"""
salience_trans.py (extension of trans.py)

Detects salience points in an image and provides new salience transformation pipeline with new helper functions:

- get_facial_features: determines facial landmarks from MediaPipe face mesh; does not support batch input.
- sample_facial_feature_points_weighted: samples fixation points weighted towards center of face; does not support batch input.

Each function returns an output with the same shape as its input. Batch support is indicated per-function above.

"""

import os
from pathlib import Path
import random
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
import mediapipe as mp
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from trans import *
from PIL import Image

#######################################
########## Helper functions ###########
#######################################

# Create MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# Determine facial landmarks
def get_facial_features(image):
    assert isinstance(image, np.ndarray), f"Expected np array, got {type(image)}."
    
    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        print("No face detected.")
        return None

    features = {'left_eye': [], 'right_eye': [], 'nose': [], 'mouth': []}
    all_landmarks = []

    # Indices for face parts
    LEFT_EYE_IDX = list(range(33, 133))
    RIGHT_EYE_IDX = list(range(263, 362))
    NOSE_IDX = list(range(1, 5)) + list(range(94, 100))
    MOUTH_IDX = list(range(78, 88)) + list(range(308, 318))

    landmarks = results.multi_face_landmarks[0]

    for i, lm in enumerate(landmarks.landmark):
        x, y = int(lm.x * w), int(lm.y * h)
        all_landmarks.append((x, y))
        if i in LEFT_EYE_IDX:
            features['left_eye'].append((x, y))
        if i in RIGHT_EYE_IDX:
            features['right_eye'].append((x, y))
        if i in NOSE_IDX:
            features['nose'].append((x, y))
        if i in MOUTH_IDX:
            features['mouth'].append((x, y))

    # Compute bounding box
    xs, ys = zip(*all_landmarks)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min
    features["face_bbox"] = (x_min, y_min, bbox_w, bbox_h)

    return features

# Randomly sample salient points
def sample_facial_feature_points_weighted(feature_dict, num_points=1):
    
    
    if "face_bbox" not in feature_dict:
        raise ValueError("Face bounding box is missing from features.")

    # Compute face center
    fx, fy, fw, fh = feature_dict["face_bbox"]
    face_center = np.array([fx + fw / 2, fy + fh / 2])

    # Collect all feature points (excluding bbox)
    feature_points = []
    for key, points in feature_dict.items():
        if key != "face_bbox":
            feature_points.extend(points)

    if len(feature_points) == 0:
        raise ValueError("No facial features found to sample from.")

    points_arr = np.array(feature_points)

    # Compute inverse distance to face center (closer = higher weight)
    dists = np.linalg.norm(points_arr - face_center, axis=1)
    # Avoid division by zero
    dists = np.clip(dists, a_min=1e-6, a_max=None)
    weights = 1.0 / dists 

    # Normalize to probabilities
    prob_weights = weights / np.sum(weights)

    # Sample without replacement
    num_to_sample = min(num_points, len(feature_points))
    sampled_indices = np.random.choice(len(feature_points), size=num_to_sample, replace=False, p=prob_weights)
    sampled_points = [feature_points[i] for i in sampled_indices]

    return sampled_points

#######################################
######### Salience Pipeline ###########
#######################################

class SaliencePipeline(torch.nn.Module):
    def __init__(self, type='train', device='cpu', logpolar=True, img_size=224, 
                 output_shape=(224, 224), num_salient_points=4, n_crops=4):
        """
        Pipeline that rotates, foveates, and log-polar transforms around a salient point (LP)
            or crops then rotates around a salient point (CNN).
        
        Args:
            type (str): 'train', 'test', or 'valid' --> 'train', 'inverted', or None
            device (str): torch device
            logpolar (bool): whether to apply log-polar transform
            img_size (int): image size (LP) or crop size (CNN)
            output_shape (tuple): output shape for log-polar transform
            num_salient_points (int): number of fixations 
            n_crops (int): number of crops for CNN
        """
        super().__init__()
        self.num_salient_points = num_salient_points
        self.device = device
        self.type = type
        self.n_crops = n_crops
        
        self.foveate = Foveate() if logpolar else torch.nn.Identity()
        self.logpolar = LogPolar(
            input_shape=(img_size, img_size),
            output_shape=output_shape,
            device=device
        ) if logpolar else torch.nn.Identity()

        self.lp_true = logpolar
        self.crop = RandomCrop(n=n_crops, crop_size=img_size)

        self.kernels = self.get_kernels()

    # Get kernels for gabor filters, each in different direction TODO: test out reducing the number of these!
    def get_kernels(self):
        kernels = []
        size = (31,31)
        lambd = [4.0, 8.0, 12.0]
        sigma = [0.56*l for l in lambd]
        psi = [0, np.pi / 2] # without both it has 2 lines, more wavy
        theta = [0.0, np.pi / 4, 2 * np.pi / 4, 3 * np.pi / 4,
                 np.pi, 5 * np.pi / 4, 6 * np.pi / 4, 7 * np.pi / 4]
        gamma = 0.5

        for l in range(len(lambd)):
            for p in range(len(psi)):
                for t in range(len(theta)):
                    kernels.append(cv2.getGaborKernel(size, sigma[l], theta[t], lambd[l], gamma, psi[p], ktype=cv2.CV_32F))

        self.num_kernels = len(kernels)
        stacked_filters = torch.from_numpy(np.stack(kernels)).unsqueeze(1) 

        # Create a single Conv2d layer to apply all filters
        filters = torch.nn.Conv2d(in_channels=1, out_channels=self.num_kernels, kernel_size=31, padding='same', bias=False)
        filters.weight.data = stacked_filters

        return filters


    def sample_salience_points(self, img, center = None):
        h, w, _ = img.shape
        filtered = []#np.zeros((len(self.kernels), h, w))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # out_img = TF.to_pil_image(img)
        # filename = f"out/img_proc_original.png"
        # out_img.save(filename)

        # gaussian mask
        if center is None:
            center_x = w / 2 - 0.5
            center_y = h / 2 - 0.5
        else:
            center_x, center_y = center

        x = torch.arange(0, w, dtype=torch.float32)
        y = torch.arange(0, h, dtype=torch.float32)
        y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')

        # -----------------------------------------------------
        # Hyperparameters towards Gaussian Filter
        # -----------------------------------------------------
        alpha = 2    # sharpness edge drop
        sigma = w / 4  # range of attention

        gaussian_mask = torch.exp(
            -((x_grid - center_x)**2 + (y_grid - center_y)**2) / (2 * sigma **2)
        ) ** alpha

        weighted_img = (gaussian_mask * img).to(self.device)
        
        filtered = self.kernels(weighted_img.unsqueeze(0))
        filtered = (filtered - torch.mean(filtered, dim=(1,2), keepdim=True)) / (torch.std(filtered, dim=(1,2), keepdim=True)+1e-9)

        # calculate variance
        # variance = torch.var(filtered, dim=0)**2
        variance = torch.var(filtered, dim=0)
        # variance = (variance - variance.min()) / (variance.max() - variance.min())
        # T = 1600
        # variance = torch.softmax((variance / T).reshape(-1), dim=0).view_as(variance)
        # variance = torch.var(filtered, dim=0)**2
        # eps = 1e-8
        # variance = (variance - variance.mean()) / (variance.std() + eps)
        T = 4
        variance = torch.softmax((variance / T).flatten(), dim=0).view_as(variance)
        # variance = torch.softmax(variance / T, dim=0).view_as(variance)

        weights = variance.flatten(start_dim=-2, end_dim=-1)
        features = torch.multinomial(weights, self.num_salient_points)
        return torch.stack(torch.unravel_index(features, variance.shape)[::-1], dim=-1)
    
    def forward(self, img): 
        assert isinstance(img, torch.Tensor), f"Expected Tensor, got {type(img)}."
        
        img = img.to(self.device)
        img = self.crop(img) if not self.lp_true else img #crop if CNN
        B,C,H,W = img.shape
        img_np = (img.permute(0,2,3,1).cpu().numpy() * 255).astype(np.uint8) 
        transformed_imgs = torch.zeros((B,self.num_salient_points,C,H,W),device=self.device)

        # Loop through each identity in batch
        for b in range(B): 
            salient_points = self.sample_salience_points(img_np[b])

            for salient_idx, center in enumerate(salient_points):
                if self.type == 'train':
                    if self.lp_true:
                        angle=torch.empty(1).uniform_(-15,15).item() #sample value in range [-15,15]
                        transformed_img = TF.rotate(img[b],angle=angle,center=(center[0],center[1]))
                    else:
                        angle=torch.empty(B).uniform_(-15,15) #sample n_crops-many values in range [-15,15]
                        transformed_img = TF.rotate(img[b],angle=angle[b].item(),center=(center[0],center[1]))
                elif self.type == 'test': 
                    transformed_img = TF.rotate(img[b],angle=180) # invert test images
                else:
                    transformed_img = img[b].clone()
                if self.lp_true:
                    transformed_img = self.foveate(transformed_img.unsqueeze(0), center=tuple(center)) # (3,224,224) --> Foveate expects batch
                    transformed_img = self.logpolar(transformed_img, center_x=center[0], center_y=center[1]) # (3,224,224)
                transformed_imgs[b,salient_idx] = transformed_img
                
        return transformed_imgs
    

if __name__ == "__main__":
    img_paths = [
                  f"data/faces/faces/32_identities/train/AdamRippon/3.jpg",
                  f"data/faces/faces/32_identities/train/BarackObama/8.jpg",
                  f"data/faces/faces/32_identities/train/EmmanuelMacron/101.jpg",
                  f"data/faces/faces/32_identities/train/GalGadot/8.jpg",
                  f"data/faces/faces/32_identities/train/EdSheeran/63.jpg",
                  f"data/faces/faces/32_identities/train/JohnLegend/18.jpg",
                  f"data/faces/faces/32_identities/train/KevinDurant/30.jpg",
                  f"data/faces/faces/32_identities/train/LiuWen/6.jpg",
                  f"data/faces/faces/32_identities/train/PrinceHarry/104.jpg",
                  f"data/faces/faces/32_identities/train/RyanReynolds/23.jpg"
                  ]
    
    img_paths_ = [
                  f"data/faces/faces/32_identities/train/LiuWen/5.jpg",
                  f"data/faces/faces/32_identities/train/LiuWen/6.jpg",
                  f"data/faces/faces/32_identities/train/LiuWen/7.jpg",
                  f"data/faces/faces/32_identities/train/LiuWen/17.jpg",
                  f"data/faces/faces/32_identities/train/LiuWen/23.jpg",
                  f"data/faces/faces/32_identities/train/LiuWen/24.jpg",
                  f"data/faces/faces/32_identities/train/LiuWen/27.jpg",
                  f"data/faces/faces/32_identities/train/LiuWen/31.jpg",
                  f"data/faces/faces/32_identities/train/LiuWen/46.jpg",
                  f"data/faces/faces/32_identities/train/LiuWen/60.jpg",
                  ]

    image_paths = [Path(p).expanduser() for p in img_paths]

    # Reuse the same pipeline object for all images
    salience_pipeline = SaliencePipeline('train', num_salient_points=64)

    # Create a 2x5 grid of subplots (10 total)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()  # make it 1D for easy indexing

    for idx, img_path in enumerate(image_paths):
        ax = axes[idx]

        # Open image
        image = Image.open(img_path).convert("RGB")

        # Convert to numpy uint8 [H, W, C] for your salience function
        tensor_img = TF.to_tensor(image)              # [C, H, W], float32 in [0, 1]
        tensor_img = (tensor_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Sample points
        points = salience_pipeline.sample_salience_points(tensor_img)

        # Show image and points on this subplot
        ax.imshow(image)
        ax.plot(points[:, 0], points[:, 1], 'o', color='red', markersize=3)
        ax.set_title(img_path.parent.name, fontsize=8)  # e.g., EmmanuelMacron
        ax.axis("off")

    # If there were fewer than 10 images, we'd turn extra axes off, but you have exactly 10.

    plt.tight_layout()
    # Save one combined canvas
    out_path = Path("./output/img_proc_points_grid.png")
    print("successful")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
