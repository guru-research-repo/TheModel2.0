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
                 output_shape=(224, 224), num_salient_points=4):
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
        self.crop_size = 180
        
        self.foveate = Foveate() if logpolar else torch.nn.Identity()
        self.logpolar = LogPolar(
            input_shape=(img_size, img_size),
            output_shape=output_shape,
            device=device
        ) if logpolar else torch.nn.Identity()

        self.kernels = self.get_kernels().to(device)

    # Get kernels for gabor filters, each in different direction
    def get_kernels(self):
        kernels = []
        size = (31,31)
        lambd = [5.0,11.0]
        sigma = [0.56*l for l in lambd]
        psi = [0.0, np.pi/2] # without both it has 2 lines, more wavy
        theta = [0.0,np.pi/4,2*np.pi/4,3*np.pi/4]#4*np.pi/4,5*np.pi/4,6*np.pi/4,7*np.pi/4] #could probably remove second half
        gamma = 0.5

        for p in range(len(psi)):
            for l in range(len(lambd)):
                for t in range(len(theta)):
                    kernels.append(cv2.getGaborKernel(size, sigma[l], theta[t], lambd[l], gamma, psi[p], ktype=cv2.CV_32F))

        self.num_kernels = len(kernels)
        stacked_filters = torch.from_numpy(np.stack(kernels)).unsqueeze(1) 

        # Create a single Conv2d layer to apply all filters
        filters = torch.nn.Conv2d(in_channels=1, out_channels=self.num_kernels, kernel_size=31, padding='same', bias=False)
        filters.weight.data = stacked_filters
        filters.weight.requires_grad_(False)

        return filters


    def sample_salience_points(self, img, center = None):
        _, h, w = img.shape
        filtered = []#np.zeros((len(self.kernels), h, w))

        img = TF.rgb_to_grayscale(img, num_output_channels=1)

        # gaussian mask
        if center is None:
            center_x = w / 2 - 0.5
            center_y = h / 2 - 0.5
        else:
            center_x, center_y = center

        x = torch.arange(0, w, dtype=torch.float32)
        y = torch.arange(0, h, dtype=torch.float32)
        y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
        gaussian_mask = torch.exp(-((x_grid - center_x)**2 + (y_grid - center_y)**2) / (2 * (w/6)**2)).to(self.device)

        weighted_img = (gaussian_mask * img).to(self.device)
        
        # apply gabor filters
        with torch.no_grad():
            filtered = self.kernels(weighted_img.unsqueeze(0))

        # mag=sqrt(real**2, imaginary**2)
        # todo: test if using this is better
        # filtered = torch.sqrt(filtered[:, :8]**2 + filtered[:,8:]**2)

        # normalize
        filtered = (filtered - torch.mean(filtered, dim=(2,3), keepdim=True)) / (torch.std(filtered, dim=(2,3), keepdim=True)+1e-9)

        # calculate variance
        variance = torch.var(filtered, dim=1)**2
        variance = (variance - variance.min()) / (variance.max() - variance.min())
        # out_img = TF.to_pil_image(variance)
        # filename = f"out/img_proc_variance332.png"
        # out_img.save(filename)

        # save top num_salient_points points
        weights = variance.flatten(start_dim=-2, end_dim=-1)
        features = torch.multinomial(weights, self.num_salient_points, replacement=False)
        ys = features // w
        xs = features % w
        return torch.stack([xs, ys], dim=-1)
    
    def forward(self, img): 
        assert isinstance(img, torch.Tensor), f"Expected Tensor, got {type(img)}."
        
        img = img.to(self.device)
        B,C,H,W = img.shape
        transformed_imgs_lp = torch.zeros((B,self.num_salient_points,C,H,W),device=self.device)
        transformed_imgs_cnn = torch.zeros((B,self.num_salient_points,C,self.crop_size,self.crop_size),device=self.device)

        # Loop through each identity in batch
        for b in range(B): 
            salient_points = self.sample_salience_points(img[b])
            for salient_idx, center in enumerate(salient_points[0]):
                # todo: do we want to crop lp as well?
                transformed_img_cnn = TF.crop(img[b],
                                               top=center[1]-self.crop_size//2,
                                               left=center[0]-self.crop_size//2, 
                                               height=self.crop_size, width=self.crop_size)
                if self.type == 'train':
                    angle=torch.empty(1).uniform_(-15,15).item() #sample value in range [-15,15]
                    transformed_img_lp = TF.rotate(img[b],angle=angle,center=(center[0],center[1]))
                    
                    transformed_img_cnn = TF.rotate(transformed_img_cnn,angle=angle)
                elif self.type == 'test': 
                    transformed_img_lp = TF.rotate(img[b],angle=180) # invert test images
                    transformed_img_cnn = TF.rotate(transformed_img_cnn,angle=180) # invert test images
                else:
                    transformed_img_lp = img[b].clone()
                    transformed_img_cnn = transformed_img_cnn.clone()
                transformed_img_lp = self.foveate(transformed_img_lp.unsqueeze(0), center=tuple(center)) # (3,224,224) --> Foveate expects batch
                transformed_img_lp = self.logpolar(transformed_img_lp, center_x=center[0], center_y=center[1]) # (3,224,224)
                transformed_imgs_lp[b,salient_idx] = transformed_img_lp
                transformed_imgs_cnn[b,salient_idx] = transformed_img_cnn
                
        return transformed_imgs_lp, transformed_imgs_cnn
    

if __name__ == "__main__":
    id = 332
    img_path = f'data/faces_cleaned/faces_cleaned/4_identities/test/EmmanuelMacron/{id}.jpg' # image or directory of images
    
    img_path = Path(img_path).expanduser()

    if os.path.exists(img_path):
        if os.path.isfile(img_path):
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                print('error')
                exit(0)

            t_image = TF.to_tensor(image)
            # tensor_img = torch.stack([t_image, t_image]) # 2 images for batch testing
            points = SaliencePipeline('train', num_salient_points=64).sample_salience_points(t_image)
            
            img = mpimg.imread(img_path)
            fig, ax = plt.subplots()

            # Display the image on the axes
            ax.imshow(image)

            # Plot the points on the image
            # 'o' specifies a circular marker, 'r' sets the color to red
            ax.plot(points[0,:,0], points[0,:,1], 'o', color='red', markersize=4)
            plt.savefig(f'./out/img_proc_points_{id}.png')
