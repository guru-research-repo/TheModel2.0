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
    def __init__(self, type='train', device='cpu', logpolar=True, crop_size=180, 
                 output_shape=(180, 180), num_salient_points=4):
        """
        Pipeline that rotates, foveates, and log-polar transforms around a salient point (LP)
            or crops then rotates around a salient point (CNN).
        
        Args:
            type (str): 'train', 'test', or 'valid' --> 'train', 'inverted', or None
            device (str): torch device
            logpolar (bool): whether to apply log-polar transform --> IGNORED: currently does both at once
            crop_size (int): crop size for both LP and CNN
            output_shape (tuple): output shape for log-polar transform
            num_salient_points (int): number of fixations 
        """
        super().__init__()
        self.num_salient_points = num_salient_points
        self.device = device
        self.type = type
        self.crop_size = crop_size
        
        if type == 'train':
            self.rotate = Rotate()
        elif type == 'test':
            self.rotate = Rotate(invert = True)
        else:
            self.rotate = torch.nn.Identity()

        self.foveate = Foveate(crop_size=crop_size)
        self.logpolar = LogPolar(input_shape=(crop_size, crop_size),
                                output_shape=output_shape, device=device)
        self.gaborlogpolar = LogPolar(input_shape=(224,224),
                                output_shape=(224,224), device=device)

        self.kernels = self.get_kernels().to(device)

    # Get kernels for gabor filters, each in different direction
    def get_kernels(self):
        kernels = []
        size = (31,31)
        lambd = [4.0,8.0]
        sigma = [0.5*l for l in lambd]
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
        img = img.to(self.device)
        B, _, H, W = img.shape

        img = TF.rgb_to_grayscale(img, num_output_channels=1)

        # gaussian mask
        if center is None:
            center_x = W / 2 - 0.5
            center_y = H / 2 - 0.5
        else:
            center_x, center_y = center

        x = torch.arange(0, W, device=self.device)
        y = torch.arange(0, H, device=self.device)
        y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
        
        # -----------------------------------------------------
        # Hyperparameters towards Gaussian Filter
        # -----------------------------------------------------
        alpha = 2    # sharpness edge drop
        sigma = W / 4  # range of attention

        gaussian_mask = torch.exp(
            -((x_grid - center_x)**2 + (y_grid - center_y)**2) / (2 * sigma **2)
        ) ** alpha

        gaussian_mask = gaussian_mask.repeat(B, 1, 1).unsqueeze(1)
        weighted_img = gaussian_mask * img

        weighted_img, xMap, yMap = self.gaborlogpolar.forwardReturnMapping(weighted_img)
        # out_img = TF.to_pil_image(weighted_img[0])
        # filename = f"out/img_proc_lp332.png"
        # out_img.save(filename)
        # apply gabor filters
        with torch.no_grad():
            filtered = self.kernels(weighted_img)
        _, _, _, W = filtered.shape
        # mask out boundaries
        filtered[...,:10,:] = 0
        filtered[...,-10:,:] = 0
        filtered[...,:,:10] = 0
        filtered[...,:,-10:] = 0

        # mag=sqrt(real**2, imaginary**2)
        num_pairs = filtered.shape[1] // 2
        filtered = torch.sqrt(filtered[:, :num_pairs]**2 + filtered[:,num_pairs:]**2 + 1e-9)

        # normalize
        # fmean = torch.mean(filtered, dim=(2,3), keepdim=True)
        # fstd = torch.std(filtered, dim=(2,3), keepdim=True)
        # filtered = (filtered - fmean) / (fstd + 1e-9)

        # calculate variance
        variance = torch.var(filtered, dim=1)
        # fmean = torch.mean(variance, dim=(1,2), keepdim=True)
        # fstd = torch.std(variance, dim=(1,2), keepdim=True)
        # variance = (variance - fmean) / (fstd + 1e-9)
        # normalize per image
        # variance_out = (variance - variance.min()) / (variance.max() - variance.min())
        # taking softmax out for now
        # temp = 3
        # variance = torch.softmax((variance / temp).flatten(-2,-1), dim=-1).view_as(variance)
        # out_img = TF.to_pil_image(variance_out[0])
        # filename = f"out/img_proc_variance332.png"
        # out_img.save(filename)

        # save top num_salient_points points
        coords = torch.zeros(B, self.num_salient_points, 2, device=self.device, dtype=torch.long)

        for b in range(B):
            flat = variance[b].flatten()
            idx = torch.multinomial(flat, self.num_salient_points, replacement=False)
            ys = idx // W
            xs = idx % W
            y_actual = yMap[ys,xs] #+ 22
            x_actual = xMap[ys,xs] #+ 22
            coords[b] = torch.stack([x_actual, y_actual], dim=-1)  # [num_points, 2]
            # coords[b] = torch.stack([xs, ys], dim=-1)  # [num_points, 2]
        return coords # [B, num_points, 2]
    
    def forward(self, img): 
        assert isinstance(img, torch.Tensor), f"Expected Tensor, got {type(img)}."
        
        img = img.to(self.device)
        B,C,H,W = img.shape
        transformed_imgs = torch.zeros((B,self.num_salient_points,C,self.crop_size,self.crop_size),device=self.device)

        salient_points = self.sample_salience_points(img)

        # crop, we are now centered on each fixation point
        for b in range(B):
            for salient_idx, center in enumerate(salient_points[b]):
                transformed_imgs[b,salient_idx] = TF.crop(img[b],
                                              top=center[1]-self.crop_size//2,
                                              left=center[0]-self.crop_size//2, 
                                              height=self.crop_size, width=self.crop_size)
        
        transformed_imgs = transformed_imgs.flatten(0,1) # output shape is (B*N,...), represented as B B B B
        transformed_imgs = self.rotate(transformed_imgs)
        transformed_imgs_cnn = transformed_imgs.clone()


        transformed_imgs = self.foveate(transformed_imgs)
        transformed_imgs = self.logpolar(transformed_imgs)

        transformed_imgs = transformed_imgs.unflatten(0, (B, self.num_salient_points))
        transformed_imgs_cnn = transformed_imgs_cnn.unflatten(0, (B, self.num_salient_points))

        return transformed_imgs, transformed_imgs_cnn # lp, cnn

if __name__ == "__main__":
    id = 309
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
            tensor_img = torch.stack([t_image, t_image]) # 2 images for batch testing
            points = SaliencePipeline('train', num_salient_points=64).sample_salience_points(tensor_img)

            fig, ax = plt.subplots()

            # Display the image on the axes
            ax.imshow(image)

            # Plot the points on the image
            # 'o' specifies a circular marker, 'r' sets the color to red
            ax.plot(points[0,:,0], points[0,:,1], 'o', color='red', markersize=4)
            plt.savefig(f'./out/img_proc_points_{id}.png')
