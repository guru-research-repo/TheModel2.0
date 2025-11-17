"""
transforms.py

Provides five image transformation functions:

- four_random_crops: generates four random square-like crops from a single image; does not support batch input.
- rotate: rotates images by a random angle within ±max_deg or 180° if inverse=True; supports both single-image (C,H,W) and batch (N,C,H,W) input, applying the same angle across the batch.
- foveation: simulates foveal blur via multi-scale Gaussian pyramids centered on a point in a tensor image; single-image only.
- logpolar_cv2: applies OpenCV's log-polar mapping to a single image (tensor or PIL); no batch support.
- logpolar_manual: manually computes a log-polar transform in PyTorch with optional smoothing and custom center selection; supports batch input.

Each function returns an output with the same shape as its input. Batch support is indicated per-function above.
"""

import random
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
# import mediapipe as mp
import os
from torch import nn
from PIL import Image
from torchvision import transforms
import math

# # Create MediaPipe face mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# # Determine facial landmarks
# def get_facial_features(image, image_path=None, log_file="unidentified_faces.txt"):
#     h, w, _ = image.shape
#     rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb)

#     if not results.multi_face_landmarks:
#         print("No face detected.")
#         if image_path is not None:
#             with open(log_file, "a") as f:
#                 f.write(f"{image_path}\n")
#         return None

#     features = {'left_eye': [], 'right_eye': [], 'nose': [], 'mouth': []}
#     all_landmarks = []

#     # Indices for face parts
#     LEFT_EYE_IDX = list(range(33, 133))
#     RIGHT_EYE_IDX = list(range(263, 362))
#     NOSE_IDX = list(range(1, 5)) + list(range(94, 100))
#     MOUTH_IDX = list(range(78, 88)) + list(range(308, 318))

#     landmarks = results.multi_face_landmarks[0]

#     for i, lm in enumerate(landmarks.landmark):
#         x, y = int(lm.x * w), int(lm.y * h)
#         all_landmarks.append((x, y))
#         if i in LEFT_EYE_IDX:
#             features['left_eye'].append((x, y))
#         if i in RIGHT_EYE_IDX:
#             features['right_eye'].append((x, y))
#         if i in NOSE_IDX:
#             features['nose'].append((x, y))
#         if i in MOUTH_IDX:
#             features['mouth'].append((x, y))

#     # Compute bounding box
#     xs, ys = zip(*all_landmarks)
#     x_min, x_max = min(xs), max(xs)
#     y_min, y_max = min(ys), max(ys)
#     bbox_w = x_max - x_min
#     bbox_h = y_max - y_min
#     features["face_bbox"] = (x_min, y_min, bbox_w, bbox_h)

#     return features

# # Randomly sample salient points
# def sample_facial_feature_points_weighted(feature_dict, num_points=10):
#     if "face_bbox" not in feature_dict:
#         raise ValueError("Face bounding box is missing from features.")

#     # Compute face center
#     fx, fy, fw, fh = feature_dict["face_bbox"]
#     face_center = np.array([fx + fw / 2, fy + fh / 2])

#     # Collect all feature points (excluding bbox)
#     feature_points = []
#     for key, points in feature_dict.items():
#         if key != "face_bbox":
#             feature_points.extend(points)

#     if len(feature_points) == 0:
#         raise ValueError("No facial features found to sample from.")

#     points_arr = np.array(feature_points)

#     # Compute inverse distance to face center (closer = higher weight)
#     dists = np.linalg.norm(points_arr - face_center, axis=1)
#     # Avoid division by zero
#     dists = np.clip(dists, a_min=1e-6, a_max=None)
#     weights = 1.0 / dists 

#     # Normalize to probabilities
#     prob_weights = weights / np.sum(weights)

#     # Sample without replacement
#     num_to_sample = min(num_points, len(feature_points))
#     sampled_indices = np.random.choice(len(feature_points), size=num_to_sample, replace=False, p=prob_weights)
#     sampled_points = [feature_points[i] for i in sampled_indices]

#     return sampled_points


# def four_random_crops(img: torch.Tensor, crop_scale: float = 0.65) -> list[torch.Tensor]:
#     """
#     Given an image tensor, return a list of 4 random square-ish crops.

#     Args:
#         img (torch.Tensor): Input image tensor of shape (C, H, W).
#         crop_scale (float): Fraction of area to keep in each crop (e.g. 0.65).

#     Returns:
#         List[torch.Tensor]: Four randomly cropped patches of shape (C, crop_h, crop_w).
#     """
#     if img.dim() != 3:
#         raise ValueError(f"Expected img tensor of shape (C, H, W), got {img.shape}")

#     C, H, W = img.shape
#     # area scale α = crop_scale → side lengths scale = sqrt(α)
#     crop_h = int(H * (crop_scale ** 0.5))
#     crop_w = int(W * (crop_scale ** 0.5))

#     crops: list[torch.Tensor] = []
#     for _ in range(4):
#         top  = random.randint(0, H - crop_h) if H != crop_h else 0
#         left = random.randint(0, W - crop_w) if W != crop_w else 0
#         patch = img[:, top : top + crop_h, left : left + crop_w]
#         crops.append(patch)

#     return crops

import os, math, cv2, torch
import numpy as np
import torch.nn.functional as F
from scipy.spatial import ConvexHull
from PIL import Image
import face_alignment

# ==============================================================
# ========== Face-aware Gabor-based salience extractor ==========
# ==============================================================

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cpu')


# --------------------------------------------------------------
# Helper: Ensure odd number for kernel sizes
# --------------------------------------------------------------
def ensure_odd(n):
    return int(n) if int(n) % 2 == 1 else int(n) + 1


# --------------------------------------------------------------
# 1️⃣ Estimate head pose from 68-point landmarks
# --------------------------------------------------------------
def estimate_head_pose(pts68, img_shape):
    indices = [30, 8, 36, 45, 48, 54]
    image_points = pts68[indices].astype(np.float32)
    model_points = np.array([
        [0.0, 0.0, 0.0],
        [0.0, -63.6, -12.5],
        [-43.3, 32.7, -26.0],
        [43.3, 32.7, -26.0],
        [-28.9, -28.9, -24.1],
        [28.9, -28.9, -24.1]
    ], dtype=np.float32)
    h, w = img_shape[:2]
    focal = w
    center = (w/2, h/2)
    K = np.array([[focal, 0, center[0]],
                  [0, focal, center[1]],
                  [0, 0, 1]], dtype=np.float32)
    success, rvec, tvec = cv2.solvePnP(model_points, image_points, K, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return None, None, None
    return rvec, tvec, K


# --------------------------------------------------------------
# 2️⃣ Face mask from landmarks
# --------------------------------------------------------------
def face_mask_from_landmarks(shape, pts68, blur=21):
    h, w = shape[:2]
    hull = ConvexHull(pts68)
    poly = pts68[hull.vertices].astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, poly, 255)
    if blur > 0:
        mask = cv2.GaussianBlur(mask, (ensure_odd(blur), ensure_odd(blur)), 0)
    return mask.astype(np.float32) / 255.0


# --------------------------------------------------------------
# 3️⃣ Build anisotropic Gaussian weighting window
# --------------------------------------------------------------
def anisotropic_gaussian_window(h, w, center, sigma_x, sigma_y, theta_rad):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    x = x.float(); y = y.float()
    cx, cy = center
    xr, yr = x - cx, y - cy
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    x_rot = c*xr + s*yr
    y_rot = -s*xr + c*yr
    g = torch.exp(-0.5*((x_rot/sigma_x)**2 + (y_rot/sigma_y)**2))
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    return g


# --------------------------------------------------------------
# 4️⃣ Gabor kernel + energy bank
# --------------------------------------------------------------
def gabor_kernel_torch(ks, wavelength, theta, sigma=None, gamma=0.5, phase=0.0):
    ks = int(ks if ks % 2 == 1 else ks + 1)
    grid = torch.arange(ks) - ks//2
    y, x = torch.meshgrid(grid, grid, indexing='ij')
    x = x.float(); y = y.float()
    c, s = math.cos(theta), math.sin(theta)
    x_theta = x*c + y*s
    y_theta = -x*s + y*c
    if sigma is None:
        sigma = 0.56 * wavelength
    gb = torch.exp(-(x_theta**2 + (gamma**2)*(y_theta**2)) / (2*(sigma**2))) * torch.cos(2*math.pi*x_theta/wavelength + phase)
    return gb - gb.mean()

def gabor_energy_bank_torch(img_t, wavelengths=[5,7,9,11], n_orient=8, ks=31, gamma=0.5):
    _, _, H, W = img_t.shape
    energies = []
    for wl in wavelengths:
        e_orients = []
        for o in range(n_orient):
            theta = o * math.pi / n_orient
            gb_even = gabor_kernel_torch(ks, wl, theta, gamma=gamma, phase=0.0)
            gb_odd  = gabor_kernel_torch(ks, wl, theta, gamma=gamma, phase=math.pi/2)
            gb_even = gb_even.view(1,1,ks,ks)
            gb_odd  = gb_odd.view(1,1,ks,ks)
            resp_even = F.conv2d(img_t, gb_even, padding=ks//2)
            resp_odd  = F.conv2d(img_t, gb_odd, padding=ks//2)
            e = torch.sqrt(resp_even**2 + resp_odd**2 + 1e-8)
            e_orients.append(e)
        e_orients = torch.stack(e_orients, dim=0).sum(dim=0)
        energies.append(e_orients[0,0])
    energy_stack = torch.stack(energies, dim=0)
    e_min = energy_stack.amin(dim=(1,2), keepdim=True)
    e_max = energy_stack.amax(dim=(1,2), keepdim=True)
    return (energy_stack - e_min) / (e_max - e_min + 1e-8)


# --------------------------------------------------------------
# 5️⃣ Top-K NMS selection
# --------------------------------------------------------------
def topk_points_nms(sal_map, K=50, radius=6):
    H, W = sal_map.shape
    sal = sal_map.copy()
    picked = []
    rad2 = radius * radius
    for _ in range(K):
        idx = np.argmax(sal)
        y, x = divmod(idx, W)
        score = sal[y, x]
        if score <= 0:
            break
        picked.append((x, y, float(score)))
        yy, xx = np.ogrid[:H, :W]
        mask = (yy - y)**2 + (xx - x)**2 <= rad2
        sal[mask] = 0.0
    return picked


# --------------------------------------------------------------
# 6️⃣ get_facial_features() (unchanged)
# --------------------------------------------------------------
def get_facial_features(image, image_path=None, log_file="unidentified_faces.txt"):
    try:
        preds = fa.get_landmarks(image)
    except Exception as e:
        preds = None

    if not preds or len(preds[0]) < 68:
        print(f"⚠️ No face detected in {image_path or 'image'}.")
        if image_path is not None:
            with open(log_file, "a") as f:
                f.write(f"{image_path}\n")
        return None

    pts68 = preds[0]
    h, w, _ = image.shape
    xs, ys = pts68[:,0], pts68[:,1]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min

    features = {
        "landmarks": pts68,
        "face_bbox": (int(x_min), int(y_min), int(bbox_w), int(bbox_h))
    }
    return features


# --------------------------------------------------------------
# 7️⃣ sample_facial_feature_points_weighted()
# --------------------------------------------------------------
def sample_facial_feature_points_weighted(feature_dict, image, num_points=50, use_pose_weighting=True):
    """
    Computes salience map with optional head-pose-based anisotropic weighting.
    Returns top-K salience points (x,y).
    """
    if "face_bbox" not in feature_dict or "landmarks" not in feature_dict:
        raise ValueError("Face bounding box or landmarks missing.")

    pts68 = feature_dict["landmarks"]
    x_min, y_min, w, h = feature_dict["face_bbox"]

    # ----- Step 1: base mask -----
    mask = face_mask_from_landmarks(image.shape, pts68, blur=31)
    h_img, w_img = mask.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # ----- Step 2: Head pose weighting (optional) -----
    if use_pose_weighting:
        rvec, tvec, K = estimate_head_pose(pts68, (h_img, w_img, 3))
        if rvec is not None:
            R, _ = cv2.Rodrigues(rvec)
            roll = math.atan2(R[1,0], R[0,0])
        else:
            pts_mean = pts68.mean(axis=0)
            U, S, Vt = np.linalg.svd(pts68 - pts_mean, full_matrices=False)
            major = Vt[0]
            roll = math.atan2(major[1], major[0])

        cx, cy = pts68.mean(axis=0)
        face_w = pts68[:,0].ptp(); face_h = pts68[:,1].ptp()
        sigma_x = max(15.0, 0.35*face_w)
        sigma_y = max(15.0, 0.50*face_h)
        gwin = anisotropic_gaussian_window(h_img, w_img, (cx, cy), sigma_x, sigma_y, -roll)
        weight = torch.from_numpy(mask).float() * gwin
        face_weight = (weight - weight.min()) / (weight.max() - weight.min() + 1e-8)
        face_weight = face_weight.numpy()
    else:
        # simple GaussianBlur mask fallback
        face_weight = mask

    # ----- Step 3: Gabor filtering -----
    img_t = torch.from_numpy(gray)[None,None,...].float()
    energy_stack = gabor_energy_bank_torch(img_t).cpu().numpy()

    wl = np.array([5,7,9,11], dtype=np.float32)
    inv_w = (1.0 / (wl + 1e-6)); inv_w = inv_w / inv_w.sum()
    sal_map = (energy_stack * inv_w[:,None,None]).sum(axis=0)

    sal_weighted = (sal_map * face_weight)
    sal_weighted = (sal_weighted - sal_weighted.min()) / (sal_weighted.max() - sal_weighted.min() + 1e-8)

    # ----- Step 4: Select top-K -----
    pts = topk_points_nms(sal_weighted.astype(np.float32), K=num_points, radius=6)
    sampled_points = [(x, y) for (x, y, s) in pts]
    return sampled_points


def rotate(
    data: torch.Tensor,
    max_deg: float = 15.0,
    inversion: int = 0,
    center = None
) -> torch.Tensor:
    """
    Rotate the image tensor by a random angle in [-max_deg, max_deg],
    or by 180° if inverse=True.

    Args:
        data (torch.Tensor): Input image of shape (C, H, W) or batch (N, C, H, W).
        max_deg (float): Maximum absolute rotation angle (±max_deg).
        inverse (bool): If True, rotate by exactly 180° instead of a random angle.
        0: Rotate randomly in [-max_deg, max_deg]
        1: Rotate by 180°
        2: Rotate by 0°
        3: Rotate randomly in [-15, 15] regardless of max_deg

    Returns:
        torch.Tensor: Rotated image(s), same shape as input.
    """
    # Choose angle
    if inversion == 1:
        angle = 180.0
        #print('angle rotated', angle)
    elif inversion == 2:
        angle = 0.0
        #print('angle rotated', angle)
    elif inversion == 3:
        angle = random.uniform(-15.0, 15.0)
        #print('angle rotated', angle)
    else:
        angle = random.uniform(-max_deg, max_deg)

    #angle = 180.0 if inverse else random.uniform(-max_deg, max_deg)

    # Use bilinear interpolation for smooth rotations
    interp = T.InterpolationMode.BILINEAR

    # Function to rotate a single image
    def rotate_single_image(img_tensor: torch.Tensor, center=None) -> torch.Tensor:
        # Convert to uint8 image for glabella detection
        # img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        # img_bgr = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # center = get_glabella_or_center(img_bgr)
        print("center in rotate ", center)

        # Rotate around fixed point
        return TF.rotate(img_tensor, angle, interpolation=interp, center=center)


    # Single image
    if data.ndim == 3:  # (C, H, W)
        return rotate_single_image(data, center=center)

    # Handle batch
    elif data.ndim == 4:  # (N, C, H, W)
        return torch.stack([rotate_single_image(img, center=center) for img in data], dim=0)

    else:
        raise ValueError(f"rotate() expected a 3D or 4D tensor, got shape {data.shape}")
        
def foveation(img: torch.Tensor, crop_size: int = 224, center=None):
    """
    Applies foveation to an image or batch of images.

    Args:
        img (torch.Tensor): Input image tensor of shape (C, H, W) or (N, C, H, W).
        crop_size (int): Diameter of the foveal region.
    Returns:
        torch.Tensor: Foveated image tensor of same shape as input.
    """
    def pyramid(tensor, sigma=1, prNum=6):
        C, H, W = tensor.shape[-3:]
        G = tensor.clone().unsqueeze(0)
        pyramids = [G]
        blur = T.GaussianBlur(5, sigma)
        # downsample
        for i in range(1, prNum):
            G = F.interpolate(blur(G), scale_factor=(0.5, 0.5), recompute_scale_factor=True)
            pyramids.append(G)
        # upsample
        for i in range(1, prNum):
            for _ in range(i):
                pyramids[i] = F.interpolate(
                    pyramids[i], scale_factor=(2, 2), mode='bilinear', align_corners=True
                )
        # fix shape back to original
        for i in range(1, prNum):
            pyramids[i] = F.interpolate(pyramids[i], size=(H, W))
        # stack and remove the extra batch-dim
        return torch.stack(pyramids).squeeze(1)

    def foveat_img(im, fixs):
        sigma = 0.248
        prNum = 6
        As = pyramid(im, sigma, prNum)  # shape: (prNum, C, H, W)
        H, W = im.shape[-2:]
        # parameters
        p = 7.5
        k = 3
        alpha = 2.5
        # grid
        x = torch.arange(W, device=As.device).float()
        y = torch.arange(H, device=As.device).float()
        x2d, y2d = torch.meshgrid(x, y, indexing='ij')
        # distance map
        theta = torch.sqrt((x2d - fixs[0][0])**2 + (y2d - fixs[0][1])**2) / p
        for fx, fy in fixs[1:]:
            theta = torch.minimum(theta, torch.sqrt((x2d - fx)**2 + (y2d - fy)**2) / p)
        R = alpha / (theta + alpha)
        # blending coefficients
        Ts = [torch.exp(-((2**(i-3) * R / sigma)**2) * k) for i in range(1, prNum)]
        Ts.append(torch.zeros_like(theta))
        # omega thresholds
        omega = np.zeros(prNum)
        for i in range(1, prNum):
            omega[i-1] = np.sqrt(np.log(2)/k) / (2**(i-3)) * sigma
        omega = np.clip(omega, None, 1)
        # layer indices
        layer_ind = torch.zeros_like(R, dtype=torch.long)
        for i in range(1, prNum):
            mask = (R >= omega[i]) & (R <= omega[i-1])
            layer_ind[mask] = i
        # blend factors
        Bs = [(0.5 - Ts[i]) / (Ts[i-1] - Ts[i] + 1e-5) for i in range(1, prNum)]
        # masks
        Ms = torch.zeros((prNum, H, W), device=As.device)
        for i in range(prNum):
            mask_i = layer_ind == i
            if i == 0:
                Ms[i][mask_i] = 1
            else:
                Ms[i][mask_i] = 1 - Bs[i-1][mask_i]
            mask_i1 = layer_ind - 1 == i
            if mask_i1.any() and i < prNum:
                Ms[i][mask_i1] = Bs[i][mask_i1]
        # combine
        im_fov = (Ms.unsqueeze(1) * As).sum(dim=0)
        return im_fov

    # handle batch vs single image
    print("center in foveate ", center) # w,h --> x,y
    if img.ndim == 4:
        out = [foveat_img(img[i], [center]) for i in range(img.shape[0])]
        return torch.stack(out).float()
    else:
        return foveat_img(img, [center]).float()

# def logpolar_cv2(
#     tensor: torch.Tensor,
#     M: float = 40,
#     center: tuple[int, int] | None = None,
#     random_center: bool = False
# ) -> torch.Tensor:
#     """
#     Apply OpenCV log-polar transform to a PyTorch tensor image.
#     **Note**: cv2.logPolar requires a permutation of input tensor.

#     Args:
#         tensor: torch.Tensor of shape (C, H, W) with values in [0,1].
#         M: scaling factor (larger = finer radial resolution).
#         center: (cx, cy) in pixel coordinates; if None, defaults to image center.
#         random_center: if True, pick a random center within the image.

#     Returns:
#         torch.Tensor of shape (C, H, W), the log-polar–warped image.
#     """
#     # move to H×W×C numpy uint8
#     C, H, W = tensor.shape
#     img = tensor.permute(1, 2, 0).cpu().numpy()
#     img = (img * 255).astype(np.uint8)

#     # choose center
#     if random_center:
#         cx = int(np.random.uniform(0, W))
#         cy = int(np.random.uniform(0, H))
#         center = (cx, cy)
#     elif center is None:
#         center = (W // 2, H // 2)

#     # apply OpenCV log-polar
#     logp = cv2.logPolar(
#         img,
#         center,
#         M,
#         flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS
#     )

#     # back to torch (C×H×W, float in [0,1])
#     logp_t = torch.from_numpy(logp).permute(2, 0, 1).float() / 255.0
#     return logp_t

def logpolar_manual(
    data: torch.Tensor,
    input_shape: tuple[int,int] = (224, 224),
    output_shape: tuple[int,int] = (224, 224),
    smoothing: float | None = None,
    apply_mask: bool = False,
    position: str = 'circumscribed',
    log_polar_distance: float = 2,
    random_center: bool = False,
    center: tuple[float,float] | None = None,
):
    """
    data:        Tensor of shape (C,H,W) or (N,C,H,W)
    input_shape, output_shape: (height, width)
    smoothing:  if None, nearest-sample; else power for interpolation weights
    apply_mask: whether to zero-out out‑of‑bounds pixels
    random_center: occasionally pick saliency-based center
    center:     explicit (y,x) center override
    """

    def getPoints(numPoints, prob_arr, threshold=0.20):
        crop_size = 0
        flat = prob_arr.reshape(-1)
        
        y_thresh = max(crop_size // 2, int(threshold * prob_arr.shape[0]))
        x_thresh = max(crop_size // 2, int(threshold * prob_arr.shape[0]))
        mask = np.zeros_like(prob_arr)
        mask[y_thresh:-y_thresh, x_thresh:-x_thresh] = 1
        mask = mask.reshape(-1)
        
        probs = flat * mask
        probs_sum = probs.sum()
        if probs_sum > 0:
            probs /= probs_sum
            idx = np.random.choice(flat.shape[0], numPoints, p=probs)
            ys, xs = np.unravel_index(idx, prob_arr.shape)
            return np.stack([ys, xs], axis=0)
        else:
            # fallback to uniform
            idx = np.random.choice(flat.shape[0], numPoints)
            ys, xs = np.unravel_index(idx, prob_arr.shape)
            return np.stack([ys, xs], axis=0)
    
    def SaliencePoints(data):
        # data: torch.Tensor C×H×W or N×C×H×W
        img = data
        if isinstance(data, torch.Tensor):
            # assume C×H×W
            img = data.cpu().numpy().transpose(1,2,0)
        cv2_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sal = cv2.saliency.StaticSaliencySpectralResidual_create()
        _, salmap = sal.computeSaliency(cv2_img)
        pts = getPoints(1, salmap)
        return int(pts[0,0]), int(pts[1,0])


    H_in, W_in = input_shape
    H_out, W_out = output_shape

    # 1) choose center
    cy, cx = center

    print("center in logpolar ", center)
    print("cy = ", cy)
    print("cx = ", cx)
    # if center is None:
    #     cy, cx = (H_in/2, W_in/2)
    # else:
    #     cy, cx = center
    # if random_center and random.random() > 0.4:
    #     cy, cx = SaliencePoints(data)

    # 2) compute max log‑radius
    if position == 'circumscribed':
        max_r = torch.log(torch.tensor((H_in**2 + W_in**2)**0.5/2 * log_polar_distance))
    else:
        max_r = torch.log(torch.tensor(max(H_in, W_in)/2 * log_polar_distance))

    # 3) build mapping grids
    device = data.device
    theta, r = torch.meshgrid(
        torch.arange(H_out, device=device),
        torch.arange(W_out, device=device),
        indexing='ij'
    )
    theta = theta.float()
    r     = r.float()
    X = torch.exp(r * max_r / W_out) * torch.cos(theta * 2*torch.pi / H_out)
    Y = torch.exp(r * max_r / W_out) * torch.sin(theta * 2*torch.pi / H_out)

    # shift by center
    X = cx + X
    Y = cy - Y

    # clamp indices
    x0 = X.long().clamp(0, W_in-1)
    y0 = Y.long().clamp(0, H_in-1)

    if smoothing is None:
        # nearest sampling
        out = data[..., y0, x0]
    else:
        # bilinear‑style interpolation with power weights
        x1 = (x0 + 1).clamp(0, W_in-1)
        y1 = (y0 + 1).clamp(0, H_in-1)
        dx = (X - x0).abs()
        dy = (Y - y0).abs()
        # weights
        w00 = ((1-dx)**smoothing * (1-dy)**smoothing).unsqueeze(0)
        w01 = ((1-dx)**smoothing * dy**smoothing).unsqueeze(0)
        w10 = (dx**smoothing   * (1-dy)**smoothing).unsqueeze(0)
        w11 = (dx**smoothing   * dy**smoothing).unsqueeze(0)
        out = (
            w00 * data[..., y0, x0] +
            w01 * data[..., y1, x0] +
            w10 * data[..., y0, x1] +
            w11 * data[..., y1, x1]
        )

    # 4) mask out‑of‑bounds
    if apply_mask:
        mask = ((X >= 0) & (X < W_in) & (Y >= 0) & (Y < H_in)).float()
        out = out * mask

    return out