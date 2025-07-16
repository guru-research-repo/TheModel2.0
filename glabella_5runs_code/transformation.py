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

def four_random_crops(img: torch.Tensor, crop_scale: float = 0.65) -> list[torch.Tensor]:
    """
    Given an image tensor, return a list of 4 random square-ish crops.

    Args:
        img (torch.Tensor): Input image tensor of shape (C, H, W).
        crop_scale (float): Fraction of area to keep in each crop (e.g. 0.65).

    Returns:
        List[torch.Tensor]: Four randomly cropped patches of shape (C, crop_h, crop_w).
    """
    if img.dim() != 3:
        raise ValueError(f"Expected img tensor of shape (C, H, W), got {img.shape}")

    C, H, W = img.shape
    # area scale α = crop_scale → side lengths scale = sqrt(α)
    crop_h = int(H * (crop_scale ** 0.5))
    crop_w = int(W * (crop_scale ** 0.5))

    crops: list[torch.Tensor] = []
    for _ in range(4):
        top  = random.randint(0, H - crop_h) if H != crop_h else 0
        left = random.randint(0, W - crop_w) if W != crop_w else 0
        patch = img[:, top : top + crop_h, left : left + crop_w]
        crops.append(patch)

    return crops

def rotate(
    data: torch.Tensor,
    max_deg: float = 15.0,
    inversion: int = 0
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

    # Single image
    if data.ndim == 3:  # (C, H, W)
        return TF.rotate(data, angle, interpolation=interp)

    # Batch of images
    elif data.ndim == 4:  # (N, C, H, W)
        return torch.stack([
            TF.rotate(img, angle, interpolation=interp) 
            for img in data
        ], dim=0)

    else:
        raise ValueError(f"rotate() expected a 3D or 4D tensor, got shape {data.shape}")

def foveation(img: torch.Tensor, crop_size: int = 224):
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
    if img.ndim == 4:
        out = [foveat_img(img[i], [(crop_size/2, crop_size/2)]) for i in range(img.shape[0])]
        return torch.stack(out).float()
    else:
        return foveat_img(img, [(crop_size/2, crop_size/2)]).float()

def logpolar_cv2(
    tensor: torch.Tensor,
    M: float = 40,
    center: tuple[int, int] | None = None,
    random_center: bool = False
) -> torch.Tensor:
    """
    Apply OpenCV log-polar transform to a PyTorch tensor image.
    **Note**: cv2.logPolar requires a permutation of input tensor.

    Args:
        tensor: torch.Tensor of shape (C, H, W) with values in [0,1].
        M: scaling factor (larger = finer radial resolution).
        center: (cx, cy) in pixel coordinates; if None, defaults to image center.
        random_center: if True, pick a random center within the image.

    Returns:
        torch.Tensor of shape (C, H, W), the log-polar–warped image.
    """
    # move to H×W×C numpy uint8
    C, H, W = tensor.shape
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)

    # choose center
    if random_center:
        cx = int(np.random.uniform(0, W))
        cy = int(np.random.uniform(0, H))
        center = (cx, cy)
    elif center is None:
        center = (W // 2, H // 2)

    # apply OpenCV log-polar
    logp = cv2.logPolar(
        img,
        center,
        M,
        flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS
    )

    # back to torch (C×H×W, float in [0,1])
    logp_t = torch.from_numpy(logp).permute(2, 0, 1).float() / 255.0
    return logp_t

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
    if center is None:
        cy, cx = (H_in/2, W_in/2)
    else:
        cy, cx = center
    if random_center and random.random() > 0.4:
        cy, cx = SaliencePoints(data)

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