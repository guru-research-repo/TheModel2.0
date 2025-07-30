'''
1. Clone the repo https://github.com/matthias-k/DeepGaze.git
2. The repo should be cloned into a folder named 'DeepGaze'.
3. This file deepgaze_preprocess.py should be outside of the 'DeepGaze' folder. 
4. Upload the centerbias_mit1003.npy file outside the 'DeepGaze' folder. This file can be downloaded from  https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy
5. Run python3 deepgaze_preprocess.py and get the preprocessed set of images. --> Currently, random 4 fixation points are chosen from the top 1000 salience points found from DeepGaze-2 model.
6. If you wanted to find top-N salience points, replace k=1000 with k=N wherever the get_topk_salience_points function is called. If you wanted to get random P points out of these salience points, replace 'chosen_points = random.sample(salience_points, P)' and 'n_final = P'.
7. If you wanted to visualize what is going on, Uncomment these lines of code below :

# if __name__ == "__main__":
#     img_file = "cleaned_faces/faces/faces/4_identities/train/JohnLegend/5.jpg"
#     out_plot = "visualizations/5_salience_grid.png"
#     visualize_salience_transformations(img_file, out_plot)
'''

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy.ndimage import zoom
from scipy.special import logsumexp
import sys
from pathlib import Path
import random

# Add the DeepGaze path
sys.path.append(str(Path(__file__).resolve().parent / "DeepGaze"))

# Now you can import
from deepgaze_pytorch.deepgaze2e import DeepGazeIIE


# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CENTERBIAS_TEMPLATE = np.load('centerbias_mit1003.npy')
model = DeepGazeIIE(pretrained=True).to(DEVICE).eval()

# -----------------------------------------
# DeepGaze Salience Point Extraction Helper
# -----------------------------------------
def get_topk_salience_points(
    pil_img, model, centerbias_template, device=DEVICE, k=1000
):
    img = np.array(pil_img)
    H, W = img.shape[:2]
    cb = zoom(centerbias_template, (H / centerbias_template.shape[0], W / centerbias_template.shape[1]), order=0, mode='nearest')
    cb -= logsumexp(cb)
    image_tensor = torch.tensor([img.transpose(2, 0, 1)]).to(DEVICE)
    cb_tensor = torch.tensor([cb]).to(DEVICE)

    with torch.no_grad():
        log_density = model(image_tensor, cb_tensor)[0, 0]  # (H, W)

    sal_map = log_density.cpu().numpy()
    # Top-k points (highest values in sal_map)
    idx = np.argpartition(sal_map.flatten(), -k)[-k:]
    ys, xs = np.unravel_index(idx, sal_map.shape)
    scores = sal_map[ys, xs]
    points = sorted(zip(xs, ys, scores), key=lambda x: -x[2])
    points = [(int(x), int(y)) for x, y, _ in points]

    # Remove duplicates in case of tied scores
    unique_points = []
    seen = set()
    for pt in points:
        if pt not in seen:
            unique_points.append(pt)
            seen.add(pt)
        if len(unique_points) == k:
            break
    return unique_points, sal_map, image_tensor.squeeze(0)

# ========= Your transform functions (replace with your exact versions) =========
def rotate(
    data: torch.Tensor,
    max_deg: float = 15.0,
    inversion: int = 0,
    center = None
) -> torch.Tensor:
    """
    Rotate the image tensor by a random angle in [-max_deg, max_deg],
    or by 180° if inversion==1, or 0° if 2, or a random [-15,+15] if 3.
    """
    # Choose angle
    if inversion == 1:        # 180 deg (inverted)
        angle = 180.0
    elif inversion == 2:      # 0 deg
        angle = 0.0
    elif inversion == 3:      # random [-15, 15]
        angle = np.random.uniform(-15.0, 15.0)
    else:                     # random [-max_deg, max_deg]
        angle = np.random.uniform(-max_deg, max_deg)

    # Bilinear gives smooth rotation, pass in center for custom pivot
    interp = T.InterpolationMode.BILINEAR
    print("center in rotate", center)

    # Single image
    if data.ndim == 3:  # (C, H, W)
        return TF.rotate(data, angle, interpolation=interp, center=center)

    # Handle batch
    elif data.ndim == 4:  # (N, C, H, W)
        return torch.stack([TF.rotate(data, angle, interpolation=interp, center=center) for img in data], dim=0)

    else:
        raise ValueError(f"rotate() expected a 3D or 4D tensor, got shape {data.shape}")

# Fill in your foveation function here (from your own code)
def foveation(img: torch.Tensor, crop_size: int = 224, center=None):
    def pyramid(tensor, sigma=1, prNum=6):
        C, H, W = tensor.shape[-3:]
        G = tensor.clone().unsqueeze(0)
        pyramids = [G]
        blur = T.GaussianBlur(5, sigma)
        for i in range(1, prNum):
            G = F.interpolate(blur(G), scale_factor=(0.5, 0.5), recompute_scale_factor=True)
            pyramids.append(G)
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

    return img

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
    # Use your provided logpolar_manual
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


# Default dataset name and identity counts for 'faces'
DEFAULT_DATASET = "faces"
IDENTITY_COUNTS = [4, 8, 16, 32, 64, 128]
CENTERBIAS_TEMPLATE = np.load('centerbias_mit1003.npy')

def process_dataset(
    dataset: str = DEFAULT_DATASET,
    root_dir: str = "cleaned_faces",
    processed_dir: str = "deepgaze2_processed_data"
):
    root = Path(root_dir).expanduser()
    dest = Path(processed_dir).expanduser()

    if dataset == "faces":
        base = root / dataset / dataset
        sub_dirs = [base / f"{n}_identities" for n in IDENTITY_COUNTS]
    else:
        base = root / dataset
        sub_dirs = [base]

    splits = ["train", "valid", "test"]

    for sub in sub_dirs:
        print(f"Now processing sub directory {sub}.")
        for split in splits:
            input_split = sub / split
            if not input_split.exists():
                continue

            rel = sub.relative_to(root)
            output_split = dest / rel / split
            output_split.mkdir(parents=True, exist_ok=True)

            for label_dir in input_split.iterdir():
                if not label_dir.is_dir():
                    continue
                out_label = output_split / label_dir.name
                out_label.mkdir(exist_ok=True)

                for img_file in label_dir.iterdir():
                    print("img_file", img_file)
                    if not img_file.is_file():
                        continue
                    try:
                        pil_img = Image.open(img_file).convert("RGB")
                    except Exception:
                        continue

                    # Get top-10 salience points
                    salience_points, _, tensor_img = get_topk_salience_points(pil_img, model, CENTERBIAS_TEMPLATE, device=DEVICE, k=1000)
                    random.seed(42)
                    chosen_points = random.sample(salience_points, 4)
                    
                    tensor_img = TF.to_tensor(pil_img)

                    # Apply transformation per point
                    for i, center in enumerate(chosen_points):
                        if split == "train":
                            targets = [("train_upright", 3)]  # random [-15, 15]
                        elif split == "valid":
                            targets = [("valid_upright", 2), ("valid_inverted", 1)]  # 0°, 180°
                        elif split == "test":
                            targets = [("test_upright", 2), ("test_inverted", 1)]  # 0°, 180°
                        else:
                            targets = []

                        for folder_name, inversion in targets:
                        # Transform in sequence: rotate → foveate → log-polar
                            rotated = rotate(tensor_img, inversion=inversion, center=center)
                            foveated = foveation(rotated, center=center)
                            C, H, W = foveated.shape
                            logpolar = logpolar_manual(
                                foveated, input_shape=(H, W), output_shape=(H, W), center=(center[1], center[0])
                            )
    
                            # Save image
                            out_img = TF.to_pil_image(logpolar.clamp(0, 1))
                            output_path = Path(processed_dir) / rel / folder_name / label_dir.name
                            output_path.mkdir(parents=True, exist_ok=True)
                            filename = f"{img_file.stem}_point{i}.png"
                            out_img.save(output_path / filename)

def get_unique_colors(n):
    """Return n visually distinct BGR colors for OpenCV."""
    import colorsys
    hsv_colors = [(i / n, 1.0, 1.0) for i in range(n)]
    rgb_colors = [colorsys.hsv_to_rgb(*hsv) for hsv in hsv_colors]
    bgr_colors = [(int(b * 255), int(g * 255), int(r * 255)) for r, g, b in rgb_colors]
    return bgr_colors

def visualize_salience_transformations(image_path, save_path):
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    from pathlib import Path
    import torchvision.transforms.functional as TF

    pil_img = Image.open(image_path).convert("RGB")
    salience_points, sal_map, tensor_img = get_topk_salience_points(
        pil_img, model, CENTERBIAS_TEMPLATE, device=DEVICE, k=1000
    )
    tensor_img = TF.to_tensor(pil_img).to(DEVICE)

    # --- Randomly select 10 unique points (no replacement) from top-100
    random.seed(42)  # For reproducibility; remove or set differently if not needed
    n_final = 4
    final_indices = random.sample(range(len(salience_points)), n_final)
    chosen_points = [salience_points[i] for i in final_indices]

    fig, axs = plt.subplots(n_final, 5, figsize=(15, 3 * n_final))
  
    unique_colors = get_unique_colors(n_final)

    for i, (x, y) in enumerate(chosen_points):
        C, H, W = tensor_img.shape

        # Original image
        axs[i, 0].imshow(pil_img)
        axs[i, 0].axis("off")
        if i == 0:
            axs[i, 0].set_title("Original")

        # Original with all salience points, each in a unique color
        img_with_points = np.array(pil_img).copy()
        for idx, (px, py) in enumerate(chosen_points):
            color = unique_colors[idx]
            cv2.circle(img_with_points, (px, py), 6, color, -1)
 
        axs[i, 1].imshow(img_with_points)
        axs[i, 1].axis("off")
        if i == 0:
            axs[i, 1].set_title("All Salience Points")

        # Rotation
        rotated = rotate(tensor_img, inversion=3, center=(x, y))
        rotated_img = TF.to_pil_image(rotated.clamp(0, 1).cpu())
        rotated_np = np.array(rotated_img)
        color = unique_colors[i]
        cv2.circle(rotated_np, (x, y), 6, color, -1)
        cv2.putText(rotated_np, f"({x},{y})", (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        axs[i, 2].imshow(rotated_np)

        axs[i, 2].imshow(rotated_np)
        axs[i, 2].axis("off")
        if i == 0:
            axs[i, 2].set_title("Rotated")

        # Foveation
        foveated = foveation(rotated, center=(x, y))
        axs[i, 3].imshow(TF.to_pil_image(foveated.clamp(0, 1).cpu()))
        axs[i, 3].axis("off")
        if i == 0:
            axs[i, 3].set_title("Foveated")

        # Log-Polar
        logpolar = logpolar_manual(
            foveated, input_shape=(H, W), output_shape=(H, W), center=(y, x)
        )
        axs[i, 4].imshow(TF.to_pil_image(logpolar.clamp(0, 1).cpu()))
        axs[i, 4].axis("off")
        if i == 0:
            axs[i, 4].set_title("Log-Polar")

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()



# if __name__ == "__main__":
#     img_file = "cleaned_faces/faces/faces/4_identities/train/JohnLegend/5.jpg"
#     out_plot = "visualizations/5_salience_grid.png"
#     visualize_salience_transformations(img_file, out_plot)


if __name__ == "__main__":
    Path(processed_dir:="deepgaze2_processed_data").mkdir(exist_ok=True)
    if len(sys.argv) > 1:
        process_dataset(sys.argv[1])
    else:
        process_dataset()
        