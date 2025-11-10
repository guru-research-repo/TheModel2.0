import sys
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformation_salience_new import get_facial_features, sample_facial_feature_points_weighted, rotate, foveation, logpolar_manual

print("Environment Initialized")

import matplotlib
matplotlib.use('Agg')  # ✅ safe for headless runs (cluster)
import matplotlib.pyplot as plt

def visualize_pipeline_per_image(base_img, fixations, rotated_list, foveated_list, logpolar_list, img_name):
    """
    Creates a 4x5 grid showing pipeline stages for each fixation of one image.
    base_img: np.ndarray (H,W,3)
    fixations: list of (x, y)
    rotated_list, foveated_list, logpolar_list: list of np.ndarrays (H,W,3)
    img_name: filename stem to save visualization as
    """
    num_fix = len(fixations)
    headers = ["Original + Fixations", "Rotate", "Foveate", "Log Polar"]

    fig, axs = plt.subplots(nrows=num_fix, ncols=4, figsize=(16, 3 * num_fix))

    for i in range(num_fix):
        # 1️⃣ original with fixation point
        overlay = base_img.copy()
        for j, (fx, fy) in enumerate(fixations):
            color = (0, 255, 0) if j == i else (200, 200, 200)
            cv2.circle(overlay, (int(fx), int(fy)), 4, color, -1)
        axs[i, 0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axs[i, 0].axis("off")
        if i == 0:
            axs[i, 0].set_title(headers[0])

        # 2️⃣ rotated
        axs[i, 1].imshow(cv2.cvtColor(rotated_list[i], cv2.COLOR_BGR2RGB))
        axs[i, 1].axis("off")
        if i == 0:
            axs[i, 1].set_title(headers[1])

        # 3️⃣ foveated
        axs[i, 2].imshow(cv2.cvtColor(foveated_list[i], cv2.COLOR_BGR2RGB))
        axs[i, 2].axis("off")
        if i == 0:
            axs[i, 2].set_title(headers[2])

        # 4️⃣ log-polar
        axs[i, 3].imshow(cv2.cvtColor(logpolar_list[i], cv2.COLOR_BGR2RGB))
        axs[i, 3].axis("off")
        if i == 0:
            axs[i, 3].set_title(headers[3])

    plt.tight_layout()
    plt.savefig(f"{img_name}_pipeline.png", dpi=150)
    plt.close(fig)


# ---------------------- Visualization (optional) ----------------------
def visualize_pipeline(samples):
    fig, axs = plt.subplots(nrows=len(samples), ncols=5, figsize=(18, 2.5 * len(samples)))
    headers = ["original", "Salience", "rotate", "foveate", "log polar"]

    for row, sample in enumerate(samples):
        for col, key in enumerate(headers):
            ax = axs[row][col] if len(samples) > 1 else axs[col]
            ax.imshow(sample[key])
            ax.axis("off")
            if row == 0:
                ax.set_title(key, fontsize=12)

    plt.tight_layout()
    plt.savefig("pipeline_visualization.png", dpi=150)
    plt.close(fig)   # ✅ release memory


# ---------------------- Defaults ----------------------
DEFAULT_DATASET = "faces"
IDENTITY_COUNTS = [32]  # Only use 32_identities
FIXATION_LEVELS = [64]  # [4, 8, 16, 32, 64, 128]


# ---------------------- Helper ----------------------
def tensor_to_bgr(tensor: torch.Tensor) -> np.ndarray:
    """Convert a (C, H, W) float tensor to uint8 BGR image."""
    img_np = tensor.permute(1, 2, 0).cpu().numpy()
    return cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


# ---------------------- Core Processing ----------------------
def process_dataset(
    dataset: str = DEFAULT_DATASET,
    root_dir: str = "cleaned_faces_dataset",
    processed_dir: str = "32_identities_64_fixations_preprocessed_faces",
    visualize_count: int = 20,
):
    """
    Processes images under `root_dir` into `processed_dir`:
      - For faces: root/faces/faces/32_identities/{train,valid,test}/<label>/<img>
      - Generates logpolar replicas for 4,8,16,32,64,128 fixations
      - Saves into train_upright, valid_upright, valid_inverted, test_upright, test_inverted
    """
    root = Path(root_dir).expanduser()
    dest = Path(processed_dir).expanduser()
    samples_to_visualize = []

    # Only process 32_identities
    if dataset == "faces":
        base = root / dataset / dataset
        sub_dirs = [base / f"{n}_identities" for n in IDENTITY_COUNTS]
    else:
        sub_dirs = [root / dataset]

    splits = ["train", "valid", "test"]

    for sub in sub_dirs:
        print(f"Now processing sub directory {sub}.")
        for split in splits:
            input_split = sub / split
            if not input_split.exists():
                continue

            rel = sub.relative_to(root)

            for label_dir in input_split.iterdir():
                if not label_dir.is_dir():
                    continue

                for img_file in tqdm(list(label_dir.iterdir()), desc=f"{split}/{label_dir.name}"):
                    if not img_file.is_file():
                        continue

                    try:
                        img = Image.open(img_file).convert("RGB")
                    except Exception:
                        print(f"⚠️ Could not open {img_file}")
                        continue

                    # # Get facial features
                    # features = get_facial_features(np.array(img), image_path=img_file)
                    # if features is None:
                    #     print(f"⚠️ No features found for {img_file}")
                    #     continue
                    
                    # # Get fixations
                    # fixations = sample_facial_feature_points_weighted(features, np.array(img), num_points=32)
                    # # fixation_subsets = {n: all_fixations[:n] for n in FIXATION_LEVELS}
                    # tensor_img = TF.to_tensor(img)
                    
                    # # --- storage for visualization ---
                    # rotated_list, foveated_list, logpolar_list = [], [], []
                    
                    # # Define targets
                    # if split == "train":
                    #     targets = [("train_upright", 3)]
                    # elif split == "valid":
                    #     targets = [("valid_upright", 2), ("valid_inverted", 1)]
                    # elif split == "test":
                    #     targets = [("test_upright", 2), ("test_inverted", 1)]
                    # else:
                    #     targets = []
                    
                    # for (fx, fy) in fixations:
                    #     for folder_name, inversion in targets:
                    #         rotated = rotate(tensor_img, inversion=inversion, center=(fx, fy))
                    #         foveated = foveation(rotated, center=(fx, fy))
                    #         C, H, W = foveated.shape
                    #         logpolar = logpolar_manual(foveated, (H, W), (224, 224), center=(fy, fx))
                    
                    #         rotated_np = tensor_to_bgr(rotated)
                    #         foveated_np = tensor_to_bgr(foveated)
                    #         logpolar_np = tensor_to_bgr(logpolar)
                    
                    #         rotated_list.append(rotated_np)
                    #         foveated_list.append(foveated_np)
                    #         logpolar_list.append(logpolar_np)
                    
                    #         # --- save transformed patch ---
                    #         out_img = TF.to_pil_image(logpolar.clamp(0, 1))
                    #         output_path = dest / f"4_fixations" / folder_name / label_dir.name
                    #         output_path.mkdir(parents=True, exist_ok=True)
                    #         filename = f"{img_file.stem}_fix{len(rotated_list):03d}.png"
                    #         out_img.save(output_path / filename)
                    
                    # # 🔍 visualize per-image pipeline
                    # visualize_pipeline_per_image(
                    #     base_img=np.array(img)[:, :, ::-1],  # convert RGB→BGR for cv2
                    #     fixations=fixations,
                    #     rotated_list=rotated_list,
                    #     foveated_list=foveated_list,
                    #     logpolar_list=logpolar_list,
                    #     img_name=f"{img_file.stem}"
                    # )

                    # Get facial features
                    features = get_facial_features(np.array(img), image_path=img_file)
                    if features is None:
                        print(f"⚠️ No features found for {img_file}")
                        continue

                    # Get 128 fixations and prepare cumulative subsets
                    all_fixations = sample_facial_feature_points_weighted(features, np.array(img), num_points=64)
                    fixation_subsets = {n: all_fixations[:n] for n in FIXATION_LEVELS}

                    tensor_img = TF.to_tensor(img)

                    # Define which folders & inversions to apply
                    if split == "train":
                        targets = [("train_upright", 3)]  # random [-15, 15]
                    elif split == "valid":
                        targets = [("valid_upright", 2), ("valid_inverted", 1)]  # 0°, 180°
                    elif split == "test":
                        targets = [("test_upright", 2), ("test_inverted", 1)]  # 0°, 180°
                    else:
                        targets = []

                    # For each fixation level
                    for n, fixations in fixation_subsets.items():
                        for i, (fx, fy) in enumerate(fixations):
                            for folder_name, inversion in targets:
                                rotated = rotate(tensor_img, inversion=inversion, center=(fx, fy))
                                foveated = foveation(rotated, center=(fx, fy))
                                C, H, W = foveated.shape
                                logpolar = logpolar_manual(
                                    foveated,
                                    (H, W),
                                    (224, 224),
                                    center=(fy, fx)
                                )

                                out_img = TF.to_pil_image(logpolar.clamp(0, 1))

                                # Save path: preprocessed_faces/<n>_fixations/train_upright/<identity>/<img>
                                output_path = dest / f"{n}_fixations" / folder_name / label_dir.name
                                output_path.mkdir(parents=True, exist_ok=True)

                                filename = f"{img_file.stem}_fix{i+1:03d}.png"
                                out_img.save(output_path / filename)


# ---------------------- Entrypoint ----------------------
if __name__ == "__main__":
    Path("32_identities_64_fixations_preprocessed_faces").mkdir(exist_ok=True)
    if len(sys.argv) > 1:
        process_dataset(sys.argv[1])
    else:
        process_dataset()

