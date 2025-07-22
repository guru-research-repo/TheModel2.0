import sys
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF

from transformation import *

print("Environment Initialized")

import matplotlib.pyplot as plt

def visualize_pipeline(samples):
    """
    samples: list of dicts with keys: 'original', 'crop', 'glabella', 'rotate', 'foveate', 'logpolar'
    """
    fig, axs = plt.subplots(nrows=len(samples), ncols=6, figsize=(18, 2.5 * len(samples)))
    headers = ["original", "crop", "glabella", "rotate", "foveate", "log polar"]

    for row, sample in enumerate(samples):
        for col, key in enumerate(headers):
            ax = axs[row][col] if len(samples) > 1 else axs[col]
            ax.imshow(sample[key])
            ax.axis("off")
            if row == 0:
                ax.set_title(key, fontsize=12)

    plt.tight_layout()
    plt.savefig("pipeline_visualization.png", dpi=150)

# Default dataset name and identity counts for 'faces'
DEFAULT_DATASET = "cars"
IDENTITY_COUNTS = [4, 8, 16, 32, 64, 128]
# IDENTITY_COUNTS = [4, 8]

def process_dataset(
    dataset: str = DEFAULT_DATASET,
    root_dir: str = "cars",
    processed_dir: str = "cars_CNN_data"
):
    """
    Processes images under `root_dir` into `processed_dir`:
      - CelebA_HQ_facial_identity_dataset:
          ~/data/CelebA_HQ_facial_identity_dataset/{train,valid,test}/<label>/images
      - faces:
          ~/data/faces/faces/{n}_identities/{train,valid,test}/<label>/images

    For each image:
      1. four_random_crops (tensor-only)
      2. rotate (random ±15° or 180° if split=='test')
      3. foveation
      4. logpolar_manual

    Outputs mirror input structure inside `processed_dir`.
    """
    root = Path(root_dir).expanduser()
    dest = Path(processed_dir).expanduser()
    subfol = "Cars_Dataset"

    # Determine subdirectories to process
    if dataset == "cars":
        base = root / subfol / subfol
        sub_dirs = [base / f"{n}_car_models" for n in IDENTITY_COUNTS]
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

            # Build corresponding output directory
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
                        img = Image.open(img_file).convert("RGB")
                    except Exception:
                        continue

                    # Convert to tensor and get four random crops
                    tensor_img = TF.to_tensor(img)
                    crops = four_random_crops(tensor_img)

                    ########################################################
                    # # Collect samples for visualization
                    samples = []  # New list to collect visual outputs
                    
                    # # Limit to 20 crops max
                    # max_crops = 20
                    # crop_count = 0
                    ########################################################


                    for i, tensor in enumerate(crops):
                        ####################################################
                    #     if crop_count >= max_crops:
                    #         break
                    
                    #     # Store original PIL and crop for visualization
                    #     crop = TF.to_pil_image(tensor)
                    
                    #     # Get glabella
                    #     center = get_glabella_or_center(tensor_to_bgr(tensor))
                    
                    #     # Glabella visualization
                    #     glabella_img = tensor_to_bgr(tensor).copy()
                    #     cv2.circle(glabella_img, center, 4, (0, 0, 255), thickness=-1)
                    #     glabella_vis = Image.fromarray(cv2.cvtColor(glabella_img, cv2.COLOR_BGR2RGB))
                    
                    #     # Rotate
                    #     rotated_tensor = rotate(tensor, inverse=(split == "test"), center=center)
                    #     rotated_img = TF.to_pil_image(rotated_tensor.clamp(0, 1))
                    
                    #     # Foveate
                    #     foveated_tensor = foveation(rotated_tensor, center=center)
                    #     foveated_np = foveated_tensor.permute(1, 2, 0).numpy()
                    #     foveated_bgr = cv2.cvtColor((foveated_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    #     cv2.circle(foveated_bgr, center, 4, (0, 0, 255), thickness=-1)
                    #     foveated_img = Image.fromarray(cv2.cvtColor(foveated_bgr, cv2.COLOR_BGR2RGB))
                    
                    #     # Log-polar
                    #     C, H, W = foveated_tensor.shape
                    #     logpolar_tensor = logpolar_manual(foveated_tensor, (H, W), (H, W), center=(center[1], center[0]))
                    #     logpolar_img = TF.to_pil_image(logpolar_tensor.clamp(0, 1))
                    
                    #     # Add to visualization samples
                    #     samples.append({
                    #         "original": img,
                    #         "crop": crop,
                    #         "glabella": glabella_vis,
                    #         "rotate": rotated_img,
                    #         "foveate": foveated_img,
                    #         "log polar": logpolar_img
                    #     })
                    #     crop_count += 1
                    # visualize_pipeline(samples)
                        ##############################################################
                        # Define which folders and angles (inversion values) to apply
                        if split == "train":
                            targets = [("train_upright", 3)]  # random [-15, 15]
                        elif split == "valid":
                            targets = [("valid_upright", 2), ("valid_inverted", 1)]  # 0°, 180°
                        elif split == "test":
                            targets = [("test_upright", 2), ("test_inverted", 1)]  # 0°, 180°
                        else:
                            targets = []

                        for folder_name, inversion in targets:
                            rotated = rotate(tensor, inversion=inversion)

                            ############################################################# UNCOMMENT BELOW
                            # foveated = foveation(rotated)
                            # C, H, W = foveated.shape
                            # logpolar = logpolar_manual(foveated, (H, W), (H, W))

                            # out_img = TF.to_pil_image(logpolar.clamp(0, 1))
                            #############################################################
                            
                            output_path = Path(processed_dir) / rel / folder_name / label_dir.name
                            # print("output_path", output_path)
                            # output_path = Path(processed_dir) / folder_name / rel / split / label_dir.name
                            output_path.mkdir(parents=True, exist_ok=True)

                            filename = f"{img_file.stem}_proc{i}.png"
                            # print("filename", filename)
                            ############################################################## UNCOMMENT BELOW
                            # out_img.save(output_path / filename)
                            ##############################################################
                            rotated_pil = TF.to_pil_image(rotated)
                            rotated_pil.save(output_path / filename)


def tensor_to_bgr(tensor: torch.Tensor) -> np.ndarray:
    """Convert a (C, H, W) float tensor to uint8 BGR image."""
    img_np = tensor.permute(1, 2, 0).numpy()
    return cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

if __name__ == "__main__":
    Path(processed_dir:="cars_CNN_data").mkdir(exist_ok=True)
    if len(sys.argv) > 1:
        process_dataset(sys.argv[1])
    else:
        process_dataset()
