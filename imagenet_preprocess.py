import os
import sys
import random
import shutil
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF
from multiprocessing import Pool

from transformation import *

# Config
DEFAULT_DATASET = "ImageNet_objects"
CLASS_GROUP_SIZES = [4, 8, 16, 32, 64, 128]
SOURCE_DIR = Path("data/ImageNet_objects/filtered_train_images_0")
TARGET_ROOT = Path("data") / DEFAULT_DATASET
PROCESSED_DIR = Path("processed_data")


def precompute_class_splits(class_folders, seed=42):
    """
    Returns a dict of {class_code: {train: [...], valid: [...], test: [...]}}
    """
    split_map = {}
    random.seed(seed)

    for class_path in class_folders:
        cls_code = class_path.name
        images = sorted(class_path.glob("*.JPEG"))
        if not images:
            continue

        random.shuffle(images)
        total = len(images)
        n_train = int(0.8 * total)
        n_valid = int(0.1 * total)

        split_map[cls_code] = {
            "train": images[:n_train],
            "valid": images[n_train:n_train + n_valid],
            "test": images[n_train + n_valid:]
        }

    return split_map

def shuffle_classes(n=128, seed=42):
    all_classes = [d for d in SOURCE_DIR.iterdir() if d.is_dir()]
    random.seed(seed)
    random.shuffle(all_classes)
    return all_classes[:n]

def split_and_copy(args):
    class_path, out_dir, split_map = args
    cls_code = class_path.name

    if cls_code not in split_map:
        return f"[{cls_code}] skipped (not in split map)"

    for split, img_list in split_map[cls_code].items():
        out_class_dir = out_dir / split / cls_code / "images"
        out_class_dir.mkdir(parents=True, exist_ok=True)
        for img in img_list:
            shutil.copy2(img, out_class_dir / img.name)

    return f"[{cls_code}] copied using consistent splits."

def structure_dataset(class_folders, split_map):
    for n in CLASS_GROUP_SIZES:
        out_dir = TARGET_ROOT / f"{n}_objects"
        selected_classes = class_folders[:n]

        print(f"\nCreating {n}_objects dataset with consistent splits...")
        with Pool(processes=os.cpu_count()) as pool:
            args = [(cls_path, out_dir, split_map) for cls_path in selected_classes]
            results = pool.map(split_and_copy, args)

def apply_transformations(dataset: str = DEFAULT_DATASET,
                          root_dir: str = "data",
                          processed_dir: str = "processed_data"):
    root = Path(root_dir).expanduser()
    dest = Path(processed_dir).expanduser()
    base = root / dataset
    sub_dirs = [base / f"{n}_objects" for n in CLASS_GROUP_SIZES]
    splits = ["train", "valid", "test"]

    for sub in sub_dirs:
        print(f"Processing subdirectory: {sub}")
        for split in splits:
            input_split = sub / split
            if not input_split.exists():
                continue

            output_split = dest / sub.relative_to(root) / split
            output_split.mkdir(parents=True, exist_ok=True)

            for label_dir in input_split.iterdir():
                if not label_dir.is_dir():
                    continue

                out_label = output_split / label_dir.name
                out_label.mkdir(exist_ok=True)
                image_dir = label_dir / "images"

                for img_file in image_dir.glob("*.*"):
                    try:
                        img = Image.open(img_file).convert("RGB")
                        tensor_img = TF.to_tensor(img)
                        crops = four_random_crops(tensor_img)

                        for i, tensor in enumerate(crops):
                            tensor = rotate(tensor, inverse=(split == "test"))
                            tensor = foveation(tensor)
                            C, H, W = tensor.shape
                            tensor = logpolar_manual(tensor, (H, W), (H, W))

                            out_img = TF.to_pil_image(tensor.clamp(0, 1))
                            out_img.save(out_label / f"{img_file.stem}_proc{i}.png")
                    except Exception as e:
                        continue


def main():
    PROCESSED_DIR.mkdir(exist_ok=True)

    # Step 1: Shuffle 128 class folders
    shuffled_classes = shuffle_classes()

    # Step 2: Precompute and lock-in train/val/test splits
    split_map = precompute_class_splits(shuffled_classes)

    # Step 3: Use consistent splits across groupings
    structure_dataset(shuffled_classes, split_map)

    # Step 4: Apply image transformations
    apply_transformations()


if __name__ == "__main__":
    main()
