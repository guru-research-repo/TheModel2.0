import sys
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF
from transformation import *
from multiprocessing import Pool

# Number of worker processes
NUM_WORKERS = 96

# Default dataset name and identity counts for 'faces'
DEFAULT_DATASET = "faces"
IDENTITY_COUNTS = [4, 8, 16, 32, 64, 128]


def process_image(task):
    img_file, out_label, split = task
    try:
        # Load and convert to RGB
        img = Image.open(img_file).convert("RGB")
    except Exception as e:
        print(f"Failed to open {img_file}: {e}")
        return

    # To tensor
    tensor_img = TF.to_tensor(img)

    # 1. Four random crops (expects tensor in / out)
    crops = four_random_crops(tensor_img)

    for i, tensor in enumerate(crops):
        # 2. Rotate (inverse for test split)
        tensor = rotate(tensor, inverse=(split == "test"))
        # 3. Foveation
        tensor = foveation(tensor)
        # 4. Log‐polar
        C, H, W = tensor.shape
        tensor = logpolar_manual(tensor, (H, W), (H, W))

        # Save result
        out_img = TF.to_pil_image(tensor.clamp(0, 1))
        filename = f"{img_file.stem}_proc{i}.png"
        out_img.save(out_label / filename)


def process_dataset(
    dataset: str = DEFAULT_DATASET,
    root_dir: str = "data",
    processed_dir: str = "processed_data"
):
    root = Path(root_dir)
    dest = Path(processed_dir)

    # Decide which subdirectories to process
    if dataset == "faces":
        sub_dirs = [root / "faces" / "faces" / f"{n}_identities" for n in IDENTITY_COUNTS]
    else:
        sub_dirs = [root / dataset]

    splits = ["train", "valid", "test"]

    for sub in sub_dirs:
        for split in splits:
            input_split = sub / split
            if not input_split.exists():
                continue

            # Mirror the folder structure under processed_dir
            rel = sub.relative_to(root)
            output_split = dest / rel / split
            output_split.mkdir(parents=True, exist_ok=True)

            # Gather all (img_path, out_label_dir, split) tasks
            tasks = []
            for label_dir in input_split.iterdir():
                if not label_dir.is_dir():
                    continue
                out_label = output_split / label_dir.name
                out_label.mkdir(exist_ok=True)
                for img_file in label_dir.iterdir():
                    if img_file.is_file():
                        tasks.append((img_file, out_label, split))

            # Parallel processing
            if tasks:
                with Pool(NUM_WORKERS) as pool:
                    pool.map(process_image, tasks)


if __name__ == "__main__":
    # Ensure the top‐level processed_data folder exists
    Path(processed_dir := "processed_data").mkdir(exist_ok=True)

    if len(sys.argv) > 1:
        # Pass the dataset name (e.g., "faces" or "CelebA_HQ_facial_identity_dataset")
        process_dataset(sys.argv[1])
    else:
        process_dataset()
