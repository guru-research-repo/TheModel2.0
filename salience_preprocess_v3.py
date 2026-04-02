from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF
import torch
from trans import *
from salience_trans import *
from Datasets import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

# Step 1: Load raw datasets
    # assumes data is stored in data/faces_cleaned/faces/{num_identities}_identities/{split}/{identity}/img#.jpg
# Step 2: Initialize Pipeline
# Step 3: Transform images using pipeline
# Step 4: Save transformed images
    #create save dir 
    #create subdir for each person
    #save each transformed image in subdir w/ img#_proc#

class FaceDataset(Dataset):
    def __init__(self, split_dir: Path):
        """
        split_dir structure:
        split_dir/
            person1/
                img1.jpg
                img2.jpg
            person2/
                img1.jpg
                ...
        """
        self.samples = []
        self.split_dir = split_dir

        for label_dir in split_dir.iterdir():
            if not label_dir.is_dir():
                continue
            self.samples.append(label_dir)
            # for img_path in label_dir.iterdir():
            #     if img_path.suffix.lower() in ['.jpg', '.png', '.jpeg']: 
            #         self.samples.append((label_dir.name, img_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label_path = self.samples[idx]

        images = []
        stems = []

        for p in label_path.iterdir():
            if p.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                img = Image.open(p).convert("RGB")
                images.append(TF.to_tensor(img))
                stems.append(p.stem)

        images = torch.stack(images)  # (N, C, H, W)

        return {
            "image": images,
            "label": label_path.name,
            "stem": stems
        }

num_identities = 128
num_fixations = 16
batch_size = 1
root = 'dogs1k-v3' # dest
dataset_name='dogs1k' # source
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for split in ['test', 'valid', 'train']: 
    split_dir = Path(f'data/{dataset_name}/{dataset_name}/{num_identities}_identities/{split}') # directory w/ subdirectories (AdamRippon,Alicia,...) with images num.jpg 
    if split == 'test':
        split_dir = Path(f'data/{dataset_name}/{dataset_name}/{num_identities}_identities/valid')
    updated_save_dir = Path(f'processed_data/{root}/updated_objects/{split}')
    # cnn_save_dir = Path(f'processed_data/{root}/cnn_objects/{split}')
    
    updated_save_dir.mkdir(parents=True, exist_ok=True)
    # cnn_save_dir.mkdir(parents=True, exist_ok=True)

    dataset = FaceDataset(split_dir)
    
    loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    pbar = tqdm(total=len(loader.dataset),
                    desc=f"{split}",
                    unit="identity")

    # Create pipeline to transform images
    pipeline = SaliencePipeline(split, num_salient_points=num_fixations, device=device).to(device) #LP and CNN
    pipeline.eval()

    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            images = batch['image'][0].to(device, non_blocking=True)
            label = batch['label'][0]
            stems = batch['stem']
            t1 = time.perf_counter()
            salience_points = pipeline.sample_salience_points(images) # torch.tensor(B,N,2)
            t2 = time.perf_counter()
            B, N, _ = salience_points.shape

            label_dir = Path(f'processed_data/{root}/updated_objects/{split}/{label}')
            label_dir.mkdir(parents=True, exist_ok=True)

            file_path = label_dir / "salience.pt"

            torch.save({
                "salience_points": salience_points.cpu(),  # (N, num_fixations, 2)
                "stems": stems,
                "label": label
            }, file_path)
                
            t3 = time.perf_counter()
            pbar.update(1)
            print(
                f"load→gpu: {t1 - t0:.3f}s | "
                f"infer: {t2 - t1:.3f}s | "
                f"save: {t3 - t2:.3f}s"
            )
            t0 = time.perf_counter()
