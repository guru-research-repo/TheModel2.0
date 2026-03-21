from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF
import torch
from trans import *
from salience_trans import *
import main_salience
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
            for img_path in label_dir.iterdir():
                if img_path.suffix in ['.jpg', '.png']: 
                    self.samples.append((label_dir.name, img_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, img_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        img_tensor = TF.to_tensor(img)  # (C,H,W)

        return {
            "image": img_tensor,
            "label": label,
            "stem": img_path.stem
        }

num_identities = 128
num_fixations = 16
batch_size = 128
root = 'face' # dest
dataset_name='faces' # source
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for split in ['test', 'valid', 'train']: 
    split_dir = Path(f'data/{dataset_name}/{dataset_name}/{num_identities}_identities/{split}') # directory w/ subdirectories (AdamRippon,Alicia,...) with images num.jpg 
    if split == 'test':
        split_dir = Path(f'data/{dataset_name}/{dataset_name}/{num_identities}_identities/valid')
    updated_save_dir = Path(f'processed_data/{root}/updated_objects/{split}')
    cnn_save_dir = Path(f'processed_data/{root}/cnn_objects/{split}')
    
    updated_save_dir.mkdir(parents=True, exist_ok=True)
    cnn_save_dir.mkdir(parents=True, exist_ok=True)

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
                    unit="img")

    # Create pipeline to transform images
    pipeline = SaliencePipeline(split, num_salient_points=num_fixations, device=device).to(device) #LP and CNN
    pipeline.eval()

    with torch.no_grad():
        for batch in loader:
            t0 = time.perf_counter()
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label']
            stems = batch['stem']
            t1 = time.perf_counter()
            transformed_imgs_lp, transformed_imgs_cnn = pipeline(images) # torch.tensor(B,N,C,H,W)
            t2 = time.perf_counter()
            B, N, C, H, W = transformed_imgs_lp.shape

            for b in range(B):
                label = labels[b]
                stem = stems[b]
                # print(b, label, stem)
                Path(f'processed_data/{root}/updated_objects/{split}/{label}').mkdir(parents=True, exist_ok=True)
                Path(f'processed_data/{root}/cnn_objects/{split}/{label}').mkdir(parents=True, exist_ok=True)
        
                torch.save({
                    'lp': (transformed_imgs_lp[b].clamp(0,1)).cpu(),
                    'cnn': (transformed_imgs_cnn[b].clamp(0,1)).cpu()
                    },
                    f'processed_data/{root}/updated_objects/{split}/{label}/{stem}_proc.pt')
                
            t3 = time.perf_counter()
            pbar.update(images.size(0))
            print(
                f"load→gpu: {t1 - t0:.3f}s | "
                f"infer: {t2 - t1:.3f}s | "
                f"save: {t3 - t2:.3f}s"
            )
