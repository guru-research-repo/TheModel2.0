import os
import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from Datasets import SalienceDatasetBatched, SalienceDataset

# ---- Create dataset ----
dataset = SalienceDatasetBatched(
    root_dir="processed_data/dogs1k/updated_objects",
    num_identities=32,
    split="train",
    num_salient_points=16,
    lp=False
)

# ---- Create dataloader ----
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# ---- Get one batch ----
images, labels = next(iter(loader))  # images: (B, C, H, W)
# b,n,c,h,w = images.shape
# images = images[:,0,...].reshape(-1,c,h,w)
labels = labels.argmax(dim=1)
print(labels)

# ---- Create grid ----
grid = vutils.make_grid(images, nrow=4, padding=2)

# ---- Save to file ----
save_path = "out/sample_dog_batch_cnn_1.png"
vutils.save_image(grid, save_path)

print(f"Saved sample grid to {save_path}")