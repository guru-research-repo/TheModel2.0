import random
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from utils import *
from pipeline import build_transform_pipeline
import yaml

# -------------------------
# Configuration
# -------------------------
# Choose the dataset to load:
#   - "celeb":  only supports splits "train" and "test"
#   - "faces":  supports identity counts {4, 8, 16, 32, 64, 128}
#               and splits "train", "valid", "test"
dataset_name   = "faces"  # Options: "celeb", "faces"
identity_count = 4        # Only used when dataset_name == "faces"
split          = "test"   # For "celeb": {"train", "test"};
                            # for "faces": {"train", "valid", "test"}
# ------------------------------------------------------------------------
# Build Transformation Pipeline
# ------------------------------------------------------------------------

# Load the config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Build the transform pipeline
train_transform = build_transform_pipeline(config)

# ------------------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------------------

# load_dataset parameters:
#   dataset  : str   -> which dataset to load ("celeb" or "faces")
#   identity : int   -> number of distinct identities (only for "faces")
#   task     : str   -> which split to load ("train", "valid", "test")
#   transform: nn.Sequential -> transformations to apply to the dataset
ds = load_dataset(
    dataset=dataset_name,
    identity=identity_count,
    task=split,
    transform=build_transform_pipeline(config, inversion=True)
)

# -------------------------
# Pick a random index and save crops
# -------------------------

import random
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

# Pick a random index
index = random.randint(0, len(ds) - 1)
index= 0
crop, label = ds[index]

print(f"Crop shape: {crop.shape}, dtype: {crop.dtype}")
print(f"LABEL in main: {label}")

# Create a figure
fig, ax = plt.subplots(figsize=(4, 4))

# If crop is not a tensor (rare), convert it
if not isinstance(crop, torch.Tensor):
    crop = TF.to_tensor(crop)

# Convert (C, H, W) → (H, W, C) for plotting
crop_img = crop.permute(1, 2, 0)

# Show the crop
ax.imshow(crop_img)
ax.axis('off')
ax.set_title(f"Label: {label}")

# Save the figure
plt.tight_layout()
plt.savefig("random_sample_crop.png")
print(f"Saved random crop index {index} with label '{label}' to random_sample_crop.png")


# # ------------------------------------------------------------------------
# # Display Example Images
# # ------------------------------------------------------------------------
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torch
import random

num_samples = 20
rows, cols = 5, 4
indices = random.sample(range(len(ds)), num_samples)  # pick 20 random indices
indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
print(f"Dataset size: {len(ds)}")

fig, axes = plt.subplots(rows, cols, figsize=(12, 15))  # (width, height)

for i, idx in enumerate(indices):
    img, label = ds[idx]
    print(f"Crop shape: {img.shape}, dtype: {img.dtype}")
    print(f"LABEL in main: {label}")

    if not isinstance(img, torch.Tensor):
        img = TF.to_tensor(img)
    img = img.permute(1, 2, 0)
    #img = img.float().clamp(0, 1)
    img = (img - img.min()) / (img.max() - img.min())
    
    row = i // cols
    col = i % cols
    ax = axes[row, col]
    ax.imshow(img)
    ax.set_title(f"Label: {label}", fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.savefig("sample_crops_grid_inverted.png")
print("Saved figure to sample_crops_grid.png!")
