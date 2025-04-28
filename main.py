from utils import *
from pipeline import build_transform_pipeline
import yaml
import random
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------

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
    transform=train_transform
)

# ------------------------------------------------------------------------
# Display Example Images
# ------------------------------------------------------------------------

num_samples = 5  
indices = random.sample(range(len(ds)), num_samples)

fig, axes = plt.subplots(num_samples, 4, figsize=(12, 10))
for row_idx, idx in enumerate(indices):
    crops, label = ds[idx]  # crops is a list of 4 images
    #print(f"Sample {idx}: Number of crops = {len(crops)}")  
    for col_idx in range(4):
        if col_idx < len(crops):  # Check if the crop exists
            ax = axes[row_idx, col_idx]
            img = crops[col_idx]
            if not isinstance(img, torch.Tensor):   
                img = TF.to_tensor(img) # Converts PIL Image to Tensor
            img = img.permute(1, 2, 0)  # (C, H, W) -> (H, W, C) for matplotlib
            ax.imshow(img)
            if col_idx == 0:
                ax.set_ylabel(f"Label: {label}", size=10)
            ax.axis('off')
        else:
            axes[row_idx, col_idx].axis('off')  
            axes[row_idx, col_idx].text(0.5, 0.5, "No crop", ha="center", va="center", fontsize=12)

plt.tight_layout()
plt.savefig("sample_crops.png")
print("Saved figure to sample_crops.png!")
