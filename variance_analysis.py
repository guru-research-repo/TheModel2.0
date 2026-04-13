import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from model_familiar import Model

# ============================================================
# ---------------- CONFIG ----------------
# ============================================================

MODEL_PATH = "familiarexp1_temp2_FACES_LPNet_32_fixations_50epoch_128_identities/familiarfaces_LP_best_model.pth"
# "FACES_LPNet_32_fixations_40_epochblock_Martha/resnet18_20260129_162806.pth" #old model before finetuning
# "familiarexp2_FACES_LPNet_16_fixations_50epoch_64_identities/familiarfaces_LP_best_model.pth" # expt 2

NUM_CLASSES = 128
TEMPERATURE = 2.0

NUM_IDENTITIES = 10
BASE_IMAGES_PER_ID = 10
FIXATIONS_PER_IMAGE = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# ---------------- PATHS (EDIT IF NEEDED) ----------------
# ============================================================

FAMILIAR_ROOT = "familiar_faces/10_identities"
UNKNOWN_ROOT = "unknown_processed_data/salience10-48lp-mag/updated_faces/10_identities"

# ============================================================
# Transform (match training!)
# ============================================================

transform = T.Compose([
    T.ToTensor()
])

# ============================================================
# Custom dataset for grouped fixations
# ============================================================
class FixationFaceDataset(Dataset):
    """
    Returns:
        image_tensor: (32, C, H, W)
        label: one-hot
    """

    def __init__(self, root_dir, split):
        self.root_dir = os.path.join(root_dir, split)
        self.samples = []
        # Filter out hidden folders like .ipynb_checkpoints
        self.identities = sorted([
            d for d in os.listdir(self.root_dir) 
            if os.path.isdir(os.path.join(self.root_dir, d)) and not d.startswith('.')
        ])
        # self.identities = sorted(os.listdir(self.root_dir))
        self.label_map = {name: i for i, name in enumerate(self.identities)}
        print("labelmap", self.label_map)

        proc_pattern = re.compile(r"(\d+)_proc(\d+)\.png")

        for ident in self.identities:
            # print("ident", ident)
            ident_dir = os.path.join(self.root_dir, ident)
            files = sorted(os.listdir(ident_dir))

            # group by base image
            base_dict = {}

            for f in files:
                m = proc_pattern.match(f)
                if m:
                    base_id = int(m.group(1))
                    base_dict.setdefault(base_id, []).append(
                        os.path.join(ident_dir, f)
                    )

            # take first N base images
            selected_bases = sorted(base_dict.keys())[:BASE_IMAGES_PER_ID]

            for base_id in selected_bases:
                fixation_paths = sorted(base_dict[base_id])

                if len(fixation_paths) < FIXATIONS_PER_IMAGE:
                    continue

                fixation_paths = fixation_paths[:FIXATIONS_PER_IMAGE]

                self.samples.append(
                    (fixation_paths, self.label_map[ident])
                )

        print(f"{split} loaded: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fixation_paths, label_id = self.samples[idx]
        # print("fixation_paths, label_id", fixation_paths, label_id)

        imgs = []
        for p in fixation_paths:
            img = Image.open(p).convert("RGB")
            img = transform(img)
            imgs.append(img)

        imgs = torch.stack(imgs)  # (32, C, H, W)

        label = torch.zeros(NUM_CLASSES)
        label[label_id] = 1.0

        return imgs, label


# ============================================================
# Hidden variance computation
# ============================================================
import torch

import torch

import torch
import torch.nn.functional as F

# def compute_hidden_variance(model, loader, device, aggregation='max'): #professor
#     """
#     Compute hidden unit variance for base images across their fixations.

#     Args:
#         model: torch model with return_rep=True
#         loader: DataLoader yielding (B, n, C, H, W), labels
#         device: torch.device
#         aggregation: str, method to collapse fixations ('max', 'mean', or 'weighted')
#     Returns:
#         float: mean hidden unit variance
#     """
#     model.eval()
#     all_base_h = []

#     with torch.no_grad():
#         for inputs, labels in loader:
#             # inputs: (B, n, C, H, W)
#             B, n, C, H, W = inputs.shape
#             inputs = inputs.to(device)

#             # Flatten fixations to treat them as independent batch items: (B*n, C, H, W)
#             inputs_flat = inputs.reshape(B * n, C, H, W)

#             # Forward pass to get hidden units and logits
#             logits, h, probs = model(inputs_flat, return_rep=True)
            
#             # h shape: (B*n, hidden_units)
#             # Reshape back to group by base image: (B, n, hidden_units)
#             h = h.view(B, n, -1)

#             # --- COLLAPSE FIXATIONS STRATEGIES ---
#             if aggregation == 'max':
#                 # Take the strongest activation for each unit across all 16 fixations
#                 h_base, _ = h.max(dim=1) 
                
#             elif aggregation == 'mean':
#                 # Standard average (dilutes strong single fixations)
#                 h_base = h.mean(dim=1)
#             else:
#                 raise ValueError("Aggregation must be 'max', 'weighted', or 'mean'.")

#             all_base_h.append(h_base.cpu())

#     # Concatenate all base images across batches
#     all_base_h = torch.cat(all_base_h, dim=0)  # (total_base_images, hidden_units)
#     print(f"Total base images collected: {all_base_h.shape[0]} using '{aggregation}' aggregation.")

#     # Variance per hidden unit over all base images
#     var_per_unit = all_base_h.var(dim=0, unbiased=True)  # (hidden_units,)
    
#     # Final mean over hidden units
#     mean_var = var_per_unit.mean().item()

#     print("Hidden unit variance shape:", var_per_unit.shape)
#     print("Mean hidden unit variance:", mean_var)

#     return mean_var


# import torch
# from collections import defaultdict

# def compute_hidden_variance(model, loader, device): #variance of each identity separately - alex
#     """
#     Compute hidden unit variance per identity, then average over identities.

#     Steps:
#     1. For each base image, collapse fixations → hidden vector.
#     2. Group hidden vectors by identity.
#     3. Compute variance per hidden unit **within each identity**.
#     4. Average variance over hidden units for that identity.
#     5. Average these per-identity variances across all identities.

#     Args:
#         model: torch model with return_rep=True
#         loader: DataLoader yielding (B, n, C, H, W), labels
#         device: torch.device
#     Returns:
#         float: mean hidden unit variance across identities
#     """

#     model.eval()
#     identity_h = defaultdict(list)

#     with torch.no_grad():
#         for inputs, labels in loader:
#             # inputs: (B, n, C, H, W)
#             B, n, C, H, W = inputs.shape
#             inputs = inputs.to(device)

#             # Flatten fixations: (B*n, C, H, W)
#             inputs_flat = inputs.reshape(B * n, C, H, W)

#             # Forward pass to get hidden units
#             logits, h, probs = model(inputs_flat, return_rep=True)
#             # h: (B*n, hidden_units)

#             # Reshape back to (B, n, hidden_units)
#             h = h.view(B, n, -1)

#             # Collapse fixations per base image by mean
#             h_base = h.mean(dim=1)  # (B, hidden_units)

#             # Convert labels to indices if one-hot
#             if labels.dim() > 1:
#                 label_ids = labels.argmax(dim=1)
#             else:
#                 label_ids = labels
#             label_ids = label_ids.cpu()

#             # Group hidden vectors by identity
#             for vec, lab in zip(h_base.cpu(), label_ids):
#                 identity_h[int(lab)].append(vec)

#     # Compute per-identity variance
#     per_identity_var = []
#     for ident in sorted(identity_h.keys()):
#         acts = torch.stack(identity_h[ident])  # (num_base_images, hidden_units)
#         var_units = acts.var(dim=0, unbiased=True)  # variance per hidden unit
#         mean_var = var_units.mean()  # mean across hidden units for this identity
#         per_identity_var.append(mean_var)

#     # Final mean across all identities
#     final_score = torch.tensor(per_identity_var).mean().item()

#     print("Per-identity variances:", per_identity_var)
    # print("Mean variance across identities:", final_score)

    # return final_score
    
# def compute_hidden_variance(model, loader, device):
#     model.eval()
#     activations = []

#     with torch.no_grad():
#         for inputs, labels in loader:
#             # print("loader",  len(loader), loader)
#             inputs, labels = inputs.to(device), labels.to(device)
#             label_ids = labels.argmax(dim=1) if labels.dim() > 1 else labels
#             # print("label_ids",label_ids)
#             #inputs = inputs.to(device) #torch.Size([8, 16, 3, 180, 180])
#             # print("inputs", inputs.shape)

#             # (B, 32, C, H, W) → flatten
#             B, n, C, H, W = inputs.shape
#             inputs = inputs.reshape(-1, C, H, W)

#             logits, h, probs = model(inputs, return_rep=True) 
#             # print("h", h)
#             ##### no need
#             logits = logits.reshape(B, 16, -1).sum(dim=1)
#             preds = logits.argmax(dim=1)
#             # print("predictions", preds)
#             # print("predictions shape", preds.shape)
#             print("accuracy",((preds == label_ids).float().mean().item()))
#             # logits torch.Size([128, 128])
#             # h torch.Size([128, 256])
#             # probs torch.Size([128, 256])
#             activations.append(h.cpu())

#     acts = torch.cat(activations, dim=0) #acts torch.Size([1600, 256])
#     var_per_unit = acts.var(dim=0) #var_per_unit torch.Size([256])

#     return var_per_unit.mean().item()

import torch

def compute_hidden_variance(model, loader, device):
    """
    Computes the variance of the sigmoids across the 16 fixations of each image,
    averaged over the hidden units, and then averaged across the dataset.
    """
    model.eval()
    all_image_variances = []

    with torch.no_grad():
        for inputs, _ in loader:  # Labels (_) are ignored
            B, n, C, H, W = inputs.shape  # B=batch_size, n=16 fixations
            inputs_flat = inputs.reshape(B * n, C, H, W).to(device)

            # Forward pass
            # Note: in your Model, 'h' is exactly the sigmoid output when stochastic=False
            _, h, _ = model(inputs_flat, return_rep=True)
            
            # Reshape h back to (B, n, hidden_units) -> (Batch, 16, 256)
            h = h.view(B, n, -1)

            # --- PROFESSOR'S INSTRUCTION STEP 1 & 2 ---
            # "take all 16 fixations of 1 image, compute the variance of each 
            # of the N (256?) sigmoids in the hidden layer over the 16 fixations"
            
            # var(dim=1) calculates the variance across the 16 fixations.
            # Shape becomes: (B, hidden_units)
            var_per_unit = h.var(dim=1, unbiased=True)

            # --- PROFESSOR'S INSTRUCTION STEP 3 ---
            # "average the N numbers you get."
            
            # mean(dim=1) averages the variances across the 256 hidden units.
            # Shape becomes: (B,) -> A single number representing the variance for ONE image.
            avg_var_per_image = var_per_unit.mean(dim=1)
            
            all_image_variances.append(avg_var_per_image.cpu())

    # Combine all the individual image variances from the loader
    all_image_variances = torch.cat(all_image_variances, dim=0)
    
    # --- PROFESSOR'S INSTRUCTION STEP 4 ---
    # "Repeat for each of the 10 identities, average the 10 numbers you get."
    # (Since there are 10 images per identity, this averages all 100 resulting numbers)
    
    final_variance = all_image_variances.mean().item()

    return final_variance

def compute_hidden_variance_array(model, loader, device):
    """
    Computes the variance of the sigmoids across the 16 fixations of each image.
    Returns the raw array of variances (one per image) instead of just the mean.
    """
    model.eval()
    all_image_variances = []

    with torch.no_grad():
        for inputs, _ in loader:  
            B, n, C, H, W = inputs.shape  
            inputs_flat = inputs.reshape(B * n, C, H, W).to(device)

            _, h, _ = model(inputs_flat, return_rep=True)
            h = h.view(B, n, -1)

            # Variance across the 16 fixations
            var_per_unit = h.var(dim=1, unbiased=True)
            # Average across the 256 hidden units
            avg_var_per_image = var_per_unit.mean(dim=1)
            
            all_image_variances.append(avg_var_per_image.cpu())

    # Combine all images (e.g., 100 images)
    all_image_variances = torch.cat(all_image_variances, dim=0)
    
    # RETURN THE FULL ARRAY (numpy array for easy plotting)
    return all_image_variances.numpy()



def main():

    # ---------------- Load model ----------------
    model = Model(
        size=180,
        num_classes=NUM_CLASSES,
        pretrained=False,
        T=TEMPERATURE,
    ).to(device)

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state, strict=False)
    model.stochastic = False

    for p in model.parameters():
        p.requires_grad = False

    print("Model loaded and frozen.")

    # ============================================================
    # Build datasets
    # ============================================================
    familiar_train = FixationFaceDataset(FAMILIAR_ROOT, "train")
    familiar_valid = FixationFaceDataset(FAMILIAR_ROOT, "valid")
    familiar_test = FixationFaceDataset(FAMILIAR_ROOT, "test")

    unknown_valid = FixationFaceDataset(UNKNOWN_ROOT, "valid")
    unknown_test = FixationFaceDataset(UNKNOWN_ROOT, "test")

    loaders = {
        "Known Train": DataLoader(familiar_train, batch_size=256//16, shuffle=False),
        "Known Up": DataLoader(familiar_valid, batch_size=256//16, shuffle=False),
        "Known Inv": DataLoader(familiar_test, batch_size=256//16, shuffle=False),
        "Unknown Up": DataLoader(unknown_valid, batch_size=256//16, shuffle=False),
        "Unknown Inv": DataLoader(unknown_test, batch_size=256//16, shuffle=False),
    }

    # ============================================================
    # Compute variances
    # ============================================================

    print("\n=== Computing variances ===")

    results = {}
    for name, loader in loaders.items():
        print(f"Processing {name}...")
        results[name] = compute_hidden_variance(model, loader, device)
        print(f"{name}: {results[name]:.6f}")

    # ============================================================
    # Plot
    # ============================================================

    plt.figure(figsize=(6, 4))
    plt.bar(results.keys(), results.values())
    plt.ylabel("Average Hidden Variance")
    plt.title("Hidden Unit Variance Comparison")
    plt.tight_layout()
    plt.savefig("familiarexp1_temp2_FACES_LPNet_32_fixations_50epoch_128_identities/variance_barplot_exp1.png", dpi=150)
    # plt.savefig("variance_barplot_oldmodelbeforefinetune.png", dpi=150)
    plt.close()

    print("Saved variance_barplot.png")

    
    # ============================================================
    # MAIN
    # ============================================================
    import seaborn as sns
    import pandas as pd
    
    # ---------------- Compute variances ----------------
    print("\n=== Computing variances ===")
    results_data = [] # List to hold data for our DataFrame
    
    for name, loader in loaders.items():
        print(f"Processing {name}...")
        
        # Get the array of 100 variances for this condition
        variances = compute_hidden_variance_array(model, loader, device)
        
        # Log the mean just to print it to the console
        print(f"{name} Mean Variance: {variances.mean():.6f}")
        
        # Append each individual image's variance to our data list
        for v in variances:
            results_data.append({"Condition": name, "Variance": v})
    
    # Convert to a Pandas DataFrame for Seaborn
    df = pd.DataFrame(results_data)
    
    # ---------------- Plot ----------------
    plt.figure(figsize=(10, 6))
    
    # 1. Draw the violin plot to show the density distribution
    sns.violinplot(
        x="Condition", 
        y="Variance", 
        data=df, 
        inner="quartile", # Shows the quartiles and median inside the violin
        palette="pastel",
        alpha=0.6
    )
    
    # 2. Overlay the swarm plot to show every single image as a dot
    sns.swarmplot(
        x="Condition", 
        y="Variance", 
        data=df, 
        color="black", 
        alpha=0.7, 
        size=4
    )
    
    plt.ylabel("Saccadic Variance (Averaged across units)")
    plt.title("Distribution of Saccadic Instability Across Identity Conditions")
    plt.tight_layout()
    plt.savefig("familiarexp1_temp2_FACES_LPNet_32_fixations_50epoch_128_identities/variance_distribution_exp1.png", dpi=150)
    plt.close()
    
    print("Saved variance_distribution_exp2.png")


# ============================================================
if __name__ == "__main__":
    main()