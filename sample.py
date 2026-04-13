import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

from model_familiar import Model
# Assuming utils contains get_label_mapping and label_to_one_hot from your training script
from utils import get_label_mapping, label_to_one_hot 

# ============================================================
# ---------------- CONFIG ----------------
# ============================================================

MODEL_PATH = "familiarexp2_FACES_LPNet_16_fixations_50epoch_64_identities/familiarfaces_LP_best_model.pth"
NUM_CLASSES = 128
TEMPERATURE = 16.0

NUM_IDENTITIES = 10
BASE_IMAGES_PER_ID = 10
FIXATIONS_PER_IMAGE = 16

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# ---------------- PATHS ----------------
# ============================================================
FAMILIAR_ROOT_BASE = "familiar_faces" 
# FixationFaceDataset expects the identity folder included in the root
UNKNOWN_ROOT_OLD = "unknown_processed_data/salience10-48lp-mag/updated_faces/10_identities"

# ============================================================
# 1. Training-Aligned Dataset Class (For Known Faces)
# ============================================================
class SalienceDataset(Dataset):
    def __init__(self, root_dir: str, num_identities: int, split: str,
                 num_salient_points: int = 16):
        self.num_salient_points = num_salient_points

        self.data_dir = os.path.join(root_dir, f"{num_identities}_identities", split)
        if not os.path.isdir(self.data_dir):
            raise ValueError(f"Directory not found: {self.data_dir}")

        self.classes = sorted(
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d)) and not d.startswith('.')
        )

        self.map = get_label_mapping(type="updated_faces")
        self.samples = []  

        for ident in self.classes:
            ident_dir = os.path.join(self.data_dir, ident)
            base_dict = {}
            for fname in sorted(os.listdir(ident_dir)):
                if fname.endswith(".png") and "_proc" in fname:
                    base_num = fname.split("_proc")[0]  
                    base_dict.setdefault(base_num, []).append(os.path.join(ident_dir, fname))

            for base_num, proc_list in base_dict.items():
                proc_list = sorted(proc_list)  
                if len(proc_list) >= self.num_salient_points:
                    self.samples.append((base_num, proc_list, ident))

        print(f"Loaded {len(self.samples)} base images for split: {split} (SalienceDataset)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base_num, proc_list, label = self.samples[idx]
        chosen = proc_list[:self.num_salient_points]

        imgs = [TF.to_tensor(Image.open(p).convert("RGB")) for p in chosen]
        imgs = torch.stack(imgs, dim=0)  # (16, C, H, W)

        label_tensor = label_to_one_hot(label, self.map)
        return imgs, label_tensor

# ============================================================
# 2. Dynamic Label Dataset Class (For Unknown Faces)
# ============================================================
class FixationFaceDataset(Dataset):
    def __init__(self, root_dir, split):
        self.root_dir = os.path.join(root_dir, split)
        self.samples = []
        
        # Filter out hidden folders
        self.identities = sorted([
            d for d in os.listdir(self.root_dir) 
            if os.path.isdir(os.path.join(self.root_dir, d)) and not d.startswith('.')
        ])
        
        # Dynamically create mapping to bypass the missing labels error
        self.label_map = {name: i for i, name in enumerate(self.identities)}

        proc_pattern = re.compile(r"(\d+)_proc(\d+)\.png")

        for ident in self.identities:
            ident_dir = os.path.join(self.root_dir, ident)
            files = sorted(os.listdir(ident_dir))
            base_dict = {}

            for f in files:
                m = proc_pattern.match(f)
                if m:
                    base_id = int(m.group(1))
                    base_dict.setdefault(base_id, []).append(os.path.join(ident_dir, f))

            selected_bases = sorted(base_dict.keys())[:BASE_IMAGES_PER_ID]

            for base_id in selected_bases:
                fixation_paths = sorted(base_dict[base_id])
                if len(fixation_paths) < FIXATIONS_PER_IMAGE:
                    continue
                fixation_paths = fixation_paths[:FIXATIONS_PER_IMAGE]
                self.samples.append((fixation_paths, self.label_map[ident]))

        print(f"Loaded {len(self.samples)} base images for split: {split} (FixationFaceDataset)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fixation_paths, label_id = self.samples[idx]
        imgs = [TF.to_tensor(Image.open(p).convert("RGB")) for p in fixation_paths]
        imgs = torch.stack(imgs)  # (16, C, H, W)

        # Create dummy one-hot to satisfy dataloader shape requirements
        label = torch.zeros(NUM_CLASSES)
        label[label_id] = 1.0
        return imgs, label

import torch
from collections import defaultdict

# ==============================================================================
# METHOD 1: Variance of 16 fixations of 1 base image (Saccadic Instability)
# ==============================================================================
def compute_variance_across_fixations(model, loader, device):
    """
    Measures how much the representation jumps around AS the eye moves 
    across a single photo.
    """
    model.eval()
    all_fixation_vars = []

    with torch.no_grad():
        for inputs, _ in loader:
            B, n, C, H, W = inputs.shape  # n is the 16 fixations
            inputs_flat = inputs.reshape(B * n, C, H, W).to(device)

            _, h, _ = model(inputs_flat, return_rep=True)
            h = h.view(B, n, -1)  # (B, 16, hidden_units)

            # Calculate variance across the 16 fixations (dim=1)
            # Shape becomes (B, hidden_units)
            var_fixations = h.var(dim=1, unbiased=True) 
            all_fixation_vars.append(var_fixations.cpu())

    all_fixation_vars = torch.cat(all_fixation_vars, dim=0) # (100, hidden_units)
    
    # Average the instability across all 100 images, then across hidden units
    return all_fixation_vars.mean().item()

# ==============================================================================
# METHOD 2: Variance of 10 base images of 1 identity (Presentation Noise)
# -> THIS PROVES YOUR HYPOTHESIS
# ==============================================================================
def compute_variance_within_identity(model, loader, device):
    """
    Measures how 'hard it is to match' different photos of the SAME person.
    """
    model.eval()
    identity_reps = defaultdict(list)

    with torch.no_grad():
        for inputs, labels in loader:
            B, n, C, H, W = inputs.shape
            inputs_flat = inputs.reshape(B * n, C, H, W).to(device)

            _, h, _ = model(inputs_flat, return_rep=True)
            h = h.view(B, n, -1)

            # Collapse the 16 fixations into 1 stable percept per image
            h_base, _ = h.max(dim=1)  # (B, hidden_units)
            
            label_ids = labels.argmax(dim=1)

            # Group the photos by who is in them
            for i in range(B):
                ident_idx = int(label_ids[i].item())
                identity_reps[ident_idx].append(h_base[i].cpu())

    mean_vars_per_id = []
    for ident, reps in identity_reps.items():
        reps_tensor = torch.stack(reps)  # (10_images, hidden_units)
        
        # Calculate variance across the 10 photos of THIS person (dim=0)
        var_per_unit = reps_tensor.var(dim=0, unbiased=True) 
        mean_vars_per_id.append(var_per_unit.mean().item())

    # Average the noise across all 10 identities
    return sum(mean_vars_per_id) / len(mean_vars_per_id)

# ==============================================================================
# METHOD 3: Variance across all 100 faces (Feature Expressivity)
# -> THIS SATISFIES THE PROFESSOR'S EXACT PROMPT
# ==============================================================================
def compute_variance_across_dataset(model, loader, device):
    """
    Measures how spread out the representations are across the ENTIRE dataset.
    """
    model.eval()
    all_base_h = []

    with torch.no_grad():
        for inputs, _ in loader:
            B, n, C, H, W = inputs.shape
            inputs_flat = inputs.reshape(B * n, C, H, W).to(device)

            _, h, _ = model(inputs_flat, return_rep=True)
            h = h.view(B, n, -1)

            # Collapse the 16 fixations into 1 stable percept per image
            h_base, _ = h.max(dim=1)  # (B, hidden_units)
            all_base_h.append(h_base.cpu())

    # Stack all 100 unrelated photos together
    all_base_h = torch.cat(all_base_h, dim=0) # (100, hidden_units)
    
    # Calculate variance down the column of 100 faces (dim=0)
    var_per_unit = all_base_h.var(dim=0, unbiased=True) 
    
    # Average across hidden units
    return var_per_unit.mean().item()
    
# # ============================================================
# # Biologically Plausible Variance Computation
# # ============================================================
# # def compute_hidden_variance(model, loader, device):
# #     """
# #     Computes variance across base images after integrating fixations via MAX pooling.
# #     """
# #     model.eval()
# #     all_base_h = []

# #     with torch.no_grad():
# #         for inputs, _ in loader:  # Labels (_) are safely ignored here
# #             # inputs: (B, 16, C, H, W)
# #             B, n, C, H, W = inputs.shape
# #             inputs_flat = inputs.reshape(B * n, C, H, W).to(device)

# #             # Forward pass
# #             _, h, _ = model(inputs_flat, return_rep=True)
            
# #             # Reshape h back to group by base image: (B, 16, hidden_units)
# #             h = h.view(B, n, -1)

# #             # Max pooling integrates the strongest feature detection across all saccades
# #             h_base, _ = h.max(dim=1)  # shape: (B, hidden_units)
            
# #             all_base_h.append(h_base.cpu())

# #     # Concatenate all base images (e.g., 100 images)
# #     all_base_h = torch.cat(all_base_h, dim=0) 
    
# #     # Compute the variance of each hidden unit across the face dataset
# #     var_per_unit = all_base_h.var(dim=0, unbiased=True) 
    
# #     # Average the results over the hidden units
# #     return var_per_unit.mean().item()

# from collections import defaultdict
# import torch

# def compute_hidden_variance(model, loader, device):
#     """
#     Computes Intra-Identity Variance: How much the hidden representation 
#     fluctuates across different images of the SAME identity.
#     """
#     model.eval()
    
#     # Dictionary to group face representations by their true identity class
#     identity_reps = defaultdict(list)

#     with torch.no_grad():
#         for inputs, labels in loader:
#             B, n, C, H, W = inputs.shape
#             inputs_flat = inputs.reshape(B * n, C, H, W).to(device)

#             # Forward pass
#             _, h, _ = model(inputs_flat, return_rep=True)
            
#             # Reshape h back to (B, n, hidden_units)
#             h = h.view(B, n, -1)

#             # Max pooling integrates the strongest feature detection across all saccades
#             h_base, _ = h.max(dim=1)  # shape: (B, hidden_units)
            
#             # Extract class index from one-hot labels
#             label_ids = labels.argmax(dim=1)

#             # Group the base representations by their identity
#             for i in range(B):
#                 ident_idx = int(label_ids[i].item())
#                 identity_reps[ident_idx].append(h_base[i].cpu())

#     # Now compute the variance WITHIN each identity group
#     mean_vars_per_id = []
    
#     for ident, reps in identity_reps.items():
#         # reps_tensor shape: (num_images_for_this_person, hidden_units)
#         reps_tensor = torch.stack(reps)  
        
#         # Variance across the different images of THIS specific identity
#         # (If the model is confident, this will be low. If confused, this will be high)
#         var_per_unit = reps_tensor.var(dim=0, unbiased=True) 
        
#         # Average across the hidden units for this identity
#         mean_vars_per_id.append(var_per_unit.mean().item())

#     # Finally, average the intra-identity variances across all 10 identities
#     final_variance = sum(mean_vars_per_id) / len(mean_vars_per_id)

#     return final_variance

# ============================================================
# MAIN
# ============================================================
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

    # ---------------- Build datasets ----------------
    # Mix and match dataloaders to bypass the KeyError
    loaders = {
        "Known Train": DataLoader(
            SalienceDataset(FAMILIAR_ROOT_BASE, NUM_IDENTITIES, "train", FIXATIONS_PER_IMAGE), 
            batch_size=8, shuffle=False
        ),
        "Known Up": DataLoader(
            SalienceDataset(FAMILIAR_ROOT_BASE, NUM_IDENTITIES, "valid", FIXATIONS_PER_IMAGE), 
            batch_size=8, shuffle=False
        ),
        "Known Inv": DataLoader(
            SalienceDataset(FAMILIAR_ROOT_BASE, NUM_IDENTITIES, "test", FIXATIONS_PER_IMAGE), 
            batch_size=8, shuffle=False
        ),
        "Unknown Up": DataLoader(
            FixationFaceDataset(UNKNOWN_ROOT_OLD, "valid"), 
            batch_size=8, shuffle=False
        ),
        "Unknown Inv": DataLoader(
            FixationFaceDataset(UNKNOWN_ROOT_OLD, "test"), 
            batch_size=8, shuffle=False
        ),
    }

    # ---------------- Compute variances ----------------
    print("\n=== Computing variances ===")
    results = {}
    for name, loader in loaders.items():
        print(f"Processing {name}...")
        results[name] = compute_variance_across_dataset(model, loader, device)
        print(f"{name}: {results[name]:.6f}")

        # results_method1[name] = compute_variance_across_fixations(model, loader, device)
        # results_method2[name] = compute_variance_within_identity(model, loader, device)
        # results_method3[name] = compute_variance_across_dataset(model, loader, device)

    # ---------------- Plot ----------------
    plt.figure(figsize=(8, 5))
    plt.bar(results.keys(), results.values(), color=['blue', 'green', 'orange', 'red', 'purple'])
    plt.ylabel("Average Hidden Variance (Inter-Image)")
    plt.title("Expressivity of Hidden Units Across Identity Conditions")
    plt.tight_layout()
    plt.savefig("variance_barplot_exp2.png", dpi=150)
    plt.close()

    print("Saved variance_barplot_exp2.png")

if __name__ == "__main__":
    main()