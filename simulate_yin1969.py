import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms.functional as TF
from model_familiar import Model

# =============================================================================
# 0. Reproducibility
# =============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =============================================================================
# 1. Helper Functions
# =============================================================================
def load_yin_trial_data(root_dir, num_identities, split, num_fixations=10, offset=0):
    data_dir = os.path.join(root_dir, f"{num_identities}_identities", split)
    if not os.path.isdir(data_dir):
        raise ValueError(f"Directory not found: {data_dir}")

    classes = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    
    samples = {}
    for ident in classes:
        ident_dir = os.path.join(data_dir, ident)
        base_dict = {}
        for fname in sorted(os.listdir(ident_dir)):
            if fname.endswith(".png") and "_proc" in fname:
                base_num = fname.split("_proc")[0]
                base_dict.setdefault(base_num, []).append(os.path.join(ident_dir, fname))
        
        if not base_dict:
            continue
            
        chosen_base = list(base_dict.keys())[0]
        proc_list = sorted(base_dict[chosen_base])
        
        chosen_fixations = proc_list[offset : offset + num_fixations]
        if len(chosen_fixations) < num_fixations:
            continue 
            
        imgs = [TF.to_tensor(Image.open(p).convert("RGB")) for p in chosen_fixations]
        imgs = torch.stack(imgs, dim=0)
        samples[ident] = imgs

    return samples

def apply_binomial_noise(binary_tensor, p_noise):
    # STRICT CLAMPING to prevent 100% inversion anomaly
    # p_noise = min(max(p_noise, 0.0), 0.5)
    if p_noise == 0.0:
        return binary_tensor
    # print("binary tensor", binary_tensor)
    
    noise_mask = torch.rand_like(binary_tensor) < p_noise
    noisy_tensor = torch.logical_xor(binary_tensor.bool(), noise_mask).float()
    # print("noisy tensor", noisy_tensor)
    return noisy_tensor

# =============================================================================
# 2. Memory Math (Barrington KDE)
# =============================================================================
def compute_p_f_given_c(f, M_c, sigma):
    dists = torch.sum((M_c - f) ** 2, dim=1) 
    kernels = torch.exp(-dists / (2 * sigma ** 2))
    return torch.mean(kernels)

def compute_familiarity_score(F_test, memory_bank, sigma):
    best_score = -float('inf')
    for c, M_c in memory_bank.items():
        # print("c,mc from memory bank=", c, M_c)
        log_likelihood = 0.0
        for i in range(F_test.size(0)):
            p = compute_p_f_given_c(F_test[i], M_c, sigma)
            log_likelihood += torch.log(p + 1e-12).item()
        if log_likelihood > best_score:
            best_score = log_likelihood
    return best_score

# =============================================================================
# 3. Yin Simulation Logic
# =============================================================================
def run_condition(model, device, familiar_root, unknown_root, study_split, test_split, p_noise, sigma=2.0):
    """
    Runs one of the 4 specific Yin conditions.
    """
    set_seed(42)
    study_data = load_yin_trial_data(familiar_root, 10, study_split, offset=0)
    test_old_data = load_yin_trial_data(familiar_root, 10, test_split, offset=10)
    unknown_data = load_yin_trial_data(unknown_root, 10, test_split, offset=0)
    
    memory_bank = {}
    
    with torch.no_grad():
        # STUDY
        for ident, imgs in study_data.items():
            imgs = imgs.to(device)
            model.stochastic = True 
            _, h, _ = model(imgs, return_rep=True) 
            memory_bank[ident] = apply_binomial_noise(h.cpu(), p_noise)
            
        # TEST
        correct_2afc = 0
        identities = list(study_data.keys())
        print("identities", identities)
        unknown_identities = list(unknown_data.keys())
        print("unknown identities", unknown_identities)
        min_len = min(len(identities), len(unknown_identities))
        
        for i in range(min_len):
            ident_old = identities[i]
            print("ident_old", ident_old)
            ident_new = unknown_identities[i]
            print("ident_new", ident_new)
            
            imgs_old = test_old_data[ident_old].to(device)
            imgs_new = unknown_data[ident_new].to(device)
            
            model.stochastic = True
            _, h_old, _ = model(imgs_old, return_rep=True)
            _, h_new, _ = model(imgs_new, return_rep=True)

            # print("h_old before noise", h_old)
            # print("h_new before noise", h_new)
            h_old = apply_binomial_noise(h_old.cpu(), p_noise)
            # print("h_old after noise", h_old)
            h_new = apply_binomial_noise(h_new.cpu(), p_noise)
            # print("h_new after noise", h_new)
            
            score_old = compute_familiarity_score(h_old, memory_bank, sigma)
            print("score old", score_old)
            
            score_new = compute_familiarity_score(h_new, memory_bank, sigma)
            print("score new", score_new)
            
            if score_old > score_new:
                correct_2afc += 1
                
    return correct_2afc / min_len

# =============================================================================
# 4. Main Execution Pipeline
# =============================================================================
if __name__ == "__main__":
    set_seed(42) # ENSURE REPRODUCIBILITY
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = Model(size=180, num_classes=128, pretrained=False, T=2.0).to(device)
    pretrained_path = "familiarexp2_temp2_FACES_LPNet_32_fixations_50epoch_128_identities/familiarfaces_LP_best_model.pth"
    model.load_state_dict(torch.load(pretrained_path, map_location=device), strict=False)
    model.eval()
    
    familiar_path = "familiar_faces"
    unknown_path = "unknown_processed_data/salience10-48lp-mag/updated_faces"
    
    # 1. Find the ideal noise for Upright-Upright to match Yin's 97.78%
    print("--- Searching for Model Noise equivalent to Human Upright Performance ---")
    ideal_noise = None
    for p in np.arange(0.0, 0.45, 0.01):
        acc = run_condition(model, device, familiar_path, unknown_path, "valid", "valid", p)
        print(f"Noise {p:.2f} -> Upright-Upright Accuracy: {acc*100:.2f}%")
        # We want the lowest noise that starts pushing accuracy off 100% towards the mid-90s
        if acc <= 0.9 and ideal_noise is None: # 80% -> p=0.33, 90% -> p=0.20, 70% -> p=0.36, 100% -> p=0.0
            ideal_noise = p
            break
        
    
    # Fallback if it stays at 100% too long
    if ideal_noise is None: ideal_noise = 0.25 
    print(f"\n[!] Selected Noise parameter p={ideal_noise:.2f} to simulate human neural noise.\n")
    
    # 2. Run all 4 Yin Conditions
    results = []
    conditions = [
        ("Upright", "Upright", "valid", "valid"), # Yin Exp 1
        ("Inverted", "Inverted", "test", "test"), # Yin Exp 1
        ("Upright", "Inverted", "valid", "test"), # Yin Exp 2
        ("Inverted", "Upright", "test", "valid")  # Yin Exp 2
    ]
    
    for study_cond, test_cond, study_split, test_split in conditions:
        acc = run_condition(model, device, familiar_path, unknown_path, study_split, test_split, ideal_noise)
        results.append({
            "Study Presentation": study_cond,
            "Test Presentation": test_cond,
            "Model Accuracy": f"{acc*100:.2f}%"
        })
        
    df_model = pd.DataFrame(results)

    print("=====================================================")
    print(" TABLE 1: YIN (1969) HUMAN BASELINES (DERIVED)")
    print("=====================================================")
    df_yin = pd.DataFrame({
        "Study Presentation": ["Upright", "Inverted", "Upright", "Inverted"],
        "Test Presentation": ["Upright", "Inverted", "Inverted", "Upright"],
        "Yin Human Accuracy": ["96.29%", "81.88%", "84.13%", "78.58%"]
    })
    print(df_yin.to_string(index=False))
    # # 3. Print Final Tables
    # print("=====================================================")
    # print(" TABLE 1: YIN (1969) HUMAN BASELINES (DERIVED)")
    # print("=====================================================")
    # df_yin = pd.DataFrame({
    #     "Study Presentation": ["Upright", "Inverted", "Upright", "Inverted"],
    #     "Test Presentation": ["Upright", "Inverted", "Inverted", "Upright"],
    #     "Yin Human Accuracy": ["97.78%", "89.13%", "90.48%", "87.15%"]
    # })
    # print(df_yin.to_string(index=False))
    print("\n")
    print("=====================================================")
    print(f" TABLE 2: MODEL PERFORMANCE (Noise p={ideal_noise:.2f})")
    print("=====================================================")
    print(df_model.to_string(index=False))

# import os
# import torch
# import random
# import numpy as np
# from PIL import Image
# import torchvision.transforms.functional as TF
# from model_familiar import Model
# from tqdm import tqdm

# # =============================================================================
# # 1. Helper Functions for Data & Noise
# # =============================================================================

# def load_yin_trial_data(root_dir, num_identities, split, num_fixations=10, offset=0):
#     """
#     Simulates the Yin (1969) paradigm. 
#     `offset` allows us to use different fixations for study vs. test.
#     """
#     data_dir = os.path.join(root_dir, f"{num_identities}_identities", split)
#     if not os.path.isdir(data_dir):
#         raise ValueError(f"Directory not found: {data_dir}")

#     classes = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    
#     samples = {}
#     for ident in classes:
#         ident_dir = os.path.join(data_dir, ident)
        
#         base_dict = {}
#         for fname in sorted(os.listdir(ident_dir)):
#             if fname.endswith(".png") and "_proc" in fname:
#                 base_num = fname.split("_proc")[0]
#                 base_dict.setdefault(base_num, []).append(os.path.join(ident_dir, fname))
        
#         if not base_dict:
#             continue
            
#         chosen_base = list(base_dict.keys())[0]
#         proc_list = sorted(base_dict[chosen_base])
        
#         # SLICE USING OFFSET: e.g., offset 0 gets 0:10. offset 10 gets 10:20.
#         chosen_fixations = proc_list[offset : offset + num_fixations]
#         if len(chosen_fixations) < num_fixations:
#             continue 
            
#         imgs = [TF.to_tensor(Image.open(p).convert("RGB")) for p in chosen_fixations]
#         imgs = torch.stack(imgs, dim=0)
#         samples[ident] = imgs

#     return samples

# def apply_binomial_noise(binary_tensor, p_noise):
#     # Clamp noise to a maximum of 0.5 to prevent wrap-around bit flipping
#     p_noise = min(max(p_noise, 0.0), 0.5)
#     if p_noise == 0.0:
#         return binary_tensor
    
#     noise_mask = torch.rand_like(binary_tensor) < p_noise
#     noisy_tensor = torch.logical_xor(binary_tensor.bool(), noise_mask).float()
#     return noisy_tensor
# # def load_yin_trial_data(root_dir, num_identities, split, num_fixations=10):
# #     # familiar root = 'familiar_faces', num_identities=10, split='valid'
# #     """
# #     Simulates the Yin (1969) paradigm by grabbing exactly ONE base image 
# #     and its first `num_fixations` for a specific set of identities.
# #     """
# #     data_dir = os.path.join(root_dir, f"{num_identities}_identities", split)
# #     if not os.path.isdir(data_dir):
# #         raise ValueError(f"Directory not found: {data_dir}")

# #     classes = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    
# #     samples = {}
# #     for ident in classes:
# #         ident_dir = os.path.join(data_dir, ident) # reached 'AdamRippon'
        
# #         # Group by base image number to ensure we only pick ONE base image per identity
# #         base_dict = {}
# #         for fname in sorted(os.listdir(ident_dir)):
# #             if fname.endswith(".png") and "_proc" in fname:
# #                 base_num = fname.split("_proc")[0]
# #                 base_dict.setdefault(base_num, []).append(os.path.join(ident_dir, fname))
        
# #         if not base_dict:
# #             continue
            
# #         # Pick the first available base image for this identity
# #         chosen_base = list(base_dict.keys())[0]
# #         proc_list = sorted(base_dict[chosen_base])
        
# #         # Take exactly the first 10 fixations
# #         chosen_fixations = proc_list[:num_fixations]
# #         if len(chosen_fixations) < num_fixations:
# #             continue # Skip if an image somehow has fewer than 10 fixations
            
# #         # Load tensors
# #         imgs = [TF.to_tensor(Image.open(p).convert("RGB")) for p in chosen_fixations]
# #         imgs = torch.stack(imgs, dim=0) # Shape: (10, 3, 180, 180)
# #         samples[ident] = imgs

# #     return samples

# def apply_binomial_noise(binary_tensor, p_noise):
#     """
#     Flips bits in the binary tensor with probability p_noise.
#     """
#     if p_noise == 0.0:
#         return binary_tensor
    
#     # Create a boolean mask where True means "flip this bit"
#     noise_mask = torch.rand_like(binary_tensor) < p_noise
#     # XOR operation flips bits: 1^1 = 0, 0^1 = 1
#     noisy_tensor = torch.logical_xor(binary_tensor.bool(), noise_mask).float()
#     return noisy_tensor

# # =============================================================================
# # 2. NIMBLE Memory Model (Barrington et al., Eq 11)
# # =============================================================================

# def compute_p_f_given_c(f, M_c, sigma):
#     """
#     Computes Equation 11: P(f|c) = (1 / |M_c|) * sum( N(f, m, sigma) )
#     f: single test fixation vector (256,)
#     M_c: memory bank for class c (10, 256)
#     sigma: kernel bandwidth parameter
#     """
#     # L2 distance squared (equivalent to Hamming distance for binary vectors)
#     # dists shape: (10,)
#     dists = torch.sum((M_c - f) ** 2, dim=1) 
    
#     # Gaussian/RBF Kernel
#     kernels = torch.exp(-dists / (2 * sigma ** 2))
    
#     # Average over the memorized fragments
#     return torch.mean(kernels)

# def compute_familiarity_score(F_test, memory_bank, sigma):
#     """
#     Given a set of test fixations F_test (10, 256), evaluate how familiar the face is.
#     We compute the log-likelihood of the face belonging to the best-matching identity.
#     """
#     best_score = -float('inf')
    
#     for c, M_c in memory_bank.items():
#         log_likelihood = 0.0
#         # Sum the log probabilities across all 10 fixations of the test face
#         for i in range(F_test.size(0)):
#             p = compute_p_f_given_c(F_test[i], M_c, sigma)
#             # Add small epsilon to avoid log(0)
#             log_likelihood += torch.log(p + 1e-12).item()
            
#         if log_likelihood > best_score:
#             best_score = log_likelihood
            
#     return best_score

# # =============================================================================
# # 3. Main Yin (1969) Simulation
# # =============================================================================

# def run_yin_simulation(model, device, familiar_root, unknown_root, split, p_noise, sigma=1.0):
#     model.eval()
    
#     # LOAD WITH OFFSETS
#     # Study uses fixations 0-9
#     study_data = load_yin_trial_data(familiar_root, 10, split, offset=0)
#     # Test OLD uses fixations 10-19 of the same base images
#     test_old_data = load_yin_trial_data(familiar_root, 10, split, offset=0)
#     # Test NEW uses fixations 0-9 of the unknown faces
#     unknown_data = load_yin_trial_data(unknown_root, 10, split, offset=0)
    
#     # -------------------------------------------------------------------------
#     # STUDY PHASE (Building the Memory Bank)
#     # -------------------------------------------------------------------------
#     memory_bank = {}
    
#     with torch.no_grad():
#         for ident, imgs in study_data.items():
#             imgs = imgs.to(device)
#             model.stochastic = True 
#             _, h, _ = model(imgs, return_rep=True) 
            
#             noisy_h = apply_binomial_noise(h.cpu(), p_noise)
#             memory_bank[ident] = noisy_h  
            
#     # -------------------------------------------------------------------------
#     # TEST PHASE (2-Alternative Forced Choice)
#     # -------------------------------------------------------------------------
#     correct_2afc = 0
#     total_trials = 0
    
#     identities = list(study_data.keys())
#     unknown_identities = list(unknown_data.keys())
#     min_len = min(len(identities), len(unknown_identities))
    
#     with torch.no_grad():
#         for i in range(min_len):
#             ident_old = identities[i]
#             ident_new = unknown_identities[i]
            
#             # CRITICAL CHANGE: Use test_old_data here so the model cannot just pixel-match
#             imgs_old = test_old_data[ident_old].to(device)
#             imgs_new = unknown_data[ident_new].to(device)
            
#             model.stochastic = True
#             _, h_old, _ = model(imgs_old, return_rep=True)
#             _, h_new, _ = model(imgs_new, return_rep=True)
            
#             h_old = apply_binomial_noise(h_old.cpu(), p_noise)
#             h_new = apply_binomial_noise(h_new.cpu(), p_noise)
            
#             score_old = compute_familiarity_score(h_old, memory_bank, sigma)
#             score_new = compute_familiarity_score(h_new, memory_bank, sigma)
            
#             if score_old > score_new:
#                 correct_2afc += 1
#             total_trials += 1
            
#     accuracy = correct_2afc / total_trials
#     return accuracy
# # def run_yin_simulation(model, device, familiar_root, unknown_root, split, p_noise, sigma=1.0):
# #     model.eval()
    
# #     # 1. Load Data (10 identities, 1 image each, 10 fixations)
# #     # Using 'valid' for upright, 'test' for inverted as per your pipeline
# #     print(f"Loading '{split}' data...")
# #     study_data = load_yin_trial_data(familiar_root, 10, split)
# #     unknown_data = load_yin_trial_data(unknown_root, 10, split)
    
# #     # -------------------------------------------------------------------------
# #     # STUDY PHASE (Building the Memory Bank)
# #     # -------------------------------------------------------------------------
# #     memory_bank = {}
    
# #     with torch.no_grad():
# #         for ident, imgs in study_data.items():
# #             imgs = imgs.to(device)
# #             # We must use stochastic=True to get the binary samples 'h'
# #             model.stochastic = True 
# #             _, h, _ = model(imgs, return_rep=True) 
            
# #             # Add post-hoc binomial noise
# #             noisy_h = apply_binomial_noise(h.cpu(), p_noise)
# #             memory_bank[ident] = noisy_h  # Store in memory M_c
            
# #     # -------------------------------------------------------------------------
# #     # TEST PHASE (2-Alternative Forced Choice)
# #     # -------------------------------------------------------------------------
# #     correct_2afc = 0
# #     total_trials = 0
    
# #     # Ensure we have the same number of unknown faces to pair with
# #     identities = list(study_data.keys())
# #     unknown_identities = list(unknown_data.keys())
# #     min_len = min(len(identities), len(unknown_identities))
    
# #     with torch.no_grad():
# #         for i in range(min_len):
# #             ident_old = identities[i] #show faces
# #             ident_new = unknown_identities[i]
            
# #             imgs_old = study_data[ident_old].to(device)
# #             imgs_new = unknown_data[ident_new].to(device)
            
# #             # Sample representations
# #             model.stochastic = True
# #             _, h_old, _ = model(imgs_old, return_rep=True)
# #             _, h_new, _ = model(imgs_new, return_rep=True)
            
# #             # Add noise to test samples
# #             h_old = apply_binomial_noise(h_old.cpu(), p_noise)
# #             h_new = apply_binomial_noise(h_new.cpu(), p_noise)
            
# #             # The Model Decision: Compare Familiarity
# #             score_old = compute_familiarity_score(h_old, memory_bank, sigma)
# #             score_new = compute_familiarity_score(h_new, memory_bank, sigma)
            
# #             # 2AFC: The model must choose the face that feels "more familiar"
# #             if score_old > score_new:
# #                 correct_2afc += 1
# #             total_trials += 1
            
# #     accuracy = correct_2afc / total_trials
# #     return accuracy

# if __name__ == "__main__":
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
#     # Initialize your finetuned Temp 2 Model
#     model = Model(size=180, num_classes=128, pretrained=False, T=2.0).to(device)
#     pretrained_path = "familiarexp2_temp2_FACES_LPNet_32_fixations_50epoch_128_identities/familiarfaces_LP_best_model.pth"
#     model.load_state_dict(torch.load(pretrained_path, map_location=device), strict=False)
    
#     familiar_path = "familiar_faces"
#     unknown_path = "unknown_processed_data/salience10-48lp-mag/updated_faces"
#     print("Offset of 0")
#     # 1. Fit noise to UPRIGHT faces ('valid' split)
#     print("--- Fitting Noise to Upright Faces ---")
#     noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 1.0]
    
#     # Sigma controls the "width" of the memory kernel. 1.0 to 5.0 are good bounds.
#     sigma = 2.0 
    
#     best_noise = 0.0
#     for p in noise_levels:
#         acc = run_yin_simulation(model, device, familiar_path, unknown_path, split="valid", p_noise=p, sigma=sigma)
#         print(f"Noise {p:.2f} -> Upright 2AFC Accuracy: {acc*100:.2f}%")
#         # Target: Yin's human performance on upright faces is generally ~85-95%. 
#         # You will manually identify which noise 'p' best matches the human baseline.
    
#     # 2. Test the selected noise level on INVERTED faces ('test' split)
#     # Manually set this to whichever 'p' gave you human-like upright performance above
#     # chosen_noise = 0.15 
#     chosen_noises = [0.0, 0.05, 0.10, 0.15, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 1.0]
#     for chosen_noise in chosen_noises:
#         inv_acc = run_yin_simulation(model, device, familiar_path, unknown_path, split="test", p_noise=chosen_noise, sigma=sigma)
#         print(f"Noise {chosen_noise:.2f} -> Inverted 2AFC Accuracy: {inv_acc*100:.2f}%")