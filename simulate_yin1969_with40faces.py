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
        # print("ident_dir", ident_dir)
        base_dict = {}
        for fname in sorted(os.listdir(ident_dir)):
            # print("fname", fname)
            if fname.endswith(".png") and "_proc" in fname:
                base_num = fname.split("_proc")[0]
                base_dict.setdefault(base_num, []).append(os.path.join(ident_dir, fname))
        
        if not base_dict:
            continue
            
        chosen_base = list(base_dict.keys())[0]
        # print("chosen_base", chosen_base)
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
# =============================================================================
# 3. Yin Simulation Logic (Updated for 40 Study / 24 Test)
# =============================================================================
def run_condition(model, device, familiar_root, unknown_root, study_split, test_split, p_noise, sigma=2.0, num_study=40, num_test=24):
    """
    Runs one of the 4 specific Yin conditions with exactly 40 study items and 24 test pairs.
    """
    set_seed(42) # Ensures the 24 sampled test identities are consistent across all runs
    # 1. LOAD DATA
    # Load 40 familiar identities for study (Only first 10 fixations)
    study_data = load_yin_trial_data(familiar_root, num_study, study_split, num_fixations=10, offset=0)
    # Load all 40 familiar identities for testing (ALL 32 fixations)
    test_old_data_all = load_yin_trial_data(familiar_root, num_study, test_split, num_fixations=32, offset=0) 
    # Load 24 unknown identities for the test pairs (ALL 32 fixations)
    unknown_data = load_yin_trial_data(unknown_root, num_test, test_split, num_fixations=32, offset=0)
    
    # # 1. LOAD DATA
    # # Load 40 familiar identities for study
    # study_data = load_yin_trial_data(familiar_root, num_study, study_split, offset=0)
    # # Load all 40 familiar identities for potential testing (using offset 10)
    # test_old_data_all = load_yin_trial_data(familiar_root, num_study, test_split, offset=10)
    # # Load 24 entirely unknown identities for the test pairs
    # unknown_data = load_yin_trial_data(unknown_root, num_test, test_split, offset=0)
    
    memory_bank = {}
    
    with torch.no_grad():
        # ---------------------------------------------------------------------
        # STUDY PHASE (40 Faces)
        # ---------------------------------------------------------------------
        for ident, imgs in study_data.items():
            imgs = imgs.to(device)
            model.stochastic = True 
            _, h, _ = model(imgs, return_rep=True) 
            memory_bank[ident] = apply_binomial_noise(h.cpu(), p_noise)
            
        # ---------------------------------------------------------------------
        # TEST PHASE SETUP (Subset 24 Old, 24 New)
        # ---------------------------------------------------------------------
        correct_2afc = 0
        all_study_identities = list(study_data.keys())
        
        # Randomly select exactly 24 identities out of the 40 to be tested
        test_old_identities = random.sample(all_study_identities, num_test)
        unknown_identities = list(unknown_data.keys())
        
        actual_test_len = min(len(test_old_identities), len(unknown_identities))
        
        # ---------------------------------------------------------------------
        # TEST LOOP (24 Pairs)
        # ---------------------------------------------------------------------
        for i in range(actual_test_len):
            ident_old = test_old_identities[i]
            ident_new = unknown_identities[i]
            
            # Fetch the specific representations
            imgs_old = test_old_data_all[ident_old].to(device)
            imgs_new = unknown_data[ident_new].to(device)
            
            model.stochastic = True
            _, h_old, _ = model(imgs_old, return_rep=True)
            _, h_new, _ = model(imgs_new, return_rep=True)

            # Apply Retrieval Noise
            h_old = apply_binomial_noise(h_old.cpu(), p_noise)
            h_new = apply_binomial_noise(h_new.cpu(), p_noise)
            
            # Compare against the 40-item Memory Bank
            score_old = compute_familiarity_score(h_old, memory_bank, sigma)
            score_new = compute_familiarity_score(h_new, memory_bank, sigma)
            
            if score_old > score_new:
                correct_2afc += 1
                
    # Return accuracy based on the 24 test pairs
    return correct_2afc / actual_test_len

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
    for p in np.arange(0.0, 0.75, 0.05):
        acc = run_condition(model, device, familiar_path, unknown_path, "valid", "valid", p)
        print(f"Noise {p:.2f} -> Upright-Upright Accuracy: {acc*100:.2f}%")
        # We want the lowest noise that starts pushing accuracy off 100% towards the mid-90s
        if acc <= 0.96 and ideal_noise is None: # 80% -> p=0.33, 90% -> p=0.20, 70% -> p=0.36, 100% -> p=0.0
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