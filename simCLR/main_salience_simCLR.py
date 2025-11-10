# main_salience.py
import os
import numpy as np
import pandas as pd
import datetime
import torch
import torch.nn.functional as F
from utils_new import *
from model import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from Datasets_new import *

# ---------------- Supervised Contrastive (label-based SimCLR) ----------------
def supcon_loss(features: torch.Tensor,
                labels: torch.Tensor,
                temperature: float = 0.2) -> torch.Tensor:
    """
    Supervised Contrastive loss (Khosla et al., NeurIPS 2020).
    Positives = all samples in the batch that share the same class label.

    Args:
        features: (N, D) L2-normalized embeddings
        labels:   (N,) int labels (class indices)
        temperature: scalar temperature

    Returns:
        scalar loss
    """
    device = features.device
    labels = labels.view(-1)
    N = features.size(0)

    # Cosine similarities scaled by temperature
    sim = torch.matmul(features, features.T) / temperature  # (N, N)

    # Remove self-sim terms
    mask = torch.eye(N, dtype=torch.bool, device=device)
    sim = sim.masked_fill(mask, -1e9)

    # Positive mask: same class (and not self)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (~mask)  # (N, N)

    # For numerical stability
    sim_max, _ = torch.max(sim, dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    exp_sim = torch.exp(sim)
    denom = exp_sim.sum(dim=1, keepdim=True)  # sum over all j≠i

    # Sum exp(sim) over positives for each anchor
    pos_exp = (exp_sim * pos_mask).sum(dim=1)

    # Avoid log(0)
    eps = 1e-8
    loss = -torch.log((pos_exp + eps) / (denom.squeeze(1) + eps))

    # If an anchor has no positives (rare if batch has one sample of a class), exclude it
    valid = pos_mask.any(dim=1)
    if valid.any():
        return loss[valid].mean()
    else:
        # Fallback: if batch accidentally has all unique labels
        return loss.mean()


def main(lp=True, dataset_name: str = "salience", faces_data="updated"):
    os.makedirs("output_4", exist_ok=True)

    # ---------------------------- Configuration -----------------------------
    identity_counts = [32]
    salient_counts  = [4]  # you said 2 fixations for now
    splits          = ["train_upright", "valid_upright", "valid_inverted"]
    epoch_block     = 40
    total_epochs    = 40
    num_workers     = 4
    batch_size      = 1024      # training batch of single images (B, C, H, W)
    lr              = 1e-3
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device:', device)

    # Contrastive hyperparams
    temperature = 0.2
    lambda_contrastive = 0.1   # weight for SupCon term (tune 0.05–0.2)

    # Helper: map epoch → identity/fixations (your logic kept)
    def identity_for_epoch(epoch: int) -> int:
        idx = (epoch - 1) // epoch_block
        return identity_counts[idx]

    def salient_points_for_epoch(epoch: int) -> int:
        idx = (epoch - 1) // epoch_block
        return salient_counts[idx]

    history     = []
    history_acc = []

    for s in salient_counts:
        torch.cuda.empty_cache()

        # IMPORTANT: ensure model knows number of classes
        num_classes = 32
        model = Model(size=224 if faces_data == 'updated' else 180,
                      num_classes=num_classes,
                      proj_dim=128).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        ce_criterion = torch.nn.CrossEntropyLoss()

        valid_batch_size = batch_size // s  # you used this earlier

        for epoch in range(1, total_epochs + 1):
            ident = identity_for_epoch(epoch)
            num_salient_points = s

            datasets = make_datasets(ident, num_salient_points, faces_data)

            train_loader = DataLoader(
                datasets["train_upright"],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
            )
            valid_loader = DataLoader(
                datasets["valid_upright"],
                batch_size=valid_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
            test_loader  = DataLoader(
                datasets["valid_inverted"],
                batch_size=valid_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )

            # ------------------------------- TRAIN ----------------------------
            model.train()
            correct = 0
            total   = 0
            train_accs = []

            pbar = tqdm(total=len(train_loader.dataset),
                        desc=f"Epoch {epoch}/{total_epochs}",
                        unit="img")

            for inputs, labels in train_loader:
                inputs = inputs.to(device)   # (B, C, H, W) from SalienceDatasetBatched
                labels = labels.to(device)   # one-hot

                # class indices
                label_ids = labels.argmax(dim=1) if labels.dim() > 1 else labels  # (B,)

                optimizer.zero_grad()

                # features + logits
                feats, logits = model(inputs, return_feat=True)  # feats: (B, d), logits: (B, num_classes)

                # classification loss
                ce_loss = ce_criterion(logits, label_ids)

                # supervised contrastive loss (label-based positives)
                con_loss = supcon_loss(feats, label_ids, temperature=temperature)

                # combine
                loss = ce_loss + lambda_contrastive * con_loss
                loss.backward()
                optimizer.step()

                # accuracy bookkeeping
                preds = logits.argmax(dim=1)
                correct += (preds == label_ids).sum().item()
                total   += label_ids.size(0)
                batch_acc = correct / total
                train_accs.append(batch_acc)

                pbar.update(inputs.size(0))
                pbar.set_postfix(acc=f"{batch_acc*100:.2f}%",
                                 ce=f"{ce_loss.item():.3f}",
                                 con=f"{con_loss.item():.3f}")

            pbar.close()
            epoch_acc = correct / total if total > 0 else 0.0
            print(f"→ Epoch {epoch}/{total_epochs} — Acc: {epoch_acc*100:.2f}%")

            train_mean = float(np.mean(train_accs)) if len(train_accs) else 0.0
            train_std  = float(np.std(train_accs))  if len(train_accs) else 0.0

            # ----------------------------- VALIDATION -------------------------
            model.eval()
            valid_accs = []
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    label_ids = labels.argmax(dim=1) if labels.dim()>1 else labels

                    # inputs shape: (B, n, C, H, W)  → flatten fixations
                    B, n, C, H, W = inputs.shape
                    inputs_flat = inputs.reshape(-1, C, H, W)
                    logits = model(inputs_flat)  # (B*n, num_classes)

                    # fuse fixations per base image
                    logits = logits.reshape(B, n, -1).sum(dim=1)  # (B, num_classes)
                    preds = logits.argmax(dim=1)

                    batch_acc = (preds == label_ids).float().mean().item()
                    valid_accs.append(batch_acc)

            valid_mean = float(np.mean(valid_accs)) if len(valid_accs) else 0.0
            valid_std  = float(np.std(valid_accs))  if len(valid_accs) else 0.0
            print(f"    Valid Acc = {valid_mean*100:.2f}% ± {valid_std*100:.2f}%")

            # ------------------------------- TEST -----------------------------
            test_accs = []
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(deviwce)
                    label_ids = labels.argmax(dim=1) if labels.dim()>1 else labels

                    B, n, C, H, W = inputs.shape
                    inputs_flat = inputs.reshape(-1, C, H, W)
                    logits = model(inputs_flat)

                    logits = logits.reshape(B, n, -1).sum(dim=1)
                    preds = logits.argmax(dim=1)
                    batch_acc = (preds == label_ids).float().mean().item()
                    test_accs.append(batch_acc)

            test_mean = float(np.mean(test_accs)) if len(test_accs) else 0.0
            test_std  = float(np.std(test_accs))  if len(test_accs) else 0.0
            print(f"    Test  Acc = {test_mean*100:.2f}% ± {test_std*100:.2f}%\n")

            history.append({
                "epoch":       epoch,
                "identity":    ident,
                "train_mean":  train_mean,
                "train_std":   train_std,
                "valid_mean":  valid_mean,
                "valid_std":   valid_std,
                "test_mean":   test_mean,
                "test_std":    test_std,
            })

        # best-of-last-5 snapshot
        best_train = max(history[-5:], key=lambda d: d['train_mean'])['train_mean']
        best_val   = max(history[-5:], key=lambda d: d['valid_mean'])['valid_mean']
        best_test  = max(history[-5:], key=lambda d: d['test_mean'])['test_mean']
        history_acc.append({
            "fixation_points": s,
            "train_mean": best_train,
            "valid_mean": best_val,
            "test_mean": best_test,
        })
        print("best accs:", best_train, best_val, best_test)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pd.DataFrame(history)
    df.to_csv(f"output_4/training_history_supcon_{ts}.csv", index=False)

    df = pd.DataFrame(history_acc)
    df.to_csv(f"output_4/overall_training_history_supcon_{ts}.csv", index=False)

    # save final model from last run
    torch.save(model.state_dict(), f"output_4/resnet18_supcon_{ts}.pth")


if __name__ == "__main__":
    print('GPU Available:', torch.cuda.is_available())
    print('Device count:', torch.cuda.device_count())
    if torch.cuda.is_available():
        print('Device name:', torch.cuda.get_device_name(0))

    for i in range(5):
        print(f"starting LP {i}...")
        main(lp=True, dataset_name="salience", faces_data="updated")


# import os
# import pandas as pd
# import datetime
# from utils_new import *
# from model import *
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from Datasets_new import *
# # from trans import Pipeline
# # from salience_trans import SaliencePipeline


# def main(lp = True, dataset_name: str = "salience", faces_data = "updated"):
#     os.makedirs("output", exist_ok=True)
#     # ------------------------------------------------------------------------
#     # Configuration
#     # ------------------------------------------------------------------------
#     dataset_name    = dataset_name
#     faces_data      = faces_data
#     identity_counts = [32]
#     salient_counts  = [2] # [2, 4, 8, 16, 32, 64]
#     splits          = ["train_upright", "valid_upright", "valid_inverted"]
#     epoch_block     = 40  # how many epochs per identity
#     total_epochs    = 40#epoch_block * len(salient_counts)
#     num_gpu         = 1
#     num_workers     = 4
#     idx_gpu         = 5   # The index of GPU that this task is about to run on
#     batch_size      = 1024  # bs --> fix: 64 --> 4, 8; 16 --> 16; 8 --> 32; 4 --> 64
#     lr              = 1e-3
#     # device = torch.device(f"cuda:{idx_gpu}" if torch.cuda.is_available() and torch.cuda.device_count() > idx_gpu else "cpu")
#     device = torch.device(f"cuda:{0}" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")

#     print('Device: ', device)

#     # ------------------------------------------------------------------------
#     # 1) Helper to map an epoch → identity
#     # ------------------------------------------------------------------------
#     def identity_for_epoch(epoch: int) -> int:
#         idx = (epoch - 1) // epoch_block
#         return identity_counts[idx]

#     def salient_points_for_epoch(epoch: int) -> int:
#         idx = (epoch - 1) // epoch_block
#         return salient_counts[idx]

#     # ------------------------------------------------------------------------
#     # 2) Training loop
#     # ------------------------------------------------------------------------
#     history         = []
#     history_acc     = []
#     for s in salient_counts:
#         torch.cuda.empty_cache()
#         model = Model(size=224) if faces_data == 'updated' else Model(size=180)
#         model = model.to(device)

#         optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#         criterion = torch.nn.CrossEntropyLoss()

#         valid_batch_size = batch_size // s

#         for epoch in range(1, total_epochs + 1):
#             # 1) figure out which identity we're on & how many salient points to use
#             ident = identity_for_epoch(epoch)
#             num_salient_points = s #salient_points_for_epoch(epoch)

#             # 2) re-create loaders for this identity
#             datasets = make_datasets(ident, num_salient_points, faces_data)

#             train_loader = DataLoader(
#                 datasets["train_upright"],
#                 batch_size=batch_size,
#                 shuffle=True,
#                 num_workers=num_workers,
#                 pin_memory=True
#             )
#             valid_loader = DataLoader(
#                 datasets["valid_upright"],
#                 batch_size=valid_batch_size,
#                 shuffle=False,
#                 num_workers=num_workers,
#                 pin_memory=True
#             )
#             test_loader  = DataLoader(
#                 datasets["valid_inverted"],
#                 batch_size=valid_batch_size,
#                 shuffle=False,
#                 num_workers=num_workers,
#                 pin_memory=True
#             )

#             # 3) ----- TRAIN -----
#             model.train()
#             correct = 0
#             total   = 0
#             train_accs = []

#             pbar = tqdm(total=len(train_loader.dataset),
#                         desc=f"Epoch {epoch}/{total_epochs}",
#                         unit="img")

#             for inputs, labels in train_loader:
#                 inputs = inputs.to(device)
#                 # print("inputs", inputs)
#                 labels = labels.to(device)
#                 # print("labels", labels)

#                 # if labels are one‑hot (B, C), convert to class indices (B,)
#                 label_ids = labels.argmax(dim=1) if labels.dim()>1 else labels
#                 ###########################################################################################
#                 # Decode one-hot labels into readable names
#                 # label_indices = labels.argmax(dim=1).cpu().numpy()
#                 # inv_map = {v: k for k, v in datasets['train_upright'].map.items()}  # reverse dict
                
#                 # for i in range(min(3, len(label_indices))):  # print first 3
#                 #     print(f"Sample {i}: class_idx={label_indices[i]}, identity={inv_map[label_indices[i]]}")
#                 # print("--------------------------")
#                 ###########################################################################################
#                 # print("label_ids", label_ids)
#                 B,C,H,W = inputs.shape
#                 # print("B, C, H, W", B, C, H, W)
        
#                 optimizer.zero_grad()
#                 outputs = model(inputs) # (B, output_dim)
#                 # print("outputs", outputs)
                
#                 loss = criterion(outputs, label_ids)
#                 # print("loss", loss)
#                 loss.backward()
#                 optimizer.step()

#                 preds = outputs.argmax(dim=1)
#                 correct += (preds == label_ids).sum().item()
#                 total   += label_ids.size(0)
#                 batch_acc = correct / total
#                 train_accs.append(batch_acc)

#                 pbar.update(inputs.size(0))
#                 pbar.set_postfix(acc=f"{batch_acc*100:.2f}%")

#             pbar.close()
#             epoch_acc = correct / total
#             print(f"→ Epoch {epoch}/{total_epochs} — Accuracy: {epoch_acc*100:.2f}%")
#             train_mean = np.mean(train_accs)
#             train_std  = np.std(train_accs)

#             # 4) ----- VALIDATION -----
#             model.eval()
#             correct = total = 0
#             valid_accs = []

#             with torch.no_grad():
#                 for inputs, labels in valid_loader:
#                     inputs, labels = inputs.to(device), labels.to(device)
#                     # print("inputs", inputs)
#                     # print("labels", labels)
                    
#                     label_ids = labels.argmax(dim=1) if labels.dim()>1 else labels
#                     # print("label_ids", label_ids)
#                     # # Decode one-hot labels into readable names
#                     # label_indices = labels.argmax(dim=1).cpu().numpy()
#                     # inv_map = {v: k for k, v in datasets['valid_upright'].map.items()}  # reverse dict
                    
#                     # for i in range(min(3, len(label_indices))):  # print first 3
#                     #     print(f"Sample {i}: class_idx={label_indices[i]}, identity={inv_map[label_indices[i]]}")
#                     # print("--------------------------")
                    
#                     # transform input data
#                     B,n,C,H,W = inputs.shape 
#                     # print("B, C, H, W", B, C, H, W)
#                     inputs = inputs.reshape(-1,C,H,W) #(B*num_salience_pts,C,H,W)
#                     outputs = model(inputs) #(B*num_salience_pts, output_dim)
#                     # print("outputs", outputs)

#                     outputs = outputs.reshape(B, num_salient_points, -1)
#                     outputs = outputs.sum(dim=1)
                    
#                     preds = outputs.argmax(dim=1)
#                     # print("preds", preds)
#                     batch_acc = (preds == label_ids).float().mean().item()
#                     valid_accs.append(batch_acc)

#             valid_mean = np.mean(valid_accs)
#             valid_std  = np.std(valid_accs)
#             print(f"    Valid Acc = {valid_mean*100:.2f}% ± {valid_std*100:.2f}%")

#             # 5) ----- TEST -----
#             correct = total = 0
#             test_accs = []
#             with torch.no_grad():
#                 for inputs, labels in test_loader:
#                     inputs, labels = inputs.to(device), labels.to(device)
#                     label_ids = labels.argmax(dim=1) if labels.dim()>1 else labels
                    
#                     # transform input data
#                     B,n,C,H,W = inputs.shape 
#                     inputs = inputs.reshape(-1,C,H,W) #(B*num_salience_pts,C,H,W)
#                     outputs = model(inputs) #(B*num_salience_pts, output_dim)
#                     outputs = outputs.reshape(B, num_salient_points, -1)
#                     outputs = outputs.sum(dim=1)
                    
#                     preds = outputs.argmax(dim=1)
#                     batch_acc = (preds == label_ids).float().mean().item()
#                     test_accs.append(batch_acc)

#             test_mean = np.mean(test_accs)
#             test_std  = np.std(test_accs)
#             print(f"    Test  Acc = {test_mean*100:.2f}% ± {test_std*100:.2f}%\n")
        
#             history.append({
#                     "epoch":       epoch,
#                     "identity":    ident,
#                     "train_mean":  train_mean,
#                     "train_std":   train_std,
#                     "valid_mean":  valid_mean,
#                     "valid_std":   valid_std,
#                     "test_mean":   test_mean,
#                     "test_std":    test_std,
#                 })
#         # take best of last 5 epochs
#         best_train = max(history[-5:], key=lambda item: item['train_mean'])['train_mean']
#         best_val = max(history[-5:], key=lambda item: item['valid_mean'])['valid_mean']
#         best_test = max(history[-5:], key=lambda item: item['test_mean'])['test_mean']
        
#         history_acc.append({
#             "fixation_points":  s,
#             "train_mean": best_train,
#             "valid_mean": best_val,
#             "test_mean": best_test,
#         })
#         print("best accs: ", best_train, best_val, best_test)

#     ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

#     df = pd.DataFrame(history)
#     df.to_csv(f"output/training_history_{'lp' if lp else 'cnn'}_{ts}.csv", index=False)

#     df = pd.DataFrame(history_acc)
#     df.to_csv(f"output/overall_training_history_{'lp' if lp else 'cnn'}_{ts}.csv", index=False)

#     torch.save(model.state_dict(), f"output/resnet18_{'lp' if lp else 'cnn'}_{ts}.pth")

# if __name__ == "__main__":
#     print('GPU Available: ', torch.cuda.is_available())
#     print('Device count: ', torch.cuda.device_count())
#     print('Current device: ', torch.cuda.current_device())
#     print('Device name: ', torch.cuda.get_device_name(0))
    
#     # main(lp=True)

#     for i in range(1):
#         print(f"starting LP {i}...")
#         main(lp=True, dataset_name="salience", faces_data="updated") 

#     # for i in range(5):
#     #     print(f"starting CNN {i}...")
#     #     main(lp=False, dataset_name="salience", faces_data="cnn")