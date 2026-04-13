# Run using CUDA_VISIBLE_DEVICES=2 python main_familiar.py
# What if I unfreeze just layer4, fc1, and fc2 while keeping the rest of the model locked?
import os
import numpy as np
import pandas as pd
import datetime
import torch
import torch.nn.functional as F
from utils import *
from model_familiar import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from Datasets import *
import matplotlib.pyplot as plt

def main(lp=True, dataset_name="salience"):
    os.makedirs("familiarexp1_temp2_FACES_LPNet_32_fixations_50epoch_128_identities", exist_ok=True)

    # ---------------------------- Curriculum Setup -----------------------------
    identity_counts = [128]    # identities introduced gradually [4, 8, 16, 32, 64, 128] 
    salient_counts  = [32]               # always use first 32 fixations [32]
    epoch_block     = 50              # each stage length of 40 epoch block
    total_epochs    = 50              #epoch_block * len(identity_counts)  # 240
    num_workers     = 4
    batch_size      = 256             
    lr              = 1e-3
    temperature = 2.0 # for now, will have to do annealing later

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Map epoch → curriculum stage
    def identity_for_epoch(epoch):
        idx = (epoch - 1) // epoch_block
        return identity_counts[idx]

    def salient_points_for_epoch(epoch):
        idx = (epoch - 1) // epoch_block
        return salient_counts[0]  # 32 fixations

    history = []
    history_acc = []

    # -------------------------------- Model -----------------------------------
    torch.cuda.empty_cache()
    num_classes = 128

    model = Model( 
        size=180 if lp else 180,
        num_classes=num_classes, pretrained=False, T = temperature
    ).to(device)

    pretrained_path = "FACES_LPNet_32_fixations_40_epochblock_Martha/resnet18_20260129_162806.pth"  # <-- change to your actual path
    state = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(state, strict=False)
    print("Loaded pretrained weights")

    # # expt part(a) : try first 
    # for p in model.parameters():
    #     p.requires_grad = False
    
    # # Unfreeze ONLY logistic hidden layer and output layer
    # for p in model.fc1.parameters():
    #     p.requires_grad = True
    # for p in model.fc2.parameters():
    #     p.requires_grad = True

    # expt part(b): allow all weights to adapt
    for p in model.parameters():
        p.requires_grad = True

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=lr)
    ce_criterion = torch.nn.CrossEntropyLoss()

    # We reuse only 1 salient-count configuration (s = 32)
    s = salient_counts[0]
    valid_batch_size = batch_size // s

    # To know when to rebuild DataLoaders
    prev_ident = None
    model.stochastic = False # for finetuning without sampling

    best_val = -1
    train_losses = []
    patience = 10
    patience_counter = 0
    best_path = "familiarexp1_temp2_FACES_LPNet_32_fixations_50epoch_128_identities/familiarfaces_LP_best_model.pth"

    # ---------------------------- Training Loop --------------------------------
    for epoch in range(1, total_epochs + 1):

        # Current curriculum stage
        ident = identity_for_epoch(epoch)
        num_salient_points = salient_points_for_epoch(epoch)

        # ------------------ REBUILD DATASET + LOADERS ONLY IF STAGE CHANGES -----
        if prev_ident != ident:
            print(f"\n=== Building datasets for ident={ident}, fix={num_salient_points} ===")

            datasets = make_datasets(ident, num_salient_points, lp, dataset=dataset_name)

            train_loader = DataLoader(
                datasets["train"],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
            )
            valid_loader = DataLoader(
                datasets["valid"],
                batch_size=valid_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
            test_loader = DataLoader(
                datasets["test"],
                batch_size=valid_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            prev_ident = ident  # update stage tracker

        # ------------------------------- TRAIN ----------------------------------
        model.train()
        correct = 0
        total = 0
        train_accs = []
        epoch_losses = []

        pbar = tqdm(
            total=len(train_loader.dataset),
            desc=f"Epoch {epoch}/{total_epochs}",
            unit="img",
        )

        for inputs, labels in train_loader:
            inputs = inputs.to(device) #[256, 3, 180, 180]
            labels = labels.to(device) #[256, 128]
            label_ids = labels.argmax(dim=1) if labels.dim() > 1 else labels

            optimizer.zero_grad()

            logits = model(inputs) #[256, 128]
            ce_loss = ce_criterion(logits, label_ids)

            ce_loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            correct += (preds == label_ids).sum().item()
            total += label_ids.size(0)
            batch_acc = correct / total
            train_accs.append(batch_acc)

            pbar.update(inputs.size(0))
            pbar.set_postfix(acc=f"{batch_acc*100:.2f}%", ce=f"{ce_loss.item():.3f}")
            epoch_losses.append(ce_loss.item())

        pbar.close()

        train_mean = float(np.mean(train_accs))
        train_loss_mean = float(np.mean(epoch_losses))
        train_std = float(np.std(train_accs))
        print(f"→ Epoch {epoch}/{total_epochs} — Train Acc: {train_mean*100:.2f}% | Loss:{train_loss_mean:.4f}")

        # ----------------------------- VALIDATION -------------------------------
        model.eval()
        valid_accs = []

        with torch.no_grad():
            pbar = tqdm(
                total=len(valid_loader.dataset),
                desc=f"Valid {epoch}/{total_epochs}",
                unit="img",
            )

            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                #print("inp", inputs.shape) #[8, 32, 3, 180, 180]
                #print("l", labels.shape) #[8, 128]
                label_ids = labels.argmax(dim=1) if labels.dim() > 1 else labels

                B, n, C, H, W = inputs.shape
                logits = model(inputs.reshape(-1, C, H, W)) #[256, 128]
                logits = logits.reshape(B, num_salient_points, -1).sum(dim=1)

                preds = logits.argmax(dim=1)
                valid_accs.append((preds == label_ids).float().mean().item())

                pbar.update(B)

            pbar.close()

        valid_mean = float(np.mean(valid_accs))
        valid_std = float(np.std(valid_accs))
        print(f"    Valid Acc = {valid_mean*100:.2f}% ± {valid_std*100:.2f}%")
        # ----- early stopping -----
        if valid_mean > best_val:
            best_val = valid_mean
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            print("Saved new best model")
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

        # ------------------------------- TEST -----------------------------------
        test_accs = []
        with torch.no_grad():
            pbar = tqdm(
                total=len(test_loader.dataset),
                desc=f"Test {epoch}/{total_epochs}",
                unit="img",
            )

            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                label_ids = labels.argmax(dim=1) if labels.dim() > 1 else labels

                B, n, C, H, W = inputs.shape
                logits = model(inputs.reshape(-1, C, H, W))
                logits = logits.reshape(B, num_salient_points, -1).sum(dim=1)

                preds = logits.argmax(dim=1)
                test_accs.append((preds == label_ids).float().mean().item())

                pbar.update(B)

            pbar.close()

        test_mean = float(np.mean(test_accs))
        test_std = float(np.std(test_accs))
        print(f"    Test  Acc = {test_mean*100:.2f}% ± {test_std*100:.2f}%\n")

        # Save metrics
        history.append(
            {
                "epoch": epoch,
                "identity": ident,
                "train_mean": train_mean,
                "train_loss": train_loss_mean,
                "train_std": train_std,
                "valid_mean": valid_mean,
                "valid_std": valid_std,
                "test_mean": test_mean,
                "test_std": test_std,
            }
        )

        # Track best in last epoch_block
        block_start = ((epoch - 1) // epoch_block) * epoch_block
        best_train = max(history[block_start:epoch], key=lambda d: d["train_mean"])["train_mean"]
        best_val = max(history[block_start:epoch], key=lambda d: d["valid_mean"])["valid_mean"]
        best_test = max(history[block_start:epoch], key=lambda d: d["test_mean"])["test_mean"]

        # Only append at boundary of stage
        if epoch % epoch_block == 0:
            history_acc.append(
                {
                    "fixation_points": s,
                    "identities": ident,
                    "train_mean": best_train,
                    "valid_mean": best_val,
                    "test_mean": best_test,
                }
            )
            print(f"Stage completed: best={best_train}, {best_val}, {best_test}")

    # ------------------------------- Save Results ------------------------------
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    pd.DataFrame(history).to_csv(
        f"familiarexp1_temp2_FACES_LPNet_32_fixations_50epoch_128_identities/training_history_{ts}.csv", index=False
    )
    pd.DataFrame(history_acc).to_csv(
        f"familiarexp1_temp2_FACES_LPNet_32_fixations_50epoch_128_identities/overall_training_history_{ts}.csv", index=False
    )

    torch.save(
        model.state_dict(), f"familiarexp1_temp2_FACES_LPNet_32_fixations_50epoch_128_identities/resnet18_{ts}.pth"
    )

    epochs = [h["epoch"] for h in history]
    train_acc = [h["train_mean"] for h in history]
    
    plt.figure()
    plt.plot(epochs, train_acc)
    plt.xlabel("Epoch")
    plt.ylabel("Train Accuracy")
    plt.title("Training Accuracy")
    plt.savefig("train_accuracy_temp2_exp1.png")
    plt.close()

    train_loss = [h["train_loss"] for h in history]
    
    plt.figure()
    plt.plot(epochs, train_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Training Loss")
    plt.savefig("train_loss_exp_temp2_exp1.png")
    plt.close()


if __name__ == "__main__":
    print("GPU Available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Device name:", torch.cuda.get_device_name(0))

    for i in range(1):
        print(f"starting LP run...")
        main(lp=True, dataset_name="salience")
