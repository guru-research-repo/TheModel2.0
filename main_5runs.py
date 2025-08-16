import os
import pandas as pd
import numpy as np
from utils import *
#from transformation import *
from model import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from Datasets import *


def main():
    os.makedirs("output", exist_ok=True)

    # ------------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------------
    dataset_name    = "objects"
    identity_counts = [4, 8, 16, 32, 64, 128]
    splits          = ["train", "valid", "test"]
    total_epochs    = 240
    epoch_block     = 40  # how many epochs per identity
    num_trials      = 5   # number of repeated runs
    #num_gpu         = 1
    idx_gpu         = 5   # The index of GPU that this task is about to run on
    num_workers     = 4
    batch_size      = 64
    lr              = 1e-3
    #device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(f"cuda:{idx_gpu}" if torch.cuda.is_available() and torch.cuda.device_count() > idx_gpu else "cpu")

    # ------------------------------------------------------------------------
    # 1) Pre‑load all datasets
    # ------------------------------------------------------------------------
    all_datasets = {
        ident: { split: load_dataset(dataset_name, ident, split)
                for split in splits }
        for ident in identity_counts
    }

    # ------------------------------------------------------------------------
    # 2) Helper to map an epoch → identity
    # ------------------------------------------------------------------------
    def identity_for_epoch(epoch: int) -> int:
        idx = (epoch - 1) // epoch_block
        return identity_counts[idx]

    # ------------------------------------------------------------------------
    # 3) Run multiple trials
    # ------------------------------------------------------------------------
    all_histories = []

    for trial in range(num_trials):
        print(f"\n====== Starting Trial {trial + 1}/{num_trials} ======")
        model = Model()

        # --- multi‑GPU wrap ---
        #if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            #n_gpu = min(num_gpu, torch.cuda.device_count())
            #print(f"→ Using {n_gpu} GPUs")
            #model = torch.nn.DataParallel(model, device_ids=list(range(n_gpu)))

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        history = []

        for epoch in range(1, total_epochs + 1):
            # 1) figure out which identity we're on
            ident = identity_for_epoch(epoch)

            # 2) re-create loaders for this identity
            train_loader = DataLoader(all_datasets[ident]["train"],
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      pin_memory=True)
            valid_loader = DataLoader(all_datasets[ident]["valid"],
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=num_workers,
                                      pin_memory=True)
            test_loader  = DataLoader(all_datasets[ident]["test"],
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=num_workers,
                                      pin_memory=True)

            # 3) ----- TRAIN -----
            model.train()
            correct = total = 0
            train_accs = []

            pbar = tqdm(total=len(train_loader.dataset),
                        desc=f"[Trial {trial+1}] Epoch {epoch}/{total_epochs}",
                        unit="img")

            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                label_ids = labels.argmax(dim=1) if labels.dim() > 1 else labels

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, label_ids)
                loss.backward()
                optimizer.step()

                preds = outputs.argmax(dim=1)
                correct += (preds == label_ids).sum().item()
                total   += label_ids.size(0)
                batch_acc = correct / total
                train_accs.append(batch_acc)

                pbar.update(inputs.size(0))
                pbar.set_postfix(acc=f"{batch_acc*100:.2f}%")

            pbar.close()
            train_mean = np.mean(train_accs)
            train_std  = np.std(train_accs)

            # 4) ----- VALIDATION -----
            model.eval()
            valid_accs = []
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    label_ids = labels.argmax(dim=1) if labels.dim() > 1 else labels
                    outputs = model(inputs)
                    preds = outputs.argmax(dim=1)
                    batch_acc = (preds == label_ids).float().mean().item()
                    valid_accs.append(batch_acc)

            valid_mean = np.mean(valid_accs)
            valid_std  = np.std(valid_accs)

            # 5) ----- TEST -----
            test_accs = []
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    label_ids = labels.argmax(dim=1) if labels.dim() > 1 else labels
                    outputs = model(inputs)
                    preds = outputs.argmax(dim=1)
                    batch_acc = (preds == label_ids).float().mean().item()
                    test_accs.append(batch_acc)

            test_mean = np.mean(test_accs)
            test_std  = np.std(test_accs)

            print(f"→ Epoch {epoch}/{total_epochs} — Train: {train_mean*100:.2f}% | "
                  f"Valid: {valid_mean*100:.2f}% | Test: {test_mean*100:.2f}%")

            history.append({
                "trial":       trial,
                "epoch":       epoch,
                "identity":    ident,
                "train_mean":  train_mean,
                "train_std":   train_std,
                "valid_mean":  valid_mean,
                "valid_std":   valid_std,
                "test_mean":   test_mean,
                "test_std":    test_std,
            })

        all_histories.append(pd.DataFrame(history))

    # ------------------------------------------------------------------------
    # Aggregate results across all trials
    # ------------------------------------------------------------------------
    combined_df = pd.concat(all_histories, ignore_index=True)

    summary_df = combined_df.groupby("epoch").agg({
        "train_mean": ["mean", "std"],
        "valid_mean": ["mean", "std"],
        "test_mean":  ["mean", "std"]
    }).reset_index()

    summary_df.columns = ["epoch",
                          "train_mean", "train_std",
                          "valid_mean", "valid_std",
                          "test_mean", "test_std"]

    summary_df.to_csv("output/objects_history.csv", index=False)
    print("→ Saved averaged results to output/objects_history.csv")

if __name__ == "__main__":
    main()