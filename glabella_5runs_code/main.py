import datetime
import os
import pandas as pd
from utils import *
from transformation import *
from model import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from Datasets import *

def run_single_training(run_id):
# def main():
    # os.makedirs("output", exist_ok=True)
    # ------------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------------
    dataset_name    = "faces"
    identity_counts = [4, 8, 16, 32, 64, 128]
    splits          = ["train_upright", "valid_upright", "valid_inverted"]
    # splits          = ["train", "valid", "test"]
    total_epochs    = 240
    epoch_block     = 40  # how many epochs per identity
    idx_gpu         = 0   # The index of GPU that this task is about to run on
    num_gpu         = 1
    num_workers     = 4

    # ------------------------------------------------------------------------
    # 1) Pre‑load all datasets
    # ------------------------------------------------------------------------
    # label_mapping = get_label_mapping(root_dir="glabella_processed_5runs", num_identities=128, split="valid_inverted")

    all_datasets = {
        ident: { split: load_dataset(dataset_name, ident, split) # label_mapping = label_mapping
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
    # 3) Training loop
    # ------------------------------------------------------------------------

    # Hyper‑parameters
    history         = []
    batch_size      = 64
    lr              = 1e-3
    num_iter        = 1
    dropout         = 0.25
    # device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(f"cuda:{idx_gpu}" if torch.cuda.is_available() and torch.cuda.device_count() > idx_gpu else "cpu")

    model = Model()
    
    model = model.to(device)

    print(f"→ Model running on {device}")

 # --- multi‑GPU wrap ---
    # if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    #     n_gpu = min(num_gpu, torch.cuda.device_count())
    #     print(f"→ Using {n_gpu} GPUs")
    #     model = torch.nn.DataParallel(model, device_ids=list(range(n_gpu)))

    # model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, total_epochs + 1):
        # 1) figure out which identity we're on
        ident = identity_for_epoch(epoch)

        # 2) re-create loaders for this identity
        train_loader = DataLoader(
            all_datasets[ident]["train_upright"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        valid_loader = DataLoader(
            all_datasets[ident]["valid_upright"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        test_loader  = DataLoader(
            all_datasets[ident]["valid_inverted"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        # 3) ----- TRAIN -----
        model.train()
        correct = 0
        total   = 0
        train_accs = []

        pbar = tqdm(total=len(train_loader.dataset),
                    desc=f"Epoch {epoch}/{total_epochs}",
                    unit="img")

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # if labels are one‑hot (B, C), convert to class indices (B,)
            if labels.dim() > 1:
                label_ids = labels.argmax(dim=1)
                # print(f"[Training] Label IDs from one-hot: {label_ids.tolist()}")
            else:
                label_ids = labels

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
        epoch_acc = correct / total
        print(f"→ Epoch {epoch}/{total_epochs} — Accuracy: {epoch_acc*100:.2f}%")
        train_mean = np.mean(train_accs)
        train_std  = np.std(train_accs)

        # 4) ----- VALIDATION -----
        model.eval()
        correct = total = 0
        valid_accs = []

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                label_ids = labels.argmax(dim=1) if labels.dim()>1 else labels
               
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
               
                batch_acc = (preds == label_ids).float().mean().item()
                valid_accs.append(batch_acc)

        valid_mean = np.mean(valid_accs)
        valid_std  = np.std(valid_accs)
        print(f"    Valid Acc = {valid_mean*100:.2f}% ± {valid_std*100:.2f}%")

        # 5) ----- TEST -----
        correct = total = 0
        test_accs = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                label_ids = labels.argmax(dim=1) if labels.dim()>1 else labels
                
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
   
                batch_acc = (preds == label_ids).float().mean().item()
                test_accs.append(batch_acc)

        test_mean = np.mean(test_accs)
        test_std  = np.std(test_accs)
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

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pd.DataFrame(history)
    df.to_csv(f"on_val_glabella_cleaned/training_history_{ts}.csv", index=False)

    torch.save(model.state_dict(), f"on_val_glabella_cleaned/resnet18_{ts}.pth")
    #####################################################################################################
    return history

def main():
    os.makedirs("on_val_glabella_cleaned", exist_ok=True)

    all_runs = []
    for run in range(1, 6):
        print(f"\n🚀 Starting Run {run}/5")
        torch.manual_seed(run)
        np.random.seed(run)
        run_history = run_single_training(run_id=run)
        all_runs.extend(run_history)

    df_all = pd.DataFrame(all_runs)
    df_all.to_csv("on_val_glabella_cleaned/training_history_5runs.csv", index=False)

    summary = df_all.groupby("epoch").agg({
        "train_mean": ["mean", "std"],
        "valid_mean": ["mean", "std"],
        "test_mean": ["mean", "std"]
    })
    summary.to_csv("on_val_glabella_cleaned/training_summary_avg_std.csv")
    ########################################################################################################

if __name__ == "__main__":
    main()