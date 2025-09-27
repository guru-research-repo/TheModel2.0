import os
import pandas as pd
import datetime
from utils import *
from model import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from Datasets import *
from trans import Pipeline
from salience_trans import SaliencePipeline


def main(lp = True, dataset_name: str = "salience", faces_data = "updated"):
    os.makedirs("output", exist_ok=True)
    # ------------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------------
    dataset_name    = dataset_name
    faces_data      = faces_data
    identity_counts = [32]
    salient_counts  = [64]
    splits          = ["train", "valid", "test"]
    epoch_block     = 40  # how many epochs per identity
    total_epochs    = epoch_block * len(salient_counts)
    num_gpu         = 1
    num_workers     = 4
    idx_gpu         = 5   # The index of GPU that this task is about to run on
    batch_size      = 4  # bs --> fix: 64 --> 4, 8; 16 --> 16; 8 --> 32; 4 --> 64
    lr              = 1e-3
    device = torch.device(f"cuda:{idx_gpu}" if torch.cuda.is_available() and torch.cuda.device_count() > idx_gpu else "cpu")

    # ------------------------------------------------------------------------
    # 1) Helper to map an epoch → identity
    # ------------------------------------------------------------------------
    def identity_for_epoch(epoch: int) -> int:
        idx = (epoch - 1) // epoch_block
        return identity_counts[idx]

    def salient_points_for_epoch(epoch: int) -> int:
        idx = (epoch - 1) // epoch_block
        return salient_counts[idx]

    # ------------------------------------------------------------------------
    # 2) Training loop
    # ------------------------------------------------------------------------
    history         = []
    model = Model(size=224) if faces_data == 'updated' else Model(size=180)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, total_epochs + 1):
        # 1) figure out which identity we're on & how many salient points to use
        ident = identity_for_epoch(epoch)
        num_salient_points = salient_points_for_epoch(epoch)

        # 2) re-create loaders for this identity
        datasets = make_datasets(ident, num_salient_points, faces_data)

        train_loader = DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        valid_loader = DataLoader(
            datasets["valid"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        test_loader  = DataLoader(
            datasets["test"],
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
            label_ids = labels.argmax(dim=1) if labels.dim()>1 else labels
            # repeat because we have num_salient_pts-many images
            label_ids = label_ids.repeat_interleave(num_salient_points) 

            B,n,C,H,W = inputs.shape
            inputs = inputs.reshape(-1,C,H,W) #(B*num_salience_pts,C,H,W)
    
            optimizer.zero_grad()
            outputs = model(inputs) #(B*num_salience_pts, output_dim)
            
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
                
                # transform input data
                B,n,C,H,W = inputs.shape 
                inputs = inputs.reshape(-1,C,H,W) #(B*num_salience_pts,C,H,W)
                outputs = model(inputs) #(B*num_salience_pts, output_dim)

                outputs = outputs.reshape(B, num_salient_points, -1)
                outputs = outputs.sum(dim=1)
                
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
                
                # transform input data
                B,n,C,H,W = inputs.shape 
                inputs = inputs.reshape(-1,C,H,W) #(B*num_salience_pts,C,H,W)
                outputs = model(inputs) #(B*num_salience_pts, output_dim)
                outputs = outputs.reshape(B, num_salient_points, -1)
                outputs = outputs.sum(dim=1)
                
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
    df.to_csv(f"output/training_history_{'lp' if lp else 'cnn'}_{ts}.csv", index=False)

    torch.save(model.state_dict(), f"output/resnet18_{'lp' if lp else 'cnn'}_{ts}.pth")

if __name__ == "__main__":
    # main(lp=True)
    # for i in range(5):
    #     print(f"starting LP {i}...")
    #     main(lp=True, dataset_name="salience", faces_data="updated") 

    for i in range(5):
        print(f"starting CNN {i}...")
        main(lp=False, dataset_name="salience", faces_data="cnn")