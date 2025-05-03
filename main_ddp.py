import os
import argparse
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from utils import *
from transformation import *
from model import Model
from Datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Local process rank. Provided by torchrun.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--epoch_block", type=int, default=40)
    return parser.parse_args()


def main():
    args = parse_args()
    # Determine local rank
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Init process group
    dist.init_process_group(backend="nccl", init_method="env://")
    is_main = (local_rank == 0)
    if is_main:
        os.makedirs("output", exist_ok=True)

    # Config
    dataset_name    = "faces"
    identity_counts = [4, 8, 16, 32, 64, 128]
    splits          = ["train", "valid", "test"]
    total_epochs    = args.epochs
    epoch_block     = args.epoch_block

    # Load datasets (CPU)
    all_datasets = {
        ident: { split: load_dataset(dataset_name, ident, split)
                 for split in splits }
        for ident in identity_counts
    }

    def identity_for_epoch(epoch: int) -> int:
        idx = (epoch - 1) // epoch_block
        return identity_counts[idx]

    # Build and wrap model with DDP, enabling unused parameter detection
    model     = Model().to(device)
    model     = DDP(model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, total_epochs + 1):
        ident = identity_for_epoch(epoch)
        # Distributed samplers
        train_ds = all_datasets[ident]["train"]
        valid_ds = all_datasets[ident]["valid"]
        test_ds  = all_datasets[ident]["test"]

        train_sampler = DistributedSampler(train_ds, shuffle=True)
        valid_sampler = DistributedSampler(valid_ds, shuffle=False)
        test_sampler  = DistributedSampler(test_ds, shuffle=False)

        # DataLoaders
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  num_workers=args.num_workers,
                                  pin_memory=True)
        valid_loader = DataLoader(valid_ds, batch_size=args.batch_size,
                                  sampler=valid_sampler,
                                  num_workers=args.num_workers,
                                  pin_memory=True)
        test_loader  = DataLoader(test_ds, batch_size=args.batch_size,
                                  sampler=test_sampler,
                                  num_workers=args.num_workers,
                                  pin_memory=True)
        train_sampler.set_epoch(epoch)

        # Training loop
        model.train()
        correct = total = 0
        if is_main:
            pbar = tqdm(total=len(train_loader.dataset),
                        desc=f"Epoch {epoch}/{total_epochs}", unit="img")
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            label_ids = labels.argmax(dim=1) if labels.dim()>1 else labels

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, label_ids)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            correct += (preds == label_ids).sum().item()
            total += label_ids.size(0)
            if is_main:
                pbar.update(inputs.size(0))
                pbar.set_postfix(acc=f"{(correct/total)*100:.2f}%")
        if is_main:
            pbar.close()
            print(f"→ Epoch {epoch} — Acc: {(correct/total)*100:.2f}%")

        # Validation
        model.eval()
        valid_accs = []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                label_ids = labels.argmax(dim=1) if labels.dim()>1 else labels
                outputs = model(inputs)
                valid_accs.append((outputs.argmax(dim=1)==label_ids).float().mean().item())
        if is_main:
            valid_mean = torch.tensor(valid_accs).mean().item()
            valid_std  = torch.tensor(valid_accs).std().item()
            print(f"    Valid Acc = {valid_mean*100:.2f}% ± {valid_std*100:.2f}%")

        # Testing
        test_accs = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                label_ids = labels.argmax(dim=1) if labels.dim()>1 else labels
                outputs = model(inputs)
                test_accs.append((outputs.argmax(dim=1)==label_ids).float().mean().item())
        if is_main:
            test_mean = torch.tensor(test_accs).mean().item()
            test_std  = torch.tensor(test_accs).std().item()
            print(f"    Test  Acc = {test_mean*100:.2f}% ± {test_std*100:.2f}%\n")

        if is_main:
            history.append({
                "epoch": epoch, "identity": ident,
                "train_acc": correct/total,
                "valid_mean": valid_mean, "valid_std": valid_std,
                "test_mean": test_mean,  "test_std": test_std
            })

    if is_main:
        pd.DataFrame(history).to_csv("output/training_history.csv", index=False)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()

# Launch:
# torchrun --nproc_per_node=6 --master_port=29500 main_ddp.py \
#   --batch_size 64 --num_workers 4 --epochs 240 --epoch_block 40
