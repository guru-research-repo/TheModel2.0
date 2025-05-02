from utils import *
from transformation import *
from model import *
from torch.utils.data import DataLoader
from Datasets import *

# ------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------

# Choose the dataset to load:
#   - "celeb":  only supports splits "train" and "test"
#   - "faces":  supports identity counts {4, 8, 16, 32, 64, 128}
#               and splits "train", "valid", "test"
dataset_name   = "faces"  # Options: "celeb", "faces"
identity_count = 4        # Only used when dataset_name == "faces"
split          = "train"   # For "celeb": {"train", "test"};
                            # for "faces": {"train", "valid", "test"}

# ------------------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------------------

# load_dataset parameters:
#   dataset : str   -> which dataset to load ("celeb" or "faces")
#   identity: int   -> number of distinct identities (only for "faces")
#   task    : str   -> which split to load ("train", "valid", "test")
ds = load_dataset(
    dataset=dataset_name,
    identity=identity_count,
    task=split
) 

model = Model()

# show_images([rotate(ds[6][0]), rotate(ds[6][0]), rotate(ds[6][0], inverse=True)])

from tqdm import tqdm

# Hyper‑parameters
batch_size = 32
epochs     = 10
lr         = 1e-3
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare data loader
# train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

# Move model to device
model = model.to(device)

# Optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0

    # tqdm bar over total # of images
    pbar = tqdm(total=len(train_loader.dataset),
                desc=f"Epoch {epoch}/{epochs}",
                unit="img")

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # accumulate loss and advance bar by batch size
        running_loss += loss.item() * inputs.size(0)
        pbar.update(inputs.size(0))
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    pbar.close()
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'  → Epoch {epoch:2d}/{epochs} done; avg loss: {epoch_loss:.4f}')
