import os
import torch
import torchvision.transforms.functional as TF
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class CelebAFaceIDDataset(Dataset):
    def __init__(self, root_dir: str = "data", split: str = "train"):
        """
        Args:
            root_dir (str): path to the folder containing
                "CelebA_HQ_facial_identity_dataset" (default=".")
            split (str): "train" or "test"
        """
        self.samples = []
        root_path  = Path(root_dir)
        dataset_dir = root_path / "CelebA_HQ_facial_identity_dataset"
        split_dir   = dataset_dir / split

        if not split_dir.is_dir():
            raise FileNotFoundError(f"Could not find split directory: {split_dir!r}")

        # each subfolder name is the integer ID
        for person_dir in sorted(split_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            try:
                person_id = int(person_dir.name)
            except ValueError:
                # skip any non‐integer‐named folders
                continue

            # gather all .jpg files under this ID
            for img_path in sorted(person_dir.glob("*.jpg")):
                self.samples.append((img_path, person_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = TF.to_tensor(img)
        label = torch.tensor(label, dtype=torch.long)
        return img, label

class CelebrityFacesDataset(Dataset):
    def __init__(self, root_dir: str, num_identities: int, split: str):
        """
        Args:
            root_dir (str): path to "/dataset"
            num_identities (int): 4, 8, …, 128
            split (str): one of "train", "valid", "test"
        """
        # build the path to e.g. "/dataset/faces/faces/8_identities/train"
        self.data_dir = os.path.join(
            root_dir, 
            "faces", 
            "faces", 
            f"{num_identities}_identities", 
            split
        )
        if not os.path.isdir(self.data_dir):
            raise ValueError(f"Directory not found: {self.data_dir}")

        # list all celebrity folders
        self.classes = sorted(
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        )

        # collect (image_path, label) tuples
        self.samples = []
        for celeb in self.classes:
            celeb_dir = os.path.join(self.data_dir, celeb)
            for fname in sorted(os.listdir(celeb_dir)):
                if fname.lower().endswith(".jpg"):
                    img_path = os.path.join(celeb_dir, fname)
                    # here label is the celebrity name (string)
                    self.samples.append((img_path, celeb))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = TF.to_tensor(img)
        # label = torch.tensor(int(label), dtype=torch.long)
        return img, label

