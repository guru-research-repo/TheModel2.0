import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from transformations import *
import torchvision.transforms.functional as TF

def load_dataset(dataset, identity=4, task="train", transform=None, inversion=False):
    if dataset == "celeb":
        ds = CelebAFaceIDDataset(root_dir="data", split=task, transform=transform, inversion=inversion)
    elif dataset == "faces":
        ds = CelebrityFacesDataset(root_dir="data", num_identities=identity, split=task, transform=transform, inversion=inversion)
    return ds

class CelebAFaceIDDataset(Dataset):
    def __init__(self, root_dir: str = "data", split: str = "train", transform=None, inversion=False):
        """
        Args:
            root_dir (str): path to the folder containing
                "CelebA_HQ_facial_identity_dataset" (default=".")
            split (str): "train" or "test"
            transform (nn.Sequential): transformations to apply
        """
        self.samples = []
        self.inversion = inversion
        root_path  = Path(root_dir)
        dataset_dir = root_path / "CelebA_HQ_facial_identity_dataset"
        split_dir   = dataset_dir / split
        self.transform = transform

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

        self.expanded_samples = []
        self.expand_samples()

    def expand_samples(self):
        for img_path, label in self.samples:
            img = Image.open(img_path).convert("RGB")
            
            if self.inversion:
                img = TF.rotate(img, 180)
    
            if self.transform:
                crops = self.transform(img)
            else:
                crops = [TF.to_tensor(img)]
    
            for crop in crops:
                self.expanded_samples.append((crop, label))


    def __len__(self):
        #return len(self.samples)
        return len(self.expanded_samples)

    def __getitem__(self, idx):
        img, label = self.expanded_samples[idx]
        return img, label
        # img_path, label = self.samples[idx]
        # img = Image.open(img_path).convert("RGB")

        # if self.inversion:
        #     img = TF.rotate(img, 180)  # <-- rotate 180 degrees

        # if self.transform:
        #     crops = self.transform(img) # Transformation pipeline
        # else:
        #     #crops = [img]
        #     crops = [TF.to_tensor(img)]

        # return crops, label

class CelebrityFacesDataset(Dataset):
    def __init__(self, root_dir: str, num_identities: int, split: str, transform=None, inversion=False):
        """
        Args:
            root_dir (str): path to "/dataset"
            num_identities (int): 4, 8, …, 128
            split (str): one of "train", "valid", "test"
            transform (nn.Sequential): transformations to apply
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

        # map names to integer labels
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        # collect (image_path, label) tuples
        self.samples = []
        self.inversion = inversion
        for celeb in self.classes:
            celeb_dir = os.path.join(self.data_dir, celeb)
            for fname in sorted(os.listdir(celeb_dir)):
                if fname.lower().endswith(".jpg"):
                    img_path = os.path.join(celeb_dir, fname)
                    # here label is the celebrity name (string)
                    self.samples.append((img_path, self.class_to_idx[celeb]))  # use integer label here
                    #self.samples.append((img_path, celeb))

        self.transform = transform
        self.expanded_samples = []
        self.expand_samples()
        print("length", len(self.expanded_samples))
        print("Type", type(self.expanded_samples[0]))

    def expand_samples(self):
        for img_path, label in self.samples:
            img = Image.open(img_path).convert("RGB")
            if self.inversion:
                img = TF.rotate(img, 180)

            if self.transform:
                crops = self.transform(img)
            else:
                crops = [TF.to_tensor(img)]

            for crop in crops:
                #print("crop no", len(crops))
                self.expanded_samples.append((crop, label))
               

    def __len__(self):
        #return len(self.samples)
        return len(self.expanded_samples)

    def __getitem__(self, idx):
        # img_path, label = self.samples[idx]
        # img = Image.open(img_path).convert("RGB")

        # if self.inversion:
        #     img = TF.rotate(img, 180)  # <-- rotate 180 degrees

        # if self.transform:
        #     crops = self.transform(img) # Transformation pipeline
        # else:
        #     #crops = [img]
        #     crops = [TF.to_tensor(img)]

        # return crops, label

        img, label = self.expanded_samples[idx]
        return img, label

