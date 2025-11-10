import os
import torch
import torchvision.transforms.functional as TF
from utils_new import get_label_mapping, label_to_one_hot
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

def load_dataset(dataset, identity=4, task="train", num_salient_points=4):
    if dataset == "celeb":
        ds = CelebAFaceIDDataset(root_dir="processed_data", split=task)
    elif dataset == "faces":
        ds = CelebrityFacesDataset(root_dir="data/faces_cleaned", num_identities=identity, split=task, type="faces")
    elif dataset == "objects":
        ds = ImageNetObjectsDataset(root_dir="processed_data", num_classes=identity, split=task)
    elif dataset == "salience":
        ds = SalienceDataset(root_dir="processed_data/salience/updated_faces", 
                             num_identities=32, 
                             split=task, 
                             num_salient_points=num_salient_points)
    else:
        ds = CelebrityFacesDataset(root_dir="data", num_identities=identity, split=task, type=dataset)
    return ds

# Used only for salience training
def make_datasets(ident, num_salient_points, faces_data="updated"):
    return {
        "train_upright": SalienceDatasetBatched(
            root_dir=f"32_identities_32_fixations_preprocessed_faces/32_fixations",
            num_identities=ident,
            split="train_upright",
            num_salient_points=num_salient_points
        ),
        "valid_upright": SalienceDataset(
            root_dir=f"32_identities_32_fixations_preprocessed_faces/32_fixations",
            num_identities=ident,
            split="valid_upright",
            num_salient_points=num_salient_points
        ),
        "valid_inverted": SalienceDataset(
            root_dir=f"32_identities_32_fixations_preprocessed_faces/32_fixations",
            num_identities=ident,
            split="valid_inverted",
            num_salient_points=num_salient_points
        ),
    }

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
    def __init__(self, root_dir: str, num_identities: int, split: str, type: str):
        """
        Args:
            root_dir (str): path to "/dataset"
            num_identities (int): 4, 8, …, 128
            split (str): one of "train", "valid", "test"
            type (str): "faces" or "dogs"
        """
        # build the path to e.g. "/dataset/faces/faces/8_identities/train"
        self.data_dir = os.path.join(
            root_dir, 
            #type, 
            type, 
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

        self.map = get_label_mapping(type=type)

        # collect (image_path, label) tuples
        self.samples = []
        for celeb in self.classes:
            celeb_dir = os.path.join(self.data_dir, celeb)
            for fname in sorted(os.listdir(celeb_dir)):
                if fname.lower().endswith(".jpg") or fname.lower().endswith(".jpeg"):
                    img_path = os.path.join(celeb_dir, fname)
                    # here label is the celebrity name (string)
                    self.samples.append((img_path, celeb))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = TF.to_tensor(img)
        label = label_to_one_hot(label, self.map)
        # label = torch.tensor(int(label), dtype=torch.long)
        return img, label

class ImageNetObjectsDataset(Dataset):
    def __init__(self, root_dir: str, num_classes: int, split: str):
        """
        Args:
            root_dir (str): path to "processed_data"
            num_classes (int): number of object classes (4, 8, ..., 128)
            split (str): one of "train", "valid", or "test"
        """
        self.samples = []
        self.data_dir = Path(root_dir) / "ImageNet1k" / "ImageNet1k" / f"{num_classes}_objects" / split
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Could not find data directory: {self.data_dir}")

        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for class_name in self.classes:
            img_dir = self.data_dir / class_name
            for img_path in img_dir.glob("*.png"):
                self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = TF.to_tensor(img)
        label = torch.tensor(label, dtype=torch.long)
        return img, label

# class SalienceShuffledDataset(Dataset):
#     def __init__(self, root_dir: str, num_identities: int, split: str, type: str,
#                  num_salient_points: int = 4, seed: int = 42):
#         """
#         Args:
#             root_dir (str): base path, e.g. "processed_data/salience/updated_faces"
#             num_identities (int): e.g. 32
#             split (str): one of "train", "valid", "test"
#             type (str): "faces" or "dogs"
#             num_salient_points (int): how many processed variants per base image
#             seed (int): global seed for reproducibility
#         """
#         self.num_salient_points = num_salient_points
#         self.seed = seed

#         # build the path to e.g. ".../faces/32_identities/train"
#         # self.data_dir = os.path.join(root_dir, type, f"{num_identities}_identities", split)
#         self.data_dir = os.path.join(root_dir, split)
#         if not os.path.isdir(self.data_dir):
#             raise ValueError(f"Directory not found: {self.data_dir}")

#         # list all identity folders
#         self.classes = sorted(
#             d for d in os.listdir(self.data_dir)
#             if os.path.isdir(os.path.join(self.data_dir, d))
#         )

#         self.map = get_label_mapping(type=type)

#         # Collect base images and all their processed variants
#         self.samples = []  # [(base_img_id, [proc_paths...], label), ...]
#         rng = np.random.RandomState(self.seed)

#         for ident in self.classes:
#             ident_dir = os.path.join(self.data_dir, ident)
#             # group by base image number (before "_proc")
#             base_dict = {}
#             for fname in sorted(os.listdir(ident_dir)):
#                 if fname.endswith(".png") and "_fix" in fname:
#                     base_num = fname.split("_fix")[0]  # base image
#                     base_dict.setdefault(base_num, []).append(os.path.join(ident_dir, fname))

#             for base_num, proc_list in base_dict.items():
#                 proc_list = sorted(proc_list)  # ensure consistent order
#                 # deterministically shuffle using seed + base_num
#                 local_rng = np.random.RandomState(self.seed + hash(base_num) % (2**32))
#                 local_rng.shuffle(proc_list)
#                 self.samples.append((base_num, proc_list, ident))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         base_num, proc_list, label = self.samples[idx]
#         # take first n processed variants
#         chosen = proc_list[:self.num_salient_points]

#         imgs = [TF.to_tensor(Image.open(p).convert("RGB")) for p in chosen]
#         imgs = torch.stack(imgs, dim=0)  # (n, C, H, W)

#         label = label_to_one_hot(label, self.map)
#         return imgs, label

class SalienceDataset(Dataset):
    def __init__(self, root_dir: str, num_identities: int, split: str,
                 num_salient_points: int = 4):
        """
        Args:
            root_dir (str): base path, e.g. "processed_data/salience/updated_faces"
            num_identities (int): e.g. 32
            split (str): one of "train", "valid", "test"
            num_salient_points (int): how many processed variants per base image
        """
        self.num_salient_points = num_salient_points

        # build the path to e.g. ".../faces/32_identities/train"
        # self.data_dir = os.path.join(root_dir, f"{num_identities}_identities", split)
        self.data_dir = os.path.join(root_dir, split)
        if not os.path.isdir(self.data_dir):
            raise ValueError(f"Directory not found: {self.data_dir}")

        # list all identity folders
        self.classes = sorted(
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        )

        self.map = get_label_mapping(type="faces")

        # Collect base images and all their processed variants
        self.samples = []  # [(base_img_id, [proc_paths...], label), ...]

        for ident in self.classes:
            ident_dir = os.path.join(self.data_dir, ident)
            # group by base image number (before "_proc")
            base_dict = {}
            for fname in sorted(os.listdir(ident_dir)):
                if fname.endswith(".png") and "_fix" in fname:
                    base_num = fname.split("_fix")[0]  # base image
                    base_dict.setdefault(base_num, []).append(os.path.join(ident_dir, fname))

            for base_num, proc_list in base_dict.items():
                proc_list = sorted(proc_list)  # ensure consistent order
                self.samples.append((base_num, proc_list, ident))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base_num, proc_list, label = self.samples[idx]
        # print("base num valid", base_num)
        
        # print("label valid", label)
        # take first n processed variants
        chosen = proc_list[:self.num_salient_points]
        # print("chosen valid", chosen)

        imgs = [TF.to_tensor(Image.open(p).convert("RGB")) for p in chosen]
        imgs = torch.stack(imgs, dim=0)  # (n, C, H, W)

        label = label_to_one_hot(label, self.map)
        # print("label valid one hot encoded", label)
        return imgs, label
    

"""
Rather than returning all fixations in the same image at once, 
compile the dataset as normal, such that a random number are present in each mini-batch.
"""
class SalienceDatasetBatched(Dataset):
    def __init__(self, root_dir: str, num_identities: int, split: str,
                 num_salient_points: int = 4):
        """
        Args:
            root_dir (str): base path, e.g. "processed_data/salience/updated_faces"
            num_identities (int): e.g. 32
            split (str): one of "train", "valid", "test"
            num_salient_points (int): how many processed variants per base image
        """
        self.num_salient_points = num_salient_points

        # build the path to e.g. ".../faces/32_identities/train"
        # self.data_dir = os.path.join(root_dir, f"{num_identities}_identities", split)
        self.data_dir = os.path.join(root_dir, split)
        if not os.path.isdir(self.data_dir):
            raise ValueError(f"Directory not found: {self.data_dir}")

        # list all identity folders
        self.classes = sorted(
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        )

        self.map = get_label_mapping(type="faces")

        # Collect base images and all their processed variants
        self.samples = []  # [(path, label), ...]

        for ident in self.classes:
            ident_dir = os.path.join(self.data_dir, ident)
            # 32_identities_32_fixations_preprocessed_faces/32_fixations/train_upright/AdamRippon/0_fix001.png
            for fname in os.listdir(ident_dir):
                for i in range(num_salient_points):
                    if fname.endswith(f"fix{i+1:03d}.png"):
                    # if fname.endswith(f'fix{i}.png'): # "procX.png" or "procXX.png"
                        self.samples.append((os.path.join(ident_dir, fname), ident))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # print("image path in Datasets_new", path)
        # print("label in Datasets_new file", label)
        img = TF.to_tensor(Image.open(path).convert("RGB"))

        label = label_to_one_hot(label, self.map)
        # print("label_to_one-hot in Datasets_new file", label)
        return img, label