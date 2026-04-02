import os
import torch
import torchvision.transforms.functional as TF
from utils import get_label_mapping, label_to_one_hot
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import time
from functools import lru_cache

def load_dataset(dataset, identity=4, task="train", num_salient_points=4, lp = True):
    if dataset == "celeb":
        ds = CelebAFaceIDDataset(root_dir="processed_data", split=task)
    elif dataset == "faces":
        ds = CelebrityFacesDataset(root_dir="data", num_identities=identity, split=task, type="faces_cleaned")
    elif dataset == "objects":
        ds = ImageNetObjectsDataset(root_dir="processed_data", num_classes=identity, split=task)
    elif dataset == "salience":
        ds = make_datasets(ident=identity, num_salient_points=num_salient_points, lp=lp)
    else:
        ds = CelebrityFacesDataset(root_dir="data", num_identities=identity, split=task, type=dataset)
    return ds

# Used only for salience training
# def make_datasets(ident, num_salient_points, lp = True, dataset="salience"):
#     root = f"processed_data/{dataset}/updated_faces/128_identities" if lp else f"processed_data/{dataset}/cnn_faces/128_identities"

#     return {
#         "train": SalienceDatasetBatched(
#             root_dir=root,
#             num_identities=ident,
#             split="train",
#             num_salient_points=num_salient_points
#         ),
#         "valid": SalienceDataset(
#             root_dir=root,
#             num_identities=ident,
#             split="valid",
#             num_salient_points=num_salient_points
#         ),
#         "test": SalienceDataset(
#             root_dir=root,
#             num_identities=ident,
#             split="test",
#             num_salient_points=num_salient_points
#         ),
#     }

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
            type, 
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

class SalienceShuffledDataset(Dataset):
    def __init__(self, root_dir: str, num_identities: int, split: str, type: str,
                 num_salient_points: int = 4, seed: int = 42):
        """
        Args:
            root_dir (str): base path, e.g. "processed_data/salience/updated_faces"
            num_identities (int): e.g. 32
            split (str): one of "train", "valid", "test"
            type (str): "faces" or "dogs"
            num_salient_points (int): how many processed variants per base image
            seed (int): global seed for reproducibility
        """
        self.num_salient_points = num_salient_points
        self.seed = seed

        # build the path to e.g. ".../faces/32_identities/train"
        self.data_dir = os.path.join(root_dir, type, f"{num_identities}_identities", split)
        if not os.path.isdir(self.data_dir):
            raise ValueError(f"Directory not found: {self.data_dir}")

        # list all identity folders
        self.classes = sorted(
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        )

        self.map = get_label_mapping(type=type)

        # Collect base images and all their processed variants
        self.samples = []  # [(base_img_id, [proc_paths...], label), ...]
        rng = np.random.RandomState(self.seed)

        for ident in self.classes:
            ident_dir = os.path.join(self.data_dir, ident)
            # group by base image number (before "_proc")
            base_dict = {}
            for fname in sorted(os.listdir(ident_dir)):
                if fname.endswith(".png") and "_proc" in fname:
                    base_num = fname.split("_proc")[0]  # base image
                    base_dict.setdefault(base_num, []).append(os.path.join(ident_dir, fname))

            for base_num, proc_list in base_dict.items():
                proc_list = sorted(proc_list)  # ensure consistent order
                # deterministically shuffle using seed + base_num
                local_rng = np.random.RandomState(self.seed + hash(base_num) % (2**32))
                local_rng.shuffle(proc_list)
                self.samples.append((base_num, proc_list, ident))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base_num, proc_list, label = self.samples[idx]
        # take first n processed variants
        chosen = proc_list[:self.num_salient_points]

        imgs = [TF.to_tensor(Image.open(p).convert("RGB")) for p in chosen]
        imgs = torch.stack(imgs, dim=0)  # (n, C, H, W)

        label = label_to_one_hot(label, self.map)
        return imgs, label

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
        class_path = os.path.join(
            'data',
            'dogs1k',
            'dogs1k',
            f"{num_identities}_identities",
            split
        )
        self.classes = sorted(
            d for d in os.listdir(class_path)
            if os.path.isdir(os.path.join(class_path, d))
        )

        self.map = get_label_mapping(type="dogs1k")

        # Collect base images and all their processed variants
        self.samples = []  # [(base_img_id, [proc_paths...], label), ...]

        for ident in self.classes:
            ident_dir = os.path.join(self.data_dir, ident)
            # group by base image number (before "_proc")
            base_dict = {}
            for fname in sorted(os.listdir(ident_dir)):
                if fname.endswith(".png") and "_proc" in fname:
                    base_num = fname.split("_proc")[0]  # base image
                    base_dict.setdefault(base_num, []).append(os.path.join(ident_dir, fname))

            for base_num, proc_list in base_dict.items():
                proc_list = sorted(proc_list)  # ensure consistent order
                self.samples.append((base_num, proc_list, ident))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base_num, proc_list, label = self.samples[idx]
        # take first n processed variants
        chosen = proc_list[:self.num_salient_points]

        imgs = [TF.to_tensor(Image.open(p).convert("RGB")) for p in chosen]
        imgs = torch.stack(imgs, dim=0)  # (n, C, H, W)

        label = label_to_one_hot(label, self.map)
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
        class_path = os.path.join(
            'data',
            'dogs1k',
            'dogs1k',
            f"{num_identities}_identities",
            split
        )
        self.classes = sorted(
            d for d in os.listdir(class_path)
            if os.path.isdir(os.path.join(class_path, d))
        )

        self.map = get_label_mapping(type="dogs1k")

        # Collect base images and all their processed variants
        self.samples = []  # [(path, label), ...]

        for ident in self.classes:
            ident_dir = os.path.join(self.data_dir, ident)

            for fname in os.listdir(ident_dir):
                for i in range(num_salient_points):
                    if fname.endswith(f'c{i}.png'): # "procX.png" or "procXX.png"
                        self.samples.append((os.path.join(ident_dir, fname), ident))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = TF.to_tensor(Image.open(path).convert("RGB"))

        label = label_to_one_hot(label, self.map)
        return img, label    

class SalienceDatasetV3(Dataset):
    def __init__(self, root_dir: str, num_identities: int, split: str, type: str, 
        num_salient_points: int = 4):
        """
        Args:
            root_dir (str): path to "/dataset"
            num_identities (int): 4, 8, …, 128
            split (str): one of "train", "valid", "test"
            type (str): "faces" or "dogs"
        """
        self.num_salient_points = num_salient_points
        self.proc_dir = os.path.join(root_dir, split)

        split = 'valid' if split == 'test' else split

        # build the path to e.g. "/dataset/faces/faces/8_identities/train"
        self.data_dir = os.path.join(
            'data', 
            type, 
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
        self.cache = {} # label -> loaded salience dict
        for label in self.classes:
            label_dir = Path(f'{self.data_dir}/{label}')
            salience_path = Path(f'{self.proc_dir}/{label}/salience.pt')

            data = torch.load(salience_path)
            stems = data["stems"]
            # print(stems)
            stem_to_idx = {stem[0]: i for i, stem in enumerate(stems)}

            # cache once
            self.cache[label] = data['salience_points']

            for img_path in label_dir.iterdir():
                if img_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:

                    stem = img_path.stem
                    if stem not in stem_to_idx:
                        print(stem_to_idx.keys())
                        print(img_path)
                    img_idx = stem_to_idx[stem]

                    # base_num = fname.split(".")[0]  # base image
                    # here label is the celebrity name (string)
                    self.samples.append((img_path, label, img_idx))

        # print(self.samples[0])
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, img_idx = self.samples[idx]
        
        # take first n processed variants
        data = self.cache[label]
        salience_points = data[img_idx,:self.num_salient_points]

        imgs = TF.to_tensor(Image.open(img_path).convert("RGB")).unsqueeze(0).repeat(self.num_salient_points,1,1,1) # (n, C, H, W)

        crop_size = 180
        cropped = torch.empty((self.num_salient_points,3,crop_size,crop_size))

        for i in range(self.num_salient_points):
            center = salience_points[i]
            cropped[i] = TF.crop(imgs[i],
                                        top=center[1]-crop_size//2,
                                        left=center[0]-crop_size//2, 
                                        height=crop_size, width=crop_size)

        label = label_to_one_hot(label, self.map)
        return imgs, label, salience_points

"""
Rather than returning all fixations in the same image at once, 
compile the dataset as normal, such that a random number are present in each mini-batch.
"""
class SalienceDatasetBatchedV3(Dataset):
    def __init__(self, root_dir: str, num_identities: int, split: str, type: str, 
        num_salient_points: int = 4):
        """
        Args:
            root_dir (str): path to "/dataset"
            num_identities (int): 4, 8, …, 128
            split (str): one of "train", "valid", "test"
            type (str): "faces" or "dogs"
        """
        self.num_salience_points = num_salient_points
        self.proc_dir = os.path.join(root_dir, split)

        split = 'valid' if split == 'test' else split
        
        # build the path to e.g. "/dataset/faces/faces/8_identities/train"
        self.data_dir = os.path.join(
            'data', 
            type, 
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
        self.cache = {} # label -> loaded salience dict
        for label in self.classes:
            label_dir = Path(f'{self.data_dir}/{label}')
            # print('label dir', label_dir)
            
            salience_path = Path(f'{self.proc_dir}/{label}/salience.pt') # fix this path
            # print('salience_path:', salience_path)

            if not salience_path.exists():
                continue

            data = torch.load(salience_path)
            stems = data["stems"]
            # print(stems)
            stem_to_idx = {stem[0]: i for i, stem in enumerate(stems)}

            # cache once
            self.cache[label] = data['salience_points']

            for img_path in label_dir.iterdir():
                if img_path.suffix.lower() not in ['.jpg', '.png', '.jpeg']:
                    continue

                stem = img_path.stem
                # print(stem_to_idx.keys())
                img_idx = stem_to_idx[stem]
                if stem not in stem_to_idx:
                    continue

                # load each salience point
                for sp_idx in range(self.num_salience_points):
                    self.samples.append({
                        "img_path": img_path,
                        "label": label,
                        "img_idx": img_idx,
                        "sp_idx": sp_idx
                    })

        # print(self.samples[0])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample['label']

        # Load image
        img = Image.open(sample["img_path"]).convert("RGB")
        img_tensor = TF.to_tensor(img)

        # Get one salience point
        data = self.cache[label]
        salience_point = data[ sample["img_idx"], sample["sp_idx"] ]  # (2,)
        crop_size = 180
        img_tensor = TF.crop(img_tensor,
                                top=salience_point[1]-crop_size//2,
                                left=salience_point[0]-crop_size//2, 
                                height=crop_size, width=crop_size)


        label = label_to_one_hot(label, self.map)

        return {
            "image": img_tensor,              # (C, H, W)
            "salience": salience_point,       # (2,)
            "label": label
        }
    
### ------------------------------------------------------
# Used only for salience training
def make_datasets(ident, num_salient_points, lp = True, dataset="salience"):
    root = f"processed_data/{dataset}/updated_objects"

    return {
        "train": SalienceDatasetBatchedV3(
            root_dir=root,
            num_identities=ident,
            split="train",
            num_salient_points=num_salient_points,
            type='dogs1k'
            # lp=lp
        ),
        "valid": SalienceDatasetV3(
            root_dir=root,
            num_identities=ident,
            split="valid",
            num_salient_points=num_salient_points,
            type='dogs1k'
            # lp=lp
        ),
        "test": SalienceDatasetV3(
            root_dir=root,
            num_identities=ident,
            split="test",
            num_salient_points=num_salient_points,
            type='dogs1k'
            # lp=lp
        ),
    }

class SalienceDataset(Dataset):
    def __init__(self, root_dir: str, num_identities: int, split: str,
                 num_salient_points: int = 4, lp=True):
        """
        Args:
            root_dir (str): base path, e.g. "processed_data/salience/updated_faces"
            num_identities (int): e.g. 32
            split (str): one of "train", "valid", "test"
            num_salient_points (int): how many processed variants per base image
            lp (bool): whether we are loading lp or cnn model images
        """
        self.num_salient_points = num_salient_points
        self.lp = lp

        # build the path to e.g. ".../faces/32_identities/train"
        self.data_dir = os.path.join(root_dir, split)
        if not os.path.isdir(self.data_dir):
            raise ValueError(f"Directory not found: {self.data_dir}")

        class_path = os.path.join(
            'data',
            'dogs1k',
            'dogs1k',
            f"{num_identities}_identities",
            split
        )
        self.classes = sorted(
            d for d in os.listdir(class_path)
            if os.path.isdir(os.path.join(class_path, d))
        )

        self.map = get_label_mapping(type="dogs1k")

        # Collect base images and all their processed variants
        self.samples = []  # [(path, label), ...]

        for ident in self.classes:
            ident_dir = os.path.join(self.data_dir, ident)

            for fname in os.listdir(ident_dir):
                if fname.endswith(f'.pt'):
                    self.samples.append((os.path.join(ident_dir, fname), ident))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data = torch.load(path)

        img = data['lp'] if self.lp else data['cnn']
        
        # Convert uint8 [0, 255] -> float [0, 1]
        img = img[:self.num_salient_points].float() #/ 255.0

        label = label_to_one_hot(label, self.map)
        return img, label
    

"""
Rather than returning all fixations in the same image at once, 
compile the dataset as normal, such that a random number are present in each mini-batch.
"""
class SalienceDatasetBatched(Dataset):
    def __init__(self, root_dir: str, num_identities: int, split: str,
                 num_salient_points: int = 4, lp=True):
        """
        Args:
            root_dir (str): base path, e.g. "processed_data/salience/updated_faces"
            num_identities (int): e.g. 32
            split (str): one of "train", "valid", "test"
            num_salient_points (int): how many processed variants per base image
            lp (bool): whether we are loading lp or cnn model images
        """
        self.num_salient_points = num_salient_points
        self.num_identities = num_identities
        self.lp = lp

        # build the path to e.g. ".../faces/32_identities/train"
        self.data_dir = os.path.join(root_dir, split)
        if not os.path.isdir(self.data_dir):
            raise ValueError(f"Directory not found: {self.data_dir}")

        # list all identity folders
        class_path = os.path.join(
            'data',
            'dogs1k',
            'dogs1k',
            f"{num_identities}_identities",
            split
        )
        self.classes = sorted(
            d for d in os.listdir(class_path)
            if os.path.isdir(os.path.join(class_path, d))
        )

        self.map = get_label_mapping(type="dogs1k")

        # Collect base images and all their processed variants
        self.samples = []  # [(path, label), ...]

        for ident in self.classes:
            ident_dir = os.path.join(self.data_dir, ident)

            for fname in os.listdir(ident_dir):
                if fname.endswith(f'.pt'):
                    self.samples.append((os.path.join(ident_dir, fname), ident))

        # Each file contributes num_salient_points individual images
        self.total_samples = len(self.samples) * self.num_salient_points


    def __len__(self):
        return self.total_samples

    @lru_cache(maxsize=512)  # tune this
    def _load_file(self, path):
        return torch.load(path, map_location="cpu")

    def __getitem__(self, idx):
        # Figure out which file and which fixation index
        file_idx = idx // self.num_salient_points
        fix_idx = idx % self.num_salient_points

        path, label = self.samples[file_idx]
        data = self._load_file(path)

        # Choose 'lp' or 'cnn' and select the specific fixation
        img = data['lp'][fix_idx] if self.lp else data['cnn'][fix_idx]
        
        # Convert uint8 [0, 255] -> float [0, 1]
        img = img.float() #/ 255.0

        label = label_to_one_hot(label, self.map)
        return img, label
        