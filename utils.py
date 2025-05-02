import math
import matplotlib.pyplot as plt
from Datasets import *
import torch
import numpy as np

def load_raw_data(name, identity=4, task="train"):
    print("Start loading raw data arrays")
    if name == "celeb":
        images, labels = load_celeba_face_id_np(split=task)
    print(f"Loaded dataset {name}, task = {task}")
    return images, labels

def load_dataset(dataset, identity=4, task="train"):
    if dataset == "celeb":
        ds = CelebAFaceIDDataset(root_dir="data", split=task)
    elif dataset == "faces":
        ds = CelebrityFacesDataset(root_dir="data", num_identities=identity, split=task)
    return ds

def load_celeba_face_id_np(root_dir: str = "data",
                           split: str = "train"
                          ) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the CelebA_HQ facial identity dataset into NumPy arrays.

    Args:
        root_dir (str): Path to folder containing "CelebA_HQ_facial_identity_dataset".
        split    (str): "train" or "test".

    Returns:
        images_np (np.ndarray): Float32 array of shape (N, 3, H, W), values in [0,1].
        labels_np (np.ndarray): Int64 array of shape (N,), identity labels.
    """
    samples = []
    root_path   = Path(root_dir)
    split_dir   = root_path / "CelebA_HQ_facial_identity_dataset" / split

    if not split_dir.is_dir():
        raise FileNotFoundError(f"Could not find split directory: {split_dir!r}")

    # collect (path, label) pairs
    for person_dir in sorted(split_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        try:
            label = int(person_dir.name)
        except ValueError:
            continue
        for img_path in sorted(person_dir.glob("*.jpg")):
            samples.append((img_path, label))

    # preallocate lists
    images = []
    labels = []

    for img_path, label in samples:
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0     # H×W×3, float32
        arr = arr.transpose(2, 0, 1)                     # → 3×H×W
        images.append(arr)
        labels.append(label)

    images_np = np.stack(images, axis=0)                # N×3×H×W
    labels_np = np.array(labels, dtype=np.int64)        # N

    return images_np, labels_np

def show_images(imgs: list[torch.Tensor], cols=3, figsize=None):
    """
    Display one or more images (PyTorch tensors or array-likes) in a grid.

    Args:
        imgs (Tensor or list of Tensors / array-likes):
            - torch.Tensor of shape (C,H,W) or (H,W)
            - torch.Tensor of shape (N,C,H,W) or (N,H,W)
            - list/tuple of the above, or list of NumPy arrays
        cols (int): Number of columns in the grid.
        figsize (tuple, optional): Figure size in inches; if None, auto-scaled.
    """
    # Normalize input to a flat list of images
    if isinstance(imgs, torch.Tensor):
        # single image or batch tensor
        if imgs.ndim == 2 or imgs.ndim == 3:
            imgs = [imgs]
        elif imgs.ndim == 4:
            imgs = list(imgs)
        else:
            raise ValueError(f"Tensor of unsupported shape {imgs.shape}")
    else:
        # assume iterable of images
        imgs = list(imgs)

    # Convert all to NumPy H×W×C (or H×W) arrays
    np_imgs = []
    for img in imgs:
        if isinstance(img, torch.Tensor):
            t = img.detach().cpu()
            if t.ndim == 3:
                # C×H×W → H×W×C
                t = t.permute(1, 2, 0)
            elif t.ndim == 2:
                # H×W — leave as is
                pass
            else:
                raise ValueError(f"Tensor of unsupported shape {img.shape}")
            arr = t.numpy()
            # If floats, clip to [0,1]
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr, 0.0, 1.0)
            np_imgs.append(arr)
        else:
            # assume array-like
            arr = np.asarray(img)
            if arr.ndim not in (2, 3):
                raise ValueError(f"Array of unsupported shape {arr.shape}")
            np_imgs.append(arr)

    n = len(np_imgs)
    if n == 0:
        return

    rows = math.ceil(n / cols)
    if figsize is None:
        figsize = (cols * 2, rows * 2)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)  # flatten even if only one subplot

    for ax, im in zip(axes, np_imgs):
        ax.imshow(im, interpolation='nearest')
        ax.axis('off')

    # Turn off any unused subplots
    for ax in axes[n:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
