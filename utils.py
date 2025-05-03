import math
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import torch.nn.functional as F

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

def get_label_mapping(root_dir="data", num_identities=128, split="test"):
    """
    Scan "<root_dir>/faces/faces/{num_identities}_identities/{split}"
    and return a dict mapping each class-name (folder name) to a unique index.

    Args:
        root_dir (str): base path to your "/dataset" folder
        num_identities (int): 4, 8, …, 128
        split (str): "train", "valid" or "test"
    Returns:
        dict: { class_name: idx, … }
    """
    data_dir = os.path.join(
        root_dir,
        "faces",
        "faces",
        f"{num_identities}_identities",
        split
    )
    if not os.path.isdir(data_dir):
        raise ValueError(f"Directory not found: {data_dir!r}")

    classes = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )
    mapping = {}
    for idx, cls_name in enumerate(classes):
        mapping[cls_name] = idx
    return mapping


def label_to_one_hot(label, mapping):
    """
    Convert a string label into a one-hot tensor.

    Args:
        label (str): the class-name (must be a key in mapping)
        mapping (dict): mapping returned by get_label_mapping()
    Returns:
        torch.FloatTensor of shape (num_classes,), e.g. [0,0,1,0,…]
    """
    idx = mapping[label]
    num_classes = len(mapping)
    # create one-hot and cast to float
    return F.one_hot(torch.tensor(idx, dtype=torch.long),
                     num_classes=num_classes).float()
