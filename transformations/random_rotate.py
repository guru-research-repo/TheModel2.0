import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class RandomRotate(torch.nn.Module):

    def __init__(self, max_rotate):
        super().__init__()
        self.max_rotate=max_rotate
        self.rotate = transforms.RandomRotation(max_rotate)

    def __call__(self, imgs, *args):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        return torch.stack([self.rotate(img) for img in imgs])