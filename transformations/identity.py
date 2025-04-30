import numpy as np
import torch
import torchvision.transforms as transforms

class Identity(torch.nn.Module):
    def __call__(self, img, *args, **kwargs):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        return img
