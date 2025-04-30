import numpy as np
import torch
import torchvision.transforms as transforms

class Resize(torch.nn.Module):

    def __init__(self, num_points, crop_size):
        super().__init__()

        self.num_points = num_points
        self.crop_tool = transforms.Resize(crop_size)

    def __call__(self, img, *args):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        img = self.crop_tool(img)
        return torch.stack([img for _ in range(self.num_points)])
