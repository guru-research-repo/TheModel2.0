import torch

class Replicate(torch.nn.Module):
    def __init__(self, count = 1):
        self.count = count

    def __call__(self, img, *args, **kwargs):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        return img.repeat(self.count, 1, 1, 1)
