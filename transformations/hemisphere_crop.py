import torch
import torchvision.transforms as transforms

class HemisphereCrop(torch.nn.Module):

    def __init__(self, crop_size, start_x = None, end_x = None, start_y = None, end_y = None):
        super().__init__()

        self.start_y = start_y
        self.end_y = end_y

        self.start_x = start_x
        self.end_x = end_x

    def __call__(self, img, *args):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        
        if self.start_y is not None and self.end_y is not None:
            start_y, end_y = [ 
                    int(value * img.shape[-2]) if isinstance(value, float) else value
                    for value 
                    in [self.start_y, self.end_y]
            ]

            if start_y > end_y:
                return torch.cat([img[..., start_y:, :].clone(), img[..., :end_y, :].clone()], axis=-2)
            
            return img[..., start_y:end_y, :].clone()
        
        elif self.start_x is not None and self.end_x is not None:
            start_x, end_x = [ 
                    int(value * img.shape[-1]) if isinstance(value, float) else value
                    for value 
                    in [self.start_x, self.end_x]
            ]

            if start_x > end_x:
                return torch.cat([img[..., start_x:].clone(), img[..., :end_x].clone()], axis=-1)
            
            return img[..., start_x:end_x].clone()

        return img
