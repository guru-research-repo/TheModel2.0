import yaml
import torch
import torchvision.transforms.functional as TF
import random
from transformations import *
import torchvision.transforms as transforms

class RandomCropper(torch.nn.Module):
    """
    Randomly crops the image into four different crops.
    Each call returns a list of four cropped versions of the input image.
    """
    def __init__(self, crop_scale=0.65):
        super().__init__()
        self.cropper = FourRandomCrops(crop_scale=crop_scale)
        
    def forward(self, img):
        return self.cropper(img)  # returns list of crops

# class RandomCropper(nn.Module):
#     """
#     Randomly crops the input image into `num_points` crops of size `crop_size`.
#     Each call returns a list of cropped versions of the input image.
#     """
#     def __init__(self, num_points=4, crop_size=224):
#         super().__init__()
#         self.num_points = num_points
#         self.crop_tool = transforms.RandomCrop(crop_size)

#     def forward(self, img):
#         """
#         Args:
#             img (PIL Image or Tensor): Input image to be cropped.

#         Returns:
#             list of images: List of cropped images.
#         """
#         return [self.crop_tool(img) for _ in range(self.num_points)]

class RandomRotator(torch.nn.Module):
    """
    Applies random rotation to each image or list of images.
    Rotates by a random angle between [-degrees, degrees].
    """
    def __init__(self, degrees=15):
        super().__init__()
        self.degrees = degrees
        
    def forward(self, img):
        if isinstance(img, list):
            # Rotate each crop separetely 
            return [TF.rotate(crop, random.uniform(-self.degrees, self.degrees)) for crop in img]
        else:
            return TF.rotate(img, random.uniform(-self.degrees, self.degrees))

# rotator for inversion
class FixedRotator(torch.nn.Module):
    """
    Rotates image or crops by a fixed angle (default: 180°) for inversion.
    """
    def __init__(self, angle=180):
        super().__init__()
        self.angle = angle

    def forward(self, img):
        if isinstance(img, list):
            return [TF.rotate(crop, self.angle) for crop in img]
        else:
            return TF.rotate(img, self.angle)

class Foveater(torch.nn.Module):
    """
    Applies foveation transform to simulate high-res center and blurry periphery.
    Works on either a single image or a list of images.
    """
    def __init__(self, crop_size=128, sigma=0.248, prNum=6):
        super().__init__()
        self.foveation = Foveation(crop_size=crop_size, sigma=sigma, prNum=prNum)

    def forward(self, img):
        if isinstance(img, list):
            out = []
            for crop in img:
                if not isinstance(crop, torch.Tensor):
                    crop = TF.to_tensor(crop)  # Convert only if needed
                out.append(self.foveation(crop))
            return out
        else:
            if not isinstance(img, torch.Tensor):
                img = TF.to_tensor(img)  # Convert only if needed
            return self.foveation(img)

class LogPolarTransformer(torch.nn.Module):
    """
    Applies a log-polar transformation to images.
    This warps the image using a logarithmic-polar coordinate system.
    """
    def __init__(self, input_shape=(128, 128), output_shape=(128, 128)):
        super().__init__()
        self.logpolar = LogPolar(
            input_shape=input_shape,
            output_shape=output_shape,
            smoothing=None,
            mask=True
        )

    def forward(self, img):
        if isinstance(img, list):
            out = []
            for crop in img:
                if not isinstance(crop, torch.Tensor):
                    crop = TF.to_tensor(crop)  # Convert only if needed
                out.append(self.logpolar(crop))
            return out
        else:
            if not isinstance(img, torch.Tensor):
                img = TF.to_tensor(img)  # Convert only if needed
            return self.logpolar(img)

# ------------------------------------------------------------------------------
# Build the Transformation Pipeline
# ------------------------------------------------------------------------------

def build_transform_pipeline(config, inversion=None):
    """
    Builds a sequence of transformations based on given config.
    
    Args:
        config (dict): Should have keys:
            - 'transformations': list of transformation names in order
            - 'params'         : dictionary of parameters for transformations

    Returns:
        torch.nn.Sequential: A sequence of transformation modules
    """
    transform_modules = []

    for transform_name in config['transformations']:
        if transform_name == 'crop':
            transform_modules.append(RandomCropper(crop_scale=config['params']['crop_scale']))
        elif transform_name == 'rotate':
            if inversion==True:
                transform_modules.append(FixedRotator(angle=180))
            elif inversion==False:
                transform_modules.append(FixedRotator(angle=0))
            else:
                transform_modules.append(RandomRotator(degrees=config['params']['rotation_degrees']))
        
            #transform_modules.append(RandomRotator(degrees=config['params']['rotation_degrees']))
        elif transform_name == 'foveation':
            transform_modules.append(Foveater(
                crop_size=config['params']['crop_size'],
                sigma=config['params']['sigma'],
                prNum=config['params']['prNum']
            ))
        elif transform_name == 'logpolar':
            transform_modules.append(LogPolarTransformer(
                input_shape=(config['params']['crop_size'], config['params']['crop_size']),
                output_shape=tuple(config['params']['logpolar_output_shape'])
            ))
        else:
            raise ValueError(f"Unknown transformation: {transform_name}")

    return torch.nn.Sequential(*transform_modules)
