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

 
# class RandomRotator(torch.nn.Module):
#     """
#     Applies random rotation to each image or list of images.
#     Rotates by a random angle between [-degrees, degrees] using RandomRotation.
#     """
#     # def __init__(self, degrees=15):
#     #     super().__init__()
#     #     self.degrees = degrees  # Just store degrees range

#     # def rotation(self, img):
#     #     """
#     #     Rotate a single image using OpenCV warpAffine.
#     #     Input: img (Tensor, shape [C, H, W])
#     #     Output: rotated_img (Tensor, shape [C, H, W])
#     #     """
#     #     if isinstance(img, torch.Tensor):
#     #         img_np = TF.to_pil_image(img)  # Convert tensor to PIL
#     #         img_np = np.array(img_np)      # Convert PIL to numpy
#     #     else:
#     #         raise ValueError(f"Expected input to be a torch.Tensor, got {type(img)} instead.")

#     #     (h, w) = img_np.shape[:2]
#     #     center = (w // 2, h // 2)
#     #     angle = random.uniform(-self.degrees, self.degrees)  # Random angle
#     #     M = cv2.getRotationMatrix2D(center, angle, 1.0)
#     #     rotated = cv2.warpAffine(img_np, M, (w, h), flags=cv2.INTER_LINEAR)

#     #     rotated_tensor = TF.to_tensor(rotated)  # Convert back to tensor
#     #     return rotated_tensor

#     def __init__(self, degrees=15):
#         super().__init__()
#         self.rotation = transforms.RandomRotation(
#             degrees=(-degrees, degrees),
#             interpolation=InterpolationMode.BILINEAR # FIX
#         )
#     def forward(self, img):
        
#         if isinstance(img, list):
#             # Rotate each crop separately
#             return [self.rotation(crop) for crop in img]
#         else:
#             return self.rotation(img)



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


        
        # if isinstance(img, list):
        #     # Apply foveation to each crop separately
        #     return [self.foveation(TF.to_tensor(crop)) for crop in img]
        # else:
        #     return self.foveation(TF.to_tensor(img))

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
            # Apply log-polar transform to each crop separately
            return [self.logpolar(crop) for crop in img]
        else:
            return self.logpolar(img)

# ------------------------------------------------------------------------------
# Build the Transformation Pipeline
# ------------------------------------------------------------------------------

def build_transform_pipeline(config):
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
            transform_modules.append(RandomRotator(degrees=config['params']['rotation_degrees']))
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
