import torch
import torch.nn as nn
from torchvision import transforms
from transformations import (
        SalienceCrop,
        SalienceSampling,
        LogPolar, 
        NRandomCrop, 
        Compose, 
        Resize, 
        Foveation, 
        RandomRotate,
        Identity
)

from config.transformation_config import *

from typing import Tuple, Union, Callable, Optional, Any

import time
# from utils import Timer

class Pipeline(torch.nn.Module):
    def __init__(self, 
            config,
            normalize = torch.nn.Identity(), 
            flatten = torch.nn.Identity()):
        
        super().__init__()

        self.tensorize = transforms.ToTensor()

        self.normalize = normalize

        if config.crop.type == 'random': 
            self.transform = transforms.RandomResizedCrop(config.crop.crop_size)
        elif config.crop.type == 'salience':
            self.transform = SalienceCrop(config.crop.points, config.crop.crop_size)
        else:
            self.transform = transforms.Resize((config.crop.crop_size, config.crop.crop_size))
            
        if config.pad.active:
            target_width = target_height = config.crop.crop_size // 2
            pad_width = (config.crop.crop_size - target_width) // 2
            pad_height = (config.crop.crop_size - target_height) // 2

            self.pad = transforms.Compose([
                transforms.Resize((target_height, target_width)),
                transforms.Pad((pad_width, pad_height, config.crop.crop_size - target_width - pad_width, config.crop.crop_size - target_height - pad_height), fill=1),
            ])
        else:
            self.pad = nn.Identity()

        if config.foveat.active:
            self.foveat = Foveation(crop_size=config.crop.crop_size)
        else:
            self.foveat = nn.Identity()

        if not config.rotate.active or config.rotate.max_rotate == 0:
            self.rotate = nn.Identity()
        elif config.rotate.invert:
            self.rotate = transforms.RandomRotation((180,180))
        else:
            self.rotate = transforms.RandomRotation(config.rotate.max_rotate)

        if config.flip.active:
            self.flip = transforms.RandomHorizontalFlip(p=config.flip.flip_probability)
        else:
            self.flip = nn.Identity()

        if config.log_polar.active:
            inp_shape = (config.crop.crop_size, config.crop.crop_size)
            if config.pad.active:
                inp_shape = (config.crop.crop_size+2*config.pad.pad_v, config.crop.crop_size+2*config.pad.pad_h)
            self.lp = LogPolar(
                    input_shape = inp_shape,
                    output_shape = config.log_polar.lp_out_shape,
                    smoothing = config.log_polar.smoothing,
                    mask = config.log_polar.mask,
                    random_center = config.log_polar.random,
            )
        else:
            self.lp = nn.Identity()

        if config.color.active:
            self.color = transforms.Compose([ 
                transforms.RandomApply(
                    [
                        transforms.ColorJitter( 
                            0.8 * config.color.jitter_strength, 
                            0.8 * config.color.jitter_strength, 
                            0.8 * config.color.jitter_strength, 
                            0.2 * config.color.jitter_strength
                        )
                    ], 
                    p=0.8
                ), 
                transforms.RandomGrayscale(p=config.color.grayscale_probability)
            ])
        else:
            self.color = nn.Identity()

        self.flatten = flatten

        self.compose = transforms.Compose([
            self.transform,
            # self.pad,
            # self.flip,
            # self.color,
            # self.rotate,
            self.foveat,
            self.lp,
            # self.normalize,
            # self.flatten,
        ])

        self.count = config.count

    def forward(self, image):
        orig_image = self.tensorize(image) 
        
        return [self.compose(orig_image) for _ in range(self.count)]
