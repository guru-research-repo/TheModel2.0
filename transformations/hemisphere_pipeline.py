import torch
import torch.nn as nn
from torchvision import transforms
from transformations import (
        HemisphereCrop,
        Pipeline
)
from config.transformation_config import *

import random

from typing import Tuple, Union, Callable, Optional, Any

import time
# from utils import Timer

class HemispherePipeline(Pipeline):
    def __init__(self, 
            config,
            normalize = transforms.Normalize( 
                mean = (0.485, 0.456, 0.406), 
                std = (0.229, 0.224, 0.225) 
            ), 
            flatten = torch.nn.Identity()):
        
        super().__init__(config)

        if config.log_polar.active:
            self.left_crop = HemisphereCrop(config.crop.crop_size, start_y = 0.25, end_y = 0.75)
            self.right_crop = HemisphereCrop(config.crop.crop_size, start_y = 0.75, end_y = 0.25)
        else:
            self.left_crop = HemisphereCrop(config.crop.crop_size, start_x = 0, end_x = 0.5)
            self.right_crop = HemisphereCrop(config.crop.crop_size, start_x = 0.5, end_x = 1.00)
        
        self.count = config.count

    def forward(self, image):
        orig_image = self.compose(self.tensorize(image))
        
        return [
                self.left_crop(orig_image),
                self.right_crop(orig_image),
                random.choice([self.left_crop, self.right_crop])(orig_image),
        ]
