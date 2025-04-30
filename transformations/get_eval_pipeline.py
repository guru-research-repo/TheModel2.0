from torchvision import transforms
from torch import nn
import torch
from transformations import (
    Pipeline, 
    LogPolar,
)

from config.transformation_config import *

from typing import *

class EvalPipeline(nn.Module):
    def __init__(self, 
            SCALE_FACTOR = 1.0,
            MIN_ROTATE = 0,
            MAX_ROTATE = 0,
            LP_ON = False,
            LP_SCALING = 'log',
            normalize = transforms.Normalize( 
                mean = (0.485, 0.456, 0.406), 
                std = (0.229, 0.224, 0.225) 
            )
    ):
        
        super().__init__()

        self.tensorize = transforms.ToTensor()

        self.normalize = normalize

        WIDTH = int(224 * SCALE_FACTOR)
        self.resize = transforms.Resize((WIDTH, WIDTH))
        self.pad = transforms.Pad((224 - WIDTH) // 2)
        self.rotate = transforms.RandomRotation((MIN_ROTATE, MAX_ROTATE))

        if LP_ON:
            if LP_SCALING == 'log':
                self.lp = LogPolar(
                        input_shape = (
                            224,
                            224
                        ),
                        output_shape = (192,164),
                        smoothing = 0,
                        mask = True,
                )
            else:
                self.lp = nn.Identity()
        else:
            self.lp = nn.Identity()

        self.compose = transforms.Compose([
            self.tensorize,
            self.resize,
            self.pad,
            self.rotate,
            self.lp,
            self.normalize,
        ])

    def forward(self, image):
        return self.compose(image)

# 5 free rotation angles X 3 scale factors
def get_eval_pipeline_1(config):
    l = []

    l.append(
        EvalPipeline(
            SCALE_FACTOR = 1.0,
            MIN_ROTATE = 0,
            MAX_ROTATE = 0,
            LP_ON = config.log_polar.active,
            LP_SCALING = 'log',
        )
    )

    for SCALE_FACTOR in torch.linspace(0.5,1,3):
        for MAX_ROTATE in torch.arange(0,181,45):
            l.append(
                EvalPipeline(
                    SCALE_FACTOR = SCALE_FACTOR,
                    MIN_ROTATE = -MAX_ROTATE,
                    MAX_ROTATE = MAX_ROTATE,
                    LP_ON = config.log_polar.active,
                    LP_SCALING = 'log',
                )
            )
    return l


# (upright, inverted) X (scaled, original size)
def get_eval_pipeline_2(config):
    l = []

    l.append(
        EvalPipeline(
            SCALE_FACTOR = 1.0,
            MIN_ROTATE = 0,
            MAX_ROTATE = 0,
            LP_ON = config.log_polar.active,
            LP_SCALING = 'log',
        )
    )
    l.append(
        EvalPipeline(
            SCALE_FACTOR = 0.5,
            MIN_ROTATE = 0,
            MAX_ROTATE = 0,
            LP_ON = config.log_polar.active,
            LP_SCALING = 'log',
        )
    )
    l.append(
        EvalPipeline(
            SCALE_FACTOR = 1.0,
            MIN_ROTATE = 180,
            MAX_ROTATE = 180,
            LP_ON = config.log_polar.active,
            LP_SCALING = 'log',
        )
    )
    l.append(
        EvalPipeline(
            SCALE_FACTOR = 0.5,
            MIN_ROTATE = 180,
            MAX_ROTATE = 180,
            LP_ON = config.log_polar.active,
            LP_SCALING = 'log',
        )
    )
    return l

# 5 fixed rotations X 5 scale factors
def get_eval_pipeline_3(config):
    l = []

    l.append(
        EvalPipeline(
            SCALE_FACTOR = 1.0,
            MIN_ROTATE = 0,
            MAX_ROTATE = 0,
            LP_ON = config.log_polar.active,
            LP_SCALING = 'log',
        )
    )

    for SCALE_FACTOR in torch.linspace(0.5,1,5):
        for MAX_ROTATE in torch.arange(0,181,45):
            l.append(
                EvalPipeline(
                    SCALE_FACTOR = SCALE_FACTOR,
                    MIN_ROTATE = MAX_ROTATE,
                    MAX_ROTATE = MAX_ROTATE,
                    LP_ON = config.log_polar.active,
                    LP_SCALING = 'log',
                )
            )
    return l

# upright and inverted - original size
def get_eval_pipeline_4(config):
    l = []

    l.append(
        EvalPipeline(
            SCALE_FACTOR = 1.0,
            MIN_ROTATE = 0,
            MAX_ROTATE = 0,
            LP_ON = config.log_polar.active,
            LP_SCALING = 'log',
        )
    )
    l.append(
        EvalPipeline(
            SCALE_FACTOR = 1.0,
            MIN_ROTATE = 180,
            MAX_ROTATE = 180,
            LP_ON = config.log_polar.active,
            LP_SCALING = 'log',
        )
    )

    return l

get_eval_pipeline = get_eval_pipeline_4
