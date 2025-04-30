from torchvision import transforms
from torch import nn
from transformations import (
    HemispherePipeline,
    Pipeline, 
    LogPolar,
    SimCLREvalDataTransform, 
    SimCLRTrainDataTransform,
)

from config.transformation_config import *

from typing import *

def get_pipeline(config: TransformationConfig) -> Tuple[
    Union[Pipeline, SimCLRTrainDataTransform],
    Union[Pipeline, SimCLREvalDataTransform],
    Optional[Callable[[Any], Any]]
]:
    if config.type == 'hemisphere':
        return (
            HemispherePipeline(
                config,
                normalize = transforms.Normalize( 
                    mean = (0.485, 0.456, 0.406), 
                    std = (0.229, 0.224, 0.225) 
                )
            ), 

            HemispherePipeline(
                config,
                normalize = transforms.Normalize( 
                    mean = (0.485, 0.456, 0.406), 
                    std = (0.229, 0.224, 0.225) 
                )
            ),
            None,
        )
    
    elif config.type == 'custom':
        return (
            Pipeline(
                config,
                normalize = transforms.Normalize(
                    mean = (0.485, 0.456, 0.406),
                    std = (0.229, 0.224, 0.225)
                )
            ), 

            Pipeline(
                config,
                normalize = transforms.Normalize(
                    mean = (0.485, 0.456, 0.406),
                    std = (0.229, 0.224, 0.225)
                )
            ),
            None,
        )
    
    elif config.type == 'custom_supervised':
        return (
            Pipeline(
                config,
                normalize = transforms.Normalize(
                    mean = (0.485, 0.456, 0.406),
                    std = (0.229, 0.224, 0.225)
                )
            ),

            Pipeline(
                config,
                normalize = transforms.Normalize(
                    mean = (0.485, 0.456, 0.406),
                    std = (0.229, 0.224, 0.225)
                )
            ),
            
            Pipeline(
                config,
                normalize = transforms.Normalize(
                    mean = (0.485, 0.456, 0.406),
                    std = (0.229, 0.224, 0.225)
                )
            ),
        )
    
    elif config.type == 'simclr':
        normalize = transforms.Compose([
            (
                LogPolar(
                    input_shape = (
                        config.crop.crop_size, 
                        config.crop.crop_size
                    ),
                    output_shape = config.log_polar.lp_out_shape,
                    smoothing = config.log_polar.smoothing
                ) if config.log_polar.active else
                nn.Identity()
            ),
            transforms.Normalize( 
                mean = (0.485, 0.456, 0.406), 
                std = (0.229, 0.224, 0.225) 
            ),
        ])

        return (
            SimCLRTrainDataTransform(
                input_height = config.crop.crop_size,
                jitter_strength = config.color.jitter_strength,
                normalize = normalize,
                count=config.count,
            ),
            SimCLREvalDataTransform(
                input_height = config.crop.crop_size,
                jitter_strength = config.color.jitter_strength,
                normalize = normalize,
                count=config.count,
            ),
            None
        )

