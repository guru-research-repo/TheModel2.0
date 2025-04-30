from pydantic import BaseModel
from typing import Tuple, Union
from enum import Enum

class LogPolarConfig(BaseModel):
    active: bool = True
    smoothing: int = 0
    lp_out_shape: Tuple[int, int] = (190, 165)
    mask: bool = True
    random: bool = False

class CropConfig(BaseModel):
    type: str = 'random'
    points: int = 4
    crop_size: int = 180

class PadConfig(BaseModel):
    active: bool = True
    pad_h: int = 40
    pad_v: int = 10
        
class FoveationConfig(BaseModel):
    active: bool = True

class RotationConfig(BaseModel):
    active: bool = True
    invert: bool = False
    max_rotate: int = 15

class ColorConfig(BaseModel):
    active: bool = False
    jitter_strength: float = 1.0
    grayscale_probability: float = 0.2

class FlipConfig(BaseModel):
    active: bool = False
    flip_probability: float = 0.5

class TransformationConfig(BaseModel):
    log_polar: LogPolarConfig
    crop: CropConfig
    foveat: FoveationConfig
    rotate: RotationConfig
    color: ColorConfig
    flip: FlipConfig
    pad: PadConfig
    count: int = 1
    type: str = 'custom'
