from pydantic import BaseModel

from .compute_config import ComputeConfig
from .training_config import TrainingConfig
from .transformation_config import TransformationConfig
from .data_config import DataConfig
from .model_config import ModelConfig

from typing import Union

class Config(BaseModel):
    training: TrainingConfig
    transformations: TransformationConfig
    data: DataConfig
    model: ModelConfig
    compute: ComputeConfig
    name: str = ''
