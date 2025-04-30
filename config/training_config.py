from pydantic import BaseModel

class TrainingConfig(BaseModel):
    batch_size: int = 48
    epochs: int = 40
    initial_lr: float = 0.0001
    lr_schedule_gamma: float = 0.99
    lr_schedule_frequency: float = 1
    pool_dim: int = 1
