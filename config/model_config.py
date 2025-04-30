from pydantic import BaseModel

class ModelConfig(BaseModel):
    name: str
    classes: int = 128
    zdim: int = 2048
    accumulate_grad_batches: int = 1
