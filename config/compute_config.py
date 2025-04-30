from pydantic import BaseModel

class ComputeConfig(BaseModel):
    num_workers: int = 0
    logging_interval: int = 5
    overfit_batches: int = 0
    num_nodes: int = 1
