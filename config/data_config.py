from pydantic import BaseModel
from typing import List
from pathlib import Path

class DataConfig(BaseModel):
    name: str
    dataset: Path
    output: Path = Path(".")
    classes: List[int] = [4, 8, 16, 32, 64, 128]
    find_classes_data_path: int = 128
    # repetitions = 1

