from dataclasses import dataclass
from typing import List, Optional

from src.entities.audio_params import AudioParams


@dataclass
class DatasetParams:
    root_path: str
    metadata: str
    years_list: List[int]
    split: Optional[str]
    audio_params: AudioParams
    feature_size: int
    overlapping: int
