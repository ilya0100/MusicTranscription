from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioParams:
    sample_rate: int
    frame_length: int
    frame_step: int
    frame_time: Optional[float]
    pad_end: bool = True
    pad_value: float = 0.0
