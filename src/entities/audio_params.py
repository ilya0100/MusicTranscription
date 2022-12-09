from dataclasses import dataclass


@dataclass
class AudioParams:
    sample_rate: int
    frame_length: int
    frame_step: int
    frame_time: float
    n_mels: int
    fmin: float
    fmax: float
    window: str
