from dataclasses import dataclass


@dataclass
class AudioParams:
    sample_rate: int
    frame_length: int
    n_mels: int
    fmin: float
    fmax: float
    window: str
