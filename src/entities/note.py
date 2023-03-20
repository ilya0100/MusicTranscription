from dataclasses import dataclass


@dataclass
class Note:
    pitch: int
    velocity: int
