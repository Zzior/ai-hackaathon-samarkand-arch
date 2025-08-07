from dataclasses import dataclass, field

import numpy as np

@dataclass
class Person:
    points: list[tuple[int, int]] = field(default_factory=list)

    num_disappearances: int = 0
    num_dangers_frames: int = 0