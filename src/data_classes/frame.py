from dataclasses import dataclass, field

import numpy as np

from data_classes.track import Person


@dataclass
class FrameData:
    frame_id: int
    timestamp: float
    frame: np.ndarray
    frame_out: np.ndarray = None

    track_xyxy: list[list[int]] = field(default_factory=list)
    track_id: list[int] = field(default_factory=list)
    track_conf: list[float] = field(default_factory=list)
    track_cls: list[str] = field(default_factory=list)

    persons: list[Person] = field(default_factory=list)