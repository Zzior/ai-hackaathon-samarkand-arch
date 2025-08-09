from typing import Sequence

import numpy as np
import cv2


def get_label(cls: Sequence[int], detect_xyxy: Sequence[int], mov_size: Sequence[int]) -> str:
    x_min, y_min, x_max, y_max = detect_xyxy
    x_center = (x_min + x_max) / 2 / mov_size[0]
    y_center = (y_min + y_max) / 2 / mov_size[1]
    width = (x_max - x_min) / mov_size[0]
    height = (y_max - y_min) / mov_size[1]

    return f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"


def calc_c_point(bbox: list[int]) -> tuple[int, int]:
    return (
        (bbox[0] + bbox[2]) // 2,
        (bbox[1] + bbox[3]) // 2,
    )


def check_intersection(obj_c_point: tuple[int, int], roi: tuple[int, int, int, int]) -> bool:
    if roi[0] < obj_c_point[0] < roi[2] and roi[1] < obj_c_point[1] < roi[3]:
        return True
    return False


def draw(frame: np.ndarray, cls: int, detect_xyxy: Sequence[int], detected_cof: float) -> np.ndarray:
    frame = cv2.putText(
        frame, f"{cls} | {round(detected_cof, 2)}",
        (detect_xyxy[0] + 5, detect_xyxy[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
        (0, 255, 255), 2,
    )
    frame = cv2.rectangle(frame, (detect_xyxy[0], detect_xyxy[1]), (detect_xyxy[2], detect_xyxy[3]), (255, 0, 0), 2)

    return frame
