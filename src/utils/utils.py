import time

import math

class FPSCounter:
    def __init__(self, buffer_size: int = 15):
        self.time_buffer = [time.time()] * buffer_size
        self.buffer_size = buffer_size

    def get_fps(self) -> float:
        self.time_buffer.append(time.time())
        return (self.buffer_size - 1) / (self.time_buffer[-1] - self.time_buffer.pop(0))


def get_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def angle_between(v1: tuple[float, float], v2: tuple[float, float]) -> float:
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)
    if mag1 == 0 or mag2 == 0:
        return 0
    cos_theta = max(-1, min(1, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_theta))

def detect_motion_anomalies(
    points: list[tuple[int, int]],
    speed_thresh: float = 20.0,         # the minimum speed that we consider to be "abnormally high"
    accel_thresh: float = 15.0,         # the minimum change in velocity to be considered acceleration
    angle_thresh: float = 60.0          # threshold for the angle of rotation (in degrees)
) -> dict[str, list[int]]:
    anomalies: dict[str, list[int]] = {}

    for i in range(2, len(points)):
        p0, p1, p2 = points[i-2], points[i-1], points[i]

        # Speed is the distance between points
        speed1 = get_distance(p0, p1)
        speed2 = get_distance(p1, p2)

        # High Speed
        if speed2 > speed_thresh:
            if "high_speed" in anomalies:
                anomalies["high_speed"].append(i)
            else:
                anomalies["high_speed"] = [i]

        # Acceleration spike
        if abs(speed2 - speed1) > accel_thresh:
            if "acceleration_spike" in anomalies:
                anomalies["acceleration_spike"].append(i)
            else:
                anomalies["acceleration_spike"] = [i]

        # Sharp change of direction (Sharp turn)
        v1 = (p1[0] - p0[0], p1[1] - p0[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])
        angle = angle_between(v1, v2)

        if angle > angle_thresh:
            if "sharp_turn" in anomalies:
                anomalies["sharp_turn"].append(i)
            else:
                anomalies["sharp_turn"] = [i]

    return anomalies