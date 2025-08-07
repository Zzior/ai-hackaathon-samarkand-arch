import cv2
import numpy as np

from data_classes.frame import FrameData
from data_classes.track import Person


class TrackObserver:
    def __init__(self, config, traffic_rois: list[np.ndarray]):
        self.track_buffer = config["detection"]["track_buffer"]

        self.traffic_rois = traffic_rois

        self.people: dict[int, Person] = {}

    def process(self, frame_data: FrameData) -> FrameData:
        for id_, track_id in enumerate(frame_data.track_id):
            object_class = frame_data.track_cls[id_]
            if object_class == "person":
                self.update_person(track_id, frame_data.track_xyxy[id_])

        frame_data.people = self.people
        return frame_data



    def update_person(self, track_id, xyxy):
        person = self.people.setdefault(track_id, Person())
        person.num_disappearances = 0

        point = self.calc_bottom_point(xyxy)
        person.points.append(point)

        in_danger_zone = False
        for roi in self.traffic_rois:
            in_danger_zone =  self.check_intersection(point, roi)

        if in_danger_zone:
            person.num_dangers_frames += 1
        else:
            person.num_dangers_frames = 0


    @staticmethod
    def calc_central_point(bbox: list[int]) -> tuple[int, int]:
        return (
            (bbox[0] + bbox[2]) // 2,
            (bbox[1] + bbox[3]) // 2,
        )

    @staticmethod
    def calc_bottom_point(bbox: list[int]) -> tuple[int, int]:
        return (
            (bbox[0] + bbox[2]) // 2,
            bbox[3],
        )

    @staticmethod
    def check_intersection(obj_c_point: tuple[int, int], roi: np.ndarray) -> bool:
        intersection = cv2.pointPolygonTest(roi, obj_c_point, False)

        if intersection >= 0:
            return True

        else:
            return False
