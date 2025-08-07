import cv2
import numpy as np

from data_classes.frame import FrameData
from data_classes.track import Person, Car


class TrackObserver:
    def __init__(self, config, traffic_rois: list[np.ndarray]):
        self.track_buffer = config["detection"]["track_buffer"]

        self.traffic_rois = traffic_rois

        self.people: dict[int, Person] = {}
        self.cars: dict[int, Car] = {}

    def process(self, frame_data: FrameData) -> FrameData:
        updated = set()

        for id_, track_id in enumerate(frame_data.track_id):
            updated.add(id_)

            object_class = frame_data.track_cls[id_]
            if object_class == "person":
                self.update_person(track_id, frame_data.track_xyxy[id_])

            elif object_class == "car":
                self.update_cars(track_id, frame_data.track_xyxy[id_])

        self.delete_objects(updated)

        frame_data.people = self.people
        return frame_data



    def update_person(self, track_id, xyxy):
        person = self.people.setdefault(track_id, Person())
        person.num_disappearances = 0

        point = self.calc_bottom_point(xyxy)
        person.points.append(point)

        in_danger_zone = False
        for roi in self.traffic_rois:
            if self.check_intersection(point, roi):
                in_danger_zone = True

        if in_danger_zone:
            person.num_dangers_frames += 1
        else:
            person.num_dangers_frames = 0

    def update_cars(self, track_id, xyxy):
        car = self.cars.setdefault(track_id, Car())
        car.num_disappearances = 0

        point = self.calc_central_point(xyxy)
        car.points.append(point)


    def delete_objects(self, updated: set[int]):
        for id_, person in list(self.people.items()):
            if id_ not in updated:
                person.num_disappearances += 1

            if person.num_disappearances >= self.track_buffer:
                del self.people[id_]


        for id_, car in list(self.cars.items()):
            if id_ not in updated:
                car.num_disappearances += 1

            if car.num_disappearances >= self.track_buffer:
                del self.cars[id_]


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
