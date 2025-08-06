import cv2

from data_classes.frame import FrameData

from utils.utils import FPSCounter


class Show:
    def __init__(self, conf):
        self.conf = conf

        self.fps_buffer = conf["show"]["fps_buffer"]
        self.show = conf["show"]["show"]

        if self.fps_buffer > 1:
            self.fps_counter = FPSCounter(self.fps_buffer)
        else:
            self.fps_counter = None

    def process(self, frame_data: FrameData) -> FrameData:
        frame_data.frame_out = frame_data.frame

        if self.fps_counter is not None:
            cv2.putText(
                frame_data.frame_out,
                f"FPS {self.fps_counter.get_fps():.1f}",
                (50, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            track_info = zip(frame_data.track_xyxy, frame_data.track_id, frame_data.track_cls, frame_data.track_conf)
            for bbox, id_, cls, conf in track_info:
                cv2.putText(
                    frame_data.frame_out,
                    f"{cls} {round(conf, 2)}",
                    (bbox[0] + 5, bbox[1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.rectangle(frame_data.frame_out, (bbox[0], bbox[1]), (bbox[2], bbox[3]), self.get_color(id_), 1)

        if self.show:
            cv2.imshow('frame', frame_data.frame_out)
            cv2.waitKey(1)

        return frame_data

    @staticmethod
    def get_color(id_: int) -> tuple[int, int, int]:
        colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Purple
            (0, 255, 255),  # Blue
            (255, 20, 147),  # Deep Pink
            (255, 165, 0),  # Orange
            (32, 178, 170),  # Light Sea
            (148, 0, 211)  # Dark Violet
        ]

        return colors[id_ % 10]