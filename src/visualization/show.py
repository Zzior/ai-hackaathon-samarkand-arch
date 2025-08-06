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

        if self.show:
            cv2.imshow('frame', frame_data.frame_out)
            cv2.waitKey(1)

        return frame_data