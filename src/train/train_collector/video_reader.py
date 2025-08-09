from typing import Generator

import numpy as np
import cv2


class VideoReader:
    def __init__(self, config) -> None:
        self.video_config = config["video_reader"]

        self.source = self.video_config["mov_source"]
        self.skip_frames = self.video_config["skip_frames"]
        self.mov_iterator = self.video_config["mov_iterator"]

        if self.mov_iterator:
            self.queue = [self.source.format(i=i) for i in range(*self.mov_iterator)]
        else:
            self.queue = []

        self.is_stream = isinstance(self.source, int) or "://" in self.source
        self.capture = None

    def _connect(self) -> bool:
        if isinstance(self.capture, cv2.VideoCapture):
            self.capture.release()

        self.capture = cv2.VideoCapture(self.queue.pop(0) if self.queue else self.source, cv2.CAP_FFMPEG)

        if self.capture.isOpened():
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 5)
            return True

        return False

    def process(self) -> Generator[np.ndarray, None, None]:
        self._connect()
        try:
            frame_id = 0
            while True:
                ret, frame = self.capture.read()

                if not ret:
                    if self.is_stream or self.queue:
                        self._connect()
                        frame_id = 0
                        continue
                    else:
                        break

                if frame is None:
                    continue

                frame_id += 1
                if frame_id % self.skip_frames != 0:
                    continue

                yield frame

        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt")
            self.capture.release()
