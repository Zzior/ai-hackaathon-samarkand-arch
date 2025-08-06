import time
import logging
from pathlib import Path
from typing import Generator

import cv2

from data_classes.frame import FrameData

logger = logging.getLogger(__name__)


class VideoReader:
    def __init__(self, source: str | int) -> None:
        assert (
            isinstance(source, int) or Path(source).is_file() or "://" in source
        ), f"VideoReader| file or source {source} not found."

        self.source = source

        self.capture = None
        self.is_stream = isinstance(self.source, int) or "://" in self.source


    def _connect_mov(self) -> bool:
        if isinstance(self.capture, cv2.VideoCapture):
            self.capture.release()

        self.capture = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)

        if self.capture.isOpened():
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 5)
            return True

        return False

    def process(self) -> Generator[FrameData, None, None]:
        self._connect_mov()
        frame_id = 1

        try:
            while True:
                ret, frame = self.capture.read()

                if not ret:
                    if self.is_stream:
                        if frame_id != 0:
                            logger.warning(f"Can't receive frame.")
                        self._connect_mov()
                        frame_id = 0
                        continue
                    else:
                        logger.info("End of video file reached.")
                        break

                if frame is None:
                    continue

                frame_id += 1
                timestamp = time.time()


                yield FrameData(frame_id, timestamp, frame)

        except KeyboardInterrupt:
            logger.info("Caught KeyboardInterrupt")
            self.capture.release()
