from pathlib import Path
from datetime import datetime

import cv2

from data_classes.frame import FrameData


class VideoWriter:
    def __init__(self, config, project_dir: Path):
        self.config = config["video_writer"]

        self.fps = self.config["fps"]
        self.fourcc = self.config["fourcc"]
        self.skip_frames = self.config["skip_frames"]
        self.resolution = config["show"]["output_size"]
        self.segment_size = self.config["segment_size"]  # секунды

        output_path = Path(self.config["output_path"]).expanduser()
        self.output_path = output_path if output_path.is_absolute() else project_dir / output_path
        Path.mkdir(self.output_path, parents=True, exist_ok=True)

        self.frames_in_segment = 0
        self.total_frames_processed = 0

        self.segment_frames = int(self.fps * self.segment_size)
        self.writer = self._create_new_writer()

    def _create_new_writer(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_path / f"{timestamp}.mkv"

        fourcc = cv2.VideoWriter_fourcc(*self.fourcc)  # noqa
        writer = cv2.VideoWriter(str(filename), fourcc, self.fps, self.resolution)
        return writer

    def process(self, frame_data: FrameData) -> None:
        if frame_data.frame_out is None:
            return

        self.total_frames_processed += 1
        if (self.total_frames_processed % self.skip_frames) != 0:
            return

        self.writer.write(frame_data.frame_out)
        self.frames_in_segment += 1

        if self.frames_in_segment >= self.segment_frames:
            self._close_current_writer()
            self.frames_in_segment = 0
            self.writer = self._create_new_writer()

    def _close_current_writer(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __del__(self):
        self._close_current_writer()
