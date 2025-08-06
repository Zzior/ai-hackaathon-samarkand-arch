from pathlib import Path

import hydra

import numpy as np

from models.video_reader import VideoReader
from models.detection_tracking import DetectionTracking

from visualization.show import Show

project_dir = Path(__file__).parent.parent
conf_dir = project_dir / "configs"

@hydra.main(version_base=None, config_path=conf_dir.__str__(), config_name="config")
def main(config) -> None:
    traffic_rois: list[np.ndarray] = []
    for roi in config["source_info"]["traffic_roi"]:
        traffic_rois.append(np.array(roi, dtype=np.int32))

    video_reader = VideoReader(str(project_dir / config["source_info"]["src"]))
    detection_tracking = DetectionTracking(config, project_dir)

    show = Show(config, traffic_rois)

    for frame_data in video_reader.process():
        frame_data = detection_tracking.process(frame_data)

        if config["show"]["show"]:
            show.process(frame_data)

if __name__ == "__main__":
    main()
