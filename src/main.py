from pathlib import Path

import hydra

from models.video_reader import VideoReader
from models.detection_tracking import DetectionTracking

from visualization.show import Show

project_dir = Path(__file__).parent.parent
conf_dir = project_dir / "configs"

@hydra.main(version_base=None, config_path=conf_dir.__str__(), config_name="config")
def main(config) -> None:
    video_reader = VideoReader(str(project_dir / config["source_info"]["src"]))
    detection_tracking = DetectionTracking(config, project_dir)

    show = Show(config)

    for frame_data in video_reader.process():
        frame_data = detection_tracking.process(frame_data)

        if config["show"]["show"]:
            show.process(frame_data)

if __name__ == "__main__":
    main()
