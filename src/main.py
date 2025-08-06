from pathlib import Path

import hydra

from models.video_reader import VideoReader

from visualization.show import Show

project_dir = Path(__file__).parent.parent
conf_dir = project_dir / "configs"

@hydra.main(version_base=None, config_path=conf_dir.__str__(), config_name="config")
def main(config) -> None:
    video_reader = VideoReader(str(project_dir / config["source_info"]["src"]))

    show = Show(config)

    for frame_data in video_reader.process():

        if config["show"]["show"]:
            show.process(frame_data)

if __name__ == "__main__":
    main()
