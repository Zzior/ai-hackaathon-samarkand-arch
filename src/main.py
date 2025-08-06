from pathlib import Path

import hydra

import cv2

project_dir = Path(__file__).parent.parent
conf_dir = project_dir / "configs"

@hydra.main(version_base=None, config_path=conf_dir.__str__(), config_name="config")
def main(config) -> None:
    src = project_dir / config["source_info"]["src"]
    cap = cv2.VideoCapture(str(src))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("frame", frame)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
