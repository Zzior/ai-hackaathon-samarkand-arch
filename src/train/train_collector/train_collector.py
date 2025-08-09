from datetime import datetime
from pathlib import Path

import cv2
import torch  # noqa
import hydra
from ultralytics import YOLO

from train.train_collector.video_reader import VideoReader
from train.train_collector.utils import get_label, draw

project_dir = Path(__file__).parent.parent.parent.parent
conf_dir = project_dir / "configs"

labels_path = project_dir / "media/tc_dataset/labels/Train"
img_path = project_dir / "media/tc_dataset/images/Train"
img_val_path = project_dir / "media/tc_dataset/images/Validation"
labels_val_path = project_dir / "media/tc_dataset/labels/Validation"
for pth in (labels_path, img_path, img_val_path, labels_val_path):
    pth.mkdir(parents=True, exist_ok=True)


@hydra.main(version_base=None, config_path=conf_dir.__str__(), config_name="config_train")
def main(config) -> None:
    show = config["show"]

    # Detect configs
    iou = config["detection"]["iou"]
    imgsz = config["detection"]["imgsz"]
    classes_param: dict[int, list[float]] = dict(config["detection"]["classes_param"])

    video_reader = VideoReader(config)
    model = YOLO(config["detection"]["weights_path"])

    for frame in video_reader.process():
        show_frame = frame.copy()

        now_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        outputs = model.predict(frame, imgsz=imgsz, verbose=False, iou=iou, classes=list(classes_param.keys()))
        if (outputs[0].boxes.cls is not None) and outputs[0].boxes.cls.numel() > 0:
            detected_cof = outputs[0].boxes.conf.cpu().tolist()
            detected_cls = outputs[0].boxes.cls.cpu().int().tolist()
            detected_xyxy = outputs[0].boxes.xyxy.cpu().int().tolist()

            yolo_label = ""
            save = False
            for obj_cls, obj_xyxy, obj_conf in zip(detected_cls, detected_xyxy, detected_cof):
                if classes_param[obj_cls][0] < obj_conf < classes_param[obj_cls][1]:
                    save = True


                yolo_label += get_label(obj_cls, obj_xyxy, (frame.shape[1], frame.shape[0]))

                if show and save:
                    show_frame = draw(show_frame, obj_cls, obj_xyxy, obj_conf)

            if yolo_label and save:
                cv2.imwrite(f"{img_path.resolve().__str__()}/{now_str}.png", frame)
                with open(f"{labels_path.resolve().__str__()}/{now_str}.txt", "w") as f:
                    f.write(yolo_label)

        if show:
            cv2.imshow("frame", show_frame)
            cv2.waitKey(1)


if __name__ == "__main__":
    main()
