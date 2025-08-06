from pathlib import Path

import torch
from ultralytics import YOLO

from data_classes.frame import FrameData


class DetectionTracking:
    def __init__(self, config, project_dir: Path) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Detection will be performed on {device}')

        config_yolo = config["detection"]
        self.model = YOLO(project_dir / config_yolo["weight_pth"], task='detect')
        self.classes_to_detect = config_yolo["classes_to_detect"]
        self.tracker = project_dir / config_yolo["tracker_pth"]
        self.classes = self.model.names
        self.conf = config_yolo["confidence"]
        self.iou = config_yolo["iou"]
        self.imgsz = config_yolo["imgsz"]

    def process(self, frame_data: FrameData) -> FrameData:
        assert isinstance(
            frame_data, FrameData
        ), f"DetectionTracking | Incorrect input element format {type(frame_data)}"

        frame = frame_data.frame.copy()

        outputs = self.model.track(
            frame, imgsz=self.imgsz, conf=self.conf, verbose=False, iou=self.iou, classes=self.classes_to_detect,
            tracker=self.tracker, persist=True
        )

        if outputs[0].boxes.id is not None:
            frame_data.track_id = outputs[0].boxes.id.cpu().numpy().astype(int).tolist()
            frame_data.track_xyxy = outputs[0].boxes.xyxy.cpu().int().tolist()
            frame_data.track_conf = outputs[0].boxes.conf.cpu().tolist()
            track_cls = outputs[0].boxes.cls.cpu().int().tolist()
            frame_data.track_cls = [self.classes[i] for i in track_cls]

        return frame_data

