from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("../../weights/yolo11n.pt")

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="../../media/dataset/data.yaml", epochs=1, imgsz=640)  # , split="val", save=True)
