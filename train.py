from ultralytics import YOLO

DATA_YAML = "helmet_data.yaml"

model = YOLO("yolo12n.pt")

model.train(
    data=DATA_YAML,
    epochs=3,
    imgsz=640,
    batch=8,
    device=0,
    project="runs/helmet_detection",
    name="yolo12n_test"
)
