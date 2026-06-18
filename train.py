from ultralytics import YOLO

DATA_YAML = "helmet_data.yaml"

model = YOLO("yolo12n.pt")

model.train(
    data=DATA_YAML,
    epochs=60,
    imgsz=960,
    batch=4,
    device=0,
    patience=15,
    workers=4,
    project="runs/helmet_detection",
    name="yolo12n_helmet_60epochs_960"
)
