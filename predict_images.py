from ultralytics import YOLO

MODEL_PATH = "runs/detect/runs/helmet_detection/yolo12n_helmet_60epochs_960/weights/best.pt"
SOURCE_PATH = "Helmet-and-Number-Plate-Detection-for-Motorbike-Safety-3/test/images"

model = YOLO(MODEL_PATH)

model.predict(
    source=SOURCE_PATH,
    conf=0.5,
    save=True,
    project="runs/helmet_predict",
    name="test_images_960"
)
