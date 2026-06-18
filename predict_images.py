from ultralytics import YOLO

MODEL_PATH = "weights/best.pt"
SOURCE_PATH = "test/images"

model = YOLO(MODEL_PATH)

model.predict(
    source=SOURCE_PATH,
    conf=0.5,
    save=True,
    project="runs/helmet_predict",
    name="test_images_960"
)
