# yolo12-helmet-detection
Helmet Detection System based on YOLO12n for rider, helmet, no_helmet and number_plate detection. Trained on Roboflow dataset and evaluated using Precision, Recall, mAP50 and mAP50-95 metrics.

1. Install

pip install -r requirements.txt

2. Train

python train.py

3. Image Detection

python predict_image.py

4. Video Detection
Used an external MP4 video for testing, and applied the trained YOLO12n model to continuous images.

