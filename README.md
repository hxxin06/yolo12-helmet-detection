# yolo12-helmet-detection
Helmet Detection System based on YOLO12n for rider, helmet, no_helmet and number_plate detection. Trained on Roboflow dataset and evaluated using Precision, Recall, mAP50 and mAP50-95 metrics.

## Dataset

The dataset used in this project is from Roboflow Universe:

**Helmet and Number Plate Detection for Motorbike Safety**

Dataset format: YOLO

The dataset contains four classes:

| Class ID | Class Name | Description |
|---|---|---|
| 0 | helmet | Rider wearing helmet |
| 1 | no_helmet | Rider without helmet |
| 2 | number_plate | Vehicle number plate |
| 3 | rider | Motorcycle or bicycle rider |

Dataset split:

| Split | Number of Images |
|---|---|
| train | 11,852 |
| valid | 1,693 |
| test | 847 |

## Method

This project uses YOLO12n as the main object detection model.

The workflow includes:

1. Set up the Python and YOLO12 environment
2. Prepare the YOLO-format dataset
3. Train YOLO12n with pretrained weights
4. Evaluate the trained model using test images
5. Perform video detection using an external MP4 video
6. Analyze the results using Precision, Recall, mAP50, mAP50-95, and F1-score

## Training Settings

The formal training settings are shown below:

| Setting | Value |
|---|---|
| Model | YOLO12n |
| Epochs | 60 |
| Image Size | 960 × 960 |
| Batch Size | 4 |
| Best Model | best.pt |

## Demo Video

The demo video result is provided in this repository:

helmet_video1_result.mp4

