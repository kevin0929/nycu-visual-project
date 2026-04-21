# NYCU Computer Vision 2026 HW2
- Student ID: 313551003
- Name: 馬停偉

## Introduction
This homework tackles the Digit Detection task: given an RGB street-number image, predict a bounding box and class label (0–9) for each digit in the image. The dataset contains 30,062 training images, 3,340 validation images, and 13,068 test images, with annotations in COCO format.
The required model for this task is DETR (End-to-End Object Detection with Transformers) with a ResNet-50 backbone. Unlike traditional detection models that rely on anchors and NMS during training, DETR formulates detection as a direct set-prediction problem, where predictions are matched to ground-truth boxes using the Hungarian algorithm. This makes the training pipeline much cleaner, but also means the model takes longer to converge compared to anchor-based methods.

## Enviroment Setup

```
pip install -r requirements.txt
```

## Usage

1. Training and auto testing
```
python train.py \
    --data_root data \
    --epochs 300 \
    --batch_size 128 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --lr_drop 200 \
    --dim_feedforward 2048 \
    --use_amp \
    --output_dir checkpoints \
```

2. Inference
```
python inference.py \
    --data_root data \
    --checkpoint checkpoints_official/best.pth \
    --output pred.json \
    --conf_thresh 0.05
```

## Performance Snapshot
![image](./assert/image.png)
