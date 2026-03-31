# NYCU Computer Vision 2026 HW1
- Student ID: 313551003
- Name: 馬停偉

## Introduction
This report describes my approach for a 100-class image classification task. The dataset has about 21K training/validation images and 2.3K unlabeled test images. I fine-tuned an ImageNet-pretrained ResNet-101 with a custom two-layer MLP classification head, using a two-stage training strategy: first training only the head with the backbone frozen, then unfreezing everything for full fine-tuning with different learning rates. I used several techniques to improve accuracy, including MixUp/CutMix augmentation, Label Smoothing, Exponential Moving Average (EMA), a Warmup + Cosine Annealing learning rate schedule, and 10-view Test Time Augmentation (TTA). The final model achieves about 95% accuracy on the test set with around 44M parameters, which is under the 100M limit.

## Enviroment Setup

```
pip install -r requirements.txt
```

## Usage

1. Training and auto testing
```
python train.py --data_dir ./data --model resnet101 --epochs 80 \
    --img_size 288 --batch_size 96 --tta
```

## Performance Snapshot
![image](https://hackmd.io/_uploads/r1SJ3rKsZg.png)
