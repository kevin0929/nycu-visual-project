# NYCU Computer Vision 2026 HW4
- Student ID: 313551003
- Name: 馬停偉

## Introduction
This homework focuses on image restoration, where the goal is to recover clean images from degraded inputs. The dataset contains two types of degradation: rain streaks and snow particles. A single model must handle both types without being told which degradation is present at test time.
We use PromptIR [1] as our model. PromptIR uses learnable prompt vectors to capture degradation-specific information, allowing one model to handle multiple degradation types. Performance is measured by PSNR (Peak Signal-to-Noise Ratio), where higher is better.

## Enviroment Setup

```
pip install -r requirements.txt
```

## Usage

1. Training and auto testing
```
python train.py --epochs 200 --batch_size 4 --patch_size 128 --accum_steps 2
```

2. Inference
```
python inference.py --ckpt ckpt/best_model.pth --test_dir data/test --output_dir output
```

## Performance Snapshot
![image](./assert/image.png)
