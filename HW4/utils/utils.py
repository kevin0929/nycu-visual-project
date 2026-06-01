import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute PSNR between two batches of images (values in [0, 1]).

    Args:
        pred:   (B, C, H, W) float tensor, values in [0, 1]
        target: (B, C, H, W) float tensor, values in [0, 1]

    Returns:
        Average PSNR (dB) over the batch.
    """
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))
    psnr = -10.0 * torch.log10(mse + 1e-8)
    return psnr.mean().item()


class CosineAnnealingWarmupLR(_LRScheduler):
    """
    Cosine annealing with linear warmup.

    During warmup_epochs, LR rises linearly from 0 to base_lr.
    Afterwards, cosine annealing from base_lr down to min_lr.
    """

    def __init__(
        self,
        optimizer,
        total_epochs: int,
        warmup_epochs: int = 5,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        lrs = []
        for base_lr in self.base_lrs:
            if epoch < self.warmup_epochs:
                lr = base_lr * (epoch + 1) / max(self.warmup_epochs, 1)
            else:
                progress = (epoch - self.warmup_epochs) / max(
                    self.total_epochs - self.warmup_epochs, 1
                )
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (
                    1.0 + math.cos(math.pi * progress)
                )
            lrs.append(lr)
        return lrs


class SSIMLoss(nn.Module):
    """
    Structural Similarity (SSIM) loss: 1 - SSIM.
    Higher SSIM = more similar, so we minimise 1 - SSIM.
    """

    def __init__(self, window_size: int = 11, channels: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.register_buffer("window", self._gaussian_window(window_size, channels))

    @staticmethod
    def _gaussian_window(size: int, channels: int) -> torch.Tensor:
        import math
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
        g /= g.sum()
        window = g.unsqueeze(1) * g.unsqueeze(0)          # (size, size)
        window = window.unsqueeze(0).unsqueeze(0)          # (1, 1, size, size)
        return window.expand(channels, 1, size, size).contiguous()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        w = self.window.to(pred.device, pred.dtype)
        pad = self.window_size // 2

        mu1 = F.conv2d(pred, w, padding=pad, groups=self.channels)
        mu2 = F.conv2d(target, w, padding=pad, groups=self.channels)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, w, padding=pad, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(target * target, w, padding=pad, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(pred * target, w, padding=pad, groups=self.channels) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()


def tensor_to_uint8(t: torch.Tensor) -> "np.ndarray":
    """Convert a (C, H, W) float tensor in [0,1] to uint8 numpy (C, H, W)."""
    import numpy as np
    arr = t.clamp(0, 1).cpu().numpy()
    return (arr * 255).round().astype(np.uint8)