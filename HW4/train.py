import os
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler, autocast

from dataset import RestoreTrainDataset
from net.promptir import PromptIR
from utils.utils import compute_psnr, CosineAnnealingWarmupLR, SSIMLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train PromptIR for HW4")
    parser.add_argument("--data_dir", type=str, default="data/train")
    parser.add_argument("--ckpt_dir", type=str, default="ckpt")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--prompt_len", type=int, default=5)
    parser.add_argument("--accum_steps", type=int, default=2)
    # Loss weights
    parser.add_argument("--w_l1", type=float, default=1.0)
    parser.add_argument("--w_ssim", type=float, default=0.2)
    parser.add_argument("--w_freq", type=float, default=0.1)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freq_loss(pred, target):
    """L1 loss in FFT frequency domain — encourages high-frequency detail."""
    pred_fft = torch.fft.rfft2(pred)
    target_fft = torch.fft.rfft2(target)
    return (torch.abs(pred_fft - target_fft)).mean()


def save_checkpoint(state, path):
    torch.save(state, path)
    print(f"  [Saved] {path}")


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Dataset ----
    full_dataset = RestoreTrainDataset(args.data_dir, patch_size=args.patch_size)
    n_val = max(1, int(len(full_dataset) * args.val_ratio))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"Train: {n_train}  |  Val: {n_val}")
    print(f"Effective batch size: {args.batch_size * args.accum_steps}")
    print(f"Loss weights — L1: {args.w_l1}  SSIM: {args.w_ssim}  Freq: {args.w_freq}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # ---- Model ----
    model = PromptIR(
        dim=args.dim,
        num_blocks=(4, 6, 6, 8),
        num_refinement_blocks=4,
        heads=(1, 2, 4, 8),
        ffn_expansion_factor=2.66,
        bias=False,
        prompt_len=args.prompt_len,
        prompt_size=args.patch_size // 4,
        use_prompt=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    # ---- Loss ----
    l1_loss = nn.L1Loss()
    ssim_loss = SSIMLoss(channels=3).to(device)

    # ---- Optimizer & Scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmupLR(
        optimizer,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
    )
    scaler = GradScaler("cuda")

    start_epoch = 0
    best_psnr = 0.0

    # ---- Resume ----
    if args.resume and Path(args.resume).exists():
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        # Reset optimizer/scheduler for new training phase
        best_psnr = ckpt.get("best_psnr", 0.0)
        print(f"  Loaded weights. Starting fresh optimizer. Best PSNR so far: {best_psnr:.2f}")

    # ---- Training loop ----
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad()

        for step, (deg, clean) in enumerate(train_loader):
            deg = deg.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            with autocast("cuda"):
                pred = model(deg)
                loss = (
                    args.w_l1 * l1_loss(pred, clean)
                    + args.w_ssim * ssim_loss(pred.clamp(0, 1), clean)
                    + args.w_freq * freq_loss(pred, clean)
                ) / args.accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * args.accum_steps

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch [{epoch+1:03d}/{args.epochs}] "
            f"Loss: {avg_loss:.4f}  LR: {current_lr:.2e}  "
            f"Time: {elapsed:.1f}s"
        )

        # ---- Validation ----
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            model.eval()
            psnr_vals = []
            with torch.no_grad():
                for deg, clean in val_loader:
                    deg = deg.to(device)
                    clean = clean.to(device)
                    with autocast("cuda"):
                        pred = model(deg).clamp(0, 1)
                    psnr_vals.append(compute_psnr(pred.float(), clean.float()))
            val_psnr = np.mean(psnr_vals)
            print(f"  [Val] PSNR: {val_psnr:.2f} dB")

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "best_psnr": best_psnr,
                    },
                    os.path.join(args.ckpt_dir, "best_model.pth"),
                )

        if (epoch + 1) % 20 == 0:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_psnr": best_psnr,
                },
                os.path.join(args.ckpt_dir, f"epoch_{epoch+1:03d}.pth"),
            )

    print(f"\nTraining complete. Best Val PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    main()