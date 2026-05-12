import argparse
import os
import time
from pathlib import Path

import torch
import torch.utils.data
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from dataset import CellDataset
from transforms import get_train_transforms


# ── Model builder ─────────────────────────────────────────────────────────────
def build_model(num_classes: int):
    """
    Mask R-CNN ResNet-50 FPN v2 with ImageNet-pretrained backbone only.

    Key design choices:
    - weights=None          : no COCO pretrained (assignment only allows ImageNet)
    - weights_backbone      : ImageNet1K pretrained ResNet-50 backbone
    - Custom box + mask heads for 4 cell classes + 1 background
    - Freeze layer1 to preserve low-level ImageNet features during early training
    - max_size=640          : prevents CUDA OOM on large images (up to 1105px in dataset)
    - Trainable params      : ~44M (well under 200M limit)
    """
    model = maskrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
        min_size=600,
        max_size=640,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=200,
        rpn_pre_nms_top_n_train=2000,
        rpn_post_nms_top_n_train=1000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_test=500,
    )

    # Replace box classifier head (91 COCO classes → num_classes)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace mask predictor head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    # Freeze layer1 to preserve low-level ImageNet features
    for p in model.backbone.body.layer1.parameters():
        p.requires_grad = False

    return model


# ── Collate fn ────────────────────────────────────────────────────────────────
def collate_fn(batch):
    return tuple(zip(*batch))


# ── Training loop ─────────────────────────────────────────────────────────────
def train_one_epoch(model, optimizer, loader, device, epoch, scaler, print_freq=20):
    model.train()
    total_loss = 0.0

    for i, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if any(len(t["boxes"]) == 0 for t in targets):
            continue

        with torch.amp.autocast(device_type="cuda"):
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())

        if not torch.isfinite(losses):
            print(f"  [WARN] Non-finite loss at step {i}, skipping.")
            continue

        optimizer.zero_grad()
        scaler.scale(losses).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += losses.item()
        if (i + 1) % print_freq == 0:
            avg = total_loss / (i + 1)
            detail = " | ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
            print(f"[Epoch {epoch}] step {i+1}/{len(loader)} | avg_loss: {avg:.4f} | {detail}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.train()
    total_loss, count = 0.0, 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if any(len(t["boxes"]) == 0 for t in targets):
            continue
        with torch.amp.autocast(device_type="cuda"):
            loss_dict = model(images, targets)
        total_loss += sum(loss_dict.values()).item()
        count += 1
    return total_loss / max(count, 1)


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Datasets ──
    if args.all_data:
        train_ds = CellDataset(args.data_root, split="all", transforms=get_train_transforms())
        val_ds = None
        print(f"Train (all data): {len(train_ds)} images | No val split")
    else:
        train_ds = CellDataset(args.data_root, split="train", transforms=get_train_transforms())
        val_ds = CellDataset(args.data_root, split="val", transforms=None)
        print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = None if val_ds is None else torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=2, collate_fn=collate_fn, pin_memory=True,
    )

    # ── Model ──
    model = build_model(num_classes=5)  # 4 cell types + 1 background
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params / 1e6:.1f}M")

    # ── Optimizer: AdamW ──
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler()

    # ── Resume ──
    os.makedirs(args.output_dir, exist_ok=True)
    start_epoch = 0
    last_ckpt = Path(args.output_dir) / "checkpoint_last.pth"
    if args.resume and last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # ── Training loop ──
    best_val_loss = float("inf")
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, scaler)
        val_loss = evaluate_loss(model, val_loader, device) if val_loader else None
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        val_str = f"{val_loss:.4f}" if val_loss is not None else "N/A (all data mode)"
        print(
            f"Epoch {epoch:03d} | train_loss: {train_loss:.4f} | "
            f"val_loss: {val_str} | lr: {lr_now:.2e} | time: {time.time()-t0:.1f}s"
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        torch.save(ckpt, last_ckpt)

        if val_loss is None:
            torch.save(model.state_dict(), Path(args.output_dir) / "model_best.pth")
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Path(args.output_dir) / "model_best.pth")
            print(f"  ↳ New best val_loss: {best_val_loss:.4f} — saved model_best.pth")

        if (epoch + 1) % args.save_freq == 0:
            torch.save(
                model.state_dict(),
                Path(args.output_dir) / f"model_epoch{epoch:03d}.pth",
            )

    torch.save(model.state_dict(), Path(args.output_dir) / "model_final.pth")
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   type=str,   required=True)
    parser.add_argument("--output_dir",  type=str,   default="output")
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--batch_size",  type=int,   default=2)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int,   default=2)
    parser.add_argument("--save_freq",   type=int,   default=5)
    parser.add_argument("--resume",      action="store_true")
    parser.add_argument("--all_data",    action="store_true",
                        help="Use all 209 images for training (no val split)")
    args = parser.parse_args()
    main(args)