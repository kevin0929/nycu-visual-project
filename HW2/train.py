import argparse
import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from dataset import DigitDetectionDataset, make_transforms, collate_fn
from criterion import DETRLoss


# ---------------------------------------------------------------------------
# Minimal vanilla DETR (clean rewrite, batch_first=True throughout)
# ---------------------------------------------------------------------------

import math
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, x, mask):
        not_mask = ~mask
        y = not_mask.cumsum(1, dtype=torch.float32)
        x_ = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y  = y  / (y[:, -1:, :] + eps) * 2 * math.pi
        x_ = x_ / (x_[:, :, -1:] + eps) * 2 * math.pi
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_[..., None] / dim_t
        pos_y = y[..., None]  / dim_t
        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], -1).flatten(3)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], -1).flatten(3)
        return torch.cat([pos_y, pos_x], dim=3).permute(0, 3, 1, 2)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = F.relu(l(x)) if i < len(self.layers) - 1 else l(x)
        return x


class VanillaDETR(nn.Module):
    """
    DETR with ResNet-50 backbone.
    - batch_first=True throughout (no permute bugs)
    - aux_loss on every decoder layer
    - FrozenBN on backbone for stability
    """

    def __init__(self, num_classes=10, num_queries=100, d_model=256,
                 nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 pretrained_backbone=True, aux_loss=True):
        super().__init__()
        self.aux_loss = aux_loss
        self.num_classes = num_classes

        # Backbone
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        bb = resnet50(weights=weights)
        # Replace BN with frozen BN for stability
        def freeze_bn(m):
            for name, child in m.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    frozen = nn.BatchNorm2d(child.num_features)
                    frozen.weight.data = child.weight.data.clone()
                    frozen.bias.data   = child.bias.data.clone()
                    frozen.running_mean = child.running_mean.clone()
                    frozen.running_var  = child.running_var.clone()
                    frozen.weight.requires_grad_(False)
                    frozen.bias.requires_grad_(False)
                    setattr(m, name, frozen)
                else:
                    freeze_bn(child)
        freeze_bn(bb)

        self.backbone = nn.Sequential(
            bb.conv1, bb.bn1, bb.relu, bb.maxpool,
            bb.layer1, bb.layer2, bb.layer3, bb.layer4)
        self.backbone_out_channels = 2048

        # Freeze layer0~layer1 (low-level, not task-specific)
        for p in list(self.backbone[:6].parameters()):
            p.requires_grad_(False)

        # Input projection
        self.input_proj = nn.Conv2d(self.backbone_out_channels, d_model, kernel_size=1)

        # Positional encoding
        self.pos_enc = PositionEmbeddingSine(d_model // 2)

        # Transformer (batch_first=True)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Object queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Prediction heads
        self.class_embed = nn.Linear(d_model, num_classes + 1)  # +1 no-object
        self.bbox_embed  = MLP(d_model, d_model, 4, 3)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0)

    def forward(self, samples):
        src  = samples.tensors   # (B, 3, H, W)
        mask = samples.mask      # (B, H, W)  True=pad

        # Backbone
        feat = self.backbone(src)                         # (B, 2048, H', W')
        src2 = self.input_proj(feat)                      # (B, d, H', W')

        # Downsample mask to feature map size
        mask_down = F.interpolate(
            mask.unsqueeze(1).float(), size=feat.shape[-2:]).bool().squeeze(1)

        # Positional encoding
        pos = self.pos_enc(src2, mask_down)               # (B, d, H', W')

        # Flatten spatial: (B, H'*W', d)
        B, d, H, W = src2.shape
        src_flat  = (src2 + pos).flatten(2).transpose(1, 2)   # (B, HW, d)
        mask_flat = mask_down.flatten(1)                        # (B, HW)

        # Object queries: (B, Q, d)
        tgt = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        # Transformer — returns (B, Q, d) for each decoder layer via hooks
        # Use standard forward which returns final decoder output
        hs = self.transformer(
            src=src_flat,
            tgt=tgt,
            src_key_padding_mask=mask_flat,
        )  # (B, Q, d)

        out_class = self.class_embed(hs)    # (B, Q, C+1)
        out_bbox  = self.bbox_embed(hs).sigmoid()  # (B, Q, 4)

        return {"pred_logits": out_class, "pred_boxes": out_bbox}


# ---------------------------------------------------------------------------
# NestedTensor
# ---------------------------------------------------------------------------

class NestedTensor:
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask    = mask

    def to(self, device):
        return NestedTensor(self.tensors.to(device), self.mask.to(device))


def nested_tensor_from_tensor_list(tensor_list):
    max_h = max(img.shape[1] for img in tensor_list)
    max_w = max(img.shape[2] for img in tensor_list)
    B = len(tensor_list)
    batched = torch.zeros(B, 3, max_h, max_w,
                          dtype=tensor_list[0].dtype,
                          device=tensor_list[0].device)
    mask = torch.ones(B, max_h, max_w, dtype=torch.bool,
                      device=tensor_list[0].device)
    for i, img in enumerate(tensor_list):
        c, h, w = img.shape
        batched[i, :c, :h, :w].copy_(img)
        mask[i, :h, :w] = False
    return NestedTensor(batched, mask)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser("Vanilla DETR Training")
    p.add_argument("--data_root",    type=str, required=True)
    p.add_argument("--train_ann",    type=str, default="train.json")
    p.add_argument("--val_ann",      type=str, default="valid.json")
    p.add_argument("--num_classes",  type=int, default=10)
    p.add_argument("--num_queries",  type=int, default=100)
    p.add_argument("--d_model",      type=int, default=256)
    p.add_argument("--nhead",        type=int, default=8)
    p.add_argument("--num_encoder_layers", type=int, default=6)
    p.add_argument("--num_decoder_layers", type=int, default=6)
    p.add_argument("--dim_feedforward",    type=int, default=2048)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--epochs",       type=int, default=150)
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--accum_steps",  type=int, default=1)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--lr_backbone",  type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lr_drop",      type=int, default=100)
    p.add_argument("--clip_max_norm",type=float, default=0.1)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--use_amp",      action="store_true", default=True)
    p.add_argument("--cost_class",   type=float, default=1.0)
    p.add_argument("--cost_bbox",    type=float, default=5.0)
    p.add_argument("--cost_giou",    type=float, default=2.0)
    p.add_argument("--eos_coef",     type=float, default=0.1)
    p.add_argument("--output_dir",   type=str, default="checkpoints_vanilla")
    p.add_argument("--resume",       type=str, default="")
    p.add_argument("--override_lr",   action="store_true", default=False,
                   help="Override lr in optimizer after resume")
    p.add_argument("--eval_freq",    type=int, default=5)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Train / Val
# ---------------------------------------------------------------------------

def train_one_epoch(model, criterion, loader, optimizer, device, epoch,
                    scaler=None, accum_steps=1, clip_max_norm=0.1):
    model.train()
    criterion.train()
    total = cls_ = l1_ = giou_ = 0.0
    optimizer.zero_grad()
    start = time.time()

    for step, (images, targets) in enumerate(loader):
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in targets]
        samples = nested_tensor_from_tensor_list([img.to(device) for img in images])

        with autocast("cuda", enabled=(scaler is not None)):
            outputs = model(samples)
            if not torch.isfinite(outputs["pred_logits"]).all():
                print(f"  [WARN] nan in output, skipping")
                optimizer.zero_grad()
                continue
            loss_dict = criterion(outputs, targets)
            loss = loss_dict["loss_total"] / accum_steps

        if not torch.isfinite(loss):
            print(f"  [WARN] nan loss, skipping")
            optimizer.zero_grad()
            continue

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            if scaler:
                if clip_max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
                optimizer.step()
            optimizer.zero_grad()

        total += loss_dict["loss_total"].item()
        cls_  += loss_dict["loss_cls"].item()
        l1_   += loss_dict["loss_l1"].item()
        giou_ += loss_dict["loss_giou"].item()

        if (step + 1) % 100 == 0:
            mem = torch.cuda.max_memory_allocated(device) / 1e9
            elapsed = time.time() - start
            print(f"  [E{epoch} {step+1}/{len(loader)}] "
                  f"loss={loss_dict['loss_total'].item():.3f} "
                  f"cls={loss_dict['loss_cls'].item():.3f} "
                  f"l1={loss_dict['loss_l1'].item():.3f} "
                  f"giou={loss_dict['loss_giou'].item():.3f} "
                  f"GPU={mem:.1f}GB ({elapsed:.0f}s)")
            start = time.time()

    n = len(loader)
    return dict(loss=total/n, loss_cls=cls_/n, loss_l1=l1_/n, loss_giou=giou_/n)


@torch.no_grad()
def validate(model, criterion, loader, device):
    model.eval(); criterion.eval()
    total = 0.0
    for images, targets in loader:
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in targets]
        samples = nested_tensor_from_tensor_list([img.to(device) for img in images])
        outputs = model(samples)
        total += criterion(outputs, targets)["loss_total"].item()
    return total / len(loader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_ds = DigitDetectionDataset(
        os.path.join(args.data_root, "train"),
        os.path.join(args.data_root, args.train_ann),
        transforms=make_transforms("train"))
    val_ds = DigitDetectionDataset(
        os.path.join(args.data_root, "valid"),
        os.path.join(args.data_root, args.val_ann),
        transforms=make_transforms("val"))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=collate_fn,
                               pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, collate_fn=collate_fn,
                               pin_memory=True)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Model
    model = VanillaDETR(
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        pretrained_backbone=True,
        aux_loss=False,
    ).to(device)

    # Separate LR for backbone
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    backbone_params = [p for p in model.parameters()
                       if id(p) in backbone_ids and p.requires_grad]
    other_params    = [p for p in model.parameters()
                       if id(p) not in backbone_ids]
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": other_params,    "lr": args.lr},
    ], weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.1)

    criterion = DETRLoss(
        num_classes=args.num_classes,
        cost_class=args.cost_class,
        cost_bbox=args.cost_bbox,
        cost_giou=args.cost_giou,
        eos_coef=args.eos_coef,
    ).to(device)

    scaler = GradScaler("cuda") if args.use_amp else None

    # Resume
    start_epoch = 1
    best_val = float("inf")
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", float("inf"))
        print(f"Resumed from epoch {ckpt['epoch']}")
        if args.override_lr:
            for i, pg in enumerate(optimizer.param_groups):
                new_lr = args.lr_backbone if i == 0 else args.lr
                pg["lr"] = new_lr
                print(f"  Override param_group[{i}] lr -> {new_lr}")

    # Loop
    log = []
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        stats = train_one_epoch(model, criterion, train_loader, optimizer,
                                device, epoch, scaler, args.accum_steps,
                                args.clip_max_norm)
        scheduler.step()

        val_loss = 0.0
        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            val_loss = validate(model, criterion, val_loader, device)
            print(f"  Val loss: {val_loss:.4f}")

        print(f"  Train: loss={stats['loss']:.4f} "
              f"cls={stats['loss_cls']:.4f} "
              f"l1={stats['loss_l1']:.4f} "
              f"giou={stats['loss_giou']:.4f}")

        ckpt = dict(epoch=epoch, model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    best_val=best_val,
                    args=vars(args), model_type="vanilla")
        torch.save(ckpt, os.path.join(args.output_dir, "last.pth"))

        if val_loss > 0 and val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(args.output_dir, "best.pth"))
            print(f"  *** Best (val={best_val:.4f})")

        # Save periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(ckpt, os.path.join(args.output_dir, f"epoch_{epoch:03d}.pth"))
            print(f"  Saved periodic checkpoint: epoch_{epoch:03d}.pth")

        log.append({"epoch": epoch, **stats, "val_loss": val_loss})

    with open(os.path.join(args.output_dir, "log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print("Done!")


if __name__ == "__main__":
    main()