import argparse
import copy
import csv
import math
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from PIL import Image


# ============================================================
# 0. Reproducibility
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


# ============================================================
# 1. Device selection
# ============================================================
def get_device():
    if torch.backends.mps.is_available():
        print("✅ Using Apple MPS (Metal) backend")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("✅ Using CUDA backend")
        return torch.device("cuda")
    else:
        print("⚠️  Using CPU backend")
        return torch.device("cpu")


# ============================================================
# 2. Test Dataset (unlabeled, flat directory)
# ============================================================
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_names = sorted([
            f for f in os.listdir(root_dir)
            if f.lower().endswith(IMG_EXTENSIONS)
        ])
        print(f"📁 Test set: {len(self.image_names)} images")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        fname = self.image_names[idx]
        img = Image.open(os.path.join(self.root_dir, fname)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, fname


# ============================================================
# 3. Data Augmentation & Transforms
# ============================================================
def get_transforms(img_size: int = 224):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


def get_tta_transforms(img_size: int = 224):
    """10-view TTA for stronger prediction."""
    _n = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tta_list = [
        # 1. Standard center crop
        transforms.Compose([
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(), _n,
        ]),
        # 2. Horizontal flip
        transforms.Compose([
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(), _n,
        ]),
        # 3. Larger resize
        transforms.Compose([
            transforms.Resize(int(img_size * 288 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(), _n,
        ]),
        # 4. Larger resize + flip
        transforms.Compose([
            transforms.Resize(int(img_size * 288 / 224)),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(), _n,
        ]),
        # 5. Smaller resize (tighter crop)
        transforms.Compose([
            transforms.Resize(int(img_size * 232 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(), _n,
        ]),
        # 6-10. FiveCrop (all 4 corners + center)
        transforms.Compose([
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.FiveCrop(img_size),
            transforms.Lambda(lambda crops: crops[0]),
            transforms.ToTensor(), _n,
        ]),
        transforms.Compose([
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.FiveCrop(img_size),
            transforms.Lambda(lambda crops: crops[1]),
            transforms.ToTensor(), _n,
        ]),
        transforms.Compose([
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.FiveCrop(img_size),
            transforms.Lambda(lambda crops: crops[2]),
            transforms.ToTensor(), _n,
        ]),
        transforms.Compose([
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.FiveCrop(img_size),
            transforms.Lambda(lambda crops: crops[3]),
            transforms.ToTensor(), _n,
        ]),
        transforms.Compose([
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.FiveCrop(img_size),
            transforms.Lambda(lambda crops: crops[4]),
            transforms.ToTensor(), _n,
        ]),
    ]
    return tta_list


# ============================================================
# 4. MixUp & CutMix
# ============================================================
def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    _, _, H, W = x.shape
    r = np.sqrt(1.0 - lam)
    ch, cw = int(H * r), int(W * r)
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = np.clip(cy - ch // 2, 0, H), np.clip(cy + ch // 2, 0, H)
    x1, x2 = np.clip(cx - cw // 2, 0, W), np.clip(cx + cw // 2, 0, W)
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - ((y2 - y1) * (x2 - x1) / (H * W))
    return x, y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# 5. Warmup + Cosine Annealing Scheduler
# ============================================================
class WarmupCosineScheduler:
    """Linear warmup for warmup_epochs, then cosine decay to eta_min."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        # Store base LRs from each param group
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step_count = 0

    def step(self):
        self._step_count += 1
        epoch = self._step_count
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if epoch <= self.warmup_epochs:
                # Linear warmup
                pg["lr"] = base_lr * epoch / max(1, self.warmup_epochs)
            else:
                # Cosine decay
                progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
                pg["lr"] = self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def state_dict(self):
        return {"step_count": self._step_count}

    def load_state_dict(self, state):
        self._step_count = state["step_count"]
        # Replay steps to restore LR
        saved = self._step_count
        self._step_count = 0
        for _ in range(saved):
            self.step()


# ============================================================
# 6. Exponential Moving Average (EMA)
# ============================================================
class EMA:
    """Maintains an exponential moving average of model parameters."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1 - self.decay)

    def apply(self, model):
        """Swap model weights with EMA weights. Call before evaluation."""
        self.backup = {n: p.data.clone() for n, p in model.named_parameters()}
        for (n, p), s in zip(model.named_parameters(), self.shadow.parameters()):
            p.data.copy_(s.data)

    def restore(self, model):
        """Restore original weights after evaluation."""
        for n, p in model.named_parameters():
            p.data.copy_(self.backup[n])
        del self.backup


# ============================================================
# 7. Model: ResNet with custom classification head
# ============================================================
RESNET_CONFIGS = {
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V2),
}


class ResNetClassifier(nn.Module):
    """
    ResNet backbone with a custom 2-layer MLP head.
    Modification: replaced single FC with [Linear → BN → ReLU → Dropout → Linear]
    to improve generalization on smaller datasets.
    """
    def __init__(self, arch="resnet50", num_classes=100, dropout=0.4, pretrained=True):
        super().__init__()
        if arch not in RESNET_CONFIGS:
            raise ValueError(f"Unsupported: {arch}")
        model_fn, weight_cls = RESNET_CONFIGS[arch]
        weights = weight_cls if pretrained else None
        backbone = model_fn(weights=weights)

        self.features = nn.Sequential(*list(backbone.children())[:-1])  # → (B, 2048, 1, 1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.classifier(self.features(x))

    def freeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = False
        print("🔒 Backbone frozen")

    def unfreeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = True
        print("🔓 Backbone unfrozen")

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"📊 Total: {total:,} ({total/1e6:.1f}M)  Trainable: {trainable:,} ({trainable/1e6:.1f}M)")
        return total, trainable


# ============================================================
# 8. Trainer
# ============================================================
class Trainer:
    def __init__(self, model, device, num_classes, label_smoothing=0.1):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.best_acc = 0.0
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        print(f"🏷️  Label smoothing: {label_smoothing}")

    def train_one_epoch(self, loader, optimizer, use_mixup=True, ema=None):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0

        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(self.device), labels.to(self.device)

            if use_mixup and random.random() < 0.5:
                if random.random() < 0.5:
                    images, y_a, y_b, lam = mixup_data(images, labels, alpha=0.4)
                else:
                    images, y_a, y_b, lam = cutmix_data(images, labels, alpha=1.0)
                outputs = self.model(images)
                loss = mixup_criterion(self.criterion, outputs, y_a, y_b, lam)
                _, pred = outputs.max(1)
                correct += lam * pred.eq(y_a).sum().item() + (1 - lam) * pred.eq(y_b).sum().item()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                _, pred = outputs.max(1)
                correct += pred.eq(labels).sum().item()

            total += labels.size(0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            optimizer.step()

            if ema is not None:
                ema.update(self.model)

            running_loss += loss.item() * labels.size(0)

            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch [{batch_idx+1}/{len(loader)}]  Loss: {loss.item():.4f}")

        return running_loss / total, 100.0 * correct / total

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
        return running_loss / total, 100.0 * correct / total

    @torch.no_grad()
    def predict(self, test_dir, transform, batch_size=32, num_workers=0, pin_memory=False):
        self.model.eval()
        ds = TestDataset(test_dir, transform=transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
        results = []
        for images, fnames in loader:
            images = images.to(self.device)
            _, pred = self.model(images).max(1)
            for f, p in zip(fnames, pred.cpu().tolist()):
                results.append((f, p))
        return results

    @torch.no_grad()
    def predict_tta(self, test_dir, tta_transforms, batch_size=32,
                    num_workers=0, pin_memory=False):
        self.model.eval()
        all_probs, all_names = None, None
        for i, tfm in enumerate(tta_transforms):
            print(f"  TTA view {i+1}/{len(tta_transforms)} ...")
            ds = TestDataset(test_dir, transform=tfm)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory)
            probs_list, names_list = [], []
            for images, fnames in loader:
                images = images.to(self.device)
                probs_list.append(torch.softmax(self.model(images), dim=1).cpu())
                names_list.extend(fnames)
            probs = torch.cat(probs_list, dim=0)
            if all_probs is None:
                all_probs, all_names = probs, names_list
            else:
                all_probs += probs
        all_probs /= len(tta_transforms)
        _, predicted = all_probs.max(1)
        return [(n, p.item()) for n, p in zip(all_names, predicted)]

    def save_checkpoint(self, epoch, optimizer, scheduler, filename, ema=None):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_acc": self.best_acc,
        }
        if ema is not None:
            state["ema_state_dict"] = ema.shadow.state_dict()
        torch.save(state, filename)

    def load_checkpoint(self, filename, ema=None):
        ckpt = torch.load(filename, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.best_acc = ckpt.get("best_acc", 0.0)
        if ema is not None and "ema_state_dict" in ckpt:
            ema.shadow.load_state_dict(ckpt["ema_state_dict"])
        print(f"📂 Loaded: {filename} (best_acc={self.best_acc:.2f}%)")
        return ckpt


# ============================================================
# 9. CSV output
# ============================================================
def save_predictions_csv(results, output_path, class_names=None):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "pred_label"])
        for fname, pred_idx in results:
            name_no_ext = os.path.splitext(fname)[0]
            label = class_names[pred_idx] if class_names else str(pred_idx)
            writer.writerow([name_no_ext, label])
    print(f"📄 Saved {len(results)} predictions → {output_path}")


# ============================================================
# 10. Main
# ============================================================
def find_data_dirs(data_dir):
    candidates = {
        "train": ["train", "training"],
        "val": ["val", "validation", "valid"],
        "test": ["test", "testing"],
    }
    found = {}
    for key, names in candidates.items():
        for name in names:
            path = os.path.join(data_dir, name)
            if os.path.isdir(path):
                found[key] = path
                break
        if key not in found:
            print(f"⚠️  {key}/ not found in {data_dir}")
    return found


def main():
    parser = argparse.ArgumentParser(description="ResNet Image Classification (Enhanced)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="resnet101",
                        choices=["resnet50", "resnet101"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=96,
                        help="Batch size (default: 96, fits 4090 at 288px)")
    parser.add_argument("--img_size", type=int, default=288,
                        help="Input image size (default: 288 for higher accuracy)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Stage 1 LR (default: 1e-3)")
    parser.add_argument("--lr_finetune", type=float, default=2e-4,
                        help="Stage 2 head LR (default: 2e-4)")
    parser.add_argument("--lr_backbone_ratio", type=float, default=0.2,
                        help="Backbone LR = lr_finetune * this ratio (default: 0.2)")
    parser.add_argument("--freeze_epochs", type=int, default=5)
    parser.add_argument("--warmup_epochs", type=int, default=3,
                        help="Warmup epochs in stage 2 (default: 3)")
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor (default: 0.1)")
    parser.add_argument("--ema_decay", type=float, default=0.999,
                        help="EMA decay rate (0 to disable, default: 0.999)")
    parser.add_argument("--num_workers", type=int, default=-1,
                        help="-1 = auto (8 CUDA, 0 MPS)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--csv_output", type=str, default="prediction.csv")
    parser.add_argument("--no_mixup", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    use_cuda = device.type == "cuda"
    if args.num_workers == -1:
        args.num_workers = 8 if use_cuda else 0
    pin_memory = use_cuda
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        print(f"⚡ cudnn.benchmark=True, workers={args.num_workers}, pin_memory=True")

    dirs = find_data_dirs(args.data_dir)
    train_tfm, val_tfm = get_transforms(args.img_size)

    train_ds = datasets.ImageFolder(dirs["train"], transform=train_tfm)
    num_classes = len(train_ds.classes)
    class_names = train_ds.classes
    print(f"📁 Classes: {num_classes}")

    # --- Model ---
    model = ResNetClassifier(arch=args.model, num_classes=num_classes,
                             dropout=args.dropout, pretrained=True)
    print(f"🏗️  {args.model}, img_size={args.img_size}")
    model.count_parameters()
    trainer = Trainer(model, device, num_classes, label_smoothing=args.label_smoothing)
    os.makedirs(args.output_dir, exist_ok=True)

    # EMA
    use_ema = args.ema_decay > 0
    ema = EMA(model, decay=args.ema_decay) if use_ema else None
    if use_ema:
        print(f"📐 EMA enabled (decay={args.ema_decay})")

    # ===========================================================
    # Predict-only mode
    # ===========================================================
    if args.predict:
        if args.checkpoint is None:
            print("❌ --checkpoint required"); return
        ckpt = trainer.load_checkpoint(args.checkpoint, ema)

        # Use EMA weights for prediction if available
        if use_ema and "ema_state_dict" in ckpt:
            ema.apply(model)
            print("📐 Using EMA weights for prediction")

        if "test" not in dirs:
            print("❌ No test dir"); return

        if args.tta:
            tta_tfms = get_tta_transforms(args.img_size)
            results = trainer.predict_tta(dirs["test"], tta_tfms, args.batch_size,
                                          args.num_workers, pin_memory)
        else:
            results = trainer.predict(dirs["test"], val_tfm, args.batch_size,
                                      args.num_workers, pin_memory)

        save_predictions_csv(results, args.csv_output, class_names)

        if use_ema and "ema_state_dict" in ckpt:
            ema.restore(model)
        return

    # ===========================================================
    # Training mode
    # ===========================================================
    val_ds = datasets.ImageFolder(dirs["val"], transform=val_tfm)
    print(f"📁 Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin_memory,
                              drop_last=True, persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin_memory,
                            persistent_workers=(args.num_workers > 0))

    # -----------------------------------------------------------
    # Stage 1: Frozen backbone
    # -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 1: Classifier head only (backbone frozen)")
    print("=" * 60)
    model.freeze_backbone()
    model.count_parameters()

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=1,
                                       total_epochs=args.freeze_epochs, eta_min=1e-5)

    start_epoch = 0
    if args.resume:
        ckpt = trainer.load_checkpoint(args.resume, ema)
        start_epoch = ckpt["epoch"] + 1
        if start_epoch < args.freeze_epochs:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if ckpt.get("scheduler_state_dict"):
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    for epoch in range(start_epoch, args.freeze_epochs):
        t0 = time.time()
        train_loss, train_acc = trainer.train_one_epoch(
            train_loader, optimizer, use_mixup=False, ema=ema
        )
        # Evaluate with EMA if available
        if use_ema:
            ema.apply(model)
        val_loss, val_acc = trainer.evaluate(val_loader)
        if use_ema:
            ema.restore(model)

        scheduler.step()
        elapsed = time.time() - t0

        print(f"[S1 {epoch+1}/{args.freeze_epochs}]  "
              f"Train: {train_loss:.4f}/{train_acc:.2f}%  "
              f"Val: {val_loss:.4f}/{val_acc:.2f}%  "
              f"LR: {scheduler.get_last_lr()[0]:.6f}  {elapsed:.0f}s")

        if val_acc > trainer.best_acc:
            trainer.best_acc = val_acc
            trainer.save_checkpoint(epoch, optimizer, scheduler,
                                    os.path.join(args.output_dir, "best_model.pth"), ema)
            print(f"  ✅ Best: {val_acc:.2f}%")

    # -----------------------------------------------------------
    # Stage 2: Full fine-tuning with layer-wise LR
    # -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 2: Full fine-tuning")
    print("=" * 60)
    model.unfreeze_backbone()
    model.count_parameters()

    backbone_lr = args.lr_finetune * args.lr_backbone_ratio
    head_lr = args.lr_finetune
    print(f"📐 LR — backbone: {backbone_lr:.6f}, head: {head_lr:.6f}")

    optimizer = optim.AdamW([
        {"params": model.features.parameters(), "lr": backbone_lr},
        {"params": model.classifier.parameters(), "lr": head_lr},
    ], weight_decay=1e-4)

    finetune_epochs = args.epochs - args.freeze_epochs
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=args.warmup_epochs,
                                       total_epochs=finetune_epochs, eta_min=1e-6)

    for epoch in range(args.freeze_epochs, args.epochs):
        t0 = time.time()
        train_loss, train_acc = trainer.train_one_epoch(
            train_loader, optimizer, use_mixup=(not args.no_mixup), ema=ema
        )

        if use_ema:
            ema.apply(model)
        val_loss, val_acc = trainer.evaluate(val_loader)
        if use_ema:
            ema.restore(model)

        scheduler.step()
        elapsed = time.time() - t0

        lrs = scheduler.get_last_lr()
        print(f"[S2 {epoch+1}/{args.epochs}]  "
              f"Train: {train_loss:.4f}/{train_acc:.2f}%  "
              f"Val: {val_loss:.4f}/{val_acc:.2f}%  "
              f"LR: {lrs[0]:.6f}/{lrs[1]:.6f}  {elapsed:.0f}s")

        if val_acc > trainer.best_acc:
            trainer.best_acc = val_acc
            trainer.save_checkpoint(epoch, optimizer, scheduler,
                                    os.path.join(args.output_dir, "best_model.pth"), ema)
            print(f"  ✅ Best: {val_acc:.2f}%")

        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(epoch, optimizer, scheduler,
                                    os.path.join(args.output_dir, f"ckpt_ep{epoch+1}.pth"), ema)

    # -----------------------------------------------------------
    # Final evaluation & prediction
    # -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL EVALUATION & PREDICTION")
    print("=" * 60)
    ckpt = trainer.load_checkpoint(os.path.join(args.output_dir, "best_model.pth"), ema)

    # Use EMA weights
    if use_ema and "ema_state_dict" in ckpt:
        ema.apply(model)
        print("📐 Using EMA weights")

    val_loss, val_acc = trainer.evaluate(val_loader)
    print(f"🏆 Best Val Accuracy: {val_acc:.2f}%")

    if "test" in dirs:
        print("\nGenerating predictions ...")
        if args.tta:
            tta_tfms = get_tta_transforms(args.img_size)
            results = trainer.predict_tta(dirs["test"], tta_tfms, args.batch_size,
                                          args.num_workers, pin_memory)
        else:
            results = trainer.predict(dirs["test"], val_tfm, args.batch_size,
                                      args.num_workers, pin_memory)
        save_predictions_csv(results, args.csv_output, class_names)

    if use_ema and "ema_state_dict" in ckpt:
        ema.restore(model)

    print("\n🎉 Done!")


if __name__ == "__main__":
    main()