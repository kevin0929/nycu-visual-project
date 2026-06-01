import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class RestoreTrainDataset(Dataset):
    """Paired degraded/clean dataset for training."""

    def __init__(self, data_dir: str, patch_size: int = 128):
        super().__init__()
        self.patch_size = patch_size
        degraded_dir = Path(data_dir) / "degraded"
        clean_dir = Path(data_dir) / "clean"

        self.pairs = []
        for fname in sorted(degraded_dir.iterdir()):
            if not fname.suffix.lower() in (".png", ".jpg", ".jpeg"):
                continue
            stem = fname.stem  # e.g. "rain-1" or "snow-1"
            if stem.startswith("rain-"):
                idx = stem.split("-")[1]
                clean_name = f"rain_clean-{idx}.png"
            elif stem.startswith("snow-"):
                idx = stem.split("-")[1]
                clean_name = f"snow_clean-{idx}.png"
            else:
                continue
            clean_path = clean_dir / clean_name
            if clean_path.exists():
                self.pairs.append((str(fname), str(clean_path)))

        print(f"[Dataset] Found {len(self.pairs)} training pairs.")

    def __len__(self):
        return len(self.pairs)

    def _load_rgb(self, path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _augment(self, degraded: Image.Image, clean: Image.Image):
        """Random crop + flip augmentations (applied identically to both)."""
        ps = self.patch_size
        w, h = degraded.size

        # Random crop
        if w > ps and h > ps:
            x = random.randint(0, w - ps)
            y = random.randint(0, h - ps)
            degraded = TF.crop(degraded, y, x, ps, ps)
            clean = TF.crop(clean, y, x, ps, ps)
        else:
            degraded = TF.resize(degraded, (ps, ps))
            clean = TF.resize(clean, (ps, ps))

        # Random horizontal flip
        if random.random() > 0.5:
            degraded = TF.hflip(degraded)
            clean = TF.hflip(clean)

        # Random vertical flip
        if random.random() > 0.5:
            degraded = TF.vflip(degraded)
            clean = TF.vflip(clean)

        # Random 90-degree rotation
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            degraded = TF.rotate(degraded, 90 * k)
            clean = TF.rotate(clean, 90 * k)

        return degraded, clean

    def __getitem__(self, idx):
        deg_path, clean_path = self.pairs[idx]
        deg_img = self._load_rgb(deg_path)
        clean_img = self._load_rgb(clean_path)
        deg_img, clean_img = self._augment(deg_img, clean_img)

        deg_t = TF.to_tensor(deg_img)    # [0, 1]
        clean_t = TF.to_tensor(clean_img)
        return deg_t, clean_t


class RestoreValDataset(Dataset):
    """Paired degraded/clean dataset for validation (no augmentation)."""

    def __init__(self, data_dir: str):
        super().__init__()
        degraded_dir = Path(data_dir) / "degraded"
        clean_dir = Path(data_dir) / "clean"

        self.pairs = []
        for fname in sorted(degraded_dir.iterdir()):
            if not fname.suffix.lower() in (".png", ".jpg", ".jpeg"):
                continue
            stem = fname.stem
            if stem.startswith("rain-"):
                idx = stem.split("-")[1]
                clean_name = f"rain_clean-{idx}.png"
            elif stem.startswith("snow-"):
                idx = stem.split("-")[1]
                clean_name = f"snow_clean-{idx}.png"
            else:
                continue
            clean_path = clean_dir / clean_name
            if clean_path.exists():
                self.pairs.append((str(fname), str(clean_path), fname.name))

        print(f"[Dataset] Found {len(self.pairs)} validation pairs.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        deg_path, clean_path, fname = self.pairs[idx]
        deg_img = Image.open(deg_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")
        deg_t = TF.to_tensor(deg_img)
        clean_t = TF.to_tensor(clean_img)
        return deg_t, clean_t, fname


class RestoreTestDataset(Dataset):
    """Test dataset — no GT, returns (tensor, filename)."""

    def __init__(self, test_dir: str):
        super().__init__()
        degraded_dir = Path(test_dir) / "degraded"
        self.files = sorted(
            [f for f in degraded_dir.iterdir()
             if f.suffix.lower() in (".png", ".jpg", ".jpeg")],
            key=lambda p: int(p.stem)
        )
        print(f"[Dataset] Found {len(self.files)} test images.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        tensor = TF.to_tensor(img)
        return tensor, path.name