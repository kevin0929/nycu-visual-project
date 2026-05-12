import json
from pathlib import Path

import numpy as np
import torch
import tifffile
from PIL import Image
from torch.utils.data import Dataset


CLASS_NAMES = ["class1", "class2", "class3", "class4"]
CLASS_TO_ID = {name: idx + 1 for idx, name in enumerate(CLASS_NAMES)}  # 1-indexed


def load_tif(path: str) -> np.ndarray:
    """Load .tif file and return HxWx3 uint8 RGB array."""
    img = tifffile.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img.astype(np.uint8)


class CellDataset(Dataset):
    """
    Medical cell instance segmentation dataset (train / val split).

    Each item returns:
        image  : FloatTensor[3, H, W]  normalized to [0, 1]
        target : dict with keys boxes, labels, masks, image_id, area, iscrowd
    """

    def __init__(self, data_root: str, split: str = "train", transforms=None):
        self.transforms = transforms
        train_dir = Path(data_root) / "train"
        image_dirs = sorted(train_dir.iterdir())

        n_train = int(len(image_dirs) * 0.8)
        if split == "all":
            self.image_dirs = image_dirs
        elif split == "train":
            self.image_dirs = image_dirs[:n_train]
        else:
            self.image_dirs = image_dirs[n_train:]

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, idx):
        img_dir = self.image_dirs[idx]
        img_np = load_tif(str(img_dir / "image.tif"))
        h, w = img_np.shape[:2]

        boxes, labels, masks = [], [], []

        for class_name in CLASS_NAMES:
            mask_path = img_dir / f"{class_name}.tif"
            if not mask_path.exists():
                continue
            mask = tifffile.imread(str(mask_path))
            instance_ids = np.unique(mask)
            instance_ids = instance_ids[instance_ids > 0]

            for inst_id in instance_ids:
                binary = (mask == inst_id).astype(np.uint8)
                if binary.sum() < 10:
                    continue

                rows = np.any(binary, axis=1)
                cols = np.any(binary, axis=0)
                y1, y2 = np.where(rows)[0][[0, -1]]
                x1, x2 = np.where(cols)[0][[0, -1]]
                if x2 <= x1 or y2 <= y1:
                    continue

                boxes.append([float(x1), float(y1), float(x2 + 1), float(y2 + 1)])
                labels.append(CLASS_TO_ID[class_name])
                masks.append(binary)

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            masks_t = torch.zeros((0, h, w), dtype=torch.bool)
            area_t = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
            masks_t = torch.as_tensor(np.stack(masks), dtype=torch.bool)
            area_t = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area_t,
            "iscrowd": torch.zeros((len(boxes_t),), dtype=torch.int64),
        }

        # Convert HxWx3 → tensor [3,H,W] in [0,1]
        img_tensor = torch.as_tensor(img_np.transpose(2, 0, 1), dtype=torch.float32) / 255.0

        if self.transforms is not None:
            img_tensor, target = self.transforms(img_tensor, target)

        return img_tensor, target


class CellTestDataset(Dataset):
    """Test dataset (no masks). Returns image tensor, image_id, filename."""

    def __init__(self, test_dir: str, id_map_path: str):
        with open(id_map_path) as f:
            id_map = json.load(f)
        self.fname_to_id = {e["file_name"]: e["id"] for e in id_map}
        self.files = sorted(Path(test_dir).glob("*.tif"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img_np = load_tif(str(path))
        image_id = self.fname_to_id.get(path.name, -1)
        img_tensor = torch.as_tensor(
            img_np.transpose(2, 0, 1), dtype=torch.float32
        ) / 255.0
        return img_tensor, image_id, path.name