import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as F
from torchvision import tv_tensors


class RandomResizeToScale(torch.nn.Module):
    """Randomly resize shortest side to one of the given sizes (DETR-style)."""

    def __init__(self, sizes, max_size=1333):
        super().__init__()
        self.sizes = sizes
        self.max_size = max_size

    def forward(self, img, target=None):
        size = self.sizes[torch.randint(len(self.sizes), ()).item()]
        # img is a Tensor (C,H,W) after ToImage
        h, w = img.shape[-2], img.shape[-1]
        scale = size / min(h, w)
        if max(h, w) * scale > self.max_size:
            scale = self.max_size / max(h, w)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        img = T.functional.resize(img, [new_h, new_w],
                                  interpolation=T.InterpolationMode.BILINEAR,
                                  antialias=True)
        if target is not None:
            if isinstance(target.get("boxes"), tv_tensors.BoundingBoxes):
                # Scale boxes proportionally
                boxes = target["boxes"].data
                scale_x = new_w / w
                scale_y = new_h / h
                boxes = boxes * torch.tensor([scale_x, scale_y, scale_x, scale_y],
                                              dtype=boxes.dtype)
                target["boxes"] = tv_tensors.BoundingBoxes(
                    boxes,
                    format=tv_tensors.BoundingBoxFormat.XYXY,
                    canvas_size=(new_h, new_w),
                )
            target["orig_size"] = torch.as_tensor([new_h, new_w])
        return img, target


def make_transforms(image_set, img_size=480):
    """Build data augmentation transforms."""
    normalize = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    resize_scales = [320, 352, 384, 416, 448, 480]

    if image_set == "train":
        return T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.RandomHorizontalFlip(),
            T.RandomPhotometricDistort(p=0.5),
            T.RandomZoomOut(fill={tv_tensors.Image: (123, 117, 104), "others": 0}),
            T.RandomIoUCrop(),
            T.SanitizeBoundingBoxes(),
            RandomResizeToScale(resize_scales, max_size=480),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            RandomResizeToScale([img_size], max_size=480),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


class DigitDetectionDataset(Dataset):
    """COCO-format dataset for digit detection."""

    def __init__(self, img_dir, ann_file, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms

        with open(ann_file, "r") as f:
            coco = json.load(f)

        # Build id -> filename map
        self.id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}
        self.image_ids = [img["id"] for img in coco["images"]]

        # Build id -> annotations map
        self.id_to_anns = {img_id: [] for img_id in self.image_ids}
        for ann in coco["annotations"]:
            self.id_to_anns[ann["image_id"]].append(ann)

        # Store image sizes for denormalization
        self.id_to_size = {img["id"]: (img["width"], img["height"])
                           for img in coco["images"]}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        filename = self.id_to_filename[image_id]
        img_path = os.path.join(self.img_dir, filename)

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        anns = self.id_to_anns[image_id]

        boxes = []
        labels = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            # Clamp to image boundaries
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + bw)
            y2 = min(h, y + bh)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(ann["category_id"])  # 1-indexed

        boxes_t = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_t = torch.as_tensor(labels, dtype=torch.int64)

        # Wrap with tv_tensors so v2 geometric transforms update boxes correctly
        boxes_tv = tv_tensors.BoundingBoxes(
            boxes_t,
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(h, w),
        )

        target = {
            "boxes": boxes_tv,
            "labels": labels_t,
            "image_id": torch.tensor([image_id]),
            "orig_size": torch.as_tensor([h, w]),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # Unwrap tv_tensors after transforms
        if isinstance(target["boxes"], tv_tensors.BoundingBoxes):
            new_h, new_w = target["boxes"].canvas_size
            target["orig_size"] = torch.as_tensor([new_h, new_w])
            target["boxes"] = target["boxes"].data

        return img, target


class DigitTestDataset(Dataset):
    """Test dataset (no annotations)."""

    def __init__(self, img_dir, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms

        # Sort to ensure consistent ordering
        self.filenames = sorted(os.listdir(img_dir))
        # Parse image IDs from filenames (e.g., "00001.png" -> 1)
        self.image_ids = []
        for fn in self.filenames:
            stem = os.path.splitext(fn)[0]
            try:
                self.image_ids.append(int(stem))
            except ValueError:
                self.image_ids.append(stem)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = os.path.join(self.img_dir, filename)
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        target = {
            "image_id": torch.tensor([self.image_ids[idx]]),
            "orig_size": torch.as_tensor([h, w]),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


def collate_fn(batch):
    """Custom collate for variable-size images."""
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)