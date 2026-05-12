import random
import torch
import torchvision.transforms.functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            w = image.shape[-1]
            image = image.flip(-1)
            if len(target["boxes"]) > 0:
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes
            if len(target["masks"]) > 0:
                target["masks"] = target["masks"].flip(-1)
        return image, target


class RandomVerticalFlip:
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            h = image.shape[-2]
            image = image.flip(-2)
            if len(target["boxes"]) > 0:
                boxes = target["boxes"].clone()
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
                target["boxes"] = boxes
            if len(target["masks"]) > 0:
                target["masks"] = target["masks"].flip(-2)
        return image, target


class RandomRotation90:
    """Rotate by 0 / 90 / 180 / 270 degrees (masks & boxes follow)."""

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target

        k = random.choice([1, 2, 3])  # 90, 180, 270
        _, h, w = image.shape
        image = torch.rot90(image, k, dims=[-2, -1])
        _, nh, nw = image.shape

        if len(target["masks"]) > 0:
            target["masks"] = torch.rot90(target["masks"].float(), k, dims=[-2, -1]).bool()

        if len(target["boxes"]) > 0:
            boxes = target["boxes"].clone()
            new_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box.tolist()
                if k == 1:   # 90 CCW
                    new_boxes.append([y1, w - x2, y2, w - x1])
                elif k == 2:  # 180
                    new_boxes.append([w - x2, h - y2, w - x1, h - y1])
                elif k == 3:  # 270 CCW
                    new_boxes.append([h - y2, x1, h - y1, x2])
            target["boxes"] = torch.tensor(new_boxes, dtype=torch.float32)

        return image, target


class ColorJitter:
    """Random brightness / contrast / saturation jitter."""

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.3):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, image, target):
        if random.random() < 0.8:
            b = 1.0 + random.uniform(-self.brightness, self.brightness)
            image = torch.clamp(image * b, 0.0, 1.0)
        if random.random() < 0.8:
            c = 1.0 + random.uniform(-self.contrast, self.contrast)
            mean = image.mean(dim=(-1, -2), keepdim=True)
            image = torch.clamp((image - mean) * c + mean, 0.0, 1.0)
        if random.random() < 0.5:
            # Grayscale simulation (useful for H&E stain variation)
            gray = image.mean(dim=0, keepdim=True).expand_as(image)
            alpha = random.uniform(0.0, 0.3)
            image = torch.clamp((1 - alpha) * image + alpha * gray, 0.0, 1.0)
        return image, target


class RandomGaussianNoise:
    """Add small Gaussian noise to simulate microscopy noise."""

    def __init__(self, std: float = 0.02, prob: float = 0.3):
        self.std = std
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            noise = torch.randn_like(image) * self.std
            image = torch.clamp(image + noise, 0.0, 1.0)
        return image, target


class RandomScaleCrop:
    """
    Random scale jitter + crop — simulates different microscope zoom levels.
    Keeps boxes and masks in sync.
    """

    def __init__(self, scale_range=(0.8, 1.2), prob=0.3):
        self.scale_range = scale_range
        self.prob = prob

    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target

        _, h, w = image.shape
        scale = random.uniform(*self.scale_range)
        new_h = int(h * scale)
        new_w = int(w * scale)

        image = torch.nn.functional.interpolate(
            image.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
        ).squeeze(0)

        if len(target["masks"]) > 0:
            masks = target["masks"].float().unsqueeze(1)
            masks = torch.nn.functional.interpolate(
                masks, size=(new_h, new_w), mode="nearest"
            ).squeeze(1).bool()
            target["masks"] = masks

        if len(target["boxes"]) > 0:
            target["boxes"] = target["boxes"] * scale
            target["boxes"][:, [0, 2]] = target["boxes"][:, [0, 2]].clamp(0, new_w)
            target["boxes"][:, [1, 3]] = target["boxes"][:, [1, 3]].clamp(0, new_h)

        if len(target["area"]) > 0:
            target["area"] = target["area"] * (scale ** 2)

        return image, target


def get_train_transforms():
    return Compose([
        RandomHorizontalFlip(prob=0.5),
        RandomVerticalFlip(prob=0.5),
        RandomRotation90(prob=0.5),
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
        RandomGaussianNoise(std=0.02, prob=0.3),
        # RandomScaleCrop removed — causes RAM OOM on large images
    ])


def get_val_transforms():
    return None