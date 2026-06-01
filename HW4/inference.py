import os
import argparse
import zipfile
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

from dataset import RestoreTestDataset
from net.promptir import PromptIR
from utils.utils import tensor_to_uint8


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for HW4 Image Restoration")
    parser.add_argument("--ckpt", type=str, default="ckpt/best_model.pth")
    parser.add_argument("--test_dir", type=str, default="data/test")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--prompt_len", type=int, default=5)
    parser.add_argument("--prompt_size", type=int, default=32,
                        help="Must match training: patch_size // 4")
    parser.add_argument("--tile", action="store_true",
                        help="Use tiled inference for large images")
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--tile_overlap", type=int, default=32)
    return parser.parse_args()


def predict(model, x, device):
    """Single forward pass."""
    with torch.no_grad():
        return model(x.to(device)).clamp(0, 1).cpu()


def tta_inference(model, img_tensor, device):
    """
    Test-Time Augmentation: average predictions over 8 augmentations
    (original + hflip + vflip + hflip&vflip + 90rot x4).
    """
    augs = [
        lambda x: x,
        lambda x: TF.hflip(x),
        lambda x: TF.vflip(x),
        lambda x: TF.hflip(TF.vflip(x)),
        lambda x: torch.rot90(x, 1, [2, 3]),
        lambda x: torch.rot90(x, 2, [2, 3]),
        lambda x: torch.rot90(x, 3, [2, 3]),
        lambda x: TF.hflip(torch.rot90(x, 1, [2, 3])),
    ]
    inv_augs = [
        lambda x: x,
        lambda x: TF.hflip(x),
        lambda x: TF.vflip(x),
        lambda x: TF.hflip(TF.vflip(x)),
        lambda x: torch.rot90(x, -1, [2, 3]),
        lambda x: torch.rot90(x, -2, [2, 3]),
        lambda x: torch.rot90(x, -3, [2, 3]),
        lambda x: torch.rot90(TF.hflip(x), -1, [2, 3]),
    ]

    preds = []
    for aug, inv in zip(augs, inv_augs):
        augmented = aug(img_tensor)
        pred = predict(model, augmented, device)
        preds.append(inv(pred))

    return torch.stack(preds).mean(dim=0)


def tile_inference(model, img_tensor, tile_size, overlap, device):
    """Run model on large images by splitting into overlapping tiles."""
    _, c, h, w = img_tensor.shape
    stride = tile_size - overlap
    output = torch.zeros_like(img_tensor)
    count = torch.zeros(1, 1, h, w)

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)

            tile = img_tensor[:, :, y_start:y_end, x_start:x_end]
            pred_tile = predict(model, tile, device)

            output[:, :, y_start:y_end, x_start:x_end] += pred_tile
            count[:, :, y_start:y_end, x_start:x_end] += 1

    return output / count.clamp(min=1)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Load model ----
    model = PromptIR(
        dim=args.dim,
        num_blocks=(4, 6, 6, 8),
        num_refinement_blocks=4,
        heads=(1, 2, 4, 8),
        ffn_expansion_factor=2.66,
        bias=False,
        prompt_len=args.prompt_len,
        prompt_size=args.prompt_size,
        use_prompt=True,
    ).to(device)

    if not Path(args.ckpt).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt}")

    # ---- Dataset ----
    test_ds = RestoreTestDataset(args.test_dir)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

    # ---- Inference ----
    images_dict = {}
    for i, (img_tensor, fname_tuple) in enumerate(test_loader):
        fname = fname_tuple[0]

        if args.tile:
            pred = tile_inference(
                model, img_tensor, args.tile_size, args.tile_overlap, device
            )
        else:
            pred = tta_inference(model, img_tensor, device)

        img_np = tensor_to_uint8(pred[0])
        images_dict[fname] = img_np

        if (i + 1) % 10 == 0 or (i + 1) == len(test_ds):
            print(f"  Processed {i+1}/{len(test_ds)}: {fname}  shape={img_np.shape}")

    # ---- Save pred.npz ----
    npz_path = os.path.join(args.output_dir, "pred.npz")
    np.savez(npz_path, **images_dict)
    print(f"\nSaved {len(images_dict)} images to {npz_path}")

    # ---- Create submission.zip ----
    zip_path = os.path.join(args.output_dir, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(npz_path, arcname="pred.npz")
    print(f"Created submission archive: {zip_path}")
    print("\nDone! Upload submission.zip to CodaBench.")


if __name__ == "__main__":
    main()