import argparse
import json

import numpy as np
import torch
from tqdm import tqdm
from pycocotools import mask as coco_mask

from dataset import CellTestDataset
from train import build_model


def encode_mask(binary_mask: np.ndarray) -> dict:
    """Encode boolean HxW mask to COCO RLE string."""
    m = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = coco_mask.encode(m)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def merge_with_nms(outputs_list: list, iou_threshold: float = 0.4) -> dict:
    """Merge multiple prediction dicts and apply NMS."""
    from torchvision.ops import nms

    merged = {}
    for key in ["boxes", "labels", "scores", "masks"]:
        merged[key] = torch.cat([o[key] for o in outputs_list], dim=0)

    if len(merged["boxes"]) == 0:
        return merged

    keep = nms(merged["boxes"], merged["scores"], iou_threshold=iou_threshold)
    for key in ["boxes", "labels", "scores", "masks"]:
        merged[key] = merged[key][keep]
    return merged


def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ──
    model = build_model(num_classes=5)
    state_dict = torch.load(args.checkpoint, map_location=device)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    model.roi_heads.score_thresh = args.score_thresh
    model.roi_heads.nms_thresh = 0.5
    model.roi_heads.detections_per_img = 300

    # ── Dataset ──
    test_ds = CellTestDataset(args.test_dir, args.id_map)
    print(f"Test images: {len(test_ds)}")

    results = []

    with torch.no_grad():
        for img_tensor, image_id, fname in tqdm(test_ds, desc="Inference"):
            if image_id == -1:
                print(f"[WARN] {fname} not in id map, skipping.")
                continue

            img = img_tensor.to(device)
            H, W = img.shape[-2], img.shape[-1]

            # Original prediction
            outputs = model([img])[0]
            tta_preds = [outputs]

            if args.tta:
                # Horizontal flip
                out_h = model([img.flip(-1)])[0]
                if len(out_h["boxes"]) > 0:
                    out_h["boxes"][:, [0, 2]] = W - out_h["boxes"][:, [2, 0]]
                    out_h["masks"] = out_h["masks"].flip(-1)
                tta_preds.append(out_h)

                # Vertical flip
                out_v = model([img.flip(-2)])[0]
                if len(out_v["boxes"]) > 0:
                    out_v["boxes"][:, [1, 3]] = H - out_v["boxes"][:, [3, 1]]
                    out_v["masks"] = out_v["masks"].flip(-2)
                tta_preds.append(out_v)

                outputs = merge_with_nms(tta_preds, iou_threshold=0.4)

            masks  = outputs["masks"].squeeze(1).cpu().numpy()   # [N, H, W]
            scores = outputs["scores"].cpu().numpy()
            labels = outputs["labels"].cpu().numpy()
            boxes  = outputs["boxes"].cpu().numpy()

            for i in range(len(scores)):
                binary_mask = masks[i] > 0.5
                if binary_mask.sum() < 5:
                    continue
                rle = encode_mask(binary_mask)
                x1, y1, x2, y2 = boxes[i].tolist()
                results.append({
                    "image_id":    int(image_id),
                    "category_id": int(labels[i]),   # 1-indexed from dataset
                    "segmentation": rle,
                    "bbox":        [x1, y1, x2 - x1, y2 - y1],
                    "score":       float(scores[i]),
                })

    with open(args.output, "w") as f:
        json.dump(results, f)

    print(f"\nSaved {len(results)} predictions → {args.output}")
    print(f"Pack: zip submission.zip {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir",     type=str,   required=True)
    parser.add_argument("--id_map",       type=str,   required=True)
    parser.add_argument("--checkpoint",   type=str,   required=True)
    parser.add_argument("--output",       type=str,   default="test-results.json")
    parser.add_argument("--score_thresh", type=float, default=0.03)
    parser.add_argument("--tta",          action="store_true",
                        help="Test-time augmentation (h-flip + v-flip)")
    args = parser.parse_args()
    run_inference(args)