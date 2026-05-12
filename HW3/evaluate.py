import argparse
import json
import tempfile

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as coco_mask
from tqdm import tqdm

from dataset import CellDataset, CLASS_NAMES
from train import build_model


def build_gt_coco(val_dataset: CellDataset) -> dict:
    """Convert val dataset to COCO-format ground-truth dict."""
    categories = [
        {"id": idx + 1, "name": name, "supercategory": "cell"}
        for idx, name in enumerate(CLASS_NAMES)
    ]
    images, annotations = [], []
    ann_id = 1

    for i in range(len(val_dataset)):
        img, target = val_dataset[i]
        h, w = img.shape[-2], img.shape[-1]
        image_id = int(target["image_id"].item())
        images.append({"id": image_id, "height": h, "width": w, "file_name": str(i)})

        masks  = target["masks"].numpy()
        labels = target["labels"].numpy()
        boxes  = target["boxes"].numpy()

        for j in range(len(labels)):
            binary = np.asfortranarray(masks[j].astype(np.uint8))
            rle = coco_mask.encode(binary)
            rle["counts"] = rle["counts"].decode("utf-8")
            x1, y1, x2, y2 = boxes[j].tolist()
            annotations.append({
                "id":          ann_id,
                "image_id":    image_id,
                "category_id": int(labels[j]),
                "segmentation": rle,
                "bbox":        [x1, y1, x2 - x1, y2 - y1],
                "area":        float((x2 - x1) * (y2 - y1)),
                "iscrowd":     0,
            })
            ann_id += 1

    return {"images": images, "annotations": annotations, "categories": categories}


@torch.no_grad()
def run_val_predictions(model, val_dataset, device, score_thresh=0.03):
    model.eval()
    model.roi_heads.score_thresh = score_thresh
    results = []

    for i in tqdm(range(len(val_dataset)), desc="Val inference"):
        img, target = val_dataset[i]
        image_id = int(target["image_id"].item())
        outputs = model([img.to(device)])[0]

        masks  = outputs["masks"].squeeze(1).cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()
        boxes  = outputs["boxes"].cpu().numpy()

        for j in range(len(scores)):
            binary = masks[j] > 0.5
            if binary.sum() < 5:
                continue
            m = np.asfortranarray(binary.astype(np.uint8))
            rle = coco_mask.encode(m)
            rle["counts"] = rle["counts"].decode("utf-8")
            x1, y1, x2, y2 = boxes[j].tolist()
            results.append({
                "image_id":    image_id,
                "category_id": int(labels[j]),
                "segmentation": rle,
                "bbox":        [x1, y1, x2 - x1, y2 - y1],
                "score":       float(scores[j]),
            })

    return results


def evaluate(gt_dict: dict, pred_list: list) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(gt_dict, f)
        gt_path = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(pred_list, f)
        pred_path = f.name

    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(pred_path)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    ap50 = coco_eval.stats[1]
    print("\n" + "=" * 50)
    print(f"  AP50 (submission metric): {ap50:.4f}")
    print("=" * 50)
    if ap50 >= 0.35:
        print("🎉  Strong baseline achieved!")
    elif ap50 >= 0.25:
        print("✅  Above weak baseline.")
    else:
        print("⚠️   Below weak baseline.")


def main(args):
    val_ds = CellDataset(args.data_root, split="val", transforms=None)

    if args.pred_json:
        with open(args.pred_json) as f:
            pred_list = json.load(f)
        evaluate(build_gt_coco(val_ds), pred_list)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=5)
    state = torch.load(args.checkpoint, map_location=device)
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.to(device)

    gt_dict = build_gt_coco(val_ds)
    pred_list = run_val_predictions(model, val_ds, device, args.score_thresh)
    evaluate(gt_dict, pred_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",    type=str, required=True)
    parser.add_argument("--checkpoint",   type=str, default=None)
    parser.add_argument("--pred_json",    type=str, default=None)
    parser.add_argument("--score_thresh", type=float, default=0.03)
    args = parser.parse_args()

    if args.checkpoint is None and args.pred_json is None:
        parser.error("Provide either --checkpoint or --pred_json")

    main(args)