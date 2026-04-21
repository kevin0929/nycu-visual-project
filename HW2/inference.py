import argparse
import json
import os
import zipfile
import torch
from torch.utils.data import DataLoader
from torchvision.ops import nms

from dataset import DigitTestDataset, make_transforms, collate_fn
from criterion import box_cxcywh_to_xyxy


def get_args():
    p = argparse.ArgumentParser("DETR Inference")
    p.add_argument("--data_root",   type=str, required=True)
    p.add_argument("--test_dir",    type=str, default="test")
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--output",      type=str, default="pred.json")
    p.add_argument("--conf_thresh", type=float, default=0.5)
    p.add_argument("--nms_thresh",  type=float, default=0.5)
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


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

    class NT:
        def __init__(self, t, m):
            self.tensors = t
            self.mask = m
        def decompose(self):
            return self.tensors, self.mask

    return NT(batched, mask)


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    saved_args = ckpt.get("args", {})
    num_classes = saved_args.get("num_classes", 10)
    num_queries = saved_args.get("num_queries", 100)
    # model_type stored in ckpt directly (newer ckpts) or in args
    model_type = ckpt.get("model_type", saved_args.get("model", "official"))

    if model_type == "official":
        from train_official import build_model
        model = build_model(num_classes, num_queries)
    elif model_type == "vanilla":
        from train_vanilla import VanillaDETR
        model = VanillaDETR(
            num_classes=num_classes,
            num_queries=num_queries,
            d_model=saved_args.get("d_model", 256),
            nhead=saved_args.get("nhead", 8),
            num_encoder_layers=saved_args.get("num_encoder_layers", 6),
            num_decoder_layers=saved_args.get("num_decoder_layers", 6),
            dim_feedforward=saved_args.get("dim_feedforward", 2048),
            dropout=saved_args.get("dropout", 0.1),
            pretrained_backbone=False,
        )
    elif model_type in ("fast", None, ""):
        from train_fast import build_fast_model, DETRWrapper
        detr  = build_fast_model(num_classes, num_queries, pretrained=False)
        model = DETRWrapper(detr)
    else:  # deformable
        from deformable_detr import build_deformable_model
        model = build_deformable_model(saved_args)

    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"Loaded '{model_type}' model from {checkpoint_path}")
    return model


@torch.no_grad()
def run_inference(model, dataloader, device, conf_thresh, nms_thresh):
    results = []
    for images, targets in dataloader:
        samples = nested_tensor_from_tensor_list([img.to(device) for img in images])
        outputs = model(samples)

        pred_logits = outputs["pred_logits"]
        pred_boxes  = outputs["pred_boxes"]
        B = pred_logits.shape[0]

        for b in range(B):
            image_id = int(targets[b]["image_id"].item())
            orig_h, orig_w = targets[b]["orig_size"].tolist()

            prob   = pred_logits[b].softmax(-1)[:, :-1]
            scores, labels = prob.max(-1)
            keep   = scores > conf_thresh
            scores = scores[keep]
            labels = labels[keep]
            boxes  = pred_boxes[b][keep]

            if scores.numel() == 0:
                continue

            boxes_xyxy = box_cxcywh_to_xyxy(boxes)
            boxes_xyxy *= torch.tensor([orig_w, orig_h, orig_w, orig_h],
                                        dtype=torch.float32, device=boxes.device)
            boxes_xywh = boxes_xyxy.clone()
            boxes_xywh[:, 2] -= boxes_xyxy[:, 0]
            boxes_xywh[:, 3] -= boxes_xyxy[:, 1]

            for cls_id in labels.unique().tolist():
                m = labels == cls_id
                k = nms(boxes_xyxy[m], scores[m], nms_thresh)
                for idx in k:
                    bbox = boxes_xywh[m][idx].cpu().tolist()
                    results.append({
                        "image_id":    image_id,
                        "bbox":        [round(v, 4) for v in bbox],
                        "score":       round(float(scores[m][idx].item()), 6),
                        "category_id": int(cls_id) + 1,
                    })
    return results


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.checkpoint, device)

    test_dir = os.path.join(args.data_root, args.test_dir)
    ds = DigitTestDataset(test_dir, transforms=make_transforms("val"))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_fn,
                        pin_memory=True)
    print(f"Test images: {len(ds)}")

    results = run_inference(model, loader, device, args.conf_thresh, args.nms_thresh)
    print(f"Total predictions: {len(results)}")

    with open(args.output, "w") as f:
        json.dump(results, f)

    zip_path = args.output.replace(".json", ".zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(args.output, "pred.json")
    print(f"Saved: {args.output}  |  Zip: {zip_path}")


if __name__ == "__main__":
    main()