"""
DETR Loss: Hungarian matching + classification + bbox regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Box utilities (cxcywh <-> xyxy, GIoU)
# ---------------------------------------------------------------------------

def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h,
                        cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def box_xyxy_to_cxcywh(boxes):
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2,
                        x2 - x1, y2 - y1], dim=-1)


def generalized_box_iou(boxes1, boxes2):
    """
    Compute GIoU between two sets of boxes.
    boxes: (N, 4) in xyxy format
    Returns: (N, M) matrix
    """
    # Intersection area
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    # Union area
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter

    iou = inter / union.clamp(min=1e-6)

    # Enclosing box
    lt_enc = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_enc = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh_enc = (rb_enc - lt_enc).clamp(min=0)
    enc_area = wh_enc[..., 0] * wh_enc[..., 1]

    giou = iou - (enc_area - union) / enc_area.clamp(min=1e-6)
    return giou


def box_l1(boxes1, boxes2):
    """Element-wise L1 between matched boxes."""
    return F.l1_loss(boxes1, boxes2, reduction="none").sum(-1)


# ---------------------------------------------------------------------------
# Hungarian Matcher
# ---------------------------------------------------------------------------

class HungarianMatcher(nn.Module):
    """
    Assigns predictions to ground-truth targets using the Hungarian algorithm.
    """

    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        outputs:
            pred_logits: (B, num_queries, num_classes+1)
            pred_boxes:  (B, num_queries, 4)  — cxcywh normalised [0,1]
        targets: list of dicts with:
            labels: (N,)
            boxes:  (N, 4) — xyxy in pixels
            orig_size: (2,) = [H, W]
        Returns: list of (row_idx, col_idx) pairs per image
        """
        B, Q = outputs["pred_logits"].shape[:2]
        pred_logits = outputs["pred_logits"].flatten(0, 1)  # (B*Q, C+1)
        pred_boxes = outputs["pred_boxes"].flatten(0, 1)    # (B*Q, 4)

        # Softmax over classes (exclude no-object for matching cost)
        out_prob = pred_logits.softmax(-1)

        indices = []
        offset = 0
        for b in range(B):
            tgt = targets[b]
            num_gt = len(tgt["labels"])
            if num_gt == 0:
                indices.append((torch.tensor([], dtype=torch.int64),
                                 torch.tensor([], dtype=torch.int64)))
                offset += Q
                continue

            # Normalize GT boxes to [0,1] cxcywh
            h, w = tgt["orig_size"].tolist()
            gt_boxes = tgt["boxes"].clone()                        # xyxy pixels
            gt_boxes_cxcywh = box_xyxy_to_cxcywh(gt_boxes)
            gt_boxes_cxcywh /= torch.tensor([w, h, w, h],
                                             dtype=torch.float32,
                                             device=gt_boxes.device)

            pred_b = pred_boxes[offset: offset + Q]
            prob_b = out_prob[offset: offset + Q]

            # Class cost: negative probability of true class
            tgt_labels = tgt["labels"] - 1  # 0-indexed
            cost_cls = -prob_b[:, tgt_labels]  # (Q, N)

            # BBox L1 cost
            cost_b = torch.cdist(pred_b, gt_boxes_cxcywh, p=1)  # (Q, N)

            # GIoU cost
            pred_xyxy = box_cxcywh_to_xyxy(pred_b)
            gt_xyxy = box_cxcywh_to_xyxy(gt_boxes_cxcywh)
            cost_g = -generalized_box_iou(pred_xyxy, gt_xyxy)   # (Q, N)

            C = (self.cost_class * cost_cls +
                 self.cost_bbox * cost_b +
                 self.cost_giou * cost_g)
            # Guard against nan/inf from fp16 overflow
            C = torch.nan_to_num(C, nan=1e4, posinf=1e4, neginf=-1e4)
            C = C.cpu().numpy()

            row, col = linear_sum_assignment(C)
            indices.append((torch.as_tensor(row, dtype=torch.int64),
                             torch.as_tensor(col, dtype=torch.int64)))
            offset += Q

        return indices


# ---------------------------------------------------------------------------
# DETR Loss
# ---------------------------------------------------------------------------

class DETRLoss(nn.Module):
    """Combined classification + L1 + GIoU loss."""

    def __init__(self, num_classes=10,
                 cost_class=1.0, cost_bbox=5.0, cost_giou=2.0,
                 weight_class=1.0, weight_bbox=5.0, weight_giou=2.0,
                 eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher(cost_class, cost_bbox, cost_giou)
        self.weight_class = weight_class
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou

        # Down-weight the "no object" class
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices):
        pred_logits = outputs["pred_logits"]  # (B, Q, C+1)
        B, Q, _ = pred_logits.shape

        # Default target: last class = no-object
        target_classes = torch.full((B, Q), self.num_classes,
                                    dtype=torch.int64,
                                    device=pred_logits.device)
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            labels = targets[b]["labels"][tgt_idx] - 1  # 0-indexed
            target_classes[b, src_idx] = labels

        # Cast to fp32 for numerical stability
        loss = F.cross_entropy(
            pred_logits.flatten(0, 1).float(),
            target_classes.flatten(),
            weight=self.empty_weight.float(),
        )
        return loss

    def loss_boxes(self, outputs, targets, indices):
        pred_boxes = outputs["pred_boxes"]  # (B, Q, 4) cxcywh [0,1]
        B = pred_boxes.shape[0]

        src_boxes_list, tgt_boxes_list = [], []
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            src_boxes_list.append(pred_boxes[b][src_idx])

            h, w = targets[b]["orig_size"].tolist()
            gt_xyxy = targets[b]["boxes"][tgt_idx].clone()
            gt_cxcywh = box_xyxy_to_cxcywh(gt_xyxy)
            gt_cxcywh /= torch.tensor([w, h, w, h],
                                       dtype=torch.float32,
                                       device=gt_xyxy.device)
            tgt_boxes_list.append(gt_cxcywh)

        if not src_boxes_list:
            return pred_boxes.sum() * 0.0, pred_boxes.sum() * 0.0

        src_boxes = torch.cat(src_boxes_list)
        tgt_boxes = torch.cat(tgt_boxes_list)

        num_boxes = src_boxes.shape[0]
        loss_l1 = F.l1_loss(src_boxes, tgt_boxes, reduction="sum") / num_boxes

        src_xyxy = box_cxcywh_to_xyxy(src_boxes)
        tgt_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
        giou = generalized_box_iou(src_xyxy, tgt_xyxy)
        loss_giou = (1 - giou.diag()).sum() / num_boxes

        return loss_l1, loss_giou

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        loss_cls = self.loss_labels(outputs, targets, indices)
        loss_l1, loss_giou = self.loss_boxes(outputs, targets, indices)

        total = (self.weight_class * loss_cls +
                 self.weight_bbox * loss_l1 +
                 self.weight_giou * loss_giou)

        # Auxiliary losses (intermediate decoder layers)
        if "aux_outputs" in outputs:
            for aux in outputs["aux_outputs"]:
                aux_indices = self.matcher(aux, targets)
                aux_cls = self.loss_labels(aux, targets, aux_indices)
                aux_l1, aux_giou = self.loss_boxes(aux, targets, aux_indices)
                total = total + (self.weight_class * aux_cls +
                                 self.weight_bbox * aux_l1 +
                                 self.weight_giou * aux_giou)

        return {
            "loss_total": total,
            "loss_cls": loss_cls.detach(),
            "loss_l1": loss_l1.detach(),
            "loss_giou": loss_giou.detach(),
        }