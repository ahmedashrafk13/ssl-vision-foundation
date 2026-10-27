"""Evaluation metrics: top-1, Dice / mIoU, COCO-style mAP@0.5:0.95 with per-class AP."""
import numpy as np
import torch
from torchvision.ops import box_iou


def top1(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


class SegMeter:
    def __init__(self, num_classes):
        self.n = num_classes
        self.cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    def update(self, logits, target):
        p = logits.argmax(1).flatten().cpu()
        t = target.flatten().cpu()
        self.cm += torch.bincount(t * self.n + p, minlength=self.n ** 2).reshape(self.n, self.n)

    def compute(self):
        cm = self.cm.double()
        tp = cm.diag()
        fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
        iou = tp / (tp + fp + fn).clamp(min=1)
        dice = 2 * tp / (2 * tp + fp + fn).clamp(min=1)
        return {"mIoU": iou.mean().item(), "dice": dice.mean().item(),
                "dice_fg": dice[1:].mean().item() if self.n > 1 else dice.mean().item(),
                "per_class_iou": iou.tolist()}


def _ap(rec, prec):
    """101-point interpolated AP (COCO)."""
    mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    mrec = np.concatenate([[0], rec, [1]])
    pts = np.linspace(0, 1, 101)
    return np.mean([mpre[np.searchsorted(mrec, p, side="left")] if p <= rec[-1] else 0 for p in pts]) if len(rec) else 0.0


def mean_ap(preds, targets, num_classes, iou_thrs=np.arange(0.5, 0.96, 0.05)):
    """preds/targets: lists of dicts with boxes (xyxy), labels, scores (preds only)."""
    aps = np.zeros((len(iou_thrs), num_classes))
    valid = np.zeros(num_classes, bool)
    for c in range(num_classes):
        dets, n_gt = [], 0
        gts = []
        for i, (p, t) in enumerate(zip(preds, targets)):
            g = t["boxes"][t["labels"] == c]
            gts.append(g)
            n_gt += len(g)
            m = p["labels"] == c
            for b, s in zip(p["boxes"][m], p["scores"][m]):
                dets.append((float(s), i, b))
        if n_gt == 0:
            continue
        valid[c] = True
        dets.sort(key=lambda d: -d[0])
        for ti, thr in enumerate(iou_thrs):
            used = [torch.zeros(len(g), dtype=torch.bool) for g in gts]
            tp = np.zeros(len(dets))
            for k, (_, i, b) in enumerate(dets):
                if len(gts[i]) == 0:
                    continue
                iou = box_iou(b[None].float(), gts[i].float())[0]
                iou[used[i]] = -1
                j = int(iou.argmax())
                if iou[j] >= thr:
                    used[i][j] = True
                    tp[k] = 1
            ctp = np.cumsum(tp)
            rec = ctp / n_gt
            prec = ctp / np.arange(1, len(dets) + 1)
            aps[ti, c] = _ap(rec, prec)
    per_class = aps.mean(0)
    return {"mAP": float(per_class[valid].mean()) if valid.any() else 0.0,
            "mAP50": float(aps[0, valid].mean()) if valid.any() else 0.0,
            "per_class_AP": {c: float(per_class[c]) for c in range(num_classes) if valid[c]}}
