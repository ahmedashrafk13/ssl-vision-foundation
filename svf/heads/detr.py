"""DETR-style detector: backbone features -> Transformer enc-dec with learned object queries,
class + box heads, Hungarian matching loss (CE + L1 + GIoU)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou, box_convert
from ..models.vit import sincos_pos_embed


class DETR(nn.Module):
    def __init__(self, backbone, num_classes, d=256, queries=100, enc_layers=3, dec_layers=3, heads=8):
        super().__init__()
        self.backbone, self.num_classes = backbone, num_classes
        self.input_proj = nn.Conv2d(backbone.out_dim, d, 1)
        self.transformer = nn.Transformer(d, heads, enc_layers, dec_layers, d * 4, 0.1, batch_first=True)
        self.query = nn.Embedding(queries, d)
        self.cls_head = nn.Linear(d, num_classes + 1)  # +1 = "no object"
        self.box_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 4))
        self.d = d

    def forward(self, x):
        f = self.input_proj(self.backbone.forward_features(x)["c5"])
        b, d, h, w = f.shape
        assert h == w, "square feature maps expected"
        pos = sincos_pos_embed(d, h).to(f)[None]
        src = f.flatten(2).transpose(1, 2) + pos
        tgt = self.query.weight[None].expand(b, -1, -1)
        hs = self.transformer(src, tgt)
        return self.cls_head(hs), self.box_head(hs).sigmoid()  # boxes: normalized cxcywh


@torch.no_grad()
def hungarian_match(logits, boxes, targets, w_cls=1.0, w_l1=5.0, w_giou=2.0):
    """targets: list of {'labels', 'boxes' normalized cxcywh}. Returns list of (pred_idx, gt_idx)."""
    out = []
    for lg, bx, t in zip(logits, boxes, targets):
        if len(t["labels"]) == 0:
            out.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
            continue
        prob = lg.softmax(-1)
        cost = (-w_cls * prob[:, t["labels"]] + w_l1 * torch.cdist(bx, t["boxes"], p=1)
                - w_giou * generalized_box_iou(box_convert(bx, "cxcywh", "xyxy"),
                                               box_convert(t["boxes"], "cxcywh", "xyxy")))
        r, c = linear_sum_assignment(cost.cpu().numpy())
        out.append((torch.as_tensor(r, dtype=torch.long), torch.as_tensor(c, dtype=torch.long)))
    return out


def detr_loss(logits, boxes, targets, eos_coef=0.1, w_l1=5.0, w_giou=2.0):
    nc = logits.shape[-1] - 1
    idx = hungarian_match(logits, boxes, targets)
    tgt_cls = torch.full(logits.shape[:2], nc, dtype=torch.long, device=logits.device)
    src_b, tgt_b = [], []
    for i, (r, c) in enumerate(idx):
        tgt_cls[i, r] = targets[i]["labels"][c]
        src_b.append(boxes[i, r])
        tgt_b.append(targets[i]["boxes"][c])
    weight = torch.ones(nc + 1, device=logits.device)
    weight[nc] = eos_coef
    loss_cls = F.cross_entropy(logits.transpose(1, 2), tgt_cls, weight)
    src_b, tgt_b = torch.cat(src_b), torch.cat(tgt_b)
    n = max(1, len(tgt_b))
    if len(tgt_b):
        l1 = F.l1_loss(src_b, tgt_b, reduction="sum") / n
        giou = (1 - torch.diag(generalized_box_iou(box_convert(src_b, "cxcywh", "xyxy"),
                                                   box_convert(tgt_b, "cxcywh", "xyxy")))).sum() / n
    else:
        l1 = giou = boxes.sum() * 0
    return {"cls": loss_cls, "l1": w_l1 * l1, "giou": w_giou * giou}


@torch.no_grad()
def detr_predict(logits, boxes, img_size, score_thr=0.3):
    prob = logits.softmax(-1)[..., :-1]
    scores, labels = prob.max(-1)
    out = []
    for s, l, b in zip(scores, labels, boxes):
        k = s > score_thr
        out.append({"boxes": box_convert(b[k], "cxcywh", "xyxy") * img_size, "scores": s[k], "labels": l[k]})
    return out
