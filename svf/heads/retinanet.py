"""FPN over CNN feature maps + RetinaNet head (focal loss, anchor boxes)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou, batched_nms, sigmoid_focal_loss


class FPN(nn.Module):
    def __init__(self, in_channels, out=256):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out, 1) for c in in_channels])
        self.smooth = nn.ModuleList([nn.Conv2d(out, out, 3, padding=1) for _ in in_channels])
        self.p6 = nn.Conv2d(out, out, 3, 2, 1)

    def forward(self, feats):
        lat = [l(f) for l, f in zip(self.lateral, feats)]
        for i in range(len(lat) - 1, 0, -1):  # top-down pathway
            lat[i - 1] = lat[i - 1] + F.interpolate(lat[i], size=lat[i - 1].shape[-2:], mode="nearest")
        outs = [s(l) for s, l in zip(self.smooth, lat)]
        outs.append(self.p6(outs[-1]))
        return outs


def make_anchors(feat_shapes, strides, img_size, device, scales=(1.0, 2 ** (1 / 3), 2 ** (2 / 3)),
                 ratios=(0.5, 1.0, 2.0)):
    all_a = []
    for (h, w), s in zip(feat_shapes, strides):
        base = 4 * s
        wh = torch.tensor([[base * sc * math.sqrt(1 / r), base * sc * math.sqrt(r)] for r in ratios for sc in scales],
                          device=device)
        ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        ctr = torch.stack([(xs + 0.5) * s, (ys + 0.5) * s], -1).reshape(-1, 1, 2)
        a = torch.cat([ctr - wh / 2, ctr + wh / 2], -1).reshape(-1, 4)
        all_a.append(a)
    return torch.cat(all_a)


def encode(anchors, boxes):
    aw, ah = anchors[:, 2] - anchors[:, 0], anchors[:, 3] - anchors[:, 1]
    ax, ay = anchors[:, 0] + aw / 2, anchors[:, 1] + ah / 2
    bw, bh = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    bx, by = boxes[:, 0] + bw / 2, boxes[:, 1] + bh / 2
    return torch.stack([(bx - ax) / aw, (by - ay) / ah, torch.log(bw / aw), torch.log(bh / ah)], 1)


def decode(anchors, d):
    aw, ah = anchors[:, 2] - anchors[:, 0], anchors[:, 3] - anchors[:, 1]
    ax, ay = anchors[:, 0] + aw / 2, anchors[:, 1] + ah / 2
    d = d.clone()
    d[:, 2:] = d[:, 2:].clamp(max=math.log(1000 / 16))
    cx, cy = ax + d[:, 0] * aw, ay + d[:, 1] * ah
    w, h = aw * d[:, 2].exp(), ah * d[:, 3].exp()
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)


class RetinaNet(nn.Module):
    """Backbone must expose forward_features -> {c3,c4,c5} (CNN) and .channels."""

    def __init__(self, backbone, num_classes, fpn_dim=256, num_anchors=9):
        super().__init__()
        self.backbone, self.num_classes = backbone, num_classes
        self.fpn = FPN(backbone.channels[1:], fpn_dim)
        self.strides = None

        def tower(out):
            layers = []
            for _ in range(4):
                layers += [nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1), nn.GroupNorm(32, fpn_dim), nn.ReLU(inplace=True)]
            layers.append(nn.Conv2d(fpn_dim, out, 3, padding=1))
            return nn.Sequential(*layers)

        self.cls_head = tower(num_anchors * num_classes)
        self.box_head = tower(num_anchors * 4)
        nn.init.constant_(self.cls_head[-1].bias, -math.log((1 - 0.01) / 0.01))  # prior p=0.01

    def forward(self, x):
        f = self.backbone.forward_features(x)
        p = self.fpn([f["c3"], f["c4"], f["c5"]])
        b = x.shape[0]
        cls = torch.cat([self.cls_head(t).permute(0, 2, 3, 1).reshape(b, -1, self.num_classes) for t in p], 1)
        box = torch.cat([self.box_head(t).permute(0, 2, 3, 1).reshape(b, -1, 4) for t in p], 1)
        strides = [x.shape[-1] // t.shape[-1] for t in p]
        anchors = make_anchors([t.shape[-2:] for t in p], strides, x.shape[-1], x.device)
        return cls, box, anchors

    def loss(self, cls, box, anchors, targets):
        """targets: list of dicts with 'boxes' (K,4 xyxy px) and 'labels' (K,)."""
        cls_l, box_l, npos = 0.0, 0.0, 0
        for i, t in enumerate(targets):
            gt = torch.zeros_like(cls[i])
            if len(t["boxes"]) == 0:
                cls_l = cls_l + sigmoid_focal_loss(cls[i], gt, reduction="sum")
                continue
            iou = box_iou(anchors, t["boxes"])
            best, idx = iou.max(1)
            best_anchor = iou.argmax(0)  # make sure every gt gets one anchor
            best[best_anchor] = 1.0
            idx[best_anchor] = torch.arange(len(best_anchor), device=idx.device)
            pos, ignore = best >= 0.5, (best >= 0.4) & (best < 0.5)
            gt[pos, t["labels"][idx[pos]]] = 1.0
            valid = ~ignore
            cls_l = cls_l + sigmoid_focal_loss(cls[i][valid], gt[valid], reduction="sum")
            if pos.any():
                box_l = box_l + F.smooth_l1_loss(box[i][pos], encode(anchors[pos], t["boxes"][idx[pos]]),
                                                 beta=1 / 9, reduction="sum")
            npos += int(pos.sum())
        n = max(1, npos)
        return {"cls": cls_l / n, "box": box_l / n}

    @torch.no_grad()
    def predict(self, x, score_thr=0.05, iou_thr=0.5, topk=100):
        cls, box, anchors = self(x)
        out = []
        for i in range(x.shape[0]):
            scores = cls[i].sigmoid()
            s, flat = scores.flatten().topk(min(1000, scores.numel()))
            keep = s > score_thr
            s, flat = s[keep], flat[keep]
            a_idx, labels = flat // self.num_classes, flat % self.num_classes
            boxes = decode(anchors[a_idx], box[i][a_idx]).clamp(0, x.shape[-1])
            k = batched_nms(boxes, s, labels, iou_thr)[:topk]
            out.append({"boxes": boxes[k], "scores": s[k], "labels": labels[k]})
        return out
