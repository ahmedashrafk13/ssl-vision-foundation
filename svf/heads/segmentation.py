"""Segmentation heads: DeepLabv3 (ASPP) for CNNs, SegFormer-style MLP decoder for ViT/multi-scale."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..models.resnet import conv_bn


class ASPP(nn.Module):
    """Parallel atrous convolutions at several rates + image pooling branch."""

    def __init__(self, cin, cout=256, rates=(6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([conv_bn(cin, cout, 1, p=0)] +
                                      [conv_bn(cin, cout, 3, dilation=r) for r in rates])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(cin, cout, 1, bias=False), nn.ReLU(inplace=True))
        self.project = nn.Sequential(conv_bn(cout * (len(rates) + 2), cout, 1, p=0), nn.Dropout(0.1))

    def forward(self, x):
        outs = [b(x) for b in self.branches]
        outs.append(F.interpolate(self.pool(x), size=x.shape[-2:], mode="bilinear", align_corners=False))
        return self.project(torch.cat(outs, 1))


class DeepLabV3(nn.Module):
    def __init__(self, backbone, num_classes, rates=(6, 12, 18)):
        super().__init__()
        self.backbone = backbone
        self.aspp = ASPP(backbone.out_dim, 256, rates)
        self.classifier = nn.Sequential(conv_bn(256, 256), nn.Conv2d(256, num_classes, 1))

    def forward(self, x):
        f = self.backbone.forward_features(x)["c5"]
        return F.interpolate(self.classifier(self.aspp(f)), size=x.shape[-2:], mode="bilinear", align_corners=False)


class SegFormerHead(nn.Module):
    """All-MLP decoder: project each feature level, upsample to the finest, fuse, classify.
    With a single-scale ViT, a simple feature pyramid (deconv/identity/pool) is built first."""

    def __init__(self, backbone, num_classes, embed=256):
        super().__init__()
        self.backbone = backbone
        self.single_scale = len(backbone.channels) == 1
        if self.single_scale:
            d = backbone.out_dim
            self.pyramid = nn.ModuleList([
                nn.Sequential(nn.ConvTranspose2d(d, d, 2, 2), nn.GELU(), nn.ConvTranspose2d(d, d, 2, 2)),
                nn.ConvTranspose2d(d, d, 2, 2), nn.Identity(), nn.MaxPool2d(2)])
            chans = [d] * 4
        else:
            chans = backbone.channels
        self.proj = nn.ModuleList([nn.Conv2d(c, embed, 1) for c in chans])
        self.fuse = conv_bn(embed * len(chans), embed, 1, p=0)
        self.cls = nn.Conv2d(embed, num_classes, 1)

    def forward(self, x):
        f = self.backbone.forward_features(x)
        feats = [p(f["c5"]) for p in self.pyramid] if self.single_scale else list(f.values())
        size = feats[0].shape[-2:]
        y = torch.cat([F.interpolate(p(t), size=size, mode="bilinear", align_corners=False)
                       for p, t in zip(self.proj, feats)], 1)
        return F.interpolate(self.cls(self.fuse(y)), size=x.shape[-2:], mode="bilinear", align_corners=False)


def dice_loss(logits, target, eps=1.0):
    """Soft multi-class Dice; target is (B, H, W) long."""
    probs = logits.softmax(1)
    onehot = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).float()
    inter = (probs * onehot).sum((0, 2, 3))
    denom = probs.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    return 1 - ((2 * inter + eps) / (denom + eps)).mean()
