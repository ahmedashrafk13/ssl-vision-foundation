"""One place that maps config names -> backbones / heads, so every experiment shares the same heads."""
import torch.nn as nn
import torchvision
from .models.resnet import resnet18, resnet34, resnet50
from .models.vit import vit_tiny, vit_small
from .heads.segmentation import DeepLabV3, SegFormerHead
from .heads.retinanet import RetinaNet
from .heads.detr import DETR


class TorchvisionResNet(nn.Module):
    """Supervised ImageNet baseline wrapped to the same interface as our ResNet."""

    def __init__(self, depth=50, pretrained=True):
        super().__init__()
        fn = getattr(torchvision.models, f"resnet{depth}")
        m = fn(weights="DEFAULT" if pretrained else None)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.stages = nn.ModuleList([m.layer1, m.layer2, m.layer3, m.layer4])
        self.channels = [64, 128, 256, 512] if depth < 50 else [256, 512, 1024, 2048]
        self.out_dim = self.channels[-1]

    def forward_features(self, x):
        x = self.stem(x)
        out = {}
        for i, s in enumerate(self.stages):
            x = s(x)
            out[f"c{i + 2}"] = x
        return out

    def forward(self, x):
        return self.forward_features(x)["c5"].mean((2, 3))


def build_backbone(name, img_size=224, **kw):
    return {
        "resnet18": lambda: resnet18(small_input=img_size <= 64, **kw),
        "resnet34": lambda: resnet34(small_input=img_size <= 64, **kw),
        "resnet50": lambda: resnet50(small_input=img_size <= 64, **kw),
        "vit_tiny": lambda: vit_tiny(img_size=img_size, patch=16 if img_size >= 128 else 4),
        "vit_small": lambda: vit_small(img_size=img_size, patch=16 if img_size >= 128 else 4),
        "sup_resnet50": lambda: TorchvisionResNet(50),
        "sup_resnet18": lambda: TorchvisionResNet(18),
    }[name]()


def build_seg(backbone, num_classes):
    """CNN -> DeepLabv3 (ASPP); ViT -> SegFormer-style decoder."""
    return SegFormerHead(backbone, num_classes) if len(backbone.channels) == 1 else DeepLabV3(backbone, num_classes)


def build_det(backbone, num_classes, kind=None):
    kind = kind or ("detr" if len(backbone.channels) == 1 else "retinanet")
    return DETR(backbone, num_classes) if kind == "detr" else RetinaNet(backbone, num_classes)
