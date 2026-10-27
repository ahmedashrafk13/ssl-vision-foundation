"""Depthwise-separable (MobileNet) and compound-scaled MBConv (EfficientNet-style) backbones."""
import math
import torch.nn as nn
from .resnet import conv_bn


class DepthwiseSeparable(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.dw = conv_bn(cin, cin, 3, stride, groups=cin)
        self.pw = conv_bn(cin, cout, 1, 1, 0)

    def forward(self, x):
        return self.pw(self.dw(x))


class SqueezeExcite(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        m = max(1, c // r)
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, m, 1), nn.SiLU(),
                                nn.Conv2d(m, c, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x)


class MBConv(nn.Module):
    def __init__(self, cin, cout, stride=1, expand=4):
        super().__init__()
        mid = cin * expand
        self.block = nn.Sequential(
            nn.Conv2d(cin, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.SiLU(),
            nn.Conv2d(mid, mid, 3, stride, 1, groups=mid, bias=False), nn.BatchNorm2d(mid), nn.SiLU(),
            SqueezeExcite(mid),
            nn.Conv2d(mid, cout, 1, bias=False), nn.BatchNorm2d(cout))
        self.residual = stride == 1 and cin == cout

    def forward(self, x):
        y = self.block(x)
        return x + y if self.residual else y


class MobileNetLite(nn.Module):
    def __init__(self, width=1.0):
        super().__init__()
        cfg = [(32, 64, 1), (64, 128, 2), (128, 128, 1), (128, 256, 2), (256, 256, 1), (256, 512, 2)]
        c = lambda v: int(v * width)
        self.stem = conv_bn(3, c(32), 3, 2)
        self.body = nn.Sequential(*[DepthwiseSeparable(c(a), c(b), s) for a, b, s in cfg])
        self.out_dim = c(512)

    def forward(self, x):
        return self.body(self.stem(x)).mean((2, 3))


class EfficientNetLite(nn.Module):
    """Compound scaling: depth=1.2^phi, width=1.1^phi, resolution=1.15^phi."""
    base = [(16, 24, 2, 2), (24, 40, 2, 2), (40, 80, 2, 3), (80, 112, 1, 3), (112, 192, 2, 4)]

    def __init__(self, phi=0):
        super().__init__()
        d, w = 1.2 ** phi, 1.1 ** phi
        self.resolution = int(224 * 1.15 ** phi)
        c = lambda v: int(math.ceil(v * w / 8) * 8)
        self.stem = conv_bn(3, c(16), 3, 2)
        blocks = []
        for cin, cout, s, n in self.base:
            for i in range(int(math.ceil(n * d))):
                blocks.append(MBConv(c(cin) if i == 0 else c(cout), c(cout), s if i == 0 else 1))
        self.body = nn.Sequential(*blocks)
        self.out_dim = c(192)

    def forward(self, x):
        return self.body(self.stem(x)).mean((2, 3))
