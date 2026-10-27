"""ResNet written from scratch (no torchvision model zoo)."""
import torch.nn as nn


def conv_bn(cin, cout, k=3, s=1, p=None, groups=1, dilation=1, act=True):
    p = (k // 2) * dilation if p is None else p
    layers = [nn.Conv2d(cin, cout, k, s, p, groups=groups, dilation=dilation, bias=False),
              nn.BatchNorm2d(cout)]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin, planes, stride=1, dilation=1):
        super().__init__()
        cout = planes * self.expansion
        self.conv1 = conv_bn(cin, planes, 3, stride, dilation=dilation)
        self.conv2 = conv_bn(planes, cout, 3, 1, dilation=dilation, act=False)
        # downsampling shortcut when shape changes
        self.shortcut = (nn.Identity() if stride == 1 and cin == cout
                         else conv_bn(cin, cout, 1, stride, 0, act=False))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv2(self.conv1(x)) + self.shortcut(x))


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, cin, planes, stride=1, dilation=1):
        super().__init__()
        cout = planes * self.expansion
        self.conv1 = conv_bn(cin, planes, 1, 1, 0)
        self.conv2 = conv_bn(planes, planes, 3, stride, dilation=dilation)
        self.conv3 = conv_bn(planes, cout, 1, 1, 0, act=False)
        nn.init.zeros_(self.conv3[1].weight)  # zero-init last BN: block starts as identity
        self.shortcut = (nn.Identity() if stride == 1 and cin == cout
                         else conv_bn(cin, cout, 1, stride, 0, act=False))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv3(self.conv2(self.conv1(x))) + self.shortcut(x))


class ResNet(nn.Module):
    """Returns multi-scale features {c2..c5} via forward_features; forward gives the pooled embedding.

    small_input=True uses a 3x3 stem without maxpool (CIFAR-sized images).
    output_stride=8/16 swaps strides for dilation in the last stages (DeepLabv3).
    """

    def __init__(self, block, layers, small_input=False, output_stride=32, width=64):
        super().__init__()
        if small_input:
            self.stem = conv_bn(3, width, 3, 1)
        else:
            self.stem = nn.Sequential(conv_bn(3, width, 7, 2, 3), nn.MaxPool2d(3, 2, 1))
        strides, dilations = [1, 2, 2, 2], [1, 1, 1, 1]
        if output_stride <= 16:
            strides[3], dilations[3] = 1, 2
        if output_stride == 8:
            strides[2], dilations[2], dilations[3] = 1, 2, 4
        cin = width
        self.stages = nn.ModuleList()
        self.channels = []
        for i, n in enumerate(layers):
            planes = width * 2 ** i
            blocks = []
            for j in range(n):
                blocks.append(block(cin, planes, strides[i] if j == 0 else 1, dilations[i]))
                cin = planes * block.expansion
            self.stages.append(nn.Sequential(*blocks))
            self.channels.append(cin)
        self.out_dim = cin

    def forward_features(self, x):
        x = self.stem(x)
        feats = {}
        for i, s in enumerate(self.stages):
            x = s(x)
            feats[f"c{i + 2}"] = x
        return feats

    def forward(self, x):
        return self.forward_features(x)["c5"].mean((2, 3))


def resnet18(**kw):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kw)


def resnet34(**kw):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kw)


def resnet50(**kw):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kw)
