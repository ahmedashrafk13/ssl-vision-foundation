"""SimCLR: two augmented views, projection head, NT-Xent loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


def simclr_augment(size):
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(0.2),
        T.RandomApply([T.GaussianBlur(max(3, size // 10 | 1))], p=0.5),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


class TwoViews:
    def __init__(self, tf):
        self.tf = tf

    def __call__(self, img):
        return self.tf(img), self.tf(img)


def nt_xent(z1, z2, temperature=0.2):
    """Normalized temperature-scaled cross entropy over 2N views; positives are the paired view."""
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    n = z1.shape[0]
    sim = z @ z.t() / temperature
    sim.fill_diagonal_(float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


class SimCLR(nn.Module):
    def __init__(self, backbone, proj_dim=128, hidden=2048):
        super().__init__()
        self.backbone = backbone
        d = backbone.out_dim
        self.projector = nn.Sequential(nn.Linear(d, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
                                       nn.Linear(hidden, proj_dim))

    def forward(self, x1, x2, temperature=0.2):
        z1, z2 = self.projector(self.backbone(x1)), self.projector(self.backbone(x2))
        return nt_xent(z1, z2, temperature)
