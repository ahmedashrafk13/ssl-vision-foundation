"""Datasets: unlabelled pre-training pool (ImageNet-100 subset, CIFAR, MVTec AD), MVTec AD cls/seg, NEU-DET.

Expected layout under data/ (see README for download commands):
  data/imagenet100/train/<class>/*.JPEG
  data/mvtec/<category>/{train/good, test/<defect>, ground_truth/<defect>/*_mask.png}
  data/neu-det/{IMAGES/*.jpg, ANNOTATIONS/*.xml}
"""
import random
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, Subset
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
NEU_CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def _images(root):
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXT)


class ImageList(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = list(paths), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB")), 0


class _Unlabel(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[i][0], 0


def unlabelled_pool(root, transform, sources=("imagenet100", "cifar", "mvtec")):
    root = Path(root)
    parts = []
    if "imagenet100" in sources and (root / "imagenet100/train").exists():
        parts.append(ImageList(_images(root / "imagenet100/train"), transform))
    if "cifar" in sources:
        parts.append(_Unlabel(torchvision.datasets.CIFAR10(root / "cifar", train=True, download=True, transform=transform)))
    if "mvtec" in sources and (root / "mvtec").exists():
        # only train/good + test images; labels/masks are never touched during pre-training
        parts.append(ImageList([p for p in _images(root / "mvtec") if "ground_truth" not in p.parts], transform))
    if not parts:
        raise FileNotFoundError(f"no pre-training data found under {root}")
    return ConcatDataset(parts)


class MVTecSeg(Dataset):
    """Per-category MVTec AD. Classification label: 0 good / 1 defective. Seg mask: 0 bg / 1 defect.
    Uses the test split (the only split with defects), divided into train/val by a fixed seed."""

    def __init__(self, root, categories, split="train", size=224, val_frac=0.3, seed=0):
        self.size, self.train = size, split == "train"
        items = []
        for cat in categories:
            for p in _images(Path(root) / cat / "test"):
                defect = p.parent.name
                mask = None if defect == "good" else Path(root) / cat / "ground_truth" / defect / f"{p.stem}_mask.png"
                items.append((p, mask))
        rng = random.Random(seed)
        rng.shuffle(items)
        k = int(len(items) * val_frac)
        self.items = items[k:] if self.train else items[:k]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, m = self.items[i]
        img = Image.open(p).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        mask = (Image.open(m).convert("L").resize((self.size, self.size), Image.NEAREST) if m
                else Image.new("L", (self.size, self.size)))
        if self.train and random.random() < 0.5:
            img, mask = TF.hflip(img), TF.hflip(mask)
        x = TF.normalize(TF.to_tensor(img), MEAN, STD)
        y = torch.from_numpy((np.array(mask) > 127).astype(np.int64))
        return x, y, int(m is not None)


class NEUDet(Dataset):
    """NEU surface-defect detection (VOC XML). Returns image, {'boxes' xyxy px, 'labels'}."""

    def __init__(self, root, split="train", size=224, val_frac=0.2, seed=0):
        # works with both the flat VOC layout (IMAGES/, ANNOTATIONS/) and the Kaggle one
        # (train|validation/{images/<class>/, annotations/}); our own seeded split is used either way
        root = Path(root)
        xmls = sorted(root.rglob("*.xml"))
        self.images = {p.stem: p for p in _images(root)}
        random.Random(seed).shuffle(xmls)
        k = int(len(xmls) * val_frac)
        self.xmls = xmls[k:] if split == "train" else xmls[:k]
        self.size, self.train = size, split == "train"

    def __len__(self):
        return len(self.xmls)

    def __getitem__(self, i):
        tree = ET.parse(self.xmls[i]).getroot()
        img = Image.open(self.images[self.xmls[i].stem]).convert("RGB")
        w, h = img.size
        boxes, labels = [], []
        for o in tree.iter("object"):
            bb = o.find("bndbox")
            boxes.append([float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")])
            labels.append(NEU_CLASSES.index(o.findtext("name")))
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes *= torch.tensor([self.size / w, self.size / h] * 2)
        img = img.resize((self.size, self.size), Image.BILINEAR)
        if self.train and random.random() < 0.5:
            img = TF.hflip(img)
            boxes[:, [0, 2]] = self.size - boxes[:, [2, 0]]
        x = TF.normalize(TF.to_tensor(img), MEAN, STD)
        return x, {"boxes": boxes, "labels": torch.tensor(labels, dtype=torch.long)}


def det_collate(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


def label_fraction(ds, frac, seed=0):
    """Subset used for the data-efficiency curve (1%, 10%, 25%, 50%, 100% of labels)."""
    if frac >= 1:
        return ds
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    return Subset(ds, idx[:max(1, int(len(ds) * frac))])


def eval_transform(size):
    return T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(), T.Normalize(MEAN, STD)])
