"""Linear probe / full fine-tune for classification, segmentation (MVTec AD) and detection (NEU-DET).

  python scripts/finetune.py --task cls --mode probe --backbone resnet18 --ckpt simclr_resnet18_96/backbone.pt
  python scripts/finetune.py --task seg --mode full  --backbone vit_tiny --ckpt mae_vit_tiny_224_m75/backbone.pt
  python scripts/finetune.py --task det --mode full  --backbone sup_resnet50 --label-frac 0.1
All results are appended to results/results.jsonl for the comparison plots.
"""
import argparse, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from common import load_backbone, log_result
from svf.builders import build_seg, build_det
from svf.data.datasets import MVTecSeg, NEUDet, NEU_CLASSES, det_collate, label_fraction, eval_transform
from svf.heads.segmentation import dice_loss
from svf.heads.detr import DETR, detr_loss, detr_predict
from svf.metrics import top1, SegMeter, mean_ap
from svf.hub import save_checkpoint
from torchvision.ops import box_convert

MVTEC = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw",
         "tile", "toothbrush", "transistor", "wood", "zipper"]


class Classifier(nn.Module):
    def __init__(self, bb, n):
        super().__init__()
        self.backbone, self.head = bb, nn.Linear(bb.out_dim, n)
        self.bn = nn.BatchNorm1d(bb.out_dim, affine=False)  # MAE-style probe normalisation

    def forward(self, x):
        return self.head(self.bn(self.backbone(x)))


def freeze(bb):
    for p in bb.parameters():
        p.requires_grad = False
    bb.eval()


def cls_data(a):
    if a.cls_data == "cifar10":
        tr = torchvision.transforms
        aug = tr.Compose([tr.RandomResizedCrop(a.img, (0.5, 1)), tr.RandomHorizontalFlip(), tr.ToTensor(),
                          tr.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        tr_ds = torchvision.datasets.CIFAR10(f"{a.data}/cifar", True, aug, download=True)
        va_ds = torchvision.datasets.CIFAR10(f"{a.data}/cifar", False, eval_transform(a.img), download=True)
        return tr_ds, va_ds, 10
    # MVTec good-vs-defective, image-level
    class ImageLevel(torch.utils.data.Dataset):
        def __init__(self, d): self.d = d
        def __len__(self): return len(self.d)
        def __getitem__(self, i): x, _, lbl = self.d[i]; return x, lbl
    return (ImageLevel(MVTecSeg(f"{a.data}/mvtec", a.mvtec_cats.split(","), "train", a.img)),
            ImageLevel(MVTecSeg(f"{a.data}/mvtec", a.mvtec_cats.split(","), "val", a.img)), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["cls", "seg", "det"], required=True)
    ap.add_argument("--mode", choices=["probe", "full"], default="full")
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--det-head", choices=["retinanet", "detr"], default=None)
    ap.add_argument("--cls-data", choices=["cifar10", "mvtec"], default="cifar10")
    ap.add_argument("--data", default="data")
    ap.add_argument("--mvtec-cats", default=",".join(MVTEC))
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--label-frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default=None, help="run name (results key / resume file)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # segmentation: dilated CNN, output stride 8 (DeepLabv3); weights are identical, only strides change
    bb = load_backbone(a.backbone, a.img, a.ckpt, output_stride=8 if a.task == "seg" and "vit" not in a.backbone else 32)
    src = a.ckpt.split("/")[0] if a.ckpt else ("supervised" if a.backbone.startswith("sup_") else "scratch")
    head = f"_{a.det_head}" if a.task == "det" and a.det_head else ""
    tag = a.tag or f"{a.task}_{a.mode}_{a.backbone}_{src}{head}_f{a.label_frac}_s{a.seed}"

    if a.task == "cls":
        tr, va, n = cls_data(a)
        model = Classifier(bb, n)
        collate = None
    elif a.task == "seg":
        tr = MVTecSeg(f"{a.data}/mvtec", a.mvtec_cats.split(","), "train", a.img)
        va = MVTecSeg(f"{a.data}/mvtec", a.mvtec_cats.split(","), "val", a.img)
        model, n, collate = build_seg(bb, 2), 2, None
    else:
        tr, va = NEUDet(f"{a.data}/neu-det", "train", a.img), NEUDet(f"{a.data}/neu-det", "val", a.img)
        model, n, collate = build_det(bb, len(NEU_CLASSES), a.det_head), len(NEU_CLASSES), det_collate
    tr = label_fraction(tr, a.label_frac, a.seed)
    if a.mode == "probe":
        freeze(bb)
    model.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": [p for p in bb.parameters() if p.requires_grad], "lr": a.lr},
                             {"params": [p for n_, p in model.named_parameters()
                                         if p.requires_grad and not n_.startswith("backbone.")], "lr": a.lr * 10}],
                            weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    dl = DataLoader(tr, a.bs, shuffle=True, num_workers=a.workers, collate_fn=collate, drop_last=len(tr) > a.bs)
    vdl = DataLoader(va, a.bs, num_workers=a.workers, collate_fn=collate)
    is_detr = isinstance(model, DETR)
    best, step, start = None, 0, 0
    resume = Path(f"checkpoints/finetune/{tag}.resume.pt")
    st = None
    if resume.exists():  # epoch-level resume after a disconnect
        try:
            st = torch.load(resume, map_location="cpu", weights_only=False)
        except Exception as e:  # truncated by an old non-atomic save: start this run over
            print(f"WARNING: {resume} is unreadable ({e}); moved aside, {tag} starts from scratch", flush=True)
            resume.replace(resume.with_name(resume.name + ".corrupt"))
    if st:
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        best, step, start = st["best"], st["step"], st["epoch"] + 1
        print(f"resumed {tag} from epoch {start}")
    for ep in range(start, a.epochs):
        model.train()
        if a.mode == "probe":
            bb.eval()
        for batch in dl:
            if a.task == "cls":
                x, y = batch[0].to(dev), batch[1].to(dev)
                loss = F.cross_entropy(model(x), y)
            elif a.task == "seg":
                x, y = batch[0].to(dev), batch[1].to(dev)
                out = model(x)
                loss = F.cross_entropy(out, y, weight=torch.tensor([1.0, 5.0], device=dev)) + dice_loss(out, y)
            else:
                x, t = batch[0].to(dev), [{k: v.to(dev) for k, v in d.items()} for d in batch[1]]
                if is_detr:
                    nt = [{"labels": d["labels"], "boxes": box_convert(d["boxes"] / a.img, "xyxy", "cxcywh")} for d in t]
                    loss = sum(detr_loss(*model(x), nt).values())
                else:
                    loss = sum(model.loss(*model(x), t).values())
            opt.zero_grad(set_to_none=True)
            step += 1
            if not torch.isfinite(loss):  # skip the batch rather than step the weights into NaN
                print(f"  step {step}: non-finite loss, batch skipped", flush=True)
                continue
            loss.backward()
            if torch.isfinite(torch.nn.utils.clip_grad_norm_(params, 1.0)):
                opt.step()
            if a.max_steps and step >= a.max_steps:
                break
        sched.step()
        metrics = evaluate(model, vdl, a, dev, n, is_detr)
        print(f"ep {ep} loss {loss.item():.4f} {metrics}", flush=True)
        key = {"cls": "top1", "seg": "dice_fg", "det": "mAP"}[a.task]
        if best is None or metrics[key] > best[key]:
            best = metrics
            save_checkpoint({"model": model.state_dict(), "args": vars(a), "metrics": metrics},
                            f"checkpoints/finetune/{tag}.pt", push=False)
        save_checkpoint({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                         "best": best, "step": step, "epoch": ep}, resume, push=False)
        if a.max_steps and step >= a.max_steps:
            break
    best_ck = torch.load(f"checkpoints/finetune/{tag}.pt", map_location="cpu", weights_only=False)
    save_checkpoint(best_ck, f"checkpoints/finetune/{tag}.pt")  # push the best checkpoint once at the end
    log_result("results/results.jsonl", {"tag": tag, **{k: v for k, v in vars(a).items() if k != "data"},
                                         "source": src, **{k: v for k, v in best.items() if not isinstance(v, (list, dict))}})
    resume.unlink(missing_ok=True)


@torch.no_grad()
def evaluate(model, vdl, a, dev, n, is_detr):
    model.eval()
    if a.task == "cls":
        acc, cnt = 0.0, 0
        for x, y in vdl:
            acc += top1(model(x.to(dev)), y.to(dev)) * len(y); cnt += len(y)
        return {"top1": acc / cnt}
    if a.task == "seg":
        m = SegMeter(n)
        for x, y, _ in vdl:
            m.update(model(x.to(dev)), y)
        return m.compute()
    preds, tgts = [], []
    for x, t in vdl:
        x = x.to(dev)
        p = detr_predict(*model(x), a.img, 0.05) if is_detr else model.predict(x)
        preds += [{k: v.cpu() for k, v in d.items()} for d in p]
        tgts += t
    return mean_ap(preds, tgts, n)


if __name__ == "__main__":
    main()
