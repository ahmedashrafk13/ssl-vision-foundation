"""Self-supervised pre-training: SimCLR (CNN) or MAE (ViT). Resumable; checkpoints mirrored to HF Hub.

  python scripts/pretrain.py --method simclr --backbone resnet18 --img 96 --epochs 200
  python scripts/pretrain.py --method mae --backbone vit_tiny --img 96 --mask-ratio 0.75 --epochs 400
"""
import argparse, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from svf.builders import build_backbone
from svf.data.datasets import unlabelled_pool, MEAN, STD
from svf.ssl.simclr import SimCLR, simclr_augment, TwoViews
from svf.ssl.mae import MAE
from svf.hub import save_checkpoint, load_checkpoint


def cosine_lr(opt, step, total, base, warmup):
    lr = base * step / warmup if step < warmup else base * 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))
    for g in opt.param_groups:
        g["lr"] = lr
    return lr


@torch.no_grad()
def amp_ok(net, img, dev, bs):
    """fp16 self-test: some GPUs (GTX 16xx) return NaN from fp16 cuDNN convs. Fall back to fp32 there."""
    import copy
    probe = copy.deepcopy(net).train()  # copy, so the test cannot touch the real BN statistics
    with torch.autocast(dev, dtype=torch.float16):
        out = probe(torch.randn(bs, 3, img, img, device=dev))
    ok_bn = all(torch.isfinite(b).all() for b in probe.buffers() if b.dtype.is_floating_point)
    ok = bool(torch.isfinite(out.float()).all()) and ok_bn
    print("mixed precision:", "fp16" if ok else "OFF (this GPU returns NaN in fp16; using fp32)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["simclr", "mae"], required=True)
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--data", default="data")
    ap.add_argument("--sources", default="imagenet100,cifar,mvtec")
    ap.add_argument("--img", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--mask-ratio", type=float, default=0.75)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=0, help="debug: stop after N steps")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tag = a.out or f"{a.method}_{a.backbone}_{a.img}" + (f"_m{int(a.mask_ratio * 100)}" if a.method == "mae" else "")
    ckdir = Path("checkpoints") / tag

    if a.method == "simclr":
        tf = TwoViews(simclr_augment(a.img))
        model = SimCLR(build_backbone(a.backbone, a.img))
        lr = a.lr or 0.3 * a.bs / 256
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6)  # LARS-free variant
    else:
        tf = T.Compose([T.RandomResizedCrop(a.img, scale=(0.2, 1.0)), T.RandomHorizontalFlip(),
                        T.ToTensor(), T.Normalize(MEAN, STD)])
        model = MAE(build_backbone(a.backbone, a.img), mask_ratio=a.mask_ratio)
        lr = a.lr or 1.5e-4 * a.bs / 256
        opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    model.to(dev)
    use_amp = dev == "cuda" and amp_ok(model.backbone if a.method == "simclr" else model.encoder, a.img, dev, a.bs)
    ds = unlabelled_pool(a.data, tf, a.sources.split(","))
    dl = DataLoader(ds, a.bs, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True,
                    persistent_workers=a.workers > 0)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    start, step = 0, 0
    last = ckdir / "last.pt"
    try:
        st = load_checkpoint(last, remote=f"{tag}/last.pt")
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        start, step = st["epoch"] + 1, st["step"]
        print(f"resumed {tag} from epoch {start}")
    except Exception:
        pass
    total = a.epochs * len(dl)
    print(f"{tag}: {len(ds)} images, {len(dl)} steps/epoch, device={dev}")
    for ep in range(start, a.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for i, (x, _) in enumerate(dl):
            cur = cosine_lr(opt, step, total, lr, warmup=10 * len(dl) if a.epochs > 20 else len(dl))
            with torch.autocast(dev, dtype=torch.bfloat16 if dev == "cpu" else torch.float16, enabled=use_amp):
                loss = model(x[0].to(dev, non_blocking=True), x[1].to(dev, non_blocking=True), a.temperature) \
                    if a.method == "simclr" else model(x.to(dev, non_blocking=True))[0]
            if not torch.isfinite(loss):  # never let one bad batch poison weights / BN statistics
                raise RuntimeError(f"non-finite loss at step {step}; restart resumes from the last epoch")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            run += loss.item(); step += 1
            if a.max_steps and step >= a.max_steps:
                break
        print(f"ep {ep} loss {run / (i + 1):.4f} lr {cur:.2e} {time.time() - t0:.0f}s", flush=True)
        bad = [k for k, v in model.state_dict().items() if v.dtype.is_floating_point and not torch.isfinite(v).all()]
        if bad:  # never save a corrupted state; a rerun resumes from the last good epoch
            raise RuntimeError(f"non-finite weights/BN stats after epoch {ep}: {bad[:4]}")
        state = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep, "step": step, "args": vars(a)}
        save_checkpoint(state, last, push=(ep + 1) % a.save_every == 0 or ep + 1 == a.epochs)
        if a.max_steps and step >= a.max_steps:
            break
    bb = model.backbone if a.method == "simclr" else model.encoder
    bad = [k for k, v in bb.state_dict().items() if v.dtype.is_floating_point and not torch.isfinite(v).all()]
    if bad:
        raise RuntimeError(f"refusing to save a backbone with non-finite weights/BN stats: {bad[:4]}")
    save_checkpoint({"backbone": bb.state_dict(), "name": a.backbone, "img": a.img, "method": a.method},
                    ckdir / "backbone.pt")


if __name__ == "__main__":
    main()
