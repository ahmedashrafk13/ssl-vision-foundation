"""FastAPI demo: upload an image -> boxes + defect mask; upload a video -> per-frame boxes with track IDs.

  SVF_DET=checkpoints/finetune/det_full_....pt SVF_SEG=checkpoints/finetune/seg_full_....pt \
    uvicorn app.main:app --port 8000
Without checkpoints the models are randomly initialised (useful only to check the API contract).
"""
import io, os, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import torchvision.transforms.functional as TF
from svf.builders import build_backbone, build_seg, build_det
from svf.data.datasets import MEAN, STD, NEU_CLASSES
from svf.heads.detr import DETR, detr_predict
from svf.track.bytetrack import ByteTracker
from svf.hub import load_checkpoint

IMG = 224
app = FastAPI(title="SSL Vision Foundation demo")
dev = "cuda" if torch.cuda.is_available() else "cpu"


def _load(kind):
    path = os.environ.get(f"SVF_{kind.upper()}")
    st = load_checkpoint(path) if path else None
    name = st["args"]["backbone"] if st else "resnet18"
    bb = build_backbone(name, IMG)
    if kind == "det":
        m = build_det(bb, len(NEU_CLASSES), st["args"].get("det_head") if st else None)
    else:
        m = build_seg(bb, 2)
    if st:
        m.load_state_dict(st["model"])
    return m.eval().to(dev)


DET, SEG = _load("det"), _load("seg")


def _prep(img):
    w, h = img.size
    x = TF.normalize(TF.to_tensor(img.convert("RGB").resize((IMG, IMG))), MEAN, STD)[None].to(dev)
    return x, (w / IMG, h / IMG)


@torch.no_grad()
def _detect(x, scale, thr=0.3):
    p = (detr_predict(*DET(x), IMG, thr) if isinstance(DET, DETR) else DET.predict(x, score_thr=thr))[0]
    b = p["boxes"].cpu().numpy() * np.array(scale * 2)
    return b, p["scores"].cpu().numpy(), p["labels"].cpu().numpy()


@app.get("/health")
def health():
    return {"ok": True, "device": dev, "classes": NEU_CLASSES}


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(400, "not an image")
    x, scale = _prep(img)
    boxes, scores, labels = _detect(x, scale)
    with torch.no_grad():
        mask = SEG(x).argmax(1)[0].byte().cpu().numpy()
    mask = np.array(Image.fromarray(mask * 255).resize(img.size, Image.NEAREST)) > 127
    ys, xs = np.nonzero(mask)
    return {"boxes": [{"box": b.round(1).tolist(), "score": float(s), "label": NEU_CLASSES[int(l)]}
                      for b, s, l in zip(boxes, scores, labels)],
            "mask": {"defect_fraction": float(mask.mean()),
                     "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None,
                     "rle": _rle(mask)}}


@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...), max_frames: int = 300):
    import cv2
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "v.mp4").suffix, delete=False) as f:
        f.write(await file.read())
    cap, tracker, frames = cv2.VideoCapture(f.name), ByteTracker(), []
    while len(frames) < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        x, scale = _prep(Image.fromarray(bgr[..., ::-1]))
        b, s, l = _detect(x, scale, thr=0.1)  # ByteTrack wants low-score dets too
        frames.append([{"id": int(i), "box": bx.round(1).tolist(), "score": float(sc), "label": NEU_CLASSES[int(lb)]}
                       for i, bx, sc, lb in tracker.update(b.reshape(-1, 4), s, l)])
    cap.release()
    os.unlink(f.name)
    return {"num_frames": len(frames), "frames": frames}


def _rle(mask):
    """Column-major run-length encoding (COCO-compatible counts)."""
    flat = mask.flatten(order="F").astype(np.uint8)
    change = np.flatnonzero(np.diff(np.concatenate([[0], flat, [0]])))
    runs = np.diff(np.concatenate([[0], change]))
    return {"size": list(mask.shape), "counts": runs.tolist()}
