"""Export each backbone to ONNX (FP32/FP16/INT8, + TensorRT if trtexec exists) and benchmark.

  python scripts/export_bench.py --backbone resnet18 --ckpt simclr_resnet18_96/backbone.pt
"""
import argparse
import numpy as np
from common import load_backbone, log_result
from svf.export import export_onnx, to_fp16, to_int8, build_trt, bench_ort, bench_torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    bb = load_backbone(a.backbone, a.img, a.ckpt)
    base = f"exports/{a.backbone}"
    rows = [bench_torch(bb, a.img, a.batch)]
    fp32 = export_onnx(bb, f"{base}.onnx", a.img)
    calib = np.random.randn(32, 3, a.img, a.img).astype(np.float32)  # swap for real val images for final numbers
    # each format is independent: a converter bug in one (e.g. FP16 on a ViT) becomes an 'error' row in the
    # deployment table instead of discarding the whole benchmark
    for name, make in (("fp32", lambda: fp32), ("fp16", lambda: to_fp16(fp32, a.img)), ("int8", lambda: to_int8(fp32, calib))):
        try:
            rows.append(bench_ort(make(), a.img, a.batch))
        except Exception as e:
            print(f"WARNING: {a.backbone} {name} ONNX failed: {type(e).__name__}: {str(e)[:300]}", flush=True)
            rows.append({"file": f"{a.backbone}_{name}.onnx", "provider": f"error: {type(e).__name__}"})
    try:
        eng = build_trt(fp32)
        if eng:
            rows.append({"file": eng.name, "provider": "tensorrt"})
    except Exception as e:
        print(f"WARNING: TensorRT build failed: {e}", flush=True)
    for r in rows:
        log_result("results/export.jsonl", {"backbone": a.backbone, **r})


if __name__ == "__main__":
    main()
