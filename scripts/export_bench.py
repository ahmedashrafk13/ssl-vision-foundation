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
    for f in (fp32, to_fp16(fp32), to_int8(fp32, calib)):
        rows.append(bench_ort(f, a.img, a.batch))
    eng = build_trt(fp32)
    if eng:
        rows.append({"file": eng.name, "provider": "tensorrt"})
    for r in rows:
        log_result("results/export.jsonl", {"backbone": a.backbone, **r})


if __name__ == "__main__":
    main()
