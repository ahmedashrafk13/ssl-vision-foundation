"""ONNX export, ONNX Runtime FP16 / INT8, TensorRT (if available), throughput / latency / size benchmark."""
import os
import time
from pathlib import Path
import numpy as np
import torch


def export_onnx(model, path, img_size=224, dynamic_batch=True):
    model = model.eval().cpu()
    x = torch.randn(1, 3, img_size, img_size)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, x, str(path), input_names=["image"], output_names=["out"], opset_version=17,
                      dynamic_axes={"image": {0: "batch"}, "out": {0: "batch"}} if dynamic_batch else None)
    return path


def to_fp16(path):
    import onnx
    from onnxconverter_common import float16
    out = Path(path).with_name(Path(path).stem + "_fp16.onnx")
    onnx.save(float16.convert_float_to_float16(onnx.load(str(path)), keep_io_types=True), str(out))
    return out


def to_int8(path, calib_images):
    """Static INT8 (QDQ) quantization with a small calibration set of (N,3,H,W) float arrays."""
    from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{"image": x[None].astype(np.float32)} for x in calib_images])

        def get_next(self):
            return next(self.it, None)

    out = Path(path).with_name(Path(path).stem + "_int8.onnx")
    quantize_static(str(path), str(out), Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, per_channel=True)
    return out


def build_trt(path, fp16=True, int8=False):
    """TensorRT engine via trtexec (present on most GPU images). Returns engine path or None."""
    import shutil, subprocess
    if not shutil.which("trtexec"):
        return None
    eng = Path(path).with_suffix(".engine")
    cmd = ["trtexec", f"--onnx={path}", f"--saveEngine={eng}", "--minShapes=image:1x3x224x224",
           "--optShapes=image:8x3x224x224", "--maxShapes=image:32x3x224x224"]
    cmd += ["--fp16"] if fp16 else []
    cmd += ["--int8"] if int8 else []
    subprocess.run(cmd, check=True, capture_output=True)
    return eng


def bench_ort(path, img_size=224, batch=8, iters=50, warmup=10):
    import onnxruntime as ort
    prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    sess = ort.InferenceSession(str(path), providers=prov)
    x = np.random.randn(batch, 3, img_size, img_size).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {"image": x})
    t = time.perf_counter()
    for _ in range(iters):
        sess.run(None, {"image": x})
    dt = (time.perf_counter() - t) / iters
    return {"file": Path(path).name, "provider": sess.get_providers()[0], "latency_ms": dt * 1000,
            "throughput_img_s": batch / dt, "size_mb": os.path.getsize(path) / 2 ** 20}


@torch.no_grad()
def bench_torch(model, img_size=224, batch=8, iters=50, warmup=10, device="cuda" if torch.cuda.is_available() else "cpu"):
    model = model.eval().to(device)
    x = torch.randn(batch, 3, img_size, img_size, device=device)
    for _ in range(warmup):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / iters
    size = sum(p.numel() * p.element_size() for p in model.parameters()) / 2 ** 20
    return {"file": "pytorch_fp32", "provider": device, "latency_ms": dt * 1000,
            "throughput_img_s": batch / dt, "size_mb": size}
