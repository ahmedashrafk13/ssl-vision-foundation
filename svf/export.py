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
    kw = dict(input_names=["image"], output_names=["out"], opset_version=17,
              dynamic_axes={"image": {0: "batch"}, "out": {0: "batch"}} if dynamic_batch else None)
    try:  # torch >= 2.9 defaults to the dynamo exporter (needs onnxscript); keep the classic one
        torch.onnx.export(model, x, str(path), dynamo=False, **kw)
    except TypeError:  # older torch: no `dynamo` argument
        torch.onnx.export(model, x, str(path), **kw)
    return path


def ort_session(path, providers):
    """ONNX Runtime session at ORT_ENABLE_EXTENDED. The default ORT_ENABLE_ALL adds a CPU layout transform
    that segfaults (access violation, no Python error) while loading the FP16 ViT on onnxruntime 1.32 CPU;
    EXTENDED keeps every fusion the GPU path uses and is applied to all formats so the comparison stays fair."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    return ort.InferenceSession(str(path), so, providers=providers)


def to_fp16(path, img_size=96):
    """FP16 copy of an ONNX model, checked to load in ONNX Runtime and to match the FP32 model.

    onnxconverter-common re-types the outputs of the Cast nodes in ViT attention (the softmax scale) to
    float16 but leaves their `to=float` attribute, so ONNX Runtime rejects the model ("Type (tensor(float16))
    of output arg ... does not match expected type (tensor(float))"). Those Casts are re-pointed at float16.
    (Do NOT retry with "Cast" in op_block_list: the converter then inserts Casts after its own Casts forever.)
    """
    import onnx
    from onnx import TensorProto
    from onnxconverter_common import float16
    out = Path(path).with_name(Path(path).stem + "_fp16.onnx")
    m = float16.convert_float_to_float16(onnx.load(str(path)), keep_io_types=True)
    vi = {v.name: v.type.tensor_type.elem_type for v in m.graph.value_info}
    for node in m.graph.node:
        if node.op_type == "Cast" and vi.get(node.output[0]) == TensorProto.FLOAT16:
            for att in node.attribute:
                if att.name == "to" and att.i == TensorProto.FLOAT:
                    att.i = TensorProto.FLOAT16
    onnx.save(m, str(out))
    x = np.random.randn(2, 3, img_size, img_size).astype(np.float32)
    ref = ort_session(path, ["CPUExecutionProvider"]).run(None, {"image": x})[0]
    got = ort_session(out, ["CPUExecutionProvider"]).run(None, {"image": x})[0]
    err = float(np.abs(got.astype(np.float32) - ref).max() / (np.abs(ref).max() + 1e-6))
    if not np.isfinite(err) or err > 0.05:
        raise RuntimeError(f"FP16 model disagrees with FP32 (max rel. error {err:.3g})")
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
    sess = ort_session(path, prov)
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
