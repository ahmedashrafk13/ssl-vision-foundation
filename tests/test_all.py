"""CPU smoke tests: shapes, loss behaviour, matching, tracking, metrics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pytest
import torch
from svf.models.resnet import resnet18, resnet50
from svf.models.mobile import DepthwiseSeparable, MobileNetLite, EfficientNetLite
from svf.models.vit import ViT
from svf.ssl.simclr import SimCLR, nt_xent
from svf.ssl.mae import MAE, patchify, unpatchify, random_masking
from svf.heads.segmentation import DeepLabV3, SegFormerHead, dice_loss
from svf.heads.retinanet import RetinaNet, encode, decode
from svf.heads.detr import DETR, detr_loss, hungarian_match, detr_predict
from svf.track.bytetrack import ByteTracker
from svf.metrics import mean_ap, SegMeter

torch.manual_seed(0)
tiny_vit = lambda img=32: ViT(img_size=img, patch=8, dim=64, depth=2, heads=4)


def test_resnet_feature_pyramid():
    f = resnet18().forward_features(torch.randn(2, 3, 64, 64))
    assert [f[k].shape[-1] for k in ("c2", "c3", "c4", "c5")] == [16, 8, 4, 2]
    assert resnet50().forward(torch.randn(1, 3, 64, 64)).shape == (1, 2048)


def test_resnet_output_stride_dilation():
    f = resnet18(output_stride=8).forward_features(torch.randn(1, 3, 64, 64))
    assert f["c5"].shape[-1] == 8


def test_mobile_blocks():
    assert DepthwiseSeparable(8, 16, 2)(torch.randn(1, 8, 16, 16)).shape == (1, 16, 8, 8)
    assert MobileNetLite(0.5)(torch.randn(1, 3, 64, 64)).shape == (1, 256)
    e0, e1 = EfficientNetLite(0), EfficientNetLite(1)
    assert e1.resolution > e0.resolution and e1.out_dim >= e0.out_dim
    assert sum(p.numel() for p in e1.parameters()) > sum(p.numel() for p in e0.parameters())


def test_vit_shapes():
    v = tiny_vit()
    x = torch.randn(2, 3, 32, 32)
    assert v(x).shape == (2, 64)
    assert v.forward_features(x)["c5"].shape == (2, 64, 4, 4)


def test_nt_xent_prefers_aligned_pairs():
    z = torch.randn(16, 32)
    assert nt_xent(z, z + 0.01 * torch.randn_like(z)) < nt_xent(z, torch.randn(16, 32))


def test_simclr_backward():
    m = SimCLR(resnet18(small_input=True), hidden=64)
    loss = m(torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32))
    loss.backward()
    assert torch.isfinite(loss)


def test_patchify_roundtrip():
    x = torch.randn(2, 3, 32, 32)
    assert torch.allclose(unpatchify(patchify(x, 8), 8), x)


def test_masking_ratio():
    keep, restore, mask = random_masking(3, 64, 0.75, "cpu")
    assert keep.shape == (3, 16) and mask.sum(1).eq(48).all()
    assert torch.equal(torch.gather(mask, 1, keep), torch.zeros(3, 16))  # kept patches are unmasked


def test_mae_loss_only_on_masked():
    m = MAE(tiny_vit(), mask_ratio=0.75, dec_dim=32, dec_depth=1, dec_heads=2)
    loss, pred, mask = m(torch.randn(2, 3, 32, 32))
    assert pred.shape == (2, 16, 8 * 8 * 3) and mask.sum() == 2 * 12
    loss.backward()
    assert m.encoder.patch_embed.proj.weight.grad is not None


def test_seg_heads():
    x = torch.randn(2, 3, 64, 64)
    assert DeepLabV3(resnet18(output_stride=8), 2)(x).shape == (2, 2, 64, 64)
    assert SegFormerHead(resnet18(), 3)(x).shape == (2, 3, 64, 64)
    assert SegFormerHead(tiny_vit(64), 2)(x).shape == (2, 2, 64, 64)
    y = torch.zeros(2, 64, 64, dtype=torch.long)
    good = torch.zeros(2, 2, 64, 64); good[:, 0] = 10
    assert dice_loss(good, y) < dice_loss(-good, y)


def test_box_codec_roundtrip():
    a = torch.tensor([[0, 0, 10, 10], [5, 5, 25, 15.]])
    b = torch.tensor([[1, 2, 12, 9], [4, 6, 30, 20.]])
    assert torch.allclose(decode(a, encode(a, b)), b, atol=1e-4)


def test_retinanet_loss_and_predict():
    m = RetinaNet(resnet18(), num_classes=3, fpn_dim=32)
    x = torch.randn(2, 3, 128, 128)
    cls, box, anchors = m(x)
    assert cls.shape[1] == box.shape[1] == anchors.shape[0]
    t = [{"boxes": torch.tensor([[10, 10, 60, 60.]]), "labels": torch.tensor([1])},
         {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}]
    l = m.loss(cls, box, anchors, t)
    sum(l.values()).backward()
    assert l["box"] > 0
    m.eval()
    assert set(m.predict(x)[0]) == {"boxes", "scores", "labels"}


def test_hungarian_matching_finds_identity():
    boxes = torch.tensor([[[0.2, 0.2, 0.1, 0.1], [0.7, 0.7, 0.2, 0.2], [0.5, 0.5, 0.3, 0.3]]])
    logits = torch.zeros(1, 3, 4)
    t = [{"labels": torch.tensor([0, 1]), "boxes": torch.tensor([[0.7, 0.7, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]])}]
    r, c = hungarian_match(logits, boxes, t)[0]
    assert dict(zip(r.tolist(), c.tolist())) == {0: 1, 1: 0}


def test_detr_forward_loss():
    m = DETR(tiny_vit(64), num_classes=3, d=32, queries=10, enc_layers=1, dec_layers=1, heads=4)
    lg, bx = m(torch.randn(2, 3, 64, 64))
    assert lg.shape == (2, 10, 4) and bx.shape == (2, 10, 4)
    t = [{"labels": torch.tensor([2]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])},
         {"labels": torch.zeros(0, dtype=torch.long), "boxes": torch.zeros(0, 4)}]
    sum(detr_loss(lg, bx, t).values()).backward()
    assert len(detr_predict(lg, bx, 64, 0.0)[0]["boxes"]) == 10


def test_bytetrack_keeps_identity():
    tr = ByteTracker()
    ids = []
    for k in range(10):
        score = 0.3 if k == 5 else 0.9  # low-score frame must still associate (second stage)
        out = tr.update(np.array([[10 + 2 * k, 10, 50 + 2 * k, 50.]]), np.array([score]), np.array([0]))
        ids.append(out[0][0])
    assert len(set(ids)) == 1


def test_map_perfect_and_empty():
    t = [{"boxes": torch.tensor([[0, 0, 10, 10.]]), "labels": torch.tensor([0])}]
    p = [{"boxes": torch.tensor([[0, 0, 10, 10.]]), "labels": torch.tensor([0]), "scores": torch.tensor([0.9])}]
    assert mean_ap(p, t, 1)["mAP"] == pytest.approx(1.0)
    e = [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long), "scores": torch.zeros(0)}]
    assert mean_ap(e, t, 1)["mAP"] == 0.0


def test_seg_meter():
    m = SegMeter(2)
    y = torch.tensor([[[0, 1], [1, 1]]])
    logits = torch.nn.functional.one_hot(y, 2).permute(0, 3, 1, 2).float()
    m.update(logits, y)
    r = m.compute()
    assert r["mIoU"] == pytest.approx(1.0) and r["dice"] == pytest.approx(1.0)


def test_onnx_export(tmp_path):
    pytest.importorskip("onnx")
    from svf.export import export_onnx
    p = export_onnx(resnet18(), tmp_path / "r18.onnx", 64)
    assert p.stat().st_size > 1e6
