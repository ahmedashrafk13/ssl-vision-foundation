# Self-Supervised Vision Foundation Model — CNN vs Vision Transformer

Two backbones pre-trained **without labels** — a from-scratch ResNet with **SimCLR** and a ViT with **MAE** — then
fine-tuned under identical heads for classification, segmentation and detection, and compared against a
supervised ImageNet ResNet on accuracy, data efficiency and deployment speed.

> Status: code complete and CPU-tested (`pytest`: 17 pass). Training runs on cloud GPUs and has not been run yet;
> the results tables below fill in from `results/*.jsonl`.

## What is implemented (all original PyTorch modules)

| Area | Module |
|---|---|
| ResNet: basic + bottleneck blocks, BN, projection shortcuts, dilated output-stride 8/16 | `svf/models/resnet.py` |
| Depthwise-separable (MobileNet) + MBConv/SE with compound scaling (EfficientNet) | `svf/models/mobile.py` |
| ViT-Tiny / ViT-Small: patch embedding, sin-cos positions, pre-norm blocks, stochastic depth | `svf/models/vit.py` |
| SimCLR: two-view augmentation, projection head, NT-Xent | `svf/ssl/simclr.py` |
| MAE: random masking, visible-only encoder, light decoder, loss on masked patches only | `svf/ssl/mae.py` |
| DeepLabv3 (ASPP, atrous convs) and SegFormer-style MLP decoder (+ simple pyramid for ViT) | `svf/heads/segmentation.py` |
| FPN + RetinaNet (anchors, focal loss, box codec, NMS) | `svf/heads/retinanet.py` |
| DETR: Transformer enc-dec, object queries, Hungarian matching, CE + L1 + GIoU | `svf/heads/detr.py` |
| ByteTrack: Kalman filter, two-stage high/low-score association | `svf/track/bytetrack.py` |
| Metrics: top-1, Dice, mIoU, COCO mAP@0.5:0.95 + per-class AP | `svf/metrics.py` |
| ONNX FP32/FP16/INT8 (ORT static QDQ), TensorRT via trtexec, benchmarks | `svf/export.py` |
| Checkpoints mirrored to the Hugging Face Hub, auto-resume | `svf/hub.py` |

## Protocol

* **Pre-training data** (unlabelled): ImageNet-100 subset, CIFAR-10 images, MVTec AD images (labels/masks never read).
* **Linear probe**: frozen backbone + BN + linear layer (seg: frozen backbone, head trained).
* **Full fine-tune**: backbone LR = head LR / 10, AdamW, cosine.
* **Same heads per task**: segmentation = DeepLabv3 on CNNs / SegFormer decoder on ViT; detection = RetinaNet+FPN
  on CNNs, DETR on ViT, plus a DETR-on-every-backbone run so one head is literally shared.
* **Data efficiency**: 1 / 10 / 25 / 50 / 100 % of labels, fixed seeds.
* **Ablations**: MAE mask ratio 0.5 / 0.6 / 0.75 / 0.9; probe-vs-finetune gap.

## Running

```bash
pip install -r requirements.txt
pytest -q tests                                   # CPU, ~20 s
python scripts/download_data.py mvtec             # HF mirror; official MVTec links are dead
python scripts/download_data.py imagenet100 --per-class 300
python scripts/download_data.py neu               # Kaggle API
bash scripts/run_all.sh                           # full grid (GPU)
python scripts/plot_results.py                    # data-efficiency curves, ablation, gap table
```

GPU runs: `notebooks/colab_train.ipynb` (Colab or Kaggle). Set `HF_TOKEN` as a notebook secret.

### Hugging Face checkpoints
`HF_TOKEN` and `HF_REPO` come from the environment or a local `.env` (gitignored, never commit it).
Pre-training pushes `checkpoints/<tag>/last.pt` every `--save-every` epochs and `backbone.pt` at the end;
fine-tunes push their best checkpoint. A restarted run pulls `last.pt` from the Hub and resumes, so
Colab disconnects only cost the epochs since the last push.

### Demo API
```bash
SVF_DET=checkpoints/finetune/<det>.pt SVF_SEG=checkpoints/finetune/<seg>.pt uvicorn app.main:app
```
`POST /predict/image`: boxes + defect mask (RLE). `POST /predict/video`: per-frame boxes with ByteTrack IDs.

## Results
_Fill from `results/` after the GPU runs: headline table, `data_eff_*.png`, `mae_mask_ratio.png`, export table._
