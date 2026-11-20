# Self-Supervised Vision Foundation Model — CNN vs Vision Transformer

Two backbones pre-trained **without labels**: a from-scratch ResNet-18 with **SimCLR** and a ViT-Tiny with **MAE**, then
fine-tuned under identical heads for classification, segmentation and detection, and compared against
scratch and supervised-ImageNet ResNet-18 baselines on accuracy, data efficiency and deployment speed.

> Status: code complete, tested, and smoke-tested end to end on GPU. Results fill in from `results/*.jsonl`
> when the Colab grid finishes.

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
| Resumable experiment grid (skips finished runs, epoch-level resume) | `scripts/run_grid.py` |

## Protocol (sized for a single free Colab T4)

All experiments run at **96×96 px** so the whole grid fits one T4 (~25–30 GPU hours, spread over sessions).

* **Pre-training (no labels):** ImageNet-100 subset (300 img/class) + CIFAR-10 train images + MVTec AD images;
  labels and masks are never read. SimCLR → ResNet-18, MAE (mask 75 %) → ViT-Tiny (patch 4), 100 epochs each.
* **Baselines:** the same ResNet-18 trained from scratch, and torchvision's supervised ImageNet-1k ResNet-18
  (an upper reference: 1.28 M labelled images vs our ~85 k unlabelled ones).
* **Tasks, identical heads per backbone family:**
  CIFAR-10 classification (linear probe + full fine-tune) · MVTec AD defect segmentation (DeepLabv3 on CNNs,
  SegFormer decoder on ViT) · NEU-DET surface-defect detection (RetinaNet+FPN on CNNs, DETR on ViT, plus DETR on
  every CNN so one head is literally shared).
* **Linear probe:** frozen backbone + BN + linear layer. **Full fine-tune:** backbone LR = head LR / 10, AdamW, cosine.
* **Data efficiency:** CIFAR-10 at 1 / 10 / 100 % labels; MVTec segmentation at 10 / 25 / 100 %.
* **Ablation:** MAE mask ratio 50 / 75 / 90 % (30-epoch pre-training, CIFAR-10 linear probe).
* **Deployment:** PyTorch vs ONNX FP32 / FP16 / INT8 latency and size.

## Running

**On Colab (recommended):** open `notebooks/colab_train.ipynb`, choose a T4 GPU, and click **Run all**, every time.
Datasets are downloaded once and cached on Google Drive as tar files; checkpoints and results are written straight
to Drive (saved every epoch). After a disconnect, *Run all* again: finished runs are skipped and an interrupted one
resumes from its last epoch. The last cell writes the results into this README and pushes them.

**Locally:**
```bash
pip install -r requirements.txt
pytest -q tests                                   # CPU, ~20 s
python scripts/download_data.py mvtec             # HF mirror; official MVTec links are dead
python scripts/download_data.py imagenet100 --per-class 300
python scripts/download_data.py neu               # or the anonymous Kaggle URL used in the notebook
python scripts/run_grid.py --smoke                # every stage for a few steps
python scripts/run_grid.py                        # full resumable grid -> results/summary.md + plots
```

### Hugging Face checkpoints
`HF_TOKEN` and `HF_REPO` come from the environment or a local `.env` (gitignored, never commit it).
Optional backup on top of Drive (enabled when `HF_PUSH=1`; the notebook sets it if an `HF_TOKEN` secret exists). Pre-training pushes `checkpoints/<tag>/last.pt` every `--save-every` epochs and `backbone.pt` at the end;
fine-tunes push their best checkpoint. A restarted run pulls `last.pt` from the Hub and resumes, so
Colab disconnects only cost the epochs since the last push.

### Demo API
```bash
SVF_DET=checkpoints/finetune/<det>.pt SVF_SEG=checkpoints/finetune/<seg>.pt uvicorn app.main:app
```
`POST /predict/image`: boxes + defect mask (RLE). `POST /predict/video`: per-frame boxes with ByteTrack IDs.

## Results
_Filled in automatically by the notebook's last cell (`scripts/update_readme.py`) once the grid finishes._
