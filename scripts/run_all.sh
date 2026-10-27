#!/usr/bin/env bash
# Full experiment grid. Run on a GPU box (Colab/Kaggle); each line is resumable and pushes to the HF Hub.
set -e
cd "$(dirname "$0")/.."

# 1) self-supervised pre-training
python scripts/pretrain.py --method simclr --backbone resnet50 --img 224 --epochs 100 --bs 256
python scripts/pretrain.py --method mae    --backbone vit_small --img 224 --epochs 200 --bs 256
for m in 0.5 0.6 0.75 0.9; do   # mask-ratio ablation (cheaper: vit_tiny, 100 epochs)
  python scripts/pretrain.py --method mae --backbone vit_tiny --img 224 --epochs 100 --mask-ratio $m
  python scripts/finetune.py --task cls --mode probe --backbone vit_tiny --ckpt mae_vit_tiny_224_m$(python -c "print(int($m*100))")/backbone.pt --epochs 30
done

# 2) headline comparison + data efficiency, identical heads per task
SIMCLR="resnet50 simclr_resnet50_224/backbone.pt"
MAE="vit_small mae_vit_small_224_m75/backbone.pt"
SUP="sup_resnet50 -"
for spec in "$SIMCLR" "$MAE" "$SUP"; do
  set -- $spec; BB=$1; CK=$2; CKARG=$([ "$CK" = "-" ] && echo "" || echo "--ckpt $CK")
  for task in cls seg; do
    python scripts/finetune.py --task $task --mode probe --backbone $BB $CKARG
  done
  for frac in 0.01 0.1 0.25 0.5 1.0; do
    for task in cls seg det; do
      python scripts/finetune.py --task $task --mode full --backbone $BB $CKARG --label-frac $frac
    done
  done
  # same-head check: DETR on every backbone
  python scripts/finetune.py --task det --mode full --backbone $BB $CKARG --det-head detr
done

# 3) deployment benchmarks
python scripts/export_bench.py --backbone resnet50 --ckpt simclr_resnet50_224/backbone.pt
python scripts/export_bench.py --backbone vit_small --ckpt mae_vit_small_224_m75/backbone.pt
python scripts/export_bench.py --backbone sup_resnet50

python scripts/plot_results.py
