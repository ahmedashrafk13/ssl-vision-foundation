"""Fetch datasets into data/.

  python scripts/download_data.py mvtec                 # HF parquet mirror (official links are dead)
  python scripts/download_data.py imagenet100 --per-class 300
  python scripts/download_data.py neu                   # needs Kaggle API creds (~/.kaggle/kaggle.json)
CIFAR-10 downloads itself through torchvision on first use.
"""
import argparse, io, shutil, subprocess, sys
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq
from PIL import Image

MVTEC_REPO = "TheoM55/mvtec_all_objects_split"
IN100_REPO = "clane9/imagenet-100"
NEU_KAGGLE = "kaustubhdikshit/neu-surface-defect-database"


def mvtec(out):
    """Rebuild the original layout: <cat>/{train/good, test/<defect>, ground_truth/<defect>/*_mask.png}."""
    files = [f for f in HfApi().list_repo_files(MVTEC_REPO, repo_type="dataset") if f.endswith(".parquet")]
    n = 0
    for f in files:
        for b in pq.ParquetFile(hf_hub_download(MVTEC_REPO, f, repo_type="dataset")).iter_batches(64):
            for r in b.to_pylist():
                stem = Path(r["image_path"]["path"] or f"{n}.png").stem
                d = out / r["object"] / r["split"] / r["defect"]
                d.mkdir(parents=True, exist_ok=True)
                name = f"{stem}_{n}"  # mirror paths are not unique across defects
                Image.open(io.BytesIO(r["image_path"]["bytes"])).convert("RGB").save(d / f"{name}.png")
                m = r.get("mask_path")
                if r["defect"] != "good" and m and m.get("bytes"):
                    g = out / r["object"] / "ground_truth" / r["defect"]
                    g.mkdir(parents=True, exist_ok=True)
                    Image.open(io.BytesIO(m["bytes"])).convert("L").save(g / f"{name}_mask.png")
                n += 1
        print(f"{f}: {n} images so far", flush=True)


def imagenet100(out, per_class):
    files = [f for f in HfApi().list_repo_files(IN100_REPO, repo_type="dataset")
             if f.endswith(".parquet") and "train" in f]
    counts = {}
    for f in files:
        for b in pq.ParquetFile(hf_hub_download(IN100_REPO, f, repo_type="dataset")).iter_batches(256):
            for r in b.to_pylist():
                c = str(r["label"])
                if counts.get(c, 0) >= per_class:
                    continue
                d = out / "train" / c
                d.mkdir(parents=True, exist_ok=True)
                img = r["image"]
                Image.open(io.BytesIO(img["bytes"])).convert("RGB").save(d / f"{counts.get(c, 0):05d}.jpg", quality=90)
                counts[c] = counts.get(c, 0) + 1
        print(f"{f}: {sum(counts.values())} images", flush=True)
        if len(counts) >= 100 and min(counts.values()) >= per_class:
            break


def neu(out):
    if not shutil.which("kaggle"):
        sys.exit("pip install kaggle and put your API token in ~/.kaggle/kaggle.json")
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", NEU_KAGGLE, "-p", str(out), "--unzip"], check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["mvtec", "imagenet100", "neu", "all"])
    ap.add_argument("--root", default="data")
    ap.add_argument("--per-class", type=int, default=300)
    a = ap.parse_args()
    root = Path(a.root)
    if a.what in ("mvtec", "all"):
        mvtec(root / "mvtec")
    if a.what in ("imagenet100", "all"):
        imagenet100(root / "imagenet100", a.per_class)
    if a.what in ("neu", "all"):
        neu(root / "neu-det")
