"""Resumable experiment grid. Safe to re-run any number of times: finished runs are skipped
(pre-training: backbone.pt exists; fine-tunes: tag already in results/results.jsonl) and an
interrupted run resumes from its last saved epoch.

  python scripts/run_grid.py            # full grid (sized for one Colab T4, ~25-30 GPU hours total)
  python scripts/run_grid.py --smoke    # few-step end-to-end check of every stage (~5 min)
"""
import argparse, atexit, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
IMG = 96


def done_tags(f="results/results.jsonl"):
    p = ROOT / f
    return {json.loads(l).get("tag") for l in p.read_text().splitlines() if l.strip()} if p.exists() else set()


def run(cmd):
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run([PY, *cmd], cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", default="2")
    ap.add_argument("--only", default="", help="comma list of stages: pretrain,ablation,finetune,export,plots")
    a = ap.parse_args()
    stages = set(a.only.split(",")) if a.only else {"pretrain", "ablation", "finetune", "export", "plots"}
    sm = a.smoke
    pre = ["--max-steps", "3", "--bs", "32"] if sm else []
    ft = ["--max-steps", "3", "--epochs", "1", "--bs", "8"] if sm else []  # small batch: smoke fits a 4 GB GPU
    W = ["--workers", a.workers]
    sfx = "_smoke" if sm else ""
    if sm:  # smoke runs stay local: never push to, or resume from, the Hugging Face Hub
        os.environ["HF_PUSH"], os.environ["HF_TOKEN"], os.environ["HF_REPO"] = "0", "", ""
    snap = {f: (ROOT / f).read_text() if (ROOT / f).exists() else None
            for f in ("results/results.jsonl", "results/export.jsonl")}

    def restore():  # smoke rows must never reach the real results, even if a stage crashes
        for f, keep in snap.items():
            (ROOT / f).write_text(keep) if keep is not None else (ROOT / f).unlink(missing_ok=True)
    if sm:
        atexit.register(restore)

    # ---------- 1) self-supervised pre-training (unlabelled ImageNet-100 subset + CIFAR + MVTec) ----------
    SIM, MAE = f"simclr_resnet18_{IMG}{sfx}", f"mae_vit_tiny_{IMG}_m75{sfx}"
    plan = [(SIM, ["--method", "simclr", "--backbone", "resnet18", "--epochs", "1" if sm else "100", "--bs", "256"]),
            (MAE, ["--method", "mae", "--backbone", "vit_tiny", "--epochs", "1" if sm else "100", "--bs", "256"])]
    abl = [] if sm else [(f"mae_vit_tiny_{IMG}_m{int(m * 100)}_e30",
                          ["--method", "mae", "--backbone", "vit_tiny", "--epochs", "30", "--bs", "256",
                           "--mask-ratio", str(m)]) for m in (0.5, 0.75, 0.9)]
    for tag, args in plan * ("pretrain" in stages) + abl * ("ablation" in stages):
        if (ROOT / "checkpoints" / tag / "backbone.pt").exists():
            print(f"skip pretrain {tag} (done)")
            continue
        run(["scripts/pretrain.py", *args, "--img", str(IMG), "--out", tag, "--save-every", "5", *W, *pre])

    # ---------- 2) downstream: identical heads per task, 4 backbones ----------
    jobs = []
    for tag, _ in abl * ("ablation" in stages):  # mask-ratio ablation: linear probe on CIFAR-10
        jobs.append(dict(task="cls", mode="probe", bb="vit_tiny", ck=f"{tag}/backbone.pt", src=tag, frac=1.0, ep=30))
    specs = [("resnet18", f"{SIM}/backbone.pt", SIM), ("vit_tiny", f"{MAE}/backbone.pt", MAE),
             ("sup_resnet18", None, "supervised"), ("resnet18", None, "scratch")]
    if "finetune" in stages:
        for bb, ck, src in specs:
            if src != "scratch":  # probing a random network is meaningless
                jobs += [dict(task="cls", mode="probe", bb=bb, ck=ck, src=src, frac=1.0, ep=30),
                         dict(task="seg", mode="probe", bb=bb, ck=ck, src=src, frac=1.0, ep=40)]
            for frac in (0.01, 0.1, 1.0):  # data efficiency on CIFAR-10 (500 / 5k / 50k labels)
                jobs.append(dict(task="cls", mode="full", bb=bb, ck=ck, src=src, frac=frac,
                                 ep={0.01: 100, 0.1: 40, 1.0: 15}[frac]))
            for frac in (0.1, 0.25, 1.0):  # data efficiency on MVTec defect segmentation
                jobs.append(dict(task="seg", mode="full", bb=bb, ck=ck, src=src, frac=frac,
                                 ep={0.1: 120, 0.25: 80, 1.0: 40}[frac]))
            jobs.append(dict(task="det", mode="full", bb=bb, ck=ck, src=src, frac=1.0, ep=60))
            if bb != "vit_tiny":  # DETR on every backbone, so one detection head is literally shared
                jobs.append(dict(task="det", mode="full", bb=bb, ck=ck, src=src, frac=1.0, ep=100, head="detr"))
    done = done_tags()
    for j in jobs:
        head = j.get("head")
        tag = f"{j['task']}_{j['mode']}_{j['bb']}_{j['src']}" + (f"_{head}" if head else "") + f"_f{j['frac']}{sfx}"
        if tag in done:
            print(f"skip {tag} (done)")
            continue
        cmd = ["scripts/finetune.py", "--task", j["task"], "--mode", j["mode"], "--backbone", j["bb"],
               "--img", str(IMG), "--label-frac", str(j["frac"]), "--epochs", str(j["ep"]), "--tag", tag, *W]
        if j["ck"]:
            cmd += ["--ckpt", j["ck"]]
        if head:
            cmd += ["--det-head", head]
        if j["task"] == "cls":
            # ViT-Tiny at patch 4 = 576 tokens: batch 64 keeps a full fine-tune well inside a 16 GB T4
            cmd += ["--bs", "64" if j["bb"] == "vit_tiny" and j["mode"] == "full" else "128", "--lr", "3e-4" if j["mode"] == "full" else "1e-3"]
        else:
            cmd += ["--bs", "16"]
        run(cmd + ft)

    # ---------- 3) deployment benchmark ----------
    if "export" in stages:
        ex = ROOT / "results/export.jsonl"
        have = {json.loads(l)["backbone"] for l in ex.read_text().splitlines() if l.strip()} if ex.exists() else set()
        for bb, ck, src in specs[:3]:
            if bb not in have or sm:
                run(["scripts/export_bench.py", "--backbone", bb, "--img", str(IMG), *(["--ckpt", ck] if ck else [])])

    if sm:
        print("\nSMOKE TEST PASSED")
        return
    if "plots" in stages:
        run(["scripts/plot_results.py"])


if __name__ == "__main__":
    main()
