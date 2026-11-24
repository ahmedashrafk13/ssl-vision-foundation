"""Resumable experiment grid. Safe to re-run any number of times: finished runs are skipped
(pre-training: backbone.pt exists; fine-tunes: tag already in results/results.jsonl) and an
interrupted run resumes from its last saved epoch.

  python scripts/run_grid.py            # full grid (sized for one Colab T4, ~25-30 GPU hours total)
  python scripts/run_grid.py --smoke    # few-step end-to-end check of every stage (~5 min)
"""
import argparse, atexit, json, os, subprocess, sys, threading, time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
IMG = 96
LOGS = ROOT / "logs" / "grid"  # on Colab logs/ is a symlink into Drive, so these survive a disconnect
FAILED = []  # (name, exit code, log path, last lines)
STRICT = False  # smoke: stop at the first failure
SMOKE_STAGE_TIMEOUT = 20 * 60  # seconds


def done_tags(f="results/results.jsonl"):
    p = ROOT / f
    return {json.loads(l).get("tag") for l in p.read_text().splitlines() if l.strip()} if p.exists() else set()


def run(cmd, name, retries=1):
    """Run one stage, teeing its output to logs/grid/<name>.log. A failed stage is retried once (it resumes
    from its last epoch, which clears transient Drive / CUDA hiccups); if it fails again it is recorded and
    the grid moves on to independent stages instead of throwing away the rest of the session."""
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{name}.log"
    for attempt in range(retries + 1):
        print("\n>>>", " ".join(cmd), f"(retry {attempt})" if attempt else "", flush=True)
        tail = deque(maxlen=40)
        t0 = time.monotonic()
        with log.open("a", encoding="utf-8", buffering=1) as f:
            f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} attempt {attempt + 1}: {' '.join(cmd)}\n")
            p = subprocess.Popen([PY, "-u", *cmd], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, errors="replace", bufsize=1)
            # smoke stages take seconds to a few minutes: one that runs 20 min is hung, so kill it and say so
            # instead of silently burning the Colab session (full-grid stages legitimately run for hours)
            watchdog = threading.Timer(SMOKE_STAGE_TIMEOUT, p.kill) if STRICT else None
            if watchdog:
                watchdog.start()
            for line in p.stdout:
                sys.stdout.write(line); f.write(line); tail.append(line)
            rc = p.wait()
            if watchdog:
                watchdog.cancel()
                if rc and time.monotonic() - t0 >= SMOKE_STAGE_TIMEOUT - 1:
                    tail.append(f"\n[run_grid] killed: stage exceeded the {SMOKE_STAGE_TIMEOUT // 60} min smoke limit (hung)\n")
        if rc == 0:
            return True
        print(f"\n!!! {name} FAILED (exit {rc}). Last lines:\n" + "".join(tail), flush=True)
        if STRICT:
            sys.exit(f"stage {name} failed; full log: {log}")
    FAILED.append((name, rc, log, "".join(list(tail)[-15:])))
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", default="2")
    ap.add_argument("--only", default="", help="comma list of stages: pretrain,ablation,finetune,export,plots")
    a = ap.parse_args()
    stages = set(a.only.split(",")) if a.only else {"pretrain", "ablation", "finetune", "export", "plots"}
    sm = a.smoke
    global STRICT
    STRICT = sm
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
        run(["scripts/pretrain.py", *args, "--img", str(IMG), "--out", tag, "--save-every", "5", *W, *pre],
            f"pretrain_{tag}")

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
        if j["ck"] and not (ROOT / "checkpoints" / j["ck"]).exists():
            print(f"skip {tag} (waiting for pre-trained backbone {j['ck']})")
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
        run(cmd + ft, tag)

    # ---------- 3) deployment benchmark ----------
    if "export" in stages:
        ex = ROOT / "results/export.jsonl"
        have = {json.loads(l)["backbone"] for l in ex.read_text().splitlines() if l.strip()} if ex.exists() else set()
        for bb, ck, src in specs[:3]:
            if bb not in have or sm:
                if ck and not (ROOT / "checkpoints" / ck).exists():
                    print(f"skip export {bb} (waiting for {ck})")
                    continue
                run(["scripts/export_bench.py", "--backbone", bb, "--img", str(IMG), *(["--ckpt", ck] if ck else [])],
                    f"export_{bb}{sfx}")

    if sm:
        print("\nSMOKE TEST PASSED")
        return
    if "plots" in stages and (ROOT / "results/results.jsonl").exists():
        run(["scripts/plot_results.py"], "plots", retries=0)  # partial tables are still useful mid-grid
    if FAILED:
        print("\n" + "=" * 70 + f"\n{len(FAILED)} stage(s) FAILED twice; everything else that could run has run.")
        for name, rc, log, tail in FAILED:
            print(f"\n--- {name} (exit {rc}) full log: {log}\n{tail}")
        print("Run all again to retry them (they resume from their last saved epoch).")
        sys.exit(1)
    print("\nGRID COMPLETE")


if __name__ == "__main__":
    main()
