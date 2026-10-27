"""Headline table, data-efficiency curves, mask-ratio ablation, probe-vs-finetune gap -> results/*.png"""
import json
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = [json.loads(l) for l in Path("results/results.jsonl").read_text().splitlines() if l.strip()]
KEY = {"cls": "top1", "seg": "dice_fg", "det": "mAP"}
label = lambda r: f"{r['backbone']} ({r['source']})"

# data-efficiency: metric vs label fraction, one line per backbone/source, full fine-tune
for task, key in KEY.items():
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["task"] == task and r["mode"] == "full" and key in r:
            groups[label(r)][r["label_frac"]].append(r[key])
    if not groups:
        continue
    plt.figure(figsize=(6, 4))
    for name, pts in groups.items():
        xs = sorted(pts)
        plt.plot(xs, [sum(pts[x]) / len(pts[x]) for x in xs], marker="o", label=name)
    plt.xscale("log"); plt.xlabel("fraction of labels"); plt.ylabel(key); plt.title(f"{task}: data efficiency")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"results/data_eff_{task}.png", dpi=150); plt.close()

# probe vs full gap (label_frac == 1)
print("| task | backbone (source) | probe | full | gap |\n|---|---|---|---|---|")
for task, key in KEY.items():
    agg = defaultdict(dict)
    for r in rows:
        if r["task"] == task and r["label_frac"] == 1.0 and key in r:
            agg[label(r)][r["mode"]] = r[key]
    for name, d in agg.items():
        p, f = d.get("probe"), d.get("full")
        gap = f"{f - p:+.3f}" if p is not None and f is not None else "-"
        print(f"| {task} | {name} | {p if p is None else f'{p:.3f}'} | {f if f is None else f'{f:.3f}'} | {gap} |")

# MAE mask-ratio ablation: sources look like mae_vit_tiny_224_m60
abl = sorted((int(r["source"].rsplit("_m", 1)[1]), r[KEY[r["task"]]]) for r in rows
             if "_m" in r["source"] and r["source"].startswith("mae") and r["mode"] == "probe" and r["task"] == "cls")
if abl:
    plt.figure(figsize=(5, 3.5))
    plt.plot(*zip(*abl), marker="o"); plt.xlabel("mask ratio (%)"); plt.ylabel("linear-probe top-1")
    plt.title("MAE mask-ratio ablation"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("results/mae_mask_ratio.png", dpi=150)
