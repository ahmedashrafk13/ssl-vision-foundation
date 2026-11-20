import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from svf.builders import build_backbone
from svf.hub import load_checkpoint


def load_backbone(name, img, ckpt=None, output_stride=32):
    """ckpt: local path or '<tag>/backbone.pt' on the hub; None = random init (or supervised for sup_*)."""
    bb = build_backbone(name, img, output_stride=output_stride)
    if ckpt:
        st = load_checkpoint(ckpt if Path(ckpt).exists() else Path("checkpoints") / ckpt, remote=ckpt)
        missing, unexpected = bb.load_state_dict(st["backbone"], strict=False)
        print(f"loaded {ckpt}: missing={len(missing)} unexpected={len(unexpected)}")
    return bb


def log_result(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(row)
