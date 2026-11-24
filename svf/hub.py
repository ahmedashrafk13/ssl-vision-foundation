"""Checkpoint save/load with optional mirroring to the Hugging Face Hub.

Reads HF_TOKEN and HF_REPO from the environment (or a local .env). Never hardcode the token.
On Colab/Kaggle put HF_TOKEN in the notebook secrets instead of uploading .env.
"""
import os
from pathlib import Path
import torch


def load_env(path=".env"):
    p = Path(path)
    if not p.exists():
        p = Path(__file__).resolve().parent.parent / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ.setdefault(k.strip(), v.strip())


def _api():
    load_env()
    token, repo = os.environ.get("HF_TOKEN"), os.environ.get("HF_REPO")
    if not token or not repo:
        return None, None
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo, private=True, exist_ok=True)
    return api, repo


def save_checkpoint(state, path, push=True, keep_remote_as=None):
    """Save locally; if HF creds are set, upload to <HF_REPO>/<keep_remote_as or relative path>."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # write-then-rename: a disconnect mid-save (Colab -> Drive) must never leave a truncated checkpoint
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    try:
        os.replace(tmp, path)
    except OSError:  # some FUSE mounts refuse rename-over-existing; fall back to delete + rename
        path.unlink(missing_ok=True)
        os.rename(tmp, path)
    if push and os.environ.get("HF_PUSH") == "1":  # opt-in: a slow upload must never stall training
        api, repo = _api()
        if api:
            remote = keep_remote_as or path.as_posix().split("checkpoints/", 1)[-1]
            try:
                api.upload_file(path_or_fileobj=str(path), path_in_repo=f"checkpoints/{remote}", repo_id=repo,
                                commit_message=f"checkpoint {remote}")
                print(f"[hub] pushed {remote} -> {repo}")
            except Exception as e:  # never kill a training run over an upload
                print(f"[hub] upload failed ({e}); local copy kept at {path}")
    return path


def load_checkpoint(path, remote=None, map_location="cpu"):
    """Load local file, or download checkpoints/<remote> from HF_REPO if missing locally."""
    path = Path(path)
    if not path.exists() and remote:
        load_env()
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(os.environ["HF_REPO"], f"checkpoints/{remote}", token=os.environ.get("HF_TOKEN")))
    return torch.load(path, map_location=map_location, weights_only=False)
