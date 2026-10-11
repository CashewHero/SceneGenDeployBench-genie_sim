"""Pinned public checkpoints, staged locally and published under a cache lock."""

from __future__ import annotations

import fcntl
import hashlib
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from runner_wrapper.files import publish_file

DA360_REVISION = "b3bebecef1237c088b66034f5033ef628de5dc0a"
ASSETS = {
    "sharp": (
        "sharp_2572gikvuh.pt",
        "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt",
        "94211a75198c47f61fca7d739ba08a215418d8d398d48fddf023baccc24f073d",
        2809738232,
    ),
    "da360_large": ("DA360_large.pth", "bb5c6039a3e24d29dbcc62e451fda63bed8d59de3cf831b6a1dfa567121ee54a", 1344095612),
    "da360_base": ("DA360_base.pth", "5004577f9741c219f5c71bba7ef64c8418df8be6a3b82c645669a795d0e71af1", 391493586),
    "da360_small": ("DA360_small.pth", "cba5dfeeb2199b4a7089a98ce08c7506c1e5ea12b22c3e4ad51cbdb15150dd74", 99638610),
}


def _valid(path: Path, digest: str, size: int) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest() == digest


def ensure_checkpoint(name: str, workspace: Path) -> Path:
    asset = ASSETS[name]
    filename = asset[0]
    if name == "sharp":
        _, url, digest, size = asset
        override = os.getenv("GENIESIM_SHARP_CHECKPOINT")
    else:
        _, digest, size = asset
        url = f"https://huggingface.co/hljiang/DA360/resolve/{DA360_REVISION}/{filename}"
        override = os.getenv("GENIESIM_DA360_CHECKPOINT")
    if override:
        path = Path(override)
        if not _valid(path, digest, size):
            raise ValueError(f"Checkpoint override does not match the selected public {name} weights: {path}")
        return path
    cache_root = os.getenv("PATH_MODEL_CACHE")
    if not cache_root:
        raise ValueError("PATH_MODEL_CACHE is required for automatic checkpoint downloads")
    target = Path(cache_root) / "genie_sim/checkpoints" / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix(target.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if _valid(target, digest, size):
            return target
        print(f"Downloading {name} checkpoint to the shared cache", flush=True)
        with tempfile.TemporaryDirectory(prefix="checkpoint-", dir=workspace) as staging:
            local = Path(staging) / filename
            try:
                with urllib.request.urlopen(url, timeout=120) as response, local.open("wb") as stream:
                    for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                        stream.write(block)
            except urllib.error.URLError as exc:
                raise ConnectionError(f"Could not download {name}: {exc.reason}") from exc
            if not _valid(local, digest, size):
                raise ValueError(f"Downloaded {name} checkpoint failed SHA256 verification")
            publish_file(local, target)
    return target
