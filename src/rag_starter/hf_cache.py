from __future__ import annotations

import os
from pathlib import Path


def configure_hf_cache(model_cache_dir: str | Path | None) -> Path | None:
    cache_dir = Path(model_cache_dir).resolve() if model_cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache_dir / "home"))
        os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))
        os.environ.setdefault("HF_XET_CACHE", str(cache_dir / "xet"))

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    return cache_dir

