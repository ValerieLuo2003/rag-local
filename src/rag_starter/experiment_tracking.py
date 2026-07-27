from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Sequence


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="JSON config file. Explicit CLI options override config values.")


def parse_args_with_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config")
    known, _ = bootstrap.parse_known_args()

    if known.config:
        config_path = Path(known.config)
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
        if not isinstance(config, dict):
            parser.error(f"Config must be a JSON object: {config_path}")

        valid_keys = {action.dest for action in parser._actions if action.dest != "help"}
        unknown_keys = sorted(set(config) - valid_keys)
        if unknown_keys:
            parser.error(f"Unknown config keys: {', '.join(unknown_keys)}")
        parser.set_defaults(**config)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(seed)


def build_run_metadata(
    args: argparse.Namespace,
    *,
    documents: int,
    chunks: int,
    examples: int,
    eval_file: str | Path,
) -> dict:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "arguments": vars(args),
        "dataset": {
            "eval_file": str(eval_file),
            "eval_file_sha256": sha256_file(eval_file),
            "documents": documents,
            "chunks": chunks,
            "examples": examples,
        },
    }


def git_commit(root: str | Path = ".") -> str | None:
    resolved_root = Path(root).resolve()
    try:
        process = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={resolved_root.as_posix()}",
                "-C",
                str(resolved_root),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return process.stdout.strip() or None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def latency_summary(values_ms: Sequence[float]) -> dict[str, float | int | None]:
    if not values_ms:
        return {"count": 0, "mean": None, "p50": None, "p95": None}
    ordered = sorted(values_ms)
    return {
        "count": len(ordered),
        "mean": mean(ordered),
        "p50": percentile(ordered, 0.50),
        "p95": percentile(ordered, 0.95),
    }


def percentile(ordered_values: Sequence[float], quantile: float) -> float:
    if not ordered_values:
        raise ValueError("ordered_values cannot be empty")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    index = (len(ordered_values) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered_values) - 1)
    weight = index - lower
    return ordered_values[lower] * (1 - weight) + ordered_values[upper] * weight


def write_json(path: str | Path, value: object) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return output_path
