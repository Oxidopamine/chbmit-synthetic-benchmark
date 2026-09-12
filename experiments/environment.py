"""Record the software environment beside every result file.

The torch build behind the Phase 1-2 grids is unrecoverable: it was printed to a log that was
never committed (AUDIT_2026-08-31.md F7). Every driver now calls ``record_environment`` before
its first cell and writes ``run_environment<tag>.json`` next to its CSV, so a result file can
always be matched to the exact library versions, device and git commit that produced it.

Never fails the run: bookkeeping must not cost a GPU job.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def collect_environment() -> dict:
    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "argv": sys.argv,
    }
    try:
        root = Path(__file__).resolve().parent.parent
        env["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                           text=True, timeout=10).stdout.strip() or None
        env["git_dirty"] = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root,
                                               capture_output=True, text=True, timeout=10).stdout.strip())
    except Exception as e:  # pragma: no cover - no git in the container is fine
        env["git_error"] = str(e)
    for mod in ("torch", "numpy", "scipy", "pandas", "sklearn", "zarr", "timescoring"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            env[mod] = None
    try:
        import torch
        env["cuda"] = torch.version.cuda
        env["cudnn"] = torch.backends.cudnn.version()
        env["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        env["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    except Exception:
        pass
    return env


def record_environment(path: str | Path) -> dict:
    """Write the environment record to ``path`` (JSON). Returns the record. Never raises."""
    env = collect_environment()
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(env, indent=2, default=str), encoding="utf-8")
    except Exception as e:  # pragma: no cover
        env["write_error"] = str(e)
    return env
