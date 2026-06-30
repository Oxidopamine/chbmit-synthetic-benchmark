"""Configuration loading and access.

A thin, dependency-light wrapper around the YAML config files in ``configs/``.
We keep the config as nested dicts (not a rigid schema) so the many knobs in
Section 22 of the plan stay editable without code changes, but we expose a few
typed convenience accessors and a deterministic seed helper.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

# Repository root = parent of this file's parent (chbmit/ -> project_root/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


@dataclass
class Config:
    """Loaded configuration plus the project root for resolving relative paths."""

    raw: Dict[str, Any]
    root: Path = PROJECT_ROOT

    # --- generic access -------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        """Fetch a nested value with a dotted key, e.g. ``get('windowing.window_seconds')``."""
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def resolve_path(self, dotted_or_path: str) -> Path:
        """Resolve a config path key (e.g. ``'paths.processed_chbmit'``) or a raw
        relative path against the project root."""
        val = self.get(dotted_or_path, dotted_or_path)
        p = Path(val)
        return p if p.is_absolute() else (self.root / p)

    # --- convenience ----------------------------------------------------
    @property
    def base_seed(self) -> int:
        return int(self.get("project.base_seed", 42))

    def fold_seed(self, fold: int, seed: int, *salt: int) -> int:
        """Deterministic per-(fold, seed, salt) integer seed in [0, 2**31)."""
        h = (self.base_seed * 1_000_003) ^ (fold * 97_003) ^ (seed * 131_071)
        for s in salt:
            h = (h * 1_000_003) ^ (int(s) * 31)
        return h & 0x7FFFFFFF


def load_config(path: Optional[str] = None) -> Config:
    """Load a YAML config. Defaults to ``configs/chbmit_synthetic.yaml``."""
    import yaml  # imported lazily so pure-stdlib modules stay importable pre-install

    if path is None:
        path = os.environ.get("CHBMIT_CONFIG", "configs/chbmit_synthetic.yaml")
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw, root=PROJECT_ROOT)
