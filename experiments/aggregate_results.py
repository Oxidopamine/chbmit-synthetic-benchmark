"""Flatten per-cell result dicts into tidy tables (plan Section 21)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def flatten_cell_result(res: Dict) -> Dict:
    spec = res.get("spec", {})
    wm = res.get("window_metrics", {})
    em = res.get("event_metrics", {})
    fa = res.get("sensitivity_at_fa", {})
    row = {
        "fold": spec.get("fold"),
        "seed": spec.get("seed"),
        "scarcity_fraction": spec.get("scarcity_fraction"),
        "detector": spec.get("detector"),
        "condition": spec.get("condition"),
        "generator": spec.get("generator"),
        "synthetic_ratio": spec.get("synthetic_ratio"),
        "selected_threshold": res.get("selected_threshold"),
        "n_train_windows": res.get("n_train_windows"),
        "n_train_pos": res.get("n_train_pos"),
        "n_params": res.get("n_params"),
        "best_val_auprc": res.get("train_log", {}).get("best_val_auprc"),
        # window-level
        "win_auroc": wm.get("auroc"),
        "win_auprc": wm.get("auprc"),
        "win_balanced_acc": wm.get("balanced_accuracy"),
        "win_macro_f1": wm.get("macro_f1"),
        # event-level (SzCORE)
        "event_sensitivity": em.get("event_sensitivity"),
        "event_precision": em.get("event_precision"),
        "event_f1": em.get("event_f1"),
        "fp_per_24h": em.get("fp_per_24h"),
        "fa_per_hour": em.get("fa_per_hour"),
        "pooled_sensitivity": em.get("pooled_sensitivity"),
        "median_latency_seconds": em.get("median_latency_seconds"),
        "n_patients": em.get("n_patients"),
    }
    for budget, val in fa.items():
        row[f"sens_at_fa_{budget}"] = val
    return row


def results_to_frame(results: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame([flatten_cell_result(r) for r in results])


def save_results(results: List[Dict], out_dir: str | Path, name: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}_raw.json").write_text(json.dumps(results, indent=2, default=float),
                                              encoding="utf-8")
    df = results_to_frame(results)
    csv_path = out_dir / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path
