"""Trainer and single-cell runner (plan Sections 9, 15-18).

A "cell" is one (fold, seed, scarcity, detector, condition[, generator]) point in
the grid. ``run_cell`` performs the whole leakage-safe protocol:

    train windows (train groups) -> event-level scarcity -> negative sampling
    -> condition assembly -> train w/ early stopping (val AUPRC)
    -> threshold selection on val event F1 -> test: window + SzCORE event +
       patient metrics + sensitivity at FA budgets

Validation/test always use full real timelines (Rules 5, 7). Synthetic windows,
when used, are training-only and supplied already fit on the same train subset.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

from augmentation.balanced_sampler import make_balanced_sampler
from augmentation.class_weighting import make_loss, pos_weight_from_labels
from augmentation.classical_aug import ClassicalAugment, ClassicalAugConfig
from chbmit.datasets import ArrayDataset, WindowDataset, negative_sample
from chbmit.scarcity import apply_scarcity_to_windows, select_seizure_events
from evaluation.event_metrics_szcore import (
    FilePrediction,
    PostprocConfig,
    SzCoreParams,
    evaluate_event_level,
    sensitivity_at_fa_budgets,
)
from evaluation.patient_metrics import aggregate_patient_metrics, per_patient_window_metrics
from evaluation.thresholds import select_threshold
from evaluation.window_metrics import window_metrics
from models import build_model, count_parameters


@dataclass
class TrainConfig:
    epochs: int = 80
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 12
    normalize_method: str = "per_window_channel_zscore"
    eps: float = 1e-6
    background_to_seizure_ratio: float = 5.0
    exclude_seconds_around_seizure: float = 60.0
    monitor_max_neg_per_pos: int = 20  # subsample val negatives for the epoch monitor only
    num_workers: int = 0
    device: str = "cpu"
    eval_batch_size: int = 256


def _loader(table, store, cfg: TrainConfig, shuffle=False, sampler=None,
            transform=None, transform_takes_label=False):
    ds = WindowDataset(table, store, cfg.normalize_method, cfg.eps,
                       transform=transform, transform_takes_label=transform_takes_label)
    return DataLoader(ds, batch_size=cfg.batch_size,
                      shuffle=(shuffle and sampler is None), sampler=sampler,
                      num_workers=cfg.num_workers, drop_last=False)


def infer_scores(model, table, store, cfg: TrainConfig) -> np.ndarray:
    """Sigmoid scores aligned to ``table`` row order."""
    if len(table) == 0:
        return np.empty((0,), dtype="float32")
    model.eval()
    ds = WindowDataset(table, store, cfg.normalize_method, cfg.eps)
    loader = DataLoader(ds, batch_size=cfg.eval_batch_size, shuffle=False,
                        num_workers=cfg.num_workers)
    out = []
    device = cfg.device
    with torch.no_grad():
        for x, _ in loader:
            logits = model(x.to(device))
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out).astype("float32")


def build_file_predictions(table, scores, index_df) -> List[FilePrediction]:
    """Group per-window scores into per-file timelines for event scoring."""
    import json

    sr_meta = index_df.set_index("file_id")
    preds: List[FilePrediction] = []
    tab = table.copy()
    tab["__score"] = scores
    for file_id, sub in tab.groupby("file_id"):
        sub = sub.sort_values("center_time_sec")
        meta = sr_meta.loc[file_id]
        ref_events = [tuple(map(float, e)) for e in json.loads(meta["seizures"])]
        preds.append(FilePrediction(
            file_id=str(file_id),
            patient=str(sub["patient"].iloc[0]),
            duration_sec=float(meta["duration_sec"]),
            center_times=sub["center_time_sec"].to_numpy(),
            scores=sub["__score"].to_numpy(),
            ref_events=ref_events,
        ))
    return preds


def _monitor_table(val_windows, ratio: int, seed: int):
    pos = val_windows[val_windows["label"] == 1]
    neg = val_windows[val_windows["label"] == 0]
    n_neg = min(len(neg), max(1, ratio * max(1, len(pos))))
    if len(neg) > n_neg:
        neg = neg.sample(n=n_neg, random_state=seed)
    return pd.concat([pos, neg]).reset_index(drop=True)


def train_model(model, train_dataset, val_monitor_table, store, loss_fn, cfg: TrainConfig,
                seed: int = 42) -> Dict[str, object]:
    torch.manual_seed(seed)
    device = cfg.device
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                            weight_decay=cfg.weight_decay)
    train_loader = train_dataset  # already a DataLoader
    best_auprc, best_state, best_epoch, since = -1.0, None, -1, 0
    history = []

    y_val = val_monitor_table["label"].to_numpy().astype(int)
    for epoch in range(cfg.epochs):
        model.train()
        tot = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).float()
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(y)
        scores = infer_scores(model, val_monitor_table, store, cfg)
        from sklearn.metrics import average_precision_score

        auprc = (float(average_precision_score(y_val, scores))
                 if len(np.unique(y_val)) == 2 else float("nan"))
        history.append({"epoch": epoch, "train_loss": tot / max(1, len(val_monitor_table)),
                        "val_auprc": auprc})
        improved = auprc == auprc and auprc > best_auprc
        if improved:
            best_auprc, best_epoch, since = auprc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since += 1
            if since >= cfg.early_stopping_patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_auprc": best_auprc, "best_epoch": best_epoch, "history": history}


@dataclass
class CellSpec:
    fold: int
    seed: int
    scarcity_fraction: float
    detector: str
    condition: str
    generator: Optional[str] = None
    synthetic_ratio: float = 1.0


def _assemble_training(condition, train_table, store, cfg, seed, synthetic=None):
    """Return (train_loader, loss_fn) for a condition. ``synthetic`` = (X, y) arrays."""
    labels = train_table["label"].to_numpy().astype(int)
    pos_w = pos_weight_from_labels(labels)

    if condition == "real_only":
        return _loader(train_table, store, cfg, shuffle=True), make_loss("bce")
    if condition == "class_weighted":
        return _loader(train_table, store, cfg, shuffle=True), make_loss("focal", pos_weight=pos_w)
    if condition == "balanced_sampler":
        sampler = make_balanced_sampler(labels)
        return _loader(train_table, store, cfg, sampler=sampler), make_loss("bce")
    if condition == "classical_aug":
        aug = ClassicalAugment(ClassicalAugConfig(), seed=seed, ictal_only=True)
        return (_loader(train_table, store, cfg, shuffle=True, transform=aug,
                        transform_takes_label=True), make_loss("bce"))
    if condition == "synthetic_aug":
        if synthetic is None:
            raise ValueError("synthetic_aug requires synthetic windows")
        real_ds = WindowDataset(train_table, store, cfg.normalize_method, cfg.eps)
        Xs, ys = synthetic
        synth_ds = ArrayDataset(Xs, ys)
        ds = ConcatDataset([real_ds, synth_ds])
        loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.num_workers)
        return loader, make_loss("bce")
    raise ValueError(f"unknown condition: {condition}")


def run_cell(
    spec: CellSpec,
    index_df,
    windows_df,
    events_df,
    store: str,
    split,
    cfg: Optional[TrainConfig] = None,
    postproc: Optional[PostprocConfig] = None,
    params: Optional[SzCoreParams] = None,
    fa_budgets=(0.5, 1.0, 2.0),
    synthetic_provider: Optional[Callable] = None,
    model_kwargs: Optional[dict] = None,
) -> Dict[str, object]:
    cfg = cfg or TrainConfig()
    postproc = postproc or PostprocConfig()
    params = params or SzCoreParams()
    model_kwargs = model_kwargs or {}

    train_ids = set(index_df[index_df["group"].isin(split.train_groups)]["file_id"])
    val_ids = set(index_df[index_df["group"].isin(split.val_groups)]["file_id"])
    test_ids = set(index_df[index_df["group"].isin(split.test_groups)]["file_id"])

    train_windows = windows_df[windows_df["file_id"].isin(train_ids) & (~windows_df["excluded"])]
    val_windows = windows_df[windows_df["file_id"].isin(val_ids)]
    test_windows = windows_df[windows_df["file_id"].isin(test_ids)]

    # Event-level scarcity on TRAIN groups, then negative sampling.
    selected = select_seizure_events(
        events_df[events_df["group"].isin(split.train_groups)],
        spec.scarcity_fraction, spec.seed, restrict_groups=split.train_groups,
    )
    scarce = apply_scarcity_to_windows(train_windows, selected)
    train_table = negative_sample(
        scarce, ratio=cfg.background_to_seizure_ratio,
        exclude_seconds=cfg.exclude_seconds_around_seizure, seed=spec.seed,
    )

    # Synthetic windows (training-only) if requested.
    synthetic = None
    if spec.condition == "synthetic_aug":
        n_pos = int((train_table["label"] == 1).sum())
        n_synth = max(1, int(round(spec.synthetic_ratio * n_pos)))
        if synthetic_provider is None:
            raise ValueError("synthetic_aug requires a synthetic_provider")
        Xs = synthetic_provider(n_synth, spec.seed)
        synthetic = (Xs, np.ones(len(Xs), dtype="int64"))

    win_samples = int(windows_df["end_sample"].iloc[0] - windows_df["start_sample"].iloc[0])
    model = build_model(spec.detector, n_channels=index_df_n_channels(store),
                        n_samples=win_samples, **model_kwargs)

    train_loader, loss_fn = _assemble_training(
        spec.condition, train_table, store, cfg, spec.seed, synthetic
    )
    monitor = _monitor_table(val_windows, cfg.monitor_max_neg_per_pos, spec.seed)
    train_log = train_model(model, train_loader, monitor, store, loss_fn, cfg, seed=spec.seed)

    # Threshold selection on validation event F1.
    val_scores = infer_scores(model, val_windows, store, cfg)
    val_preds = build_file_predictions(val_windows, val_scores, index_df)
    sel = select_threshold(val_preds, postproc, params)

    # Test evaluation on full timelines.
    test_scores = infer_scores(model, test_windows, store, cfg)
    test_preds = build_file_predictions(test_windows, test_scores, index_df)
    event_res = evaluate_event_level(test_preds, sel.selected_threshold, postproc, params)
    win_m = window_metrics(test_windows["label"].to_numpy(), test_scores, sel.selected_threshold)
    patient_arrays = _patient_arrays(test_windows, test_scores)
    patient_win = aggregate_patient_metrics(
        per_patient_window_metrics(patient_arrays, sel.selected_threshold)
    )
    fa_sens = sensitivity_at_fa_budgets(test_preds, fa_budgets, postproc=postproc, params=params)

    return {
        "spec": spec.__dict__,
        "n_train_windows": int(len(train_table)),
        "n_train_pos": int((train_table["label"] == 1).sum()),
        "n_params": count_parameters(model),
        "selected_threshold": sel.selected_threshold,
        "threshold_selection": sel.as_dict(),
        "train_log": {"best_val_auprc": train_log["best_val_auprc"],
                       "best_epoch": train_log["best_epoch"]},
        "window_metrics": win_m,
        "event_metrics": event_res["aggregate"],
        "event_metrics_per_patient": event_res["per_patient"],
        "patient_window_metrics": patient_win,
        "sensitivity_at_fa": {str(k): v for k, v in fa_sens.items()},
    }


def _patient_arrays(test_windows, test_scores):
    tw = test_windows.reset_index(drop=True)
    out = {}
    for pat, idx in tw.groupby("patient").groups.items():
        rows = np.asarray(list(idx))
        out[str(pat)] = (tw.loc[rows, "label"].to_numpy(), test_scores[rows])
    return out


def index_df_n_channels(store: str) -> int:
    import zarr

    root = zarr.open_group(str(store), mode="r")
    return int(root.attrs["n_channels"])
