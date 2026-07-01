"""Shared grid runner used by every tier (plan Sections 8, 18, 20).

For each (fold, seed, scarcity) the generators are fit once on the exact training
subset (Rule 4) and reused across detectors. Synthetic conditions
(``ungated_synthetic_aug``, ``trust_gated_synthetic_aug``) iterate over generators;
skipped (below-threshold) generators are recorded, never silently dropped. The
real_only model is trained first and reused as the gated condition's teacher
(v5.3 Sec 5.1). Per-generator quality checks run alongside (Section 20).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from evaluation.event_metrics_szcore import PostprocConfig, SzCoreParams
from experiments.training import (
    GATED_SYNTHETIC,
    UNGATED_SYNTHETIC,
    CellSpec,
    TrainConfig,
    is_synthetic_condition,
    run_cell,
)
from synthetic.quality_checks import run_quality_suite
from synthetic.train_provider import build_provider, fit_provider_for_cell
from synthetic.trust_gate import TrustGateConfig

# real_only is trained first so it can serve as the gated condition's teacher (v5.3 Sec 5.1).
_CONDITION_ORDER = {"real_only": 0, "class_weighted": 1, "balanced_sampler": 1,
                    "classical_aug": 1, UNGATED_SYNTHETIC: 2, GATED_SYNTHETIC: 3}


@dataclass
class GridSpec:
    detectors: Sequence[str]
    conditions: Sequence[str]
    scarcity_fractions: Sequence[float]
    seeds: Sequence[int]
    generators: Sequence[str] = field(default_factory=list)
    synthetic_ratio: float = 1.0
    folds: Optional[Sequence[int]] = None  # None => all
    do_quality: bool = True


def run_grid(
    prepared,
    grid: GridSpec,
    train_cfg: Optional[TrainConfig] = None,
    gen_configs: Optional[Dict[str, object]] = None,
    postproc: Optional[PostprocConfig] = None,
    params: Optional[SzCoreParams] = None,
    fa_budgets=(0.5, 1.0, 2.0),
    results_dir: Optional[str | Path] = None,
    gate_cfg: Optional[TrustGateConfig] = None,
    log=print,
) -> Dict[str, list]:
    train_cfg = train_cfg or TrainConfig()
    gen_configs = gen_configs or {}
    gate_cfg = gate_cfg or TrustGateConfig()
    results: List[dict] = []
    quality: List[dict] = []
    folds = grid.folds if grid.folds is not None else range(len(prepared.splits))
    scarcity_dir = (Path(results_dir) / "scarcity") if results_dir else None
    quality_dir = (Path(results_dir) / "figures" / "quality") if results_dir else None

    conditions = sorted(grid.conditions, key=lambda c: _CONDITION_ORDER.get(c, 1))
    needs_gen = any(is_synthetic_condition(c) for c in conditions) and grid.generators
    needs_teacher = GATED_SYNTHETIC in conditions

    for fold in folds:
        split = prepared.splits[fold]
        for seed in grid.seeds:
            for frac in grid.scarcity_fractions:
                fitted: Dict[str, dict] = {}
                if needs_gen:
                    for gen in grid.generators:
                        log(f"[grid] fit generator={gen} fold={fold} seed={seed} frac={frac}")
                        prov = build_provider(gen, gen_configs.get(gen))
                        info = fit_provider_for_cell(
                            prov, prepared.index_df, prepared.windows_df, prepared.events_df,
                            prepared.store, split, fold, seed, frac,
                            normalize_method=train_cfg.normalize_method, eps=train_cfg.eps,
                            save_events_dir=str(scarcity_dir) if scarcity_dir else None,
                        )
                        fitted[gen] = info
                        if grid.do_quality and info["fitted"]:
                            _run_quality(info, gen, fold, seed, frac, prepared, quality,
                                         quality_dir, train_cfg)

                for detector in grid.detectors:
                    teacher_model = None
                    teacher_result = None

                    def _real_only_cell(record: bool):
                        spec = CellSpec(fold, seed, frac, detector, "real_only")
                        log(f"[grid] cell {detector} real_only fold={fold} seed={seed} frac={frac}")
                        res = run_cell(
                            spec, prepared.index_df, prepared.windows_df,
                            prepared.events_df, prepared.store, split, cfg=train_cfg,
                            postproc=postproc, params=params, fa_budgets=fa_budgets,
                            return_model=True,
                        )
                        model = res.pop("model", None)
                        if record:
                            results.append(res)
                        return model, res

                    for cond in conditions:
                        if cond == "real_only":
                            teacher_model, teacher_result = _real_only_cell(record=True)
                            continue

                        if is_synthetic_condition(cond):
                            if cond == GATED_SYNTHETIC and teacher_model is None:
                                teacher_model, teacher_result = _real_only_cell(record=False)
                            for gen in grid.generators:
                                info = fitted.get(gen, {})
                                if not info.get("fitted"):
                                    results.append(_skipped_record(
                                        fold, seed, frac, detector, cond, gen,
                                        reason=info.get("provider").metadata.extra.get("skip_reason")
                                        if info.get("provider") and info["provider"].metadata else "no_provider"))
                                    continue
                                prov = info["provider"]
                                spec = CellSpec(fold, seed, frac, detector, cond,
                                                generator=gen, synthetic_ratio=grid.synthetic_ratio)
                                log(f"[grid] cell {detector} {cond}/{gen} fold={fold} seed={seed} frac={frac}")
                                res = run_cell(
                                    spec, prepared.index_df, prepared.windows_df,
                                    prepared.events_df, prepared.store, split, cfg=train_cfg,
                                    postproc=postproc, params=params, fa_budgets=fa_budgets,
                                    synthetic_provider=lambda n, s, _p=prov: _p.generate(n, seed=s),
                                    teacher_model=(teacher_model if cond == GATED_SYNTHETIC else None),
                                    teacher_result=(teacher_result if cond == GATED_SYNTHETIC else None),
                                    gate_cfg=gate_cfg,
                                )
                                results.append(res)
                        else:
                            spec = CellSpec(fold, seed, frac, detector, cond)
                            log(f"[grid] cell {detector} {cond} fold={fold} seed={seed} frac={frac}")
                            res = run_cell(
                                spec, prepared.index_df, prepared.windows_df,
                                prepared.events_df, prepared.store, split, cfg=train_cfg,
                                postproc=postproc, params=params, fa_budgets=fa_budgets,
                            )
                            results.append(res)
    return {"results": results, "quality": quality}


def _run_quality(info, gen, fold, seed, frac, prepared, quality, quality_dir, train_cfg):
    real = info["real_ictal"]
    if real is None or len(real) < 8:
        return
    prov = info["provider"]
    synth = prov.generate(min(len(real), 500), seed=seed)
    fs = int(__import__("zarr").open_group(str(prepared.store), mode="r").attrs["sampling_rate"])
    out_dir = quality_dir if quality_dir else Path(".")
    qm = run_quality_suite(real, synth, fs, out_dir, f"{gen}_f{fold}_s{seed}_frac{frac}",
                           make_figures=quality_dir is not None)
    qm.update({"generator": gen, "fold": fold, "seed": seed, "scarcity_fraction": frac})
    quality.append(qm)


def _skipped_record(fold, seed, frac, detector, condition, gen, reason):
    return {
        "spec": {"fold": fold, "seed": seed, "scarcity_fraction": frac,
                 "detector": detector, "condition": condition, "generator": gen,
                 "synthetic_ratio": None},
        "skipped": True, "skip_reason": reason,
        "window_metrics": {}, "event_metrics": {}, "sensitivity_at_fa": {},
    }
