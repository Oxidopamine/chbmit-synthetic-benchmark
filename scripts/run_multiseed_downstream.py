"""Multi-seed x multi-fold downstream gated help/harm confirmation run.

The fold-0/single-seed downstream experiment (run_downstream_gated.py) produced one
"upside" cell (TCN, gated q=0.90: test event-F1 0.174->0.204). Its +0.03 gain is within
the between-fold noise (+/-0.11), so it must be replicated across seeds and folds before
it can be believed. This runs:

  folds {0,1,2} x seeds {42,123,2024} x detectors {eegnet, lct, tcn}
  x conditions {real_only, class_weighted, classical_aug, ungated,
                gated q=0.90, gated q=0.50, random_gated q=0.90}

class_weighted and classical_aug are the PRE-REGISTERED harm reference (PREREGISTRATION.md
Sec 3 registers "the best simple baseline per (fold, seed)", not real_only). random_gated is
the matched-volume admission control: it injects exactly as many windows as gated q=0.90 but
draws them uniformly from the same pool, so admission QUALITY is separated from DOSE.

with a band-limited WGAN-GP retrained PER (fold, seed) on that fold's training patients
(leakage-safe). Reports per-cell event-F1 / FP-24h + gate admission/decision, so the
gated-vs-real_only delta can be summarised with mean +/- CI and a paired test afterwards.

RunPod-safety: the processed store is on a network fs, so each fold's windows are
prefetched into the in-process cache once (per-fold; val/test shared across seeds).
The run is RESUMABLE -- completed (fold,seed,detector) blocks are skipped via the output
CSV, WGAN checkpoints are cached on disk, and results are committed+pushed after each fold
so a crash costs at most one fold.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import zarr

from chbmit.splits import Split
from chbmit.scarcity import select_seizure_events, apply_scarcity_to_windows
from chbmit.datasets import negative_sample, prefetch_windows, clear_window_cache
from synthetic.wgan_gp_provider import WGANConfig
from synthetic.train_provider import build_provider, fit_provider_for_cell
from synthetic.band_limit import band_limit_windows
from synthetic.trust_gate import TrustGateConfig
from experiments.training import (
    run_cell, CellSpec, TrainConfig, UNGATED_SYNTHETIC, GATED_SYNTHETIC)

# Paths are overridable so the same driver runs unchanged on a workstation, a pod, or a
# Vertex AI custom job (where code and data are staged onto the container's local disk).
# Defaults reproduce the original behaviour exactly.
STORE = os.environ.get("CHBMIT_STORE", "data/processed_chbmit_real/eeg.zarr")
PROC = os.environ.get("CHBMIT_PROC", "data/processed_chbmit_real/processed_index.csv")
RES = Path(os.environ.get("CHBMIT_RESULTS", "results_chbmit_synthetic/real_validation"))
OUT = RES / "analysis_tierB"
GEN_DIR = RES / "generators"
DEVICE = "cuda"
FRAC = 1.0
# Conditions per (fold,seed,detector): (condition_label, q). q=None for non-gated.
CONDS = [("real_only", None), ("class_weighted", None), ("classical_aug", None),
         ("ungated", None), ("gated", 0.90), ("gated", 0.50), ("random_gated", 0.90)]
SIMPLE_BASELINES = ("class_weighted", "classical_aug")   # registered harm reference
RANDOM_GATED_Q = 0.90                                    # control matches the core q


def n_conds(qs) -> int:
    """Rows in a COMPLETE (fold,seed,detector) block for this invocation.

    Derived from --qs rather than len(CONDS) so resume-completeness stays correct if the
    q grid is changed on the command line (a fixed 7 would never mark a 1-q run complete).
    """
    return len([c for c in CONDS if c[0] != "gated"]) + len(qs)


def _evt(res):
    em = res.get("event_metrics", {}) or {}
    g = res.get("gate")
    aug = (res.get("gated_model_metrics", {}) or {}).get("event_metrics", em) if res.get(
        "reverted_to_real_only") else em
    return {
        "event_f1": em.get("event_f1"), "event_sensitivity": em.get("event_sensitivity"),
        "event_precision": em.get("event_precision"), "fp_per_24h": em.get("fp_per_24h"),
        "aug_event_f1": aug.get("event_f1"), "aug_fp_per_24h": aug.get("fp_per_24h"),
        "aug_event_sensitivity": aug.get("event_sensitivity"),
        "reverted_to_real_only": res.get("reverted_to_real_only"),
        "gate_admission_rate": (g or {}).get("admission_rate"),
        "gate_n_admitted": (g or {}).get("n_admitted"),
        "gate_reason": (g or {}).get("reason"),
        "gate_reference": (g or {}).get("reference"),
        "gate_selection": (g or {}).get("selection"),
        "gate_min_admitted": (g or {}).get("min_admitted"),
    }


def _git_backup(paths, msg):
    """Commit + push the given paths (force-add past .gitignore). Best-effort."""
    try:
        subprocess.run(["git", "add", "-f", *[str(p) for p in paths]],
                       cwd=str(Path(__file__).resolve().parent.parent), check=True)
        r = subprocess.run(["git", "commit", "-q", "-m", msg],
                           cwd=str(Path(__file__).resolve().parent.parent))
        if r.returncode == 0:
            subprocess.run(["git", "push", "-q", "origin", "master"],
                           cwd=str(Path(__file__).resolve().parent.parent))
    except Exception as e:  # never let backup failure stop the experiment
        print(f"  [git backup failed: {e}]", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 123, 2024])
    ap.add_argument("--detectors", nargs="*", default=["eegnet", "lct", "tcn"])
    ap.add_argument("--qs", nargs="*", type=float, default=[0.90, 0.50])
    ap.add_argument("--epochs", type=int, default=80)          # detector training epochs
    ap.add_argument("--gen-epochs", type=int, default=300)     # WGAN epochs (matches Tier B)
    # DataLoader workers. KEEP THIS AT 0 FOR ANY RUN THAT WILL BE COMPARED WITH ANOTHER.
    # Measured, not assumed (fold 0 / seed 42 / lct / 3 epochs, identical otherwise): going from
    # 0 to 8 workers is 2.16x faster (1733 s -> 803 s per block) but CHANGES 6 of 7 conditions --
    # real_only event-F1 0.093776 -> 0.160457, and gate_n_admitted 198 -> 436. With workers,
    # PyTorch draws a base seed from the global RNG to seed them, which perturbs the same stream
    # that drives dropout and the shuffle permutation, so a different model is trained. All
    # published results use 0; this flag exists only for throughput experiments.
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--tag", default="_multiseed")
    ap.add_argument("--no-backup", action="store_true")  # skip git commit/push (for smoke tests)
    args = ap.parse_args()
    out_csv = OUT / f"downstream_gated{args.tag}.csv"
    OUT.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    index_df = pd.read_csv(PROC)
    win = pd.read_csv(RES / "windows/windows.csv")
    ev = pd.read_csv(RES / "windows/events.csv")
    sp = json.load(open(RES / "splits/splits_seed42.json"))
    fs = int(zarr.open_group(STORE, mode="r").attrs["sampling_rate"])
    cfg = TrainConfig(epochs=args.epochs, device=DEVICE, num_workers=args.num_workers)

    # Resume: keep only rows from fully-complete (fold,seed,detector) blocks.
    n_expected = n_conds(args.qs)
    rows = []
    done_blocks = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        for (fo, se, det), grp in prev.groupby(["fold", "seed", "detector"]):
            if len(grp) >= n_expected:
                rows.extend(grp.to_dict("records"))
                done_blocks.add((int(fo), int(se), str(det)))
        print(f"[resume] {len(done_blocks)} complete blocks loaded from {out_csv} "
              f"({n_expected} conditions per block)", flush=True)

    for fold in args.folds:
        f = sp["folds"][fold]
        split = Split(fold=fold, train_groups=f["train_groups"],
                      val_groups=f["val_groups"], test_groups=f["test_groups"])

        # Is any block in this fold still outstanding? If not, skip the (expensive) prefetch.
        outstanding = [(s, d) for s in args.seeds for d in args.detectors
                       if (fold, s, d) not in done_blocks]
        if not outstanding:
            print(f"[{time.time()-t0:6.0f}s] fold {fold} already complete; skipping", flush=True)
            continue

        # Prefetch val + test (seed-independent) once per fold.
        clear_window_cache()
        val_ids = set(index_df[index_df["group"].isin(split.val_groups)]["file_id"])
        test_ids = set(index_df[index_df["group"].isin(split.test_groups)]["file_id"])
        prefetch_windows(win[win["file_id"].isin(val_ids)], STORE, workers=16, dtype="float16")
        prefetch_windows(win[win["file_id"].isin(test_ids)], STORE, workers=16, dtype="float16")
        print(f"[{time.time()-t0:6.0f}s] fold {fold}: val/test prefetched", flush=True)

        train_ids = set(index_df[index_df["group"].isin(split.train_groups)]["file_id"])
        train_windows = win[win["file_id"].isin(train_ids) & (~win["excluded"])]

        for seed in args.seeds:
            if all((fold, seed, d) in done_blocks for d in args.detectors):
                continue
            # Prefetch this (fold,seed)'s real training table (small).
            selected = select_seizure_events(ev[ev["group"].isin(split.train_groups)], FRAC,
                                              seed, restrict_groups=split.train_groups)
            scarce = apply_scarcity_to_windows(train_windows, selected)
            train_table = negative_sample(scarce, ratio=cfg.background_to_seizure_ratio,
                                          exclude_seconds=cfg.exclude_seconds_around_seizure,
                                          seed=seed)
            prefetch_windows(train_table, STORE, workers=16, dtype="float16")

            # WGAN-GP for (fold,seed): load cached checkpoint or train + persist.
            gdir = GEN_DIR / f"wgan_f{fold}_s{seed}"
            wgan = build_provider("wgan_gp", WGANConfig(epochs=args.gen_epochs,
                                                        min_ictal_windows=256,
                                                        device=DEVICE, seed=seed))
            if (gdir / "generator.pt").exists():
                wgan._load_state(gdir)
                wgan.fitted = True
                print(f"[{time.time()-t0:6.0f}s] f{fold} s{seed}: WGAN loaded from cache", flush=True)
            else:
                fit_provider_for_cell(wgan, index_df, win, ev, STORE, split, fold, seed, FRAC)
                gdir.mkdir(parents=True, exist_ok=True)
                wgan._save_state(gdir)
                print(f"[{time.time()-t0:6.0f}s] f{fold} s{seed}: WGAN trained+saved", flush=True)

            def bl_wgan(n, s):
                return band_limit_windows(wgan.generate(n, seed=s), fs=fs)

            for det in args.detectors:
                if (fold, seed, det) in done_blocks:
                    continue

                def _spec(cond, **kw):
                    return CellSpec(fold=fold, seed=seed, scarcity_fraction=FRAC,
                                    detector=det, condition=cond, **kw)
                base = {"fold": fold, "seed": seed, "detector": det}

                teacher_res = run_cell(_spec("real_only"), index_df, win, ev, STORE, split,
                                       cfg=cfg, return_model=True)
                teacher_model = teacher_res.pop("model", None)
                rows.append({**base, "condition": "real_only", "q": None, **_evt(teacher_res)})

                # Registered simple baselines -- generator-independent, no synthetic injected.
                for simple in SIMPLE_BASELINES:
                    r = run_cell(_spec(simple), index_df, win, ev, STORE, split, cfg=cfg)
                    rows.append({**base, "condition": simple, "q": None, **_evt(r)})

                r = run_cell(_spec(UNGATED_SYNTHETIC, generator="wgan_bl", synthetic_ratio=1.0),
                             index_df, win, ev, STORE, split, cfg=cfg, synthetic_provider=bl_wgan)
                rows.append({**base, "condition": "ungated", "q": None, **_evt(r)})

                for q in args.qs:
                    r = run_cell(_spec(GATED_SYNTHETIC, generator="wgan_bl", synthetic_ratio=1.0),
                                 index_df, win, ev, STORE, split, cfg=cfg,
                                 synthetic_provider=bl_wgan, teacher_model=teacher_model,
                                 teacher_result=copy.deepcopy(teacher_res),
                                 gate_cfg=TrustGateConfig(q=q))
                    rows.append({**base, "condition": "gated", "q": q, **_evt(r)})

                # Matched-volume control: same pool, same teacher, same admitted COUNT as the
                # q=0.90 sibling (identical seed => identical pool), uniformly random selection.
                r = run_cell(_spec(GATED_SYNTHETIC, generator="wgan_bl", synthetic_ratio=1.0),
                             index_df, win, ev, STORE, split, cfg=cfg,
                             synthetic_provider=bl_wgan, teacher_model=teacher_model,
                             teacher_result=copy.deepcopy(teacher_res),
                             gate_cfg=TrustGateConfig(q=RANDOM_GATED_Q, selection="random",
                                                      selection_seed=seed))
                rows.append({**base, "condition": "random_gated", "q": RANDOM_GATED_Q, **_evt(r)})

                done_blocks.add((fold, seed, det))
                pd.DataFrame(rows).to_csv(out_csv, index=False)
                rr = rows[-1]
                print(f"[{time.time()-t0:6.0f}s] DONE f{fold} s{seed} {det}: "
                      f"real={[x for x in rows if x['fold']==fold and x['seed']==seed and x['detector']==det and x['condition']=='real_only'][0]['event_f1']:.3f}",
                      flush=True)

        # Per-fold backup: commit + push CSV and this fold's WGAN checkpoints.
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        if not args.no_backup:
            ckpts = [GEN_DIR / f"wgan_f{fold}_s{seed}" for seed in args.seeds]
            _git_backup([out_csv, *ckpts],
                        f"Multiseed downstream: fold {fold} complete ({args.tag})")
            print(f"[{time.time()-t0:6.0f}s] fold {fold} committed+pushed", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("\n=== MULTI-SEED DOWNSTREAM (band-limited WGAN) ===", flush=True)
    show = ["fold", "seed", "detector", "condition", "q", "event_f1", "fp_per_24h",
            "aug_event_f1", "gate_admission_rate", "reverted_to_real_only"]
    pd.set_option("display.width", 220, "display.max_rows", 200)
    print(df[show].to_string(index=False), flush=True)
    print(f"\n[{time.time()-t0:6.0f}s] wrote {out_csv}", flush=True)
    clear_window_cache()


if __name__ == "__main__":
    main()
