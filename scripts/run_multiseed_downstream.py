"""Multi-seed x multi-fold downstream gated help/harm confirmation run.

The fold-0/single-seed downstream experiment (run_downstream_gated.py) produced one
"upside" cell (TCN, gated q=0.90: test event-F1 0.174->0.204). Its +0.03 gain is within
the between-fold noise (+/-0.11), so it must be replicated across seeds and folds before
it can be believed. This runs:

  folds {0,1,2} x seeds {42,123,2024} x detectors {eegnet, lct, tcn}
  x {real_only, class_weighted, classical_aug}                        (inject nothing)
  x --ratios x {ungated, gated x --qs, random_gated x --random-qs}    (synthetic arms)

Defaults reproduce the Phase 1 grid. The Phase 2 ladder is
``--ratios 0.02 0.05 0.10 0.20 0.30 --qs 0.90`` (18 conditions per block).

class_weighted and classical_aug are the PRE-REGISTERED harm reference (PREREGISTRATION.md
Sec 3 registers "the best simple baseline per (fold, seed)", not real_only). random_gated is
the matched-volume admission control: it injects exactly as many windows as its gated sibling
at the same (q, ratio), but draws them uniformly from the same pool, so admission QUALITY is
separated from DOSE. Two axes matter and Phase 1 got both wrong:

  q      -- the control ran only at q=0.90, where the gate admits ~3% of the intended dose
            (median 80 of ~2508 windows), so no selection rule could have moved the result.
  ratio  -- r was pinned at 1.0, which is 3.3x above the parent method's ladder ceiling of
            0.30, while gated q=0.90 realises r ~ 0.032. Nothing ever sampled r in
            [0.05, 0.30], the band the parent actually operates in, so the negative result
            against the simple baselines is confounded with an out-of-range dose.

See reports/DECISION_GATE_1.md CORRECTION 2 and reports/LITERATURE_AUDIT_2026-08-29.md.

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
from experiments.environment import record_environment

# Paths are overridable so the same driver runs unchanged on a workstation, a pod, or a
# Vertex AI custom job (where code and data are staged onto the container's local disk).
# Defaults reproduce the original behaviour exactly.
STORE = os.environ.get("CHBMIT_STORE", "data/processed_chbmit_real/eeg.zarr")
PROC = os.environ.get("CHBMIT_PROC", "data/processed_chbmit_real/processed_index.csv")
RES = Path(os.environ.get("CHBMIT_RESULTS", "results_chbmit_synthetic/real_validation"))
OUT = RES / "analysis_tierB"
GEN_DIR = RES / "generators"
DEVICE = "cuda"
FRAC = 1.0   # overridden by --scarcity
SIMPLE_BASELINES = ("class_weighted", "classical_aug")   # registered harm reference
# Conditions that inject nothing, so they run once per (fold,seed,detector) whatever the grid.
# Everything else is a synthetic arm crossed with --ratios: ungated, gated x --qs, and
# random_gated x --random-qs. Block size is therefore computed, never a literal.
FIXED_CONDS = ("real_only", "class_weighted", "classical_aug")


def n_conds(qs, random_qs, ratios) -> int:
    """Rows in a COMPLETE (fold,seed,detector) block for this invocation.

    Computed from the actual grid so resume-completeness stays correct when any axis changes on
    the command line. A literal would both fail to mark a reduced run complete and wrongly mark
    a smaller legacy block complete under a larger grid -- e.g. a 7-row Phase 1 block must NOT
    count as complete once a random control or a ratio rung is added.
    """
    return len(FIXED_CONDS) + len(ratios) * (1 + len(qs) + len(random_qs))


def _evt(res):
    em = res.get("event_metrics", {}) or {}
    g = res.get("gate")
    # Validation metrics of the model whose TEST metrics this row reports. Needed so the
    # harm reference can be chosen on validation and reported on test -- selecting the
    # reference on the metric being reported is the bias documented in DECISION_GATE_1.md.
    ts = res.get("threshold_selection", {}) or {}
    aug = (res.get("gated_model_metrics", {}) or {}).get("event_metrics", em) if res.get(
        "reverted_to_real_only") else em
    return {
        "event_f1": em.get("event_f1"), "event_sensitivity": em.get("event_sensitivity"),
        "event_precision": em.get("event_precision"), "fp_per_24h": em.get("fp_per_24h"),
        "val_event_f1": ts.get("validation_event_f1"),
        "val_fp_per_24h": ts.get("validation_fp_per_24h"),
        # Threshold-free selector statistic. The fail-closed rule as built compares two maxima
        # of a 19-point sweep; emitting AUPRC per arm lets the parent's threshold-free rule
        # (margin 0.01) be evaluated in ANALYSIS instead of costing a second grid.
        "val_auprc": ts.get("validation_auprc"),
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
    # Matched-volume control per gated arm. Defaults to EVERY q in --qs: Phase 1 ran the control
    # only at q=0.90, where the gate admits ~3% of the intended dose, so the null it produced was
    # underpowered by construction -- at q=0.50 the gate admits the full dose and the control can
    # actually discriminate. See reports/DECISION_GATE_1.md, "CORRECTION 2".
    ap.add_argument("--random-qs", nargs="*", type=float, default=None,
                    help="qs to run random_gated at (default: same as --qs)")
    # Injection ratio r = n_synth / n_train_pos, crossed with every synthetic arm. Default [1.0]
    # reproduces the Phase 1 grid exactly. The parent method's validation ladder selects r <= 0.30
    # with a median injection ~58 windows, so r = 1.0 sits 3.3x above its ceiling while gated
    # q=0.90 realises r ~ 0.032 -- nothing in this benchmark has ever sampled r in [0.05, 0.30],
    # which is where the parent actually operates. Phase 2 ladder:
    #   --ratios 0.02 0.05 0.10 0.20 0.30 --qs 0.90
    # See reports/LITERATURE_AUDIT_2026-08-29.md Sec 2.3 and 5.
    # DO NOT PASS 0. run_cell floors the count at `max(1, round(r * n_pos))`, so r=0 injects one
    # window rather than none. The r=0 rung of the ladder is the `real_only` arm, which every
    # block already runs -- use that in analysis instead.
    ap.add_argument("--ratios", nargs="*", type=float, default=[1.0],
                    help="synthetic injection ratios to cross with every synthetic arm "
                         "(do not pass 0; the r=0 rung is the real_only arm)")
    # Admission reference. TrustGateConfig has carried this field since Phase 1 but the driver
    # never set it, so every run so far used "real_ictal" -- a threshold calibrated on windows the
    # teacher has MEMORISED. Measured consequence: the gate admits 6 windows against a target of
    # 251 and 23 against 752, i.e. the admitted count is decoupled from the requested dose, and no
    # ratio ladder can move it (reports/DECISION_GATE_1.md CORRECTION 2; Phase 2 _p2 results).
    # With "pool" -- the rank cut TGA actually publishes -- admitted = min(oversample*(1-q), 1) *
    # n_synth exactly, so q becomes direct dose control: q = 1 - r/oversample hits any target r.
    # DEFAULT FLIPPED TO "pool" 2026-08-31. It used to default to "real_ictal", so a clone of
    # this repository reproduced the disabled mechanism -- the exact silent failure the project
    # is about. Pass --gate-reference real_ictal to reproduce Phases 1-2.
    ap.add_argument("--gate-reference", default="pool", choices=["pool", "real_ictal"],
                    help="admission reference distribution (default: pool, the published TGA "
                         "rank cut; real_ictal reproduces Phases 1-2 and cannot inject a dose)")
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
    # Phase 3 axes. --splits selects the fold file: the committed splits_seed42.json (Phases 1-2,
    # carved validation -- 7 of 23 groups ever serve as validation), splits_rot5_seed42.json (same
    # test folds, rotating validation) or splits_logo23_seed42.json (leave-one-group-out, the
    # registered Phase 3 design). Fold indices in --folds index into whichever file is given.
    ap.add_argument("--splits", default=str(RES / "splits/splits_seed42.json"),
                    help="split file; see scripts/make_phase3_splits.py for the Phase 3 files")
    ap.add_argument("--scarcity", type=float, default=1.0,
                    help="event-level scarcity fraction for the real training set (default 1.0)")
    # The same-seed floor (PREREGISTRATION_PHASE3.md Sec 2, item 1). Offsetting ONLY the seed the
    # synthetic provider draws with leaves weight initialisation (torch.manual_seed(spec.seed)
    # before build_model), the real training table and the shuffle order identical, and changes
    # nothing but which synthetic windows are injected. Pairing a cell against the same cell at
    # offset 0 therefore measures the floor that applies to a paired synthetic-vs-reference delta,
    # which the different-initialisation floor (sigma 0.10-0.17) does not.
    ap.add_argument("--synth-seed-offset", type=int, default=0,
                    help="added to the provider seed only; non-zero runs the same-seed floor arm")
    args = ap.parse_args()
    random_qs = args.qs if args.random_qs is None else args.random_qs
    global FRAC
    FRAC = args.scarcity
    out_csv = OUT / f"downstream_gated{args.tag}.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    # Environment record beside the CSV, before the first cell (AUDIT_2026-08-31 F7).
    record_environment(OUT / f"run_environment{args.tag}.json")

    t0 = time.time()
    index_df = pd.read_csv(PROC)
    win = pd.read_csv(RES / "windows/windows.csv")
    ev = pd.read_csv(RES / "windows/events.csv")
    sp = json.load(open(args.splits))
    print(f"[splits] {args.splits}: {sp.get('n_folds')} folds, "
          f"val_strategy={sp.get('val_strategy', 'carve')}, scarcity={FRAC}", flush=True)
    fs = int(zarr.open_group(STORE, mode="r").attrs["sampling_rate"])
    cfg = TrainConfig(epochs=args.epochs, device=DEVICE, num_workers=args.num_workers)

    # Resume: keep only rows from fully-complete (fold,seed,detector) blocks.
    n_expected = n_conds(args.qs, random_qs, args.ratios)
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

            # WGAN-GP for (fold,seed): load cached checkpoint or train + persist. Skipped
            # entirely when no synthetic arm is requested (`--ratios` with no values: the
            # baseline-only re-run), so that job never pays for a generator it does not use.
            # The checkpoint is keyed by the split FILE as well as (fold, seed): a fold index means
            # a different training set under a different file, and a generator fit on one must
            # never be reused for the other (leakage rule 4).
            split_tag = Path(args.splits).stem.replace("splits_", "")
            gkey = (f"wgan_f{fold}_s{seed}" if split_tag == "seed42" and FRAC == 1.0
                    else f"wgan_{split_tag}_sc{FRAC:g}_f{fold}_s{seed}")
            gdir = GEN_DIR / gkey
            bl_wgan = None
            if args.ratios:
                wgan = build_provider("wgan_gp", WGANConfig(epochs=args.gen_epochs,
                                                            min_ictal_windows=256,
                                                            device=DEVICE, seed=seed))
                if (gdir / "generator.pt").exists():
                    wgan._load_state(gdir)
                    wgan.fitted = True
                    print(f"[{time.time()-t0:6.0f}s] f{fold} s{seed}: WGAN loaded from cache {gkey}", flush=True)
                else:
                    fit_provider_for_cell(wgan, index_df, win, ev, STORE, split, fold, seed, FRAC)
                    gdir.mkdir(parents=True, exist_ok=True)
                    wgan._save_state(gdir)
                    print(f"[{time.time()-t0:6.0f}s] f{fold} s{seed}: WGAN trained+saved {gkey}", flush=True)

                def bl_wgan(n, s, _w=wgan, _off=args.synth_seed_offset):
                    return band_limit_windows(_w.generate(n, seed=s + _off), fs=fs)

            for det in args.detectors:
                if (fold, seed, det) in done_blocks:
                    continue

                def _spec(cond, **kw):
                    return CellSpec(fold=fold, seed=seed, scarcity_fraction=FRAC,
                                    detector=det, condition=cond, **kw)
                base = {"fold": fold, "seed": seed, "detector": det,
                        "synth_seed_offset": args.synth_seed_offset, "scarcity": FRAC,
                        "splits_file": Path(args.splits).name}

                teacher_res = run_cell(_spec("real_only"), index_df, win, ev, STORE, split,
                                       cfg=cfg, return_model=True)
                teacher_model = teacher_res.pop("model", None)
                rows.append({**base, "condition": "real_only", "q": None, **_evt(teacher_res)})

                # Registered simple baselines -- generator-independent, no synthetic injected.
                for simple in SIMPLE_BASELINES:
                    r = run_cell(_spec(simple), index_df, win, ev, STORE, split, cfg=cfg)
                    rows.append({**base, "condition": simple, "q": None, **_evt(r)})

                # Every synthetic arm is crossed with the injection ratio r. r is the parent
                # method's real knob (its validation ladder selects r <= 0.30); this grid used to
                # pin r = 1.0, which is 3.3x above that ceiling, leaving r in [0.05, 0.30]
                # unsampled -- the coverage hole the negative result currently rests on.
                for ratio in args.ratios:
                    rb = {**base, "ratio": ratio}
                    r = run_cell(_spec(UNGATED_SYNTHETIC, generator="wgan_bl",
                                       synthetic_ratio=ratio),
                                 index_df, win, ev, STORE, split, cfg=cfg,
                                 synthetic_provider=bl_wgan)
                    rows.append({**rb, "condition": "ungated", "q": None, **_evt(r)})

                    for q in args.qs:
                        r = run_cell(_spec(GATED_SYNTHETIC, generator="wgan_bl",
                                           synthetic_ratio=ratio),
                                     index_df, win, ev, STORE, split, cfg=cfg,
                                     synthetic_provider=bl_wgan, teacher_model=teacher_model,
                                     teacher_result=copy.deepcopy(teacher_res),
                                     gate_cfg=TrustGateConfig(q=q,
                                                              reference=args.gate_reference))
                        rows.append({**rb, "condition": "gated", "q": q, **_evt(r)})

                    # Matched-volume control, one per gated arm at the SAME ratio: same pool,
                    # same teacher, same admitted COUNT as that (q, ratio) sibling (identical
                    # seed => identical pool), uniformly random selection. The control is only
                    # matched if it shares the teacher that set the count, so it must be run in
                    # the SAME block as its sibling -- never bolted onto an earlier run's rows.
                    for q in random_qs:
                        r = run_cell(_spec(GATED_SYNTHETIC, generator="wgan_bl",
                                           synthetic_ratio=ratio),
                                     index_df, win, ev, STORE, split, cfg=cfg,
                                     synthetic_provider=bl_wgan, teacher_model=teacher_model,
                                     teacher_result=copy.deepcopy(teacher_res),
                                     gate_cfg=TrustGateConfig(q=q, selection="random",
                                                              selection_seed=seed,
                                                              reference=args.gate_reference))
                        rows.append({**rb, "condition": "random_gated", "q": q, **_evt(r)})

                done_blocks.add((fold, seed, det))
                pd.DataFrame(rows).to_csv(out_csv, index=False)
                rr = rows[-1]
                print(f"[{time.time()-t0:6.0f}s] DONE f{fold} s{seed} {det}: "
                      f"real={[x for x in rows if x['fold']==fold and x['seed']==seed and x['detector']==det and x['condition']=='real_only'][0]['event_f1']:.3f}",
                      flush=True)

        # Per-fold backup: commit + push CSV and this fold's WGAN checkpoints.
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        if not args.no_backup:
            ckpts = [p for p in GEN_DIR.glob(f"wgan_*f{fold}_s*") if p.is_dir()]
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
