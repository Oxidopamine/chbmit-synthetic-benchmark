"""Channel montage audit (plan Section 13).

Given a manifest, decide whether the candidate 18-channel bipolar montage is
acceptable under the ``drop_file`` missing-channel policy. A file is dropped if
it lacks any montage channel; the montage is accepted only if dropping those
files loses <= ``max_dropped_fraction`` of BOTH seizure events and seizure
duration. We also emit the artifacts Section 13 requires:

    channel_coverage_report.csv   per-file present/missing montage channels
    channel_alias_map.json        raw label -> canonical (or null)
    final_channel_set.json        the accepted montage channel order

If the gate fails we surface a recommended reduced montage (channels present in
all included files) so the montage can be revised before experiments run.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from chbmit.channel_map import CANONICAL_MONTAGE_18, canonical_channel, match_montage


@dataclass
class AuditResult:
    montage: List[str]
    accepted: bool
    max_dropped_fraction: float
    n_files_total: int
    n_files_kept: int
    n_files_dropped: int
    dropped_seizure_event_fraction: float
    dropped_seizure_duration_fraction: float
    final_channel_set: List[str]
    alias_map: Dict[str, Optional[str]]
    channel_presence: Dict[str, int]  # montage channel -> # included files containing it
    recommended_full_coverage_montage: List[str]
    coverage_rows: List[dict] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "montage": self.montage,
            "accepted": self.accepted,
            "max_dropped_fraction": self.max_dropped_fraction,
            "n_files_total": self.n_files_total,
            "n_files_kept": self.n_files_kept,
            "n_files_dropped": self.n_files_dropped,
            "dropped_seizure_event_fraction": self.dropped_seizure_event_fraction,
            "dropped_seizure_duration_fraction": self.dropped_seizure_duration_fraction,
            "final_channel_set": self.final_channel_set,
            "channel_presence": self.channel_presence,
            "recommended_full_coverage_montage": self.recommended_full_coverage_montage,
        }


def audit_channels(
    manifest_df,
    montage: Sequence[str] = CANONICAL_MONTAGE_18,
    max_dropped_fraction: float = 0.10,
) -> AuditResult:
    montage = list(montage)
    inc = manifest_df[manifest_df["included"]].copy()

    total_events = float(inc["n_seizures"].sum())
    total_duration = float(inc["seizure_duration_sec"].sum())

    alias_map: Dict[str, Optional[str]] = {}
    presence = {ch: 0 for ch in montage}
    coverage_rows: List[dict] = []
    dropped_events = 0.0
    dropped_duration = 0.0
    n_kept = 0

    for _, row in inc.iterrows():
        raw_labels = json.loads(row["channels_hdr"]) or json.loads(row["channels_summary"])
        for lbl in raw_labels:
            if lbl not in alias_map:
                alias_map[lbl] = canonical_channel(lbl)
        selected, missing = match_montage(raw_labels, montage)
        present = [c for c, _ in selected]
        for c in present:
            presence[c] += 1
        kept = len(missing) == 0
        if kept:
            n_kept += 1
        else:
            dropped_events += float(row["n_seizures"])
            dropped_duration += float(row["seizure_duration_sec"])
        coverage_rows.append({
            "edf_path": row["edf_path"],
            "patient": row["patient"],
            "group": row["group"],
            "n_present": len(present),
            "n_missing": len(missing),
            "kept": kept,
            "missing_channels": ";".join(missing),
            "n_seizures": int(row["n_seizures"]),
        })

    frac_events = (dropped_events / total_events) if total_events > 0 else 0.0
    frac_duration = (dropped_duration / total_duration) if total_duration > 0 else 0.0
    accepted = (frac_events <= max_dropped_fraction) and (frac_duration <= max_dropped_fraction)

    n_inc = int(len(inc))
    full_coverage = [ch for ch in montage if presence[ch] == n_inc] if n_inc else list(montage)

    return AuditResult(
        montage=montage,
        accepted=accepted,
        max_dropped_fraction=max_dropped_fraction,
        n_files_total=n_inc,
        n_files_kept=n_kept,
        n_files_dropped=n_inc - n_kept,
        dropped_seizure_event_fraction=frac_events,
        dropped_seizure_duration_fraction=frac_duration,
        final_channel_set=list(montage) if accepted else full_coverage,
        alias_map=alias_map,
        channel_presence=presence,
        recommended_full_coverage_montage=full_coverage,
        coverage_rows=coverage_rows,
    )


def write_audit(result: AuditResult, out_dir: str | Path) -> Path:
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cov_path = out_dir / "channel_coverage_report.csv"
    pd.DataFrame(result.coverage_rows).to_csv(cov_path, index=False)
    (out_dir / "channel_alias_map.json").write_text(
        json.dumps(result.alias_map, indent=2), encoding="utf-8"
    )
    (out_dir / "final_channel_set.json").write_text(
        json.dumps(result.final_channel_set, indent=2), encoding="utf-8"
    )
    (out_dir / "channel_audit_summary.json").write_text(
        json.dumps(result.summary(), indent=2), encoding="utf-8"
    )
    return cov_path


if __name__ == "__main__":  # pragma: no cover
    import argparse

    from chbmit.make_manifest import build_manifest

    ap = argparse.ArgumentParser(description="Audit CHB-MIT channel montage coverage")
    ap.add_argument("--raw-root", default="data/raw_chbmit")
    ap.add_argument("--out-dir", default="results_chbmit_synthetic/manifests")
    ap.add_argument("--max-dropped-fraction", type=float, default=0.10)
    args = ap.parse_args()
    df = build_manifest(args.raw_root)
    res = audit_channels(df, max_dropped_fraction=args.max_dropped_fraction)
    write_audit(res, args.out_dir)
    print(json.dumps(res.summary(), indent=2))
