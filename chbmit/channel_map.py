"""Channel-label normalization and montage matching for CHB-MIT.

CHB-MIT EDF channels are bipolar pairs (e.g. ``FP1-F7``). Real files carry a
range of label quirks the audit (Section 13) must tolerate:

* case and whitespace (``Fp1-F7``, ``FP1 - F7``)
* an ``EEG `` prefix or ``-REF`` / ``-LE`` suffix on some referential exports
* CHB-MIT's duplicated channels exported with a dedup digit (``T8-P8-0`` /
  ``T8-P8-1``)
* the old 10-20 nomenclature ``T3/T4/T5/T6`` vs. modern ``T7/T8/P7/P8``

Pure stdlib so it is importable before the heavy deps install.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "CANONICAL_MONTAGE_18",
    "canonical_electrode",
    "canonical_channel",
    "build_alias_map",
    "match_montage",
]

# Candidate 18-channel bipolar montage (plan Section 13). Already canonical.
CANONICAL_MONTAGE_18: List[str] = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FZ-CZ", "CZ-PZ",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
]

# Old -> modern 10-20 electrode names.
_OLD_TO_NEW = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

_RE_EEG_PREFIX = re.compile(r"^EEG\s+", re.IGNORECASE)
_RE_REF_SUFFIX = re.compile(r"-(REF|LE)$", re.IGNORECASE)


def canonical_electrode(name: str) -> Optional[str]:
    """Normalize a single electrode label, mapping old 10-20 names to modern."""
    e = name.upper().strip()
    if not e:
        return None
    return _OLD_TO_NEW.get(e, e)


def canonical_channel(label: str) -> Optional[str]:
    """Normalize a bipolar channel label to ``A-B`` canonical form.

    Returns ``None`` for non-bipolar / non-EEG labels (``ECG``, ``VNS``, ``.``,
    ``LOC-ROC`` style with unknown electrodes), so the audit can drop them.
    """
    if label is None:
        return None
    s = _RE_EEG_PREFIX.sub("", label.upper().strip())
    s = _RE_REF_SUFFIX.sub("", s)
    s = s.replace(" ", "")
    if not s:
        return None
    parts = s.split("-")
    # Drop a trailing dedup digit suffix, e.g. "T8-P8-0" -> ["T8", "P8"].
    if len(parts) == 3 and parts[2].isdigit():
        parts = parts[:2]
    if len(parts) != 2:
        return None
    a, b = canonical_electrode(parts[0]), canonical_electrode(parts[1])
    if not a or not b:
        return None
    return f"{a}-{b}"


def build_alias_map(raw_labels: Sequence[str]) -> Dict[str, Optional[str]]:
    """Map each raw label to its canonical form (or None if undecodable)."""
    return {lbl: canonical_channel(lbl) for lbl in raw_labels}


def match_montage(
    raw_labels: Sequence[str],
    montage: Sequence[str] = CANONICAL_MONTAGE_18,
) -> Tuple[List[Tuple[str, int]], List[str]]:
    """Match available raw labels against a montage.

    Returns ``(selected, missing)`` where ``selected`` is an ordered list of
    ``(montage_channel, source_index)`` (montage order preserved, first matching
    source index used for duplicates) and ``missing`` lists montage channels not
    present in ``raw_labels``.
    """
    canon_to_index: Dict[str, int] = {}
    for idx, lbl in enumerate(raw_labels):
        c = canonical_channel(lbl)
        if c is not None and c not in canon_to_index:
            canon_to_index[c] = idx  # keep first occurrence (dedup duplicates)

    selected: List[Tuple[str, int]] = []
    missing: List[str] = []
    for ch in montage:
        if ch in canon_to_index:
            selected.append((ch, canon_to_index[ch]))
        else:
            missing.append(ch)
    return selected, missing
