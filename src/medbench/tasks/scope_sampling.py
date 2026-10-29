"""Where the 5-min ECG epochs of a SCOPE stay come from (task scope-arrest@0). Pure: numpy and hashlib only, so the
same file runs inside compute pods (scripts/scope_build.py) and on the Mac.

All times are seconds relative to the outcome anchor that SCOPE puts at 2100-01-01 00:00:00 UTC (negative = before
the outcome). The anchor, not the end of the file, is the reference: 78% of arrest files end exactly an hour after the
outcome, against 33% of death and 12% of discharge files, so anything placed relative to the file end would reveal the
label.

- Arrest stays (label 1): contiguous epochs from 24 h to 0.5 h before the arrest (Lee et al. 2023, npj Digit Med).
- Other stays (label 0): one contiguous 24 h block at a position drawn uniformly over the stay's ECG record, from a
  random number that depends only on (seed, stay id); never anchored to the end of the record.
- No stay contributes data from the last 0.5 h before its outcome or after it.
"""
import hashlib
import math

import numpy as np

ANCHOR_UTC = 4102444800.0  # 2100-01-01 00:00:00 UTC
EPOCH_S = 300
BLACKOUT_S = 1800  # 0.5 h before the outcome: never used
CASE_HORIZON_S = 24 * 3600  # arrest epochs start no earlier than 24 h before the arrest
CONTROL_BLOCK_S = 24 * 3600
CASE_EPOCHS = (CASE_HORIZON_S - BLACKOUT_S) // EPOCH_S  # 282
CONTROL_EPOCHS = CONTROL_BLOCK_S // EPOCH_S  # 288
SEED = 20261029
ARREST = "Cardiac Arrest"
CONTROL_OUTCOMES = ("Discharge", "Death")


def label_for(outcome: str) -> int:
    if outcome == ARREST:
        return 1
    if outcome in CONTROL_OUTCOMES:
        return 0
    raise ValueError(f"unknown SCOPE outcome {outcome!r}")


def _usable(rec_start: float, rec_end: float) -> tuple[int, int]:
    """Whole seconds of the record that may be used: from its first ECG sample to 0.5 h before the outcome."""
    return math.ceil(rec_start), math.floor(min(rec_end, -BLACKOUT_S))


def case_epoch_starts(rec_start: float, rec_end: float) -> list[int]:
    lo, hi = _usable(rec_start, rec_end)
    starts = [-BLACKOUT_S - EPOCH_S * (k + 1) for k in range(CASE_EPOCHS)]
    return sorted(s for s in starts if s >= lo and s + EPOCH_S <= hi)


def control_block_fraction(case_id: int, seed: int = SEED) -> float:
    """u ~ U(0, 1) from the seed and the stay id alone: nothing about the record or its outcome enters the draw."""
    return float(np.random.default_rng([seed, int(case_id)]).random())


def control_epoch_starts(case_id: int, rec_start: float, rec_end: float, seed: int = SEED) -> list[int]:
    lo, hi = _usable(rec_start, rec_end)
    span = hi - lo
    if span < EPOCH_S:
        return []
    if span <= CONTROL_BLOCK_S:
        begin, n = lo, span // EPOCH_S
    else:
        begin, n = lo + int(control_block_fraction(case_id, seed) * (span - CONTROL_BLOCK_S)), CONTROL_EPOCHS
    return [begin + EPOCH_S * j for j in range(n)]


def epoch_starts(case_id: int, label: int, rec_start: float, rec_end: float, seed: int = SEED) -> list[int]:
    return case_epoch_starts(rec_start, rec_end) if label == 1 else control_epoch_starts(case_id, rec_start, rec_end, seed)


def instance_id(case_id: int, start_s: int) -> str:
    """Stable and opaque: neither the stay id nor the position relative to the outcome can be read off it."""
    return "scope-" + hashlib.sha256(f"scope-arrest|{int(case_id)}|{int(start_s)}".encode()).hexdigest()[:16]
