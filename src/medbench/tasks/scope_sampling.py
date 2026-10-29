"""Where the 5-min ECG epochs of a SCOPE stay come from (task scope-arrest@0), as in the dataset authors' SCOPE
processing script (Lee et al. 2023 setup). Pure: the standard library only, so the same file runs inside compute
pods and on a laptop.

The ECG is read at 250 Hz as one array per stay (vitaldb `to_numpy` with a 1/250 s interval).
- Arrest stays (label 1): the file is first cropped to the 24 h before the outcome anchor (2100-01-01 00:00 in the
  file's time zone; all SCOPE files but one are in UTC), then cut into 5-min epochs from the start of the crop;
  epochs ending 0.5-24 h before the arrest are kept (282 epochs).
- Discharge stays (label 0): one 24 h block at a position drawn with Python's `random` seeded with 42 plus the stay's
  rank among discharge stays in the metadata sheet; a record of 24 h or less is used whole. The block is cut into
  5-min epochs from its start.
- Death stays are not part of the cohort.
Epoch positions and minutes to the outcome are for audits only; they are never model inputs.
"""
import hashlib
import random
from collections.abc import Iterable

ANCHOR_UTC = 4102444800.0  # 2100-01-01 00:00:00 UTC
HZ = 250
EPOCH_S = 300
EPOCH_SAMPLES = EPOCH_S * HZ  # 75,000
HORIZON_MIN_S = 1800  # epochs end at least 0.5 h before the arrest
HORIZON_MAX_S = 24 * 3600  # and at most 24 h before it
BLOCK_SAMPLES = HORIZON_MAX_S * HZ  # the controls' 24 h block
CASE_EPOCHS = (HORIZON_MAX_S - HORIZON_MIN_S) // EPOCH_S  # 282
SEED = 42
ARREST, CONTROL = "Cardiac Arrest", "Discharge"
EXCLUDED = ("Death",)


def label_for(outcome: str) -> int | None:
    """1 for an arrest, 0 for a discharge, None for a stay outside the cohort (death)."""
    if outcome == ARREST:
        return 1
    if outcome == CONTROL:
        return 0
    if outcome in EXCLUDED:
        return None
    raise ValueError(f"unknown SCOPE outcome {outcome!r}")


def stay_seeds(case_ids: Iterable[int], outcomes: Iterable[str], seed: int = SEED) -> dict[int, int]:
    """seed + the stay's rank within its group (arrests, discharges) in the order given, which must be the order of
    the release's metadata sheet: the authors seed each stay's worker this way."""
    out, rank = {}, {ARREST: 0, CONTROL: 0}
    for case_id, outcome in zip(case_ids, outcomes):
        if label_for(outcome) is None:
            continue
        out[int(case_id)] = seed + rank[outcome]
        rank[outcome] += 1
    return out


def case_epochs(total_samples: int) -> list[tuple[int, float]]:
    """(start sample, minutes from the epoch's end to the arrest) in the 24 h crop ending at the arrest."""
    out = []
    for start in range(0, total_samples - EPOCH_SAMPLES + 1, EPOCH_SAMPLES):
        from_end = total_samples - (start + EPOCH_SAMPLES)
        if HORIZON_MIN_S * HZ <= from_end <= HORIZON_MAX_S * HZ:
            out.append((start, from_end / (HZ * 60)))
    return out


def control_block(total_samples: int, seed: int) -> tuple[int, int]:
    """[start, end) of the control's 24 h block in its whole record; the draw depends on the seed and the record
    length only."""
    if total_samples <= BLOCK_SAMPLES:
        return 0, total_samples
    start = random.Random(seed).randint(0, total_samples - BLOCK_SAMPLES)
    return start, start + BLOCK_SAMPLES


def control_epochs(block_samples: int) -> list[tuple[int, float]]:
    """(start sample within the block, minutes from the epoch's end to the block's end)."""
    return [(start, (block_samples - start - EPOCH_SAMPLES) / (HZ * 60))
            for start in range(0, block_samples - EPOCH_SAMPLES + 1, EPOCH_SAMPLES)]


def instance_id(case_id: int, start_tick: int) -> str:
    """Stable and opaque: neither the stay id nor the position (in 250 Hz samples relative to the anchor) can be read
    off it."""
    return "scope-" + hashlib.sha256(f"scope-arrest|{int(case_id)}|{int(start_tick)}".encode()).hexdigest()[:16]
