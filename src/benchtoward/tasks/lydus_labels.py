"""The LYDUS ECG tasks' labels: the dataset's own reviewed label columns, nothing BenchToward-made.

Owner decision (2026-11-04): lydus-ecg12-dx@0 and lydus-ecg-image@0 are scored only on the classes the release itself
provides (SNUH LYDUS ECG 50K 1.1.1, KHDP, doi:10.23421/2025.0704; reviewed by an anesthesiologist and an
intensivist), in its four categorical columns `rhythm`, `premature beat`, `bbb` and `avb`. Design:
docs/design/2026-11-04-lydus-provided-labels-design.md.

- 25 binary targets, one per definite value, named `column:value`, in the fixed order of TARGETS. A target is positive
  when the record's column holds exactly that value; `APC,VPC` makes both premature-beat targets positive. Composite
  bundle-branch values (BFB(...)) are their own targets and do not switch on their parts.
- Known and unknown: `NSP` in premature beat, bbb or avb means no named finding, so that column's targets are known
  negatives. In rhythm, `NSP`, `Possible pacing`, `Undetermined` and `Others` are not a definite rhythm, so every
  rhythm target of that record is masked (unknown). A value outside the release's vocabulary masks its whole column and
  is reported by `audit`.
- PRIMARY: the 15 targets with at least 20 test positives, 20 test negatives and 10 positive groups in the test split
  of both tasks (waveform and image), fixed before any model run; the other 10 are reported, never in the headline.
- Metrics: macro AUROC over PRIMARY (each target's AUROC on its known records, equally weighted), with per-target
  AUROC, and macro F1 over PRIMARY at thresholds tuned on validation (0.5 below 10 validation positives). 95%
  intervals from 1,000 whole-group bootstrap draws (seed 0); replicates where a target's AUROC is undefined are left
  out of that target, and a macro replicate needs every PRIMARY target defined (the share left out is reported).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ("rhythm", "premature beat", "bbb", "avb")
RHYTHM_UNKNOWN = ("NSP", "Possible pacing", "Undetermined", "Others")
DEFINITE = {
    "rhythm": ("A.fib", "Other sinus rhythm", "NSR", "S.brady", "Junctional rhythm", "A.flutter", "S.tachy",
               "S.arrhythmia", "SVT", "WPW"),
    "premature beat": ("APC", "VPC"),
    "bbb": ("RBBB", "LBBB", "LAFB", "BFB(RBBB+LAFB)", "BFB(RBBB+LPFB)", "LPFB", "BFB"),
    "avb": ("1AVB", "Variable AVB", "IVB", "2AVB", "3AVB", "AV dissociation"),
}
NEGATIVE = {"premature beat": ("NSP",), "bbb": ("NSP",), "avb": ("NSP",), "rhythm": ()}
SPLIT_VALUES = {("premature beat", "APC,VPC"): ("APC", "VPC")}
TARGETS = tuple(f"{c}:{v}" for c in COLUMNS for v in DEFINITE[c])
PRIMARY = ("rhythm:A.fib", "rhythm:A.flutter", "rhythm:Junctional rhythm", "rhythm:NSR", "rhythm:Other sinus rhythm",
           "rhythm:S.brady", "rhythm:S.tachy", "premature beat:VPC", "bbb:RBBB", "bbb:LBBB", "bbb:LAFB",
           "bbb:BFB(RBBB+LAFB)", "avb:1AVB", "avb:Variable AVB", "avb:IVB")
MIN_POSITIVES, MIN_NEGATIVES, MIN_POSITIVE_GROUPS = 20, 20, 10
N_BOOT, SEED = 1000, 0


def _values(column: str, value: str) -> tuple[str, ...] | None:
    """The definite values a cell names, () for a known negative, None when the column is unknown for this record."""
    value = str(value).strip()
    if (column, value) in SPLIT_VALUES:
        return SPLIT_VALUES[(column, value)]
    if value in DEFINITE[column]:
        return (value,)
    if value in NEGATIVE[column]:
        return ()
    return None   # rhythm's NSP/Possible pacing/Undetermined/Others, or a value outside the vocabulary


def encode(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(y, known): booleans (records, len(TARGETS)); y is meaningful only where known."""
    y = np.zeros((len(meta), len(TARGETS)), dtype=bool)
    known = np.zeros_like(y)
    index = {t: k for k, t in enumerate(TARGETS)}
    for column in COLUMNS:
        cols = [index[f"{column}:{v}"] for v in DEFINITE[column]]
        for r, cell in enumerate(meta[column].to_numpy(object)):
            names = _values(column, cell)
            if names is None:
                continue
            known[r, cols] = True
            for name in names:
                y[r, index[f"{column}:{name}"]] = True
    return y, known


def audit(meta: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Values outside the vocabulary (each masks its column for that record), per column; empty for the release."""
    out = {}
    for column in COLUMNS:
        vocab = set(DEFINITE[column]) | set(NEGATIVE[column]) | {v for (c, v) in SPLIT_VALUES if c == column}
        if column == "rhythm":
            vocab |= set(RHYTHM_UNKNOWN)
        odd = meta.loc[~meta[column].astype(str).str.strip().isin(vocab), column].astype(str).value_counts()
        if len(odd):
            out[column] = {k: int(v) for k, v in odd.items()}
    return out


def target_names(row_y: np.ndarray, row_known: np.ndarray) -> tuple[list[str], list[str]]:
    """(positive targets, unknown targets) of one record, for the instances file."""
    return ([TARGETS[k] for k in np.flatnonzero(row_y & row_known)],
            [TARGETS[k] for k in np.flatnonzero(~row_known)])


def matrices(positives, unknown) -> tuple[np.ndarray, np.ndarray]:
    """(y, known) back from the instances' `targets` and `unknown_targets` lists."""
    index = {t: k for k, t in enumerate(TARGETS)}
    y = np.zeros((len(positives), len(TARGETS)), dtype=bool)
    known = np.ones_like(y)
    for r, (pos, unk) in enumerate(zip(positives, unknown)):
        y[r, [index[t] for t in pos]] = True
        known[r, [index[t] for t in unk]] = False
    return y, known


def eligible(y: np.ndarray, known: np.ndarray, groups) -> list[str]:
    """Targets meeting the PRIMARY rule in one cohort's test split (y, known and groups of the test records)."""
    groups = np.asarray(groups)
    out = []
    for k, name in enumerate(TARGETS):
        pos = y[:, k] & known[:, k]
        neg = ~y[:, k] & known[:, k]
        if pos.sum() >= MIN_POSITIVES and neg.sum() >= MIN_NEGATIVES and len(set(groups[pos])) >= MIN_POSITIVE_GROUPS:
            out.append(name)
    return out


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank AUROC with ties counted half (the Mann-Whitney form); nan when a class is absent."""
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(np.asarray(scores, dtype=float)).rank(method="average").to_numpy()
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def per_target_auroc(y, known, scores, targets=PRIMARY) -> dict[str, float]:
    index = {t: k for k, t in enumerate(TARGETS)}
    out = {}
    for name in targets:
        k = index[name]
        m = known[:, k]
        out[name] = auroc(y[m, k], scores[m, k])
    return out


def thresholds_from_validation(y, known, scores, targets=PRIMARY, min_positives: int = 10) -> dict[str, float]:
    """The F1-best threshold per target on validation known labels; 0.5 with fewer than `min_positives` positives."""
    index = {t: k for k, t in enumerate(TARGETS)}
    out = {}
    for name in targets:
        k = index[name]
        m = known[:, k]
        lab, sc = y[m, k], scores[m, k]
        if lab.sum() < min_positives:
            out[name] = 0.5
            continue
        best, best_f1 = 0.5, -1.0
        for t in np.unique(sc):
            f1 = _f1(lab, sc >= t)
            if f1 > best_f1:
                best, best_f1 = float(t), f1
        out[name] = best
    return out


def _f1(lab: np.ndarray, pred: np.ndarray) -> float:
    tp = int((lab & pred).sum())
    denom = int(lab.sum()) + int(pred.sum())
    return 2 * tp / denom if denom else float("nan")


def per_target_f1(y, known, scores, thresholds: dict[str, float], targets=PRIMARY) -> dict[str, float]:
    index = {t: k for k, t in enumerate(TARGETS)}
    out = {}
    for name in targets:
        k = index[name]
        m = known[:, k]
        out[name] = _f1(y[m, k], scores[m, k] >= thresholds.get(name, 0.5))
    return out


def _json_float(v) -> float | None:
    """A finite float, or None for an undefined value (strict JSON has no NaN)."""
    v = float(v)
    return v if np.isfinite(v) else None


def _nanmean(values) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def metrics(y, known, scores, groups, thresholds: dict[str, float] | None = None, n_boot: int = N_BOOT,
            seed: int = SEED, targets=PRIMARY) -> dict:
    """macro_auroc (primary) and macro_f1 with 95% whole-group bootstrap intervals, plus per-target values."""
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups)
    thresholds = thresholds or {t: 0.5 for t in targets}

    def point(rows):
        a = per_target_auroc(y[rows], known[rows], scores[rows], targets)
        f = per_target_f1(y[rows], known[rows], scores[rows], thresholds, targets)
        return a, f

    a, f = point(np.arange(len(y)))
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    macro_a, macro_f, dropped = [], [], 0
    per_a = {t: [] for t in targets}
    for _ in range(n_boot):
        rows = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        ba, bf = point(rows)
        for t, v in ba.items():
            if np.isfinite(v):
                per_a[t].append(v)
        if all(np.isfinite(v) for v in ba.values()):
            macro_a.append(float(np.mean(list(ba.values()))))
            macro_f.append(_nanmean(list(bf.values())))
        else:
            dropped += 1

    def ci(values):
        return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))) if values else (None, None)

    lo_a, hi_a = ci(macro_a)
    lo_f, hi_f = ci(macro_f)
    return {
        "macro_auroc": {"value": _json_float(np.mean(list(a.values()))), "ci_low": lo_a, "ci_high": hi_a,
                        "n": int(len(y)), "n_groups": int(len(uniq))},
        "macro_f1": {"value": _json_float(_nanmean(list(f.values()))), "ci_low": lo_f, "ci_high": hi_f,
                     "n": int(len(y)), "n_groups": int(len(uniq))},
        "per_target": {t: {"auroc": _json_float(a[t]), "auroc_ci": list(ci(per_a[t])), "f1": _json_float(f[t]),
                           "positives": int((y[:, TARGETS.index(t)] & known[:, TARGETS.index(t)]).sum()),
                           "known": int(known[:, TARGETS.index(t)].sum())} for t in targets},
        "bootstrap": {"draws": n_boot, "seed": seed, "macro_dropped": dropped},
    }
