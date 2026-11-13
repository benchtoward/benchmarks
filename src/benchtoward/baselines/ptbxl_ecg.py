"""ptbxl-superdiagnostic@0 baselines: the train prevalence (prior) and per-class gradient boosting on beat features.

The features are those of the LYDUS beat-feature baseline (benchtoward.baselines.lydus_ecg.record_features, version
lydus-beat-v1): lead II RR features, each lead's median beat and amplitude percentiles of the band-passed leads, read
from PTB-XL's eight independent leads (I, II, V1-V6) in uV; age and sex are added. One
HistGradientBoostingClassifier per class (300 iterations, learning rate 0.1, random_state 0) on the train folds.
"""
import json

import numpy as np
import pandas as pd

from benchtoward.baselines import lydus_ecg
from benchtoward.tasks import ptbxl
from benchtoward.tasks.ptbxl_superdiagnostic import CLASSES

FEATURE_VERSION = f"{lydus_ecg.FEATURE_VERSION} on PTB-XL I, II, V1-V6 (uV)"
HGB = lydus_ecg.HGB


def _chunk(args) -> np.ndarray:
    path, rows = args
    signals = np.load(path, mmap_mode="r")
    return np.stack([lydus_ecg.record_features(ptbxl.stored_eight(np.asarray(signals[r]))) for r in rows])


def features_for(signals_path, rows: np.ndarray, workers: int = 4, chunk: int = 500) -> np.ndarray:
    """Beat features (lydus_ecg.FEATURE_NAMES order) of the rows `rows` of signals-500hz.npy."""
    from multiprocessing import get_context

    rows = np.asarray(rows, dtype=int)
    jobs = [(str(signals_path), rows[s:s + chunk]) for s in range(0, len(rows), chunk)]
    if workers <= 1:
        return np.concatenate([_chunk(j) for j in jobs])
    with get_context("spawn").Pool(workers) as pool:
        return np.concatenate(pool.map(_chunk, jobs))


def with_demographics(X: np.ndarray, inst: pd.DataFrame) -> np.ndarray:
    return lydus_ecg.with_demographics(X, inst)


def fit(X: np.ndarray, y: np.ndarray) -> "lydus_ecg.ClassModels":
    """One gradient-boosting model per class (every label is known in PTB-XL)."""
    return lydus_ecg.fit_classes(X, y, np.ones_like(y, dtype=bool), min_positives=1, params=HGB)


def run_rows(inst: pd.DataFrame, y: np.ndarray, scores: np.ndarray) -> list[dict]:
    """Per-record rows: the five scores, the labels, and whether every class at score >= 0.5 matches its label."""
    rows = []
    for i, iid in enumerate(inst["instance_id"]):
        outputs = scores[i] >= 0.5
        rows.append({"instance_id": iid,
                     "prediction": json.dumps({"scores": {c: float(scores[i, k]) for k, c in enumerate(CLASSES)}}),
                     "label": json.dumps({c: int(y[i, k]) for k, c in enumerate(CLASSES)}),
                     "correct": bool(np.array_equal(outputs, y[i])), "score": None})
    return rows
