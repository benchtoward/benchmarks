"""Two medbench baselines for vitaldb-arrhythmia-abnormal, trained on the train split.

- medbench/prior: the train-split share of abnormal segments, for every segment (ignores the ECG).
- medbench/arrhythmia-hgb-rr: gradient boosting on RR-interval and amplitude features. R-peaks come from a simple
  QRS detector run on the 5-second segment itself; the annotated beat times are never used as input.
"""
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

HZ = 500
FEATURE_VERSION = "rr-v1"
FEATURES = ["n_peaks", "hr_bpm", "rr_mean", "rr_sd", "rr_cv", "rmssd", "rr_min", "rr_max", "rr_range_ratio",
            "irregularity", "pnn50", "peak_amp_cv", "signal_sd", "signal_ptp", "missing_fraction"]
THRESHOLD = 0.5


@lru_cache(maxsize=4)
def _bandpass(hz: int):
    from scipy.signal import butter

    return butter(2, [5, 15], btype="band", fs=hz)


def _fill(x: np.ndarray) -> np.ndarray | None:
    """Linear interpolation over missing samples; None if nothing is left."""
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    if ok.all():
        return x
    if not ok.any():
        return None
    idx = np.arange(len(x))
    return np.interp(idx, idx[ok], x[ok])


def _filtered(x: np.ndarray, hz: int) -> tuple[np.ndarray, np.ndarray]:
    """Band-passed signal and its smoothed squared slope (Pan-Tompkins style), along the last axis."""
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import filtfilt

    b, a = _bandpass(hz)
    y = filtfilt(b, a, x, axis=-1)
    energy = uniform_filter1d(np.gradient(y, axis=-1) ** 2, size=int(0.12 * hz) | 1, axis=-1, mode="constant")
    return y, energy


def _peaks(y: np.ndarray, energy: np.ndarray, hz: int) -> np.ndarray:
    from scipy.signal import find_peaks

    top = np.percentile(energy, 99)
    if top <= 0:
        return np.array([], dtype=int)
    found, _ = find_peaks(energy, height=0.3 * top, distance=int(0.2 * hz))
    half = int(0.075 * hz)
    refined = [max(0, p - half) + int(np.argmax(np.abs(y[max(0, p - half):p + half]))) for p in found]
    return np.unique(np.asarray(refined, dtype=int))


def detect_r_peaks(x: np.ndarray, hz: int = HZ) -> np.ndarray:
    """Sample indices of R-peaks in one ECG segment."""
    filled = _fill(x)
    if filled is None or np.ptp(filled) < 1e-6:
        return np.array([], dtype=int)
    y, energy = _filtered(filled, hz)
    return _peaks(y, energy, hz)


def _features(raw: np.ndarray, filled: np.ndarray | None, y, energy, hz: int) -> np.ndarray:
    f = dict.fromkeys(FEATURES, np.nan)
    f["missing_fraction"] = float(np.isnan(np.asarray(raw, dtype=float)).mean())
    if filled is None:
        return np.array([f[k] for k in FEATURES])
    f["signal_sd"], f["signal_ptp"] = float(filled.std()), float(np.ptp(filled))
    peaks = _peaks(y, energy, hz) if np.ptp(filled) >= 1e-6 else np.array([], dtype=int)
    f["n_peaks"] = len(peaks)
    if len(peaks) >= 2:
        rr = np.diff(peaks) / hz
        mean = rr.mean()
        f.update(rr_mean=mean, rr_sd=rr.std(), rr_cv=rr.std() / mean, rr_min=rr.min(), rr_max=rr.max(),
                 rr_range_ratio=(rr.max() - rr.min()) / mean, hr_bpm=60 / mean)
        amp = np.abs(y[peaks])
        f["peak_amp_cv"] = float(amp.std() / amp.mean()) if amp.mean() > 0 else np.nan
        if len(rr) >= 2:
            d = np.diff(rr)
            f.update(rmssd=float(np.sqrt(np.mean(d ** 2))), irregularity=float(np.mean(np.abs(d)) / mean),
                     pnn50=float(np.mean(np.abs(d) > 0.05)))
    return np.array([f[k] for k in FEATURES], dtype=float)


def rr_features(x: np.ndarray, hz: int = HZ) -> np.ndarray:
    """Feature vector (order: FEATURES) for one segment; undefined features are NaN."""
    filled = _fill(x)
    y = energy = None
    if filled is not None:
        y, energy = _filtered(filled, hz)
    return _features(x, filled, y, energy, hz)


def feature_matrix(segments: np.ndarray, hz: int = HZ) -> np.ndarray:
    """Features for a batch of equal-length segments (rows), filtering the batch at once."""
    segments = np.asarray(segments, dtype=float)
    if len(segments) == 0:
        return np.empty((0, len(FEATURES)))
    filled = [_fill(row) for row in segments]
    stacked = np.stack([row if row is not None else np.zeros(segments.shape[1]) for row in filled])
    y, energy = _filtered(stacked, hz)
    return np.vstack([_features(segments[i], filled[i], y[i], energy[i], hz) for i in range(len(segments))])


def features_for(instances: pd.DataFrame, data_dir, hz: int = HZ) -> np.ndarray:
    """Feature matrix for task instances, read case by case from the ECG cache."""
    from benchtoward.tasks.vitaldb_arrhythmia_abnormal import materialise

    out = np.full((len(instances), len(FEATURES)), np.nan)
    positions = np.arange(len(instances))
    for _, idx in pd.Series(positions).groupby(instances["caseid"].to_numpy()):
        rows = idx.to_numpy()
        out[rows] = feature_matrix(materialise(instances.iloc[rows], data_dir), hz)
    return out


def fit_predict_hgb(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    model = HistGradientBoostingClassifier(random_state=0).fit(train_x, train_y)
    return model.predict_proba(test_x)[:, 1]


def score_test(test: pd.DataFrame, scores: np.ndarray, n_boot: int = 1000) -> dict:
    """Task metrics: AUROC (primary), AUPRC, Brier, thresholded metrics, and the flagged share per rhythm."""
    from benchtoward.scoring import binary_metrics_ci, threshold_metrics_ci

    y, groups = test["label"].to_numpy(dtype=int), test["subjectid"].to_numpy()
    metrics = binary_metrics_ci(y, scores, groups, n_boot=n_boot)
    metrics.update(threshold_metrics_ci(y, scores, groups, threshold=THRESHOLD, n_boot=n_boot))
    flagged = scores >= THRESHOLD
    by_category = {}
    for rhythm in sorted(test["rhythm"].unique(), key=lambda r: (r != "N", r)):
        mask = (test["rhythm"] == rhythm).to_numpy()
        name = "N: false-positive rate" if rhythm == "N" else f"{rhythm}: sensitivity"
        by_category[name] = {"value": float(flagged[mask].mean()), "n": int(mask.sum())}
    metrics["by_category"] = by_category
    return metrics


def run_arrhythmia_baselines(session, task, instances: pd.DataFrame, features: np.ndarray, *, data_dir,
                             n_boot: int = 1000) -> list:
    """Train each baseline on the train split, score the test split, and record one run per baseline."""
    from benchtoward.baselines.ioh import data_hash
    from benchtoward.db import Model
    from benchtoward.evaluate import record_baseline_run

    train = (instances["split"] == "train").to_numpy()
    test = (instances["split"] == "test").to_numpy()
    fingerprint = data_hash(instances)
    y_train = instances.loc[train, "label"].to_numpy(dtype=int)
    predictions = {
        "medbench/prior": (np.full(test.sum(), y_train.mean()), {"prior": "train share of abnormal segments"}),
        "medbench/arrhythmia-hgb-rr": (fit_predict_hgb(features[train], y_train, features[test]),
                                       {"features": FEATURE_VERSION, "model": "HistGradientBoostingClassifier",
                                        "random_state": 0}),
    }
    test_rows = instances[test]
    runs = []
    for slug, (scores, extra) in predictions.items():
        metrics = score_test(test_rows, scores, n_boot=n_boot)
        rows = [{"instance_id": iid, "score": float(s), "label": int(y)}
                for iid, s, y in zip(test_rows["instance_id"], scores, test_rows["label"])]
        config = {"data_hash": fingerprint, "train_split": "train", "eval_split": "test", **extra}
        runs.append(record_baseline_run(session, session.get(Model, slug), task, rows, metrics, config,
                                        data_dir=data_dir))
    return runs
