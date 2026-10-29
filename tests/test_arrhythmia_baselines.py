"""Baselines for vitaldb-arrhythmia-abnormal@0: R-peak detector, RR features, prior and gradient boosting."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from medbench.baselines import arrhythmia as base

HZ = 500


def ecg(beat_times, seconds=5.0, noise=0.02, seed=0, width=0.012):
    t = np.arange(int(seconds * HZ)) / HZ
    rng = np.random.default_rng(seed)
    x = rng.normal(0, noise, len(t)) + 0.1 * np.sin(2 * np.pi * 0.3 * t)   # baseline wander
    for b in beat_times:
        x += np.exp(-0.5 * ((t - b) / width) ** 2)
    return x.astype(np.float32)


def regular(seed):
    rr = np.random.default_rng(seed).uniform(0.7, 1.0)
    return ecg(np.arange(0.3, 5.0, rr), seed=seed)


def irregular(seed):
    rng = np.random.default_rng(seed)
    times = np.cumsum(rng.uniform(0.35, 1.1, 12)) + 0.2
    return ecg(times[times < 4.9], seed=seed)


def test_r_peaks_found_at_the_beats():
    beats = np.array([0.4, 1.2, 1.9, 2.8, 3.5, 4.4])
    peaks = base.detect_r_peaks(ecg(beats))
    assert len(peaks) == len(beats)
    assert np.max(np.abs(peaks / HZ - beats)) < 0.03


def test_features_named_and_nan_safe():
    f = base.rr_features(regular(1))
    assert f.shape == (len(base.FEATURES),) and np.isfinite(f).all()
    names = dict(zip(base.FEATURES, f))
    assert 60 <= names["hr_bpm"] <= 90 and names["rr_sd"] < 0.02
    for x in (np.full(5 * HZ, np.nan, dtype=np.float32), np.zeros(5 * HZ, dtype=np.float32)):
        g = base.rr_features(x)
        assert g.shape == (len(base.FEATURES),)   # undefined features are NaN, never an exception
    one_gap = regular(2)
    one_gap[1000:1100] = np.nan
    assert np.isfinite(base.rr_features(one_gap)[base.FEATURES.index("rr_mean")])


def _toy(n, seed):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n)
    x = np.stack([irregular(seed * 1000 + i) if y else regular(seed * 1000 + i) for i, y in enumerate(labels)])
    return x, labels


def test_gradient_boosting_separates_regular_from_irregular():
    x_train, y_train = _toy(200, 1)
    x_test, y_test = _toy(100, 2)
    scores = base.fit_predict_hgb(base.feature_matrix(x_train), y_train, base.feature_matrix(x_test))
    assert scores.shape == (100,) and roc_auc_score(y_test, scores) > 0.9


def _instances(n=300, seed=3):
    x, labels = _toy(n, seed)
    df = pd.DataFrame({"instance_id": [f"arr-{i // 3}-{5 * (i % 3)}" for i in range(n)], "caseid": np.arange(n) // 3,
                       "subjectid": np.arange(n) // 6, "label": labels,
                       "rhythm": np.where(labels == 1, "AFIB/AFL", "N")})
    n_subjects = n // 6
    df["split"] = np.where(df["subjectid"] < 0.7 * n_subjects, "train",
                           np.where(df["subjectid"] < 0.8 * n_subjects, "validation", "test"))
    return df, base.feature_matrix(x)


def test_run_baselines_records_prior_and_gradient_boosting(tmp_session, tmp_path):
    from conftest import REGISTRY_DIR
    from medbench.db import ItemResult
    from medbench.registry import load_registry, resolve_task

    load_registry(tmp_session, REGISTRY_DIR)
    df, features = _instances()
    runs = base.run_arrhythmia_baselines(tmp_session, resolve_task(tmp_session, "vitaldb-arrhythmia-abnormal"), df,
                                         features, data_dir=tmp_path, n_boot=50)
    by_slug = {r.model_slug: r for r in runs}
    assert sorted(by_slug) == ["medbench/arrhythmia-hgb-rr", "medbench/prior"]
    n_test = int((df["split"] == "test").sum())
    for run in runs:
        m = json.loads(run.metrics_json)
        assert {"auroc", "auprc", "brier", "accuracy", "sensitivity", "specificity", "f1"} <= set(m)
        assert m["n"] == n_test and m["n_groups"] == 10 and "AFIB/AFL: sensitivity" in m["by_category"]
        assert "N: false-positive rate" in m["by_category"]
        assert tmp_session.query(ItemResult).filter_by(run_id=run.id).count() == n_test
        assert (tmp_path / "artifacts" / run.id / "predictions.parquet").exists()
    prior = json.loads(by_slug["medbench/prior"].metrics_json)
    hgb = json.loads(by_slug["medbench/arrhythmia-hgb-rr"].metrics_json)
    assert prior["auroc"]["value"] == 0.5 and hgb["auroc"]["value"] > 0.9


def test_baseline_runs_change_when_data_changes(tmp_session, tmp_path):
    from conftest import REGISTRY_DIR
    from medbench.registry import load_registry, resolve_task

    load_registry(tmp_session, REGISTRY_DIR)
    task = resolve_task(tmp_session, "vitaldb-arrhythmia-abnormal")
    df, features = _instances(120, seed=4)
    first = base.run_arrhythmia_baselines(tmp_session, task, df, features, data_dir=tmp_path, n_boot=10)
    assert {r.id for r in first} == {r.id for r in base.run_arrhythmia_baselines(
        tmp_session, task, df, features, data_dir=tmp_path, n_boot=10)}   # run once
    changed = df.copy()
    changed.loc[changed.index[-1], "label"] = 1 - changed.loc[changed.index[-1], "label"]
    second = base.run_arrhythmia_baselines(tmp_session, task, changed, features, data_dir=tmp_path, n_boot=10)
    assert {r.id for r in first}.isdisjoint({r.id for r in second})
