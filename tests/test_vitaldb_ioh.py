import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from benchtoward.baselines.ioh import features, fit_predict_hgb, fit_predict_logreg_lastmap
from benchtoward.tasks.vitaldb_ioh import build_task, make_windows, split_subjects

N = 150  # samples per 300 s window at 2 s


def series(input_values, horizon_values):
    return np.concatenate([input_values, horizon_values]).astype(float)


def test_30_consecutive_low_in_horizon_is_positive():
    horizon = np.r_[np.full(60, 80.0), np.full(30, 60.0), np.full(60, 80.0)]
    [(start, x, label)] = make_windows(series(np.full(N, 80.0), horizon))
    assert start == 0 and len(x) == N and label == 1


def test_29_consecutive_low_is_negative():
    horizon = np.r_[np.full(29, 60.0), np.full(121, 80.0)]
    [(_, _, label)] = make_windows(series(np.full(N, 80.0), horizon))
    assert label == 0


def test_artifact_zeros_are_not_hypotension():
    windows = make_windows(series(np.full(N, 80.0), np.r_[np.zeros(30), np.full(120, 80.0)]))
    assert all(label == 0 for _, _, label in windows)
    assert windows == []  # 30 of 150 horizon samples missing (>10%) -> dropped


def test_missing_value_breaks_a_low_run():
    horizon = np.r_[np.full(20, 60.0), np.zeros(10), np.full(20, 60.0), np.full(100, 80.0)]
    [(_, _, label)] = make_windows(series(np.full(N, 80.0), horizon))
    assert label == 0


def test_window_with_hypotensive_input_is_dropped():
    inp = np.r_[np.full(40, 55.0), np.full(110, 80.0)]
    assert make_windows(series(inp, np.full(N, 80.0))) == []


def test_trailing_incomplete_window_dropped_and_stride():
    windows = make_windows(np.full(N * 4 + 10, 80.0))
    assert [start for start, _, _ in windows] == [0, 150, 300]


def test_split_is_deterministic_and_subject_disjoint():
    ids = list(range(1000))
    a = split_subjects(ids)
    assert a == split_subjects(reversed(ids))
    assert set(a) == set(ids)
    share = {s: sum(v == s for v in a.values()) / 1000 for s in ("train", "validation", "test")}
    assert abs(share["train"] - 0.7) <= 0.03 and abs(share["validation"] - 0.1) <= 0.03
    assert abs(share["test"] - 0.2) <= 0.03


def test_build_task_from_parts(tmp_path):
    root = tmp_path / "datasets" / "vitaldb" / "ioh-v0"
    (root / "tracks").mkdir(parents=True)
    low = series(np.full(N, 80.0), np.r_[np.full(30, 60.0), np.full(120, 80.0)])
    pd.DataFrame({"caseid": [1, 2], "map": [low.tolist(), np.full(2 * N, 80.0).tolist()]}).to_parquet(
        root / "tracks" / "part-00000.parquet")
    pd.DataFrame({"caseid": [1, 2], "subjectid": [10, 20], "ane_type": ["General", "General"]}).to_parquet(
        root / "clinical.parquet")
    df = pd.read_parquet(build_task(tmp_path))
    assert list(df["instance_id"]) == ["ioh-1-0", "ioh-2-0"]
    assert list(df["label"]) == [1, 0] and list(df["subjectid"]) == [10, 20]
    assert len(df["x"].iloc[0]) == N and set(df["split"]) <= {"train", "validation", "test"}


def test_features_are_nan_safe():
    x = np.full(N, 80.0)
    x[[3, 50, 149]] = np.nan
    f = features(x)
    assert f.shape == (8,) and np.isfinite(f).all()


def _toy(n, rng):
    labels = rng.integers(0, 2, n)
    xs = [np.full(N, 70.0 if y else 90.0) + rng.normal(0, 3, N) for y in labels]
    return pd.DataFrame({"x": xs, "label": labels})


def test_baselines_learn_a_separable_signal():
    rng = np.random.default_rng(0)
    train, test = _toy(200, rng), _toy(100, rng)
    for fit in (fit_predict_logreg_lastmap, fit_predict_hgb):
        scores = fit(train, test)
        assert scores.shape == (100,) and roc_auc_score(test["label"], scores) > 0.9


def test_run_ioh_baselines_records_two_runs(tmp_session, tmp_path):
    import json

    from conftest import REGISTRY_DIR
    from benchtoward.baselines.ioh import run_ioh_baselines
    from benchtoward.db import ItemResult
    from benchtoward.registry import load_registry, resolve_task

    load_registry(tmp_session, REGISTRY_DIR)
    rng = np.random.default_rng(1)
    df = _toy(400, rng)
    df["subjectid"] = np.arange(400) // 2
    df["instance_id"] = [f"ioh-{i}-0" for i in range(400)]
    df["split"] = np.where(df["subjectid"] < 140, "train", np.where(df["subjectid"] < 160, "validation", "test"))
    runs = run_ioh_baselines(tmp_session, resolve_task(tmp_session, "vitaldb-ioh"), df, data_dir=tmp_path, n_boot=50)
    assert sorted(r.model_slug for r in runs) == ["benchtoward/ioh-hgb-summary", "benchtoward/ioh-logreg-lastmap"]
    for r in runs:
        metrics = json.loads(r.metrics_json)
        assert metrics["auroc"]["value"] > 0.9 and metrics["n"] == 80 and metrics["n_groups"] == 40
        assert tmp_session.query(ItemResult).filter_by(run_id=r.id).count() == 80
        assert (tmp_path / "artifacts" / r.id / "predictions.parquet").exists()


def test_series_completeness_check():
    from benchtoward.tasks.vitaldb_ioh import series_is_complete

    assert series_is_complete([80.0] * 600, caseend_s=1200, interval_s=2)
    assert not series_is_complete([80.0] * 300, caseend_s=1200, interval_s=2)
    assert not series_is_complete([], caseend_s=1200, interval_s=2)


def test_build_task_refuses_missing_cohort_cases(tmp_path):
    import json

    import pytest

    root = tmp_path / "datasets" / "vitaldb" / "ioh-v0"
    (root / "tracks").mkdir(parents=True)
    pd.DataFrame({"caseid": [1], "map": [np.full(2 * N, 80.0).tolist()]}).to_parquet(root / "tracks" / "part-00000.parquet")
    pd.DataFrame({"caseid": [1, 2], "subjectid": [10, 20], "ane_type": ["General", "General"]}).to_parquet(
        root / "clinical.parquet")
    (root / "cohort.json").write_text(json.dumps([1, 2]))
    with pytest.raises(RuntimeError, match="1 cohort case"):
        build_task(tmp_path)


def test_baseline_runs_change_when_data_changes(tmp_session, tmp_path):
    from conftest import REGISTRY_DIR
    from benchtoward.baselines.ioh import run_ioh_baselines
    from benchtoward.registry import load_registry, resolve_task

    load_registry(tmp_session, REGISTRY_DIR)
    task = resolve_task(tmp_session, "vitaldb-ioh")
    df = _toy(400, np.random.default_rng(2))
    df["subjectid"] = np.arange(400) // 2
    df["instance_id"] = [f"ioh-{i}-0" for i in range(400)]
    df["split"] = np.where(df["subjectid"] < 160, "train", "test")
    first = run_ioh_baselines(tmp_session, task, df, data_dir=tmp_path, n_boot=20)
    changed = df.copy()
    changed.loc[changed.index[-1], "label"] = 1 - changed.loc[changed.index[-1], "label"]
    second = run_ioh_baselines(tmp_session, task, changed, data_dir=tmp_path, n_boot=20)
    assert {r.id for r in first}.isdisjoint({r.id for r in second})
