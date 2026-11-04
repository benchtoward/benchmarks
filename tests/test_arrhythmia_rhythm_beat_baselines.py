"""Baselines for vitaldb-arrhythmia-rhythm@0 and vitaldb-arrhythmia-beat@0: beat features at given positions,
segment features, one pass over the shared segments, prior and gradient boosting, leakage checks."""
import json

import numpy as np
import pandas as pd
import pytest

from benchtoward.baselines import arrhythmia_rhythm_beat as rbb

HZ = 500


def ecg(beats, seconds=5.0, noise=0.01, seed=0):
    """`beats`: (time_s, kind) with kind N (narrow), S (narrow, no P wave) or V (wide, tall, inverted T)."""
    t = np.arange(int(seconds * HZ)) / HZ
    x = np.random.default_rng(seed).normal(0, noise, len(t)) + 0.05 * np.sin(2 * np.pi * 0.3 * t)
    for b, kind in beats:
        if kind == "V":
            x += 1.6 * np.exp(-0.5 * ((t - b) / 0.035) ** 2) - 0.4 * np.exp(-0.5 * ((t - b - 0.25) / 0.06) ** 2)
        else:
            x += np.exp(-0.5 * ((t - b) / 0.01) ** 2) + 0.2 * np.exp(-0.5 * ((t - b - 0.28) / 0.05) ** 2)
            if kind == "N":
                x += 0.12 * np.exp(-0.5 * ((t - b + 0.16) / 0.03) ** 2)   # P wave
    return x.astype(np.float32)


def pattern(seed):
    """A 5-s segment of sinus beats at 0.8 s with one premature beat (S or V) or none; labels per beat."""
    rng = np.random.default_rng(seed)
    times, kinds, t = [], [], 0.3 + rng.uniform(0, 0.3)
    ectopic = rng.choice(["N", "S", "V"])
    where = rng.integers(2, 4)
    while t < 4.9:
        k = ectopic if len(times) == where else "N"
        times.append(t)
        kinds.append(k)
        t += 0.8 * (0.6 if len(times) == where else (1.35 if k == "V" else 1.0)) * rng.uniform(0.95, 1.05)
    return ecg(list(zip(times, kinds)), seed=seed), (np.array(times) * HZ).round().astype(int), np.array(kinds)


def test_beat_features_tell_wide_premature_beats_from_narrow_ones():
    x = ecg([(0.4, "N"), (1.2, "N"), (2.0, "N"), (2.5, "V"), (3.6, "N"), (4.0, "S"), (4.6, "N")])
    samples = (np.array([0.4, 1.2, 2.0, 2.5, 3.6, 4.0, 4.6]) * HZ).astype(int)
    f = pd.DataFrame(rbb.beat_features(x, samples), columns=rbb.BEAT_FEATURES)
    assert f.shape == (7, len(rbb.BEAT_FEATURES))
    assert np.isnan(f.loc[0, "pre_rr"]) and np.isnan(f.loc[6, "post_rr"]) and f.loc[1, "pre_rr"] == pytest.approx(0.8)
    assert f.loc[3, "pre_ratio"] < 0.85 and f.loc[5, "pre_ratio"] < 0.85 and f.loc[1, "pre_ratio"] > 0.9
    assert f.loc[3, "qrs_width_ms"] > 1.5 * f.loc[1, "qrs_width_ms"]
    assert f.loc[3, "template_corr"] < f.loc[1, "template_corr"] and f.loc[1, "template_corr"] > 0.9
    assert np.isfinite(f.drop(columns=["pre_rr", "post_rr", "pre_post_ratio", "pre_ratio", "post_ratio"])
                       .loc[1:5].to_numpy()).all()


def test_beat_features_are_nan_safe():
    samples = np.array([300, 700, 1100])
    blank = rbb.beat_features(np.full(5 * HZ, np.nan, dtype=np.float32), samples)
    f = pd.DataFrame(blank, columns=rbb.BEAT_FEATURES)
    assert f.loc[1, "pre_rr"] == pytest.approx(0.8)          # RR intervals come from the positions alone
    assert f["template_corr"].isna().all() and f["qrs_width_ms"].isna().all()
    flat = rbb.beat_features(np.zeros(5 * HZ, dtype=np.float32), samples)
    assert flat.shape == (3, len(rbb.BEAT_FEATURES))
    assert rbb.beat_features(np.zeros(5 * HZ, dtype=np.float32), np.array([], dtype=int)).shape == \
        (0, len(rbb.BEAT_FEATURES))
    edge = rbb.beat_features(pattern(1)[0], np.array([2, 2497]))   # windows mostly outside the segment
    assert edge.shape == (2, len(rbb.BEAT_FEATURES))


def test_rhythm_features_extend_the_rr_features_with_morphology():
    x, _, _ = pattern(3)
    f = rbb.rhythm_features(x)
    assert f.shape == (len(rbb.RHYTHM_FEATURES),) and rbb.RHYTHM_FEATURES[:15] == rbb.base.FEATURES
    assert np.isfinite(f[:15]).all() and np.isfinite(f[15:]).all()
    assert rbb.rhythm_features(np.full(5 * HZ, np.nan, dtype=np.float32)).shape == (len(rbb.RHYTHM_FEATURES),)


def _beat_set(n, seed):
    xs, rows = [], []
    for i in range(n):
        x, samples, kinds = pattern(seed * 10_000 + i)
        xs.append(rbb.beat_features(x, samples))
        rows.extend(kinds)
    return np.vstack(xs), np.array(rows)


def test_gradient_boosting_classifies_synthetic_beats():
    from sklearn.metrics import f1_score

    train_x, train_y = _beat_set(150, 1)
    test_x, test_y = _beat_set(60, 2)
    proba = rbb.fit_predict(train_x, train_y, test_x, ["N", "S", "V"])
    assert proba.shape == (len(test_y), 3) and np.allclose(proba.sum(axis=1), 1)
    pred = np.array(["N", "S", "V"])[proba.argmax(axis=1)]
    assert f1_score(test_y, pred, average="macro") > 0.9


# --- one pass ---------------------------------------------------------------------------------------------------


def _tables(n_cases=40, per_case=6, seed=0):
    """Shared segments, rhythm instances and beat instances like build_task writes, with synthetic ECG per case."""
    rng = np.random.default_rng(seed)
    segs, rhythm, beat, signals = [], [], [], {}
    for c in range(1, n_cases + 1):
        part = "train" if c <= 0.7 * n_cases else ("validation" if c <= 0.8 * n_cases else "test")
        pieces = []
        for k in range(per_case):
            x, samples, kinds = pattern(c * 100 + k)
            start = 5 * (k + 1)
            sid = f"arr-{c}-{start}"
            label = "VT" if "V" in kinds else ("SVTA" if "S" in kinds else "N")
            af = k % 3 == 1   # every third segment is called atrial fibrillation (its beats keep their labels)
            rhythms = f"AFIB/AFL|{label}" if af else label
            segs.append({"segment_id": sid, "caseid": c, "subjectid": 500 + c, "split": part, "start_s": start,
                         "duration_s": 5, "rhythm": label if k else None, "rhythm_reason": None if k else "mixed",
                         "rhythms": rhythms, "rhythm_from_neighbours": False, "n_beats": len(samples),
                         "bad_signal_quality": False, "beats_scored": True, "beat_samples": list(samples),
                         "beat_types": list(kinds)})
            if k:
                rhythm.append({"instance_id": sid, "caseid": c, "subjectid": 500 + c, "split": part,
                               "start_s": start, "duration_s": 5, "label": label})
            for s, kind in zip(samples, kinds):
                beat.append({"instance_id": f"{sid}-{s}", "segment_id": sid, "caseid": c, "subjectid": 500 + c,
                             "split": part, "start_s": start, "sample": int(s), "label": kind, "rhythms": rhythms,
                             "in_af_segment": af})
            pieces.append(x)
        signals[c] = np.concatenate([np.zeros(5 * HZ, dtype=np.float32), *pieces])   # cached span starts at 0 s
    return pd.DataFrame(segs), pd.DataFrame(rhythm), pd.DataFrame(beat), signals, rng


@pytest.fixture
def one_pass(monkeypatch):
    from benchtoward.tasks import vitaldb_arrhythmia_abnormal as binary

    seg, rhythm, beat, signals, _ = _tables()
    reads = []

    def cached(data_dir, caseid):
        reads.append(int(caseid))
        return signals[int(caseid)], 0

    monkeypatch.setattr(binary, "_cached", cached)
    return seg, rhythm, beat, reads


def test_one_pass_reads_each_case_once_and_aligns_both_feature_sets(one_pass, tmp_path):
    seg, rhythm, beat, reads = one_pass
    features = rbb.one_pass_features(seg, rhythm, beat, tmp_path)
    assert sorted(reads) == sorted(seg["caseid"].unique())   # one read per case serves both tasks
    assert features["rhythm"].shape == (len(rhythm), len(rbb.RHYTHM_FEATURES))
    assert features["beat"].shape == (len(beat), len(rbb.BEAT_FEATURES))
    first = beat.iloc[7]
    row = seg.set_index("segment_id").loc[first["segment_id"]]
    x = signals_from(seg, first["segment_id"], one_pass)
    expected = rbb.beat_features(x, np.asarray(row["beat_samples"]))[list(row["beat_samples"]).index(first["sample"])]
    assert np.allclose(features["beat"][7], expected, equal_nan=True)
    assert np.allclose(features["rhythm"][0], rbb.rhythm_features(signals_from(seg, rhythm.iloc[0]["instance_id"],
                                                                               one_pass)), equal_nan=True)


def signals_from(seg, segment_id, one_pass):
    from benchtoward.tasks import vitaldb_arrhythmia_rhythm_beat as rb

    return rb.segment_ecg(seg[seg["segment_id"] == segment_id], None)[0]


def test_run_baselines_records_prior_and_gradient_boosting_for_both_tasks(one_pass, tmp_session, tmp_path):
    from conftest import REGISTRY_DIR
    from benchtoward.db import ItemResult
    from benchtoward.registry import load_registry, resolve_task
    from benchtoward.tasks import vitaldb_arrhythmia_rhythm_beat as rb

    load_registry(tmp_session, REGISTRY_DIR)
    seg, rhythm, beat, _ = one_pass
    tasks = {slug: resolve_task(tmp_session, f"{slug}@0", include_hidden=True)
             for slug in (rb.RHYTHM_SLUG, rb.BEAT_SLUG)}
    features = rbb.one_pass_features(seg, rhythm, beat, tmp_path)
    runs = rbb.run_rhythm_beat_baselines(tmp_session, tasks, rhythm, beat, features, data_dir=tmp_path, n_boot=20)
    by_key = {(r.task_id, r.model_slug): r for r in runs}
    assert len(runs) == 4 and {m for _, m in by_key} == {"benchtoward/prior", "benchtoward/arrhythmia-hgb-rr-morph"}
    for slug, table in ((rb.RHYTHM_SLUG, rhythm), (rb.BEAT_SLUG, beat)):
        n_test = int((table["split"] == "test").sum())
        for model in ("benchtoward/prior", "benchtoward/arrhythmia-hgb-rr-morph"):
            run = by_key[(tasks[slug].id, model)]
            m = json.loads(run.metrics_json)
            assert {"macro_f1", "accuracy", "by_class", "confusion"} <= set(m) and m["n"] == n_test
            assert m["n_groups"] == int(table.loc[table["split"] == "test", "subjectid"].nunique())
            assert tmp_session.query(ItemResult).filter_by(run_id=run.id).count() == n_test
            assert (tmp_path / "artifacts" / run.id / "predictions.parquet").exists()
            config = json.loads(run.config_json)
            assert config["eval_split"] == "test" and config["one_pass"] == "segments.parquet"
    beat_hgb = json.loads(by_key[(tasks[rb.BEAT_SLUG].id, "benchtoward/arrhythmia-hgb-rr-morph")].metrics_json)
    assert {"veb_sensitivity", "veb_ppv", "veb_fpr", "sveb_sensitivity", "sveb_ppv", "sveb_fpr",
            "sveb_sensitivity_outside_af", "sveb_ppv_outside_af"} <= set(beat_hgb)
    assert beat_hgb["n_outside_af"] == int(((beat["split"] == "test") & ~beat["in_af_segment"]).sum())
    rhythm_hgb = json.loads(by_key[(tasks[rb.RHYTHM_SLUG].id, "benchtoward/arrhythmia-hgb-rr-morph")].metrics_json)
    assert rhythm_hgb["confusion"]["labels"] == rb.scored_classes(rb.RHYTHM_SLUG)
    assert "AVB" not in rhythm_hgb["by_class"]
    assert set(rhythm_hgb["bootstrap"]) == {"n", "by", "redraws"} and rhythm_hgb["bootstrap"]["n"] == 20
    assert beat_hgb["macro_f1"]["value"] > 0.8
    prior = json.loads(by_key[(tasks[rb.BEAT_SLUG].id, "benchtoward/prior")].metrics_json)
    assert prior["by_class"]["N"]["sensitivity"]["value"] == 1.0 and prior["by_class"]["V"]["ppv"]["value"] is None
    item = tmp_session.query(ItemResult).filter_by(run_id=by_key[(tasks[rb.BEAT_SLUG].id, "benchtoward/prior")].id).first()
    assert item.prediction == "N" and item.label in ("N", "S", "V") and item.correct == (item.label == "N")
    again = rbb.run_rhythm_beat_baselines(tmp_session, tasks, rhythm, beat, features, data_dir=tmp_path, n_boot=20)
    assert {r.id for r in again} == {r.id for r in runs}   # run once


def test_validation_report_compares_class_weights_without_test_labels(one_pass, tmp_path):
    seg, rhythm, beat, _ = one_pass
    features = rbb.one_pass_features(seg, rhythm, beat, tmp_path)
    for table in (rhythm, beat):
        table.loc[table["split"] == "test", "label"] = "not a class"   # the choice must never read test labels
    report = rbb.validation_report(rhythm, beat, features)
    assert set(report) == {"rhythm", "beat"}
    for name, entry in report.items():
        assert set(entry) == {"none", "balanced", "used"} and entry["used"] == rbb.CLASS_WEIGHT[name]
        assert 0 <= entry["none"] <= 1 and 0 <= entry["balanced"] <= 1


def test_leakage_checks_score_position_alone(one_pass, tmp_path):
    seg, rhythm, beat, _ = one_pass
    features = rbb.one_pass_features(seg, rhythm, beat, tmp_path)
    report = rbb.leakage_report(rhythm, beat, features, seg, n_boot=20)
    assert set(report) == {"rhythm_position_only", "rhythm_beat_positions_only", "beat_position_only",
                           "beat_positions_rr_only"}
    for value in report.values():
        assert 0 <= value["macro_f1"]["value"] <= 1 and value["features"]
    assert report["rhythm_beat_positions_only"]["features"] == rbb.POSITION_RHYTHM_FEATURES


def test_rhythm_features_from_the_given_beat_positions_alone():
    f = dict(zip(rbb.POSITION_RHYTHM_FEATURES, rbb.position_rhythm_features([250, 650, 1050, 1250, 1850])))
    assert f["n_beats"] == 5 and f["rr_mean"] == pytest.approx(0.8) and f["rr_min"] == pytest.approx(0.4)
    assert f["rr_max"] == pytest.approx(1.2) and f["hr_bpm"] == pytest.approx(75)   # RR 0.8, 0.8, 0.4, 1.2 s
    assert f["premature_share"] == pytest.approx(1 / 4)   # one RR interval below 85% of the mean
    one = rbb.position_rhythm_features([1000])
    assert one[0] == 1 and np.isnan(one[1:]).all()
    seg = pd.DataFrame({"segment_id": ["a", "b"], "beat_samples": [[250, 650, 1050], [100]]})
    x = rbb.position_rhythm_matrix(seg, pd.DataFrame({"instance_id": ["b", "a"]}))
    assert x.shape == (2, len(rbb.POSITION_RHYTHM_FEATURES)) and x[0, 0] == 1 and x[1, 0] == 3


def test_beat_score_adds_supraventricular_statistics_outside_atrial_fibrillation():
    from benchtoward.scoring import multiclass_metrics_ci

    test = pd.DataFrame({"label": ["S", "S", "N", "S", "N", "V", "S", "N"],
                         "subjectid": [1, 1, 1, 2, 2, 2, 3, 3],
                         "in_af_segment": [True, True, False, False, False, False, False, True]})
    predicted = np.array(["S", "N", "N", "S", "S", "V", "N", "S"])
    m = rbb.score("beat", test, predicted, ["N", "S", "V"], n_boot=20)
    outside = ~test["in_af_segment"].to_numpy()
    expected = multiclass_metrics_ci(test["label"][outside], predicted[outside], test["subjectid"][outside],
                                     classes=["N", "S", "V"], n_boot=20)
    assert m["sveb_sensitivity_outside_af"] == expected["by_class"]["S"]["sensitivity"]
    assert m["sveb_ppv_outside_af"] == expected["by_class"]["S"]["ppv"]
    assert m["sveb_sensitivity_outside_af"]["value"] == 0.5 and m["sveb_ppv_outside_af"]["value"] == 0.5
    assert m["n_outside_af"] == 5 and m["sveb_sensitivity"]["value"] == 0.5
