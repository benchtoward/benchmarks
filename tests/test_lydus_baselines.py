"""Provided-label baseline training, thresholds and prediction artifacts."""
import json
import numpy as np
import pandas as pd
import pytest
from benchtoward.baselines import lydus_ecg as base
from benchtoward.tasks import lydus_labels as ll
from test_lydus_split import ecg


def test_prior_uses_known_train_prevalence_and_f1_constant_answer():
    y = np.zeros((4, 25), bool)
    known = np.ones_like(y)
    y[0, 0] = True
    known[2:, 0] = False
    known[:, 1] = False
    outputs, scores = base.prior(y, known, 3)
    assert scores.shape == (3, 25)
    assert scores[:, 0].tolist() == [0.5] * 3
    assert scores[:, 1].tolist() == [0.0] * 3
    assert outputs[:, 0].all() and not outputs[:, 1:].any()


def test_hgb_fits_only_known_entries_and_rare_targets_use_prevalence(monkeypatch):
    fitted = []
    class Model:
        def __init__(self, **kw): pass
        def fit(self, X, y):
            fitted.append((X.copy(), y.copy()))
            return self
        def predict_proba(self, X): return np.tile([0.2, 0.8], (len(X), 1))
    monkeypatch.setattr(base, "HistGradientBoostingClassifier", Model)
    X = np.arange(50)[:, None]
    y = np.zeros((50, 25), bool)
    known = np.ones_like(y)
    y[:20, 0] = True
    known[30:, 0] = False
    y[:19, 1] = True
    y[40:, 1] = True
    known[40:, 1] = False
    model = base.fit_classes(X, y, known)
    assert len(fitted) == 1 and fitted[0][0].ravel().tolist() == list(range(30))
    assert fitted[0][1].sum() == 20
    scores = model.predict_scores(X[:2])
    assert scores[:, 0].tolist() == [0.8] * 2
    assert scores[:, 1].tolist() == [19 / 40] * 2


def test_thresholds_delegate_to_provided_label_validation_masks():
    y = np.zeros((30, 25), bool)
    y[:10, 0] = True
    known = np.ones_like(y)
    known[:5, 0] = False
    scores = np.where(y, 0.2, 0.1)
    expected = ll.thresholds_from_validation(y, known, scores, targets=ll.TARGETS)
    assert base.f1_thresholds(y, known, scores).tolist() == [expected[t] for t in ll.TARGETS]
    assert expected[ll.TARGETS[0]] == 0.5


def test_run_rows_store_named_scores_and_known_labels():
    inst = pd.DataFrame({"instance_id": ["a", "b"]})
    y = np.zeros((2, 25), bool)
    y[0, 0] = True
    known = np.ones_like(y)
    known[1, 0] = False
    outputs = y.copy()
    outputs[1, 0] = True
    rows = base.run_rows(inst, y, known, outputs, y * 0.9 + 0.05)
    assert rows[0]["correct"] is True and rows[1]["correct"] is True
    pred = json.loads(rows[0]["prediction"])
    assert pred["outputs"] == [ll.TARGETS[0]] and set(pred["scores"]) == set(ll.TARGETS)
    assert json.loads(rows[1]["label"])[ll.TARGETS[0]] is None
    assert json.loads(rows[0]["label"])[ll.TARGETS[0]] == 1


def test_record_features_cover_twelve_leads_and_rr():
    f = base.record_features(ecg())
    assert f.shape == (len(base.FEATURE_NAMES),) and np.isfinite(f[: len(base.RR_NAMES)]).all()
    flat = base.record_features(np.zeros((8, 5000), np.int16))
    assert np.isnan(flat[len(base.RR_NAMES):]).all()


def test_feature_columns_keep_only_the_lead_set():
    twelve, two = base.feature_columns("12"), base.feature_columns("2")
    assert len(twelve) == len(base.FEATURE_NAMES)
    names = [base.FEATURE_NAMES[k] for k in two]
    assert all(n.split(":")[0] in ("rr", "I", "II") for n in names)
    assert any(n.startswith("II:") for n in names) and not any(n.startswith("V1:") for n in names)


def test_measurement_and_order_features_for_leakage_checks():
    inst = pd.DataFrame({"new_id": ["000010", "000002"], "age": [70.0, np.nan], "sex": ["M", None],
                         "vrate": [80.0, 60.0], "arate": [80.0, 60.0], "pri": [150.0, np.nan], "qrsd": [90.0, 120.0],
                         "qti": [380.0, 400.0], "qtc": [420.0, 410.0], "paxis": [np.nan, 40.0], "raxis": [10.0, -40.0],
                         "taxis": [20.0, 30.0]})
    assert base.leakage_features(inst, "measurements").shape == (2, 11)
    assert base.leakage_features(inst, "demographics").tolist()[0] == [70.0, 1.0]
    assert base.leakage_features(inst, "order").tolist() == [[10.0], [2.0]]




def test_baseline_report_uses_fixed_macro_auroc_for_leakage(monkeypatch):
    n = 60
    inst = pd.DataFrame({"instance_id": [f"i{k}" for k in range(n)], "row": range(n),
                         "targets": [[ll.TARGETS[k % 25]] for k in range(n)], "unknown_targets": [[]] * n,
                         "split": ["train"] * 20 + ["validation"] * 20 + ["test"] * 20,
                         "group": [f"g{k}" for k in range(n)], "age": [30] * n, "sex": ["F"] * n})
    monkeypatch.setattr(base, "leakage_features", lambda inst, kind: np.ones((len(inst), 1)))
    runs, report = base.run_lydus_baselines(None, None, inst, np.ones((n, len(base.FEATURE_NAMES))),
                                           data_dir=None, record=False, lead_sets=("12",),
                                           n_boot=0, report_boot=0, record_level_split=inst["split"].to_numpy())
    assert runs == [] and set(report["prior"]) == {"macro_auroc", "macro_f1"}
    assert all(set(m) >= {"macro_auroc", "macro_f1"} for m in report["leakage"].values())


def test_record_predictions_scores_and_records_macro_auroc(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from benchtoward.tasks import lydus_cli as cli, lydus_ecg12_dx as task
    import benchtoward.cli
    import benchtoward.evaluate
    import benchtoward.registry
    from types import SimpleNamespace
    inst = pd.DataFrame({"instance_id": ["a", "b"], "row": [0, 1], "split": ["test"] * 2,
                         "group": ["a", "b"], "targets": [list(ll.PRIMARY), []], "unknown_targets": [[], []]})
    monkeypatch.setattr(task, "load_instances", lambda _: inst)
    monkeypatch.setattr(cli, "get_settings", lambda: SimpleNamespace(data_dir=tmp_path))
    monkeypatch.setattr(benchtoward.cli, "open_session", lambda: nullcontext(SimpleNamespace(get=lambda *args: None)))
    monkeypatch.setattr(benchtoward.cli, "_summary", lambda _: "ok")
    monkeypatch.setattr(benchtoward.registry, "resolve_task", lambda *args, **kw: None)
    captured = []
    monkeypatch.setattr(benchtoward.evaluate, "record_baseline_run", lambda *args, **kw: captured.append(args))
    y, _ = task.label_matrices(inst)
    file = tmp_path / "preds.npz"
    np.savez(file, instance_id=["a", "b"], scores=np.where(y, 0.9, 0.1))
    cli.record_predictions(model="test", file=file, config="{}", n_boot=0)
    metrics = captured[0][4]
    assert metrics["macro_auroc"]["value"] == 1.0 and metrics["macro_f1"]["value"] == 1.0
    assert len(json.loads(captured[0][3][0]["prediction"])["scores"]) == 25


def test_reporting_keeps_fixed_primary_and_discloses_all_target_support():
    from benchtoward.tasks.lydus_reporting import metrics
    y = np.zeros((4, 25), bool)
    y[0, :] = True
    known = np.ones_like(y)
    known[-1, 0] = False
    m = metrics(y, known, np.where(y, 0.9, 0.1), ["g1", "g1", "g2", "g3"], n_boot=0)
    assert m["macro_auroc"]["value"] == 1.0
    assert set(m["per_target"]) == set(ll.TARGETS)
    assert m["per_target"][ll.TARGETS[0]]["negatives"] == 2
    assert m["per_target"][ll.TARGETS[0]]["positive_groups"] == 1
    assert m["per_target"]["rhythm:WPW"]["primary"] is False
