import numpy as np
import pandas as pd
import pytest

from benchtoward.khdp import KhdpFile
from benchtoward.tasks.inspire_mortality import (assign_split, build_cohort, build_task, dataset_dir, fetch_inspire,
                                              load_spec)

DAY = 24 * 60
LAB_ITEMS = ('hb', 'platelet', 'aptt', 'wbc', 'ptinr', 'glucose', 'bun', 'albumin', 'ast', 'alt', 'creatinine',
             'sodium', 'potassium')
SCRIPT_INPUTS = ['age', 'sex', 'emop', 'bmi', 'andur', 'preop_hb', 'preop_platelet', 'preop_wbc', 'preop_aptt',
                 'preop_ptinr', 'preop_glucose', 'preop_bun', 'preop_albumin', 'preop_ast', 'preop_alt',
                 'preop_creatinine', 'preop_sodium', 'preop_potassium']


# ---- fetch ----

class FakeKhdp:
    """Stands in for KhdpClient: a release listing plus downloads that write the listed bytes."""

    def __init__(self, contents: dict[str, bytes]):
        self.contents = contents
        self.downloaded: list[str] = []

    def files(self, code, version):
        assert (code, version) == ("inspire", "1.3")
        return [KhdpFile(f"1.3/{name}", len(body), f"https://storage.test/{name}") for name, body in self.contents.items()]

    def download(self, file, dest):
        name = file.key.removeprefix("1.3/")
        self.downloaded.append(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.contents[name])
        return dest


RELEASE = {"operations.csv": b"op_id\n1\n", "labs.csv": b"subject_id\n1\n", "vitals.csv": b"x" * 100}


def test_fetch_downloads_only_operations_and_labs(tmp_path):
    client = FakeKhdp(RELEASE)
    root = fetch_inspire(tmp_path, client)
    assert root == dataset_dir(tmp_path) == tmp_path / "datasets" / "inspire" / "1.3"
    assert sorted(client.downloaded) == ["labs.csv", "operations.csv"]
    assert (root / "operations.csv").read_bytes() == RELEASE["operations.csv"]
    assert not (root / "vitals.csv").exists()


def test_fetch_skips_complete_files_and_redownloads_short_ones(tmp_path):
    root = dataset_dir(tmp_path)
    root.mkdir(parents=True)
    (root / "operations.csv").write_bytes(RELEASE["operations.csv"])
    (root / "labs.csv").write_bytes(b"sub")  # cut short
    client = FakeKhdp(RELEASE)
    fetch_inspire(tmp_path, client)
    assert client.downloaded == ["labs.csv"]
    assert (root / "labs.csv").read_bytes() == RELEASE["labs.csv"]


def test_fetch_fails_when_a_file_is_missing_from_the_release(tmp_path):
    with pytest.raises(FileNotFoundError, match="labs.csv"):
        fetch_inspire(tmp_path, FakeKhdp({"operations.csv": b"op_id\n1\n"}))


def test_cli_registers_fetch_build_and_baseline(monkeypatch):
    from typer.testing import CliRunner

    from benchtoward.cli import app

    runner = CliRunner()
    assert "inspire" in runner.invoke(app, ["fetch", "--help"]).output
    assert "inspire-mortality" in runner.invoke(app, ["build", "--help"]).output
    assert "inspire-mortality" in runner.invoke(app, ["baseline", "--help"]).output
    monkeypatch.delenv("KHDP_TOKEN", raising=False)
    result = runner.invoke(app, ["fetch", "inspire"])
    assert result.exit_code == 1 and "KHDP_TOKEN" in result.output


# ---- cohort, label, inputs and split ----

def op(op_id, subject_id, orin_time, **kw):
    row = {"op_id": op_id, "subject_id": subject_id, "case_id": np.nan, "age": 60, "sex": "F", "weight": 60.0,
           "height": 160.0, "asa": 2.0, "emop": 0, "orin_time": orin_time, "orout_time": orin_time + 120,
           "anstart_time": orin_time + 5, "anend_time": orin_time + 115, "inhosp_death_time": np.nan}
    return row | kw


def ops_frame(*rows):
    return pd.DataFrame(list(rows))


def labs_frame(*rows):
    if not rows:
        return pd.DataFrame({"subject_id": pd.Series(dtype="int64"), "chart_time": pd.Series(dtype="int64"),
                             "item_name": pd.Series(dtype="str"), "value": pd.Series(dtype="float64")})
    return pd.DataFrame(list(rows), columns=["subject_id", "chart_time", "item_name", "value"])


def test_spec_matches_the_script_constants():
    spec = load_spec()
    assert spec["inputs"]["script"] == SCRIPT_INPUTS and spec["inputs"]["extra"] == ["asa"]
    assert tuple(spec["inputs"]["lab_items"]) == LAB_ITEMS
    assert spec["inputs"]["lab_window_min"] == 6 * 30 * DAY and spec["label"]["window_min"] == 30 * DAY
    assert spec["split"]["seed"] == 1 and spec["split"]["train_fraction"] == 0.7
    assert len(spec["protocol"]["commit"]) == 40 and spec["protocol"]["commit"] in spec["protocol"]["script"]


def test_cohort_takes_each_patients_earliest_operation_not_the_smallest_op_id():
    ops = ops_frame(op(40, 1, 9000, age=70), op(41, 1, 5000, age=50), op(39, 1, 7000, age=80), op(42, 2, 100))
    df = build_cohort(ops, labs_frame())
    assert sorted(df["op_id"]) == [41, 42]  # ops 39 and 40 have smaller op_ids but are later operations
    assert df.set_index("op_id").loc[41, "age"] == 50


def test_orin_time_ties_within_a_patient_break_by_smallest_op_id():
    ops = ops_frame(op(52, 1, 300, age=20), op(51, 1, 300, age=30), op(53, 1, 300, age=40), op(50, 1, 900))
    df = build_cohort(ops, labs_frame())
    assert list(df["op_id"]) == [51] and df["age"].tolist() == [30]
    assert "smallest op_id" in load_spec()["cohort"]["operation"]


def test_label_is_in_hospital_death_within_30_days_after_or_exit():
    ops = ops_frame(op(1, 1, 0, inhosp_death_time=120 + 30 * DAY - 5),   # just inside
                    op(2, 2, 0, inhosp_death_time=120 + 30 * DAY),       # exactly 30 days: strict <, negative
                    op(3, 3, 0),                                         # no in-hospital death
                    op(4, 4, 0, inhosp_death_time=60))                   # died before leaving the OR
    df = build_cohort(ops, labs_frame()).set_index("op_id")
    assert df["label"].to_dict() == {1: 1, 2: 0, 3: 0, 4: 1}


def test_asa_6_and_missing_asa_are_excluded_after_choosing_the_operation():
    ops = ops_frame(op(1, 1, 0, asa=5.0), op(2, 2, 0, asa=6.0), op(3, 3, 0, asa=np.nan),
                    op(4, 4, 0, asa=6.0), op(5, 4, 50, asa=2.0))  # subject 4: chosen op 4 is ASA 6 -> dropped
    df = build_cohort(ops, labs_frame())
    assert list(df["op_id"]) == [1] and df["asa"].tolist() == [5.0]


def test_bmi_andur_sex_and_vitaldb_link():
    ops = ops_frame(op(1, 1, 0, weight=80.0, height=200.0, sex="M", anstart_time=10, anend_time=70, case_id=1234.0),
                    op(2, 2, 0, height=10.0),  # height <= 10 -> bmi missing
                    op(3, 3, 0, anend_time=np.nan))
    df = build_cohort(ops, labs_frame()).set_index("op_id")
    assert df.loc[1, "bmi"] == pytest.approx(20.0) and np.isnan(df.loc[2, "bmi"])
    assert df.loc[1, "andur"] == 60 and np.isnan(df.loc[3, "andur"])
    assert df["sex"].to_dict() == {1: 1.0, 2: 0.0, 3: 0.0}
    assert df["vitaldb_linked"].to_dict() == {1: True, 2: False, 3: False}


def test_preop_labs_take_the_latest_value_up_to_orin_time_within_6_months():
    orin = 300 * DAY
    ops = ops_frame(op(1, 1, orin), op(2, 2, orin))
    labs = labs_frame((1, orin - 10 * DAY, "hb", 11.0), (1, orin - 5, "hb", 12.0), (1, orin + 5, "hb", 99.0),
                      (1, orin, "sodium", 140.0),                    # at orin_time: included
                      (1, orin - 6 * 30 * DAY - 5, "albumin", 3.0),  # just beyond 6 months: excluded
                      (1, orin - 6 * 30 * DAY, "bun", 20.0),         # exactly 6 months: included
                      (2, orin - DAY, "wbc", 7.0),
                      (2, orin - DAY, "wbc", 8.0),                   # same time: the later file row wins
                      (1, orin - DAY, "lactate", 2.0))               # not one of the 13 items
    df = build_cohort(ops, labs).set_index("op_id")
    assert df.loc[1, "preop_hb"] == 12.0 and df.loc[1, "preop_sodium"] == 140.0 and df.loc[1, "preop_bun"] == 20.0
    assert np.isnan(df.loc[1, "preop_albumin"]) and np.isnan(df.loc[1, "preop_wbc"])
    assert df.loc[2, "preop_wbc"] == 8.0 and np.isnan(df.loc[2, "preop_hb"])
    assert "preop_lactate" not in df.columns
    assert set(SCRIPT_INPUTS + ["asa", "subject_id", "label", "vitaldb_linked"]) <= set(df.columns)


def test_cohort_order_is_orin_time_then_subject_id():
    ops = ops_frame(op(10, 3, 500), op(11, 1, 500), op(12, 2, 100), op(13, 4, 500))
    assert list(build_cohort(ops, labs_frame())["subject_id"]) == [2, 1, 3, 4]


def test_split_reproduces_the_script_shuffle_and_keeps_the_overlap_row_out_of_train():
    cohort = pd.DataFrame({"op_id": range(100, 120), "subject_id": range(20), "label": [0, 1] * 10})
    out = assign_split(cohort)
    script = cohort.sample(frac=1, random_state=1).reset_index(drop=True)
    ntrain = int(len(script) * 0.7)
    assert list(out["op_id"]) == list(script["op_id"])
    assert list(out.loc[out["split"] == "train", "op_id"]) == list(script.loc[:ntrain - 1, "op_id"])
    assert list(out.loc[out["split"] == "test", "op_id"]) == list(script.loc[ntrain:, "op_id"])
    overlap = script.loc[ntrain, "op_id"]  # in both of the script's sets (df.loc[:ntrain] is inclusive)
    assert out.set_index("op_id").loc[overlap, "split"] == "test"
    assert (out["split"] == "train").sum() == ntrain == 14 and (out["split"] == "test").sum() == 6


def test_build_task_writes_instances(tmp_path):
    root = dataset_dir(tmp_path)
    root.mkdir(parents=True)
    ops_frame(*[op(400 + k, 100 + k, 50 * k, case_id=(7.0 if k == 0 else np.nan),
                   inhosp_death_time=(60.0 if k % 4 == 0 else np.nan)) for k in range(10)]).to_csv(
        root / "operations.csv", index=False)
    labs_frame((100, 0, "hb", 13.0)).to_csv(root / "labs.csv", index=False)
    df = pd.read_parquet(build_task(tmp_path))
    assert len(df) == 10 and set(df["split"]) == {"train", "test"} and (df["split"] == "train").sum() == 7
    assert df["instance_id"].is_unique
    first = df.set_index("op_id").loc[400]
    assert first["instance_id"] == "inspire-100-400" and first["preop_hb"] == 13.0 and bool(first["vitaldb_linked"])
    assert set(SCRIPT_INPUTS + ["asa", "label", "subject_id"]) <= set(df.columns)
    assert df["label"].sum() == 3


# ---- metrics and baselines ----

def test_binary_metrics_can_skip_brier_for_scores_that_are_not_probabilities():
    from benchtoward.scoring import binary_metrics_ci

    y = np.array([0, 0, 1, 0, 1, 1, 0, 0])
    asa = np.array([1, 2, 3, 1, 4, 3, 2, 2], dtype=float)
    m = binary_metrics_ci(y, asa, np.arange(8), n_boot=50, metrics=("auroc", "auprc"))
    assert set(m) == {"auroc", "auprc", "n", "n_groups", "prevalence"}
    assert m["auroc"]["value"] > 0.9 and m["n_groups"] == 8


def _toy(n, rng, start=0):
    """Death is likelier with low albumin and high ASA; a fifth of the lab values are missing."""
    df = pd.DataFrame({c: rng.normal(0, 1, n) for c in SCRIPT_INPUTS})
    df["asa"] = rng.integers(1, 6, n).astype(float)
    risk = -2.5 * df["preop_albumin"] + 0.8 * df["asa"]
    df["label"] = (risk + rng.normal(0, 0.5, n) > np.quantile(risk, 0.8)).astype(int)
    for c in [c for c in SCRIPT_INPUTS if c.startswith("preop_") and c != "preop_albumin"]:
        df.loc[rng.random(n) < 0.2, c] = np.nan
    df["subject_id"] = np.arange(start, start + n)
    df["op_id"] = df["subject_id"] + 400_000
    df["instance_id"] = [f"inspire-{s}-{o}" for s, o in zip(df["subject_id"], df["op_id"])]
    return df


def test_asa_baseline_scores_are_the_asa_class():
    from benchtoward.baselines.inspire import fit_predict_asa

    rng = np.random.default_rng(0)
    train, test = _toy(50, rng), _toy(30, rng, start=50)
    np.testing.assert_array_equal(fit_predict_asa(train, test), test["asa"].to_numpy())


def test_logreg_and_xgb_learn_a_signal_despite_missing_values():
    from sklearn.metrics import roc_auc_score

    from benchtoward.baselines.inspire import fit_predict_logreg, fit_predict_xgb

    rng = np.random.default_rng(1)
    train, test = _toy(600, rng), _toy(300, rng, start=600)
    for fit in (fit_predict_logreg, fit_predict_xgb):
        scores = fit(train, test)
        assert scores.shape == (300,) and np.isfinite(scores).all() and ((0 <= scores) & (scores <= 1)).all()
        assert roc_auc_score(test["label"], scores) > 0.85


def test_baseline_settings_are_the_scripts():
    from benchtoward.baselines.inspire import LOGREG_PARAMS, XGB_PARAMS, model_inputs

    assert XGB_PARAMS == {"max_depth": 4, "n_estimators": 50, "subsample": 0.8, "colsample_bytree": 0.8,
                          "eval_metric": "logloss"}
    assert LOGREG_PARAMS == {"max_iter": 5000}
    assert model_inputs() == SCRIPT_INPUTS  # ASA is an input column, but the script's models do not use it


def _instances(rng):
    df = _toy(500, rng)
    df["split"] = np.where(np.arange(500) < 350, "train", "test")
    df["vitaldb_linked"] = False
    return df


def test_run_inspire_baselines_records_three_runs(tmp_session, tmp_path):
    import json

    from conftest import REGISTRY_DIR
    from benchtoward.baselines.inspire import run_inspire_baselines
    from benchtoward.db import ItemResult
    from benchtoward.registry import load_registry, resolve_task

    load_registry(tmp_session, REGISTRY_DIR)
    df = _instances(np.random.default_rng(2))
    runs = run_inspire_baselines(tmp_session, resolve_task(tmp_session, "inspire-mortality"), df, data_dir=tmp_path,
                                 n_boot=50)
    assert [r.model_slug for r in runs] == ["medbench/inspire-asa", "medbench/inspire-logreg", "medbench/inspire-xgb"]
    for r in runs:
        metrics = json.loads(r.metrics_json)
        assert set(metrics) == {"auroc", "auprc", "n", "n_groups", "prevalence"}
        assert metrics["auroc"]["value"] > 0.7 and metrics["n"] == 150 and metrics["n_groups"] == 150
        assert metrics["auroc"]["ci_low"] <= metrics["auroc"]["value"] <= metrics["auroc"]["ci_high"]
        assert tmp_session.query(ItemResult).filter_by(run_id=r.id).count() == 150
        assert (tmp_path / "artifacts" / r.id / "predictions.parquet").exists()
    asa_scores = {i.instance_id: i.score for i in tmp_session.query(ItemResult).filter_by(run_id=runs[0].id)}
    test = df[df["split"] == "test"]
    assert asa_scores == dict(zip(test["instance_id"], test["asa"]))
    config = json.loads(runs[2].config_json)
    assert config["params"]["max_depth"] == 4 and config["train_split"] == "train" and "xgboost" in config["versions"]


def test_inspire_baseline_runs_change_when_data_changes(tmp_session, tmp_path):
    from conftest import REGISTRY_DIR
    from benchtoward.baselines.inspire import run_inspire_baselines
    from benchtoward.registry import load_registry, resolve_task

    load_registry(tmp_session, REGISTRY_DIR)
    task = resolve_task(tmp_session, "inspire-mortality")
    df = _instances(np.random.default_rng(3))
    first = run_inspire_baselines(tmp_session, task, df, data_dir=tmp_path, n_boot=20)
    again = run_inspire_baselines(tmp_session, task, df, data_dir=tmp_path, n_boot=20)
    assert [r.id for r in first] == [r.id for r in again]  # run once
    changed = df.copy()
    changed.loc[changed.index[-1], "label"] = 1 - changed.loc[changed.index[-1], "label"]
    second = run_inspire_baselines(tmp_session, task, changed, data_dir=tmp_path, n_boot=20)
    assert {r.id for r in first}.isdisjoint({r.id for r in second})


# ---- registry ----

def test_registry_has_the_inspire_dataset_task_and_baselines(tmp_session):
    import yaml

    from conftest import REGISTRY_DIR
    from benchtoward.db import Dataset, Model
    from benchtoward.registry import load_registry, resolve_task

    load_registry(tmp_session, REGISTRY_DIR)
    dataset = tmp_session.get(Dataset, "inspire")
    assert dataset.modality == "structured" and dataset.version == "1.3" and dataset.khdp_code == "inspire"
    assert dataset.source_url == "https://khdp.net/database/data-search-detail/660/inspire/1.3"
    assert "10.13026/46m4-f655" in dataset.citation and dataset.license == "CC BY 4.0"
    task = resolve_task(tmp_session, "inspire-mortality")
    assert (task.version, task.dataset_slug, task.kind, task.primary_metric, task.preview) == (0, "inspire", "binary",
                                                                                               "auroc", True)
    for slug in ("medbench/inspire-asa", "medbench/inspire-logreg", "medbench/inspire-xgb"):
        model = tmp_session.get(Model, slug)
        assert (model.type, model.provider, model.developer) == ("baseline", "medbench", "medbench.kr")
    planned = yaml.safe_load((REGISTRY_DIR / "planned.yaml").read_text(encoding="utf-8"))
    assert not any("INSPIRE" in p["dataset"] for p in planned)
