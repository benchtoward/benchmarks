"""inspire-aki@0: cohort, inputs, labels (the paper's and the earlier CRRT rule), GS-AKI, split, outer folds and
baselines on miniature INSPIRE tables."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from benchtoward.khdp import KhdpFile
from benchtoward.tasks.inspire_aki import (
    DAY,
    FILES,
    aki_labels,
    assign_fold,
    assign_split,
    audit_actions,
    card_event_counts,
    crrt_flags,
    dataset_dir,
    fetch_inspire_aki,
    gs_aki,
    input_columns,
    kidney_failure_before,
    latest_before,
    load_spec,
    outer_folds_path,
    postop_max,
    preop_max,
    published_split_path,
    read_audit_table,
    read_outer_folds,
    read_split,
    same_day_card_events,
    select_operations,
)

# ---- fetch ----

class FakeKhdp:
    def __init__(self, contents):
        self.contents, self.downloaded = contents, []

    def files(self, code, version):
        assert (code, version) == ("inspire", "1.3")
        return [KhdpFile(f"1.3/{n}", len(b), f"https://storage.test/{n}") for n, b in self.contents.items()]

    def download(self, file, dest):
        name = file.key.removeprefix("1.3/")
        self.downloaded.append(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.contents[name])
        return dest


def test_fetch_downloads_the_five_tables_and_skips_complete_ones(tmp_path):
    release = {name: name.encode() * 3 for name in FILES} | {"medications.csv": b"x" * 50}
    root = dataset_dir(tmp_path)
    root.mkdir(parents=True)
    (root / "labs.csv").write_bytes(release["labs.csv"])
    client = FakeKhdp(release)
    fetch_inspire_aki(tmp_path, client)
    assert sorted(client.downloaded) == sorted(set(FILES) - {"labs.csv"})
    assert not (root / "medications.csv").exists()


def test_fetch_fails_when_a_table_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="ward_vitals.csv"):
        fetch_inspire_aki(tmp_path, FakeKhdp({n: b"1" for n in FILES if n != "ward_vitals.csv"}))


# ---- cohort ----

def _op(op_id, subject_id=1, **kw):
    row = dict(op_id=op_id, subject_id=subject_id, age=60, sex="M", height=170.0, weight=70.0, asa=2, emop=0,
               department="GS", antype="General", icd10_pcs="0DTJ4", opdate=0, orin_time=0, orout_time=200,
               opstart_time=30, opend_time=150, cpbon_time=np.nan, cpboff_time=np.nan)
    row.update(kw)
    return row


AUDIT = pd.DataFrame({"pcs_prefix": ["0DTJ4", "0DTJ4", "0BTC4", "0JH60"], "is_cts": [False, True, True, False],
                      "cpb_present": [False, False, True, False], "action": ["retain", "retain", "exclude", "exclude"]})
NO_DIAG = pd.DataFrame({"subject_id": [], "chart_time": [], "icd10_cm": []})


def test_select_operations_applies_the_authors_filters_in_order():
    ops = pd.DataFrame([
        _op(1),
        _op(2, asa=6), _op(3, asa=np.nan), _op(4, age=17), _op(5, opend_time=np.nan), _op(6, opend_time=30),
        _op(7, height=0.0), _op(8, weight=np.nan), _op(9, antype="Regional"), _op(10, department="PED"),
        _op(11, icd10_pcs="02703"), _op(12, icd10_pcs="10D00"), _op(13, icd10_pcs="0TY00"),
        _op(14, icd10_pcs="0BTC4", department="CTS", cpbon_time=100.0),   # audit: exclude
        _op(15, icd10_pcs="0DTJ4", department="CTS"),                    # audit: retain
        _op(16, antype="MAC"),
    ])
    df, steps = select_operations(ops, NO_DIAG, AUDIT)
    assert df["op_id"].tolist() == [1, 15, 16]
    assert [s["step"] for s in steps][-1] == "procedure_audit"
    assert {s["step"]: s["operations"] for s in steps}["raw_operations"] == 16
    row = df.iloc[0]
    assert row["sex"] == 1.0 and row["op_len"] == 120 and row["booking_case_length"] == 200
    assert row["BSA"] == pytest.approx(np.sqrt(170 * 70 / 3600)) and row["BMI"] == pytest.approx(70 / 1.7 ** 2)
    assert df["antype"].tolist() == [0.0, 0.0, 1.0]


def test_audit_lookup_refuses_an_unknown_combination():
    ops = pd.DataFrame([_op(1, icd10_pcs="0DTJ4", cpbon_time=5.0)])  # CPB with this code is not in the table
    with pytest.raises(ValueError, match="no decision"):
        audit_actions(ops, AUDIT)


def test_published_audit_table_is_a_function_of_its_keys():
    table = read_audit_table()
    assert not table.duplicated(["pcs_prefix", "is_cts", "cpb_present"]).any()
    assert set(table["action"]) == {"retain", "exclude"}
    assert (table["pcs_prefix"].str.len() == 5).all()
    assert table.loc[table["pcs_prefix"].str.startswith("02"), "action"].empty  # removed earlier by the prefix rule


def test_card_events_count_rows_with_I_codes_before_the_start():
    ops = pd.DataFrame([_op(1, opstart_time=1000), _op(2, subject_id=2, opstart_time=1000)])
    diag = pd.DataFrame({"subject_id": [1, 1, 1, 1, 2], "chart_time": [10, 20, 1000, 20, 5],
                         "icd10_cm": ["I10", "I50", "I21", "E11", "J18"]})
    assert card_event_counts(ops, diag).tolist() == [2, 0]


def test_same_day_diagnoses_pass_the_before_the_start_rule_and_are_counted_apart():
    """Diagnosis times are whole days (midnight), so a row stamped on the day of the operation counts as charted before
    the start (the authors' rule, kept), though it may have been made after it. The report counts those rows apart."""
    ops = pd.DataFrame([_op(1, opdate=10 * DAY, opstart_time=10 * DAY + 480),
                        _op(2, subject_id=2, opdate=10 * DAY, opstart_time=10 * DAY + 480)])
    diag = pd.DataFrame({"subject_id": [1, 1, 1, 1, 1, 2],
                         "chart_time": [9 * DAY, 10 * DAY, 10 * DAY, 11 * DAY, 10 * DAY, 3],
                         "icd10_cm": ["I10", "I50", "E11", "I21", "I48", "I10"]})
    assert card_event_counts(ops, diag).tolist() == [3, 1]
    assert same_day_card_events(ops, diag).tolist() == [2, 0]


# ---- inputs ----

def test_latest_before_takes_the_latest_value_within_90_days_up_to_the_start():
    ops = pd.DataFrame([_op(1, opstart_time=100 * DAY), _op(2, subject_id=2, opstart_time=100 * DAY),
                        _op(3, subject_id=3, opstart_time=100 * DAY)])
    labs = pd.DataFrame({
        "subject_id": [1, 1, 1, 2, 3, 3],
        "chart_time": [20 * DAY, 99 * DAY, 100 * DAY + 1, 9 * DAY, 100 * DAY, 100 * DAY],
        "item_name": ["creatinine"] * 6,
        "value": [0.8, 1.1, 5.0, 0.9, 1.0, 1.3],
    })
    out = latest_before(ops, labs, ["creatinine", "hb"], "preop_", 90 * DAY)
    assert out["preop_creatinine"].iloc[0] == 1.1          # after the start never counts
    assert np.isnan(out["preop_creatinine"].iloc[1])       # 91 days before: outside the window
    assert out["preop_creatinine"].iloc[2] == 1.3          # at the start counts; ties take the later row
    assert out["preop_hb"].isna().all()


# ---- labels ----

def _label_frame(**cols):
    base = dict(preop_creatinine=1.0, postop_creatinine_2_days=np.nan, postop_creatinine_7_days=np.nan,
                crrt_ever=False, crrt_postop=False)
    n = max(len(v) for v in cols.values())
    return pd.DataFrame({k: cols.get(k, [v] * n) for k, v in base.items()})


def test_kdigo_stages_follow_labels_py():
    df = _label_frame(postop_creatinine_7_days=[1.4, 1.6, 2.0, 2.9, 3.0, 4.1, np.nan],
                      postop_creatinine_2_days=[1.31, 1.2, 1.0, 1.0, 1.0, 1.0, np.nan],
                      preop_creatinine=[1.0, 1.0, 1.0, 1.0, 1.0, 3.9, 1.0])
    out = aki_labels(df)
    assert out["aki_1"].tolist() == [True, True, False, False, False, False, False]
    assert out["aki_2"].tolist() == [False, False, True, True, False, False, False]
    assert out["aki_3"].tolist() == [False, False, False, False, True, True, False]  # > 4 mg/dl without a rise
    assert out["label"].tolist() == [0, 0, 1, 1, 1, 1, 0]
    assert out["in_cohort"].tolist() == [True] * 6 + [False]


def test_headline_label_counts_any_crrt_record_as_labels_py_does():
    """The paper's label (owner decision 2026-10-30): a CRRT record of the patient at any time is stage 3, and keeps
    the operation in the cohort without a post-operative creatinine. The earlier BenchToward rule (CRRT in the 7 days
    after the operation end) is kept as label_crrt_7d for the report."""
    df = _label_frame(postop_creatinine_7_days=[1.0, np.nan, 1.0, np.nan],
                      crrt_ever=[True, True, True, False], crrt_postop=[False, False, True, False])
    out = aki_labels(df)
    assert out["label"].tolist() == [1, 1, 1, 0]
    assert out["aki_3"].tolist() == [True, True, True, False]
    assert out["in_cohort"].tolist() == [True, True, True, False]
    assert out["label_crrt_7d"].tolist() == [0, 0, 1, 0]
    assert out["in_crrt_7d_cohort"].tolist() == [True, False, True, False]
    assert not {"label_authors", "in_authors_cohort"} & set(out.columns)


def test_postop_windows_and_crrt_flags():
    ops = pd.DataFrame([_op(1, opstart_time=10 * DAY, opend_time=10 * DAY + 100)])
    end = 10 * DAY + 100
    cr = pd.DataFrame({"subject_id": [1] * 4, "chart_time": [end, end + 1, end + 2 * DAY, end + 7 * DAY + 1],
                       "value": [9.0, 1.2, 1.5, 8.0]})
    assert postop_max(ops, cr, 2).iloc[0] == 1.5
    assert postop_max(ops, cr, 7).iloc[0] == 1.5
    crrt = pd.DataFrame({"subject_id": [1, 1], "chart_time": [3 * DAY + 1, end + 8 * DAY]})
    f = crrt_flags(ops, crrt, 7).iloc[0]
    assert bool(f["crrt_ever"]) and not f["crrt_postop"] and bool(f["crrt_preop_7d"])


def test_pre_existing_kidney_failure_reads_the_whole_90_day_window():
    """INSPIRE bins creatinine (the top bins are 3.04 and 5.55), so labels.py's '> 4 mg/dl' means a post-operative value
    in the top bin, while the '< 4.5' exclusion reads only the latest pre-operative value. A top-bin value anywhere in
    the 90 days up to the start, or CRRT in the 7 days up to it, marks kidney failure that predates the operation."""
    start = 100 * DAY
    ops = pd.DataFrame([_op(i, subject_id=i, opstart_time=start) for i in (1, 2, 3, 4, 5)])
    cr = pd.DataFrame({"subject_id": [1, 1, 2, 2, 3, 4, 5],
                       "chart_time": [start - 10 * DAY, start - DAY, start - 90 * DAY - 1, start - DAY, start + 1,
                                      start - 90 * DAY, start - DAY],
                       "value": [5.55, 3.04, 5.55, 3.04, 5.55, 5.55, 1.0]})
    df = ops.assign(preop_creatinine_max_90d=preop_max(ops, cr, 90 * DAY))
    assert df["preop_creatinine_max_90d"].tolist()[:2] == [5.55, 3.04]
    assert np.isnan(df["preop_creatinine_max_90d"].iloc[2])                 # only a value after the start
    assert df["preop_creatinine_max_90d"].iloc[3] == 5.55                    # exactly 90 days before: inside
    df["crrt_preop_7d"] = [False, False, False, False, True]
    assert kidney_failure_before(df).tolist() == [True, False, False, True, True]


# ---- GS-AKI ----

def test_gs_aki_factors():
    df = pd.DataFrame([dict(op_id=1, subject_id=1, age=56, sex=1.0, emop=1, icd10_pcs="0DTJ4", opstart_time=100 * DAY,
                            preop_creatinine=1.2),
                       dict(op_id=2, subject_id=2, age=55, sex=0.0, emop=0, icd10_pcs="0JH60", opstart_time=100 * DAY,
                            preop_creatinine=1.19)])
    diag = pd.DataFrame({"subject_id": [1, 1, 1, 2, 2], "icd10_cm": ["E119", "I50", "R18", "I10", "I50"],
                         "chart_time": [1, 99 * DAY, 50 * DAY, 100 * DAY, 99 * DAY]})
    proxy = pd.DataFrame({"icd10_pcs_5char": ["0DTJ4", "0JH60"], "intraperitoneal_proxy": [1, 0]})
    out = gs_aki(df, diag, proxy)
    # op 1: age, male, emergency, intraperitoneal, diabetes, chf (30 d), creatinine; ascites 50 days before: no
    assert out["gs_aki_count"].tolist() == [7, 1]  # op 2: heart failure; hypertension charted at the start: no
    assert out.loc[1, "gs_aki_chf_30d"] == 1 and out.loc[1, "gs_aki_hypertension"] == 0


# ---- split ----

def test_published_split_matches_the_spec():
    spec = load_spec()
    path = published_split_path()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == spec["split"]["sha256"]
    split = read_split()
    assert split["subject_id"].is_unique and set(split["split"]) == {"train", "test"}
    counts = split["split"].value_counts()
    assert counts["test"] == spec["split"]["counts"]["test"]["patients"]
    assert counts["train"] == spec["split"]["counts"]["train"]["patients"]


def test_outer_folds_hold_every_patient_once_and_fold_0_is_the_test_split():
    spec = load_spec()
    assert hashlib.sha256(outer_folds_path().read_bytes()).hexdigest() == spec["split"]["outer_folds"]["sha256"]
    folds = read_outer_folds()
    assert folds["subject_id"].is_unique and set(folds["outer_fold"]) == {0, 1, 2, 3, 4}
    split = read_split().merge(folds, on="subject_id", how="outer", validate="1:1")
    assert split[["split", "outer_fold"]].notna().all().all()
    assert ((split["outer_fold"] == 0) == (split["split"] == "test")).all()


def test_assign_fold_maps_patients_and_refuses_unknown_ones():
    folds = pd.DataFrame({"subject_id": [1, 2], "outer_fold": [3, 0]})
    assert assign_fold(pd.DataFrame({"subject_id": [2, 1, 1]}), folds).tolist() == [0, 3, 3]
    with pytest.raises(ValueError, match="missing from the outer folds"):
        assign_fold(pd.DataFrame({"subject_id": [9]}), folds)


def test_assign_split_is_by_patient_and_refuses_unknown_patients():
    split = pd.DataFrame({"subject_id": [1, 2], "split": ["train", "test"]})
    ops = pd.DataFrame({"subject_id": [1, 1, 2]})
    assert assign_split(ops, split).tolist() == ["train", "train", "test"]
    with pytest.raises(ValueError, match="missing from the split"):
        assign_split(pd.DataFrame({"subject_id": [3]}), split)


def test_inputs_are_the_authors_preoperative_feature_set():
    """build_preop_features' columns minus identifiers, times and death times (datasets/tabular.py): the two
    durations are inputs, as in the authors' pre-operative model; outcomes never are."""
    spec = load_spec()
    cols = input_columns(spec)
    assert len(cols) == len(set(cols)) == 43
    assert {"op_len", "booking_case_length"} <= set(cols)
    assert set(spec["inputs"]["end_of_surgery"]) == {"op_len", "booking_case_length"}
    assert not set(cols) & {"label", "label_crrt_7d", "gs_aki_count", "postop_creatinine_7_days", "crrt_ever",
                            "crrt_postop", "preexisting_kidney_failure", "op_id", "subject_id", "opstart_time",
                            "opend_time", "orin_time", "orout_time"}


# ---- baselines ----

def test_preprocessing_is_fitted_on_train_only():
    from benchtoward.baselines.inspire_aki import preprocess

    rng = np.random.default_rng(0)
    train = pd.DataFrame({"preop_hb": rng.normal(13, 2, 400), "preop_crp": rng.normal(1, 1, 400)})
    train.loc[:9, "preop_hb"] = np.nan          # 2.5% missing: KNN
    train.loc[:99, "preop_crp"] = np.nan        # 25% missing: -99
    test1 = pd.DataFrame({"preop_hb": [13.0, np.nan], "preop_crp": [1.0, np.nan]})
    test2 = pd.DataFrame({"preop_hb": [13.0, 500.0], "preop_crp": [1.0, 900.0]})
    tr1, te1 = preprocess(train, test1)
    tr2, te2 = preprocess(train, test2)
    pd.testing.assert_frame_equal(tr1, tr2)     # test rows never change what is fitted
    assert te1.loc[1, "preop_crp"] == -99 and not np.isnan(te1.loc[1, "preop_hb"])
    assert te1.loc[0, "preop_hb"] == te2.loc[0, "preop_hb"]
    assert abs(te2.loc[1, "preop_hb"]) < 5      # beyond the train 99th percentile: replaced, then scaled


def test_balanced_weights_make_classes_equal():
    from benchtoward.baselines.inspire_aki import balance_weights

    y = np.array([1, 0, 0, 0, 0])
    w = balance_weights(y)
    assert w[y == 1].sum() == w[y == 0].sum() == 4


def test_design_one_hot_uses_train_departments():
    from benchtoward.baselines.inspire_aki import design

    df = pd.DataFrame({"age": [50, 60], "department": ["GS", "NS"]})
    x = design(df, ["age", "department"], ["GS", "OS"])
    assert list(x.columns) == ["age", "department_GS", "department_OS"]
    assert x["department_GS"].tolist() == [1.0, 0.0] and x["department_OS"].tolist() == [0.0, 0.0]


TOY_PARAMS = {"log_reg": {"C": 1.0}, "xgb": {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.1}}


def _toy_instances(n=600, seed=0):
    rng = np.random.default_rng(seed)
    spec = load_spec()
    df = pd.DataFrame({c: rng.normal(size=n) for c in input_columns(spec) if c != "department"})
    df["department"] = rng.choice(["GS", "OS", "NS"], n)
    df["asa"] = rng.integers(1, 5, n)
    df["gs_aki_count"] = rng.integers(0, 6, n)
    risk = 1 / (1 + np.exp(-(2 * df["preop_creatinine"] - 3)))
    df["label"] = (rng.random(n) < risk).astype(int)
    df["subject_id"] = np.arange(n) // 2
    df["op_id"] = np.arange(n) + 400000000
    df["instance_id"] = "inspire-aki-" + df["op_id"].astype(str)
    df["outer_fold"] = (df["subject_id"] * 7) % 5
    df["split"] = np.where(df["outer_fold"] == 0, "test", "train")
    return df


def test_baselines_learn_the_signal(monkeypatch):
    pytest.importorskip("xgboost")
    from sklearn.metrics import roc_auc_score

    from benchtoward.baselines import inspire_aki as b

    monkeypatch.setattr(b, "hpo_params", lambda variant="headline": TOY_PARAMS)
    inst = _toy_instances()
    scores = b.score_all(inst)
    test = inst[inst["split"] == "test"]
    assert set(scores) >= set(b.RECORDED)
    assert np.unique(scores["benchtoward/prior"]).size == 1
    for slug in ("benchtoward/inspire-aki-logreg", "benchtoward/inspire-aki-xgb"):
        assert roc_auc_score(test["label"], scores[slug + ":raw"]) > 0.75
        assert ((scores[slug] >= 0) & (scores[slug] <= 1)).all()


def test_cli_registers_fetch_build_and_baseline(monkeypatch):
    from typer.testing import CliRunner

    from benchtoward.cli import app

    runner = CliRunner()
    for group in ("fetch", "build", "baseline"):
        assert "inspire-aki" in runner.invoke(app, [group, "--help"]).output
    monkeypatch.delenv("KHDP_TOKEN", raising=False)
    result = runner.invoke(app, ["fetch", "inspire-aki"])
    assert result.exit_code == 1 and "KHDP_TOKEN" in result.output


def _toy_report_instances():
    inst = _toy_instances()
    rng = np.random.default_rng(1)
    flip = rng.random(len(inst)) < 0.1
    inst["label_crrt_7d"] = np.where(flip, 0, inst["label"])
    inst["in_crrt_7d_cohort"] = ~(flip & (inst.index % 2 == 0))
    inst["crrt_ever"] = inst["crrt_postop"] = False
    inst["crrt_preop_7d"] = inst.index % 7 == 0
    inst["preexisting_kidney_failure"] = inst["crrt_preop_7d"] | (inst.index % 11 == 0)
    inst["num_card_events_same_day"] = rng.integers(0, 2, len(inst))
    return inst


def test_report_rows_change_one_headline_choice_each(monkeypatch):
    """The earlier BenchToward choices are report rows: the CRRT-in-7-days label (its own cohort), the inputs without
    the two durations, and pre-existing kidney failure left out (headline models scored without it, and retrained)."""
    pytest.importorskip("xgboost")
    from sklearn.metrics import roc_auc_score

    from benchtoward.baselines import inspire_aki as b

    used = []
    monkeypatch.setattr(b, "hpo_params", lambda variant="headline": used.append(variant) or TOY_PARAMS)
    monkeypatch.setattr(b, "five_fold", lambda instances, **kw: {"folds": []})
    inst = _toy_report_instances()
    out = b.report(inst, n_boot=20)

    test = inst[inst["split"] == "test"].reset_index(drop=True)
    assert out["headline"]["n"] == len(test) and out["headline"]["positives"] == int(test["label"].sum())
    crrt = test[test["in_crrt_7d_cohort"]]
    assert out["crrt_7d_label"]["n"] == len(crrt)
    assert out["crrt_7d_label"]["positives"] == int(crrt["label_crrt_7d"].sum())
    assert out["without_durations"]["n"] == len(test)
    assert out["earlier_headline"]["positives"] == int(crrt["label_crrt_7d"].sum())
    assert {"headline", "crrt_7d_label", "without_durations", "earlier_headline"} <= set(used)

    keep = ~test["preexisting_kidney_failure"].to_numpy()
    sub = out["headline_models__without_preexisting_kidney_failure"]
    assert sub["n"] == int(keep.sum()) and sub["positives"] == int(test.loc[keep, "label"].sum())
    xgb = b.score_all(inst)["benchtoward/inspire-aki-xgb"]
    assert sub["models"]["benchtoward/inspire-aki-xgb"]["auroc"]["value"] == pytest.approx(
        roc_auc_score(test.loc[keep, "label"], xgb[keep]), abs=1e-4)
    assert out["headline_models__preexisting_kidney_failure"]["n"] == int((~keep).sum())
    assert out["retrained_without_preexisting_kidney_failure"]["n"] == int(keep.sum())
    for probe in ("label:op_len", "label:preexisting_kidney_failure", "label_crrt_7d:crrt_ever",
                  "label:num_card_events_same_day"):
        assert probe in out["leak_probes"]


def test_without_durations_drops_only_the_two_end_of_surgery_columns():
    from benchtoward.baselines import inspire_aki as b

    cols = b.variant_columns("without_durations")
    assert set(input_columns()) - set(cols) == {"op_len", "booking_case_length"}
    assert b.variant_columns("headline") == input_columns()


def test_five_fold_trains_each_outer_fold_with_the_authors_seeds_and_averages(monkeypatch):
    """Fold k: its own hyperparameters (hpo fold_k; fold 0 is the headline), model seed 42 + k, inner calibration
    splitter seed 42 + k and inner model seeds 42 + 100 k + j (pipelines/train.py, split_manager.py). The summary is
    the mean over the five folds with a t interval, as the authors' reporting/tables.py makes Table 2."""
    pytest.importorskip("xgboost")
    from scipy import stats

    from benchtoward.baselines import inspire_aki as b

    variants, seeds, inner = [], [], []
    monkeypatch.setattr(b, "hpo_params", lambda variant="headline": variants.append(variant) or TOY_PARAMS)
    real_fit, real_inner = b.fit_predict_model, b.inner_folds
    monkeypatch.setattr(b, "fit_predict_model", lambda kind, params, xtr, ytr, xte, seed, **kw: (
        seeds.append((kind, seed)), real_fit(kind, params, xtr, ytr, xte, seed, **kw))[1])
    monkeypatch.setattr(b, "inner_folds", lambda train, label, n_splits=5, seed=42: (
        inner.append(seed), real_inner(train, label, n_splits, seed))[1])
    inst = _toy_instances()
    out = b.five_fold(inst)

    assert [f["fold"] for f in out["folds"]] == [0, 1, 2, 3, 4]
    assert sum(f["n"] for f in out["folds"]) == len(inst)
    assert variants.count("headline") == 1 and all(f"fold_{k}" in variants for k in range(1, 5))
    assert sorted(set(inner)) == [42, 43, 44, 45, 46]
    lr = [seed for kind, seed in seeds if kind == "log_reg"]
    assert {42, 43, 44, 45, 46} <= set(lr) and {42 + 100 * 3 + j for j in range(5)} <= set(lr)
    values = [f["models"]["benchtoward/inspire-aki-xgb"]["auroc"] for f in out["folds"]]
    mean = out["mean"]["benchtoward/inspire-aki-xgb"]["auroc"]
    low, high = stats.t.interval(0.95, df=4, loc=np.mean(values), scale=stats.sem(values))
    assert mean["value"] == pytest.approx(np.mean(values), abs=1e-4)
    assert (mean["ci_low"], mean["ci_high"]) == (pytest.approx(low, abs=1e-4), pytest.approx(high, abs=1e-4))
    prior = out["mean"]["benchtoward/prior"]["auroc"]
    assert prior == {"value": 0.5, "ci_low": 0.5, "ci_high": 0.5}
    assert set(out["mean"]["benchtoward/inspire-aki-asa"]) == {"auroc", "auprc", "auprc_trapezoid"}


def test_trapezoid_auprc_is_the_papers_table_2_area():
    """reporting/tables.py _safe_trapezoidal_auprc: sklearn auc(recall, precision), which reads higher than average
    precision for a score with few values (the ASA class)."""
    from sklearn.metrics import auc, average_precision_score, precision_recall_curve

    from benchtoward.baselines.inspire_aki import trapezoid_auprc

    y = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1])
    s = np.array([1, 1, 2, 2, 3, 3, 1, 2, 1, 3]) / 6
    precision, recall, _ = precision_recall_curve(y, s)
    assert trapezoid_auprc(y, s) == pytest.approx(auc(recall, precision))
    assert trapezoid_auprc(y, s) > average_precision_score(y, s)


def test_hpo_matrix_holds_the_train_rows_preprocessed_on_train():
    from benchtoward.baselines import inspire_aki as b

    inst = _toy_instances()
    x = b.hpo_matrix(inst, "label", input_columns())
    train = inst[inst["split"] == "train"]
    assert x["op_id"].tolist() == train["op_id"].tolist()
    assert x["patient_id"].tolist() == train["subject_id"].tolist()
    assert x["aki_boolean"].tolist() == train["label"].astype(bool).tolist()
    assert "department_GS" in x.columns and not {"split", "label", "subject_id"} & set(x.columns)
    inst = _toy_report_instances()
    matrices = b.hpo_matrices(inst)
    assert set(matrices) == {"headline", "fold_1", "fold_2", "fold_3", "fold_4", "crrt_7d_label", "without_durations",
                             "earlier_headline"}
    assert len(matrices["fold_2"]) == int((inst["outer_fold"] != 2).sum())
    crrt_train = inst[inst["in_crrt_7d_cohort"] & (inst["split"] == "train")]
    assert matrices["crrt_7d_label"]["aki_boolean"].tolist() == crrt_train["label_crrt_7d"].astype(bool).tolist()
    assert "op_len" not in matrices["earlier_headline"] and "op_len" in matrices["crrt_7d_label"]
