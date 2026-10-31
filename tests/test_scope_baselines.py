"""Baselines of scope-arrest@0: the input-blind prior and the dataset authors' training pipeline (LightGBM on 33 HRV
measures, Bayesian optimization in a stay-grouped 5-fold CV, boosting rounds from a second CV, Beta calibration)."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from benchtoward.baselines.scope_arrest import (BASELINES, FIXED_PARAMS, PARAM_BOUNDS, feature_matrix,
                                             fit_predict_hrv_lgbm, fit_predict_prior, impute_train_medians)
from benchtoward.tasks.scope_hrv import CANDIDATES, FEATURES

FAST = {"opt_inits": 1, "opt_iters": 1, "max_rounds": 60, "patience": 10, "n_jobs": 2}


def toy(n_stays: int, rng, split: str, start: int = 0) -> pd.DataFrame:
    rows = []
    for k in range(n_stays):
        case_id, label = 100000000 + start + k, int(k % 5 == 0)
        for j in range(20):
            row = {name: rng.normal(0, 1) for name in CANDIDATES}
            row["MeanNN"] = rng.normal(700 if label else 850, 60)
            row.update(instance_id=f"scope-{case_id}-{j}", case_id=case_id, split=split, label=label,
                       outcome="Cardiac Arrest" if label else "Discharge", start_s=-1800.0 - 300 * (j + 1),
                       minutes_to_end=30.0 + 5 * j, quality_ratio=0.9)
            rows.append(row)
    return pd.DataFrame(rows)


def data(seed=0):
    rng = np.random.default_rng(seed)
    return pd.concat([toy(60, rng, "train"), toy(30, rng, "test", 2000)], ignore_index=True)


def test_the_authors_settings():
    assert FIXED_PARAMS == {"objective": "binary", "learning_rate": 0.005, "bagging_freq": 1, "force_row_wise": True,
                            "max_depth": 5, "verbose": -1, "random_state": 42}
    assert PARAM_BOUNDS == {"num_leaves": (16, 32), "lambda_l1": (0.7, 0.9), "lambda_l2": (0.9, 1),
                            "feature_fraction": (0.6, 0.7), "bagging_fraction": (0.6, 0.9),
                            "min_child_samples": (6, 10), "min_child_weight": (10, 40)}


def test_feature_matrix_is_the_33_measures_only():
    X = feature_matrix(data())
    assert list(X.columns) == FEATURES
    assert not {"start_s", "minutes_to_end", "quality_ratio", "case_id", "label", "split"} & set(X.columns)


def test_missing_measures_take_the_train_medians():
    train = pd.DataFrame({"a": [1.0, 2.0, np.nan, 10.0], "b": [np.inf, 1.0, 1.0, 1.0]})
    test = pd.DataFrame({"a": [np.nan, 5.0], "b": [np.nan, 2.0]})
    filled_train, filled_test = impute_train_medians(train, test)
    assert filled_train["a"].tolist() == [1.0, 2.0, 2.0, 10.0] and filled_test["a"].tolist() == [2.0, 5.0]
    assert filled_test["b"].tolist() == [1.0, 2.0]  # the median ignores infinite values


def test_prior_is_the_train_prevalence():
    df = data()
    scores, info = fit_predict_prior(df[df.split == "train"], df[df.split == "test"])
    assert np.allclose(scores, df[df.split == "train"]["label"].mean()) and info == {}


def test_authors_pipeline_learns_a_separable_signal_and_ignores_time():
    df = data()
    train, test = df[df.split == "train"], df[df.split == "test"]
    scores, info = fit_predict_hrv_lgbm(train, test, **FAST)
    assert roc_auc_score(test["label"], scores) > 0.9 and ((0 <= scores) & (scores <= 1)).all()
    assert len(info["fold_best_auprc"]) == 5 and len(info["cv_best_iterations"]) == 5
    assert info["final_iterations"] == int(np.mean(info["cv_best_iterations"]))  # the authors truncate the mean
    assert info["params"]["num_leaves"] == int(info["params"]["num_leaves"]) and info["calibration"] == "beta"
    leaky = df.assign(start_s=np.where(df["label"] == 1, -2100.0, -80000.0), minutes_to_end=0.0)
    again, _ = fit_predict_hrv_lgbm(leaky[leaky.split == "train"], leaky[leaky.split == "test"], **FAST)
    assert np.array_equal(scores, again)


def test_calibration_keeps_the_ranking():
    df = data(2)
    train, test = df[df.split == "train"], df[df.split == "test"]
    scores, info = fit_predict_hrv_lgbm(train, test, **FAST)
    raw = np.asarray(info.pop("_raw_test"))
    order = np.argsort(raw, kind="stable")
    assert (np.diff(scores[order]) >= -1e-12).all()


def test_run_records_runs_with_a_stay_bootstrap(tmp_session, tmp_path, monkeypatch):
    from conftest import REGISTRY_DIR
    from benchtoward.baselines import scope_arrest
    from benchtoward.db import ItemResult
    from benchtoward.registry import load_registry, resolve_task

    monkeypatch.setitem(scope_arrest.BASELINES, "benchtoward/scope-hrv-lgbm",
                        lambda train, test: fit_predict_hrv_lgbm(train, test, **FAST))
    load_registry(tmp_session, REGISTRY_DIR)
    df = data(1)
    task = resolve_task(tmp_session, "scope-arrest", include_hidden=True)
    runs = scope_arrest.run_scope_baselines(tmp_session, task, df, data_dir=tmp_path, n_boot=30)
    assert sorted(r.model_slug for r in runs) == sorted(BASELINES) == ["benchtoward/prior", "benchtoward/scope-hrv-lgbm"]
    n_test = int((df.split == "test").sum())
    for run in runs:
        metrics = json.loads(run.metrics_json)
        assert metrics["n"] == n_test and metrics["n_groups"] == 30 and "auprc" in metrics
        assert tmp_session.query(ItemResult).filter_by(run_id=run.id).count() == n_test
        assert (tmp_path / "artifacts" / run.id / "predictions.parquet").exists()
    lgbm = next(r for r in runs if r.model_slug == "benchtoward/scope-hrv-lgbm")
    assert json.loads(lgbm.metrics_json)["auroc"]["value"] > 0.9
    config = json.loads(lgbm.config_json)
    assert config["calibration"] == "beta" and "params" in config and "_raw_test" not in config

