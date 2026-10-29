import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from medbench.baselines.scope_arrest import BASELINES, feature_matrix, fit_predict_hrv_lgbm, fit_predict_prior
from medbench.tasks.scope_hrv import CANDIDATES, FEATURES


def toy(n_stays: int, rng, split: str, start: int = 0) -> pd.DataFrame:
    rows = []
    for k in range(n_stays):
        case_id, label = 100000000 + start + k, int(k % 5 == 0)
        for j in range(20):
            row = {name: rng.normal(0, 1) for name in CANDIDATES}
            row["MeanNN"] = rng.normal(700 if label else 850, 60)
            row.update(instance_id=f"scope-{case_id}-{j}", case_id=case_id, split=split, label=label,
                       outcome="Cardiac Arrest" if label else "Discharge", start_s=-1800 - 300 * (j + 1))
            rows.append(row)
    return pd.DataFrame(rows)


def data(seed=0):
    rng = np.random.default_rng(seed)
    return pd.concat([toy(60, rng, "train"), toy(20, rng, "validation", 1000), toy(30, rng, "test", 2000)],
                     ignore_index=True)


def test_feature_matrix_is_the_33_measures_only():
    df = data()
    X = feature_matrix(df)
    assert list(X.columns) == FEATURES
    assert not {"start_s", "case_id", "label", "split", "outcome"} & set(X.columns)


def test_prior_is_the_train_prevalence():
    df = data()
    scores, info = fit_predict_prior(df[df.split == "train"], df[df.split == "validation"], df[df.split == "test"])
    assert np.allclose(scores, df[df.split == "train"]["label"].mean()) and info == {}


def test_lgbm_learns_a_separable_signal_and_ignores_time():
    df = data()
    train, val, test = (df[df.split == s] for s in ("train", "validation", "test"))
    scores, info = fit_predict_hrv_lgbm(train, val, test)
    assert roc_auc_score(test["label"], scores) > 0.9 and 1 <= info["best_iteration"] <= 2000
    # rounds = mean of the best iterations of a 5-fold, stay-grouped CV on the train split (the paper's second CV)
    assert len(info["cv_best_iterations"]) == 5 and info["best_iteration"] == round(np.mean(info["cv_best_iterations"]))
    leaky = df.assign(start_s=np.where(df["label"] == 1, -2100, -80000))  # a time column that would leak the label
    again, _ = fit_predict_hrv_lgbm(*(leaky[leaky.split == s] for s in ("train", "validation", "test")))
    assert np.array_equal(scores, again)


def test_run_records_runs_with_a_stay_bootstrap(tmp_session, tmp_path):
    from conftest import REGISTRY_DIR
    from medbench.baselines.scope_arrest import run_scope_baselines
    from medbench.db import ItemResult
    from medbench.registry import load_registry, resolve_task

    load_registry(tmp_session, REGISTRY_DIR)
    df = data(1)
    runs = run_scope_baselines(tmp_session, resolve_task(tmp_session, "scope-arrest", include_hidden=True), df, data_dir=tmp_path, n_boot=30)
    assert sorted(r.model_slug for r in runs) == sorted(BASELINES) == ["medbench/prior", "medbench/scope-hrv-lgbm"]
    n_test = int((df.split == "test").sum())
    for run in runs:
        metrics = json.loads(run.metrics_json)
        assert metrics["n"] == n_test and metrics["n_groups"] == 30 and "auprc" in metrics
        assert tmp_session.query(ItemResult).filter_by(run_id=run.id).count() == n_test
        assert (tmp_path / "artifacts" / run.id / "predictions.parquet").exists()
    lgbm = next(r for r in runs if r.model_slug == "medbench/scope-hrv-lgbm")
    assert json.loads(lgbm.metrics_json)["auroc"]["value"] > 0.9
    assert "best_iteration" in json.loads(lgbm.config_json)
