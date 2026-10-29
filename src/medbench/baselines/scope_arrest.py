"""medbench baselines for scope-arrest@0, trained on the train split and scored on the test split.

- medbench/prior: the train-split prevalence for every epoch (ignores the input; an unranked reference row).
- medbench/scope-hrv-lgbm: LightGBM on the 33 HRV measures of Lee et al. 2023 (npj Digit Med 6:215), with the
  paper's fixed parameters and the midpoints of its search ranges (github.com/HyeonhoonLee/hrvarrest, main.py);
  boosting rounds from the paper's second 5-fold cross-validation (early stopping on AUPRC, mean best iteration),
  run on the train split grouped by stay. No Bayesian search and no Beta calibration (a monotone map: AUROC and
  AUPRC do not change).
Models see only the HRV measures: never the stay id or the epoch's position relative to the outcome.
"""
import hashlib

import numpy as np
import pandas as pd

from medbench.tasks.scope_hrv import FEATURES

LGBM_PARAMS = {
    "objective": "binary", "learning_rate": 0.005, "bagging_freq": 1, "force_row_wise": True, "max_depth": 5,
    "verbose": -1, "seed": 42, "deterministic": True, "num_threads": 8,
    # midpoints of the paper's Bayesian-optimization bounds
    "num_leaves": 24, "lambda_l1": 0.8, "lambda_l2": 0.95, "feature_fraction": 0.65, "bagging_fraction": 0.75,
    "min_child_samples": 8, "min_child_weight": 25.0,
}
MAX_ROUNDS, PATIENCE, CV_FOLDS = 2000, 300, 5


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURES].astype(float).replace([np.inf, -np.inf], np.nan)


def fit_predict_prior(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, dict]:
    return np.full(len(test), float(train["label"].mean())), {}


def fit_predict_hrv_lgbm(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Rounds as in the paper's second cross-validation: mean best iteration (early stopping on AUPRC) over 5 folds
    of the train split, grouped by stay; then one model on the whole train split. The validation split is unused."""
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import StratifiedGroupKFold

    def auprc(preds, dataset):
        return "auprc", average_precision_score(dataset.get_label(), preds), True

    X, y, groups = feature_matrix(train), train["label"].to_numpy(), train["case_id"].to_numpy()
    params = {**LGBM_PARAMS, "metric": "None"}
    best = []
    for fit_idx, stop_idx in StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42).split(X, y, groups):
        dfit = lgb.Dataset(X.iloc[fit_idx], y[fit_idx])
        dstop = lgb.Dataset(X.iloc[stop_idx], y[stop_idx], reference=dfit)
        booster = lgb.train(params, dfit, num_boost_round=MAX_ROUNDS, valid_sets=[dstop], feval=auprc,
                            callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
        best.append(int(booster.best_iteration or MAX_ROUNDS))
    rounds = max(1, round(float(np.mean(best))))
    booster = lgb.train(params, lgb.Dataset(X, y), num_boost_round=rounds)
    return booster.predict(feature_matrix(test)), {"best_iteration": rounds, "cv_best_iterations": best}


BASELINES = {
    "medbench/prior": fit_predict_prior,
    "medbench/scope-hrv-lgbm": fit_predict_hrv_lgbm,
}


def data_hash(instances: pd.DataFrame) -> str:
    """Fingerprint of the instances, splits, labels and features, so a rebuilt task never reuses stale runs."""
    cols = ["instance_id", "split", "label", *FEATURES]
    key = pd.util.hash_pandas_object(instances[cols].sort_values("instance_id"), index=False).to_numpy()
    return hashlib.sha256(key.tobytes()).hexdigest()


def run_scope_baselines(session, task, instances: pd.DataFrame, *, data_dir, n_boot: int = 1000) -> list:
    from medbench.db import Model
    from medbench.evaluate import record_baseline_run
    from medbench.scoring import binary_metrics_ci

    train, validation, test = (instances[instances["split"] == s] for s in ("train", "validation", "test"))
    fingerprint = data_hash(instances)
    runs = []
    for slug, fit in BASELINES.items():
        scores, info = fit(train, validation, test)
        metrics = binary_metrics_ci(test["label"].to_numpy(), scores, test["case_id"].to_numpy(), n_boot=n_boot)
        rows = [{"instance_id": iid, "score": float(s), "label": int(y)}
                for iid, s, y in zip(test["instance_id"], scores, test["label"])]
        config = {"data_hash": fingerprint, "train_split": "train", "eval_split": "test", "features": "hrv33", **info}
        if slug == "medbench/scope-hrv-lgbm":
            config["params"] = LGBM_PARAMS
        runs.append(record_baseline_run(session, session.get(Model, slug), task, rows, metrics, config,
                                        data_dir=data_dir))
    return runs
