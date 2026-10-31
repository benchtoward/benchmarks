"""BenchToward baselines for scope-arrest@0, trained on the train split and scored on the test split.

- benchtoward/prior: the train-split prevalence for every epoch (ignores the input; an unranked reference row).
- benchtoward/scope-hrv-lgbm: the dataset authors' training script (Lee et al. 2023 setup) on the 33 HRV measures:
  missing measures filled with medians; a 5-fold CV of the train split grouped by stay and stratified by label
  (StratifiedGroupKFold, shuffled, random_state 42); in each fold a Bayesian optimization (3 random and 50 guided
  points) of LightGBM's parameters, scored by AUPRC on the held-out fold with early stopping (300 rounds, at most
  2000); the parameters of the fold with the best AUPRC; boosting rounds = the mean best iteration of a second CV
  with those parameters, truncated; one model on the whole train split; Beta calibration fitted on its train
  predictions.
Models see only the HRV measures: never the stay id or the epoch's position relative to the outcome.
"""
import hashlib
import os

import numpy as np
import pandas as pd

from benchtoward.tasks.scope_hrv import FEATURES

SEED = 42
FIXED_PARAMS = {"objective": "binary", "learning_rate": 0.005, "bagging_freq": 1, "force_row_wise": True,
                "max_depth": 5, "verbose": -1, "random_state": SEED}
PARAM_BOUNDS = {"num_leaves": (16, 32), "lambda_l1": (0.7, 0.9), "lambda_l2": (0.9, 1),
                "feature_fraction": (0.6, 0.7), "bagging_fraction": (0.6, 0.9), "min_child_samples": (6, 10),
                "min_child_weight": (10, 40)}
OPT_INITS, OPT_ITERS = 3, 50
MAX_ROUNDS, PATIENCE, N_FOLDS = 2000, 300, 5


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURES].astype(float).replace([np.inf, -np.inf], np.nan)


def impute_train_medians(train: pd.DataFrame, *others: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """Fill missing values with the train split's column medians (infinite values are not counted)."""
    medians = train.replace([np.inf, -np.inf], np.nan).median()
    return tuple(frame.fillna(medians) for frame in (train, *others))


def fit_predict_prior(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, dict]:
    return np.full(len(test), float(train["label"].mean())), {}


def _auprc(preds, dataset):
    from sklearn.metrics import average_precision_score

    return "auprc", average_precision_score(dataset.get_label(), preds), True


def _tune_fold(args) -> dict:
    """One fold of the authors' first CV: Bayesian optimization of the parameters, AUPRC on the held-out fold."""
    import lightgbm as lgb
    from bayes_opt import BayesianOptimization
    from sklearn.metrics import average_precision_score

    X, y, fit_idx, stop_idx, opts = args
    dfit, dstop = lgb.Dataset(X[fit_idx], y[fit_idx]), lgb.Dataset(X[stop_idx], y[stop_idx])

    def score(num_leaves, lambda_l1, lambda_l2, feature_fraction, bagging_fraction, min_child_samples,
              min_child_weight):
        params = {"num_leaves": int(round(num_leaves)), "lambda_l1": lambda_l1, "lambda_l2": lambda_l2,
                  "feature_fraction": feature_fraction, "bagging_fraction": bagging_fraction,
                  "min_child_samples": int(round(min_child_samples)), "min_child_weight": min_child_weight,
                  "feature_pre_filter": False, **FIXED_PARAMS, "n_jobs": opts["n_jobs"]}
        model = lgb.train(params, dfit, num_boost_round=opts["max_rounds"], valid_sets=dstop, feval=_auprc,
                          callbacks=[lgb.early_stopping(stopping_rounds=opts["patience"], verbose=False)])
        return average_precision_score(y[stop_idx], model.predict(X[stop_idx]))

    optimizer = BayesianOptimization(f=score, pbounds=PARAM_BOUNDS, random_state=SEED, verbose=0)
    optimizer.maximize(init_points=opts["opt_inits"], n_iter=opts["opt_iters"])
    return {"params": dict(optimizer.max["params"]), "auprc": float(optimizer.max["target"])}


def _best_iteration(args) -> int:
    """One fold of the authors' second CV: the best iteration with early stopping."""
    import lightgbm as lgb

    X, y, fit_idx, stop_idx, params, opts = args
    model = lgb.train(params, lgb.Dataset(X[fit_idx], y[fit_idx]), num_boost_round=opts["max_rounds"],
                      valid_sets=lgb.Dataset(X[stop_idx], y[stop_idx]), feval=_auprc,
                      callbacks=[lgb.early_stopping(stopping_rounds=opts["patience"], verbose=False)])
    return int(model.best_iteration)


def _map(fn, jobs: list, processes: int) -> list:
    if processes <= 1:
        return [fn(job) for job in jobs]
    import multiprocessing as mp

    with mp.get_context("spawn").Pool(processes) as pool:  # LightGBM's OpenMP does not survive fork on macOS
        return pool.map(fn, jobs)


def fit_predict_hrv_lgbm(train: pd.DataFrame, test: pd.DataFrame, *, opt_inits: int = OPT_INITS,
                         opt_iters: int = OPT_ITERS, max_rounds: int = MAX_ROUNDS, patience: int = PATIENCE,
                         n_jobs: int | None = None, fold_processes: int = 1, log=None) -> tuple[np.ndarray, dict]:
    import lightgbm as lgb
    from betacal import BetaCalibration
    from sklearn.model_selection import StratifiedGroupKFold

    n_jobs = n_jobs or os.cpu_count() or 1
    opts = {"opt_inits": opt_inits, "opt_iters": opt_iters, "max_rounds": max_rounds, "patience": patience,
            "n_jobs": max(1, n_jobs // max(1, fold_processes))}
    X_df, X_test_df = impute_train_medians(feature_matrix(train), feature_matrix(test))
    X, X_test = X_df.to_numpy(dtype=float), X_test_df.to_numpy(dtype=float)
    y, groups = train["label"].to_numpy(dtype=int), train["case_id"].to_numpy()
    folds = list(StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X, y, groups))

    tuned = _map(_tune_fold, [(X, y, a, b, opts) for a, b in folds], fold_processes)
    best_fold = int(np.argmax([t["auprc"] for t in tuned]))
    params = dict(tuned[best_fold]["params"])
    params["num_leaves"] = int(round(params["num_leaves"]))
    params["min_child_samples"] = int(round(params["min_child_samples"]))
    params.update(FIXED_PARAMS)
    if log:
        log(f"tuned: fold AUPRC {[round(t['auprc'], 4) for t in tuned]}; params from fold {best_fold + 1}: {params}")

    run_params = {**params, "n_jobs": opts["n_jobs"]}
    best_iters = _map(_best_iteration, [(X, y, a, b, run_params, opts) for a, b in folds], fold_processes)
    final_iter = int(np.mean(best_iters))
    model = lgb.train({**params, "n_jobs": n_jobs}, lgb.Dataset(X, y), num_boost_round=final_iter)
    calibrator = BetaCalibration()
    calibrator.fit(model.predict(X).reshape(-1, 1), y)
    raw_test = model.predict(X_test)
    scores = calibrator.predict(raw_test.reshape(-1, 1))
    info = {"params": params, "fold_best_auprc": [t["auprc"] for t in tuned], "best_fold": best_fold + 1,
            "cv_best_iterations": best_iters, "final_iterations": final_iter, "calibration": "beta",
            "imputed_values": {"train": int(feature_matrix(train).isna().sum().sum()),
                               "test": int(feature_matrix(test).isna().sum().sum())},
            "bayes_opt": {"init_points": opt_inits, "n_iter": opt_iters}, "_raw_test": raw_test}
    return np.asarray(scores, dtype=float), info


BASELINES = {
    "benchtoward/prior": fit_predict_prior,
    "benchtoward/scope-hrv-lgbm": fit_predict_hrv_lgbm,
}


def data_hash(instances: pd.DataFrame) -> str:
    """Fingerprint of the instances, splits, labels and features, so a rebuilt task never reuses stale runs."""
    cols = ["instance_id", "split", "label", *FEATURES]
    key = pd.util.hash_pandas_object(instances[cols].sort_values("instance_id"), index=False).to_numpy()
    return hashlib.sha256(key.tobytes()).hexdigest()


def run_scope_baselines(session, task, instances: pd.DataFrame, *, data_dir, n_boot: int = 1000,
                        fit_kwargs: dict | None = None) -> list:
    from benchtoward.db import Model
    from benchtoward.evaluate import record_baseline_run
    from benchtoward.scoring import binary_metrics_ci

    train, test = (instances[instances["split"] == s] for s in ("train", "test"))
    fingerprint = data_hash(instances)
    runs = []
    for slug, fit in BASELINES.items():
        kwargs = (fit_kwargs or {}) if slug == "benchtoward/scope-hrv-lgbm" else {}
        scores, info = fit(train, test, **kwargs)
        metrics = binary_metrics_ci(test["label"].to_numpy(), scores, test["case_id"].to_numpy(), n_boot=n_boot)
        rows = [{"instance_id": iid, "score": float(s), "label": int(y)}
                for iid, s, y in zip(test["instance_id"], scores, test["label"])]
        config = {"data_hash": fingerprint, "train_split": "train", "eval_split": "test", "features": "hrv33",
                  **{k: v for k, v in info.items() if not k.startswith("_")}}
        runs.append(record_baseline_run(session, session.get(Model, slug), task, rows, metrics, config,
                                        data_dir=data_dir))
    return runs
