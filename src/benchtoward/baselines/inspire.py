"""The three models of the INSPIRE authors' mortality script (gbm_mortality.py), as BenchToward baselines.

Each is trained on the train split only and scored on the test split. The settings are the script's; xgboost is a
dev dependency: baselines run on a workstation, and the web service does not need it.
"""
import hashlib

import numpy as np
import pandas as pd

from benchtoward.tasks.inspire_mortality import input_columns, load_spec

LOGREG_PARAMS = {"max_iter": 5000}
XGB_PARAMS = {"max_depth": 4, "n_estimators": 50, "subsample": 0.8, "colsample_bytree": 0.8, "eval_metric": "logloss"}


def model_inputs() -> list[str]:
    """The script's INPUT_VARS (ASA is an input column of the task, but the script's models do not use it)."""
    return list(load_spec()["inputs"]["script"])


def _x(df: pd.DataFrame) -> pd.DataFrame:
    return df[model_inputs()].astype(float)


def fit_predict_asa(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return test["asa"].to_numpy(dtype=float)


def fit_predict_logreg(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression

    imputer = SimpleImputer().fit(_x(train))
    model = LogisticRegression(**LOGREG_PARAMS).fit(imputer.transform(_x(train)), train["label"].astype(int))
    return model.predict_proba(imputer.transform(_x(test)))[:, 1]


def fit_predict_xgb(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    from xgboost import XGBClassifier

    model = XGBClassifier(**XGB_PARAMS).fit(_x(train), train["label"].astype(int))
    return model.predict_proba(_x(test))[:, 1]


def _sklearn_version() -> dict:
    import sklearn

    return {"scikit-learn": sklearn.__version__}


def _xgboost_version() -> dict:
    import xgboost

    return {"xgboost": xgboost.__version__}


# slug -> (fit, settings recorded in the run config, library versions)
BASELINES = {
    "benchtoward/inspire-asa": (fit_predict_asa, {"score": "asa"}, dict),
    "benchtoward/inspire-logreg": (fit_predict_logreg, {"imputer": "SimpleImputer(mean)", **LOGREG_PARAMS},
                                _sklearn_version),
    "benchtoward/inspire-xgb": (fit_predict_xgb, dict(XGB_PARAMS), _xgboost_version),
}


def data_hash(instances: pd.DataFrame) -> str:
    """Fingerprint of ids, splits, labels and inputs, so rebuilt data never reuses stale baseline runs."""
    columns = ["instance_id", "split", "label", *input_columns()]
    key = instances[columns].sort_values("instance_id").to_csv(index=False)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def run_inspire_baselines(session, task, instances: pd.DataFrame, *, data_dir, n_boot: int = 1000) -> list:
    """Train each baseline on the train split, score the test split, and record one run per baseline."""
    from benchtoward.db import Model
    from benchtoward.evaluate import record_baseline_run
    from benchtoward.scoring import binary_metrics_ci

    spec = load_spec()
    metric_names = [spec["metrics"]["primary"], *spec["metrics"]["secondary"]]
    train = instances[instances["split"] == "train"]
    test = instances[instances["split"] == "test"]
    fingerprint = data_hash(instances)
    runs = []
    for slug, (fit, params, versions) in BASELINES.items():
        scores = fit(train, test)
        metrics = binary_metrics_ci(test["label"].to_numpy(), scores, test["subject_id"].to_numpy(), n_boot=n_boot,
                                    metrics=metric_names)
        rows = [{"instance_id": iid, "score": float(s), "label": int(y)}
                for iid, s, y in zip(test["instance_id"], scores, test["label"])]
        config = {"data_hash": fingerprint, "train_split": "train", "eval_split": "test", "params": params,
                  "versions": versions()}
        runs.append(record_baseline_run(session, session.get(Model, slug), task, rows, metrics, config,
                                        data_dir=data_dir))
    return runs
