"""Two BenchToward baselines for vitaldb-ioh, trained on the train split."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

INTERVAL_MIN = 2 / 60  # minutes per sample
LAST = 30  # last 60 s


def features(x: np.ndarray) -> np.ndarray:
    """mean, min, max, std, slope (mmHg/min), mean of last 60 s, last valid value, fraction missing."""
    x = np.asarray(x, dtype=float)
    valid = ~np.isnan(x)
    if not valid.any():
        return np.zeros(8)
    t = np.flatnonzero(valid) * INTERVAL_MIN
    v = x[valid]
    slope = float(np.polyfit(t, v, 1)[0]) if len(v) > 1 else 0.0
    tail = x[-LAST:]
    last_mean = float(np.nanmean(tail)) if (~np.isnan(tail)).any() else float(v.mean())
    return np.array([v.mean(), v.min(), v.max(), v.std(), slope, last_mean, v[-1], 1 - valid.mean()])


def _matrix(df: pd.DataFrame) -> np.ndarray:
    return np.vstack([features(np.asarray(x, dtype=float)) for x in df["x"]])


def fit_predict_logreg_lastmap(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    col = 5  # mean of last 60 s
    model = LogisticRegression().fit(_matrix(train)[:, [col]], train["label"])
    return model.predict_proba(_matrix(test)[:, [col]])[:, 1]


def fit_predict_hgb(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    model = HistGradientBoostingClassifier(random_state=0).fit(_matrix(train), train["label"])
    return model.predict_proba(_matrix(test))[:, 1]


BASELINES = {
    "benchtoward/ioh-logreg-lastmap": fit_predict_logreg_lastmap,
    "benchtoward/ioh-hgb-summary": fit_predict_hgb,
}


def data_hash(instances: pd.DataFrame) -> str:
    """Fingerprint of the windows, splits and labels, so rebuilt data never reuses stale baseline runs."""
    import hashlib

    key = instances[["instance_id", "split", "label"]].sort_values("instance_id").to_csv(index=False)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def run_ioh_baselines(session, task, instances: pd.DataFrame, *, data_dir, n_boot: int = 1000) -> list:
    """Train each baseline on the train split, score the test split, and record one run per baseline."""
    from benchtoward.db import Model
    from benchtoward.evaluate import record_baseline_run
    from benchtoward.scoring import binary_metrics_ci

    train = instances[instances["split"] == "train"]
    test = instances[instances["split"] == "test"]
    fingerprint = data_hash(instances)
    runs = []
    for slug, fit in BASELINES.items():
        scores = fit(train, test)
        metrics = binary_metrics_ci(test["label"].to_numpy(), scores, test["subjectid"].to_numpy(), n_boot=n_boot)
        rows = [{"instance_id": iid, "score": float(s), "label": int(y)}
                for iid, s, y in zip(test["instance_id"], scores, test["label"])]
        config = {"data_hash": fingerprint, "train_split": "train", "eval_split": "test"}
        runs.append(record_baseline_run(session, session.get(Model, slug), task, rows, metrics, config,
                                        data_dir=data_dir))
    return runs
