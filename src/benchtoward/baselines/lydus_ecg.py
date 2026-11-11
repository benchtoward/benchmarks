"""LYDUS provided-label baselines: known train prevalence and per-target gradient boosting.

Fit each target only on known train entries; fewer than 20 known positives use prevalence.
Thresholds use known validation labels. Metrics use the fixed PRIMARY macro AUROC and macro F1.
Reduced leads and measurement/demographic/order leakage checks remain report-only.
"""
import json

import numpy as np

from benchtoward.tasks.lydus_reporting import metrics as classification_metrics
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from benchtoward.tasks import lydus_labels as ll
from benchtoward.baselines import arrhythmia as rr
from benchtoward.tasks import lydus

FEATURE_VERSION = "lydus-beat-v1"
HZ = 500
BEFORE, AFTER, STEP = 100, 200, 5
TEMPLATE_MS = [int((k * STEP - BEFORE) * 1000 / HZ) for k in range((BEFORE + AFTER) // STEP)]
RR_NAMES = [f"rr:{n}" for n in rr.FEATURES]
TEMPLATE_NAMES = [f"{lead}:t{ms}" for lead in lydus.LEADS12 for ms in TEMPLATE_MS]
AMPLITUDE_NAMES = [f"{lead}:{stat}" for lead in lydus.LEADS12 for stat in ("sd", "p1", "p99")]
FEATURE_NAMES = RR_NAMES + TEMPLATE_NAMES + AMPLITUDE_NAMES
MIN_POSITIVES = 20
HGB = {"max_iter": 300, "learning_rate": 0.1, "random_state": 0}


def record_features(x8: np.ndarray) -> np.ndarray:
    """Feature vector (order: FEATURE_NAMES) of one record's 8 stored leads; undefined features are NaN."""
    from scipy.signal import butter, filtfilt

    out = np.full(len(FEATURE_NAMES), np.nan)
    x12 = lydus.to_12_leads(x8).astype(np.float64)
    out[:len(RR_NAMES)] = rr.rr_features(x12[1], HZ)
    if np.ptp(x12[1]) == 0 and np.ptp(x12[0]) == 0:
        return out
    b, a = butter(2, [0.5, 40], btype="band", fs=HZ)
    y = filtfilt(b, a, x12, axis=1)
    amp = np.c_[y.std(axis=1), np.percentile(y, 1, axis=1), np.percentile(y, 99, axis=1)]
    out[len(RR_NAMES) + len(TEMPLATE_NAMES):] = amp.ravel()
    peaks = rr.detect_r_peaks(x12[1], HZ)
    peaks = peaks[(peaks >= BEFORE) & (peaks < x12.shape[1] - AFTER)]
    if len(peaks) >= 2:
        windows = np.stack([y[:, p - BEFORE:p + AFTER] for p in peaks])
        out[len(RR_NAMES):len(RR_NAMES) + len(TEMPLATE_NAMES)] = np.median(windows, axis=0)[:, ::STEP].ravel()
    return out


def _chunk(args) -> np.ndarray:
    signals_file, rows = args
    values = lydus.open_signals_file(signals_file).values
    return np.stack([record_features(np.asarray(values[r])) for r in rows])


def features_for(signals_file, rows: np.ndarray, workers: int = 8, chunk: int = 500) -> np.ndarray:
    """Features of the release rows `rows` of the 500 Hz NPZ, computed in `workers` processes."""
    from multiprocessing import get_context

    rows = np.asarray(rows, dtype=int)
    jobs = [(str(signals_file), rows[s:s + chunk]) for s in range(0, len(rows), chunk)]
    if workers <= 1:
        return np.concatenate([_chunk(j) for j in jobs])
    with get_context("spawn").Pool(workers) as pool:
        return np.concatenate(pool.map(_chunk, jobs))


def feature_columns(lead_set: str) -> list[int]:
    leads = set(lydus.LEAD_SETS[lead_set])
    return [k for k, name in enumerate(FEATURE_NAMES) if name.startswith("rr:") or name.split(":")[0] in leads]


def with_demographics(X: np.ndarray, inst: pd.DataFrame) -> np.ndarray:
    sex = inst["sex"].map({"M": 1.0, "F": 0.0}).astype(float).to_numpy()
    return np.c_[X, pd.to_numeric(inst["age"], errors="coerce").to_numpy(float), sex]


def leakage_features(inst: pd.DataFrame, kind: str) -> np.ndarray:
    """measurements: the electrocardiograph's rates, intervals and axes with age and sex; demographics: age and
    sex; order: new_id (the record's index in the 160K release)."""
    demo = with_demographics(np.empty((len(inst), 0)), inst)
    if kind == "measurements":
        return np.c_[inst[list(lydus.MEASUREMENTS)].to_numpy(float), demo]
    if kind == "demographics":
        return demo
    if kind == "order":
        return inst["new_id"].astype(int).to_numpy(float)[:, None]
    raise ValueError(f"unknown leakage feature set {kind!r}")


def prevalence(y: np.ndarray, known: np.ndarray) -> np.ndarray:
    return np.divide((y & known).sum(0), known.sum(0), out=np.zeros(y.shape[1], float), where=known.sum(0) > 0)


def prior(train_labels: np.ndarray, known: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    p = prevalence(train_labels, known)
    # The F1-optimal constant answer predicts every target observed positive in train.
    return np.tile(p > 0, (n, 1)), np.tile(p, (n, 1))


class ClassModels:
    def __init__(self, models: list, prevalence: np.ndarray):
        self.models = models
        self.prevalence = prevalence
        self.fitted = np.array([m is not None for m in models])

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        out = np.tile(self.prevalence, (len(X), 1)).astype(float)
        for k, model in enumerate(self.models):
            if model is not None:
                out[:, k] = model.predict_proba(X)[:, 1]
        return out


def fit_classes(X: np.ndarray, y: np.ndarray, known: np.ndarray, min_positives: int = MIN_POSITIVES, params: dict = HGB) -> ClassModels:
    models = []
    for k in range(y.shape[1]):
        m = known[:, k]
        positives = int(y[m, k].sum())
        if positives < min_positives or positives == int(m.sum()):
            models.append(None)
            continue
        models.append(HistGradientBoostingClassifier(**params).fit(X[m], y[m, k]))
    return ClassModels(models, prevalence(y, known))


MIN_VALIDATION_POSITIVES = 10
FALLBACK_THRESHOLD = 0.5
THRESHOLD_RULE = (f"per class, the highest validation F-measure; {FALLBACK_THRESHOLD} for a class with fewer than "
                  f"{MIN_VALIDATION_POSITIVES} validation positives")


def f1_thresholds(y: np.ndarray, known: np.ndarray, scores: np.ndarray) -> np.ndarray:
    thresholds = ll.thresholds_from_validation(y, known, scores, targets=ll.TARGETS)
    return np.array([thresholds[t] for t in ll.TARGETS])


def run_rows(inst: pd.DataFrame, labels: np.ndarray, known: np.ndarray,
             outputs: np.ndarray, scores: np.ndarray) -> list[dict]:
    rows = []
    for i, iid in enumerate(inst["instance_id"]):
        rows.append({"instance_id": iid,
                     "prediction": json.dumps({"outputs": [ll.TARGETS[k] for k in np.flatnonzero(outputs[i])],
                                               "scores": dict(zip(ll.TARGETS, map(float, scores[i])))}),
                     "label": json.dumps({t: int(labels[i, k]) if known[i, k] else None
                                          for k, t in enumerate(ll.TARGETS)}),
                     "correct": bool(np.all((labels[i] == outputs[i]) | ~known[i])), "score": None})
    return rows


# ---- runs and report -----------------------------------------------------------------------------------------------

def _evaluate(y_test, known, scores, groups, n_boot, thresholds=None) -> dict:
    return classification_metrics(y_test, known, scores, groups, thresholds=thresholds, n_boot=n_boot, seed=0)


def _short(metrics: dict) -> dict:
    return {k: {kk: metrics[k][kk] for kk in ("value", "ci_low", "ci_high")}
            for k in ("macro_auroc", "macro_f1")}


def hgb_lead_set(X_all: np.ndarray, inst: pd.DataFrame, y: np.ndarray, known: np.ndarray, masks: dict, lead_set: str):
    """Fit on train, thresholds on validation; (outputs, scores, thresholds, fitted) on test."""
    X = with_demographics(X_all[:, feature_columns(lead_set)], inst)
    model = fit_classes(X[masks["train"]], y[masks["train"]], known[masks["train"]])
    thresholds = f1_thresholds(y[masks["validation"]], known[masks["validation"]], model.predict_scores(X[masks["validation"]]))
    scores = model.predict_scores(X[masks["test"]])
    return scores >= thresholds, scores, thresholds, model.fitted


def run_lydus_baselines(session, task, inst: pd.DataFrame, features: np.ndarray, *, data_dir, n_boot: int = 1000,
                        lead_sets=("12", "6", "4", "3", "2"), record: bool = True, report_boot: int = 200,
                        record_level_split: np.ndarray | None = None, leakage: bool = True) -> tuple[list, dict]:
    """Prior and gradient boosting on the 12-lead set as runs; reduced lead sets and leakage checks in the report."""
    from benchtoward.db import Model
    from benchtoward.evaluate import record_baseline_run
    from benchtoward.tasks.lydus_ecg12_dx import label_matrices

    y, known = label_matrices(inst)
    masks = {p: (inst["split"] == p).to_numpy() for p in ("train", "validation", "test")}
    test = inst[masks["test"]]
    groups = test["group"].to_numpy(object)
    runs, report = [], {"lead_sets": {}, "leakage": {}}

    outputs, scores = prior(y[masks["train"]], known[masks["train"]], int(masks["test"].sum()))
    metrics = _evaluate(y[masks["test"]], known[masks["test"]], scores, groups, n_boot,
                        {t: 0.0 if outputs[0, k] else 1.0 for k, t in enumerate(ll.TARGETS)}) | {"lead_set": "12"}
    report["prior"] = _short(metrics)
    if record:
        runs.append(record_baseline_run(session, session.get(Model, "benchtoward/prior"), task,
                                        run_rows(test, y[masks["test"]], known[masks["test"]], outputs, scores), metrics,
                                        {"baseline": "prior", "outputs": "every target observed positive in train",
                                         "scores": "train prevalence"}, data_dir=data_dir))
    for lead_set in lead_sets:
        outputs, scores, thresholds, fitted = hgb_lead_set(features, inst, y, known, masks, lead_set)
        boot = n_boot if lead_set == "12" else report_boot
        metrics = _evaluate(y[masks["test"]], known[masks["test"]], scores, groups, boot,
                            dict(zip(ll.TARGETS, map(float, thresholds)))) | {"lead_set": lead_set}
        report["lead_sets"][lead_set] = {"hgb": _short(metrics)}
        if lead_set == "12":
            report["hgb_12_per_class"] = metrics["per_target"]
            report["hgb_12_thresholds"] = {ll.TARGETS[k]: (None if not np.isfinite(t) else float(t))
                                           for k, t in enumerate(thresholds)}
            report["hgb_12_fitted_classes"] = [ll.TARGETS[k] for k in np.flatnonzero(fitted)]
            if record:
                config = {"baseline": "hgb-beat-features", "feature_version": FEATURE_VERSION, "lead_set": "12",
                          "min_positives": MIN_POSITIVES, "hgb": HGB, "thresholds": THRESHOLD_RULE}
                runs.append(record_baseline_run(session, session.get(Model, "benchtoward/lydus-hgb-beat-features"),
                                                task, run_rows(test, y[masks["test"]], known[masks["test"]], outputs, scores), metrics,
                                                config, data_dir=data_dir))
    for kind in ("measurements", "demographics", "order") if leakage else ():
        Xk = leakage_features(inst, kind)
        model = fit_classes(Xk[masks["train"]], y[masks["train"]], known[masks["train"]])
        thresholds = f1_thresholds(y[masks["validation"]], known[masks["validation"]], model.predict_scores(Xk[masks["validation"]]))
        scores = model.predict_scores(Xk[masks["test"]])
        report["leakage"][f"{kind}_only"] = _short(_evaluate(y[masks["test"]], known[masks["test"]], scores, groups,
                      report_boot, dict(zip(ll.TARGETS, map(float, thresholds)))))
    if record_level_split is not None:
        rmasks = {p: (record_level_split == p) for p in ("train", "validation", "test")}
        outputs, scores, thresholds, _ = hgb_lead_set(features, inst, y, known, rmasks, "12")
        rtest = inst[rmasks["test"]]
        report["leakage"]["record_level_split_hgb_12"] = _short(_evaluate(
            y[rmasks["test"]], known[rmasks["test"]], scores, rtest["group"].to_numpy(object), report_boot,
            dict(zip(ll.TARGETS, map(float, thresholds))))) | {
            "test_records": int(rmasks["test"].sum())}
    return runs, report


# ---- fitting on a CPU pod ------------------------------------------------------------------------------------------

CONFIGS = ("hgb-12", "hgb-6", "hgb-4", "hgb-3", "hgb-2", "measurements_only", "demographics_only", "order_only",
           "record_level_split_hgb-12")


def fit_config(name: str, X_all: np.ndarray, inst: pd.DataFrame, y: np.ndarray, known: np.ndarray) -> dict:
    """Test-split outputs and scores of one configuration (thresholds from validation F-measure). `inst` carries
    split, record_level_split, the measurements, age, sex and new_id."""
    split_col = "record_level_split" if name.startswith("record_level_split") else "split"
    masks = {p: (inst[split_col] == p).to_numpy() for p in ("train", "validation", "test")}
    if name.startswith("hgb-") or name.startswith("record_level_split"):
        lead_set = name.rsplit("-", 1)[1]
        X = with_demographics(X_all[:, feature_columns(lead_set)], inst)
    else:
        X = leakage_features(inst, name.removesuffix("_only"))
    model = fit_classes(X[masks["train"]], y[masks["train"]], known[masks["train"]])
    val_scores = model.predict_scores(X[masks["validation"]])
    thresholds = f1_thresholds(y[masks["validation"]], known[masks["validation"]], val_scores)
    scores = model.predict_scores(X[masks["test"]])
    return {"test_idx": np.flatnonzero(masks["test"]), "outputs": scores >= thresholds, "scores": scores,
            "val_idx": np.flatnonzero(masks["validation"]), "val_scores": val_scores, "thresholds": thresholds,
            "fitted": model.fitted, "labels_sha256": labels_digest(inst, y, known)}


def labels_digest(inst, y, known) -> str:
    from benchtoward.baselines.lydus_resnet import labels_digest as digest
    return digest(y, known, inst["split"].to_numpy(), inst["row"].to_numpy())


def main(argv=None) -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Fit the gradient-boosting configurations of lydus-ecg12-dx@0")
    parser.add_argument("--signals", required=True)
    parser.add_argument("--instances", required=True, help="parquet: row, split, record_level_split, targets, unknown_targets, ...")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--configs", default=",".join(CONFIGS))
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inst = pd.read_parquet(args.instances)
    y, known = ll.matrices(inst["targets"], inst["unknown_targets"])
    X = features_for(args.signals, inst["row"].to_numpy(), workers=args.workers)
    np.save(out / "features.npy", X)
    for name in args.configs.split(","):
        result = fit_config(name, X, inst, y, known)
        np.savez(out / f"preds-{name}.npz", **result)
        print(name, "done", flush=True)


if __name__ == "__main__":
    main()
