"""Baselines of inspire-aki@0, ported from the authors' code (Inspire-AKI-prediction @ 4dc799c, MIT).

- benchtoward/prior: the train prevalence for every operation (ignores the input).
- benchtoward/inspire-aki-asa: ASA class / 6, the authors' asa_rule score (no training).
- benchtoward/inspire-aki-gs-aki: the adapted GS-AKI count / 9, the authors' gs_aki_rule score (no training).
- benchtoward/inspire-aki-logreg and benchtoward/inspire-aki-xgb: the authors' ridge logistic regression and gradient
  boosted trees (models/tabular.py fit_tabular_model) on their pre-processing (features/normalization.py,
  features/missingness.py), with the hyperparameters their Optuna search (models/hpo.py) chose on this train split,
  and their isotonic calibration on patient-grouped inner-fold predictions (evaluation/calibration.py).

The authors fit outlier limits, scaling and imputation on every operation, test operations included; here they are
fitted on the train split and applied to both (a listed deviation). The report also trains every outer fold of the
authors' five-fold split once (five_fold), to check the paper's Table 2. xgboost is a dev dependency.
"""
import hashlib

import numpy as np
import pandas as pd

from benchtoward.tasks.inspire_aki import input_columns, load_spec

SEED = 42
N_FOLDS = 5
# features/normalization.py and missingness.py, with the authors' configs/aki/default.yaml values
OUTLIER_Q = {"lower_extreme": 0.01, "upper_extreme": 0.99, "lower_fill_low": 0.005, "lower_fill_high": 0.05,
             "upper_fill_low": 0.95, "upper_fill_high": 0.995}
HIGH_MISSING_PCT = 10.0
KNN_NEIGHBORS = 5
NOT_NORMALISED = {"age", "emop", "num_card_events", "antype", "sex", "asa"}  # base_ignore_cols, and department_*


def design(df: pd.DataFrame, columns: list[str], departments: list[str]) -> pd.DataFrame:
    """Model inputs as the authors' tabular dataset holds them: numbers as floats, department one-hot."""
    x = df[[c for c in columns if c != "department"]].astype(float).reset_index(drop=True)
    if "department" in columns:
        for d in departments:
            x[f"department_{d}"] = (df["department"].to_numpy() == d).astype(float)
    return x


def _outlier_plan(col: pd.Series) -> dict | None:
    q = {k: col.quantile(v) for k, v in OUTLIER_Q.items()}
    if pd.isna(q["lower_extreme"]) or pd.isna(q["upper_extreme"]):
        return None
    return q


def _apply_outliers(col: pd.Series, plan: dict | None, seed: int) -> pd.Series:
    out = col.copy()
    if plan is None:
        return out
    rng = np.random.default_rng(seed)
    low, high = out < plan["lower_extreme"], out > plan["upper_extreme"]
    if low.sum():
        out[low] = rng.uniform(plan["lower_fill_low"], plan["lower_fill_high"], size=int(low.sum()))
    if high.sum():
        out[high] = rng.uniform(plan["upper_fill_low"], plan["upper_fill_high"], size=int(high.sum()))
    return out


def preprocess(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The authors' build_tabular_datasets steps, fitted on train: outliers beyond the 1st and 99th percentiles are
    replaced by uniform draws from 0.5-5% and 95-99.5% (seed 42 + 104,729 per column), z-scores, then columns with at
    least 10% missing (in train) get -99 and those with less are filled by KNNImputer(5) over those columns."""
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import StandardScaler

    train, test = train.copy(), test.copy()
    numeric = [c for c in train.columns if c not in NOT_NORMALISED and not c.startswith("department")]
    for idx, col in enumerate(numeric):
        plan = _outlier_plan(train[col])
        train[col] = _apply_outliers(train[col], plan, SEED + idx * 104_729)
        test[col] = _apply_outliers(test[col], plan, SEED + idx * 104_729)
    if numeric:
        scaler = StandardScaler().fit(train[numeric])
        train[numeric] = scaler.transform(train[numeric])
        test[numeric] = scaler.transform(test[numeric])
    missing = train.isna().mean() * 100
    high = missing[missing >= HIGH_MISSING_PCT].index.tolist()
    low = missing[(missing > 0) & (missing < HIGH_MISSING_PCT)].index.tolist()
    train[high], test[high] = train[high].fillna(-99), test[high].fillna(-99)
    if low:
        imputer = KNNImputer(n_neighbors=KNN_NEIGHBORS).fit(train[low])
        train[low] = imputer.transform(train[low])
        test[low] = imputer.transform(test[low])
    test = test.fillna(-99)  # a column with no missing value in train but some in test
    return train, test


def balance_weights(y: np.ndarray) -> np.ndarray:
    """models/weighting.py: positives weighted negatives / positives (1:1 overall)."""
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    w = neg / pos if pos and neg else 1.0
    return np.where(y == 1, w, 1.0).astype(float)


def fit_predict_model(kind: str, params: dict, x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame,
                      seed: int, n_jobs: int = 8) -> np.ndarray:
    """fit_tabular_model and predict_tabular_bundle for log_reg and xgb: a StandardScaler fitted on train, balanced
    sample weights, the model's predict_proba."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x_train)
    xs_train, xs_test = scaler.transform(x_train), scaler.transform(x_test)
    weights = balance_weights(y_train)
    if kind == "log_reg":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(**params, max_iter=10000, tol=0.01, random_state=seed)
    elif kind == "xgb":
        from xgboost import XGBClassifier

        model = XGBClassifier(**params, objective="binary:logistic", eval_metric="logloss", random_state=seed,
                              n_jobs=n_jobs)
    else:
        raise ValueError(kind)
    model.fit(xs_train, y_train, sample_weight=weights)
    return model.predict_proba(xs_test)[:, 1]


def inner_folds(train: pd.DataFrame, label: str, n_splits: int = 5, seed: int = SEED) -> list:
    """split_manager._build_grouped_kfold_splitter on the outer-train operations (outer fold k: random_state 42 + k)."""
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(train[["op_id"]], train[label].astype(int), groups=train["subject_id"]))


def fit_predict_calibrated(kind: str, params: dict, train: pd.DataFrame, test: pd.DataFrame, label: str,
                           x_train: pd.DataFrame, x_test: pd.DataFrame, fold: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Raw and calibrated test scores of outer fold `fold` (0 is the headline): the model on all train rows (seed
    42 + fold), and isotonic regression fitted on out-of-fold predictions of five patient-grouped inner folds (splitter
    seed 42 + fold, model seeds 42 + 100 fold + inner fold), as the authors' pipelines/train.py seeds them."""
    from sklearn.isotonic import IsotonicRegression

    y = train[label].to_numpy().astype(int)
    raw = fit_predict_model(kind, params, x_train, y, x_test, SEED + fold)
    oof_y, oof_p = [], []
    for k, (tr, va) in enumerate(inner_folds(train.reset_index(drop=True), label, seed=SEED + fold)):
        oof_p.append(fit_predict_model(kind, params, x_train.iloc[tr], y[tr], x_train.iloc[va],
                                       SEED + 100 * fold + k))
        oof_y.append(y[va])
    iso = IsotonicRegression(out_of_bounds="clip").fit(np.concatenate(oof_p), np.concatenate(oof_y))
    return raw, iso.predict(raw)


def hpo_params(variant: str = "headline") -> dict:
    """The hyperparameters the authors' Optuna search chose on a train split (spec baselines_detail.hpo): headline
    (outer fold 0), fold_1 to fold_4 (the other outer folds, report) and one per report variant."""
    return load_spec()["baselines_detail"]["hpo"][variant]


def variant_columns(name: str = "headline") -> list[str]:
    """Model inputs: the paper's (headline), or without the two end-of-surgery durations (inputs.end_of_surgery)."""
    columns = input_columns()
    if name == "without_durations":
        drop = set(load_spec()["inputs"]["end_of_surgery"])
        return [c for c in columns if c not in drop]
    return columns


# Report variants: each changes the headline (paper's label, cohort and inputs) as listed; the earlier BenchToward
# headline changes the label (with its cohort) and the inputs. name -> (cohort column or None, label, inputs).
VARIANTS = {
    "crrt_7d_label": ("in_crrt_7d_cohort", "label_crrt_7d", "headline"),
    "without_durations": (None, "label", "without_durations"),
    "earlier_headline": ("in_crrt_7d_cohort", "label_crrt_7d", "without_durations"),
}


def variant_frame(instances: pd.DataFrame, name: str) -> tuple[pd.DataFrame, str, list[str]]:
    cohort, label, inputs = VARIANTS[name]
    frame = instances if cohort is None else instances[instances[cohort].astype(bool)]
    return frame.reset_index(drop=True), label, variant_columns(inputs)


def fold_frame(instances: pd.DataFrame, fold: int) -> pd.DataFrame:
    """Outer fold `fold` of the authors' split as test, the other four as train (fold 0 is the headline split)."""
    return instances.assign(split=np.where(instances["outer_fold"].to_numpy() == fold, "test", "train"))


def score_all(instances: pd.DataFrame, label: str = "label", columns: list[str] | None = None,
              hpo_variant: str = "headline", calibrate: bool = True, fold: int = 0) -> dict[str, np.ndarray]:
    """Test scores of every baseline, trained on the train split. Returns slug -> scores (test rows in order)."""
    columns = columns or input_columns()
    train = instances[instances["split"] == "train"].reset_index(drop=True)
    test = instances[instances["split"] == "test"].reset_index(drop=True)
    departments = sorted(train["department"].dropna().unique()) if "department" in columns else []
    x_train, x_test = preprocess(design(train, columns, departments), design(test, columns, departments))
    params = hpo_params(hpo_variant)
    scores = {
        "benchtoward/prior": np.full(len(test), float(train[label].mean())),
        "benchtoward/inspire-aki-asa": test["asa"].to_numpy(dtype=float) / 6.0,
        "benchtoward/inspire-aki-gs-aki": np.clip(test["gs_aki_count"].to_numpy(dtype=float) / 9.0, 0, 1),
    }
    for slug, kind in (("benchtoward/inspire-aki-logreg", "log_reg"), ("benchtoward/inspire-aki-xgb", "xgb")):
        raw, calibrated = fit_predict_calibrated(kind, params[kind], train, test, label, x_train, x_test, fold)
        scores[slug] = calibrated if calibrate else raw
        scores[slug + ":raw"] = raw
    return scores


def hpo_matrix(instances: pd.DataFrame, label: str, columns: list[str]) -> pd.DataFrame:
    """The pre-processed train matrix the authors' Optuna search reads (scripts/inspire_aki_reference.py hpo):
    op_id, patient_id, the model inputs (pre-processing fitted on train) and aki_boolean."""
    train = instances[instances["split"] == "train"].reset_index(drop=True)
    test = instances[instances["split"] == "test"].reset_index(drop=True)
    departments = sorted(train["department"].dropna().unique()) if "department" in columns else []
    x, _ = preprocess(design(train, columns, departments), design(test, columns, departments))
    x.insert(0, "op_id", train["op_id"].to_numpy())
    x.insert(1, "patient_id", train["subject_id"].to_numpy())
    x["aki_boolean"] = train[label].astype(bool).to_numpy()
    return x


def hpo_matrices(instances: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One train matrix per search of spec baselines_detail.hpo: the headline (outer fold 0), outer folds 1-4 and the
    report variants (their own cohort, label and inputs)."""
    out = {"headline": hpo_matrix(instances, "label", variant_columns())}
    for fold in range(1, N_FOLDS):
        out[f"fold_{fold}"] = hpo_matrix(fold_frame(instances, fold), "label", variant_columns())
    for name in VARIANTS:
        frame, label, columns = variant_frame(instances, name)
        out[name] = hpo_matrix(frame, label, columns)
    return out


def data_hash(instances: pd.DataFrame) -> str:
    columns = ["instance_id", "split", "label", *input_columns(), "gs_aki_count"]
    key = instances[columns].sort_values("instance_id").to_csv(index=False)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


RECORDED = ("benchtoward/prior", "benchtoward/inspire-aki-asa", "benchtoward/inspire-aki-gs-aki", "benchtoward/inspire-aki-logreg",
            "benchtoward/inspire-aki-xgb")


def run_inspire_aki_baselines(session, task, instances: pd.DataFrame, *, data_dir, n_boot: int = 1000) -> list:
    """Record one run per baseline on the test split (headline label and inputs)."""
    import sklearn
    import xgboost

    from benchtoward.db import Model
    from benchtoward.evaluate import record_baseline_run
    from benchtoward.scoring import binary_metrics_ci

    spec = load_spec()
    test = instances[instances["split"] == "test"].reset_index(drop=True)
    scores = score_all(instances)
    hpo = hpo_params("headline")
    model_params = {"benchtoward/prior": {"score": "train prevalence"}, "benchtoward/inspire-aki-asa": {"score": "asa / 6"},
                    "benchtoward/inspire-aki-gs-aki": {"score": "gs_aki_count / 9"},
                    "benchtoward/inspire-aki-logreg": {**hpo["log_reg"], "max_iter": 10000, "tol": 0.01, "seed": SEED,
                                                    "calibration": "isotonic, 5 grouped inner folds"},
                    "benchtoward/inspire-aki-xgb": {**hpo["xgb"], "seed": SEED, "calibration": "isotonic, 5 grouped inner folds"}}
    fingerprint = data_hash(instances)
    runs = []
    for slug in RECORDED:
        s = scores[slug]
        metrics = binary_metrics_ci(test["label"].to_numpy(), s, test["subject_id"].to_numpy(), n_boot=n_boot,
                                    metrics=[spec["metrics"]["primary"], *spec["metrics"]["secondary"]])
        rows = [{"instance_id": iid, "score": float(v), "label": int(y)}
                for iid, v, y in zip(test["instance_id"], s, test["label"])]
        config = {"data_hash": fingerprint, "train_split": "train", "eval_split": "test", "label": "label",
                  "params": model_params.get(slug, {}),
                  "versions": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__}}
        runs.append(record_baseline_run(session, session.get(Model, slug), task, rows, metrics, config,
                                        data_dir=data_dir))
    return runs


def _metrics(y, s, groups, n_boot) -> dict:
    from benchtoward.scoring import binary_metrics_ci

    m = binary_metrics_ci(np.asarray(y), np.asarray(s, dtype=float), np.asarray(groups), n_boot=n_boot,
                          metrics=["auroc", "auprc"])
    return {k: (v if not isinstance(v, dict) else {kk: round(float(vv), 4) for kk, vv in v.items()})
            for k, v in m.items()}


def _block(test: pd.DataFrame, label: str, scores: dict, n_boot: int) -> dict:
    return {"n": len(test), "positives": int(test[label].sum()),
            "models": {slug: _metrics(test[label], s, test["subject_id"], n_boot) for slug, s in scores.items()}}


def trapezoid_auprc(y: np.ndarray, s: np.ndarray) -> float:
    """The paper's Table 2 AUPRC (reporting/tables.py _safe_trapezoidal_auprc): the trapezoidal area under the
    precision-recall curve. BenchToward's auprc is average precision, as the authors' evaluation/metrics.py."""
    from sklearn.metrics import auc, precision_recall_curve

    precision, recall, _ = precision_recall_curve(y, s)
    return float(auc(recall, precision))


def five_fold(instances: pd.DataFrame) -> dict:
    """Every outer fold of the authors' split as test once (fold 0 is the headline), each with the hyperparameters
    the authors' search chose on that fold's train part and the authors' seeds. AUROC, AUPRC (average precision) and
    the trapezoidal AUPRC of the paper's table per fold, and their mean over the five folds with a t-based 95%
    interval, as the authors' reporting/tables.py makes the paper's Table 2 (calibrated scores for LR and GBT, the
    rule scores as they are)."""
    from scipy import stats
    from sklearn.metrics import average_precision_score, roc_auc_score

    folds = []
    for fold in range(N_FOLDS):
        frame = fold_frame(instances, fold)
        test = frame[frame["split"] == "test"].reset_index(drop=True)
        scores = score_all(frame, hpo_variant="headline" if fold == 0 else f"fold_{fold}", fold=fold)
        y = test["label"].to_numpy()
        folds.append({"fold": fold, "n": len(test), "positives": int(y.sum()),
                      "models": {slug: {"auroc": round(float(roc_auc_score(y, s)), 4),
                                        "auprc": round(float(average_precision_score(y, s)), 4),
                                        "auprc_trapezoid": round(trapezoid_auprc(y, s), 4)}
                                 for slug, s in scores.items()}})
    mean: dict = {}
    for slug in folds[0]["models"]:
        mean[slug] = {}
        for metric in ("auroc", "auprc", "auprc_trapezoid"):
            values = np.array([f["models"][slug][metric] for f in folds])
            sem = float(stats.sem(values))
            low, high = ((values.mean(), values.mean()) if sem == 0 else  # the prior: 0.5 in every fold
                         stats.t.interval(0.95, df=len(values) - 1, loc=values.mean(), scale=sem))
            mean[slug][metric] = {"value": round(float(values.mean()), 4), "ci_low": round(float(low), 4),
                                  "ci_high": round(float(high), 4)}
    return {"folds": folds, "mean": mean, "paper_table_2": load_spec()["protocol"]["paper_table_2"]}


def report(instances: pd.DataFrame, n_boot: int = 1000) -> dict:
    """Report-only numbers. The headline (all models, raw and calibrated scores); each earlier BenchToward choice as a
    row (spec report): the CRRT-in-7-days label, the inputs without the two durations, both (the earlier headline);
    pre-existing kidney failure: the headline models scored on the test operations without it and with it, and
    models retrained without it (headline hyperparameters); the five outer folds of the authors' split; single-column
    leak probes."""
    out: dict = {}
    test = instances[instances["split"] == "test"].reset_index(drop=True)
    headline = score_all(instances)
    out["headline"] = _block(test, "label", headline, n_boot)
    for name in VARIANTS:
        frame, label, columns = variant_frame(instances, name)
        out[name] = _block(frame[frame["split"] == "test"].reset_index(drop=True), label,
                           score_all(frame, label=label, columns=columns, hpo_variant=name), n_boot)
    failed = test["preexisting_kidney_failure"].astype(bool).to_numpy()
    for name, mask in (("without", ~failed), ("with", failed)):
        key = "headline_models__" + ("without_" if name == "without" else "") + "preexisting_kidney_failure"
        out[key] = _block(test[mask].reset_index(drop=True), "label", {slug: s[mask] for slug, s in headline.items()},
                          n_boot)
    kept = instances[~instances["preexisting_kidney_failure"].astype(bool)].reset_index(drop=True)
    out["retrained_without_preexisting_kidney_failure"] = _block(
        kept[kept["split"] == "test"].reset_index(drop=True), "label", score_all(kept), n_boot)
    out["five_fold"] = five_fold(instances)
    probes = {}
    for label, frame in (("label", instances), ("label_crrt_7d", instances[instances["in_crrt_7d_cohort"]])):
        part = frame[frame["split"] == "test"].reset_index(drop=True)
        for col in ("op_len", "booking_case_length", "crrt_ever", "crrt_postop", "preop_creatinine",
                    "preexisting_kidney_failure", "num_card_events_same_day"):
            probes[f"{label}:{col}"] = _metrics(part[label], part[col].astype(float).fillna(-1), part["subject_id"],
                                                n_boot)
    out["leak_probes"] = probes
    return out
