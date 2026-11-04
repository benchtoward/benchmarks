"""Baselines for vitaldb-arrhythmia-rhythm@0 and vitaldb-arrhythmia-beat@0, trained on the train split and computed
in one pass over the shared segments (each case's ECG is read once and serves both tasks).

- benchtoward/prior: the train split's most frequent class for every instance (ignores the ECG).
- benchtoward/arrhythmia-hgb-rr-morph: gradient boosting on RR-interval and beat morphology features.
  Rhythm: the binary baseline's RR features (rr-v1, BenchToward's QRS detector run on the segment) plus summaries of
  the detected beats' morphology; the annotated beat positions are not used, as the rhythm task requires.
  Beat: RR intervals from the given beat positions (previous, next, segment mean and their ratios) and the beat's
  morphology relative to the segment's median beat (width, amplitude, correlation, a coarse shape), in the spirit of
  de Chazal et al. 2004 but with the segment as the only context.

Both baselines answer the classes each task scores (`scored_classes`) and train on the instances of those classes;
beat instances already leave out cases whose annotations are offset from the ECG.
"""
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from benchtoward.baselines import arrhythmia as base

HZ = 500
FEATURE_VERSION = "rr-morph-v1"
SHAPE_MS = (-250, -200, -150, -100, -50, 0, 50, 100, 150, 200, 250, 300, 350, 400)
RR_FEATURES = ["pre_rr", "post_rr", "mean_rr", "pre_ratio", "post_ratio", "pre_post_ratio", "rr_cv", "n_beats"]
MORPH_FEATURES = ["amp_ratio", "signed_amp_ratio", "qrs_width_ms", "width_ratio", "template_corr", "template_rmse",
                  "p_region_sd", *[f"shape_{ms}" for ms in SHAPE_MS]]
BEAT_FEATURES = RR_FEATURES + MORPH_FEATURES
MORPH_SUMMARY = ["corr_min", "corr_mean", "low_corr_share", "width_mean_ms", "width_max_ratio", "amp_ratio_sd",
                 "rmse_max", "p_region_sd_mean", "premature_share"]
RHYTHM_FEATURES = base.FEATURES + MORPH_SUMMARY
POSITION_RHYTHM_FEATURES = ["n_beats", "hr_bpm", "rr_mean", "rr_sd", "rr_cv", "rr_min", "rr_max", "rr_range_ratio",
                            "rmssd", "irregularity", "pnn50", "premature_share"]
PREMATURE_RATIO = 0.85        # RR before the beat below 85% of the segment mean
LOW_CORR = 0.9
# Chosen by validation_report on the validation split, never on test (2026-10-30 build after the review; macro-F1
# without / with balanced class weights): rhythm 0.377 / 0.499, beat 0.922 / 0.907.
CLASS_WEIGHT = {"rhythm": "balanced", "beat": None}


def _quiet():
    """Silence numpy's all-NaN slice warnings: an all-missing window gives NaN on purpose."""
    return warnings.catch_warnings(action="ignore", category=RuntimeWarning)


@lru_cache(maxsize=4)
def _wide_band(hz: int):
    from scipy.signal import butter

    return butter(2, [0.5, 40], btype="band", fs=hz)


def _rr(samples: np.ndarray, hz: int) -> np.ndarray:
    t = np.asarray(samples, dtype=float) / hz
    n = len(t)
    out = np.full((n, len(RR_FEATURES)), np.nan)
    out[:, RR_FEATURES.index("n_beats")] = n
    if n < 2:
        return out
    rr = np.diff(t)
    pre, post = np.r_[np.nan, rr], np.r_[rr, np.nan]
    mean = rr.mean()
    out[:, 0], out[:, 1], out[:, 2] = pre, post, mean
    out[:, 3], out[:, 4] = pre / mean, post / mean
    with np.errstate(divide="ignore", invalid="ignore"):
        out[:, 5] = pre / post
    out[:, 6] = rr.std() / mean if len(rr) >= 2 else np.nan
    return out


def beat_features(x: np.ndarray, samples, hz: int = HZ) -> np.ndarray:
    """Feature matrix (order: BEAT_FEATURES) for the beats at `samples` (offsets in the segment); undefined
    features are NaN. RR features need only the positions; morphology features need the ECG."""
    from scipy.signal import filtfilt

    samples = np.asarray(samples, dtype=int)
    out = np.full((len(samples), len(BEAT_FEATURES)), np.nan)
    if len(samples) == 0:
        return out
    out[:, :len(RR_FEATURES)] = _rr(samples, hz)
    filled = base._fill(x)
    if filled is None or np.ptp(filled) < 1e-6:
        return out
    b, a = _wide_band(hz)
    y = filtfilt(b, a, filled - filled.mean())
    before, after = int(0.3 * hz), int(0.45 * hz)
    padded = np.r_[np.full(before, np.nan), y, np.full(after, np.nan)]
    windows = padded[samples[:, None] + np.arange(before + after)]           # y[p - 0.3 s : p + 0.45 s]
    near = windows[:, before - int(0.03 * hz):before + int(0.03 * hz) + 1]  # within 30 ms of the position
    with np.errstate(invalid="ignore", divide="ignore"), _quiet():
        peak = np.nanargmax(np.where(np.isfinite(near), np.abs(near), -1), axis=1)
        signed = near[np.arange(len(samples)), peak]
        amp = np.abs(signed)
        scale = np.nanmedian(amp)
        if not np.isfinite(scale) or scale <= 0:
            return out
        qrs = windows[:, before - int(0.1 * hz):before + int(0.1 * hz) + 1]
        slope = np.abs(np.diff(qrs, axis=1))
        top = np.nanmax(slope, axis=1, keepdims=True)
        width = np.where(np.isfinite(top[:, 0]), (slope > 0.2 * top).sum(axis=1) * 1000 / hz, np.nan)
        shape = windows[:, before - int(0.2 * hz):before + int(0.4 * hz)]      # -200 .. +400 ms
        template = np.nanmedian(shape, axis=0)
        corr, rmse = np.full(len(samples), np.nan), np.full(len(samples), np.nan)
        for i, w in enumerate(shape):
            ok = np.isfinite(w) & np.isfinite(template)
            if ok.mean() >= 0.5 and np.std(w[ok]) > 0 and np.std(template[ok]) > 0:
                corr[i] = np.corrcoef(w[ok], template[ok])[0, 1]
                rmse[i] = np.sqrt(np.mean((w[ok] - template[ok]) ** 2)) / scale
        p_region = windows[:, before - int(0.25 * hz):before - int(0.08 * hz)]
        p_sd = np.where(np.isfinite(p_region).mean(axis=1) >= 0.5, np.nanstd(p_region, axis=1), np.nan) / scale
        offsets = before + (np.asarray(SHAPE_MS) * hz // 1000)
        morph = np.column_stack([amp / scale, signed / scale, width, width / np.nanmedian(width), corr, rmse, p_sd,
                                 windows[:, offsets] / scale])
    out[:, len(RR_FEATURES):] = morph
    return out


def morphology_summary(beats: np.ndarray) -> np.ndarray:
    """Segment-level summaries (order: MORPH_SUMMARY) of a beat feature matrix."""
    if len(beats) == 0:
        return np.full(len(MORPH_SUMMARY), np.nan)
    col = {name: beats[:, BEAT_FEATURES.index(name)] for name in BEAT_FEATURES}
    corr = col["template_corr"]
    with _quiet():
        values = [np.nanmin(corr), np.nanmean(corr),
                  np.mean(corr[np.isfinite(corr)] < LOW_CORR) if np.isfinite(corr).any() else np.nan,
                  np.nanmean(col["qrs_width_ms"]), np.nanmax(col["width_ratio"]), np.nanstd(col["amp_ratio"]),
                  np.nanmax(col["template_rmse"]), np.nanmean(col["p_region_sd"]),
                  np.mean(col["pre_ratio"][np.isfinite(col["pre_ratio"])] < PREMATURE_RATIO)
                  if np.isfinite(col["pre_ratio"]).any() else np.nan]
    return np.asarray(values, dtype=float)


def position_rhythm_features(samples, hz: int = HZ) -> np.ndarray:
    """Segment-level RR features (order: POSITION_RHYTHM_FEATURES) from beat positions alone, without the ECG: what
    the beat task's given positions would tell a rhythm answer (a report-only leakage check)."""
    t = np.sort(np.asarray(samples, dtype=float)) / hz
    out = dict.fromkeys(POSITION_RHYTHM_FEATURES, np.nan)
    out["n_beats"] = len(t)
    if len(t) >= 2:
        rr = np.diff(t)
        mean = rr.mean()
        out.update(hr_bpm=60 / mean if mean > 0 else np.nan, rr_mean=mean, rr_sd=rr.std(), rr_min=rr.min(),
                   rr_max=rr.max(), premature_share=float(np.mean(rr < PREMATURE_RATIO * mean)))
        if mean > 0:
            out.update(rr_cv=rr.std() / mean, rr_range_ratio=(rr.max() - rr.min()) / mean)
        if len(rr) >= 2:
            d = np.diff(rr)
            out.update(rmssd=float(np.sqrt(np.mean(d ** 2))), pnn50=float(np.mean(np.abs(d) > 0.05)),
                       irregularity=float(np.mean(np.abs(d)) / mean) if mean > 0 else np.nan)
    return np.array([out[k] for k in POSITION_RHYTHM_FEATURES], dtype=float)


def position_rhythm_matrix(segments: pd.DataFrame, rhythm: pd.DataFrame, hz: int = HZ) -> np.ndarray:
    """position_rhythm_features for the rows of `rhythm`, from the shared segments' given beat positions."""
    samples = dict(zip(segments["segment_id"], segments["beat_samples"]))
    return np.vstack([position_rhythm_features(samples[i], hz) for i in rhythm["instance_id"]]) if len(rhythm) else \
        np.empty((0, len(POSITION_RHYTHM_FEATURES)))


def rhythm_features(x: np.ndarray, hz: int = HZ) -> np.ndarray:
    """Feature vector (order: RHYTHM_FEATURES) for one segment, from the ECG alone (detected R-peaks)."""
    peaks = base.detect_r_peaks(x, hz)
    return np.r_[base.rr_features(x, hz), morphology_summary(beat_features(x, peaks, hz))]


def one_pass_features(segments: pd.DataFrame, rhythm: pd.DataFrame, beat: pd.DataFrame, data_dir,
                      hz: int = HZ) -> dict[str, np.ndarray]:
    """Rhythm features (rows of `rhythm`) and beat features (rows of `beat`) from one read of each case's ECG.
    Beat features use every given position of the segment (unscored U and P beats included, as the input lists
    them); rhythm features use only the ECG."""
    from benchtoward.tasks.vitaldb_arrhythmia_rhythm_beat import segment_ecg

    rhythm_rows = {sid: i for i, sid in enumerate(rhythm["instance_id"])}
    beat_rows = {bid: i for i, bid in enumerate(beat["instance_id"])}
    out = {"rhythm": np.full((len(rhythm), len(RHYTHM_FEATURES)), np.nan, dtype=np.float32),
           "beat": np.full((len(beat), len(BEAT_FEATURES)), np.nan, dtype=np.float32)}
    for _, group in segments.groupby("caseid", sort=True):
        x = segment_ecg(group, data_dir)
        for k, s in enumerate(group.itertuples(index=False)):
            if s.segment_id in rhythm_rows:
                out["rhythm"][rhythm_rows[s.segment_id]] = rhythm_features(x[k], hz)
            samples = np.asarray(s.beat_samples, dtype=int)
            wanted = [(j, beat_rows.get(f"{s.segment_id}-{int(p)}")) for j, p in enumerate(samples)]
            wanted = [(j, r) for j, r in wanted if r is not None]
            if wanted:
                f = beat_features(x[k], samples, hz)
                for j, r in wanted:
                    out["beat"][r] = f[j]
    return out


def fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, classes, class_weight=None,
                ) -> np.ndarray:
    """Class probabilities (n, len(classes)) in the order of `classes`."""
    model = HistGradientBoostingClassifier(random_state=0, class_weight=class_weight).fit(train_x, train_y)
    proba = model.predict_proba(test_x)
    out = np.zeros((len(test_x), len(classes)))
    for j, c in enumerate(model.classes_):
        out[:, list(classes).index(c)] = proba[:, j]
    return out


EC57 = {"veb": "V", "sveb": "S"}


def score(task: str, test: pd.DataFrame, predicted, classes, n_boot: int = 1000) -> dict:
    """Task metrics: macro-F1 (primary), accuracy, per-class statistics; for beats also EC57's VEB and SVEB
    sensitivity, positive predictivity and false-positive rate, and SVEB sensitivity and positive predictivity on
    the beats outside atrial fibrillation segments (where the database labels every beat S)."""
    from benchtoward.scoring import multiclass_metrics_ci

    predicted = np.asarray(predicted)
    metrics = multiclass_metrics_ci(test["label"].to_numpy(), predicted, test["subjectid"].to_numpy(),
                                    classes=classes, n_boot=n_boot)
    if task == "beat":
        for prefix, cls in EC57.items():
            for name, key in (("sensitivity", "sensitivity"), ("ppv", "ppv"), ("fpr", "fpr")):
                metrics[f"{prefix}_{name}"] = metrics["by_class"][cls][key]
        outside = ~test["in_af_segment"].to_numpy(dtype=bool)
        sub = multiclass_metrics_ci(test["label"].to_numpy()[outside], predicted[outside],
                                    test["subjectid"].to_numpy()[outside], classes=classes, n_boot=n_boot)
        metrics["sveb_sensitivity_outside_af"] = sub["by_class"][EC57["sveb"]]["sensitivity"]
        metrics["sveb_ppv_outside_af"] = sub["by_class"][EC57["sveb"]]["ppv"]
        metrics["n_outside_af"] = sub["n"]
        metrics["bootstrap_outside_af"] = sub["bootstrap"]
    return metrics


def run_rhythm_beat_baselines(session, tasks: dict, rhythm: pd.DataFrame, beat: pd.DataFrame,
                              features: dict[str, np.ndarray], *, data_dir, n_boot: int = 1000) -> list:
    """Train each baseline on the train split, score the test split, and record one run per baseline and task.
    `tasks` maps the two task slugs to their Task rows."""
    from benchtoward.baselines.ioh import data_hash
    from benchtoward.db import Model
    from benchtoward.evaluate import record_baseline_run
    from benchtoward.tasks.vitaldb_arrhythmia_rhythm_beat import BEAT_SLUG, RHYTHM_SLUG, scored_classes

    runs = []
    for name, slug, table in (("rhythm", RHYTHM_SLUG, rhythm), ("beat", BEAT_SLUG, beat)):
        classes = scored_classes(slug)
        train, test = _split(table, "train", classes), (table["split"] == "test").to_numpy()
        y_train = table.loc[train, "label"].to_numpy()
        shares = pd.Series(y_train).value_counts(normalize=True)
        majority = shares.index[0]
        proba = fit_predict(features[name][train], y_train, features[name][test], classes, CLASS_WEIGHT[name])
        predictions = {
            "benchtoward/prior": (np.full(test.sum(), majority), np.full(test.sum(), shares.iloc[0]),
                               {"prior": f"train split's most frequent class ({majority})"}),
            "benchtoward/arrhythmia-hgb-rr-morph": (np.asarray(classes)[proba.argmax(axis=1)], proba.max(axis=1),
                                                 {"features": FEATURE_VERSION,
                                                  "model": "HistGradientBoostingClassifier", "random_state": 0,
                                                  "class_weight": CLASS_WEIGHT[name]}),
        }
        rows_test = table[test]
        fingerprint = data_hash(table)
        for model_slug, (labels, scores, extra) in predictions.items():
            metrics = score(name, rows_test, labels, classes, n_boot=n_boot)
            rows = [{"instance_id": iid, "prediction": str(p), "label": str(y), "correct": bool(p == y),
                     "score": float(s)} for iid, p, y, s in zip(rows_test["instance_id"], labels, rows_test["label"],
                                                                scores)]
            config = {"data_hash": fingerprint, "train_split": "train", "eval_split": "test",
                      "one_pass": "segments.parquet", **extra}
            runs.append(record_baseline_run(session, session.get(Model, model_slug), tasks[slug], rows, metrics,
                                            config, data_dir=data_dir))
    return runs


def validation_report(rhythm: pd.DataFrame, beat: pd.DataFrame, features: dict[str, np.ndarray]) -> dict:
    """Macro-F1 on the validation split, training on train with and without balanced class weights: the basis of
    CLASS_WEIGHT. Test labels are never read."""
    from benchtoward.scoring import multiclass_metrics_ci
    from benchtoward.tasks.vitaldb_arrhythmia_rhythm_beat import BEAT_SLUG, RHYTHM_SLUG, scored_classes

    report = {}
    for name, slug, table in (("rhythm", RHYTHM_SLUG, rhythm), ("beat", BEAT_SLUG, beat)):
        classes = scored_classes(slug)
        train, val = _split(table, "train", classes), _split(table, "validation", classes)
        entry = {}
        for key, weight in (("none", None), ("balanced", "balanced")):
            proba = fit_predict(features[name][train], table.loc[train, "label"].to_numpy(), features[name][val],
                                classes, weight)
            metrics = multiclass_metrics_ci(table.loc[val, "label"].to_numpy(), np.asarray(classes)[proba.argmax(1)],
                                            table.loc[val, "subjectid"].to_numpy(), classes=classes, n_boot=0)
            entry[key] = metrics["macro_f1"]["value"]
        report[name] = {**entry, "used": CLASS_WEIGHT[name]}
    return report


def _split(table: pd.DataFrame, part: str, classes) -> np.ndarray:
    """Training rows of one split, of the classes the task scores (the build already leaves out any other class).
    Test rows are never filtered: a test label outside the classes makes scoring fail."""
    return ((table["split"] == part) & table["label"].isin(list(classes))).to_numpy()


def _position_columns(table: pd.DataFrame, with_sample: bool) -> tuple[np.ndarray, list[str]]:
    first = table.groupby("caseid")["start_s"].transform("min")
    last = table.groupby("caseid")["start_s"].transform("max")
    cols = {"start_s": table["start_s"], "from_first_segment_s": table["start_s"] - first,
            "relative_position": ((table["start_s"] - first) / (last - first).replace(0, np.nan))}
    if with_sample:
        cols["sample_in_segment"] = table["sample"]
    return np.column_stack([np.asarray(v, dtype=float) for v in cols.values()]), list(cols)


def leakage_report(rhythm: pd.DataFrame, beat: pd.DataFrame, features: dict[str, np.ndarray],
                   segments: pd.DataFrame, n_boot: int = 1000) -> dict:
    """Report-only checks: what the segment's position in its case (and a beat's position in its segment) is worth
    alone; what the beat task's given positions are worth for the rhythm answer without the ECG (the reason a
    combined run must answer rhythm without them); and what they are worth for the beat answer without the
    waveform (RR features only)."""
    from benchtoward.tasks.vitaldb_arrhythmia_rhythm_beat import BEAT_SLUG, RHYTHM_SLUG, scored_classes

    rr = features["beat"][:, :len(RR_FEATURES)]
    checks = {"rhythm_position_only": ("rhythm", RHYTHM_SLUG, rhythm, *_position_columns(rhythm, False)),
              "rhythm_beat_positions_only": ("rhythm", RHYTHM_SLUG, rhythm, position_rhythm_matrix(segments, rhythm),
                                             POSITION_RHYTHM_FEATURES),
              "beat_position_only": ("beat", BEAT_SLUG, beat, *_position_columns(beat, True)),
              "beat_positions_rr_only": ("beat", BEAT_SLUG, beat, rr, RR_FEATURES)}
    report = {}
    for key, (name, slug, table, x, columns) in checks.items():
        classes = scored_classes(slug)
        train, test = _split(table, "train", classes), (table["split"] == "test").to_numpy()
        proba = fit_predict(x[train], table.loc[train, "label"].to_numpy(), x[test], classes, CLASS_WEIGHT[name])
        metrics = score(name, table[test], np.asarray(classes)[proba.argmax(axis=1)], classes, n_boot=n_boot)
        report[key] = {"features": columns, "macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"],
                       "bootstrap": metrics["bootstrap"]}
    return report
