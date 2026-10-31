from collections.abc import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _interval(samples: list[float]) -> tuple[float | None, float | None]:
    if not samples:
        return None, None
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def accuracy_ci(correct: Sequence[bool | None], n_boot: int = 1000, seed: int = 0) -> dict:
    """Accuracy with a 95% item-bootstrap interval. None (unparsed) counts as wrong."""
    hits = np.array([bool(c) for c in correct], dtype=float)
    n = len(hits)
    value = float(hits.mean()) if n else 0.0
    rng = np.random.default_rng(seed)
    boots = [float(hits[rng.integers(0, n, n)].mean()) for _ in range(n_boot)] if n else []
    low, high = _interval(boots)
    return {"value": value, "ci_low": low, "ci_high": high, "n": n, "n_unparsed": sum(c is None for c in correct)}


def edit_distance(ref: Sequence, hyp: Sequence) -> int:
    """Levenshtein distance (substitutions + deletions + insertions) between two sequences."""
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i]
        for j, h in enumerate(hyp, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1]


def error_rate_ci(errors: Sequence[float], lengths: Sequence[float], groups: Sequence, n_boot: int = 1000,
                  seed: int = 0) -> dict:
    """Pooled error rate sum(errors) / sum(lengths) (as CER and WER are defined), with a 95% interval from a cluster
    bootstrap: whole groups (e.g. all voices of one utterance) are resampled together."""
    e = np.asarray(errors, dtype=float)
    n = np.asarray(lengths, dtype=float)
    unique, inverse = np.unique(np.asarray(groups), return_inverse=True)
    e_group = np.bincount(inverse, weights=e, minlength=len(unique))
    n_group = np.bincount(inverse, weights=n, minlength=len(unique))
    value = float(e.sum() / n.sum()) if n.sum() else 0.0
    boots: list[float] = []
    if len(unique):
        picked = np.random.default_rng(seed).integers(0, len(unique), (n_boot, len(unique)))
        denominators = n_group[picked].sum(axis=1)
        keep = denominators > 0
        boots = list(e_group[picked].sum(axis=1)[keep] / denominators[keep])
    low, high = _interval(boots)
    return {"value": value, "ci_low": low, "ci_high": high, "n": int(len(e)), "n_groups": int(len(unique))}


BINARY_METRICS = {"auroc": roc_auc_score, "auprc": average_precision_score, "brier": brier_score_loss}


def _binary(y: np.ndarray, s: np.ndarray, metrics: Sequence[str] = tuple(BINARY_METRICS)) -> dict[str, float]:
    return {name: float(BINARY_METRICS[name](y, s)) for name in metrics}


def binary_metrics_ci(y_true: np.ndarray, y_score: np.ndarray, groups: np.ndarray,
                      n_boot: int = 1000, seed: int = 0, metrics: Sequence[str] = tuple(BINARY_METRICS)) -> dict:
    """AUROC, AUPRC and Brier (or the subset in `metrics`) with 95% intervals from a cluster bootstrap over groups.
    Leave Brier out when scores are not probabilities (e.g. a risk class)."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    g = np.asarray(groups)
    point = _binary(y, s, metrics)

    unique, inverse = np.unique(g, return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(len(unique))]
    rng = np.random.default_rng(seed)
    boots: dict[str, list[float]] = {name: [] for name in point}
    for _ in range(n_boot):
        picked = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[k] for k in picked])
        if y[idx].min() == y[idx].max():
            continue
        for name, value in _binary(y[idx], s[idx], metrics).items():
            boots[name].append(value)

    result: dict = {}
    for name, value in point.items():
        low, high = _interval(boots[name])
        result[name] = {"value": value, "ci_low": low, "ci_high": high}
    result.update({"n": int(len(y)), "n_groups": int(len(unique)), "prevalence": float(y.mean())})
    return result


def _threshold(y: np.ndarray, pred: np.ndarray) -> dict[str, float | None]:
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "accuracy": (tp + tn) / len(y) if len(y) else None,
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fn else None,
    }


def threshold_metrics_ci(y_true: np.ndarray, y_score: np.ndarray, groups: np.ndarray, threshold: float = 0.5,
                         n_boot: int = 1000, seed: int = 0) -> dict:
    """Accuracy, sensitivity, specificity and F1 of `score >= threshold`, with 95% intervals from a cluster
    bootstrap over groups. A metric that is undefined in a resample (e.g. no positives) skips that resample."""
    y = np.asarray(y_true, dtype=int)
    pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    unique, inverse = np.unique(np.asarray(groups), return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(len(unique))]
    point = _threshold(y, pred)
    rng = np.random.default_rng(seed)
    boots: dict[str, list[float]] = {name: [] for name in point}
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        for name, value in _threshold(y[idx], pred[idx]).items():
            if value is not None:
                boots[name].append(value)
    result: dict = {}
    for name, value in point.items():
        low, high = _interval(boots[name]) if value is not None else (None, None)
        result[name] = {"value": value, "ci_low": low, "ci_high": high}
    result["threshold"] = threshold
    return result


def _multiclass(conf: np.ndarray, classes: Sequence[str], scored: np.ndarray) -> dict:
    """Metrics from a confusion matrix whose rows are the reference classes and whose columns are the same classes
    plus one column for every other answer (a miss of the reference class, never a false positive). Macro-F1 is the
    mean F1 of the classes at the indices `scored`."""
    k = len(classes)
    tp = np.diag(conf[:, :k]).astype(float)
    ref = conf.sum(axis=1).astype(float)
    pred = conf[:, :k].sum(axis=0).astype(float)
    total = float(conf.sum())
    per_class = {}
    for i, name in enumerate(classes):
        fp, fn, negatives = pred[i] - tp[i], ref[i] - tp[i], total - ref[i]
        per_class[name] = {"sensitivity": tp[i] / ref[i] if ref[i] else None,
                           "ppv": tp[i] / pred[i] if pred[i] else None,
                           "f1": 2 * tp[i] / (2 * tp[i] + fp + fn) if ref[i] else None,
                           "fpr": fp / negatives if negatives else None}
    f1s = [per_class[classes[i]]["f1"] for i in scored]
    macro = float(np.mean(f1s)) if f1s and all(f is not None for f in f1s) else None
    return {"macro_f1": macro, "accuracy": tp.sum() / total if total else None, "by_class": per_class}


MAX_REDRAWS_PER_RESAMPLE = 100


def multiclass_metrics_ci(y_true: Sequence[str], y_pred: Sequence[str | None], groups: Sequence, classes: Sequence[str],
                          n_boot: int = 1000, seed: int = 0) -> dict:
    """Macro-F1 (primary), accuracy and per-class sensitivity, positive predictivity (ppv), F1 and false-positive
    rate, with 95% intervals from a cluster bootstrap over groups (e.g. patients).

    Macro-F1 is the mean F1 of the classes present in the full reference (`classes_scored`), as the PhysioNet/CinC
    2017 score averages per-class F1. Every resample keeps that fixed class set: a resample whose reference lacks
    one of these classes is drawn again (counted in `bootstrap.redraws`), so a resample never averages fewer
    classes than the point estimate; a RuntimeError is raised when one resample needs more than
    MAX_REDRAWS_PER_RESAMPLE redraws. A prediction outside `classes` (or None) is a miss of the reference class and
    no false positive for any class, as bxb counts AAMI F and Q answers."""
    classes = list(classes)
    y = np.asarray(y_true, dtype=object)
    unknown = set(y) - set(classes)
    if unknown:
        raise ValueError(f"reference labels outside the classes: {sorted(map(str, unknown))}")
    index = {name: i for i, name in enumerate(classes)}
    rows = np.array([index[v] for v in y], dtype=int)
    cols = np.array([index.get(v, len(classes)) for v in np.asarray(y_pred, dtype=object)], dtype=int)
    unique, inverse = np.unique(np.asarray(groups), return_inverse=True)
    k = len(classes)
    by_group = np.zeros((len(unique), k, k + 1), dtype=np.int64)
    np.add.at(by_group, (inverse, rows, cols), 1)
    conf = by_group.sum(axis=0)
    scored = np.flatnonzero(conf.sum(axis=1) > 0)
    point = _multiclass(conf, classes, scored)

    boots: dict[tuple, list[float]] = {}
    rng = np.random.default_rng(seed)
    accepted = redraws = tries = 0   # tries: redraws for the resample being drawn
    while len(unique) and accepted < n_boot:
        sample = by_group[rng.integers(0, len(unique), len(unique))].sum(axis=0)
        if (sample[scored].sum(axis=1) == 0).any():
            redraws += 1
            tries += 1
            if tries > MAX_REDRAWS_PER_RESAMPLE:
                raise RuntimeError(f"resample {accepted + 1} of {n_boot} needed more than {MAX_REDRAWS_PER_RESAMPLE} "
                                   "redraws: the reference classes are too rare across groups for this bootstrap")
            continue
        tries = 0
        accepted += 1
        m = _multiclass(sample, classes, scored)
        for key in ("macro_f1", "accuracy"):
            if m[key] is not None:
                boots.setdefault((key,), []).append(m[key])
        for name, values in m["by_class"].items():
            for key, value in values.items():
                if value is not None:
                    boots.setdefault((name, key), []).append(value)

    def entry(path: tuple, value) -> dict:
        low, high = _interval(boots.get(path, [])) if value is not None else (None, None)
        return {"value": None if value is None else float(value), "ci_low": low, "ci_high": high}

    result: dict = {"macro_f1": entry(("macro_f1",), point["macro_f1"]),
                    "accuracy": entry(("accuracy",), point["accuracy"]), "by_class": {}}
    for i, name in enumerate(classes):
        stats = point["by_class"][name]
        result["by_class"][name] = {"n": int(conf[i].sum()), **{key: entry((name, key), v) for key, v in stats.items()}}
    result["confusion"] = {"labels": classes, "predicted": [*classes, "other"], "matrix": conf.tolist()}
    result.update({"classes_scored": [classes[i] for i in scored], "n": int(len(y)), "n_groups": int(len(unique)),
                   "bootstrap": {"n": accepted, "by": "group", "redraws": redraws}})
    return result


def paired_rank_stats(stat_fn, n_items: int, groups: np.ndarray | None = None, higher_is_better: bool = True,
                      n_boot: int = 1000, seed: int = 0) -> list[dict]:
    """Rank statistics under a paired bootstrap.

    All models are scored on the same resampled items (or whole groups, e.g. patients), so shared item difficulty
    cancels out. `stat_fn(indices)` returns one metric value per model. For each model:
    - `range`: 2.5-97.5 percentile of its rank across resamples (1 = best, ties share the better rank);
    - `shared`: 1 + the number of models that beat it in at least 97.5% of resamples, so models that cannot be
      told apart share a rank (as on Arena).
    """
    rng = np.random.default_rng(seed)
    if groups is not None:
        unique, inverse = np.unique(np.asarray(groups), return_inverse=True)
        members = [np.flatnonzero(inverse == k) for k in range(len(unique))]
    keyed = []
    for _ in range(n_boot):
        if groups is None:
            idx = rng.integers(0, n_items, n_items)
        else:
            idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        values = np.asarray(stat_fn(idx), dtype=float)
        keyed.append(-values if higher_is_better else values)
    keyed = np.asarray(keyed)  # n_boot x n_models, lower is better
    ranks = 1 + (keyed[:, None, :] < keyed[:, :, None]).sum(axis=2)  # [b, i] = 1 + #models strictly better than i
    beats = (keyed[:, :, None] < keyed[:, None, :]).mean(axis=0)  # [j, i] = share of resamples where j beats i
    n_models = keyed.shape[1]
    return [{"range": (int(np.floor(np.percentile(ranks[:, m], 2.5))), int(np.ceil(np.percentile(ranks[:, m], 97.5)))),
             "shared": 1 + int(sum(beats[j, m] >= 0.975 for j in range(n_models) if j != m))}
            for m in range(n_models)]


def paired_rank_ranges(stat_fn, n_items: int, groups: np.ndarray | None = None, higher_is_better: bool = True,
                       n_boot: int = 1000, seed: int = 0) -> list[tuple[int, int]]:
    """95% range of each model's rank under a paired bootstrap (see paired_rank_stats)."""
    return [s["range"] for s in paired_rank_stats(stat_fn, n_items, groups, higher_is_better, n_boot, seed)]
