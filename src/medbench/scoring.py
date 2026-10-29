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
