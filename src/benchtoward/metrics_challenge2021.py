"""PhysioNet/Computing in Cardiology Challenge 2021 scoring (multi-label 12-lead ECG classification).

Ported from https://github.com/physionetchallenges/evaluation-2021 at commit e2a75fc01f729cb74cc4e853e054ce81e28381fc
(evaluate_model.py and helper_code.py), vendored with its licence, weights.csv and class tables in
third_party/evaluation-2021/. The functions load_table, load_weights, compute_accuracy, compute_confusion_matrices,
compute_f_measure, compute_auc, compute_modified_confusion_matrix and compute_challenge_metric are copied unchanged
apart from load_weights' default path; tests/test_metrics_challenge2021.py checks them against the vendored file,
also end to end on Challenge-format label and output files.

BenchToward adds vectorised forms of the same quantities that accept per-record counts, so a cluster bootstrap can
reweight records instead of copying them (the tests check that both agree), and `metrics_ci`, which reports the
Challenge score, macro AUROC and macro F-measure with 95% intervals.

Copyright (c) 2020, 2021 PhysioNet/Computing in Cardiology Challenges. All rights reserved. Redistribution and use in
source and binary forms, with or without modification, are permitted provided that the following conditions are
met: (1) Redistributions of source code must retain the above copyright notice, this list of conditions and the
following disclaimer. (2) Redistributions in binary form must reproduce the above copyright notice, this list of
conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np

COMMIT = "e2a75fc01f729cb74cc4e853e054ce81e28381fc"
REPOSITORY = "https://github.com/physionetchallenges/evaluation-2021"
VENDORED = Path(__file__).resolve().parents[2] / "third_party" / "evaluation-2021"
WEIGHTS_FILE = VENDORED / "weights.csv"
WEIGHTS_SHA256 = "668cb04555a97e8977488cb4bb6088d0d2bf9e59557945deedf29d5065c14d2b"
SINUS_RHYTHM = {"426783006"}


# ---- helper_code.py ---------------------------------------------------------------------------------------------

def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False


def is_finite_number(x):
    if is_number(x):
        return np.isfinite(float(x))
    else:
        return False


# ---- evaluate_model.py (unchanged apart from load_weights' default path) -----------------------------------------

# Load a table with row and column names.
def load_table(table_file):
    # The table should have the following form:
    #
    # ,    a,   b,   c
    # a, 1.2, 2.3, 3.4
    # b, 4.5, 5.6, 6.7
    # c, 7.8, 8.9, 9.0
    #
    table = list()
    with open(table_file, 'r') as f:
        for i, l in enumerate(f):
            arrs = [arr.strip() for arr in l.split(',')]
            table.append(arrs)

    # Define the numbers of rows and columns and check for errors.
    num_rows = len(table)-1
    if num_rows<1:
        raise Exception('The table {} is empty.'.format(table_file))
    row_lengths = set(len(table[i])-1 for i in range(num_rows))
    if len(row_lengths)!=1:
        raise Exception('The table {} has rows with different lengths.'.format(table_file))
    num_cols = min(row_lengths)
    if num_cols<1:
        raise Exception('The table {} is empty.'.format(table_file))

    # Find the row and column labels.
    rows = [table[0][j+1] for j in range(num_rows)]
    cols = [table[i+1][0] for i in range(num_cols)]

    # Find the entries of the table.
    values = np.zeros((num_rows, num_cols), dtype=np.float64)
    for i in range(num_rows):
        for j in range(num_cols):
            value = table[i+1][j+1]
            if is_finite_number(value):
                values[i, j] = float(value)
            else:
                values[i, j] = float('nan')

    return rows, cols, values

# Load weights.
def load_weights(weight_file=WEIGHTS_FILE):
    # Load the table with the weight matrix.
    rows, cols, values = load_table(weight_file)

    # Split the equivalent classes.
    rows = [set(row.split('|')) for row in rows]
    cols = [set(col.split('|')) for col in cols]
    assert(rows == cols)

    # Identify the classes and the weight matrix.
    classes = rows
    weights = values

    return classes, weights

# Compute recording-wise accuracy.
def compute_accuracy(labels, outputs):
    num_recordings, num_classes = np.shape(labels)

    num_correct_recordings = 0
    for i in range(num_recordings):
        if np.all(labels[i, :]==outputs[i, :]):
            num_correct_recordings += 1

    return float(num_correct_recordings) / float(num_recordings)

# Compute confusion matrices.
def compute_confusion_matrices(labels, outputs, normalize=False):
    # Compute a binary confusion matrix for each class k:
    #
    #     [TN_k FN_k]
    #     [FP_k TP_k]
    #
    # If the normalize variable is set to true, then normalize the contributions
    # to the confusion matrix by the number of labels per recording.
    num_recordings, num_classes = np.shape(labels)

    if not normalize:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            for j in range(num_classes):
                if labels[i, j]==1 and outputs[i, j]==1: # TP
                    A[j, 1, 1] += 1
                elif labels[i, j]==0 and outputs[i, j]==1: # FP
                    A[j, 1, 0] += 1
                elif labels[i, j]==1 and outputs[i, j]==0: # FN
                    A[j, 0, 1] += 1
                elif labels[i, j]==0 and outputs[i, j]==0: # TN
                    A[j, 0, 0] += 1
                else: # This condition should not happen.
                    raise ValueError('Error in computing the confusion matrix.')
    else:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            normalization = float(max(np.sum(labels[i, :]), 1))
            for j in range(num_classes):
                if labels[i, j]==1 and outputs[i, j]==1: # TP
                    A[j, 1, 1] += 1.0/normalization
                elif labels[i, j]==0 and outputs[i, j]==1: # FP
                    A[j, 1, 0] += 1.0/normalization
                elif labels[i, j]==1 and outputs[i, j]==0: # FN
                    A[j, 0, 1] += 1.0/normalization
                elif labels[i, j]==0 and outputs[i, j]==0: # TN
                    A[j, 0, 0] += 1.0/normalization
                else: # This condition should not happen.
                    raise ValueError('Error in computing the confusion matrix.')

    return A

# Compute macro F-measure.
def compute_f_measure(labels, outputs):
    num_recordings, num_classes = np.shape(labels)

    A = compute_confusion_matrices(labels, outputs)

    f_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
        if 2 * tp + fp + fn:
            f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            f_measure[k] = float('nan')

    if np.any(np.isfinite(f_measure)):
        macro_f_measure = np.nanmean(f_measure)
    else:
        macro_f_measure = float('nan')

    return macro_f_measure, f_measure

# Compute macro AUROC and macro AUPRC.
def compute_auc(labels, outputs):
    num_recordings, num_classes = np.shape(labels)

    # Compute and summarize the confusion matrices for each class across at distinct output values.
    auroc = np.zeros(num_classes)
    auprc = np.zeros(num_classes)

    for k in range(num_classes):
        # We only need to compute TPs, FPs, FNs, and TNs at distinct output values.
        thresholds = np.unique(outputs[:, k])
        thresholds = np.append(thresholds, thresholds[-1]+1)
        thresholds = thresholds[::-1]
        num_thresholds = len(thresholds)

        # Initialize the TPs, FPs, FNs, and TNs.
        tp = np.zeros(num_thresholds)
        fp = np.zeros(num_thresholds)
        fn = np.zeros(num_thresholds)
        tn = np.zeros(num_thresholds)
        fn[0] = np.sum(labels[:, k]==1)
        tn[0] = np.sum(labels[:, k]==0)

        # Find the indices that result in sorted output values.
        idx = np.argsort(outputs[:, k])[::-1]

        # Compute the TPs, FPs, FNs, and TNs for class k across thresholds.
        i = 0
        for j in range(1, num_thresholds):
            # Initialize TPs, FPs, FNs, and TNs using values at previous threshold.
            tp[j] = tp[j-1]
            fp[j] = fp[j-1]
            fn[j] = fn[j-1]
            tn[j] = tn[j-1]

            # Update the TPs, FPs, FNs, and TNs at i-th output value.
            while i < num_recordings and outputs[idx[i], k] >= thresholds[j]:
                if labels[idx[i], k]:
                    tp[j] += 1
                    fn[j] -= 1
                else:
                    fp[j] += 1
                    tn[j] -= 1
                i += 1

        # Summarize the TPs, FPs, FNs, and TNs for class k.
        tpr = np.zeros(num_thresholds)
        tnr = np.zeros(num_thresholds)
        ppv = np.zeros(num_thresholds)
        for j in range(num_thresholds):
            if tp[j] + fn[j]:
                tpr[j] = float(tp[j]) / float(tp[j] + fn[j])
            else:
                tpr[j] = float('nan')
            if fp[j] + tn[j]:
                tnr[j] = float(tn[j]) / float(fp[j] + tn[j])
            else:
                tnr[j] = float('nan')
            if tp[j] + fp[j]:
                ppv[j] = float(tp[j]) / float(tp[j] + fp[j])
            else:
                ppv[j] = float('nan')

        # Compute AUROC as the area under a piecewise linear function with TPR/
        # sensitivity (x-axis) and TNR/specificity (y-axis) and AUPRC as the area
        # under a piecewise constant with TPR/recall (x-axis) and PPV/precision
        # (y-axis) for class k.
        for j in range(num_thresholds-1):
            auroc[k] += 0.5 * (tpr[j+1] - tpr[j]) * (tnr[j+1] + tnr[j])
            auprc[k] += (tpr[j+1] - tpr[j]) * ppv[j+1]

    # Compute macro AUROC and macro AUPRC across classes.
    if np.any(np.isfinite(auroc)):
        macro_auroc = np.nanmean(auroc)
    else:
        macro_auroc = float('nan')
    if np.any(np.isfinite(auprc)):
        macro_auprc = np.nanmean(auprc)
    else:
        macro_auprc = float('nan')

    return macro_auroc, macro_auprc, auroc, auprc

# Compute a modified confusion matrix for multi-class, multi-label tasks.
def compute_modified_confusion_matrix(labels, outputs):
    # Compute a binary multi-class, multi-label confusion matrix, where the rows
    # are the labels and the columns are the outputs.
    num_recordings, num_classes = np.shape(labels)
    A = np.zeros((num_classes, num_classes))

    # Iterate over all of the recordings.
    for i in range(num_recordings):
        # Calculate the number of positive labels and/or outputs.
        normalization = float(max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1))
        # Iterate over all of the classes.
        for j in range(num_classes):
            # Assign full and/or partial credit for each positive class.
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        A[j, k] += 1.0/normalization

    return A

# Compute the evaluation metric for the Challenge.
def compute_challenge_metric(weights, labels, outputs, classes, sinus_rhythm):
    num_recordings, num_classes = np.shape(labels)
    if sinus_rhythm in classes:
        sinus_rhythm_index = classes.index(sinus_rhythm)
    else:
        raise ValueError('The sinus rhythm class is not available.')

    # Compute the observed score.
    A = compute_modified_confusion_matrix(labels, outputs)
    observed_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the correct label(s).
    correct_outputs = labels
    A = compute_modified_confusion_matrix(labels, correct_outputs)
    correct_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the sinus rhythm class.
    inactive_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool_)
    inactive_outputs[:, sinus_rhythm_index] = 1
    A = compute_modified_confusion_matrix(labels, inactive_outputs)
    inactive_score = np.nansum(weights * A)

    if correct_score != inactive_score:
        normalized_score = float(observed_score - inactive_score) / float(correct_score - inactive_score)
    else:
        normalized_score = 0.0

    return normalized_score


# ---- BenchToward additions ---------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _weights() -> tuple[tuple[frozenset, ...], np.ndarray]:
    classes, weights = load_weights()
    return tuple(frozenset(c) for c in classes), weights


def classes() -> list[set[str]]:
    """The 26 scored classes in weights.csv order; equivalent SNOMED CT codes share a class."""
    return [set(c) for c in _weights()[0]]


def class_names() -> list[str]:
    """'|'-joined codes of each class, as the vendored script names them in its class output."""
    return ["|".join(sorted(c)) for c in classes()]


@lru_cache(maxsize=1)
def abbreviations() -> list[str]:
    rows, _, _ = load_table(VENDORED / "weights_abbreviations.csv")
    return rows


def class_index(code: str) -> int | None:
    for k, c in enumerate(_weights()[0]):
        if code in c:
            return k
    return None


SINUS_INDEX = [k for k, c in enumerate(load_weights()[0]) if c == SINUS_RHYTHM][0]


def _counts(n: int, counts) -> np.ndarray:
    return np.ones(n) if counts is None else np.asarray(counts, dtype=float)


def challenge_score_fast(labels, outputs, counts=None) -> float:
    """compute_challenge_metric as matrix products; `counts[i]` copies of record i (bootstrap weights)."""
    _, weights = _weights()
    L = np.asarray(labels, dtype=float)
    O = np.asarray(outputs, dtype=float)
    w = _counts(len(L), counts)

    def score(out):
        norm = np.maximum(np.logical_or(L > 0, out > 0).sum(axis=1), 1)
        A = (L * (w / norm)[:, None]).T @ out
        return float(np.nansum(weights * A))

    inactive = np.zeros_like(L)
    inactive[:, SINUS_INDEX] = 1
    observed, correct, base = score(O), score(L), score(inactive)
    return (observed - base) / (correct - base) if correct != base else 0.0


def auroc_per_class_fast(labels, scores, counts=None) -> np.ndarray:
    """compute_auc's per-class AUROC (ties as one threshold, trapezoids); NaN for a class with no positives or no
    negatives."""
    L = np.asarray(labels, dtype=bool)
    S = np.asarray(scores, dtype=float)
    w = _counts(len(L), counts)
    out = np.full(L.shape[1], np.nan)
    for k in range(L.shape[1]):
        pos, neg = (w * L[:, k]).sum(), (w * ~L[:, k]).sum()
        if pos == 0 or neg == 0:
            continue
        values, inverse = np.unique(S[:, k], return_inverse=True)
        p = np.bincount(inverse, weights=w * L[:, k], minlength=len(values))[::-1]   # descending thresholds
        q = np.bincount(inverse, weights=w * ~L[:, k], minlength=len(values))[::-1]
        tpr = np.r_[0, np.cumsum(p)] / pos
        tnr = 1 - np.r_[0, np.cumsum(q)] / neg
        out[k] = float(np.sum(0.5 * np.diff(tpr) * (tnr[1:] + tnr[:-1])))
    return out


def f1_per_class_fast(labels, outputs, counts=None) -> np.ndarray:
    L = np.asarray(labels, dtype=bool)
    O = np.asarray(outputs, dtype=bool)
    w = _counts(len(L), counts)[:, None]
    tp, fp, fn = (w * (L & O)).sum(0), (w * (~L & O)).sum(0), (w * (L & ~O)).sum(0)
    denom = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), np.nan)


def _nanmean(x) -> float:
    return float(np.nanmean(x)) if np.any(np.isfinite(x)) else float("nan")


def evaluate(labels, outputs, scores) -> dict:
    """The vendored script's numbers for one set of outputs (point estimates, original loops)."""
    classes_, weights = load_weights()
    labels, outputs, scores = (np.asarray(labels, bool), np.asarray(outputs, bool), np.asarray(scores, float))
    macro_auroc, macro_auprc, auroc, auprc = compute_auc(labels, scores)
    macro_f1, f1 = compute_f_measure(labels, outputs)
    return {"challenge_score": compute_challenge_metric(weights, labels, outputs, classes_, SINUS_RHYTHM),
            "macro_auroc": float(macro_auroc), "macro_auprc": float(macro_auprc), "macro_f1": float(macro_f1),
            "accuracy": compute_accuracy(labels, outputs), "auroc": auroc, "auprc": auprc, "f1": f1}


def _interval(samples) -> tuple[float | None, float | None]:
    samples = [s for s in samples if np.isfinite(s)]
    if not samples:
        return None, None
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def metrics_ci(labels, outputs, scores, groups, n_boot: int = 1000, seed: int = 0) -> dict:
    """Challenge score (headline), macro AUROC, macro F-measure and accuracy as the vendored script computes them,
    with 95% intervals from a cluster bootstrap: whole groups are resampled and their records reweighted. Per-class
    AUROC, AUPRC and F-measure with the number of positives."""
    labels, outputs, scores = (np.asarray(labels, bool), np.asarray(outputs, bool), np.asarray(scores, float))
    point = evaluate(labels, outputs, scores)
    unique, inverse = np.unique(np.asarray(groups), return_inverse=True)
    rng = np.random.default_rng(seed)
    boots = {"challenge_score": [], "macro_auroc": [], "macro_f1": [], "accuracy": []}
    exact = np.all(labels == outputs, axis=1).astype(float)
    for _ in range(n_boot):
        picked = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        counts = picked[inverse].astype(float)
        boots["challenge_score"].append(challenge_score_fast(labels, outputs, counts))
        boots["macro_auroc"].append(_nanmean(auroc_per_class_fast(labels, scores, counts)))
        boots["macro_f1"].append(_nanmean(f1_per_class_fast(labels, outputs, counts)))
        boots["accuracy"].append(float((counts * exact).sum() / counts.sum()))
    out = {}
    for name, samples in boots.items():
        low, high = _interval(samples)
        out[name] = {"value": float(point[name]), "ci_low": low, "ci_high": high, "n": int(len(labels)),
                     "n_groups": int(len(unique)), "bootstrap": {"n": n_boot, "by": "group"}}
    out["macro_auprc"] = {"value": point["macro_auprc"]}
    names, abbrev = class_names(), abbreviations()
    out["per_class"] = [{"codes": names[k], "abbreviation": abbrev[k], "positives": int(labels[:, k].sum()),
                         "predicted": int(outputs[:, k].sum()),
                         "auroc": None if not np.isfinite(point["auroc"][k]) else float(point["auroc"][k]),
                         "auprc": None if not np.isfinite(point["auprc"][k]) else float(point["auprc"][k]),
                         "f1": None if not np.isfinite(point["f1"][k]) else float(point["f1"][k])}
                        for k in range(len(names))]
    return out
