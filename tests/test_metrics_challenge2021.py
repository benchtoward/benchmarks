"""PhysioNet/CinC Challenge 2021 scoring: the port in benchtoward.metrics_challenge2021 against the vendored original
(third_party/evaluation-2021 at e2a75fc), and the vectorised forms used by the bootstrap."""
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from benchtoward import metrics_challenge2021 as cm

VENDORED = Path(__file__).resolve().parents[1] / "third_party" / "evaluation-2021"


def _original():
    """The vendored evaluate_model.py, imported unchanged (it imports helper_code from its own folder)."""
    sys.path.insert(0, str(VENDORED))
    try:
        spec = importlib.util.spec_from_file_location("challenge2021_original", VENDORED / "evaluate_model.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(VENDORED))
    return module


def _random(n=300, seed=0):
    rng = np.random.default_rng(seed)
    classes, _ = cm.load_weights()
    k = len(classes)
    labels = rng.random((n, k)) < 0.12
    labels[rng.random(n) < 0.3, cm.SINUS_INDEX] = True
    scores = np.round(np.clip(labels * 0.4 + rng.random((n, k)) * 0.7, 0, 1), 2)   # ties on purpose
    outputs = scores >= 0.5
    return labels, outputs, scores


def test_vendored_files_are_the_pinned_ones():
    assert cm.COMMIT == "e2a75fc01f729cb74cc4e853e054ce81e28381fc"
    digest = hashlib.sha256((VENDORED / "weights.csv").read_bytes()).hexdigest()
    assert digest == cm.WEIGHTS_SHA256
    assert "BSD 2-Clause" in (VENDORED / "LICENSE").read_text()


def test_weights_have_26_classes_with_equivalences_and_sinus_rhythm():
    classes, weights = cm.load_weights()
    assert len(classes) == 26 and weights.shape == (26, 26)
    assert {"733534002", "164909002"} in classes and {"713427006", "59118001"} in classes
    assert {"284470004", "63593006"} in classes and {"427172004", "17338001"} in classes
    assert classes[cm.SINUS_INDEX] == {"426783006"}
    assert np.allclose(np.diag(weights), 1.0)


def test_class_index_maps_every_equivalent_code():
    assert cm.class_index("59118001") == cm.class_index("713427006")
    assert cm.class_index("63593006") == cm.class_index("284470004")
    assert cm.class_index("164873001") is None          # LVH is not scored


def test_perfect_outputs_score_one_and_sinus_only_scores_zero():
    labels, _, _ = _random()
    classes, weights = cm.load_weights()
    assert cm.compute_challenge_metric(weights, labels, labels.copy(), classes, {"426783006"}) == pytest.approx(1.0)
    sinus = np.zeros_like(labels)
    sinus[:, cm.SINUS_INDEX] = True
    assert cm.compute_challenge_metric(weights, labels, sinus, classes, {"426783006"}) == pytest.approx(0.0)


def test_port_matches_the_vendored_original_function_by_function():
    original = _original()
    labels, outputs, scores = _random()
    classes, weights = cm.load_weights()
    assert cm.compute_challenge_metric(weights, labels, outputs, classes, {"426783006"}) == pytest.approx(
        original.compute_challenge_metric(weights, labels, outputs, classes, {"426783006"}), abs=1e-12)
    ours, theirs = cm.compute_auc(labels, scores), original.compute_auc(labels, scores)
    for a, b in zip(ours, theirs):
        assert np.allclose(a, b, equal_nan=True)
    assert cm.compute_f_measure(labels, outputs)[0] == pytest.approx(original.compute_f_measure(labels, outputs)[0])
    assert cm.compute_accuracy(labels, outputs) == original.compute_accuracy(labels, outputs)


def test_end_to_end_against_the_vendored_script_on_challenge_files(tmp_path, monkeypatch):
    """Write Challenge label (.hea) and output (.csv) files and run the original evaluate_model() on them."""
    original = _original()
    labels, outputs, scores = _random(n=60, seed=3)
    classes, _ = cm.load_weights()
    names = ["|".join(sorted(c)) for c in classes]
    (tmp_path / "labels").mkdir()
    (tmp_path / "outputs").mkdir()
    for i in range(len(labels)):
        dx = ",".join(sorted(next(iter(sorted(classes[j]))) for j in np.flatnonzero(labels[i]))) or "426783006"
        if not labels[i].any():
            labels[i, cm.SINUS_INDEX] = True
        (tmp_path / "labels" / f"r{i:03d}.hea").write_text(f"r{i:03d} 12 500 5000\n#Dx: {dx}\n")
        (tmp_path / "outputs" / f"r{i:03d}.csv").write_text(
            f"#r{i:03d}\n{','.join(names)}\n{','.join(str(int(v)) for v in outputs[i])}\n"
            f"{','.join(f'{v:.2f}' for v in scores[i])}\n")
    monkeypatch.chdir(VENDORED)   # the script reads weights.csv from the working directory
    result = original.evaluate_model(str(tmp_path / "labels"), str(tmp_path / "outputs"))
    ours = cm.evaluate(labels, outputs, scores)
    assert ours["challenge_score"] == pytest.approx(result[8], abs=1e-12)
    assert ours["macro_auroc"] == pytest.approx(result[1], abs=1e-12)
    assert ours["macro_f1"] == pytest.approx(result[6], abs=1e-12)


def test_vectorised_forms_equal_the_loops():
    labels, outputs, scores = _random(seed=5)
    classes, weights = cm.load_weights()
    reference = cm.compute_challenge_metric(weights, labels, outputs, classes, {"426783006"})
    assert cm.challenge_score_fast(labels, outputs) == pytest.approx(reference, abs=1e-12)
    auroc = cm.compute_auc(labels, scores)[2]
    assert np.allclose(cm.auroc_per_class_fast(labels, scores), auroc, equal_nan=True)
    assert np.allclose(cm.f1_per_class_fast(labels, outputs), cm.compute_f_measure(labels, outputs)[1],
                       equal_nan=True)


def test_counts_weighting_equals_repeating_rows():
    labels, outputs, scores = _random(n=80, seed=7)
    rng = np.random.default_rng(1)
    counts = rng.integers(0, 4, len(labels))
    idx = np.repeat(np.arange(len(labels)), counts)
    classes, weights = cm.load_weights()
    expected = cm.compute_challenge_metric(weights, labels[idx], outputs[idx], classes, {"426783006"})
    assert cm.challenge_score_fast(labels, outputs, counts) == pytest.approx(expected, abs=1e-12)
    assert np.allclose(cm.auroc_per_class_fast(labels, scores, counts), cm.compute_auc(labels[idx], scores[idx])[2],
                       equal_nan=True)
    assert np.allclose(cm.f1_per_class_fast(labels, outputs, counts),
                       cm.compute_f_measure(labels[idx], outputs[idx])[1], equal_nan=True)


def test_metrics_ci_bootstraps_groups_and_reports_classes():
    labels, outputs, scores = _random(n=200, seed=11)
    groups = np.arange(200) // 2
    out = cm.metrics_ci(labels, outputs, scores, groups, n_boot=50, seed=0)
    assert set(out) >= {"challenge_score", "macro_auroc", "macro_f1", "accuracy", "per_class"}
    for name in ("challenge_score", "macro_auroc", "macro_f1"):
        assert out[name]["ci_low"] <= out[name]["value"] <= out[name]["ci_high"]
        assert out[name]["n"] == 200 and out[name]["n_groups"] == 100
    assert out["challenge_score"]["bootstrap"] == {"n": 50, "by": "group"}
    abbreviations = [c["abbreviation"] for c in out["per_class"]]
    assert "NSR" in abbreviations and "CRBBB|RBBB" in abbreviations and len(abbreviations) == 26
    first = out["per_class"][0]
    assert set(first) >= {"codes", "abbreviation", "positives", "auroc", "f1"}
