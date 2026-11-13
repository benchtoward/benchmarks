"""ptbxl-superdiagnostic@0 baselines: beat features from the 500 Hz signals, per-class gradient boosting, run rows."""
import json

import numpy as np
import pandas as pd

from benchtoward.baselines import ptbxl_ecg as B
from benchtoward.baselines.lydus_ecg import FEATURE_NAMES
from benchtoward.tasks import ptbxl_superdiagnostic as task


def _beats(seed: int, rate: float = 1.0) -> np.ndarray:
    """(12, 5000) int16 in uV: narrow R waves every `rate` seconds on every lead, with noise."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 20, (12, 5000))
    for t in np.arange(0.3, 9.8, rate):
        i = int(t * 500)
        x[:, i - 5:i + 5] += 1000 * np.hanning(10)
    return x.astype(np.int16)


def test_features_read_the_eight_independent_leads(tmp_path):
    signals = np.stack([_beats(k, 0.6 + 0.1 * k) for k in range(3)])
    path = tmp_path / "s.npy"
    np.save(path, signals)
    X = B.features_for(path, np.array([2, 0]), workers=1)
    assert X.shape == (2, len(FEATURE_NAMES))
    rate = FEATURE_NAMES.index("rr:hr_bpm")
    assert X[0, rate] < X[1, rate]              # row 2 beats more slowly than row 0


def test_run_rows_carry_scores_labels_and_exact_match():
    inst = pd.DataFrame({"instance_id": ["a", "b"]})
    y = np.array([[1, 0, 0, 1, 0], [0, 0, 0, 0, 1]], dtype=bool)
    scores = np.array([[0.9, 0.1, 0.2, 0.7, 0.1], [0.6, 0.0, 0.0, 0.1, 0.9]])
    rows = B.run_rows(inst, y, scores)
    assert rows[0]["correct"] is True and rows[1]["correct"] is False     # CD scored 0.6 >= 0.5 for b
    assert json.loads(rows[0]["prediction"])["scores"]["NORM"] == 0.7
    assert json.loads(rows[1]["label"]) == {"CD": 0, "HYP": 0, "MI": 0, "NORM": 0, "STTC": 1}


def test_gradient_boosting_learns_a_separable_class():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4))
    y = np.zeros((400, 5), dtype=bool)
    y[:, 0] = X[:, 0] > 0
    y[:, 3] = X[:, 1] > 0.5
    y[:, 1] = rng.random(400) < 0.2
    y[:, 2] = rng.random(400) < 0.2
    y[:, 4] = rng.random(400) < 0.2
    model = B.fit(X[:300], y[:300])
    scores = model.predict_scores(X[300:])
    m = task.metrics(y[300:], scores, np.arange(100), n_boot=0)
    assert m["per_class"]["CD"]["auroc"] > 0.9 and m["per_class"]["NORM"]["auroc"] > 0.9
