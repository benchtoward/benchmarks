"""benchtoward/ptbxl-resnet1d: inputs, label file, pod commands, score loading and the SOP comparison (no network)."""
import json

import numpy as np
import pandas as pd
import pytest

from benchtoward.baselines import ptbxl_resnet as R
from benchtoward.tasks import ptbxl
from test_ptbxl import write_record


def test_resample_pad_converts_microvolts_to_millivolts_at_400_hz():
    x = np.full((2, 8, 5000), 1000, dtype=np.int16)          # 1,000 uV = 1 mV
    out = R.resample_pad_uv(x)
    assert out.shape == (2, 8, 4096) and out.dtype == np.float16
    assert out[:, :, :48].max() == 0 and out[:, :, -48:].max() == 0   # 4,000 samples centred in 4,096
    assert np.allclose(out[:, :, 200:3800].astype(float), 1.0, atol=1e-2)


def _labels(n=6):
    ecg_ids = np.arange(1, n + 1)
    split = np.array([0, 0, 0, 1, 2, 2][:n], dtype=np.int8)
    y = np.eye(5, dtype=bool)[np.arange(n) % 5]
    return ecg_ids, split, y


def test_labels_digest_binds_ids_split_and_labels():
    ecg_ids, split, y = _labels()
    d = R.labels_digest(ecg_ids, split, y)
    assert d == R.labels_digest(ecg_ids.copy(), split.copy(), y.copy())
    assert d != R.labels_digest(ecg_ids, np.roll(split, 1), y)
    assert d != R.labels_digest(ecg_ids, split, np.roll(y, 1, axis=0))
    assert d != R.labels_digest(ecg_ids[::-1], split, y)


def test_label_file_holds_no_text_but_filenames_and_matches_instances():
    inst = pd.DataFrame({"ecg_id": [3, 1], "split": ["test", "train"], "targets": [["MI"], ["CD", "NORM"]],
                         "patient_id": [7, 8]})
    data = np.load(__import__("io").BytesIO(R.labels_file(inst, filenames={1: "records500/00000/00001_hr",
                                                                           3: "records500/00000/00003_hr"})))
    assert list(data["ecg_id"]) == [3, 1] and list(data["split"]) == [2, 0]
    assert data["y"].tolist() == [[False, False, True, False, False], [True, False, False, True, False]]
    assert list(data["filename"]) == ["records500/00000/00003_hr", "records500/00000/00001_hr"]
    assert "patient_id" not in data.files
    assert str(data["labels_sha256"]) == R.labels_digest(data["ecg_id"], data["split"], data["y"])


def test_cache_reads_the_eight_independent_leads_in_label_order(tmp_path):
    raws = {}
    for k, name in enumerate(["00001_hr", "00002_hr"]):
        raw = np.zeros((12, 5000), dtype=np.int16)
        raw[:, :] = (np.arange(12) * 100 + k * 1000)[:, None]   # lead index and record visible in the value
        write_record(tmp_path / "records500" / "00000", name, raw)
        raws[name] = raw
    cache = R.build_cache(tmp_path, ["records500/00000/00002_hr", "records500/00000/00001_hr"], tmp_path / "x.npy")
    x = np.load(cache)
    assert x.shape == (2, 8, 4096)
    # first row is record 2, lead V1 (stored index 2 of the eight; header index 6): (6 * 100 + 1000) uV = 1.6 mV
    assert abs(float(x[0, 2, 2048]) - 1.6) < 0.02
    assert abs(float(x[1, 0, 2048]) - 0.0) < 0.02 and abs(float(x[1, 1, 2048]) - 0.1) < 0.02


def test_pod_scores_must_match_the_current_labels_and_splits(tmp_path):
    inst = pd.DataFrame({"ecg_id": [1, 2, 3, 4], "split": ["train", "validation", "test", "test"],
                         "targets": [["CD"], ["MI"], ["NORM"], ["HYP", "STTC"]]})
    ecg_ids, split, y = R.label_arrays(inst)
    digest = R.labels_digest(ecg_ids, split, y)
    np.savez(tmp_path / "scores-12.npz", val_rows=np.array([1]), val_scores=np.zeros((1, 5)),
             test_rows=np.array([2, 3]), test_scores=np.full((2, 5), 0.5), labels_sha256=digest)
    rows, scores = R.load_pod_scores(tmp_path, inst)
    assert rows.tolist() == [2, 3] and scores.shape == (2, 5)
    np.savez(tmp_path / "scores-12.npz", val_rows=np.array([1]), val_scores=np.zeros((1, 5)),
             test_rows=np.array([2, 3]), test_scores=np.full((2, 5), 0.5), labels_sha256="0" * 64)
    with pytest.raises(ValueError, match="other labels"):
        R.load_pod_scores(tmp_path, inst)
    np.savez(tmp_path / "scores-12.npz", val_rows=np.array([1]), val_scores=np.zeros((1, 5)),
             test_rows=np.array([1, 3]), test_scores=np.full((2, 5), 0.5), labels_sha256=digest)
    with pytest.raises(ValueError, match="test split"):
        R.load_pod_scores(tmp_path, inst)


def test_paired_difference_is_zero_for_identical_scores_and_positive_for_better_ones():
    rng = np.random.default_rng(1)
    y = rng.random((300, 5)) < 0.3
    groups = np.arange(300) // 2
    weak = rng.random((300, 5))
    strong = np.where(y, 0.7, 0.3) + rng.normal(0, 0.1, (300, 5))
    same = R.paired_difference(y, weak, weak, groups, n_boot=50)
    assert same["value"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0
    better = R.paired_difference(y, strong, weak, groups, n_boot=50)
    assert better["value"] > 0.3 and better["ci_low"] > 0 and better["share_at_or_below_zero"] == 0


def test_sop_verdict_follows_the_four_criteria():
    ok = R.sop_verdict(best={"value": 0.93, "ci_low": 0.92, "ci_high": 0.94}, prior=0.5,
                       weak={"value": 0.90, "ci_low": 0.89, "ci_high": 0.91},
                       strong={"value": 0.93, "ci_low": 0.92, "ci_high": 0.94}, format_failures=0.0)
    assert ok["verdict"] == "pass" and ok["headroom"]["H"] == pytest.approx(0.14)
    overlap = R.sop_verdict(best={"value": 0.93, "ci_low": 0.92, "ci_high": 0.94}, prior=0.5,
                            weak={"value": 0.924, "ci_low": 0.916, "ci_high": 0.931},
                            strong={"value": 0.93, "ci_low": 0.92, "ci_high": 0.94}, format_failures=0.0)
    assert overlap["separation"]["result"] == "fail" and overlap["verdict"] == "fail"
    warn = R.sop_verdict(best={"value": 0.96, "ci_low": 0.955, "ci_high": 0.965}, prior=0.5,
                         weak={"value": 0.90, "ci_low": 0.89, "ci_high": 0.91},
                         strong={"value": 0.96, "ci_low": 0.955, "ci_high": 0.965}, format_failures=0.0)
    assert warn["headroom"]["result"] == "warning"
    json.dumps(ok)


def test_training_runs_on_cpu_and_writes_scores(tmp_path):
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.1, (12, 8, 4096)).astype(np.float16)
    y = rng.random((12, 5)) < 0.4
    split = np.array([0] * 8 + [1] * 2 + [2] * 2)
    settings = dict(R.TRAINING, max_epochs=1, batch_size=4)
    R.train_cache(x, y, split, tmp_path, device="cpu", settings=settings, labels_sha256="d" * 64)
    data = np.load(tmp_path / "scores-12.npz")
    assert data["test_scores"].shape == (2, 5) and str(data["labels_sha256"]) == "d" * 64
