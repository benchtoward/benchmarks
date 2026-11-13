"""ptbxl-superdiagnostic@0: spec, build on a miniature release, patient-safe split, metric and the prior (no network)."""
import json

import numpy as np
import pandas as pd
import pytest

from benchtoward.tasks import ptbxl
from benchtoward.tasks import ptbxl_superdiagnostic as task
from test_ptbxl import STATEMENTS, write_record


def test_spec_is_an_open_adopted_protocol_on_ptbxl():
    spec = task.load_spec()
    # Published 2026-11-13 with the failed separation check stated in its note.
    assert spec["slug"] == "ptbxl-superdiagnostic" and spec["version"] == 0 and spec["visibility"] == "open"
    assert "failed on separation" in spec["publication_note"]
    assert spec["dataset"] == ptbxl.DATASET_SLUG and spec["kind"] == "multilabel"
    assert spec["primary_metric"] == "macro_auroc" and spec["metrics"]["primary"] == "macro_auroc"
    assert spec["label"]["classes"] == list(ptbxl.SUPERCLASSES)
    protocol = spec["protocol"]
    assert protocol["repository"] == "https://github.com/helme/ecg_ptbxl_benchmarking"
    assert len(protocol["commit"]) == 40 and protocol["experiment"] == "exp1.1.1 superdiagnostic"
    assert "GPL-3.0" in protocol["license"] and "not copied" in protocol["license"]
    assert spec["split"] == {"by": "patient_id", "folds": {"train": [1, 2, 3, 4, 5, 6, 7, 8], "validation": [9],
                                                           "test": [10]}, "source": "strat_fold"}
    assert spec["metrics"]["bootstrap"] == {"n": 1000, "by": "patient_id", "seed": 0}
    assert spec["input"]["hz"] == 500 and spec["input"]["samples"] == 5000
    assert {d.split(":")[0] for d in spec["deviations"]} >= {"Sampling rate", "Intervals"}
    assert spec["counts"]["records"] == {"train": 17084, "validation": 2146, "test": 2158}
    assert spec["counts"]["excluded"] == {"no_superdiagnostic_label": 411}
    assert spec["data"]["database_sha256"] == "7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b"


def test_dataset_entry_cites_the_release_the_paper_and_physionet():
    import yaml

    from conftest import REGISTRY_DIR

    entry = next(d for d in yaml.safe_load((REGISTRY_DIR / "datasets.yaml").read_text()) if d["slug"] == "ptb-xl")
    assert entry["license"] == "CC BY 4.0" and entry["version"] == "1.0.3" and entry.get("khdp_code") is None
    assert entry["source_url"] == "https://physionet.org/content/ptb-xl/1.0.3/"
    for part in ("10.13026/kfzx-aw45", "10.1038/s41597-020-0495-6", "10.1038/s44360-026-00096-z"):
        assert part in entry["citation"]


def test_instance_ids_are_opaque_and_stable():
    a = task.instance_id(17)
    assert a.startswith("ptbxl5-") and len(a) == len("ptbxl5-") + 16
    assert a == task.instance_id(17) != task.instance_id(18)


def _release(tmp_path):
    """Ten records of six patients over folds 1, 9 and 10; record 4 has no diagnostic statement."""
    root = ptbxl.release_dir(tmp_path)
    root.mkdir(parents=True)
    STATEMENTS.to_csv(root / ptbxl.STATEMENTS_FILE)
    codes = ["{'NORM': 100.0, 'SR': 0.0}", "{'IMI': 15.0}", "{'NDT': 100.0, 'CRBBB': 100.0}", "{'LVH': 50.0}",
             "{'SR': 0.0}", "{'NORM': 80.0}", "{'IMI': 100.0, 'LVH': 0.0}", "{'CRBBB': 100.0}", "{'NORM': 100.0}",
             "{'NDT': 50.0}"]
    patients = [11, 11, 12, 13, 13, 14, 15, 15, 16, 16]
    folds = [1, 1, 1, 9, 9, 9, 10, 10, 10, 10]
    rows = []
    rng = np.random.default_rng(0)
    for k in range(10):
        name = f"{k + 1:05d}_hr"
        write_record(root / "records500" / "00000", name, rng.integers(-500, 500, (12, 5000)).astype(np.int16))
        rows.append({"ecg_id": k + 1, "patient_id": float(patients[k]), "age": 40.0 + k, "sex": k % 2,
                     "scp_codes": codes[k], "strat_fold": folds[k], "filename_hr": f"records500/00000/{name}"})
    pd.DataFrame(rows).to_csv(root / ptbxl.DATABASE_FILE, index=False)
    return root


def test_build_follows_the_protocol_and_keeps_patients_in_one_split(tmp_path):
    _release(tmp_path)
    paths = task.build_task(tmp_path, check_counts=False)
    inst = pd.read_parquet(paths["instances"])
    summary = json.loads(paths["summary"].read_text())
    assert len(inst) == 9 and summary["excluded"] == {"no_superdiagnostic_label": 1}   # the protocol's selection
    assert summary["records"] == {"train": 3, "validation": 2, "test": 4}
    assert summary["patients"] == {"train": 2, "validation": 2, "test": 2} and summary["patients_in_two_splits"] == 0
    # record 7's "LVH: 0.0" still counts: the protocol takes every diagnostic code whatever its likelihood
    assert summary["positives"]["test"] == {"CD": 1, "HYP": 1, "MI": 1, "NORM": 1, "STTC": 1}
    row = inst.set_index("ecg_id").loc[3]
    assert list(row["targets"]) == ["CD", "STTC"] and row["split"] == "train"
    signals = np.load(paths["signals"], mmap_mode="r")
    assert signals.shape == (9, 12, 5000) and signals.dtype == np.int16
    y = task.label_matrix(inst)
    assert y.shape == (9, 5) and y.dtype == bool


def test_build_refuses_counts_that_differ_from_the_spec(tmp_path):
    _release(tmp_path)
    with pytest.raises(ValueError, match="differs from the spec"):
        task.build_task(tmp_path)          # the miniature release is not the real one


def test_macro_auroc_averages_the_classes_with_a_patient_bootstrap():
    y = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 1], [0, 0]], dtype=bool)
    scores = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.6], [0.6, 0.7], [0.3, 0.2]])
    groups = np.array([1, 1, 2, 2, 3, 3])
    m = task.metrics(y, scores, groups, classes=("A", "B"), n_boot=200)
    assert m["macro_auroc"]["value"] == pytest.approx(1.0)
    assert m["per_class"]["A"]["auroc"] == pytest.approx(1.0) and m["per_class"]["B"]["positives"] == 3
    assert m["macro_auroc"]["n"] == 6 and m["macro_auroc"]["n_groups"] == 3
    worse = scores.copy()
    worse[:, 0] = 1 - worse[:, 0]
    assert task.metrics(y, worse, groups, classes=("A", "B"), n_boot=0)["macro_auroc"]["value"] == pytest.approx(0.5)


def test_macro_auroc_matches_sklearn_average_macro():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(1)
    y = rng.random((300, 5)) < 0.3
    scores = rng.random((300, 5)) + y * 0.3
    m = task.metrics(y, scores, np.arange(300), n_boot=0)
    assert m["macro_auroc"]["value"] == pytest.approx(roc_auc_score(y, scores, average="macro"))


def test_prior_scores_train_prevalence():
    y_train = np.array([[1, 0, 0, 1, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 1], [1, 0, 0, 0, 0]], dtype=bool)
    scores = task.prior_scores(y_train, 3)
    assert scores.shape == (3, 5) and np.allclose(scores[0], [0.5, 0, 0.25, 0.75, 0.25])
