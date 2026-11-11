"""lydus-ecg12-dx@0: spec, instance ids, labels, split file check and build on a miniature release (no network)."""
import json

import numpy as np
import pandas as pd
import pytest

from conftest import REGISTRY_DIR
from benchtoward.tasks import lydus
from benchtoward.tasks import lydus_ecg12_dx as task
from benchtoward.tasks import lydus_labels as L
from test_lydus import write_release
from test_lydus_split import ecg


def test_spec_is_a_hidden_task_scored_on_the_dataset_provided_labels():
    spec = task.load_spec()
    assert spec["slug"] == "lydus-ecg12-dx" and spec["version"] == 0 and "preview" not in spec
    # hidden until the maintainers publish it (the image tasks go out after it, so their export opens it too)
    assert spec["visibility"] in ("hidden", "open") and spec["dataset"] == lydus.DATASET_SLUG
    assert spec["kind"] == "multilabel" and spec["primary_metric"] == "macro_auroc"
    assert spec["metrics"]["primary"] == "macro_auroc" and spec["metrics"]["secondary"] == ["macro_f1"]
    assert spec["label"]["columns"] == list(L.COLUMNS)
    assert spec["label"]["targets"] == list(L.TARGETS) and spec["label"]["primary"] == list(L.PRIMARY)
    assert "map" not in spec["label"] and "challenge" not in spec["primary_metric"]
    assert spec["input"]["hz"] == 500 and spec["input"]["samples"] == 5000
    assert spec["input"]["lead_sets"] == {k: list(v) for k, v in lydus.LEAD_SETS.items()}
    assert spec["split"]["file"] == "registry/splits/lydus-snuh-ecg-50k-1.1.1-split.csv"
    assert spec["split"]["seed"] == 20261030 and spec["split"]["fractions"] == [0.7, 0.1, 0.2]
    assert spec["metrics"]["bootstrap"] == {"n": 1000, "by": "group", "seed": 0}
    assert spec["deviations"] and spec["leakage_checks"] and spec["owner_decisions"]


def test_instance_ids_are_opaque_and_stable():
    a = task.instance_id("000123")
    assert a.startswith("lydus12-") and len(a) == len("lydus12-") + 16 and "000123" not in a
    assert a == task.instance_id("000123") != task.instance_id("000124")


def test_class_names_are_the_provided_label_targets():
    assert task.class_names() == list(L.TARGETS)


def _release(tmp_path):
    ids = [f"{i:06d}" for i in (1, 2, 40, 41, 80, 120, 160, 200, 240, 280, 320, 360)]
    d, _ = write_release(tmp_path, ids=ids)
    meta = pd.read_csv(d / lydus.METADATA_FILE, dtype={"new_id": str})
    dx = ["Sinus rhythm\nwith 1st degree A-V block\nAbnormal ECG\n", "Atrial fibrillation\nAbnormal ECG\n",
          "Normal sinus rhythm\nPossible\nLeft atrial enlargement\n", "*** No QRS complexes found, no ECG analysis possible\n"]
    meta["dx"] = [dx[k % 4] for k in range(len(meta))]
    meta["sex"] = ["M", "M"] + ["F", "M"] * 5
    meta["age"] = [70, 70] + list(range(50, 60))
    labels = [("A.fib", "VPC", "NSP", "NSP"), ("NSR", "NSP", "RBBB", "1AVB"), ("Possible pacing", "NSP", "NSP", "NSP")]
    for k, column in enumerate(L.COLUMNS):
        meta[column] = [labels[r % 3][k] for r in range(len(meta))]
    meta.to_csv(d / lydus.METADATA_FILE, index=False)
    vals = np.stack([ecg(shape=1 + k, seed=k) for k in range(len(ids))])
    vals[5] = 0                                                # all leads flat
    np.savez(d / lydus.SIGNAL_FILE, keys=np.array(ids), vals=vals)
    return ids


def test_build_accepts_the_public_split_without_groups_and_reports_the_full_hash(tmp_path):
    from benchtoward.tasks import lydus_split
    _release(tmp_path)
    full_path = tmp_path / "split.csv"
    task.build_task(tmp_path, split_path=full_path, write_split=True, workers=1, check_map_counts=False)
    full_sha = json.loads((task.task_dir(tmp_path) / "build_summary.json").read_text())["split_sha256"]
    public_path = tmp_path / "public-split.csv"
    public_path.write_bytes(lydus_split.public_split_bytes(full_path.read_bytes()))
    paths = task.build_task(tmp_path, split_path=public_path, workers=1, check_map_counts=False)
    # The public copy has no group column; the build re-derives the groups and its full-table hash must match the spec.
    assert json.loads(paths["summary"].read_text())["split_sha256"] == full_sha
    wrong = pd.read_csv(public_path, dtype=str)
    wrong.loc[0, "split"] = "test" if wrong.loc[0, "split"] != "test" else "train"
    wrong.to_csv(public_path, index=False, lineterminator="\n")
    with pytest.raises(ValueError):
        task.build_task(tmp_path, split_path=public_path, workers=1, check_map_counts=False)


def test_build_writes_instances_and_checks_the_split_file(tmp_path, monkeypatch):
    ids = _release(tmp_path)
    split_path = tmp_path / "split.csv"
    with pytest.raises(FileNotFoundError):
        task.build_task(tmp_path, split_path=split_path, workers=1, check_map_counts=False)
    paths = task.build_task(tmp_path, split_path=split_path, write_split=True, workers=1, check_map_counts=False)
    inst = pd.read_parquet(paths["instances"])
    summary = json.loads(paths["summary"].read_text())
    split = pd.read_csv(split_path, dtype=str)
    assert split.columns.tolist() == ["new_id", "group", "split"] and split["new_id"].tolist() == ids
    assert split.loc[0, "group"] == split.loc[1, "group"] == "000001"   # adjacent ids, same sex and age
    # left out: the record without analysis (every 4th, k = 3, 7, 11) and the flat one (k = 5)
    assert summary["excluded"] == {"no_analysis": 3, "flat_signal": 1}
    assert len(inst) == 8 and inst["instance_id"].is_monotonic_increasing
    assert set(inst.columns) >= {"instance_id", "new_id", "split", "group", "age", "sex", "targets",
                                 "unknown_targets"}
    assert not {"labels", "hedged", "hedged_labels", "codes"} & set(inst.columns)
    by_id = inst.set_index("new_id")
    assert sorted(by_id.loc["000001", "targets"]) == ["premature beat:VPC", "rhythm:A.fib"]
    assert list(by_id.loc["000001", "unknown_targets"]) == []
    assert sorted(by_id.loc["000002", "targets"]) == ["avb:1AVB", "bbb:RBBB", "rhythm:NSR"]
    third = by_id.loc["000040"]                                  # Possible pacing: rhythm unknown, the rest known
    assert list(third["targets"]) == []
    assert sorted(third["unknown_targets"]) == sorted(t for t in L.TARGETS if t.startswith("rhythm:"))
    assert summary["audit"] == {}
    assert set(summary["positives"]["train"]) == set(L.TARGETS)
    assert set(summary["known"]["test"]) == set(L.TARGETS)
    assert summary["split_sha256"] and set(summary["records"]) == {"train", "validation", "test"}
    # a second build checks the file and refuses one that differs
    task.build_task(tmp_path, split_path=split_path, workers=1, check_map_counts=False)
    split.loc[0, "split"] = "test" if split.loc[0, "split"] != "test" else "train"
    split.to_csv(split_path, index=False)
    with pytest.raises(ValueError, match="split"):
        task.build_task(tmp_path, split_path=split_path, workers=1, check_map_counts=False)


def test_build_refuses_a_map_whose_counts_differ(tmp_path):
    _release(tmp_path)
    with pytest.raises(ValueError, match="n_records"):
        task.build_task(tmp_path, split_path=tmp_path / "split.csv", write_split=True, workers=1)


def test_label_matrices_and_inputs():
    inst = pd.DataFrame({"targets": [["rhythm:NSR"], ["rhythm:A.fib", "bbb:LBBB"], []],
                         "unknown_targets": [[], [], [t for t in L.TARGETS if t.startswith("rhythm:")]]})
    y, known = task.label_matrices(inst)
    assert y.shape == known.shape == (3, 25)
    assert y[0, L.TARGETS.index("rhythm:NSR")] and y[1, L.TARGETS.index("bbb:LBBB")] and not y[2].any()
    assert not known[2, L.TARGETS.index("rhythm:NSR")] and known[2, L.TARGETS.index("bbb:LBBB")]
    x8 = np.ones((8, 5000), np.int16)
    assert task.model_input(x8, "12").shape == (12, 5000) and task.model_input(x8, "2").shape == (2, 5000)


def test_count_mismatches_compares_records_and_positives():
    spec = {"counts": {"records": {"train": 2}, "positives": {"train": {"rhythm:NSR": 1}}}, "split": {"sha256": "x"}}
    summary = {"records": {"train": 2}, "positives": {"train": {"rhythm:NSR": 2}}, "split_sha256": "x"}
    assert task.count_mismatches(summary, spec) == ["positives train: built {'rhythm:NSR': 2}, spec {'rhythm:NSR': 1}"]


def test_count_mismatches_refuses_a_build_where_a_primary_target_is_not_eligible():
    spec = {"counts": {}, "split": {}}
    summary = {"records": {}, "positives": {}, "split_sha256": "x", "eligible_test": list(L.PRIMARY)}
    assert task.count_mismatches(summary, spec) == []
    summary["eligible_test"] = [t for t in L.PRIMARY if t != "bbb:LAFB"]
    assert task.count_mismatches(summary, spec) == ["primary targets below the test-split minimum: ['bbb:LAFB']"]
