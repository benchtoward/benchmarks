"""ptbxl-superdiagnostic@0: the five diagnostic superclasses of a 10-second 12-lead ECG (PTB-XL 1.0.3), macro AUROC.

The task is experiment exp1.1.1 ("superdiagnostic") of Strodthoff et al., Deep Learning for ECG Analysis:
Benchmarks and Insights from PTB-XL, IEEE JBHI 2021 (helme/ecg_ptbxl_benchmarking @ cdbf4e6, GPL-3.0), reimplemented
here from that code, not copied: labels from the release's scp_statements.csv aggregation (`ptbxl.superdiagnostic`),
records without a superclass left out, folds 1-8 train, 9 validation, 10 test, and macro AUROC over the five classes
(sklearn's roc_auc_score average="macro"). BenchToward's changes are the spec's deviations (500 Hz input, 95%
patient-grouped intervals).

Build (`benchtoward build ptbxl-superdiagnostic`): read the release, label each record, split by fold and write the
instances, the 500 Hz signals as one int16 array (records x 12 leads x 5000, 1 uV per unit) and build_summary.json.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from benchtoward.config import registry_dir
from benchtoward.tasks import ptbxl
from benchtoward.tasks.lydus_labels import auroc

SLUG = "ptbxl-superdiagnostic"
VERSION = 0
REF = f"{SLUG}@{VERSION}"
CLASSES = ptbxl.SUPERCLASSES
PARTS = ("train", "validation", "test")
N_BOOT, SEED = 1000, 0


def load_spec() -> dict:
    return yaml.safe_load((registry_dir() / "tasks" / f"{SLUG}.yaml").read_text(encoding="utf-8"))


def task_dir(data_dir: Path) -> Path:
    return ptbxl.dataset_dir(data_dir) / "superdiagnostic-v0"


def instance_id(ecg_id: int) -> str:
    return "ptbxl5-" + hashlib.sha256(f"{REF}|{int(ecg_id)}".encode()).hexdigest()[:16]


def label_matrix(inst: pd.DataFrame) -> np.ndarray:
    """(records, 5) booleans in CLASSES order from the instances' `targets`."""
    index = {c: k for k, c in enumerate(CLASSES)}
    y = np.zeros((len(inst), len(CLASSES)), dtype=bool)
    for r, targets in enumerate(inst["targets"]):
        y[r, [index[t] for t in targets]] = True
    return y


def build_task(data_dir: Path, check_counts: bool = True) -> dict[str, Path]:
    """Instances (instances.parquet), signals (signals-500hz.npy) and build_summary.json in task_dir(data_dir)."""
    root = ptbxl.release_dir(data_dir)
    db = ptbxl.read_database(root)
    statements = ptbxl.read_statements(root)
    labels = db["scp_codes"].apply(lambda codes: ptbxl.superdiagnostic(codes, statements))
    keep = labels.apply(len) > 0                               # the protocol's select_data: superdiagnostic_len > 0
    db, labels = db[keep], labels[keep]
    split = db["strat_fold"].map(ptbxl.split_of)
    inst = pd.DataFrame({
        "instance_id": [instance_id(i) for i in db.index],
        "ecg_id": db.index.to_numpy(int),
        "patient_id": db["patient_id"].to_numpy(int),
        "row": np.arange(len(db)),                             # position in signals-500hz.npy
        "split": split.to_numpy(object),
        "strat_fold": db["strat_fold"].to_numpy(int),
        "age": db["age"].to_numpy(float),
        "sex": db["sex"].to_numpy(object),
        "targets": labels.to_list(),
    })
    out = task_dir(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"instances": out / "instances.parquet", "signals": out / "signals-500hz.npy",
             "summary": out / "build_summary.json"}
    signals = np.lib.format.open_memmap(paths["signals"], mode="w+", dtype=np.int16,
                                        shape=(len(db), len(ptbxl.HEADER_LEADS), ptbxl.SAMPLES))
    for r, filename in enumerate(db["filename_hr"]):
        signals[r] = ptbxl.read_record(root, filename)
    signals.flush()
    del signals
    inst = inst.sort_values("instance_id", ignore_index=True)
    inst.to_parquet(paths["instances"], index=False)
    y = label_matrix(inst)
    masks = {p: (inst["split"] == p).to_numpy() for p in PARTS}
    patients = {p: set(inst.loc[m, "patient_id"]) for p, m in masks.items()}
    summary = {
        "release": {"physionet": f"ptb-xl/{ptbxl.VERSION}", "doi": ptbxl.DOI, "records": int(len(keep))},
        "excluded": {"no_superdiagnostic_label": int((~keep).sum())},
        "records": {p: int(m.sum()) for p, m in masks.items()},
        "patients": {p: len(s) for p, s in patients.items()},
        "patients_in_two_splits": len((patients["train"] & patients["validation"]) | (patients["train"] & patients["test"])
                                      | (patients["validation"] & patients["test"])),
        "positives": {p: {c: int(y[m, k].sum()) for k, c in enumerate(CLASSES)} for p, m in masks.items()},
        "database_sha256": ptbxl.sha256_file(root / ptbxl.DATABASE_FILE),
        "statements_sha256": ptbxl.sha256_file(root / ptbxl.STATEMENTS_FILE),
    }
    paths["summary"].write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    if check_counts:
        mismatches = count_mismatches(summary)
        if mismatches:
            raise ValueError("the build differs from the spec: " + "; ".join(mismatches))
    return paths


def count_mismatches(summary: dict, spec: dict | None = None) -> list[str]:
    spec = spec or load_spec()
    out = []
    counts = spec.get("counts") or {}
    for key in ("records", "patients", "positives", "excluded"):
        if key in counts and summary.get(key) != counts[key]:
            out.append(f"{key}: built {summary.get(key)}, spec {counts[key]}")
    data = spec.get("data") or {}
    for key, field in (("database_sha256", "database_sha256"), ("statements_sha256", "statements_sha256")):
        if data.get(field) and summary.get(key) != data[field]:
            out.append(f"{field}: built {summary.get(key)}, spec {data[field]}")
    if summary.get("patients_in_two_splits"):
        out.append(f"{summary['patients_in_two_splits']} patients in more than one split")
    return out


def metrics(y: np.ndarray, scores: np.ndarray, groups, classes=CLASSES, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """macro_auroc (each class's AUROC, ties half, averaged with equal weight: sklearn's average="macro") with a 95%
    interval from `n_boot` whole-patient bootstrap draws (seed 0); draws where a class has one label only are left
    out (their count is reported). Per-class AUROC with its interval."""
    y = np.asarray(y, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups)

    def point(rows):
        return [auroc(y[rows, k], scores[rows, k]) for k in range(len(classes))]

    per = point(np.arange(len(y)))
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    macro, per_draws, dropped = [], [[] for _ in classes], 0
    for _ in range(n_boot):
        rows = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        values = point(rows)
        for k, v in enumerate(values):
            if np.isfinite(v):
                per_draws[k].append(v)
        if all(np.isfinite(v) for v in values):
            macro.append(float(np.mean(values)))
        else:
            dropped += 1

    def ci(values):
        return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))) if values else (None, None)

    lo, hi = ci(macro)
    value = float(np.mean(per)) if all(np.isfinite(per)) else None
    return {
        "macro_auroc": {"value": value, "ci_low": lo, "ci_high": hi, "n": int(len(y)), "n_groups": int(len(uniq))},
        "per_class": {c: {"auroc": None if not np.isfinite(per[k]) else float(per[k]), "auroc_ci": list(ci(per_draws[k])),
                          "positives": int(y[:, k].sum()), "negatives": int((~y[:, k]).sum())}
                      for k, c in enumerate(classes)},
        "bootstrap": {"draws": n_boot, "seed": seed, "by": "patient_id", "macro_dropped": dropped},
    }


def prior_scores(y_train: np.ndarray, n: int) -> np.ndarray:
    """Train prevalence of each class as every record's score (input-blind; macro AUROC 0.5 by construction)."""
    return np.tile(np.asarray(y_train, dtype=bool).mean(axis=0), (n, 1))


def load_instances(data_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(task_dir(data_dir) / "instances.parquet")


def open_signals(data_dir: Path) -> np.ndarray:
    return np.load(task_dir(data_dir) / "signals-500hz.npy", mmap_mode="r")
