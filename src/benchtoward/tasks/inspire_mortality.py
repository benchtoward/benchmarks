"""INSPIRE 30-day postoperative in-hospital mortality, version 0 (preview).

Cohort, label, inputs and split follow the INSPIRE authors' published script (vitaldb/inspire gbm_mortality.py,
pinned in registry/tasks/inspire-mortality.yaml), with the deviations listed there (notably: each patient's earliest
operation, not the smallest op_id). Data: KHDP `inspire` 1.3 (CC BY 4.0), operations.csv and labs.csv.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from benchtoward.config import registry_dir

CODE, VERSION = "inspire", "1.3"
FILES = ("operations.csv", "labs.csv")
STABLE = "stable"  # the script's default quicksort orders ties by platform; see the spec's deviations


@lru_cache(maxsize=1)
def load_spec() -> dict:
    return yaml.safe_load((registry_dir() / "tasks" / "inspire-mortality.yaml").read_text(encoding="utf-8"))


def dataset_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "datasets" / CODE / VERSION


def task_dir(data_dir: Path) -> Path:
    return dataset_dir(data_dir) / "mortality-v0"


def fetch_inspire(data_dir: Path, client) -> Path:
    """Download operations.csv and labs.csv into data/datasets/inspire/1.3/ (resumable: complete files are kept).

    The release has no checksum file, so a file counts as complete when its size matches the KHDP listing.
    """
    root = dataset_dir(data_dir)
    listing = {f.key.removeprefix(f"{VERSION}/"): f for f in client.files(CODE, VERSION)}
    for name in FILES:
        f = listing.get(name)
        if f is None:
            raise FileNotFoundError(f"{name} is not in the KHDP {CODE} {VERSION} release")
        dest = root / name
        if dest.exists() and dest.stat().st_size == f.size:
            continue
        client.download(f, dest)
        if dest.stat().st_size != f.size:
            raise ValueError(f"{name}: downloaded {dest.stat().st_size} bytes, KHDP lists {f.size}")
    return root


def input_columns(spec: dict | None = None) -> list[str]:
    spec = spec or load_spec()
    return list(spec["inputs"]["script"]) + list(spec["inputs"]["extra"])


def build_cohort(ops: pd.DataFrame, labs: pd.DataFrame, spec: dict | None = None) -> pd.DataFrame:
    """One row per patient in the script's order before its shuffle: cohort, label and inputs as in gbm_mortality.py,
    except that each patient's operation is the earliest by orin_time (ties: smallest op_id). The published code
    takes the smallest op_id, which INSPIRE assigns at random (spec deviations)."""
    spec = spec or load_spec()
    df = ops.sort_values(["subject_id", "orin_time", "op_id"], kind=STABLE)
    df = df.groupby("subject_id").head(1)  # rows stay in subject_id order, as the script's .loc[idxmin] leaves them
    df = df.assign(label=(df["inhosp_death_time"] < df["orout_time"] + spec["label"]["window_min"]).astype(int))
    df = df[df["asa"] < 6].copy()
    df["andur"] = df["anend_time"] - df["anstart_time"]
    df["bmi"] = np.where(df["height"] > 10, df["weight"] / (df["height"] / 100) ** 2, np.nan)
    for item in spec["inputs"]["lab_items"]:
        right = labs.loc[labs["item_name"] == item, ["subject_id", "chart_time", "value"]]
        df = pd.merge_asof(df.sort_values("orin_time", kind=STABLE), right.sort_values("chart_time", kind=STABLE),
                           left_on="orin_time", right_on="chart_time", by="subject_id", direction="backward",
                           tolerance=spec["inputs"]["lab_window_min"])
        df = df.drop(columns="chart_time").rename(columns={"value": f"preop_{item}"})
    df["sex"] = (df["sex"] == "M").astype(float)
    df["vitaldb_linked"] = df["case_id"].notna()
    return df[["op_id", "subject_id", "label", *input_columns(spec), "vitaldb_linked"]].reset_index(drop=True)


def assign_split(cohort: pd.DataFrame, spec: dict | None = None) -> pd.DataFrame:
    """The script's shuffle; the first int(0.7 n) rows are train. Row ntrain, which the script also puts in train, is
    test only."""
    spec = spec or load_spec()
    df = cohort.sample(frac=1, random_state=spec["split"]["seed"]).reset_index(drop=True)
    ntrain = int(len(df) * spec["split"]["train_fraction"])
    df["split"] = np.where(df.index < ntrain, "train", "test")
    return df


def build_task(data_dir: Path) -> Path:
    spec = load_spec()
    root = dataset_dir(data_dir)
    ops = pd.read_csv(root / "operations.csv")
    labs = pd.read_csv(root / "labs.csv")
    df = assign_split(build_cohort(ops, labs, spec), spec)
    df.insert(0, "instance_id", "inspire-" + df["subject_id"].astype(str) + "-" + df["op_id"].astype(str))
    out = task_dir(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "instances.parquet"
    df.to_parquet(path, index=False)
    return path
