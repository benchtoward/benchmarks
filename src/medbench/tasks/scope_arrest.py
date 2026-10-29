"""SCOPE (KHDP `icu-cardiac-arrest` 1.0, CC BY 4.0): in-unit cardiac arrest within 0.5-24 h from a 5-min ECG epoch,
version 0 (preview). Spec: registry/tasks/scope-arrest.yaml. Window rule: medbench.tasks.scope_sampling. HRV features:
medbench.tasks.scope_hrv. The ~376 GB of .vital files are read on compute pods (scripts/scope_build.py,
medbench.tasks.scope_pods); only per-epoch features and the test split's 5-min ECG epochs come back.

Layout under the data dir:
  datasets/scope/1.0/metadata.parquet    the release's clinical metadata plus the stay split
  datasets/scope/1.0/split.csv           case_id,split (also published as registry/splits/scope-1.0.csv)
  datasets/scope/1.0/pod-out/            shards copied back from the pods
  datasets/scope/arrest-v0/              instances.parquet, epochs.parquet, stays.parquet, test_ecg_index.parquet
"""
import json
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from medbench.config import registry_dir
from medbench.tasks.scope_hrv import CANDIDATES
from medbench.tasks.scope_sampling import SEED, label_for
from medbench.tasks.vitaldb_ioh import split_subjects

CODE, VERSION = "icu-cardiac-arrest", "1.0"
FRACTIONS = (0.7, 0.1, 0.2)
INSTANCE_COLUMNS = ["instance_id", "case_id", "split", "label", "outcome", "start_s", *CANDIDATES]
# start_s (seconds before the outcome) and case_id are for scoring and audits only; they are never model inputs.
NOT_INPUTS = ("case_id", "split", "label", "outcome", "start_s")


@lru_cache(maxsize=1)
def load_spec() -> dict:
    return yaml.safe_load((registry_dir() / "tasks" / "scope-arrest.yaml").read_text(encoding="utf-8"))


def dataset_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "datasets" / "scope" / VERSION


def pod_out_dir(data_dir: Path) -> Path:
    return dataset_dir(data_dir) / "pod-out"


def task_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "datasets" / "scope" / "arrest-v0"


def stay_split(meta: pd.DataFrame, seed: int = SEED, fractions: Iterable[float] = FRACTIONS) -> dict[int, str]:
    """Dataset-level split by ICU stay, stratified by outcome (SCOPE pseudonymizes stays, so repeat stays of one
    patient cannot be grouped). Every SCOPE task uses this split."""
    out: dict[int, str] = {}
    for _, group in sorted(meta.groupby("outcome"), key=lambda kv: kv[0]):
        out.update(split_subjects(group["case_id"].astype(int), seed=seed, fractions=tuple(fractions)))
    return out


def read_metadata_xlsx(path: Path) -> pd.DataFrame:
    meta = pd.read_excel(path)
    meta["case_id"] = meta["case_id"].astype(int)
    meta["circulatory_support"] = meta["circulatory_support"].astype(str)
    return meta


def write_metadata(data_dir: Path, meta: pd.DataFrame, split: dict[int, str]) -> Path:
    root = dataset_dir(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    meta = meta.assign(case_id=meta["case_id"].astype(int))
    meta["split"] = meta["case_id"].map(split)
    if meta["split"].isna().any():
        raise ValueError("stays without a split")
    meta.to_parquet(root / "metadata.parquet", index=False)
    meta.sort_values("case_id")[["case_id", "split"]].to_csv(root / "split.csv", index=False)
    return root


def load_metadata(data_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(dataset_dir(data_dir) / "metadata.parquet")


def remaining_stays(stays: pd.DataFrame, shard_dir: Path) -> pd.DataFrame:
    """Stays not yet finished (without an error) in any collected stays-*.parquet, so a restarted pod build
    spreads only the remaining work over its pods."""
    done: set[int] = set()
    for path in Path(shard_dir).glob("stays-*.parquet"):
        part = pd.read_parquet(path)
        ok = part["error"].isna() if "error" in part else pd.Series(True, index=part.index)
        done |= set(part.loc[ok, "case_id"].astype(int))
    return stays[~stays["case_id"].astype(int).isin(done)].reset_index(drop=True)


def _read_all(pattern: str, data_dir: Path) -> pd.DataFrame:
    paths = sorted(pod_out_dir(data_dir).glob(pattern))
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True) if paths else pd.DataFrame()


def build_task(data_dir: Path) -> Path:
    """Merge the pod shards into instances.parquet (kept epochs with label and split) plus audit tables."""
    meta = load_metadata(data_dir)
    stays = _read_all("stays-*.parquet", data_dir)
    if "error" not in stays:
        stays["error"] = None
    finished = stays[stays["error"].isna()].drop_duplicates("case_id", keep="last") if len(stays) else stays
    missing = sorted(set(meta["case_id"].astype(int)) - set(finished.get("case_id", pd.Series(dtype=int)).astype(int)))
    if missing:
        raise RuntimeError(f"{len(missing)} stay(s) not processed by the pods, e.g. {missing[:5]}; rerun the pod build")
    epochs = _read_all("epochs-*.parquet", data_dir).drop_duplicates("instance_id", keep="last")
    epochs = epochs[epochs["case_id"].isin(meta["case_id"])]
    labelled = epochs.merge(meta[["case_id", "outcome", "split"]], on="case_id", how="left")
    labelled["label"] = labelled["outcome"].map(label_for).astype(int)
    check = finished.set_index("case_id")["label"].astype(int)
    if (labelled["case_id"].map(check) != labelled["label"]).any():
        raise RuntimeError("labels computed in the pod disagree with the metadata")
    instances = labelled[labelled["status"] == "ok"][INSTANCE_COLUMNS].sort_values(["case_id", "start_s"])

    out = task_dir(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    instances.to_parquet(out / "instances.parquet", index=False)
    labelled.to_parquet(out / "epochs.parquet", index=False)
    finished.merge(meta, on="case_id", how="left", suffixes=("", "_meta")).to_parquet(out / "stays.parquet", index=False)

    index = []
    for path in sorted(pod_out_dir(data_dir).glob("ecg-test-*.npz")):
        with np.load(path) as shard:
            index += [{"instance_id": str(i), "file": path.name, "row": k} for k, i in enumerate(shard["instance_id"])]
    index = pd.DataFrame(index, columns=["instance_id", "file", "row"]).drop_duplicates("instance_id", keep="last")
    index.to_parquet(out / "test_ecg_index.parquet", index=False)
    test_ids = set(instances.loc[instances["split"] == "test", "instance_id"])
    summary = {
        "stays": int(meta["case_id"].nunique()), "instances": int(len(instances)),
        "epochs_by_status": labelled["status"].value_counts().to_dict(),
        "by_split": {s: {"instances": int(len(g)), "stays": int(g["case_id"].nunique()),
                         "arrest_stays": int(g.loc[g["label"] == 1, "case_id"].nunique()),
                         "prevalence": float(g["label"].mean())} for s, g in instances.groupby("split")},
        "test_ecg_missing": len(test_ids - set(index["instance_id"])),
        "file_end_after_anchor_s_by_outcome": {o: {"median": float(g["dtend_s"].median()),
                                                   "max": float(g["dtend_s"].max())}
                                               for o, g in finished.merge(meta[["case_id", "outcome"]], on="case_id")
                                               .groupby("outcome") if "dtend_s" in g},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return out / "instances.parquet"


def load_test_ecg(data_dir: Path, instance_ids: Iterable[str]) -> np.ndarray:
    """Raw 250 Hz test epochs (float16, n x 75000) in the order of `instance_ids`."""
    index = pd.read_parquet(task_dir(data_dir) / "test_ecg_index.parquet").set_index("instance_id")
    wanted = list(instance_ids)
    out = np.empty((len(wanted), 75_000), dtype=np.float16)
    rows = index.loc[wanted].assign(pos=range(len(wanted)))
    for name, group in rows.groupby("file"):
        with np.load(pod_out_dir(data_dir) / name) as shard:
            out[group["pos"].to_numpy()] = shard["ecg"][group["row"].to_numpy()]
    return out
