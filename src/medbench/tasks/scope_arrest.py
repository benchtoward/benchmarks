"""SCOPE (KHDP `icu-cardiac-arrest` 1.0, CC BY 4.0): in-unit cardiac arrest within 0.5-24 h from a 5-min ECG epoch,
version 0 (preview), built with the dataset authors' SCOPE processing and training scripts (Lee et al. 2023 setup).
Spec: registry/tasks/scope-arrest.yaml. Window rule: medbench.tasks.scope_sampling. Epoch pipeline and HRV measures:
medbench.tasks.scope_hrv. The ~376 GB of .vital files are read on compute pods by a deployment-only build that is
not part of the public code; only per-epoch measures and the test split's 5-min ECG epochs come back.

Layout under the data dir:
  datasets/scope/1.0/metadata.parquet    the release's clinical metadata, in the order of its sheet (`sheet_row`)
  datasets/scope/1.0/pod-out/            shards copied back from the pods (all cohort stays, measures only)
  datasets/scope/1.0/pod-out-ecg/        shards of a second pod run over the test stays that keeps their ECG epochs
  datasets/scope/arrest-v0/              instances.parquet, epochs.parquet, stays.parquet, split.csv,
                                         test_ecg_index.parquet, summary.json
"""
import json
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from medbench.config import registry_dir
from medbench.tasks.scope_hrv import CANDIDATES, FEATURES
from medbench.tasks.scope_sampling import ARREST, CONTROL, label_for, stay_seeds

CODE, VERSION = "icu-cardiac-arrest", "1.0"
TEST_SIZE, SPLIT_SEED = 0.2, 42  # the authors' training script: 80/20 by stay, stratified, random_state 42
INSTANCE_COLUMNS = ["instance_id", "case_id", "split", "label", "outcome", "start_s", "minutes_to_end",
                    "quality_ratio", *CANDIDATES]
# For scoring and audits only; never model inputs.
NOT_INPUTS = ("case_id", "split", "label", "outcome", "start_s", "minutes_to_end", "quality_ratio")


@lru_cache(maxsize=1)
def load_spec() -> dict:
    return yaml.safe_load((registry_dir() / "tasks" / "scope-arrest.yaml").read_text(encoding="utf-8"))


def published_split_path() -> Path:
    return registry_dir().parent / load_spec()["split"]["file"]


def dataset_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "datasets" / "scope" / VERSION


def pod_out_dir(data_dir: Path) -> Path:
    return dataset_dir(data_dir) / "pod-out"


def ecg_out_dir(data_dir: Path) -> Path:
    return dataset_dir(data_dir) / "pod-out-ecg"


def task_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "datasets" / "scope" / "arrest-v0"


def read_metadata_xlsx(path: Path) -> pd.DataFrame:
    meta = pd.read_excel(path)
    meta["case_id"] = meta["case_id"].astype(int)
    if "circulatory_support" in meta:
        meta["circulatory_support"] = meta["circulatory_support"].astype(str)
    return meta


def write_metadata(data_dir: Path, meta: pd.DataFrame) -> Path:
    """Keep the sheet's row order (`sheet_row`): the authors' seeds and split depend on it."""
    root = dataset_dir(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    meta = meta.reset_index(drop=True).assign(case_id=meta["case_id"].astype(int).to_numpy())
    meta["sheet_row"] = np.arange(len(meta))
    meta.to_parquet(root / "metadata.parquet", index=False)
    return root


def load_metadata(data_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(dataset_dir(data_dir) / "metadata.parquet").sort_values("sheet_row")


def cohort(meta: pd.DataFrame) -> pd.DataFrame:
    """Arrest and discharge stays (deaths are left out, as in the authors' script) with label and worker seed."""
    meta = meta.sort_values("sheet_row") if "sheet_row" in meta else meta
    seeds = stay_seeds(meta["case_id"], meta["outcome"])
    out = meta[meta["case_id"].astype(int).isin(seeds)].copy()
    out["label"] = out["outcome"].map(label_for).astype(int)
    out["seed"] = out["case_id"].astype(int).map(seeds)
    return out.reset_index(drop=True)


def authors_split(stays_with_instances: Iterable[int], meta: pd.DataFrame) -> dict[int, str]:
    """The authors' train/test split: the stays that have at least one epoch, in the order of their feature table
    (arrest stays, then discharge stays, each in sheet order), split 80/20 with scikit-learn's train_test_split,
    stratified by the stay's label, random_state 42."""
    from sklearn.model_selection import train_test_split

    keep = {int(c) for c in stays_with_instances}
    meta = meta.sort_values("sheet_row") if "sheet_row" in meta else meta
    ordered = pd.concat([meta[meta["outcome"] == outcome] for outcome in (ARREST, CONTROL)])
    ordered = ordered[ordered["case_id"].astype(int).isin(keep)]
    ids = ordered["case_id"].astype(int).to_numpy()
    labels = (ordered["outcome"] == ARREST).astype(int).to_numpy()
    train, test = train_test_split(ids, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=labels)
    return {**{int(c): "train" for c in train}, **{int(c): "test" for c in test}}


def remaining_stays(stays: pd.DataFrame, shard_dir: Path) -> pd.DataFrame:
    """Stays not yet finished (without an error) in any collected stays-*.parquet, so a restarted pod build
    spreads only the remaining work over its pods."""
    done: set[int] = set()
    for path in Path(shard_dir).glob("stays-*.parquet"):
        part = pd.read_parquet(path)
        ok = part["error"].isna() if "error" in part else pd.Series(True, index=part.index)
        done |= set(part.loc[ok, "case_id"].astype(int))
    return stays[~stays["case_id"].astype(int).isin(done)].reset_index(drop=True)


def _read_all(pattern: str, folder: Path) -> pd.DataFrame:
    paths = sorted(Path(folder).glob(pattern))
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True) if paths else pd.DataFrame()


def is_instance(epochs: pd.DataFrame) -> pd.Series:
    """Kept epochs whose 33 model measures are not infinite: the authors impute missing measures but drop rows with
    an infinite one (their training script turns inf into NaN, then drops rows with NaN)."""
    finite = ~np.isinf(epochs[FEATURES].astype(float)).any(axis=1)
    return (epochs["status"] == "ok") & finite


def build_task(data_dir: Path, published_split: Path | None = None) -> Path:
    """Merge the pod shards into instances.parquet (kept epochs with label and split) plus audit tables. The split is
    the authors' (`authors_split`); if `published_split` is given, it must match."""
    meta = cohort(load_metadata(data_dir))
    stays = _read_all("stays-*.parquet", pod_out_dir(data_dir))
    if "error" not in stays:
        stays["error"] = None
    finished = stays[stays["error"].isna()].drop_duplicates("case_id", keep="last") if len(stays) else stays
    missing = sorted(set(meta["case_id"].astype(int)) - set(finished.get("case_id", pd.Series(dtype=int)).astype(int)))
    if missing:
        raise RuntimeError(f"{len(missing)} stay(s) not processed by the pods, e.g. {missing[:5]}; rerun the pod build")
    epochs = _read_all("epochs-*.parquet", pod_out_dir(data_dir)).drop_duplicates("instance_id", keep="last")
    epochs = epochs[epochs["case_id"].isin(meta["case_id"])]
    labelled = epochs.merge(meta[["case_id", "outcome", "label"]], on="case_id", how="left")
    check = finished.set_index("case_id")["label"].astype(int)
    if (labelled["case_id"].map(check) != labelled["label"]).any():
        raise RuntimeError("labels computed in the pod disagree with the metadata")
    kept = labelled[is_instance(labelled)]
    split = authors_split(kept["case_id"].unique(), meta)
    if published_split is not None:
        published = pd.read_csv(published_split)
        if dict(zip(published["case_id"].astype(int), published["split"])) != split:
            raise RuntimeError("the authors' split computed from these shards differs from the published split")
    instances = kept.assign(split=kept["case_id"].map(split))[INSTANCE_COLUMNS].sort_values(["case_id", "start_s"])

    out = task_dir(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    instances.to_parquet(out / "instances.parquet", index=False)
    labelled.to_parquet(out / "epochs.parquet", index=False)
    finished.merge(meta, on=["case_id", "label"], how="left").to_parquet(out / "stays.parquet", index=False)
    pd.DataFrame(sorted(split.items()), columns=["case_id", "split"]).to_csv(out / "split.csv", index=False)

    index = []
    for folder in (pod_out_dir(data_dir), ecg_out_dir(data_dir)):
        for path in sorted(Path(folder).glob("ecg-test-*.npz")):
            with np.load(path) as shard:
                index += [{"instance_id": str(i), "file": f"{folder.name}/{path.name}", "row": k}
                          for k, i in enumerate(shard["instance_id"])]
    index = pd.DataFrame(index, columns=["instance_id", "file", "row"]).drop_duplicates("instance_id", keep="last")
    test_ids = set(instances.loc[instances["split"] == "test", "instance_id"])
    index = index[index["instance_id"].isin(test_ids)]
    index.to_parquet(out / "test_ecg_index.parquet", index=False)
    all_meta = load_metadata(data_dir)
    excluded = all_meta.loc[~all_meta["case_id"].isin(meta["case_id"]), "outcome"]
    summary = {
        "stays": int(meta["case_id"].nunique()), "instances": int(len(instances)),
        "excluded_outcomes": excluded.value_counts().to_dict(),
        "stays_without_instances": int(meta["case_id"].nunique() - len(split)),
        "epochs_by_status": labelled["status"].value_counts().to_dict(),
        "dropped_infinite": int(((labelled["status"] == "ok") & ~is_instance(labelled)).sum()),
        "by_split": {s: {"instances": int(len(g)), "stays": int(g["case_id"].nunique()),
                         "arrest_stays": int(g.loc[g["label"] == 1, "case_id"].nunique()),
                         "prevalence": float(g["label"].mean())} for s, g in instances.groupby("split")},
        "test_ecg_missing": len(test_ids - set(index["instance_id"])),
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
        with np.load(dataset_dir(data_dir) / name) as shard:
            out[group["pos"].to_numpy()] = shard["ecg"][group["row"].to_numpy()]
    return out
