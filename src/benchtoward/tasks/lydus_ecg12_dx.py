"""lydus-ecg12-dx@0: multi-label classification of 10-second 12-lead ECGs (LYDUS-SNUH-ECG-50K 1.1.1) on the labels the
release itself provides (its reviewed rhythm, premature beat, bbb and avb columns; benchtoward.tasks.lydus_labels),
scored by macro AUROC over 15 primary targets (owner, 2026-11-04).

Build (`benchtoward build lydus-ecg12-dx`): read the release, encode the four label columns as 25 binary targets with
their known/unknown mask, link likely repeat ECGs and split by group (lydus_split; the file
registry/splits/lydus-snuh-ecg-50k-1.1.1-split.csv is checked, or written with --write-split), and write the
instances. Records the electrocardiograph did not analyse (read from `dx` with lydus_dx, for this exclusion only) or
without any signal are left out, as before, so the record set is unchanged.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from benchtoward.config import registry_dir
from benchtoward.tasks import lydus, lydus_dx, lydus_labels, lydus_split

SLUG = "lydus-ecg12-dx"
VERSION = 0
REF = f"{SLUG}@{VERSION}"


def load_spec() -> dict:
    return yaml.safe_load((registry_dir() / "tasks" / f"{SLUG}.yaml").read_text(encoding="utf-8"))


def engine_dir(data_dir: Path) -> Path:
    """Shared LYDUS build outputs (templates), reused by the other LYDUS tasks."""
    return lydus.dataset_dir(data_dir) / "engine"


def task_dir(data_dir: Path) -> Path:
    return lydus.dataset_dir(data_dir) / "ecg12-dx-v0"


def published_split_path() -> Path:
    return registry_dir().parent / load_spec()["split"]["file"]


def map_path() -> Path:
    """The dx statement map, used only to find records without an interpretation (not for labels)."""
    return registry_dir() / "maps" / lydus_dx.MAP_FILE


def instance_id(new_id: str) -> str:
    return "lydus12-" + hashlib.sha256(f"{REF}|{new_id}".encode()).hexdigest()[:16]


def class_names() -> list[str]:
    return list(lydus_labels.TARGETS)


def label_matrices(instances: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(y, known), booleans (records, 25) in lydus_labels.TARGETS order, from the instances' `targets` and
    `unknown_targets`; score and train a target only where it is known."""
    return lydus_labels.matrices(instances["targets"], instances["unknown_targets"])


def model_input(x8: np.ndarray, lead_set: str = "12") -> np.ndarray:
    """What a model reads: the record's leads of `lead_set` as float32 (raw counts as stored), (leads, 5000)."""
    return lydus.select_leads(lydus.to_12_leads(x8), lead_set)


def flat_leads(signals: lydus.Signals, chunk: int = 2000) -> np.ndarray:
    """(records,) number of stored leads that are constant over the 10 s."""
    out = np.zeros(len(signals.keys), dtype=int)
    for s in range(0, len(out), chunk):
        block = np.asarray(signals.values[s:s + chunk])
        out[s:s + chunk] = (block.max(axis=2) == block.min(axis=2)).sum(axis=1)
    return out


def _templates(data_dir: Path, n: int, workers: int) -> np.ndarray:
    path = engine_dir(data_dir) / "templates-v0.npy"
    if path.exists():
        cached = np.load(path)
        if len(cached) == n:
            return cached
    T = lydus_split.templates(data_dir, n, workers=workers)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, T)
    return T


def split_table(data_dir: Path, meta: pd.DataFrame, workers: int = 8) -> tuple[pd.DataFrame, dict]:
    """new_id, group, split for every record of the release, and the link summary."""
    spec = load_spec()["split"]
    T = _templates(data_dir, len(meta), workers)
    Z, ok = lydus_split.unit_vectors(T)
    rules = spec["rules"]
    pairs = lydus_split.template_pairs(Z, ok, rules["template"]["min_corr"])
    edges = lydus_split.link_pairs(meta, pairs, rules)
    groups = lydus_split.groups_from_links(meta["new_id"].to_numpy(object), edges)
    split = lydus_split.assign_split(groups, seed=spec["seed"], fractions=tuple(spec["fractions"]))
    table = pd.DataFrame({"new_id": meta["new_id"].to_numpy(object), "group": groups, "split": split})
    info = lydus_split.summary(edges, groups) | {"templates_missing": int((~ok).sum())}
    return table, {"info": info, "Z": Z, "ok": ok}


def check_split(table: pd.DataFrame, path: Path) -> str:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path}: no published split file (build with --write-split for a new version)")
    published = lydus_split.read_split_file(path)
    columns = published.columns.tolist()
    if columns not in (lydus_split.FULL_COLUMNS, lydus_split.PUBLIC_COLUMNS) or not published.astype(str).equals(
            table[columns].astype(str).reset_index(drop=True)):
        raise ValueError(f"the split built here differs from {path}")
    if columns == lydus_split.PUBLIC_COLUMNS:
        # The public copy omits the groups; the hash of the full table built here, groups included, is what the
        # spec's sha256 pins, so it also checks the groups this build re-derived.
        return lydus_split.table_sha256(table["new_id"], table["group"], table["split"])
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_map(table_path: Path, meta: pd.DataFrame) -> None:
    rows = pd.read_csv(table_path, dtype=str, keep_default_na=False)
    stored = dict(zip(rows["statement"], rows["n_records"].astype(int)))
    built = lydus_dx.statement_counts(meta["dx"])
    differs = [s for s in set(stored) | set(built) if stored.get(s) != built.get(s)]
    if differs:
        raise ValueError(f"{len(differs)} map rows whose n_records differ from the data, e.g. {sorted(differs)[:3]}")


def build_task(data_dir: Path, split_path: Path | None = None, write_split: bool = False, workers: int = 8,
               check_map_counts: bool = True, leakage: bool = True) -> dict[str, Path]:
    """Instances (instances.parquet) and build_summary.json in task_dir(data_dir)."""
    spec = load_spec()
    meta = lydus.read_metadata(data_dir)
    signals = lydus.open_signals(data_dir)
    lydus.check_alignment(meta, signals)
    table_path = map_path()
    if check_map_counts:
        check_map(table_path, meta)
    parsed = lydus_dx.parse_all(meta["dx"], lydus_dx.load_map(table_path))   # no_analysis only
    y_all, known_all = lydus_labels.encode(meta)
    names = [lydus_labels.target_names(y_all[r], known_all[r]) for r in range(len(meta))]
    flat = flat_leads(signals)

    split_path = Path(split_path) if split_path else published_split_path()
    if not write_split and not split_path.exists():
        raise FileNotFoundError(f"{split_path}: no published split file (build with --write-split)")
    table, extra = split_table(data_dir, meta, workers=workers)
    if write_split:
        lydus_split.write_split_file(split_path, table["new_id"], table["group"], table["split"])
    split_sha = check_split(table, split_path)

    excluded = {"no_analysis": parsed.no_analysis, "flat_signal": flat == len(lydus.STORED_LEADS)}
    keep = ~(excluded["no_analysis"] | excluded["flat_signal"])
    age = pd.to_numeric(meta["age"], errors="coerce")
    inst = pd.DataFrame({
        "instance_id": [instance_id(i) for i in meta["new_id"]],
        "new_id": meta["new_id"].to_numpy(object),
        "row": np.arange(len(meta)),                       # position in the release files
        "split": table["split"].to_numpy(object),
        "group": table["group"].to_numpy(object),
        "age": age.to_numpy(float),
        "sex": meta["sex"].to_numpy(object),
        "targets": [pos for pos, _ in names],               # positive targets (dataset-provided labels)
        "unknown_targets": [unk for _, unk in names],       # masked targets (e.g. rhythm "Possible pacing")
        "flat_leads": flat,
    })[keep].sort_values("instance_id", ignore_index=True)

    root = task_dir(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {"instances": root / "instances.parquet", "summary": root / "build_summary.json"}
    inst.to_parquet(paths["instances"], index=False)
    y, known = label_matrices(inst)
    parts = {p: (inst["split"] == p).to_numpy() for p in lydus_split.PARTS}
    test = parts["test"]
    summary = {
        "release": {"khdp": f"{lydus.KHDP_CODE}@{lydus.VERSION}", "records": int(len(meta)),
                    "sha256": lydus.SHA256},
        "excluded": {k: int(v.sum()) for k, v in excluded.items()},
        "records": {p: int(m.sum()) for p, m in parts.items()},
        "groups": {p: int(inst.loc[m, "group"].nunique()) for p, m in parts.items()},
        "positives": {p: {t: int((y[m, k] & known[m, k]).sum()) for k, t in enumerate(lydus_labels.TARGETS)}
                      for p, m in parts.items()},
        "known": {p: {t: int(known[m, k].sum()) for k, t in enumerate(lydus_labels.TARGETS)}
                  for p, m in parts.items()},
        "audit": lydus_labels.audit(meta),
        "eligible_test": lydus_labels.eligible(y[test], known[test], inst.loc[test, "group"].to_numpy(object)),
        "split_file": str(split_path.name), "split_sha256": split_sha,
        "links": extra["info"],
    }
    if leakage:
        rng = np.random.default_rng(spec["split"]["seed"])
        record_level = np.array(lydus_split.PARTS, dtype=object)[
            np.searchsorted(np.cumsum(spec["split"]["fractions"]), rng.random(len(meta)), side="right").clip(0, 2)]
        summary["leakage"] = {
            "group_split": lydus_split.leakage_estimate(table["split"].to_numpy(object), meta, extra["Z"], extra["ok"]),
            "record_level_split": lydus_split.leakage_estimate(record_level, meta, extra["Z"], extra["ok"]),
        }
    paths["summary"].write_text(lydus_split.dumps(summary), encoding="utf-8")
    return paths


def count_mismatches(summary: dict, spec: dict | None = None) -> list[str]:
    spec = spec or load_spec()
    out = []
    for key in ("records", "positives"):
        for part, expected in (spec.get("counts", {}).get(key) or {}).items():
            built = summary[key].get(part)
            if built != expected:
                out.append(f"{key} {part}: built {built}, spec {expected}")
    if "eligible_test" in summary:
        short = [t for t in lydus_labels.PRIMARY if t not in summary["eligible_test"]]
        if short:
            out.append(f"primary targets below the test-split minimum: {short}")
    if spec["split"].get("sha256") and spec["split"]["sha256"] != summary["split_sha256"]:
        out.append(f"split sha256: built {summary['split_sha256']}, spec {spec['split']['sha256']}")
    return out


def load_instances(data_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(task_dir(data_dir) / "instances.parquet")


def read_summary(data_dir: Path) -> dict:
    return json.loads((task_dir(data_dir) / "build_summary.json").read_text(encoding="utf-8"))
