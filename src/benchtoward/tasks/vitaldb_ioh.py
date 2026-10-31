"""VitalDB intraoperative hypotension, version 0 (preview): numeric MAP every 2 s, 5 min ahead."""
import io
import json
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import yaml

from benchtoward.config import registry_dir

TASK_REF = "vitaldb-ioh@0"  # explicit: once a later version is published it becomes the latest
CASES_URL = "https://api.vitaldb.net/cases"
PART_SIZE = 200
MIN_COMPLETE_RATIO = 0.95  # downloaded samples x interval / recorded case length; all 3,695 cases were 1.0
LOAD_ATTEMPTS = 3


@lru_cache(maxsize=1)
def load_spec() -> dict:
    return yaml.safe_load((registry_dir() / "tasks" / "vitaldb-ioh.yaml").read_text(encoding="utf-8"))


def task_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "datasets" / "vitaldb" / "ioh-v0"


def _has_event(segment: np.ndarray, threshold: float, run_length: int) -> bool:
    """True if `run_length` consecutive non-missing values are below `threshold`. NaN breaks a run."""
    run = 0
    for value in segment:
        run = run + 1 if (not np.isnan(value) and value < threshold) else 0
        if run >= run_length:
            return True
    return False


def make_windows(map_2s: np.ndarray, spec: dict | None = None) -> list[tuple[int, np.ndarray, int]]:
    spec = spec or load_spec()
    step = spec["interval_s"]
    n_in, n_h, stride = spec["input_window_s"] // step, spec["horizon_s"] // step, spec["stride_s"] // step
    low, high = spec["valid_range_mmhg"]
    threshold = spec["hypotension"]["threshold_mmhg"]
    run_length = spec["hypotension"]["min_duration_s"] // step
    max_missing = spec["max_missing_fraction"]

    x = np.asarray(map_2s, dtype=float).copy()
    x[(x < low) | (x > high)] = np.nan
    windows = []
    for start in range(0, len(x) - n_in - n_h + 1, stride):
        inp, horizon = x[start:start + n_in], x[start + n_in:start + n_in + n_h]
        if np.isnan(inp).mean() > max_missing or np.isnan(horizon).mean() > max_missing:
            continue
        if spec["exclude_if_input_hypotensive"] and _has_event(inp, threshold, run_length):
            continue
        windows.append((start, inp, int(_has_event(horizon, threshold, run_length))))
    return windows


def split_subjects(subject_ids: Iterable[int], seed: int = 20261029,
                   fractions=(0.7, 0.1, 0.2)) -> dict[int, str]:
    ids = np.array(sorted(set(int(s) for s in subject_ids)))
    order = np.random.default_rng(seed).permutation(len(ids))
    n_train = round(fractions[0] * len(ids))
    n_val = round(fractions[1] * len(ids))
    split = {}
    for rank, idx in enumerate(order):
        split[int(ids[idx])] = "train" if rank < n_train else ("validation" if rank < n_train + n_val else "test")
    return split


def series_is_complete(values, caseend_s: float, interval_s: float) -> bool:
    """A cut-short download (vitaldb treats a truncated stream as end of file) yields a short series."""
    return len(values) > 0 and len(values) * interval_s >= MIN_COMPLETE_RATIO * float(caseend_s)


def _read_cases() -> pd.DataFrame:
    raw = httpx.get(CASES_URL, timeout=120, follow_redirects=True).content
    try:
        return pd.read_csv(io.BytesIO(raw), compression="gzip")
    except OSError:
        return pd.read_csv(io.BytesIO(raw))


def fetch_tracks(data_dir: Path, workers: int = 8) -> Path:
    """Download the MAP track for the cohort into resumable parquet parts of PART_SIZE cases."""
    import vitaldb

    spec = load_spec()
    root = task_dir(data_dir)
    (root / "tracks").mkdir(parents=True, exist_ok=True)
    clinical_path = root / "clinical.parquet"
    if not clinical_path.exists():
        _read_cases().to_parquet(clinical_path, index=False)
    clinical = pd.read_parquet(clinical_path)
    with_track = set(int(c) for c in vitaldb.find_cases([spec["track"]]))
    cohort = clinical[(clinical["ane_type"] == spec["cohort"]["ane_type"]) & clinical["caseid"].isin(with_track)]
    caseids = sorted(int(c) for c in cohort["caseid"])
    (root / "cohort.json").write_text(json.dumps(caseids))
    caseend = clinical.set_index("caseid")["caseend"]

    def load(caseid: int) -> list[float]:
        for _ in range(LOAD_ATTEMPTS):
            values = vitaldb.load_case(caseid, [spec["track"]], spec["interval_s"])
            values = [] if values is None else np.asarray(values, dtype=np.float32).reshape(-1).tolist()
            if series_is_complete(values, caseend[caseid], spec["interval_s"]):
                return values
        raise RuntimeError(f"case {caseid}: incomplete download after {LOAD_ATTEMPTS} attempts")

    for part, start in enumerate(range(0, len(caseids), PART_SIZE)):
        path = root / "tracks" / f"part-{part:05d}.parquet"
        if path.exists():
            continue
        chunk = caseids[start:start + PART_SIZE]
        with ThreadPoolExecutor(workers) as pool:
            series = list(pool.map(load, chunk))
        pd.DataFrame({"caseid": chunk, "map": series}).to_parquet(path, index=False)
    return root


def build_task(data_dir: Path) -> Path:
    spec = load_spec()
    root = task_dir(data_dir)
    clinical = pd.read_parquet(root / "clinical.parquet").set_index("caseid")
    split = split_subjects(clinical["subjectid"], spec["split"]["seed"], tuple(spec["split"]["fractions"]))
    parts = [pd.read_parquet(part) for part in sorted((root / "tracks").glob("part-*.parquet"))]
    cohort_path = root / "cohort.json"
    if cohort_path.exists():
        fetched = {int(c) for part in parts for c in part["caseid"]}
        missing = sorted(set(json.loads(cohort_path.read_text())) - fetched)
        if missing:
            raise RuntimeError(f"{len(missing)} cohort case(s) not downloaded, e.g. {missing[:5]}; rerun fetch")
    records = []
    for part in parts:
        for caseid, values in part.itertuples(index=False):
            subject = int(clinical.loc[caseid, "subjectid"])
            for start, x, label in make_windows(np.asarray(values, dtype=float), spec):
                records.append({"instance_id": f"ioh-{caseid}-{start}", "caseid": int(caseid), "subjectid": subject,
                                "split": split[subject], "x": x.astype(np.float32).tolist(), "label": label})
    path = root / "instances.parquet"
    pd.DataFrame(records).to_parquet(path, index=False)
    return path
