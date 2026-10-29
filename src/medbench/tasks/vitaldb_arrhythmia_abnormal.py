"""vitaldb-arrhythmia-abnormal, version 0 (preview): normal vs abnormal rhythm in 5-second ECG lead II segments.

Follows the evaluation in the dataset paper's Technical Validation (Eun et al., Sci Data 2026,
doi:10.1038/s41597-026-07076-8); the reading of each rule and the deviations are in
registry/tasks/vitaldb-arrhythmia-abnormal.yaml. Labels come from the KHDP annotation release; ECG lead II comes from
VitalDB and is cached per case for the annotated span only (float32 .npy plus a JSON sidecar written last).

Instances are references (caseid, start_s, duration_s) with labels; `materialise` turns them into arrays.
"""
import json
import math
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from medbench.config import registry_dir
from medbench.tasks.vitaldb_arrhythmia import (CODE, ECG_HZ, ECG_TRACK, dataset_dir, load_annotations,  # noqa: F401
                                               load_metadata)

SLUG = "vitaldb-arrhythmia-abnormal"
BEATS = ("N", "S", "V", "U", "P")   # beat_type values that are beats (Start/End in one file are quality markers)
LOAD_ATTEMPTS = 3
REASONS = ("annotation_gap", "noise", "no_beats", "premature_beats", "short_nsr", "ecg_missing", "bad_signal_quality")
QC_BEATS = 300   # beats sampled per case for the alignment check


@lru_cache(maxsize=1)
def load_spec() -> dict:
    return yaml.safe_load((registry_dir() / "tasks" / f"{SLUG}.yaml").read_text(encoding="utf-8"))


def task_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "datasets" / CODE / "abnormal-v0"


def ecg_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "datasets" / CODE / "ecg"


def instance_id(caseid: int, start_s: int) -> str:
    return f"arr-{int(caseid)}-{int(start_s)}"


# --- segments and labels ----------------------------------------------------------------------------------------


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First and last index of each run of True values."""
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return starts, ends


def segment_case(rows: pd.DataFrame, spec: dict | None = None) -> pd.DataFrame:
    """Every grid segment inside one case's annotated span, with its label (0, 1, or missing with a `reason`).

    A segment's rhythm labels are those of the annotation rows inside [start, start + duration) plus the last row
    at or before the start and the first row at or after the end, because the time between two rows belongs to
    both of them. Rows without a rhythm label (quality markers) are not used for labels.
    """
    spec = spec or load_spec()
    seg, lab = spec["segments"], spec["label"]
    duration, grid, max_gap = spec["input"]["duration_s"], seg["grid_s"], seg["max_annotation_gap_s"]
    normal, not_scored = set(lab["normal"]), set(lab["not_scored"])
    columns = ["start_s", "label", "reason", "rhythm", "rhythms", "n_beats", "bad_signal_quality"]

    rows = rows.sort_values("time_second", kind="stable")
    unknown = set(rows["rhythm_label"].dropna()) - normal - set(lab["abnormal"]) - not_scored
    if unknown:
        raise ValueError(f"unknown rhythm label(s) {sorted(unknown)}; the task spec lists the allowed labels")
    t_all = rows["time_second"].to_numpy(dtype=float)
    beat_t = t_all[rows["beat_type"].isin(BEATS).to_numpy()]
    premature_t = t_all[rows["beat_type"].isin(lab["premature_beats"]).to_numpy()]
    bad_lo, bad_hi = (t_all[i] for i in _runs(rows["bad_signal_quality"].to_numpy(dtype=bool)))
    labelled = rows["rhythm_label"].notna().to_numpy()
    lt, ll = t_all[labelled], rows["rhythm_label"].to_numpy()[labelled].astype(str)
    if len(lt) < 2:
        return pd.DataFrame(columns=columns)

    step = np.diff(lt)
    gap = np.flatnonzero(step > max_gap)                     # unannotated time between lt[k] and lt[k + 1]
    new_run = np.r_[True, (ll[1:] != ll[:-1]) | (step > max_gap)]
    run_id = np.cumsum(new_run) - 1
    run_first = lt[new_run]
    run_last = lt[np.r_[new_run[1:], True]]

    out = []
    for start in range(math.ceil(lt[0] / grid) * grid, math.floor(lt[-1] - duration) + 1, grid):
        end = start + duration
        i0 = int(np.searchsorted(lt, start, "right")) - 1     # last labelled row at or before the start
        i1 = int(np.searchsorted(lt, end, "left"))            # first labelled row at or after the end
        labels = ll[i0:i1 + 1]
        present = set(labels)
        n_beats = int(np.searchsorted(beat_t, end, "left") - np.searchsorted(beat_t, start, "left"))
        n_premature = int(np.searchsorted(premature_t, end, "left") - np.searchsorted(premature_t, start, "left"))
        bad = bool(((bad_lo < end) & (bad_hi >= start)).any())
        label, reason, rhythm = None, None, None
        if ((lt[gap] < end) & (lt[gap + 1] > start)).any():
            reason = "annotation_gap"
        elif present & not_scored:
            reason = "noise"
        elif n_beats < seg["min_beats"]:
            reason = "no_beats"
        elif present - normal:
            label = 1
            counts = Counter(x for x in labels if x not in normal)
            rhythm = min(counts, key=lambda k: (-counts[k], k))
        else:
            rhythm = "N"
            run = run_id[i0]
            if n_premature:
                reason = "premature_beats"
            elif run_last[run] - run_first[run] < lab["normal_context_s"]:
                reason = "short_nsr"
            else:
                label = 0
        out.append({"start_s": start, "label": label, "reason": reason, "rhythm": rhythm,
                    "rhythms": "|".join(sorted(present)), "n_beats": n_beats, "bad_signal_quality": bad})
    df = pd.DataFrame(out, columns=columns)
    df["label"] = df["label"].astype("Int64")
    return df


# --- ECG cache --------------------------------------------------------------------------------------------------


def cached_span(start_s: float, end_s: float, grid: int) -> tuple[int, int]:
    """Grid-aligned [first, last) seconds that hold every segment of a case."""
    return int(math.floor(start_s / grid) * grid), int(math.ceil(end_s / grid) * grid)


def alignment_offset_ms(x: np.ndarray, x0_s: float, beat_times, hz: int = ECG_HZ) -> float | None:
    """Median lag (ms) from annotated R-peaks to the band-passed ECG maximum within 200 ms. 0 means aligned."""
    from scipy.signal import butter, filtfilt

    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).any():
        return None
    b, a = butter(2, [5, 30], btype="band", fs=hz)
    y = np.abs(filtfilt(b, a, np.nan_to_num(x - np.nanmean(x))))
    half = int(0.2 * hz)
    beat_times = np.asarray(beat_times, dtype=float)
    if len(beat_times) > QC_BEATS:
        beat_times = beat_times[np.linspace(0, len(beat_times) - 1, QC_BEATS).astype(int)]
    lags = []
    for t in beat_times:
        i = int(round((t - x0_s) * hz))
        if i - half >= 0 and i + half <= len(y):
            lags.append(int(np.argmax(y[i - half:i + half])) - half)
    return float(np.median(lags) * 1000 / hz) if lags else None


def _vitaldb_load(caseid: int) -> np.ndarray | None:
    import vitaldb

    values = vitaldb.load_case(int(caseid), [ECG_TRACK], 1 / ECG_HZ)
    return None if values is None or len(values) == 0 else np.asarray(values, dtype=np.float32)[:, 0]


def _write_atomic(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        write(fh)
    os.replace(tmp, path)


def fetch_ecg(data_dir: Path, workers: int = 8, load=None) -> dict:
    """Download ECG lead II for every annotated case and keep the annotated span (resumable: the JSON sidecar is
    written last and marks a finished case). Raises after the loop if any case stayed incomplete."""
    spec = load_spec()
    load = load or _vitaldb_load
    hz, grid = spec["input"]["hz"], spec["segments"]["grid_s"]
    out = ecg_dir(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = load_metadata(data_dir).sort_values("caseid")
    ann = load_annotations(data_dir)
    beats = {int(c): g.loc[g["beat_type"].isin(["N", "S", "V"]), "time_second"].to_numpy()
             for c, g in ann.groupby("caseid")}
    todo = [r for r in meta.itertuples(index=False) if not (out / f"{int(r.caseid)}.json").exists()]
    started = time.time()

    def one(r) -> str | None:
        caseid = int(r.caseid)
        first, last = cached_span(r.analysis_start_time_sec, r.analysis_end_time_sec, grid)
        x = None
        for _ in range(LOAD_ATTEMPTS):
            try:
                x = load(caseid)
            except Exception:   # a dropped connection is retried like a short download
                x = None
            if x is not None and len(x) / hz >= r.analysis_end_time_sec:
                break
        else:
            return f"case {caseid}: incomplete ECG after {LOAD_ATTEMPTS} attempts"
        window = np.full((last - first) * hz, np.nan, dtype=np.float32)
        piece = np.asarray(x[first * hz:last * hz], dtype=np.float32)
        window[:len(piece)] = piece
        _write_atomic(out / f"{caseid}.npy", lambda fh: np.save(fh, window))
        offset = alignment_offset_ms(window, first, beats.get(caseid, []), hz)
        sidecar = {"caseid": caseid, "start_s": first, "end_s": last, "hz": hz, "track": ECG_TRACK,
                   "case_samples": int(len(x)), "nan_fraction": float(np.isnan(window).mean()),
                   "qc": {"median_offset_ms": offset}}
        _write_atomic(out / f"{caseid}.json", lambda fh: fh.write(json.dumps(sidecar).encode("utf-8")))
        return None

    with ThreadPoolExecutor(max(1, workers)) as pool:
        errors = [e for e in pool.map(one, todo) if e]
    cached = sorted(out.glob("*.npy"))
    summary = {"cases": len(meta), "fetched": len(todo) - len(errors), "failed": len(errors),
               "seconds": round(time.time() - started, 1), "cache_bytes": sum(p.stat().st_size for p in cached)}
    if errors:
        raise RuntimeError(f"{len(errors)} case(s) could not be downloaded completely, e.g. {errors[:3]}; rerun fetch")
    return summary


def _cached(data_dir: Path, caseid: int) -> tuple[np.ndarray, int]:
    side = json.loads((ecg_dir(data_dir) / f"{int(caseid)}.json").read_text(encoding="utf-8"))
    return np.load(ecg_dir(data_dir) / f"{int(caseid)}.npy", mmap_mode="r"), int(side["start_s"])


def materialise(instances: pd.DataFrame, data_dir: Path) -> np.ndarray:
    """ECG arrays (n, duration * hz) float32 for instances, in row order, from the cache."""
    spec = load_spec()
    hz = spec["input"]["hz"]
    n = spec["input"]["duration_s"] * hz
    out = np.full((len(instances), n), np.nan, dtype=np.float32)
    positions = np.arange(len(instances))
    for caseid, idx in pd.Series(positions).groupby(instances["caseid"].to_numpy()):
        x, first = _cached(data_dir, caseid)
        for pos in idx.to_numpy():
            a = (int(instances["start_s"].iat[pos]) - first) * hz
            out[pos] = x[a:a + n]
    return out


# --- build ------------------------------------------------------------------------------------------------------


def _vitaldb_cases(root: Path) -> pd.DataFrame:
    path = root / "vitaldb_cases.parquet"
    if not path.exists():
        from medbench.tasks.vitaldb_split import load_cases

        load_cases()[["caseid", "subjectid"]].to_parquet(path, index=False)
    return pd.read_parquet(path)


def build_task(data_dir: Path, cases: pd.DataFrame | None = None) -> Path:
    """Segments, labels and the dataset-level VitalDB split -> instances.parquet (+ build_summary.json)."""
    from medbench.tasks.vitaldb_split import caseid_split

    spec = load_spec()
    hz, duration = spec["input"]["hz"], spec["input"]["duration_s"]
    root = task_dir(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    meta = load_metadata(data_dir).sort_values("caseid")
    if cases is None:
        cases = _vitaldb_cases(root)
    vitaldb_subject = dict(zip(cases["caseid"].astype(int), cases["subjectid"]))
    wrong = [int(c) for c, s in zip(meta["caseid"], meta["subjectid"]) if vitaldb_subject.get(int(c)) != s]
    if wrong:
        raise ValueError(f"{len(wrong)} case(s) whose subjectid differs from VitalDB's, e.g. {wrong[:5]}")
    missing = [int(c) for c in meta["caseid"] if not (ecg_dir(data_dir) / f"{int(c)}.json").exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} case(s) without cached ECG, e.g. {missing[:5]}; rerun fetch")
    split = caseid_split(cases)
    subject = dict(zip(meta["caseid"].astype(int), meta["subjectid"].astype(int)))
    ann = load_annotations(data_dir)
    excluded = {s: dict.fromkeys(REASONS, 0) for s in ("train", "validation", "test")}
    records = []
    for caseid, rows in ann.groupby("caseid", sort=True):
        caseid = int(caseid)
        part = split[caseid]
        x, first = _cached(data_dir, caseid)
        for w in segment_case(rows, spec).itertuples(index=False):
            reason = None if pd.isna(w.reason) else w.reason
            if reason is None:
                a = (w.start_s - first) * hz
                if np.isnan(x[a:a + duration * hz]).mean() > spec["segments"]["max_ecg_missing_fraction"]:
                    reason = "ecg_missing"
                elif w.bad_signal_quality and part in spec["bad_signal_quality"]["exclude_from"]:
                    reason = "bad_signal_quality"
            if reason is not None:
                excluded[part][reason] += 1
                continue
            records.append({"instance_id": instance_id(caseid, w.start_s), "caseid": caseid,
                            "subjectid": subject[caseid], "split": part, "start_s": int(w.start_s),
                            "duration_s": duration, "label": int(w.label), "rhythm": w.rhythm, "rhythms": w.rhythms,
                            "n_beats": int(w.n_beats), "bad_signal_quality": bool(w.bad_signal_quality)})
    df = pd.DataFrame(records).sort_values(["caseid", "start_s"], ignore_index=True)
    path = root / "instances.parquet"
    df.to_parquet(path, index=False)
    summary = {"instances": len(df), "excluded": excluded, "splits": {}}
    for part, g in df.groupby("split"):
        summary["splits"][part] = {"instances": len(g), "subjects": int(g["subjectid"].nunique()),
                                   "cases": int(g["caseid"].nunique()),
                                   "label": {str(k): int(v) for k, v in g["label"].value_counts().sort_index().items()},
                                   "rhythm": {str(k): int(v) for k, v in g["rhythm"].value_counts().items()}}
    (root / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path
