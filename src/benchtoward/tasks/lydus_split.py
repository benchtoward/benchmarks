"""The LYDUS split: one file for every LYDUS task, keeping likely repeat ECGs of one patient on one side.

The release has no patient id: `new_id` is the record's index in the 160K release, and the 160K page counts 167,199
patients for about as many records. The data show repeat ECGs nonetheless (measured 2026-10-30, see the task spec's
`split.evidence`): records of the same sex whose median beats nearly coincide are much more common than such records
of opposite sex, and records next to each other in the index are often of the same sex and age. Two rules link
records; a group is a connected component of the links; groups are assigned whole to train, validation or test.

- template: same sex, ages at most 2 years apart, and median-beat templates (8 stored leads, 200 ms before to 400 ms
  after each lead II R peak, every 10 ms, z-normalised over all leads) with correlation >= 0.96;
- index: same sex, ages at most 1 year apart, and `new_id` at most 10 apart in the 160K index.
Records with unknown sex or age are never linked.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

TEMPLATE_BEFORE = 100     # samples at 500 Hz (200 ms)
TEMPLATE_AFTER = 200      # 400 ms
TEMPLATE_STEP = 5         # every 10 ms -> 60 points per lead
RULES = {"template": {"min_corr": 0.96, "max_age_gap": 2}, "index": {"max_id_gap": 10, "max_age_gap": 1}}
SEED = 20261030
FRACTIONS = (0.7, 0.1, 0.2)
PARTS = ("train", "validation", "test")


def beat_template(x8: np.ndarray, hz: int = 500) -> np.ndarray:
    """(8, 60) median beat around lead II R peaks of a band-passed (0.5-40 Hz) record; NaN if fewer than 2 beats."""
    from scipy.signal import butter, filtfilt

    from benchtoward.baselines.arrhythmia import detect_r_peaks

    x = np.asarray(x8, dtype=np.float64)
    out = np.full((x.shape[0], (TEMPLATE_BEFORE + TEMPLATE_AFTER) // TEMPLATE_STEP), np.nan, dtype=np.float32)
    if np.ptp(x[1]) == 0:
        return out
    try:
        peaks = np.asarray(detect_r_peaks(x[1], hz), dtype=int)
    except Exception:  # noqa: BLE001 - a record the detector cannot read gets no template
        return out
    peaks = peaks[(peaks >= TEMPLATE_BEFORE) & (peaks < x.shape[1] - TEMPLATE_AFTER)]
    if len(peaks) < 2:
        return out
    b, a = butter(2, [0.5, 40], btype="band", fs=hz)
    y = filtfilt(b, a, x, axis=1)
    windows = np.stack([y[:, p - TEMPLATE_BEFORE:p + TEMPLATE_AFTER] for p in peaks])
    return np.median(windows, axis=0)[:, ::TEMPLATE_STEP].astype(np.float32)


def _templates_chunk(args) -> np.ndarray:
    data_dir, start, stop = args
    from benchtoward.tasks import lydus

    values = lydus.open_signals(Path(data_dir)).values
    return np.stack([beat_template(np.asarray(values[i])) for i in range(start, stop)])


def templates(data_dir: Path, n: int, workers: int = 8, chunk: int = 1000) -> np.ndarray:
    """Templates of every record, in release order, computed in `workers` processes."""
    from multiprocessing import get_context

    jobs = [(str(data_dir), s, min(s + chunk, n)) for s in range(0, n, chunk)]
    if workers <= 1:
        return np.concatenate([_templates_chunk(j) for j in jobs])
    with get_context("spawn").Pool(workers) as pool:
        return np.concatenate(pool.map(_templates_chunk, jobs))


def unit_vectors(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flattened templates centred and scaled to unit length (dot product = correlation); rows without a template
    are zero and marked False in `ok`."""
    X = np.asarray(T, dtype=np.float64).reshape(len(T), -1)
    ok = np.isfinite(X).all(axis=1)
    Z = np.zeros_like(X)
    centred = X[ok] - X[ok].mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centred, axis=1, keepdims=True)
    ok_idx = np.flatnonzero(ok)
    good = norm[:, 0] > 0
    Z[ok_idx[good]] = centred[good] / norm[good]
    ok[ok_idx[~good]] = False
    return Z.astype(np.float32), ok


def template_pairs(Z: np.ndarray, ok: np.ndarray, min_corr: float, block: int = 2000):
    """All pairs i < j of usable records with correlation >= min_corr: (i, j, corr)."""
    Z = np.asarray(Z, dtype=np.float32)
    idx = np.flatnonzero(ok)
    Zo = Z[idx]
    out_i, out_j, out_c = [], [], []
    for s in range(0, len(idx), block):
        S = Zo[s:s + block] @ Zo.T
        a, b = np.nonzero(S >= min_corr)
        a_global = a + s
        keep = a_global < b
        out_i.append(idx[a_global[keep]])
        out_j.append(idx[b[keep]])
        out_c.append(S[a[keep], b[keep]])
    cat = (lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt))
    return cat(out_i, np.int64), cat(out_j, np.int64), cat(out_c, np.float32)


def _demo_match(sex, age, a, b, max_age_gap):
    known = pd.notna(sex[a]) & pd.notna(sex[b]) & np.isfinite(age[a]) & np.isfinite(age[b])
    return known & (sex[a] == sex[b]) & (np.abs(age[a] - age[b]) <= max_age_gap)


def link_pairs(meta: pd.DataFrame, pairs, rules: dict = RULES) -> pd.DataFrame:
    """Edges (i, j, rule) between record positions. `pairs`: template_pairs output (searched at the template rule's
    threshold or lower)."""
    sex = meta["sex"].to_numpy(object)
    age = pd.to_numeric(meta["age"], errors="coerce").to_numpy(float)
    i, j, corr = pairs
    t = rules["template"]
    keep = (corr >= t["min_corr"]) & _demo_match(sex, age, i, j, t["max_age_gap"])
    edges = [pd.DataFrame({"i": i[keep], "j": j[keep], "rule": "template"})]
    ids = meta["new_id"].astype(int).to_numpy()
    order = np.argsort(ids, kind="stable")
    r = rules["index"]
    for lag in range(1, r["max_id_gap"] + 1):
        a, b = order[:-lag], order[lag:]
        near = (ids[b] - ids[a] <= r["max_id_gap"]) & _demo_match(sex, age, a, b, r["max_age_gap"])
        if near.any():
            lo, hi = np.minimum(a[near], b[near]), np.maximum(a[near], b[near])
            edges.append(pd.DataFrame({"i": lo, "j": hi, "rule": "index"}))
    return pd.concat(edges, ignore_index=True)


def groups_from_links(ids: np.ndarray, edges: pd.DataFrame) -> np.ndarray:
    """Connected components of the links, each named by its smallest new_id."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    ids = np.asarray(ids, dtype=object)
    n = len(ids)
    graph = coo_matrix((np.ones(len(edges)), (edges["i"].to_numpy(int), edges["j"].to_numpy(int))), shape=(n, n))
    _, component = connected_components(graph, directed=False)
    name = pd.Series(ids).groupby(component).transform("min")
    return name.to_numpy(object)


def assign_split(groups: np.ndarray, seed: int = SEED, fractions=FRACTIONS) -> np.ndarray:
    """Shuffle the sorted group names with numpy's default_rng(seed); fill train, then validation, then test by
    cumulative share of records (a group goes where its last record falls)."""
    groups = np.asarray(groups, dtype=object)
    names, inverse, sizes = np.unique(groups.astype(str), return_inverse=True, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(names))
    share = np.cumsum(sizes[order]) / len(groups)
    part = np.where(share <= fractions[0] + 1e-12, 0, np.where(share <= fractions[0] + fractions[1] + 1e-12, 1, 2))
    by_group = np.empty(len(names), dtype=int)
    by_group[order] = part
    return np.array(PARTS, dtype=object)[by_group[inverse]]


def leakage_estimate(split: np.ndarray, meta: pd.DataFrame, Z: np.ndarray, ok: np.ndarray,
                     thresholds=(0.93, 0.95, 0.96), id_gaps=(10, 30), max_age_gap: int = 2) -> dict:
    """How many test records still have a likely repeat outside test. Template: a non-test record with
    correlation >= t and the same sex (ages within `max_age_gap`), against the same count with the other sex (which
    cannot be the same patient); the excess estimates repeats. Index: a non-test record within `id_gap` of new_id
    with the same sex and age, against the other sex."""
    split = np.asarray(split, dtype=object)
    sex = meta["sex"].to_numpy(object)
    age = pd.to_numeric(meta["age"], errors="coerce").to_numpy(float)
    known = pd.notna(sex) & np.isfinite(age)
    test = np.flatnonzero((split == "test") & ok & known)
    other = np.flatnonzero((split != "test") & ok & known)
    out = {"template": {}, "index": {}}
    same_any = {t: 0 for t in thresholds}
    opp_any = {t: 0 for t in thresholds}
    Zo = np.asarray(Z, dtype=np.float32)[other]
    for s in range(0, len(test), 2000):
        rows = test[s:s + 2000]
        S = np.asarray(Z, dtype=np.float32)[rows] @ Zo.T
        age_ok = np.abs(age[rows][:, None] - age[other][None, :]) <= max_age_gap
        same_sex = sex[rows][:, None] == sex[other][None, :]
        for t in thresholds:
            hit = (S >= t) & age_ok
            same_any[t] += int((hit & same_sex).any(axis=1).sum())
            opp_any[t] += int((hit & ~same_sex).any(axis=1).sum())
    for t in thresholds:
        out["template"][str(t)] = {"test_records_with_same_sex_neighbour": same_any[t],
                                   "with_other_sex_neighbour": opp_any[t], "excess": same_any[t] - opp_any[t]}
    ids = meta["new_id"].astype(int).to_numpy()
    order = np.argsort(ids, kind="stable")
    for gap in id_gaps:
        same_hit, opp_hit = np.zeros(len(ids), bool), np.zeros(len(ids), bool)
        for lag in range(1, gap + 1):
            a, b = order[:-lag], order[lag:]
            near = (ids[b] - ids[a] <= gap) & known[a] & known[b] & (age[a] == age[b])
            cross = (split[a] == "test") != (split[b] == "test")
            for x, y in ((a, b), (b, a)):
                sel = near & cross & (split[x] == "test")
                same_hit[x[sel & (sex[x] == sex[y])]] = True
                opp_hit[x[sel & (sex[x] != sex[y])]] = True
        out["index"][str(gap)] = {"test_records_with_same_sex_age_neighbour": int(same_hit.sum()),
                                  "with_other_sex_neighbour": int(opp_hit.sum()),
                                  "excess": int(same_hit.sum() - opp_hit.sum())}
    return out


def write_split_file(path: Path, ids, groups, split) -> str:
    """CSV new_id, group, split in release order; returns its sha256."""
    import hashlib

    df = pd.DataFrame({"new_id": ids, "group": groups, "split": split})
    text = df.to_csv(index=False, lineterminator="\n")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode()).hexdigest()


def read_split_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str)


FULL_COLUMNS = ["new_id", "group", "split"]
PUBLIC_COLUMNS = ["new_id", "split"]   # the likely-same-patient groups stay private (decision 2026-11-11)


def public_split_bytes(data: bytes) -> bytes:
    """The split file as published: the full file without its group column, rows and order unchanged."""
    lines = data.decode("utf-8").splitlines()
    if not lines or lines[0].split(",") != FULL_COLUMNS:
        raise ValueError("expected a full split file with columns new_id, group, split")
    out = []
    for line in lines:
        new_id, _group, split = line.split(",")
        out.append(f"{new_id},{split}")
    return ("\n".join(out) + "\n").encode("utf-8")


def table_sha256(ids, groups, split) -> str:
    """sha256 of the full split file write_split_file would write for this table."""
    import hashlib

    df = pd.DataFrame({"new_id": ids, "group": groups, "split": split})
    return hashlib.sha256(df.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def summary(edges: pd.DataFrame, groups: np.ndarray) -> dict:
    sizes = pd.Series(groups).value_counts()
    return {"links": {k: int(v) for k, v in edges["rule"].value_counts().items()},
            "groups": int(len(sizes)), "records_in_multi_record_groups": int(sizes[sizes > 1].sum()),
            "largest_group": int(sizes.max()), "group_size_counts": {str(k): int(v) for k, v in
                                                                     sizes.value_counts().sort_index().items()}}


def dumps(obj) -> str:
    return json.dumps(obj, indent=2, default=float) + "\n"
