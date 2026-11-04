"""vitaldb-arrhythmia-rhythm@0 and vitaldb-arrhythmia-beat@0 (hidden): the rhythm class of a 5-second ECG
lead II segment, and the AAMI class (N, S, V) of each annotated beat in it.

Both tasks cut the binary task's segments (vitaldb_arrhythmia_abnormal.segment_case: same grid, gap, noise and
beat rules) and use its dataset-level subject split, so the three tasks never put a patient on two sides. They share
one segment list, which lets a model answer both in one pass: segments.parquet holds every segment that passes the
shared segment and signal rules, with the positions of all its annotated beats; the rhythm task scores the subset
with one classifiable rhythm of a scored class, the beat task the N, S and V beats of the segments whose case has
annotations verified as aligned with the ECG (`beats_scored`). Readings and deviations are in
registry/tasks/vitaldb-arrhythmia-rhythm.yaml and -beat.yaml.
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from benchtoward.config import registry_dir
from benchtoward.tasks import vitaldb_arrhythmia_abnormal as binary
from benchtoward.tasks.vitaldb_arrhythmia import CODE, load_annotations, load_metadata

RHYTHM_SLUG, BEAT_SLUG = "vitaldb-arrhythmia-rhythm", "vitaldb-arrhythmia-beat"
TASK_KEYS = (f"{RHYTHM_SLUG}@0", f"{BEAT_SLUG}@0")
BEATS = binary.BEATS                 # beat_type values that are beats: N, S, V, U, P
SEGMENT_REASONS = ("annotation_gap", "noise", "no_beats", "ecg_missing", "bad_signal_quality")
RHYTHM_REASONS = ("unclassifiable", "mixed", "class_not_scored")
SPLITS = ("train", "validation", "test")


@lru_cache(maxsize=2)
def load_spec(slug: str) -> dict:
    return yaml.safe_load((registry_dir() / "tasks" / f"{slug}.yaml").read_text(encoding="utf-8"))


def task_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "datasets" / CODE / "rhythm-beat-v0"


def published_split_path() -> Path:
    return registry_dir().parent / load_spec(RHYTHM_SLUG)["split"]["file"]


def beat_instance_id(segment_id: str, sample: int) -> str:
    return f"{segment_id}-{int(sample)}"


# --- segments, rhythm classes and beats -------------------------------------------------------------------------


def _segment_rules(spec: dict) -> dict:
    """The binary segmenter's spec shape with this task's rules: every rhythm label is accepted (the class is chosen
    here), and N has no context or premature-beat condition, so the segmenter only marks gaps, noise and empty
    segments."""
    others = [c for c in spec["label"]["classes"] if c != "N"] + ["Unclassifiable"]   # rhythm_class drops it later
    return {"input": spec["input"], "segments": spec["segments"],
            "label": {"normal": ["N"], "abnormal": others, "not_scored": spec["segments"]["not_scored"],
                      "normal_context_s": 0, "premature_beats": []}}


def scored_classes(slug: str) -> list[str]:
    """The classes a task scores and a model answers: the rhythm spec's build-time `scored_classes` (classes in
    enough test patients), the beat spec's `classes`."""
    label = load_spec(slug)["label"]
    return list(label["scored_classes"] if "scored_classes" in label else label["classes"])


def rhythm_class(labels: set[str]) -> tuple[str | None, str | None]:
    """(class, None) for a scored segment, (None, reason) otherwise. N alone is N; one other label (with or without N)
    is that label; Unclassifiable anywhere, or two different other labels, is not scored."""
    if "Unclassifiable" in labels:
        return None, "unclassifiable"
    other = sorted(labels - {"N"})
    if len(other) > 1:
        return None, "mixed"
    return (other[0] if other else "N"), None


def beat_positions(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Annotated beat times and types of one case, in time order. A beat listed twice at the same time (the release
    has a few) is one position; if its two rows disagree, its type joins both (e.g. "S|U"), which no class maps."""
    beats = rows.loc[rows["beat_type"].isin(BEATS), ["time_second", "beat_type"]]
    grouped = beats.groupby("time_second", sort=True)["beat_type"].agg(lambda s: "|".join(sorted(set(s))))
    return grouped.index.to_numpy(dtype=float), grouped.to_numpy().astype(str)


def segment_case(rows: pd.DataFrame, spec: dict | None = None) -> pd.DataFrame:
    """Every grid segment of one case: the segmenter's reason (gap, noise, no beats, or none), the rhythm class or
    the reason it is not scored, and the sample offsets and types of the annotated beats in [start, start + 5 s)
    (`n_beats` counts these positions).

    The rhythm class comes from the labelled rows inside [start, start + 5 s) only (`rhythms`). Only a segment with
    no labelled row inside takes the binary segmenter's labels, the last row before and the first row after it
    (`rhythm_from_neighbours`). The segmenter's noise and gap rules still look at those neighbouring rows."""
    spec = spec or load_spec(RHYTHM_SLUG)
    hz, duration = spec["input"]["hz"], spec["input"]["duration_s"]
    out = binary.segment_case(rows, _segment_rules(spec))
    beat_t, beat_type = beat_positions(rows)
    labelled = rows.loc[rows["rhythm_label"].notna(), ["time_second", "rhythm_label"]].sort_values(
        "time_second", kind="stable")
    lt, ll = labelled["time_second"].to_numpy(dtype=float), labelled["rhythm_label"].to_numpy().astype(str)
    rhythm, rhythm_reason, rhythms, neighbours, samples, types = [], [], [], [], [], []
    for w in out.itertuples(index=False):
        a, b = np.searchsorted(beat_t, [w.start_s, w.start_s + duration], "left")
        samples.append([int(round((t - w.start_s) * hz)) for t in beat_t[a:b]])
        types.append([str(x) for x in beat_type[a:b]])
        i, j = np.searchsorted(lt, [w.start_s, w.start_s + duration], "left")
        inside = set(ll[i:j])
        labels = inside or set(w.rhythms.split("|"))
        label, reason = (None, None) if pd.notna(w.reason) else rhythm_class(labels)
        rhythm.append(label)
        rhythm_reason.append(reason)
        rhythms.append("|".join(sorted(labels)))
        neighbours.append(not inside)
    out = out.drop(columns=["label", "rhythm"]).assign(rhythm=rhythm, rhythm_reason=rhythm_reason, rhythms=rhythms,
                                                        rhythm_from_neighbours=neighbours, beat_samples=samples,
                                                        beat_types=types, n_beats=[len(s) for s in samples])
    return out[["start_s", "reason", "rhythm", "rhythm_reason", "rhythms", "rhythm_from_neighbours", "n_beats",
                "bad_signal_quality", "beat_samples", "beat_types"]]


def close_pairs(times, max_s: float) -> int:
    """Pairs of consecutive distinct beat positions less than `max_s` apart."""
    t = np.unique(np.asarray(times, dtype=float))
    return int((np.diff(t) < max_s).sum())


def rhythm_classes(seg: pd.DataFrame, classes, min_test_patients: int) -> tuple[list[str], dict[str, dict]]:
    """Scored classes (spec order: those in at least `min_test_patients` test patients) and, for every other class,
    its patients and segments per split. A class in too few test patients would put macro-F1 at the mercy of one
    or two patients, so it is not scored."""
    labelled = seg[seg["rhythm"].notna()]
    scored, not_scored = [], {}
    for c in classes:
        rows = labelled[labelled["rhythm"] == c]
        stats = {}
        for part in ("test", "validation", "train"):
            r = rows[rows["split"] == part]
            stats[f"{part}_patients"], stats[f"{part}_segments"] = int(r["subjectid"].nunique()), len(r)
        if stats["test_patients"] >= min_test_patients:
            scored.append(c)
        else:
            not_scored[c] = stats
    return scored, not_scored


def alignment_offsets(data_dir: Path, caseids) -> dict[int, float | None]:
    """Median annotation-to-ECG offset (ms) per case from the ECG cache's QC sidecars (`qc.median_offset_ms`);
    None where the sidecar has none (alignment not verified)."""
    out = {}
    for c in caseids:
        side = json.loads((binary.ecg_dir(data_dir) / f"{int(c)}.json").read_text(encoding="utf-8"))
        out[int(c)] = (side.get("qc") or {}).get("median_offset_ms")
    return out


# --- split file -------------------------------------------------------------------------------------------------


def split_table(meta: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    """caseid, subjectid, split for the database's cases under the dataset-level VitalDB split."""
    from benchtoward.tasks.vitaldb_split import caseid_split

    split = caseid_split(cases)
    df = meta[["caseid", "subjectid"]].astype(int).sort_values("caseid", ignore_index=True)
    return df.assign(split=[split[c] for c in df["caseid"]])


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_split(table: pd.DataFrame, path: Path) -> None:
    published = pd.read_csv(path)
    if not published.equals(table.reset_index(drop=True)):
        merged = table.merge(published, on="caseid", how="outer", suffixes=("", "_file"), indicator=True)
        differ = merged[(merged["_merge"] != "both") | (merged["split"] != merged["split_file"])]
        raise ValueError(f"the computed split differs from {path} for {len(differ)} case(s), e.g. "
                         f"{differ['caseid'].head(5).tolist()}; rebuild with --write-split only if the split rules "
                         "changed (that is a new task version)")


def count_mismatches(summary: dict, specs: dict | None = None) -> list[str]:
    """What a build computed that differs from the specs (empty: identical): scored instances per split and class
    (`counts`), the rhythm classes scored and not scored, and the cases whose beats are left out for alignment
    (`left_out`: offset too large; `offset_unknown`: no offset in the QC sidecar, none in the spec today)."""
    specs = specs or {"rhythm": load_spec(RHYTHM_SLUG), "beat": load_spec(BEAT_SLUG)}
    out = []
    for task, spec in specs.items():
        for part, expected in spec["counts"].items():
            built = summary[task][part]["label"]
            if built != expected:
                out.append(f"{task} {part}: built {built}, spec {expected}")
    label, classes = specs["rhythm"]["label"], summary["rhythm_classes"]
    if classes["scored"] != label["scored_classes"]:
        out.append(f"rhythm scored classes: built {classes['scored']}, spec {label['scored_classes']}")
    if classes["not_scored"] != label["classes_not_scored"]:
        out.append(f"rhythm classes not scored: built {classes['not_scored']}, spec {label['classes_not_scored']}")
    for key in ("left_out", "offset_unknown"):
        built = summary["beat_alignment"].get(key) or {}
        expected = {str(c): v for c, v in (specs["beat"]["alignment"].get(key) or {}).items()}
        if built != expected:
            out.append(f"beat alignment {key}: built {built}, spec {expected}")
    return out


# --- ECG --------------------------------------------------------------------------------------------------------


def segment_ecg(segments: pd.DataFrame, data_dir: Path) -> np.ndarray:
    """ECG arrays (n, 5 s * 500 Hz) float32 for segments, in row order, from the binary task's ECG cache."""
    spec = load_spec(RHYTHM_SLUG)
    hz, n = spec["input"]["hz"], spec["input"]["duration_s"] * spec["input"]["hz"]
    out = np.full((len(segments), n), np.nan, dtype=np.float32)
    positions = np.arange(len(segments))
    for caseid, idx in pd.Series(positions).groupby(segments["caseid"].to_numpy()):
        x, first = binary._cached(data_dir, caseid)
        for pos in idx.to_numpy():
            a = (int(segments["start_s"].iat[pos]) - first) * hz
            out[pos] = x[a:a + n]
    return out


# --- build ------------------------------------------------------------------------------------------------------


def _counts(series: pd.Series, order) -> dict[str, int]:
    counts = series.value_counts()
    return {str(k): int(counts[k]) for k in order if k in counts.index}


def build_task(data_dir: Path, cases: pd.DataFrame | None = None, split_path: Path | None = None,
               write_split: bool = False, min_test_patients: int | None = None) -> dict[str, Path]:
    """Shared segments, rhythm instances and beat instances -> task_dir (segments.parquet, rhythm.parquet,
    beat.parquet, build_summary.json). The split must equal the published file unless `write_split`.

    Rhythm classes in fewer test patients than `min_test_patients` (default: the spec's) are not scored: their
    segments leave the rhythm instances in every split. Beats of cases whose annotations are offset from the ECG
    by more than the beat spec's `alignment.max_median_offset_ms` (QC sidecar), or whose sidecar has no offset
    (alignment not verified, counted apart as `offset_unknown`), are not beat instances."""
    rhythm_spec, beat_spec = load_spec(RHYTHM_SLUG), load_spec(BEAT_SLUG)
    hz, duration = rhythm_spec["input"]["hz"], rhythm_spec["input"]["duration_s"]
    classes, beat_classes = list(rhythm_spec["label"]["classes"]), list(beat_spec["label"]["classes"])
    mapping = beat_spec["label"]["mapping"]
    if min_test_patients is None:
        min_test_patients = int(rhythm_spec["label"]["min_test_patients"])
    max_offset_ms = beat_spec["alignment"]["max_median_offset_ms"]
    close_s = beat_spec["alignment"]["close_pair_ms"] / 1000
    root = task_dir(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    meta = load_metadata(data_dir).sort_values("caseid")
    if cases is None:
        cases = binary._vitaldb_cases(binary.task_dir(data_dir))
    vitaldb_subject = dict(zip(cases["caseid"].astype(int), cases["subjectid"]))
    wrong = [int(c) for c, s in zip(meta["caseid"], meta["subjectid"]) if vitaldb_subject.get(int(c)) != s]
    if wrong:
        raise ValueError(f"{len(wrong)} case(s) whose subjectid differs from VitalDB's, e.g. {wrong[:5]}")
    missing = [int(c) for c in meta["caseid"] if not (binary.ecg_dir(data_dir) / f"{int(c)}.json").exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} case(s) without cached ECG, e.g. {missing[:5]}; rerun fetch")
    table = split_table(meta, cases)
    split_path = Path(split_path) if split_path else published_split_path()
    if write_split:
        split_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(split_path, index=False)
    check_split(table, split_path)
    split = dict(zip(table["caseid"], table["split"]))
    subject = dict(zip(table["caseid"], table["subjectid"]))
    offsets = alignment_offsets(data_dir, table["caseid"])
    unverified = {c for c, v in offsets.items() if v is None}
    misaligned = {c for c, v in offsets.items() if v is not None and abs(v) > max_offset_ms}

    ann = load_annotations(data_dir)
    zero = lambda keys: {s: dict.fromkeys(keys, 0) for s in SPLITS}  # noqa: E731
    rhythm_excluded = zero(SEGMENT_REASONS + RHYTHM_REASONS)
    beat_excluded = zero(SEGMENT_REASONS + ("misaligned_annotations", "offset_unknown", "outside_grid"))
    not_scored = zero([mapping.get(b, b) for b in BEATS if mapping.get(b, b) not in beat_classes] + ["conflicting"])
    close = {"release": zero(["pairs"]), "scored": zero(["pairs"])}
    left_out: dict[str, dict] = {}       # offset above the limit
    offset_unknown: dict[str, dict] = {}  # no offset in the QC sidecar: alignment not verified
    records = []
    for caseid, rows in ann.groupby("caseid", sort=True):
        caseid = int(caseid)
        part = split[caseid]
        x, first = binary._cached(data_dir, caseid)
        segs = segment_case(rows, rhythm_spec)
        in_grid = int(segs["n_beats"].sum())
        positions = beat_positions(rows)[0]
        beat_excluded[part]["outside_grid"] += len(positions) - in_grid
        close["release"][part]["pairs"] += close_pairs(positions, close_s)
        beats_scored = caseid not in misaligned and caseid not in unverified
        for w in segs.itertuples(index=False):
            reason = None if pd.isna(w.reason) else w.reason
            if reason is None:
                a = (w.start_s - first) * hz
                if np.isnan(x[a:a + duration * hz]).mean() > rhythm_spec["segments"]["max_ecg_missing_fraction"]:
                    reason = "ecg_missing"
                elif w.bad_signal_quality and part in rhythm_spec["bad_signal_quality"]["exclude_from"]:
                    reason = "bad_signal_quality"
            if reason is not None:
                rhythm_excluded[part][reason] += 1
                beat_excluded[part][reason] += int(w.n_beats)
                continue
            rhythm_label = None if pd.isna(w.rhythm) else w.rhythm
            rhythm_reason = None if pd.isna(w.rhythm_reason) else w.rhythm_reason
            if rhythm_reason is not None:
                rhythm_excluded[part][rhythm_reason] += 1
            if beats_scored:
                for t in w.beat_types:
                    if mapping.get(t) not in beat_classes:
                        not_scored[part]["conflicting" if "|" in t else mapping.get(t, t)] += 1
            elif caseid in unverified:
                beat_excluded[part]["offset_unknown"] += int(w.n_beats)
                entry = offset_unknown.setdefault(str(caseid), {"split": part, "scored_beats_left_out": 0})
                entry["scored_beats_left_out"] += sum(mapping.get(t) in beat_classes for t in w.beat_types)
            else:
                beat_excluded[part]["misaligned_annotations"] += int(w.n_beats)
                entry = left_out.setdefault(str(caseid), {"split": part, "median_offset_ms": offsets[caseid],
                                                          "scored_beats_left_out": 0})
                entry["scored_beats_left_out"] += sum(mapping.get(t) in beat_classes for t in w.beat_types)
            records.append({"segment_id": binary.instance_id(caseid, w.start_s), "caseid": caseid,
                            "subjectid": subject[caseid], "split": part, "start_s": int(w.start_s),
                            "duration_s": duration, "rhythm": rhythm_label, "rhythm_reason": rhythm_reason,
                            "rhythms": w.rhythms, "rhythm_from_neighbours": bool(w.rhythm_from_neighbours),
                            "n_beats": int(w.n_beats), "bad_signal_quality": bool(w.bad_signal_quality),
                            "beats_scored": beats_scored, "beat_samples": list(w.beat_samples),
                            "beat_types": list(w.beat_types)})
    seg = pd.DataFrame(records).sort_values(["caseid", "start_s"], ignore_index=True)
    unknown = set(seg["rhythm"].dropna()) - set(classes)
    if unknown:
        raise ValueError(f"rhythm class(es) outside the spec: {sorted(unknown)}")
    scored, classes_not_scored = rhythm_classes(seg, classes, min_test_patients)
    dropped = seg["rhythm"].isin(list(classes_not_scored))
    for part, n in seg.loc[dropped, "split"].value_counts().items():
        rhythm_excluded[part]["class_not_scored"] += int(n)
    seg.loc[dropped, "rhythm_reason"] = "class_not_scored"
    seg.loc[dropped, "rhythm"] = None
    keep = ["caseid", "subjectid", "split", "start_s", "duration_s"]
    rhythm = (seg[seg["rhythm"].notna()].rename(columns={"segment_id": "instance_id", "rhythm": "label"})
              [["instance_id", *keep, "label", "rhythms", "rhythm_from_neighbours", "n_beats",
                "bad_signal_quality"]].reset_index(drop=True))
    beats = []
    for s in seg[seg["beats_scored"]].itertuples(index=False):
        in_af = "AFIB/AFL" in s.rhythms.split("|")
        kept = []
        for sample, t in zip(s.beat_samples, s.beat_types):
            label = mapping.get(t)
            if label in beat_classes:
                kept.append(sample)
                beats.append({"instance_id": beat_instance_id(s.segment_id, sample), "segment_id": s.segment_id,
                              "caseid": s.caseid, "subjectid": s.subjectid, "split": s.split, "start_s": s.start_s,
                              "sample": int(sample), "label": label, "rhythms": s.rhythms, "in_af_segment": in_af})
        close["scored"][s.split]["pairs"] += close_pairs(np.asarray(kept) / hz, close_s)
    columns = ["instance_id", "segment_id", "caseid", "subjectid", "split", "start_s", "sample", "label", "rhythms",
               "in_af_segment"]
    beat = pd.DataFrame(beats, columns=columns)
    if not beat["instance_id"].is_unique:
        raise ValueError("two annotated beats share a sample in one segment")

    paths = {"segments": root / "segments.parquet", "rhythm": root / "rhythm.parquet", "beat": root / "beat.parquet",
             "summary": root / "build_summary.json"}
    seg.to_parquet(paths["segments"], index=False)
    rhythm.to_parquet(paths["rhythm"], index=False)
    beat.to_parquet(paths["beat"], index=False)
    summary = {"split_file": str(split_path.name), "split_sha256": file_sha256(split_path),
               "segments": {s: int((seg["split"] == s).sum()) for s in SPLITS},
               "rhythm_classes": {"min_test_patients": min_test_patients, "scored": scored,
                                  "not_scored": classes_not_scored},
               "beat_alignment": {"max_median_offset_ms": max_offset_ms, "left_out": left_out,
                                  "offset_unknown": offset_unknown},
               "close_beat_pairs": {k: {p: v[p]["pairs"] for p in SPLITS} for k, v in close.items()},
               "rhythm": {}, "beat": {}}
    for part in SPLITS:
        r, b = rhythm[rhythm["split"] == part], beat[beat["split"] == part]
        summary["rhythm"][part] = {"instances": len(r), "subjects": int(r["subjectid"].nunique()),
                                   "cases": int(r["caseid"].nunique()), "label": _counts(r["label"], classes),
                                   "label_subjects": {c: int(r.loc[r["label"] == c, "subjectid"].nunique())
                                                      for c in classes if (r["label"] == c).any()},
                                   "from_neighbours": int(r["rhythm_from_neighbours"].sum()),
                                   "excluded": rhythm_excluded[part]}
        summary["beat"][part] = {"instances": len(b), "subjects": int(b["subjectid"].nunique()) if len(b) else 0,
                                 "cases": int(b["caseid"].nunique()) if len(b) else 0,
                                 "label": _counts(b["label"], beat_classes) if len(b) else {},
                                 "S_in_af_segments": int(((b["label"] == "S") & b["in_af_segment"]).sum()),
                                 "not_scored": not_scored[part], "excluded": beat_excluded[part]}
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return paths
