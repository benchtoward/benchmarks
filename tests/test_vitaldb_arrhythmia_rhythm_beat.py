"""vitaldb-arrhythmia-rhythm@0 and vitaldb-arrhythmia-beat@0: shared segments, rhythm classes, beat classes, split
file and build (no network)."""
import hashlib
import json
import re

import numpy as np
import pandas as pd
import pytest

from conftest import REGISTRY_DIR
from benchtoward.tasks import vitaldb_arrhythmia_abnormal as binary
from benchtoward.tasks import vitaldb_arrhythmia_rhythm_beat as rb
from benchtoward.tasks.vitaldb_split import caseid_split

HZ = 500
RHYTHMS = ["N", "AFIB/AFL", "AVB", "Patterned Atrial Ectopy", "Patterned Ventricular Ectopy", "SND", "SVTA", "VT",
           "WAP/MAT"]


def case_rows(*parts, start=0.5, rr=1.0, beats=None):
    """Annotation rows for one case: `parts` are (rhythm_label, seconds); one row every `rr` s from `start`.
    Beat types: S for AFIB/AFL, V for VT, none for Noise, else N; `beats` overrides single rows by time."""
    default_beat = {"AFIB/AFL": "S", "VT": "V", "Noise": None}
    rows, t = [], start
    for label, seconds in parts:
        for _ in range(int(round(seconds / rr))):
            rows.append({"time_second": round(t, 6), "beat_type": default_beat.get(label, "N"), "rhythm_label": label,
                         "bad_signal_quality": False, "bad_signal_quality_label": None})
            t += rr
    df = pd.DataFrame(rows)
    for time_second, beat in (beats or {}).items():
        df.loc[np.isclose(df["time_second"], time_second), "beat_type"] = beat
    return df


def segments(df):
    return rb.segment_case(df).set_index("start_s")


# --- specs ------------------------------------------------------------------------------------------------------


def test_specs_are_previews_with_the_adopted_standards():
    rhythm, beat = rb.load_spec(rb.RHYTHM_SLUG), rb.load_spec(rb.BEAT_SLUG)
    for spec in (rhythm, beat):
        assert spec["version"] == 0 and "preview" not in spec
        assert spec["visibility"] in ("hidden", "open")   # hidden until the maintainers publish both
        assert spec["kind"] == "multiclass" and spec["primary_metric"] == "macro_f1"
        assert spec["dataset"] == "vitaldb-arrhythmia" and spec["input"]["duration_s"] == 5
        assert spec["input"]["hz"] == HZ
        assert spec["segments"]["grid_s"] == 5 and spec["metrics"]["bootstrap"] == {"n": 1000, "by": "subjectid"}
        assert "drawn again" in spec["metrics"]["bootstrap_classes"] and spec["decisions"] and spec["leakage_checks"]
        assert spec["split"]["seed"] == 20261029 and spec["split"]["fractions"] == [0.7, 0.1, 0.2]
        assert len(spec["protocol"]["annotations"]["commit"]) == 40 and spec["deviations"] and spec["one_pass"]
    assert list(rhythm["label"]["classes"]) == RHYTHMS
    assert "10.22489/CinC.2017.065-469" in rhythm["protocol"]["source"]
    assert "10.1038/s41597-026-07076-8" in rhythm["protocol"]["dataset_paper"]
    assert "owner, 2026-10-30" in rhythm["label"]["not_scored"]["Unclassifiable"]
    assert rhythm["input"]["beat_positions"] == "not given" and "without the beat positions" in rhythm["one_pass"]
    assert "rhythm_beat_positions_only" in rhythm["leakage_checks"]
    assert {"sveb_sensitivity_outside_af", "sveb_ppv_outside_af"} <= set(beat["metrics"]["secondary"])
    assert "91.6%" in " ".join(beat["deviations"]) and "14 pairs" in beat["input"]["beat_positions"]
    assert "10.1109/TBME.2004.827359" in beat["protocol"]["source"] and "EC57" in beat["protocol"]["standard"]
    assert beat["label"]["mapping"] == {"N": "N", "S": "S", "V": "V", "U": "Q"} and list(beat["label"]["classes"]) == \
        ["N", "S", "V"]
    assert set(beat["label"]["not_scored"]) == {"Q", "P", "conflicting"}


def _texts(value):
    """Every string in a loaded spec (keys and values), for checks on its wording."""
    if isinstance(value, dict):
        return [t for k, v in value.items() for t in (*_texts(k), *_texts(v))]
    if isinstance(value, list):
        return [t for v in value for t in _texts(v)]
    return [value] if isinstance(value, str) else []


def test_rhythm_spec_wording_counts_the_scored_classes_not_the_labelled_ones():
    """The database labels nine rhythm classes and the task scores eight (AVB is in too few test patients): the
    spec's title and text must not call the nine classes scored (review of PR #89)."""
    words = {"seven": 7, "eight": 8, "nine": 9, "ten": 10}
    spec = rb.load_spec(rb.RHYTHM_SLUG)
    scored = len(spec["label"]["scored_classes"])
    assert (scored, len(spec["label"]["classes"])) == (8, 9)
    title = re.search(r"\(lead II, (\w+) rhythms\)", spec["title"])
    assert title and words[title.group(1)] == scored
    text = " ".join(_texts(spec))
    claims = re.findall(r"\b(seven|eight|nine|ten) scored\b", text)
    assert claims and {words[w] for w in claims} == {scored}
    assert re.search(r"\bnine rhythm classes the database labels\b", text)


def test_both_tasks_share_the_segment_rules_of_the_binary_task():
    abnormal = binary.load_spec()
    for slug in (rb.RHYTHM_SLUG, rb.BEAT_SLUG):
        spec = rb.load_spec(slug)
        for key in ("grid_s", "max_annotation_gap_s", "min_beats", "max_ecg_missing_fraction"):
            assert spec["segments"][key] == abnormal["segments"][key], key
        assert spec["bad_signal_quality"] == abnormal["bad_signal_quality"]
        assert spec["split"]["file"] == rb.load_spec(rb.RHYTHM_SLUG)["split"]["file"]
        assert {k: spec["split"][k] for k in ("by", "seed", "fractions")} == \
            {k: abnormal["split"][k] for k in ("by", "seed", "fractions")}


def test_published_split_file_matches_its_checksum_and_keeps_subjects_together():
    spec = rb.load_spec(rb.RHYTHM_SLUG)
    path = REGISTRY_DIR.parent / spec["split"]["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == spec["split"]["sha256"]
    assert rb.load_spec(rb.BEAT_SLUG)["split"]["sha256"] == spec["split"]["sha256"]
    df = pd.read_csv(path)
    assert list(df.columns) == ["caseid", "subjectid", "split"] and len(df) == 482 and df["caseid"].is_unique
    assert set(df["split"]) == {"train", "validation", "test"}
    assert df.groupby("subjectid")["split"].nunique().max() == 1
    assert list(df["caseid"]) == sorted(df["caseid"])


# --- segments and rhythm classes --------------------------------------------------------------------------------


def test_normal_segments_carry_their_beat_positions():
    w = segments(case_rows(("N", 60), beats={12.5: "U", 41.5: "P"}))   # rows at 0.5 .. 59.5
    assert list(w.index) == list(range(5, 55, 5))
    assert w["reason"].isna().all() and (w["rhythm"] == "N").all() and w["rhythm_reason"].isna().all()
    assert list(w.loc[5, "beat_samples"]) == [250, 750, 1250, 1750, 2250]
    assert list(w.loc[10, "beat_types"]) == ["N", "N", "U", "N", "N"]   # U and P stay as positions
    assert list(w.loc[40, "beat_types"]) == ["N", "P", "N", "N", "N"] and (w["n_beats"] == 5).all()


def test_a_segment_with_one_other_label_takes_it_and_normal_needs_no_45_s_context():
    w = segments(case_rows(("AFIB/AFL", 20), ("N", 30), ("AFIB/AFL", 20)))   # AF 0.5-19.5, N 20.5-49.5, AF 50.5-
    assert list(w.loc[[5, 10, 15, 50, 55, 60], "rhythm"]) == ["AFIB/AFL"] * 6
    assert (w.loc[[20, 25, 30, 35, 40, 45], "rhythm"] == "N").all()   # short runs of N are N here (binary: not scored)
    w = segments(case_rows(("N", 30), ("VT", 3), ("N", 30), beats={45.5: "V"}))   # VT rows 30.5 .. 32.5
    assert w.loc[30, "rhythm"] == "VT" and w.loc[30, "rhythms"] == "N|VT"
    assert w.loc[45, "rhythm"] == "N"   # an isolated ventricular beat does not change the rhythm label


def test_the_rhythm_class_comes_from_the_labelled_rows_inside_the_segment():
    w = segments(case_rows(("N", 30), ("VT", 3), ("N", 30)))   # N 25.5-29.5 inside [25, 30); VT from 30.5
    assert w.loc[25, "rhythm"] == "N" and w.loc[25, "rhythms"] == "N"   # the VT row after the end is not used
    w = segments(case_rows(("AFIB/AFL", 20), ("N", 30)))   # AF up to 19.5: [20, 25) is all N inside
    assert w.loc[20, "rhythm"] == "N" and w.loc[15, "rhythm"] == "AFIB/AFL"
    assert not w["rhythm_from_neighbours"].any()


def test_without_a_labelled_row_inside_the_class_comes_from_the_neighbouring_rows():
    df = case_rows(("N", 15), ("VT", 3), ("N", 15))   # N 0.5-14.5, VT 15.5-17.5
    df.loc[df["time_second"].between(10.4, 14.6), "rhythm_label"] = None   # beats without a rhythm label
    w = segments(df)
    assert w.loc[10, "n_beats"] == 5 and w.loc[10, "rhythm"] == "VT"     # neighbours 9.5 (N) and 15.5 (VT)
    assert w.loc[10, "rhythms"] == "N|VT" and bool(w.loc[10, "rhythm_from_neighbours"])
    assert not w.drop(index=10)["rhythm_from_neighbours"].any()


def test_unclassifiable_and_mixed_segments_are_not_scored_for_rhythm_but_keep_their_beats():
    w = segments(case_rows(("N", 30), ("Unclassifiable", 10), ("N", 30)))   # rows 30.5 .. 39.5
    unclassifiable = w[w["rhythm_reason"] == "unclassifiable"]
    assert list(unclassifiable.index) == [30, 35] and unclassifiable["rhythm"].isna().all()   # rows inside only
    assert unclassifiable["reason"].isna().all() and (unclassifiable["n_beats"] > 0).all()
    assert w.loc[25, "rhythm"] == "N" and w.loc[40, "rhythm"] == "N"
    w = segments(case_rows(("Patterned Atrial Ectopy", 18), ("SND", 20)))     # switch at 17.5 -> 18.5
    assert w.loc[15, "rhythm_reason"] == "mixed" and pd.isna(w.loc[15, "rhythm"])
    assert w.loc[15, "rhythms"] == "Patterned Atrial Ectopy|SND"
    assert w.loc[10, "rhythm"] == "Patterned Atrial Ectopy" and w.loc[20, "rhythm"] == "SND"


def test_noise_gaps_and_empty_segments_are_left_out_of_both_tasks():
    w = segments(case_rows(("N", 60), ("Noise", 10), ("N", 60)))
    assert list(w.index[w["reason"] == "noise"]) == [55, 60, 65, 70]
    assert w.loc[w["reason"] == "noise", "rhythm"].isna().all()
    df = pd.concat([case_rows(("N", 60)), case_rows(("N", 60), start=80.5)], ignore_index=True)
    assert set(segments(df).loc[[55, 60, 65, 70, 75, 80], "reason"]) == {"annotation_gap"}
    empty = pd.concat([case_rows(("SND", 30)), case_rows(("SND", 30), start=36.5)], ignore_index=True)
    assert segments(empty).loc[30, "reason"] == "no_beats"


def test_a_beat_listed_twice_is_one_position_and_conflicting_types_are_kept_as_both():
    df = case_rows(("N", 30))
    twice = df[np.isclose(df["time_second"], 7.5) | np.isclose(df["time_second"], 12.5)].copy()
    twice.loc[np.isclose(twice["time_second"], 12.5), "beat_type"] = "U"   # 12.5: N and U at the same time
    w = segments(pd.concat([df, twice], ignore_index=True))
    assert list(w.loc[5, "beat_samples"]) == [250, 750, 1250, 1750, 2250] and w.loc[5, "n_beats"] == 5
    assert list(w.loc[10, "beat_types"]) == ["N", "N", "N|U", "N", "N"]


def test_unknown_rhythm_label_raises():
    with pytest.raises(ValueError, match="unknown rhythm label"):
        rb.segment_case(case_rows(("Torsades", 30)))


# --- build ------------------------------------------------------------------------------------------------------


def synthetic_ecg(n_seconds, beat_times=(), nan_spans=()):
    t = np.arange(int(n_seconds * HZ)) / HZ
    x = np.random.default_rng(0).normal(0, 0.01, len(t))
    for b in beat_times:
        x += np.exp(-0.5 * ((t - b) / 0.012) ** 2)
    for a, b in nan_spans:
        x[int(a * HZ):int(b * HZ)] = np.nan
    return x.astype(np.float32)


def write_dataset(data_dir, cases: dict[int, pd.DataFrame], subjects: dict[int, int]):
    root = binary.dataset_dir(data_dir)
    (root / "Annotation_Files").mkdir(parents=True)
    meta = []
    for caseid, df in cases.items():
        df.to_csv(root / "Annotation_Files" / f"Annotation_file_{caseid}.csv", index=False)
        meta.append({"case_id": caseid, "analysis_start_time_sec": df["time_second"].min(),
                     "analysis_end_time_sec": df["time_second"].max(), "subjectid": subjects[caseid]})
    pd.DataFrame(meta).to_csv(root / "metadata.csv", index=False)


def vitaldb_cases(n=300):
    return pd.DataFrame({"caseid": range(1, n + 1), "subjectid": [1000 + i // 2 for i in range(n)]})


def pick_cases(cases):
    split = caseid_split(cases)
    return {s: next(c for c in sorted(split) if split[c] == s) for s in ("train", "test")}


FIXTURE_BEATS = {12.5: "V", 31.5: "U", 41.5: "S", 100.5: "P"}


FIXTURE_PARTS = (("N", 60), ("VT", 10), ("Unclassifiable", 10), ("Noise", 10), ("N", 40))


def build_fixture(tmp_path, cases, bad_in=(), nan_in=None, parts=FIXTURE_PARTS, beats=None):
    """Two cases (one train, one test): N 0.5-59.5, VT 60.5-69.5, Unclassifiable 70.5-79.5, Noise 80.5-89.5,
    N 90.5-129.5, with a few V, U, S and P beats."""
    picked = pick_cases(cases)
    subjects = dict(zip(cases["caseid"], cases["subjectid"]))
    annotations, signals = {}, {}
    for caseid in picked.values():
        df = case_rows(*parts, beats=FIXTURE_BEATS if beats is None else beats)
        if caseid in bad_in:
            df.loc[df["time_second"].between(20.4, 22.6), "bad_signal_quality"] = True
        annotations[caseid] = df
        signals[caseid] = synthetic_ecg(df["time_second"].max() + 10.5, df["time_second"],
                                        nan_spans=[nan_in] if nan_in else ())
    write_dataset(tmp_path, annotations, {c: subjects[c] for c in annotations})

    def load(caseid):
        return signals[caseid]

    binary.fetch_ecg(tmp_path, workers=1, load=load)
    return picked


def build(tmp_path, cases, min_test_patients=1, **kwargs):
    """The fixture has one test patient, so every class has fewer than the spec's minimum: the tests lower it."""
    split_file = tmp_path / "split.csv"
    return rb.build_task(tmp_path, cases=cases, split_path=split_file, write_split=not split_file.exists(),
                         min_test_patients=min_test_patients, **kwargs)


def set_offset(tmp_path, caseid, offset_ms):
    path = binary.ecg_dir(tmp_path) / f"{caseid}.json"
    side = json.loads(path.read_text())
    side["qc"]["median_offset_ms"] = offset_ms
    path.write_text(json.dumps(side))


def test_build_writes_shared_segments_rhythm_and_beat_instances(tmp_path):
    cases = vitaldb_cases()
    picked = build_fixture(tmp_path, cases)
    paths = build(tmp_path, cases)
    seg, rhythm, beat = (pd.read_parquet(paths[k]) for k in ("segments", "rhythm", "beat"))
    expected = caseid_split(cases)
    assert all(expected[c] == s for c, s in zip(seg["caseid"], seg["split"])) and set(seg["split"]) == {"train", "test"}
    one = seg[seg["caseid"] == picked["train"]].set_index("start_s")
    assert list(one.index) == [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 95, 100, 105, 110, 115, 120]
    assert one.loc[5, "segment_id"] == f"arr-{picked['train']}-5" and one.loc[5, "duration_s"] == 5
    assert one.loc[55, "rhythm"] == "N" and one.loc[60, "rhythm"] == "VT" and one.loc[65, "rhythm"] == "VT"
    assert list(one.index[one["rhythm_reason"] == "unclassifiable"]) == [70]
    assert one["beats_scored"].all()
    # rhythm instances: the scored subset of the shared segments, with the same ids
    assert set(rhythm["instance_id"]) == set(seg.loc[seg["rhythm"].notna(), "segment_id"])
    assert set(rhythm["label"]) == {"N", "VT"} and rhythm["instance_id"].is_unique
    # beat instances: every N, S and V beat of every shared segment; U and P beats are positions only
    assert set(beat["label"]) == {"N", "S", "V"} and beat["instance_id"].is_unique
    assert set(beat["segment_id"]) == set(seg["segment_id"])
    v = beat[(beat["caseid"] == picked["train"]) & (beat["start_s"] == 10) & (beat["label"] == "V")].iloc[0]
    assert v["instance_id"] == f"arr-{picked['train']}-10-1250" and v["sample"] == 1250
    n_scored = sum(sum(t in ("N", "S", "V") for t in types) for types in seg["beat_types"])
    assert len(beat) == n_scored and n_scored == seg["n_beats"].sum() - 2 * 2   # minus one U and one P per case
    again = build(tmp_path, cases.sample(frac=1, random_state=1))
    assert pd.read_parquet(again["beat"]).equals(beat) and pd.read_parquet(again["segments"]).equals(seg)


def test_binary_task_segments_are_a_subset_of_the_shared_segments(tmp_path):
    cases = vitaldb_cases()
    picked = build_fixture(tmp_path, cases, bad_in=tuple(pick_cases(cases).values()))
    seg = pd.read_parquet(build(tmp_path, cases)["segments"])
    abnormal = pd.read_parquet(binary.build_task(tmp_path, cases=cases))
    assert set(abnormal["instance_id"]) <= set(seg["segment_id"])
    assert set(abnormal["split"]) == {"train", "test"} and len(picked) == 2


def test_build_summary_counts_labels_and_exclusions_per_split(tmp_path):
    cases = vitaldb_cases()
    picked = build_fixture(tmp_path, cases, bad_in=tuple(pick_cases(cases).values()))
    paths = build(tmp_path, cases)
    seg = pd.read_parquet(paths["segments"])
    test = seg[seg["split"] == "test"]
    assert 20 not in set(test["start_s"]) and 20 in set(seg.loc[seg["split"] == "train", "start_s"])
    summary = json.loads(paths["summary"].read_text())
    assert summary["rhythm"]["test"]["excluded"]["bad_signal_quality"] == 1
    assert summary["rhythm"]["train"]["excluded"]["unclassifiable"] == 1
    assert summary["rhythm"]["train"]["excluded"]["noise"] == 4
    assert summary["rhythm"]["train"]["label"] == {"N": 17, "VT": 2}
    assert summary["rhythm"]["train"]["from_neighbours"] == 0
    classes = summary["rhythm_classes"]
    assert classes["min_test_patients"] == 1 and classes["scored"] == ["N", "VT"]
    assert set(classes["not_scored"]) == set(RHYTHMS) - {"N", "VT"}   # in no test patient of the fixture
    assert summary["beat_alignment"] == {"max_median_offset_ms": 100, "left_out": {}, "offset_unknown": {}}
    assert summary["beat"]["train"]["label"] == {"N": 86, "S": 1, "V": 11}
    assert summary["beat"]["train"]["not_scored"]["Q"] == 1 and summary["beat"]["train"]["not_scored"]["P"] == 1
    assert summary["beat"]["test"]["excluded"]["bad_signal_quality"] == 5
    assert summary["beat"]["train"]["excluded"]["outside_grid"] == 5 + 5   # 0.5-4.5 and 125.5-129.5
    assert summary["rhythm"]["train"]["subjects"] == 1 and summary["beat"]["test"]["cases"] == 1
    assert picked


def test_build_checks_the_published_split_file(tmp_path):
    cases = vitaldb_cases()
    picked = build_fixture(tmp_path, cases)
    build(tmp_path, cases)
    split_file = tmp_path / "split.csv"
    written = pd.read_csv(split_file)
    assert list(written.columns) == ["caseid", "subjectid", "split"] and sorted(written["caseid"]) == \
        sorted(picked.values())
    rb.build_task(tmp_path, cases=cases, split_path=split_file)   # unchanged: accepted
    written.loc[written["caseid"] == picked["train"], "split"] = "test"
    written.to_csv(split_file, index=False)
    with pytest.raises(ValueError, match="split"):
        rb.build_task(tmp_path, cases=cases, split_path=split_file)


def test_build_excludes_segments_with_missing_ecg(tmp_path):
    cases = vitaldb_cases()
    build_fixture(tmp_path, cases, nan_in=(30.0, 31.0))
    paths = build(tmp_path, cases)
    seg = pd.read_parquet(paths["segments"])
    assert 30 not in set(seg["start_s"]) and 35 in set(seg["start_s"])
    assert json.loads(paths["summary"].read_text())["rhythm"]["train"]["excluded"]["ecg_missing"] == 1


def test_segment_ecg_returns_the_cached_windows(tmp_path):
    cases = vitaldb_cases()
    picked = build_fixture(tmp_path, cases)
    seg = pd.read_parquet(build(tmp_path, cases)["segments"])
    x = rb.segment_ecg(seg, tmp_path)
    assert x.shape == (len(seg), 5 * HZ) and x.dtype == np.float32
    cache = np.load(binary.ecg_dir(tmp_path) / f"{picked['train']}.npy")   # cached span starts at 0 s
    row = seg.index[(seg["caseid"] == picked["train"]) & (seg["start_s"] == 10)][0]
    assert np.array_equal(x[row], cache[10 * HZ:15 * HZ])


def test_spec_counts_are_checked_against_a_build():
    specs = {"rhythm": {"counts": {"train": {"N": 2}, "validation": {}, "test": {"VT": 1}}},
             "beat": {"counts": {"train": {"N": 9, "V": 1}, "validation": {}, "test": {"S": 4}}}}
    specs["rhythm"]["label"] = {"min_test_patients": 5, "scored_classes": ["VT"],
                                "classes_not_scored": {"AVB": {"test_patients": 1}}}
    specs["beat"]["alignment"] = {"max_median_offset_ms": 100,
                                  "left_out": {746: {"split": "test", "median_offset_ms": -168.0}}}
    summary = {task: {part: {"label": dict(counts)} for part, counts in spec["counts"].items()}
               for task, spec in specs.items()}
    summary["rhythm_classes"] = {"min_test_patients": 5, "scored": ["VT"], "not_scored": {"AVB": {"test_patients": 1}}}
    summary["beat_alignment"] = {"max_median_offset_ms": 100,
                                 "left_out": {"746": {"split": "test", "median_offset_ms": -168.0}}}
    assert rb.count_mismatches(summary, specs) == []
    summary["beat"]["test"]["label"] = {"S": 5}
    assert rb.count_mismatches(summary, specs) == ["beat test: built {'S': 5}, spec {'S': 4}"]
    summary["beat"]["test"]["label"] = {"S": 4}
    summary["rhythm_classes"]["scored"] = ["VT", "AVB"]
    summary["beat_alignment"]["left_out"] = {}
    assert [m.split(":")[0] for m in rb.count_mismatches(summary, specs)] == ["rhythm scored classes",
                                                                             "beat alignment left_out"]
    summary["rhythm_classes"]["scored"] = ["VT"]
    summary["beat_alignment"]["left_out"] = {"746": {"split": "test", "median_offset_ms": -168.0}}
    summary["beat_alignment"]["offset_unknown"] = {"12": {"split": "train", "scored_beats_left_out": 3}}
    assert [m.split(":")[0] for m in rb.count_mismatches(summary, specs)] == ["beat alignment offset_unknown"]
    specs["beat"]["alignment"]["offset_unknown"] = {12: {"split": "train", "scored_beats_left_out": 3}}
    assert rb.count_mismatches(summary, specs) == []   # a case without a QC offset must be listed in the spec
    real ={"rhythm": rb.load_spec(rb.RHYTHM_SLUG), "beat": rb.load_spec(rb.BEAT_SLUG)}
    for spec in real.values():
        assert set(spec["counts"]) == {"train", "validation", "test"}
        for counts in spec["counts"].values():
            assert set(counts) <= set(spec["label"]["classes"])
    rhythm_label = real["rhythm"]["label"]
    assert rhythm_label["min_test_patients"] == 5 and set(rhythm_label["classes_not_scored"]) == {"AVB"}
    assert rhythm_label["scored_classes"] == [c for c in RHYTHMS if c != "AVB"]
    assert rhythm_label["classes_not_scored"]["AVB"]["test_patients"] == 1
    assert set(real["rhythm"]["counts"]["test"]) == set(rhythm_label["scored_classes"])
    assert set(real["beat"]["counts"]["test"]) == {"N", "S", "V"}
    assert real["beat"]["alignment"]["max_median_offset_ms"] == 100
    assert {c: v["split"] for c, v in real["beat"]["alignment"]["left_out"].items()} == {746: "test", 1165: "train"}


def test_rhythm_classes_with_fewer_test_patients_than_the_minimum_are_not_scored():
    seg = pd.DataFrame({"split": ["test"] * 5 + ["train"] * 3 + ["validation"],
                        "subjectid": [1, 2, 2, 3, 3, 7, 8, 8, 9],
                        "rhythm": ["N", "N", "VT", "N", None, "VT", "AVB", "AVB", "N"]})
    scored, not_scored = rb.rhythm_classes(seg, RHYTHMS, min_test_patients=2)
    assert scored == ["N"]   # N in three test patients; VT in one; the others in none
    assert not_scored["VT"] == {"test_patients": 1, "test_segments": 1, "validation_patients": 0,
                                "validation_segments": 0, "train_patients": 1, "train_segments": 1}
    assert not_scored["AVB"]["test_patients"] == 0 and not_scored["AVB"]["train_segments"] == 2
    assert set(not_scored) == set(RHYTHMS) - {"N"}


def test_build_leaves_classes_below_the_test_patient_minimum_out_of_the_rhythm_instances(tmp_path):
    cases = vitaldb_cases()
    build_fixture(tmp_path, cases)
    paths = build(tmp_path, cases, min_test_patients=2)   # one test patient: N and VT are not scored
    seg, rhythm = pd.read_parquet(paths["segments"]), pd.read_parquet(paths["rhythm"])
    summary = json.loads(paths["summary"].read_text())
    assert summary["rhythm_classes"]["scored"] == [] and set(summary["rhythm_classes"]["not_scored"]) == \
        set(RHYTHMS)
    assert summary["rhythm_classes"]["not_scored"]["VT"] == {"test_patients": 1, "test_segments": 2,
                                                             "validation_patients": 0, "validation_segments": 0,
                                                             "train_patients": 1, "train_segments": 2}
    assert len(rhythm) == 0 and seg["rhythm"].isna().all()
    assert (seg["rhythm_reason"] == "class_not_scored").sum() == 2 * (17 + 2)
    assert summary["rhythm"]["test"]["excluded"]["class_not_scored"] == 19
    assert len(pd.read_parquet(paths["beat"])) > 0   # beats are scored as before


def test_beats_of_cases_whose_annotations_are_offset_from_the_ecg_are_not_scored(tmp_path):
    cases = vitaldb_cases()
    picked = build_fixture(tmp_path, cases)
    set_offset(tmp_path, picked["test"], -150.0)
    set_offset(tmp_path, picked["train"], 99.0)   # within 100 ms: kept
    paths = build(tmp_path, cases)
    seg, rhythm, beat = (pd.read_parquet(paths[k]) for k in ("segments", "rhythm", "beat"))
    summary = json.loads(paths["summary"].read_text())
    assert set(beat["caseid"]) == {picked["train"]}
    test_seg = seg[seg["caseid"] == picked["test"]]
    assert len(test_seg) == 20 and not test_seg["beats_scored"].any()   # the rhythm task keeps the segments
    assert seg.loc[seg["caseid"] == picked["train"], "beats_scored"].all()
    assert set(rhythm["caseid"]) == {picked["train"], picked["test"]}
    assert summary["beat_alignment"]["left_out"] == {str(picked["test"]): {"split": "test", "median_offset_ms": -150.0,
                                                                         "scored_beats_left_out": 98}}
    assert summary["beat"]["test"]["instances"] == 0
    assert summary["beat"]["test"]["excluded"]["misaligned_annotations"] == test_seg["n_beats"].sum()
    assert summary["beat"]["test"]["excluded"]["offset_unknown"] == 0
    assert summary["beat_alignment"]["offset_unknown"] == {}


@pytest.mark.parametrize("sidecar", ["null_offset", "no_qc"])
def test_beats_of_cases_without_a_known_offset_are_not_verified_and_not_scored(tmp_path, sidecar):
    """A QC sidecar without median_offset_ms does not show the annotations are aligned: the case's beats are left
    out like a misaligned case's, and counted apart from them (review of PR #89)."""
    cases = vitaldb_cases()
    picked = build_fixture(tmp_path, cases)
    path = binary.ecg_dir(tmp_path) / f"{picked['test']}.json"
    side = json.loads(path.read_text())
    if sidecar == "null_offset":
        side["qc"]["median_offset_ms"] = None
    else:
        del side["qc"]
    path.write_text(json.dumps(side))
    paths = build(tmp_path, cases)
    seg, rhythm, beat = (pd.read_parquet(paths[k]) for k in ("segments", "rhythm", "beat"))
    summary = json.loads(paths["summary"].read_text())
    assert set(beat["caseid"]) == {picked["train"]}   # neither scored nor in the baseline's training rows
    test_seg = seg[seg["caseid"] == picked["test"]]
    assert len(test_seg) == 20 and not test_seg["beats_scored"].any()
    assert picked["test"] in set(rhythm["caseid"])   # the rhythm task keeps the segments
    assert summary["beat_alignment"]["left_out"] == {}
    assert summary["beat_alignment"]["offset_unknown"] == {str(picked["test"]): {"split": "test",
                                                                                "scored_beats_left_out": 98}}
    assert summary["beat"]["test"]["excluded"]["offset_unknown"] == test_seg["n_beats"].sum()
    assert summary["beat"]["test"]["excluded"]["misaligned_annotations"] == 0
    assert summary["beat"]["test"]["instances"] == 0


def test_beats_in_atrial_fibrillation_segments_are_marked_and_counted(tmp_path):
    cases = vitaldb_cases()
    build_fixture(tmp_path, cases, parts=(("N", 30), ("AFIB/AFL", 20), ("N", 30)), beats={12.5: "S", 62.5: "V"})
    paths = build(tmp_path, cases)
    beat = pd.read_parquet(paths["beat"])
    af = beat[beat["in_af_segment"]]
    assert set(af["start_s"]) == {30, 35, 40, 45} and set(af["label"]) == {"S"}
    assert set(beat.loc[~beat["in_af_segment"], "rhythms"]) == {"N"}
    summary = json.loads(paths["summary"].read_text())
    assert summary["beat"]["train"]["S_in_af_segments"] == 20 and summary["beat"]["train"]["label"]["S"] == 21


def test_close_beat_pairs_are_counted():
    times = np.array([1.0, 1.05, 2.0, 3.0, 3.099, 3.2])
    assert rb.close_pairs(times, 0.1) == 2   # 1.0-1.05 and 3.0-3.099; 3.099-3.2 is 101 ms


# --- registry and CLI -------------------------------------------------------------------------------------------


def test_registry_loads_both_tasks_and_the_baseline_model(tmp_session):
    from benchtoward.db import Model
    from benchtoward.registry import load_registry, resolve_task

    load_registry(tmp_session, REGISTRY_DIR)
    for slug in (rb.RHYTHM_SLUG, rb.BEAT_SLUG):
        row = resolve_task(tmp_session, f"{slug}@0", include_hidden=True)
        assert (row.kind, row.primary_metric, row.preview) == ("multiclass", "macro_f1", False)
        if row.visibility == "hidden":
            with pytest.raises(LookupError):
                resolve_task(tmp_session, slug)
    model = tmp_session.get(Model, "benchtoward/arrhythmia-hgb-rr-morph")
    assert (model.type, model.provider, model.developer) == ("baseline", "benchtoward", "BenchToward")


def test_cli_registers_build_and_baseline():
    from typer.testing import CliRunner

    from benchtoward.cli import app

    runner = CliRunner()
    assert "vitaldb-arrhythmia-rhythm-beat" in runner.invoke(app, ["build", "--help"]).output
    assert "vitaldb-arrhythmia-rhythm-beat" in runner.invoke(app, ["baseline", "--help"]).output
