"""vitaldb-arrhythmia-abnormal@0: segmentation, labels, split, ECG cache and build (no network)."""
import json

import numpy as np
import pandas as pd
import pytest

from medbench.tasks import vitaldb_arrhythmia_abnormal as task
from medbench.tasks.vitaldb_split import caseid_split

HZ = 500


def case_rows(*parts, start=0.5, rr=1.0, beats=None):
    """Annotation rows for one case: `parts` are (rhythm_label, seconds); one row every `rr` s from `start`.
    Beat types: N for normal sinus rhythm, S for AFIB/AFL, V for VT, none for Noise, else N."""
    default_beat = {"N": "N", "AFIB/AFL": "S", "VT": "V", "Noise": None}
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


def windows(df):
    out = task.segment_case(df)
    return out.set_index("start_s")


def test_spec_follows_the_dataset_paper():
    spec = task.load_spec()
    assert spec["slug"] == "vitaldb-arrhythmia-abnormal" and spec["kind"] == "binary"
    assert spec["primary_metric"] == "auroc" and spec["input"]["duration_s"] == 5 and spec["input"]["hz"] == HZ
    assert "10.1038/s41597-026-07076-8" in spec["protocol"]["source"]
    assert len(spec["protocol"]["annotations"]["commit"]) == 40
    assert spec["label"]["normal_context_s"] == 45 and spec["label"]["not_scored"] == ["Noise"]
    assert spec["split"]["seed"] == 20261029 and spec["split"]["fractions"] == [0.7, 0.1, 0.2]
    assert isinstance(spec["deviations"], list) and spec["deviations"]


def test_long_normal_sinus_rhythm_is_normal_on_a_5s_grid():
    w = windows(case_rows(("N", 60)))   # rows at 0.5 .. 59.5
    assert list(w.index) == list(range(5, 55, 5))
    assert (w["label"] == 0).all() and w["reason"].isna().all()
    assert (w["rhythm"] == "N").all() and (w["n_beats"] == 5).all()


def test_any_other_rhythm_label_makes_a_segment_abnormal_and_short_nsr_is_not_scored():
    w = windows(case_rows(("AFIB/AFL", 20), ("N", 30), ("AFIB/AFL", 20)))
    assert list(w.index) == list(range(5, 65, 5))
    assert list(w.loc[[5, 10, 15, 20, 45, 50, 55, 60], "label"]) == [1] * 8
    assert set(w.loc[[15, 20, 45], "rhythms"]) == {"AFIB/AFL|N"}   # transition segments count as abnormal
    assert (w.loc[[15, 20], "rhythm"] == "AFIB/AFL").all()
    short = w.loc[[25, 30, 35, 40]]
    assert short["label"].isna().all() and (short["reason"] == "short_nsr").all()


def test_normal_needs_45_seconds_of_uninterrupted_nsr():
    w = windows(case_rows(("VT", 5), ("N", 45.0), ("VT", 5)))       # N rows 5.5 .. 49.5: span 44 s
    assert (w.loc[w["rhythms"] == "N", "reason"] == "short_nsr").all()
    w = windows(case_rows(("VT", 5), ("N", 46.0), ("VT", 5)))       # span 45 s
    assert (w.loc[w["rhythms"] == "N", "label"] == 0).all()


def test_nsr_segments_with_premature_beats_are_omitted():
    w = windows(case_rows(("N", 60), beats={12.5: "V", 31.5: "S", 41.5: "U"}))
    assert w.loc[10, "reason"] == "premature_beats" and w.loc[30, "reason"] == "premature_beats"
    assert w.loc[40, "label"] == 0   # unclassifiable beats are not premature beats
    assert (w.drop(index=[10, 30])["label"] == 0).all()


def test_noise_is_not_scored_including_segments_bordering_it():
    w = windows(case_rows(("N", 60), ("Noise", 10), ("N", 60)))   # noise rows 60.5 .. 69.5
    noisy = w[w["reason"] == "noise"]
    assert list(noisy.index) == [55, 60, 65, 70]   # 55 and 70 border noise rows (59.5 -> 60.5, 69.5 -> 70.5)
    assert w.loc[50, "label"] == 0 and w.loc[75, "label"] == 0
    assert "Noise" not in set(w["rhythm"].dropna())


def test_annotation_gaps_are_not_scored_and_break_nsr_runs():
    df = pd.concat([case_rows(("N", 60)), case_rows(("N", 60), start=80.5)], ignore_index=True)
    w = windows(df)
    assert set(w.loc[[55, 60, 65, 70, 75, 80], "reason"]) == {"annotation_gap"}
    assert w.loc[50, "label"] == 0 and w.loc[85, "label"] == 0


def test_short_pauses_are_kept():
    df = pd.concat([case_rows(("SND", 32)), case_rows(("SND", 30), start=38.5)], ignore_index=True)  # 7 s pause
    w = windows(df)
    assert w["reason"].isna().all() and (w["label"] == 1).all() and (w["rhythm"] == "SND").all()
    empty = pd.concat([case_rows(("SND", 30)), case_rows(("SND", 30), start=36.5)], ignore_index=True)
    assert windows(empty).loc[30, "reason"] == "no_beats"   # a pause that holds a whole segment


def test_bad_signal_quality_flag_by_overlap():
    df = case_rows(("N", 60))
    df.loc[df["time_second"].between(20.4, 22.6), "bad_signal_quality"] = True
    w = windows(df)
    assert list(w.index[w["bad_signal_quality"]]) == [20]
    assert w.loc[20, "label"] == 0   # the flag does not change the label; the builder decides per split


def test_marker_rows_are_not_beats():
    df = case_rows(("N", 60))
    marker = {"time_second": 12.2, "beat_type": None, "rhythm_label": None, "bad_signal_quality": True,
              "bad_signal_quality_label": "Start1"}
    df = pd.concat([df, pd.DataFrame([marker])], ignore_index=True).sort_values("time_second")
    w = windows(df)
    assert w.loc[10, "n_beats"] == 5 and w.loc[10, "bad_signal_quality"]


def test_unknown_rhythm_label_raises():
    with pytest.raises(ValueError, match="unknown rhythm label"):
        task.segment_case(case_rows(("Torsades", 30)))


# --- ECG cache --------------------------------------------------------------------------------------------------


def synthetic_ecg(n_seconds, beat_times=(), offset_s=0.0, nan_spans=()):
    t = np.arange(int(n_seconds * HZ)) / HZ
    x = np.random.default_rng(0).normal(0, 0.01, len(t))
    for b in beat_times:
        x += np.exp(-0.5 * ((t - b - offset_s) / 0.012) ** 2)
    for a, b in nan_spans:
        x[int(a * HZ):int(b * HZ)] = np.nan
    return x.astype(np.float32)


def write_dataset(data_dir, cases: dict[int, pd.DataFrame], subjects: dict[int, int]):
    root = task.dataset_dir(data_dir)
    (root / "Annotation_Files").mkdir(parents=True)
    meta = []
    for caseid, df in cases.items():
        df.to_csv(root / "Annotation_Files" / f"Annotation_file_{caseid}.csv", index=False)
        meta.append({"case_id": caseid, "analysis_start_time_sec": df["time_second"].min(),
                     "analysis_end_time_sec": df["time_second"].max(), "subjectid": subjects[caseid]})
    pd.DataFrame(meta).to_csv(root / "metadata.csv", index=False)


class FakeLoader:
    def __init__(self, signals):
        self.signals, self.calls = signals, []

    def __call__(self, caseid):
        self.calls.append(caseid)
        value = self.signals[caseid]
        return value() if callable(value) else value


def test_alignment_offset_detects_a_shift():
    beats = np.arange(1.0, 19.0, 0.9)
    assert task.alignment_offset_ms(synthetic_ecg(20, beats), 0.0, beats) == pytest.approx(0, abs=4)
    assert task.alignment_offset_ms(synthetic_ecg(20, beats, offset_s=0.1), 0.0, beats) == pytest.approx(100, abs=4)


def test_fetch_ecg_caches_only_the_annotated_span_and_resumes(tmp_path):
    df = case_rows(("N", 60), start=12.5)      # 12.5 .. 71.5 -> cached span [10, 75)
    write_dataset(tmp_path, {7: df}, {7: 70})
    loader = FakeLoader({7: synthetic_ecg(100, df["time_second"])})
    summary = task.fetch_ecg(tmp_path, workers=1, load=loader)
    x = np.load(task.ecg_dir(tmp_path) / "7.npy")
    side = json.loads((task.ecg_dir(tmp_path) / "7.json").read_text())
    assert x.dtype == np.float32 and len(x) == 65 * HZ and side["start_s"] == 10 and side["hz"] == HZ
    assert np.array_equal(x, loader.signals[7][10 * HZ:75 * HZ])
    assert abs(side["qc"]["median_offset_ms"]) <= 4 and summary["fetched"] == 1
    assert task.fetch_ecg(tmp_path, workers=1, load=loader)["fetched"] == 0 and loader.calls == [7]


def test_fetch_ecg_retries_short_downloads_and_reports_failures(tmp_path):
    df = case_rows(("N", 60))
    write_dataset(tmp_path, {1: df, 2: df}, {1: 10, 2: 20})
    attempts = iter([synthetic_ecg(30), synthetic_ecg(30), synthetic_ecg(30)])
    loader = FakeLoader({1: synthetic_ecg(80, df["time_second"]), 2: lambda: next(attempts)})
    with pytest.raises(RuntimeError, match="1 case"):
        task.fetch_ecg(tmp_path, workers=1, load=loader)
    assert (task.ecg_dir(tmp_path) / "1.npy").exists() and not (task.ecg_dir(tmp_path) / "2.npy").exists()
    assert loader.calls.count(2) == task.LOAD_ATTEMPTS


# --- build ------------------------------------------------------------------------------------------------------


def vitaldb_cases(n=300):
    return pd.DataFrame({"caseid": range(1, n + 1), "subjectid": [1000 + i // 2 for i in range(n)]})


def pick_cases(cases, wanted=("train", "test")):
    split = caseid_split(cases)
    return {s: next(c for c in sorted(split) if split[c] == s) for s in wanted}


def build_fixture(tmp_path, cases, bad_in=(), nan_in=None):
    picked = pick_cases(cases)
    subjects = dict(zip(cases["caseid"], cases["subjectid"]))
    annotations, signals = {}, {}
    for caseid in picked.values():
        df = case_rows(("N", 60), ("AFIB/AFL", 30), ("Noise", 10), ("N", 20))   # rows 0.5 .. 119.5
        if caseid in bad_in:
            df.loc[df["time_second"].between(20.4, 22.6), "bad_signal_quality"] = True
        annotations[caseid] = df
        signals[caseid] = synthetic_ecg(130, df["time_second"], nan_spans=[nan_in] if nan_in else ())
    write_dataset(tmp_path, annotations, {c: subjects[c] for c in annotations})
    task.fetch_ecg(tmp_path, workers=1, load=FakeLoader(signals))
    return picked


def test_build_uses_the_dataset_level_split_and_stable_ids(tmp_path):
    cases = vitaldb_cases()
    picked = build_fixture(tmp_path, cases)
    df = pd.read_parquet(task.build_task(tmp_path, cases=cases))
    expected = caseid_split(cases)
    assert all(expected[c] == s for c, s in zip(df["caseid"], df["split"]))
    assert set(df["split"]) == {"train", "test"} and df["instance_id"].is_unique
    first = df[df["caseid"] == picked["train"]].iloc[0]
    assert first["instance_id"] == f"arr-{picked['train']}-5" and first["start_s"] == 5 and first["duration_s"] == 5
    assert set(df["label"]) == {0, 1} and df["label"].dtype.kind == "i"
    assert {"rhythm", "rhythms", "n_beats", "bad_signal_quality", "subjectid"} <= set(df.columns)
    again = pd.read_parquet(task.build_task(tmp_path, cases=cases.sample(frac=1, random_state=1)))
    assert df.equals(again)


def test_build_excludes_bad_quality_from_test_only(tmp_path):
    cases = vitaldb_cases()
    picked = pick_cases(cases)
    build_fixture(tmp_path, cases, bad_in=tuple(picked.values()))
    df = pd.read_parquet(task.build_task(tmp_path, cases=cases))
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    assert train["bad_signal_quality"].sum() == 1 and 20 in set(train["start_s"])
    assert not test["bad_signal_quality"].any() and 20 not in set(test["start_s"])
    summary = json.loads((task.task_dir(tmp_path) / "build_summary.json").read_text())
    assert summary["excluded"]["test"]["bad_signal_quality"] == 1


def test_build_excludes_segments_with_missing_ecg(tmp_path):
    cases = vitaldb_cases()
    build_fixture(tmp_path, cases, nan_in=(30.0, 31.0))   # 1 s of 5 s missing in segment 30
    df = pd.read_parquet(task.build_task(tmp_path, cases=cases))
    assert 30 not in set(df["start_s"]) and 35 in set(df["start_s"])
    summary = json.loads((task.task_dir(tmp_path) / "build_summary.json").read_text())
    assert summary["excluded"]["train"]["ecg_missing"] == 1


def test_build_refuses_missing_ecg_and_subject_mismatch(tmp_path):
    cases = vitaldb_cases()
    picked = build_fixture(tmp_path, cases)
    (task.ecg_dir(tmp_path) / f"{picked['test']}.json").unlink()
    with pytest.raises(RuntimeError, match="rerun fetch"):
        task.build_task(tmp_path, cases=cases)
    other = tmp_path / "other"
    build_fixture(other, cases)
    wrong = cases.copy()
    wrong.loc[wrong["caseid"] == picked["train"], "subjectid"] = 99999
    with pytest.raises(ValueError, match="subjectid"):
        task.build_task(other, cases=wrong)


def test_materialise_returns_the_cached_windows(tmp_path):
    cases = vitaldb_cases()
    picked = build_fixture(tmp_path, cases)
    df = pd.read_parquet(task.build_task(tmp_path, cases=cases))
    x = task.materialise(df, tmp_path)
    assert x.shape == (len(df), 5 * HZ) and x.dtype == np.float32
    cache = np.load(task.ecg_dir(tmp_path) / f"{picked['train']}.npy")   # cached span starts at 0 s
    row = df.index[(df["caseid"] == picked["train"]) & (df["start_s"] == 10)][0]
    assert np.array_equal(x[row], cache[10 * HZ:15 * HZ])


# --- registry and CLI -------------------------------------------------------------------------------------------


def test_registry_has_the_task_and_baseline_and_no_coming_soon_card(tmp_session):
    import yaml

    from conftest import REGISTRY_DIR
    from medbench.db import Model
    from medbench.registry import load_registry, resolve_task

    load_registry(tmp_session, REGISTRY_DIR)
    row = resolve_task(tmp_session, "vitaldb-arrhythmia-abnormal")
    assert (row.version, row.kind, row.primary_metric, row.dataset_slug, row.preview) == \
        (0, "binary", "auroc", "vitaldb-arrhythmia", True)
    model = tmp_session.get(Model, "medbench/arrhythmia-hgb-rr")
    assert (model.type, model.provider, model.developer) == ("baseline", "medbench", "medbench.kr")
    planned = yaml.safe_load((REGISTRY_DIR / "planned.yaml").read_text(encoding="utf-8"))
    assert not any("arrhythmia" in p["title"].lower() for p in planned)


def test_cli_registers_fetch_build_and_baseline():
    from typer.testing import CliRunner

    from medbench.cli import app

    runner = CliRunner()
    assert "vitaldb-arrhythmia-ecg" in runner.invoke(app, ["fetch", "--help"]).output
    assert "vitaldb-arrhythmia-abnormal" in runner.invoke(app, ["build", "--help"]).output
    assert "vitaldb-arrhythmia-abnormal" in runner.invoke(app, ["baseline", "--help"]).output
