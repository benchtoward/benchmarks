"""Commands for vitaldb-arrhythmia-rhythm@0 and vitaldb-arrhythmia-beat@0: one build and one baseline pass for both.

    benchtoward fetch vitaldb-arrhythmia && benchtoward fetch vitaldb-arrhythmia-ecg   # as for vitaldb-arrhythmia-abnormal
    benchtoward build vitaldb-arrhythmia-rhythm-beat                 # checks the published split file
    benchtoward baseline vitaldb-arrhythmia-rhythm-beat --report out.json   # also validation and leakage checks
"""
import json
from pathlib import Path

import typer

from benchtoward.config import get_settings

NAME = "vitaldb-arrhythmia-rhythm-beat"


def build_cmd(write_split: bool = typer.Option(False, help="write registry/splits/ from the dataset-level VitalDB "
                                               "split instead of checking it (a new task version only)")) -> None:
    """Build the shared 5-second segments, rhythm instances and beat instances (dataset-level VitalDB split)."""
    from benchtoward.tasks.vitaldb_arrhythmia_rhythm_beat import build_task, count_mismatches

    try:
        paths = build_task(get_settings().data_dir, write_split=write_split)
    except (RuntimeError, ValueError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(1)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    typer.echo(f"split {summary['split_file']} sha256 {summary['split_sha256']}; segments {summary['segments']}")
    for task in ("rhythm", "beat"):
        for part in ("train", "validation", "test"):
            s = summary[task][part]
            excluded = {k: v for k, v in s["excluded"].items() if v}
            extra = f", not scored {s['not_scored']}" if "not_scored" in s else ""
            typer.echo(f"{task} {part}: {s['instances']} instances, {s['subjects']} subjects, labels {s['label']}"
                       f"{extra}; left out {excluded}")
    classes = summary["rhythm_classes"]
    typer.echo(f"rhythm classes scored (>= {classes['min_test_patients']} test patients): {classes['scored']}; "
               f"not scored: {classes['not_scored']}")
    alignment = summary["beat_alignment"]
    typer.echo(f"beats left out for annotation offset > {alignment['max_median_offset_ms']} ms: "
               f"{alignment['left_out']}; for no offset in the QC sidecar: {alignment['offset_unknown']}; "
               f"close beat pairs {summary['close_beat_pairs']}")
    mismatches = count_mismatches(summary)
    for line in mismatches:
        typer.echo(f"differs from the spec's counts: {line}")
    if mismatches and not write_split:
        raise typer.Exit(1)


def _features(paths: dict, data_dir: Path, segments, rhythm, beat) -> dict:
    """One-pass features, cached next to the instances and reused while the instance lists are unchanged."""
    import hashlib

    import numpy as np

    from benchtoward.baselines.arrhythmia_rhythm_beat import FEATURE_VERSION, one_pass_features

    key = hashlib.sha256("\n".join([*rhythm["instance_id"], *beat["instance_id"]]).encode()).hexdigest()
    cache = paths["segments"].parent / f"features-{FEATURE_VERSION}.npz"
    if cache.exists():
        stored = np.load(cache)
        if str(stored["key"]) == key:
            return {"rhythm": stored["rhythm"], "beat": stored["beat"]}
    features = one_pass_features(segments, rhythm, beat, data_dir)
    np.savez(cache, key=key, **features)
    return features


def baseline_cmd(n_boot: int = 1000,
                 report: Path = typer.Option(None, help="also write the validation choice and the leakage checks "
                                             "(report only) to this file"),
                 record: bool = typer.Option(True, help="record the runs in the database")) -> None:
    """Train and score the baselines of both tasks in one pass (prior; gradient boosting on RR and morphology)."""
    import pandas as pd

    from benchtoward.baselines.arrhythmia_rhythm_beat import leakage_report, run_rhythm_beat_baselines, validation_report
    from benchtoward.cli import _summary, open_session
    from benchtoward.registry import resolve_task
    from benchtoward.tasks.vitaldb_arrhythmia_rhythm_beat import BEAT_SLUG, RHYTHM_SLUG, task_dir

    data_dir = get_settings().data_dir
    root = task_dir(data_dir)
    paths = {k: root / f"{k}.parquet" for k in ("segments", "rhythm", "beat")}
    segments, rhythm, beat = (pd.read_parquet(paths[k]) for k in ("segments", "rhythm", "beat"))
    features = _features(paths, data_dir, segments, rhythm, beat)
    if record:
        with open_session() as session:
            tasks = {slug: resolve_task(session, f"{slug}@0", include_hidden=True) for slug in (RHYTHM_SLUG, BEAT_SLUG)}
            for run in run_rhythm_beat_baselines(session, tasks, rhythm, beat, features, data_dir=data_dir,
                                                 n_boot=n_boot):
                typer.echo(f"{run.task_id}: {_summary(run)}")
    if report:
        out = {"validation": validation_report(rhythm, beat, features),
               "leakage": leakage_report(rhythm, beat, features, segments, n_boot=n_boot)}
        report.write_text(json.dumps(out, indent=2, default=float) + "\n")
        typer.echo(f"wrote {report}")


def register(build_app: typer.Typer, baseline_app: typer.Typer) -> None:
    build_app.command(NAME)(build_cmd)
    baseline_app.command(NAME)(baseline_cmd)
