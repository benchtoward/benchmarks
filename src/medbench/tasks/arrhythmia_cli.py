"""medbench commands for vitaldb-arrhythmia-abnormal@0: ECG cache, build, baselines.

    medbench fetch vitaldb-arrhythmia                 # annotations from KHDP (needs KHDP_TOKEN)
    medbench fetch vitaldb-arrhythmia-ecg             # ECG lead II from VitalDB, annotated spans only
    medbench build vitaldb-arrhythmia-abnormal
    medbench baseline vitaldb-arrhythmia-abnormal
"""
import json

import typer

from medbench.config import get_settings

SLUG = "vitaldb-arrhythmia-abnormal"


def fetch_ecg_cmd(workers: int = 8) -> None:
    """Download ECG lead II (VitalDB, 500 Hz) for the annotated span of every case (resumable)."""
    from medbench.tasks.vitaldb_arrhythmia import dataset_dir
    from medbench.tasks.vitaldb_arrhythmia_abnormal import ecg_dir, fetch_ecg

    data_dir = get_settings().data_dir
    if not (dataset_dir(data_dir) / "metadata.csv").exists():
        typer.echo("annotations missing: run `medbench fetch vitaldb-arrhythmia` first")
        raise typer.Exit(1)
    try:
        summary = fetch_ecg(data_dir, workers=workers)
    except RuntimeError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1)
    offsets = [json.loads(p.read_text(encoding="utf-8"))["qc"]["median_offset_ms"]
               for p in sorted(ecg_dir(data_dir).glob("*.json"))]
    shifted = [o for o in offsets if o is None or abs(o) > 20]
    typer.echo(f"{ecg_dir(data_dir)}: {summary['fetched']} case(s) downloaded in {summary['seconds']} s, "
               f"{len(offsets)}/{summary['cases']} cached ({summary['cache_bytes'] / 1e6:.0f} MB); "
               f"annotation vs ECG R-peak offset over 20 ms (or unknown) in {len(shifted)} case(s)")


def build_cmd() -> None:
    """Build 5-second segments, labels and the dataset-level VitalDB split."""
    from medbench.tasks.vitaldb_arrhythmia_abnormal import build_task, task_dir

    data_dir = get_settings().data_dir
    try:
        build_task(data_dir)
    except (RuntimeError, ValueError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(1)
    summary = json.loads((task_dir(data_dir) / "build_summary.json").read_text(encoding="utf-8"))
    for part in ("train", "validation", "test"):
        s = summary["splits"].get(part)
        if s is None:
            continue
        excluded = {k: v for k, v in summary["excluded"][part].items() if v}
        typer.echo(f"{part}: {s['instances']} segments, {s['subjects']} subjects, labels {s['label']}, "
                   f"rhythms {s['rhythm']}; not scored {excluded}")


def baseline_cmd(n_boot: int = 1000) -> None:
    """Train and score the two medbench baselines (prior, gradient boosting on RR features)."""
    import pandas as pd

    from medbench.baselines.arrhythmia import features_for, run_arrhythmia_baselines
    from medbench.cli import _summary, open_session
    from medbench.registry import resolve_task
    from medbench.tasks.vitaldb_arrhythmia_abnormal import task_dir

    data_dir = get_settings().data_dir
    instances = pd.read_parquet(task_dir(data_dir) / "instances.parquet")
    features = features_for(instances, data_dir)
    with open_session() as session:
        for run in run_arrhythmia_baselines(session, resolve_task(session, SLUG), instances, features,
                                            data_dir=data_dir, n_boot=n_boot):
            typer.echo(_summary(run))


def register(fetch_app: typer.Typer, build_app: typer.Typer, baseline_app: typer.Typer) -> None:
    fetch_app.command("vitaldb-arrhythmia-ecg")(fetch_ecg_cmd)
    build_app.command(SLUG)(build_cmd)
    baseline_app.command(SLUG)(baseline_cmd)
