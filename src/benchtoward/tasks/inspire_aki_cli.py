"""Commands for inspire-aki@0 (postoperative AKI on INSPIRE 1.3, the cohort and models of Do et al. 2026).

    benchtoward fetch inspire-aki                    # operations, labs, ward_vitals, diagnosis, vitals (about 4.3 GB)
    benchtoward build inspire-aki                    # the paper's cohort, label, inputs, GS-AKI, split and outer folds
    benchtoward baseline inspire-aki                 # prior, ASA, GS-AKI, the authors' ridge LR and GBT
    benchtoward baseline inspire-aki --report r.json # also the earlier choices (CRRT in 7 days, no durations,
                                                  # pre-existing kidney failure left out), the five outer folds
                                                  # against the paper's Table 2, raw scores, leak probes
    benchtoward baseline inspire-aki --no-record --hpo-matrices DIR  # train matrices for the authors' Optuna search
"""
import json
from pathlib import Path

import typer

from benchtoward.config import get_settings


def fetch_inspire_aki_cmd() -> None:
    """Download the five INSPIRE 1.3 tables inspire-aki reads (resumable)."""
    from benchtoward.khdp import KhdpClient
    from benchtoward.tasks.inspire_aki import fetch_inspire_aki

    settings = get_settings()
    if not settings.khdp_token:
        typer.echo("KHDP_TOKEN is not set (KHDP account settings -> API Token)")
        raise typer.Exit(1)
    root = fetch_inspire_aki(settings.data_dir, KhdpClient(settings.khdp_token))
    typer.echo(f"{root}: " + ", ".join(f"{p.name} {p.stat().st_size:,} bytes" for p in sorted(root.glob("*.csv"))))


def build_inspire_aki_cmd() -> None:
    """Build the cohort, labels, inputs and split; write instances and a summary of every cohort step."""
    from benchtoward.tasks.inspire_aki import build_task

    path = build_task(get_settings().data_dir)
    typer.echo(f"{path}\n{(path.parent / 'summary.json').read_text()}")


def baseline_inspire_aki_cmd(n_boot: int = 1000,
                             report: Path = typer.Option(None, help="write report-only numbers to this file"),
                             record: bool = typer.Option(True, help="record the headline runs in the database"),
                             hpo_matrices: Path = typer.Option(None, help="write the train matrix of every Optuna "
                                                               "search (spec baselines_detail.hpo) to this folder")
                             ) -> None:
    """Train on the train split, score the test split, record the five baselines."""
    from benchtoward.baselines.inspire_aki import hpo_matrices as make_matrices
    from benchtoward.baselines.inspire_aki import report as make_report
    from benchtoward.baselines.inspire_aki import run_inspire_aki_baselines
    from benchtoward.cli import _summary, open_session
    from benchtoward.registry import resolve_task
    from benchtoward.tasks.inspire_aki import TASK_KEY, load_instances

    data_dir = get_settings().data_dir
    instances = load_instances(data_dir)
    if hpo_matrices:
        hpo_matrices.mkdir(parents=True, exist_ok=True)
        for name, matrix in make_matrices(instances).items():
            matrix.to_parquet(hpo_matrices / f"{name}.parquet", index=False)
            typer.echo(f"{name}: {len(matrix)} rows, {int(matrix['aki_boolean'].sum())} positives")
    if record:
        with open_session() as session:
            task = resolve_task(session, TASK_KEY, include_hidden=True)
            for run in run_inspire_aki_baselines(session, task, instances, data_dir=data_dir, n_boot=n_boot):
                typer.echo(_summary(run))
    if report:
        out = make_report(instances, n_boot=n_boot)
        report.write_text(json.dumps(out, indent=2, default=float) + "\n")
        typer.echo(f"wrote {report}")


def register(fetch_app: typer.Typer, build_app: typer.Typer, baseline_app: typer.Typer) -> None:
    fetch_app.command("inspire-aki")(fetch_inspire_aki_cmd)
    build_app.command("inspire-aki")(build_inspire_aki_cmd)
    baseline_app.command("inspire-aki")(baseline_inspire_aki_cmd)
