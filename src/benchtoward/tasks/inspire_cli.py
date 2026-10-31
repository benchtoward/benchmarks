"""medbench commands for INSPIRE: `fetch inspire`, `build inspire-mortality`, `baseline inspire-mortality`.

Kept out of cli.py so INSPIRE work merges without touching the other commands; cli.py only calls `register`.
"""
import typer

from benchtoward.config import get_settings


def fetch_inspire_cmd() -> None:
    """Download INSPIRE 1.3 operations.csv and labs.csv from KHDP (resumable)."""
    from benchtoward.khdp import KhdpClient
    from benchtoward.tasks.inspire_mortality import fetch_inspire

    settings = get_settings()
    if not settings.khdp_token:
        typer.echo("KHDP_TOKEN is not set (KHDP account settings -> API Token)")
        raise typer.Exit(1)
    root = fetch_inspire(settings.data_dir, KhdpClient(settings.khdp_token))
    typer.echo(f"{root}: " + ", ".join(f"{p.name} {p.stat().st_size:,} bytes" for p in sorted(root.glob("*.csv"))))


def build_inspire_mortality_cmd() -> None:
    """Build the cohort, label, inputs and the published train/test split."""
    import pandas as pd

    from benchtoward.tasks.inspire_mortality import build_task

    path = build_task(get_settings().data_dir)
    df = pd.read_parquet(path, columns=["split", "label", "subject_id", "vitaldb_linked"])
    typer.echo(f"{path}: {len(df)} patients")
    for split, part in df.groupby("split"):
        typer.echo(f"{split}: {len(part)} patients, {int(part['label'].sum())} deaths, "
                   f"prevalence {part['label'].mean():.4f}, VitalDB-linked {int(part['vitaldb_linked'].sum())}")


def baseline_inspire_mortality_cmd() -> None:
    """Train and score the three baselines from the INSPIRE authors' script (ASA, logistic regression, XGBoost)."""
    import pandas as pd

    from benchtoward.baselines.inspire import run_inspire_baselines
    from benchtoward.cli import _summary, open_session
    from benchtoward.registry import resolve_task
    from benchtoward.tasks.inspire_mortality import task_dir

    settings = get_settings()
    instances = pd.read_parquet(task_dir(settings.data_dir) / "instances.parquet")
    with open_session() as session:
        for run in run_inspire_baselines(session, resolve_task(session, "inspire-mortality"), instances,
                                         data_dir=settings.data_dir):
            typer.echo(_summary(run))


def register(fetch_app: typer.Typer, build_app: typer.Typer, baseline_app: typer.Typer) -> None:
    fetch_app.command("inspire")(fetch_inspire_cmd)
    build_app.command("inspire-mortality")(build_inspire_mortality_cmd)
    baseline_app.command("inspire-mortality")(baseline_inspire_mortality_cmd)
