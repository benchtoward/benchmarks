import asyncio
import json
import typer

from medbench.config import get_settings, registry_dir
from medbench.db import Model, init_db, make_engine, session_factory
from medbench.evaluate import (constant_answer_baseline, import_runs, max_tokens_for, record_baseline_run,
                               rekey_runs, run_llm_task)
from medbench.llm import MissingCredential, client_for
from medbench.registry import load_registry, resolve_task
from medbench.tasks.clinicalqa import fetch_clinicalqa, load_items

app = typer.Typer(help="medbench.kr benchmark commands", no_args_is_help=True)
fetch_app = typer.Typer(help="Download datasets", no_args_is_help=True)
build_app = typer.Typer(help="Build task instances", no_args_is_help=True)
baseline_app = typer.Typer(help="Run medbench baselines", no_args_is_help=True)
app.add_typer(fetch_app, name="fetch")
app.add_typer(build_app, name="build")
app.add_typer(baseline_app, name="baseline")

try:  # the paper agent is part of the medbench.kr deployment, not of the published evaluation code
    from medbench.agent.cli import agent_app  # noqa: E402
except ModuleNotFoundError:
    agent_app = None
from medbench.tasks.arrhythmia_cli import register as register_arrhythmia  # noqa: E402

if agent_app is not None:
    app.add_typer(agent_app, name="agent")
register_arrhythmia(fetch_app, build_app, baseline_app)

try:  # draft harness, published with its task after owner review
    from medbench.tasks.text2sql_cli import text2sql_app  # noqa: E402
except ModuleNotFoundError:
    text2sql_app = None

if text2sql_app is not None:
    app.add_typer(text2sql_app, name="text2sql")

from medbench.tasks.speech_cli import register as register_speech  # noqa: E402

register_speech(app, fetch_app)


def open_session():
    settings = get_settings()
    engine = make_engine(settings.db_url)
    init_db(engine)
    return session_factory(engine)()


def _summary(run) -> str:
    metrics = json.loads(run.metrics_json) if run.metrics_json else {}
    head = {k: v["value"] for k, v in metrics.items() if isinstance(v, dict) and "value" in v}
    return f"run {run.id} [{run.status}] {run.model_slug} n={run.n_items} {head} cost=${run.cost_usd:.4f}"


@app.command()
def init() -> None:
    """Create tables and load the registry (idempotent)."""
    with open_session() as session:
        load_registry(session, registry_dir())
    typer.echo("database ready")


@app.command()
def rekey() -> None:
    """Recompute run keys and task spec hashes after the spec-hash definition changes."""
    with open_session() as session:
        load_registry(session, registry_dir())
        typer.echo(f"{rekey_runs(session)} run(s) updated")


@app.command("import-bundle")
def import_bundle_cmd(source: str = typer.Argument(..., help="path to another medbench SQLite database")) -> None:
    """Merge the paper agent's state and finished runs from another database (tasks remapped by slug and version)."""
    from medbench.evaluate import import_bundle

    with open_session() as session:
        load_registry(session, registry_dir())
        counts = import_bundle(session, f"sqlite:///{source}")
        typer.echo(" ".join(f"{k}={v}" for k, v in counts.items()))


@app.command("import-runs")
def import_runs_cmd(source: str = typer.Argument(..., help="path to another medbench SQLite database")) -> None:
    """Merge finished runs from another database by run key, keeping this database's other data."""
    with open_session() as session:
        load_registry(session, registry_dir())
        typer.echo(f"{import_runs(session, f'sqlite:///{source}')} run(s) imported")


@fetch_app.command("clinicalqa")
def fetch_clinicalqa_cmd() -> None:
    typer.echo(fetch_clinicalqa(get_settings().data_dir))


def _eval_one(session, model: Model, task_ref: str, force: bool, limit: int | None) -> None:
    settings = get_settings()
    task = resolve_task(session, task_ref)
    items = load_items(fetch_clinicalqa(settings.data_dir))
    client = client_for(model, settings)
    run = asyncio.run(run_llm_task(session, model, task, items, client, force=force, limit=limit,
                                   max_tokens=max_tokens_for(model),
                                   data_dir=settings.data_dir))
    typer.echo(_summary(run))


@app.command("eval")
def eval_cmd(task: str = typer.Option(..., help="e.g. clinicalqa@0"),
             model: str = typer.Option(..., help="model slug, e.g. google/medgemma-27b"),
             force: bool = False, limit: int | None = None) -> None:
    """Evaluate one LLM on an LLM task (run once; resumes unfinished runs)."""
    with open_session() as session:
        row = session.get(Model, model)
        if row is None:
            raise typer.BadParameter(f"unknown model {model}")
        _eval_one(session, row, task, force, limit)


@app.command("eval-all")
def eval_all(task: str = typer.Option(..., help="e.g. clinicalqa@0")) -> None:
    """Evaluate every enabled LLM whose credentials are available."""
    with open_session() as session:
        for model in session.query(Model).filter_by(type="llm", enabled=True).order_by(Model.slug):
            try:
                _eval_one(session, model, task, False, None)
            except MissingCredential as exc:
                typer.echo(f"skip {model.slug}: {exc}")
            except Exception as exc:  # one model's failure must not stop the others
                session.rollback()
                typer.echo(f"error {model.slug}: {type(exc).__name__}: {exc}")


@baseline_app.command("clinicalqa")
def baseline_clinicalqa() -> None:
    """Record the majority-class reference (always answer B)."""
    settings = get_settings()
    with open_session() as session:
        task = resolve_task(session, "clinicalqa")
        rows, metrics = constant_answer_baseline(load_items(fetch_clinicalqa(settings.data_dir)), "B")
        run = record_baseline_run(session, session.get(Model, "medbench/majority-b"), task, rows, metrics,
                                  {"letter": "B"}, data_dir=settings.data_dir)
        typer.echo(_summary(run))


@fetch_app.command("vitaldb-ioh")
def fetch_vitaldb_ioh(workers: int = 8) -> None:
    """Download the MAP track for the vitaldb-ioh cohort (resumable)."""
    from medbench.tasks.vitaldb_ioh import fetch_tracks

    typer.echo(fetch_tracks(get_settings().data_dir, workers=workers))


@fetch_app.command("vitaldb-arrhythmia")
def fetch_vitaldb_arrhythmia() -> None:
    """Download the VitalDB Arrhythmia Database annotations from KHDP and verify checksums (resumable)."""
    from medbench.khdp import KhdpClient
    from medbench.tasks.vitaldb_arrhythmia import fetch_annotations, load_annotations, load_metadata

    settings = get_settings()
    if not settings.khdp_token:
        typer.echo("KHDP_TOKEN is not set (KHDP account settings -> API Token)")
        raise typer.Exit(1)
    root = fetch_annotations(settings.data_dir, KhdpClient(settings.khdp_token))
    ann, meta = load_annotations(settings.data_dir), load_metadata(settings.data_dir)
    typer.echo(f"{root}: {meta['caseid'].nunique()} cases, {meta['subjectid'].nunique()} subjects, "
               f"{len(ann)} annotated beats, rhythms: {', '.join(sorted(ann['rhythm_label'].dropna().unique()))}")


@build_app.command("vitaldb-ioh")
def build_vitaldb_ioh() -> None:
    """Build windows, labels and the subject split."""
    import pandas as pd

    from medbench.tasks.vitaldb_ioh import build_task

    path = build_task(get_settings().data_dir)
    df = pd.read_parquet(path, columns=["split", "label", "subjectid"])
    for split, part in df.groupby("split"):
        typer.echo(f"{split}: {len(part)} windows, {part['subjectid'].nunique()} subjects, "
                   f"prevalence {part['label'].mean():.3f}")


@baseline_app.command("vitaldb-ioh")
def baseline_vitaldb_ioh() -> None:
    """Train and score the two medbench baselines."""
    import pandas as pd

    from medbench.baselines.ioh import run_ioh_baselines
    from medbench.tasks.vitaldb_ioh import task_dir

    settings = get_settings()
    instances = pd.read_parquet(task_dir(settings.data_dir) / "instances.parquet")
    with open_session() as session:
        for run in run_ioh_baselines(session, resolve_task(session, "vitaldb-ioh"), instances,
                                     data_dir=settings.data_dir):
            typer.echo(_summary(run))


from medbench.tasks.inspire_cli import register as register_inspire  # noqa: E402

register_inspire(fetch_app, build_app, baseline_app)
