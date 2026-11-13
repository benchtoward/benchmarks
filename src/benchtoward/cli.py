import asyncio
import json
import typer

from benchtoward.config import get_settings, registry_dir
from benchtoward.db import Model, Run, Task, init_db, make_engine, session_factory
from benchtoward.evaluate import (constant_answer_baseline, import_runs, max_tokens_for, record_baseline_run,
                               rekey_runs, run_llm_task)
from benchtoward.llm import MissingCredential, client_for
from benchtoward.registry import load_registry, resolve_task
from benchtoward.tasks.clinicalqa import fetch_clinicalqa, load_items

app = typer.Typer(help="BenchToward benchmark commands", no_args_is_help=True)
fetch_app = typer.Typer(help="Download datasets", no_args_is_help=True)
build_app = typer.Typer(help="Build task instances", no_args_is_help=True)
baseline_app = typer.Typer(help="Run BenchToward baselines", no_args_is_help=True)
app.add_typer(fetch_app, name="fetch")
app.add_typer(build_app, name="build")
app.add_typer(baseline_app, name="baseline")

try:  # the paper agent is part of the BenchToward deployment, not of the published evaluation code
    from benchtoward.agent.cli import agent_app  # noqa: E402
except ModuleNotFoundError:
    agent_app = None
from benchtoward.tasks.arrhythmia_cli import register as register_arrhythmia  # noqa: E402

if agent_app is not None:
    app.add_typer(agent_app, name="agent")
register_arrhythmia(fetch_app, build_app, baseline_app)

# Text-to-SQL: the question files, the frontier-model jobs and the KHDP release (public ahead of the task, owner
# 2026-10-30)
from benchtoward.tasks.text2sql_cli import text2sql_app  # noqa: E402

app.add_typer(text2sql_app, name="text2sql")

try:  # syn-snuh-note-errors rules and selection stay private while a test version is live
    from benchtoward.tasks.note_errors_cli import register as register_note_errors  # noqa: E402
except ModuleNotFoundError:
    register_note_errors = None
if register_note_errors is not None:
    register_note_errors(app)

from benchtoward.tasks.speech_cli import register as register_speech  # noqa: E402

register_speech(app, fetch_app)

try:  # SCOPE's pod build uses the deployment's compute client, which is not part of the published code
    from benchtoward.tasks.scope_cli import register as register_scope  # noqa: E402
except ModuleNotFoundError:
    register_scope = None
if register_scope is not None:
    register_scope(app, fetch_app, build_app, baseline_app)


def open_session():
    settings = get_settings()
    engine = make_engine(settings.db_url)
    init_db(engine)
    return session_factory(engine)()


def _summary(run) -> str:
    metrics = json.loads(run.metrics_json) if run.metrics_json else {}
    head = {k: v["value"] for k, v in metrics.items() if isinstance(v, dict) and "value" in v}
    return f"run {run.id} [{run.status}] {run.model_slug} n={run.n_items} {head} cost=${run.cost_usd:.4f}"


def _raise_failed_run(run) -> None:
    if run.status == "failed":
        typer.echo(json.loads(run.internal_json or "{}").get("failed_reason", f"run {run.id} failed"))
        raise typer.Exit(1)


@app.command("audit-api-errors")
def audit_api_errors(apply: bool = typer.Option(False, "--apply", help="mark the listed runs failed")) -> None:
    """List done model runs whose stored final call failures reach the API error threshold; dry run by default."""
    from types import SimpleNamespace

    from benchtoward.db import ItemResult
    from benchtoward.evaluate import finish_model_run

    with open_session() as session:
        matches = []
        for run, task in (session.query(Run, Task).join(Task, Run.task_id == Task.id)
                          .join(Model, Run.model_slug == Model.slug)
                          .filter(Run.status == "done", Model.type != "baseline").order_by(Run.id)):
            items = session.query(ItemResult).filter_by(run_id=run.id).all()
            candidate = SimpleNamespace(status=run.status, internal_json=run.internal_json)
            reason = finish_model_run(candidate, items)
            if reason and reason.startswith("API errors "):
                typer.echo(f"{run.id}\t{task.slug}@{task.version}\t{run.model_slug}\t{reason}")
                matches.append((run, items))
        if apply:
            for run, items in matches:
                finish_model_run(run, items)
            session.commit()
        typer.echo(f"{len(matches)} run(s) {'marked failed' if apply else 'would be marked failed (dry run)'}")


@app.command("privacy-retention")
def privacy_retention_command(apply: bool = typer.Option(False, "--apply", help="Apply privacy deadlines")) -> None:
    """Delete inactive accounts and expired records; anonymise old downloads. Dry run by default.

    Run daily on the work host against the site database, never as a Mac data job.
    """
    from benchtoward.privacy import privacy_retention

    settings = get_settings()
    # No init_db: a dry run must not migrate/create tables or create a missing database.
    if settings.db_url.startswith("sqlite:///") and settings.db_url != "sqlite:///:memory:":
        from pathlib import Path
        if not Path(settings.db_url.removeprefix("sqlite:///")).is_file():
            raise typer.BadParameter("The site database must already exist.")
    engine = make_engine(settings.db_url)
    try:
        with session_factory(engine)() as session:
            counts = privacy_retention(session, apply=apply, idle_seconds=settings.session_idle_seconds)
        typer.echo(("applied: " if apply else "dry run: ") + " ".join(f"{k}={v}" for k, v in counts.items()))
    finally:
        engine.dispose()


@app.command()
def init() -> None:
    """Create tables and load the registry (idempotent)."""
    with open_session() as session:
        load_registry(session, registry_dir())
    typer.echo("database ready")


@app.command()
def rekey() -> None:
    """Recompute run keys from each run's own spec hash (after a digest or run-key rule change). Never moves a run's
    spec hash: a rename-only spec edit goes through `rename-rekey`."""
    with open_session() as session:
        load_registry(session, registry_dir())
        typer.echo(f"{rekey_runs(session)} run(s) updated")


@app.command("rename-rekey")
def rename_rekey(plan: str = typer.Option(None, help="plan file for --write-to-hashes (default: every plan in "
                                                       "registry/renames/); the other modes always use them all"),
                 db: str = typer.Option(None, help="database URL (default: the configured one), e.g. a scratch DB"),
                 apply: bool = typer.Option(False, "--apply", help="back up the database, then apply the plans"),
                 check: bool = typer.Option(False, "--check", help="dry run; exit 1 on a refusal, a scan hit, lag or "
                                                                   "a skipped pair of an open task in a database "
                                                                   "the plan has not migrated"),
                 status: bool = typer.Option(False, "--status", help="applied plans and lag, read-only"),
                 write_to_hashes: bool = typer.Option(False, "--write-to-hashes",
                                                      help="fill the plan's `to` hashes from the registry specs")) -> None:
    """Rename plans (registry/renames/): proofs, run moves and rekeys. Default: a dry run on a copy of the database
    that prints the report and changes nothing."""
    from pathlib import Path

    from benchtoward import rename

    if sum((apply, check, status, write_to_hashes)) > 1:
        raise typer.BadParameter("choose one of --apply, --check, --status, --write-to-hashes")
    registry = registry_dir()
    url = db or get_settings().db_url
    try:
        if write_to_hashes:
            paths = [Path(plan)] if plan else sorted((registry / "renames").glob("*.yaml"))
            if not paths:
                typer.echo("no rename plans")
                raise typer.Exit(1)
            for path in paths:
                for line in rename.write_to_hashes(path, registry):
                    typer.echo(line)
            return
        plans = rename.load_plans(registry)
        if plan and Path(plan).resolve() not in {p.path.resolve() for p in plans}:
            raise typer.BadParameter(f"{plan} is not a plan in {registry / 'renames'}")
        if status:
            lines, failed = rename.status_lines(url, plans)
        elif apply:
            src = rename.sqlite_file(url)
            if src is not None and not src.is_file():
                raise rename.NoDatabase(f"no database at {src}: nothing to apply")
            engine = make_engine(url)
            init_db(engine)
            with session_factory(engine)() as session:
                report = load_registry(session, registry)
                report.lag = rename.rename_lag(session, plans)
            lines, failed = report.lines(), False
        else:
            report = rename.dry_run(url, registry)
            lines = report.lines()
            failed = check and bool(report.scan_hits or report.lag or report.open_skips)
    except rename.RenameRefused as exc:
        typer.echo(f"refused: {exc}")
        raise typer.Exit(1)
    except rename.UnsupportedURL as exc:
        typer.echo(str(exc))
        raise typer.Exit(1)
    except rename.NoDatabase as exc:
        # only the configured database may be missing (the deploy preflight on a first deploy); a --db that names no
        # file is a typo, and --status has nothing to report
        typer.echo(str(exc))
        raise typer.Exit(0 if db is None and not (status or apply) else 1)
    for line in lines:
        typer.echo(line)
    if failed:
        raise typer.Exit(1)


@app.command("import-bundle")
def import_bundle_cmd(source: str = typer.Argument(..., help="path to another BenchToward SQLite database")) -> None:
    """Merge the paper agent's state and finished runs from another database (tasks remapped by slug and version)."""
    from benchtoward.evaluate import import_bundle

    with open_session() as session:
        load_registry(session, registry_dir())
        counts = import_bundle(session, f"sqlite:///{source}", registry_dir=registry_dir())
        typer.echo(" ".join(f"{k}={v}" for k, v in counts.items()))


@app.command("rescore-clinicalqa")
def rescore_clinicalqa_cmd(apply: bool = typer.Option(False, "--apply", help="write the new runs (default: preview only)")
                           ) -> None:
    """Re-read finished ClinicalQA runs from their stored replies with the current parser, without model calls. Each
    becomes a new run (its config names the parser) that replaces the old one on the leaderboard; preview first."""
    from benchtoward.evaluate import rescore_clinicalqa
    from benchtoward.tasks.clinicalqa import fetch_clinicalqa, load_items

    settings = get_settings()
    items = load_items(fetch_clinicalqa(settings.data_dir))
    with open_session() as session:
        load_registry(session, registry_dir())
        task = session.query(Task).filter_by(slug="clinicalqa").order_by(Task.version.desc()).first()
        made = rescore_clinicalqa(session, task, items, data_dir=settings.data_dir, dry_run=not apply)
        for new in made:
            old = session.get(Run, json.loads(new.internal_json)["rescored_from"])
            before, after = json.loads(old.metrics_json), json.loads(new.metrics_json)
            typer.echo(f"{new.model_slug} [{new.status}] {old.id} -> {'(preview)' if not apply else new.id}: "
                       f"accuracy {before['accuracy']['value']:.3f} -> {after['accuracy']['value']:.3f}, "
                       f"parse rate {before['parse_rate']:.3f} -> {after['parse_rate']:.3f}")
        typer.echo(f"{len(made)} run(s) {'re-scored' if apply else 'would be re-scored (preview; --apply writes)'}")


@app.command("import-runs")
def import_runs_cmd(source: str = typer.Argument(..., help="path to another BenchToward SQLite database")) -> None:
    """Import finished runs (with their item results and missing models) from another database by run key. Papers,
    attempts, proposals, data needs and tasks are never written: a review's scratch DB cannot undo the main one."""
    with open_session() as session:
        load_registry(session, registry_dir())
        counts = import_runs(session, f"sqlite:///{source}", registry_dir=registry_dir())
        typer.echo(f"{counts['runs']} run(s) imported ({counts['items']} item results, {counts['models']} new models)")
        if counts["skipped"]:
            typer.echo(f"skipped {len(counts['skipped'])}: " + "; ".join(counts["skipped"]))



@app.command("public-artifacts")
def public_artifacts_cmd(db: str = typer.Option(None, "--db", help="SQLite file to read (default: the configured "
                                                                  "database); sync passes its snapshot"),
                         artifacts: str = typer.Option(None, "--artifacts", help="folder to list (default: "
                                                       "DATA_DIR/artifacts); the scripts pass the one rsync copies")
                         ) -> None:
    """Print the artifact folders (run ids under DATA_DIR/artifacts, or --artifacts) whose run has a public result
    page, one per line: the list deploy/sync.sh and deploy/deploy.sh give `rsync --files-from`, so hidden tasks'
    files never reach the site. Those scripts pass `--artifacts data/artifacts`, the folder they copy from, so the
    list and the copy read the same folder whatever BENCHTOWARD_DATA_DIR says. Only reads the database."""
    from pathlib import Path

    from benchtoward.web.data import public_artifact_dirs

    if db is not None and not Path(db).is_file():
        typer.echo(f"public-artifacts: no database at {db}", err=True)
        raise typer.Exit(2)
    folder = Path(artifacts) if artifacts is not None else get_settings().data_dir / "artifacts"
    if artifacts is not None and not folder.is_dir():
        typer.echo(f"public-artifacts: no folder at {artifacts}", err=True)
        raise typer.Exit(2)
    engine = make_engine(f"sqlite:///{db}" if db is not None else get_settings().db_url)
    with session_factory(engine)() as session:
        for name in public_artifact_dirs(session, folder):
            typer.echo(name)

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
    _raise_failed_run(run)


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
    failed = False
    with open_session() as session:
        for model in session.query(Model).filter_by(type="llm", enabled=True).order_by(Model.slug):
            try:
                _eval_one(session, model, task, False, None)
            except MissingCredential as exc:
                typer.echo(f"skip {model.slug}: {exc}")
            except typer.Exit:
                failed = True
            except Exception as exc:  # one model's failure must not stop the others
                session.rollback()
                typer.echo(f"error {model.slug}: {type(exc).__name__}: {exc}")
                failed = True
    if failed:
        raise typer.Exit(1)


@baseline_app.command("clinicalqa")
def baseline_clinicalqa() -> None:
    """Record the majority-class reference (always answer B)."""
    settings = get_settings()
    with open_session() as session:
        task = resolve_task(session, "clinicalqa")
        rows, metrics = constant_answer_baseline(load_items(fetch_clinicalqa(settings.data_dir)), "B")
        run = record_baseline_run(session, session.get(Model, "benchtoward/majority-b"), task, rows, metrics,
                                  {"letter": "B"}, data_dir=settings.data_dir)
        typer.echo(_summary(run))


@fetch_app.command("vitaldb-ioh")
def fetch_vitaldb_ioh(workers: int = 8) -> None:
    """Download the MAP track for the vitaldb-ioh cohort (resumable)."""
    from benchtoward.tasks.vitaldb_ioh import fetch_tracks

    typer.echo(fetch_tracks(get_settings().data_dir, workers=workers))


@fetch_app.command("vitaldb-arrhythmia")
def fetch_vitaldb_arrhythmia() -> None:
    """Download the VitalDB Arrhythmia Database annotations from KHDP and verify checksums (resumable)."""
    from benchtoward.khdp import KhdpClient
    from benchtoward.tasks.vitaldb_arrhythmia import fetch_annotations, load_annotations, load_metadata

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

    from benchtoward.tasks.vitaldb_ioh import build_task

    path = build_task(get_settings().data_dir)
    df = pd.read_parquet(path, columns=["split", "label", "subjectid"])
    for split, part in df.groupby("split"):
        typer.echo(f"{split}: {len(part)} windows, {part['subjectid'].nunique()} subjects, "
                   f"prevalence {part['label'].mean():.3f}")


@baseline_app.command("vitaldb-ioh")
def baseline_vitaldb_ioh() -> None:
    """Train and score the two BenchToward baselines."""
    import pandas as pd

    from benchtoward.baselines.ioh import run_ioh_baselines
    from benchtoward.tasks.vitaldb_ioh import TASK_REF, task_dir

    settings = get_settings()
    instances = pd.read_parquet(task_dir(settings.data_dir) / "instances.parquet")
    with open_session() as session:
        for run in run_ioh_baselines(session, resolve_task(session, TASK_REF), instances,
                                     data_dir=settings.data_dir):
            typer.echo(_summary(run))


from benchtoward.tasks.inspire_cli import register as register_inspire  # noqa: E402

register_inspire(fetch_app, build_app, baseline_app)

try:  # vitaldb-ioh@1 is exported with its task once the maintainers publish it
    from benchtoward.tasks.ioh_v1_cli import register as register_ioh_v1  # noqa: E402
except ModuleNotFoundError:
    register_ioh_v1 = None
if register_ioh_v1 is not None:
    register_ioh_v1(build_app, baseline_app)

try:  # vitaldb-arrhythmia-rhythm@0 and -beat@0 are exported with their tasks once the maintainers publish them
    from benchtoward.tasks.arrhythmia_rhythm_beat_cli import register as register_rhythm_beat  # noqa: E402
except ModuleNotFoundError:
    register_rhythm_beat = None
if register_rhythm_beat is not None:
    register_rhythm_beat(build_app, baseline_app)

try:  # inspire-aki@0 is exported with its task once the maintainers publish it
    from benchtoward.tasks.inspire_aki_cli import register as register_inspire_aki  # noqa: E402
except ModuleNotFoundError:
    register_inspire_aki = None
if register_inspire_aki is not None:
    register_inspire_aki(fetch_app, build_app, baseline_app)
try:  # vitaldb-bis@0 is exported with its task once the maintainers publish it
    from benchtoward.tasks.bis_cli import register as register_bis  # noqa: E402
except ModuleNotFoundError:
    register_bis = None
if register_bis is not None:
    register_bis(build_app, baseline_app)

try:  # its pod extraction is deployment-only
    from benchtoward.tasks.bis_pods_cli import register as register_bis_pods  # noqa: E402
except ModuleNotFoundError:
    register_bis_pods = None
if register_bis_pods is not None:
    register_bis_pods(app)

from benchtoward.tasks.voice_command_cli import register as register_voice_command  # noqa: E402

try:  # syn-icu-sofa@0 is exported with its task once the maintainers publish it
    from benchtoward.tasks.syn_icu_sofa_cli import register as register_syn_icu_sofa  # noqa: E402
except ModuleNotFoundError:
    register_syn_icu_sofa = None
if register_syn_icu_sofa is not None:
    register_syn_icu_sofa(app, fetch_app, build_app, baseline_app)

register_voice_command(app, fetch_app, baseline_app)

try:  # syn-snuh-chatqa@0 is exported with its task once the maintainers publish it
    from benchtoward.tasks.chatqa_cli import register as register_chatqa  # noqa: E402
except ModuleNotFoundError:
    register_chatqa = None
if register_chatqa is not None:
    register_chatqa(app)

from benchtoward.tasks.text2sql_task_cli import register as register_text2sql_task  # noqa: E402

register_text2sql_task(app, fetch_app, baseline_app)  # synthetic-snuh-text2sql@0 (its code is public, owner 2026-10-30)

try:  # deployment scheduling and job bookkeeping, not part of the published evaluation code
    from benchtoward.orch.cli import orch_app, register_jobs  # noqa: E402
except ModuleNotFoundError:
    orch_app = register_jobs = None

if orch_app is not None:
    app.add_typer(orch_app, name="orch")
    register_jobs(app)  # eval commands hold a ledger lease while they run

try:  # lydus-ecg12-dx@0 is exported with its task once the maintainers publish it
    from benchtoward.tasks.lydus_cli import register as register_lydus  # noqa: E402
except ModuleNotFoundError:
    register_lydus = None
if register_lydus is not None:
    register_lydus(app, fetch_app, build_app, baseline_app)

try:  # ptbxl-superdiagnostic@0 is exported with its task once the maintainers publish it
    from benchtoward.tasks.ptbxl_cli import register as register_ptbxl  # noqa: E402
except ModuleNotFoundError:
    register_ptbxl = None
if register_ptbxl is not None:
    register_ptbxl(app, fetch_app, build_app, baseline_app)

try:  # lydus-ecg-image@0 is exported with its task once the maintainers publish it
    from benchtoward.tasks.lydus_ecg_image_cli import register as register_ecg_image  # noqa: E402
except ModuleNotFoundError:
    register_ecg_image = None
if register_ecg_image is not None:
    register_ecg_image(app, build_app, baseline_app)
