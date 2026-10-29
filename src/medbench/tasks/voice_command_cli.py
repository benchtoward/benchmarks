"""CLI commands (`medbench ...`) for kor-surgical-robot-voice-command@0: fetch, baseline, eval-voice-command, compare-voice-command.

    medbench fetch kor-surgical-robot-voice-command        # KHDP_TOKEN; every file SHA-256 checked
    medbench baseline kor-surgical-robot-voice-command     # medbench/prior (input-blind)
    medbench eval-voice-command --model <slug>             # all four subsets, run once, resumable
    medbench compare-voice-command --a <slug> --b <slug>   # paired difference per subset (cluster bootstrap)

The items need no build step: the release files are the instances.
"""
import asyncio
import json

import typer

from medbench.config import get_settings
from medbench.db import Model
from medbench.llm import MissingCredential, client_for
from medbench.tasks.voice_command import (
    SUBSETS,
    compare,
    dataset_dir,
    fetch_dataset,
    is_full_run,
    load_instances,
    load_rules,
    load_scenes,
    load_train,
    prior_output,
    prior_run,
    route_request,
    run_command_task,
    score_item,
)

DEFAULT_TASK = "kor-surgical-robot-voice-command@0"
DATASET = "kor-surgical-robot-voice-command"
OVERHEARD_LABELS = {"exact": "no-action rate (exact)", "unsafe_exec": "false activation (unsafe_exec)"}


def fetch_voice_command() -> None:
    """Download the Korean Surgical Robot Voice-Command Benchmark 1.0.0 from KHDP and check every file."""
    from medbench.khdp import KhdpClient

    settings = get_settings()
    if not settings.khdp_token:
        typer.echo("KHDP_TOKEN is not set (KHDP account settings -> API Token)")
        raise typer.Exit(1)
    root = fetch_dataset(settings.data_dir, KhdpClient(settings.khdp_token))
    inst = load_instances(settings.data_dir)
    counts = ", ".join(f"{s} {sum(i.subset == s for i in inst)}" for s in SUBSETS)
    typer.echo(f"{root}: {counts}; {len(load_scenes(settings.data_dir))} scenes")


def _require_data(data_dir) -> None:
    missing = [rel for rel in SUBSETS.values() if not (dataset_dir(data_dir) / rel).exists()]
    if missing:
        raise typer.BadParameter(f"run `medbench fetch {DATASET}` first ({missing[0]} is missing)")


def _subsets(subset: list[str] | None) -> list[str]:
    chosen = subset or list(SUBSETS)
    unknown = [s for s in chosen if s not in SUBSETS]
    if unknown:
        raise typer.BadParameter(f"unknown subset {unknown[0]} (choose from {', '.join(SUBSETS)})")
    return [s for s in SUBSETS if s in chosen]


def baseline_voice_command(task: str = typer.Option(DEFAULT_TASK, help="task, slug@version")) -> None:
    """Record medbench/prior: the most frequent training output (no action) for every item."""
    from medbench.cli import _summary, open_session
    from medbench.evaluate import record_baseline_run
    from medbench.registry import resolve_task

    settings = get_settings()
    _require_data(settings.data_dir)
    inst = load_instances(settings.data_dir)
    output = prior_output(load_train(settings.data_dir))
    rows, metrics = prior_run(inst, output)
    config = {"output": output, "rule": "most frequent (response, actions) of train/train_v0.2.jsonl",
              "subsets": list(SUBSETS)}
    with open_session() as session:
        run = record_baseline_run(session, session.get(Model, "medbench/prior"), resolve_task(session, task, include_hidden=True), rows,
                                  metrics, config, data_dir=settings.data_dir)
        typer.echo(_summary(run))


def _eval_one(session, model: Model, task_ref: str, subsets: list[str], force: bool, limit: int | None,
              concurrency: int) -> None:
    from medbench.cli import _summary
    from medbench.registry import resolve_task

    settings = get_settings()
    if model.type != "llm":
        raise typer.BadParameter(f"{model.slug} is not an LLM")
    _require_data(settings.data_dir)
    task = resolve_task(session, task_ref, include_hidden=True)  # also while in review
    inst = load_instances(settings.data_dir, subsets)
    route = settings.route_requests.get(model.provider)
    try:
        route_request(route, model.provider_model_id or model.slug)
    except ValueError as exc:
        raise typer.BadParameter(f"{model.slug} (route {model.provider}): {exc}")
    client = client_for(model, settings)

    def progress(done: int, total: int) -> None:
        if done % 250 == 0 or done == total:
            typer.echo(f"  {model.slug}: {done}/{total}")

    run = asyncio.run(run_command_task(session, model, task, inst, client, scenes=load_scenes(settings.data_dir),
                                       rules=load_rules(settings.data_dir), concurrency=concurrency, force=force,
                                       limit=limit, data_dir=settings.data_dir, progress=progress, route=route))
    typer.echo(_summary(run))
    if run.status == "failed":
        typer.echo("  some items still need a reply; run the same command again to resume")


def eval_voice_command(model: str = typer.Option(..., help="LLM slug, e.g. alibaba/qwen3.8-flash-next-180b-a6b"),
                       task: str = typer.Option(DEFAULT_TASK, help="task, slug@version"),
                       subset: list[str] = typer.Option(None, help="subset(s) to run (default: all four)"),
                       force: bool = False, limit: int | None = None,
                       concurrency: int = typer.Option(8, help="requests in flight")) -> None:
    """Ask one LLM for a robot command on every item and score it (run once; resumes unfinished runs)."""
    from medbench.cli import open_session

    with open_session() as session:
        row = session.get(Model, model)
        if row is None:
            raise typer.BadParameter(f"unknown model {model}")
        try:
            _eval_one(session, row, task, _subsets(subset), force, limit, concurrency)
        except MissingCredential as exc:
            typer.echo(f"skip {model}: {exc}")
            raise typer.Exit(1)


def _latest_rows(session, task, slug: str, inst) -> dict[str, list[dict]]:
    """Per-subset scored rows of a model's latest full run on the current spec (the run the leaderboard shows),
    rebuilt from stored predictions. Smoke runs (--limit) and subset runs (--subset) are skipped."""
    from medbench.db import ItemResult, Run

    runs = (session.query(Run).filter_by(task_id=task.id, model_slug=slug, status="done",
                                         task_spec_hash=task.spec_hash).order_by(Run.created_at.desc()))
    run = next((r for r in runs if is_full_run(json.loads(r.config_json or "{}"))), None)
    if run is None:
        raise typer.BadParameter(f"no finished full run of {slug} on {task.slug}@{task.version} (current spec)")
    stored = {r.instance_id: r for r in session.query(ItemResult).filter_by(run_id=run.id)}
    out: dict[str, list[dict]] = {}
    for i in inst:
        r = stored.get(i.instance_id)
        if r is None:
            continue
        parsed = json.loads(r.prediction) if r.prediction else None
        out.setdefault(i.subset, []).append(score_item(i.item, {"parsed": parsed, "error": r.error}))
    return out


def compare_voice_command(a: str = typer.Option(..., help="model slug A"), b: str = typer.Option(..., help="model slug B"),
                          task: str = typer.Option(DEFAULT_TASK, help="task, slug@version"),
                          metric: str = typer.Option("exact", help="per-item field: exact, unsafe_exec, ... "
                                                      "(overheard also prints false activation)"),
                          n_boot: int = 2000, seed: int = 0) -> None:
    """B minus A per subset on the same items, with a 95% cluster-bootstrap interval over wording variants."""
    from medbench.cli import open_session
    from medbench.registry import resolve_task

    settings = get_settings()
    _require_data(settings.data_dir)
    inst = load_instances(settings.data_dir)
    with open_session() as session:
        t = resolve_task(session, task, include_hidden=True)
        rows_a, rows_b = _latest_rows(session, t, a, inst), _latest_rows(session, t, b, inst)
    for subset in SUBSETS:
        if subset not in rows_a or subset not in rows_b:
            continue
        # On overheard talk `exact` is the no-action rate; false activation (an executed action) is `unsafe_exec`,
        # so it is always printed there too.
        fields = [metric] + (["unsafe_exec"] if subset == "overheard" and metric != "unsafe_exec" else [])
        for field in fields:
            c = compare(rows_a[subset], rows_b[subset], field, n_boot=n_boot, seed=seed)
            label = OVERHEARD_LABELS.get(field, field) if subset == "overheard" else field
            typer.echo(f"{subset} {label}: {c['a_%']} -> {c['b_%']} ({c['diff_pp']:+} pp, 95% CI {c['ci95_lo']} to "
                       f"{c['ci95_hi']}; {c['n_items']} items in {c['n_groups']} groups; discordant "
                       f"{c['only_a']}/{c['only_b']}, McNemar p={c['mcnemar_p']:.2g})")


def register(app: typer.Typer, fetch_app: typer.Typer, baseline_app: typer.Typer) -> None:
    fetch_app.command(DATASET)(fetch_voice_command)
    baseline_app.command(DATASET)(baseline_voice_command)
    app.command("eval-voice-command")(eval_voice_command)
    app.command("compare-voice-command")(compare_voice_command)

