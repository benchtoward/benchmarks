"""CLI commands for synthetic-snuh-text2sql@0: fetch, baseline, eval-text2sql, check-text2sql.

    benchtoward fetch synthetic-snuh                          # KHDP_TOKEN; MD5-checked, loaded into DuckDB
    benchtoward baseline synthetic-snuh-text2sql              # the two fixed-rule priors
    benchtoward eval-text2sql --model <slug> [--reuse hard=DIR --reuse easy=DIR]   # run once, resumable
    benchtoward check-text2sql [--json report.json]           # the SOP check with the spec's roster

The items are the frozen question files the spec pins (registry/drafts/, checked by sha256); there is no build step.
"""
import asyncio
import json
from pathlib import Path

import typer

from benchtoward.config import get_settings, registry_dir
from benchtoward.db import Model
from benchtoward.tasks.text2sql_task import (TASK_REF, TIERS, load_tiers, paired_difference, record_priors,
                                          run_text2sql_task, sop_check)

DATASET = "synthetic-snuh"


def _connect():
    from benchtoward.tasks.synthetic_snuh import connect_readonly, db_path

    path = db_path(get_settings().data_dir)
    if not path.exists():
        typer.echo(f"{path} not found: run `benchtoward fetch {DATASET}` first (needs KHDP_TOKEN)")
        raise typer.Exit(1)
    return connect_readonly(path)


def _task_and_tiers(session, ref: str):
    from benchtoward.registry import resolve_task

    task = resolve_task(session, ref, include_hidden=True)  # also while in review
    spec = json.loads(task.spec_json)
    try:
        return task, spec, load_tiers(spec, registry_dir())
    except (OSError, ValueError) as exc:
        typer.echo(f"{task.slug}@{task.version}: {exc}")
        raise typer.Exit(1)


def fetch_synthetic_snuh(force: bool = typer.Option(False, help="rebuild the DuckDB file even if it is current")) -> None:
    """Download Synthetic-SNUH 4.0.0 (OMOP CDM Parquet) from KHDP, check MD5 sums and load it into DuckDB."""
    from benchtoward.khdp import KhdpClient
    from benchtoward.tasks.synthetic_snuh import build_duckdb, connect_readonly, fetch_cdm, table_counts

    settings = get_settings()
    if not settings.khdp_token:
        typer.echo("KHDP_TOKEN is not set (KHDP account settings -> API Token)")
        raise typer.Exit(1)
    root = fetch_cdm(settings.data_dir, KhdpClient(settings.khdp_token))
    path = build_duckdb(settings.data_dir, force=force)
    counts = table_counts(connect_readonly(path))
    typer.echo(f"{root} -> {path}")
    typer.echo(", ".join(f"{t} {n:,}" for t, n in counts.items()))


def baseline_text2sql(task: str = typer.Option(TASK_REF, help="task, slug@version")) -> None:
    """Record the fixed-rule priors: abstain on every item (medbench/prior) and SELECT 0 for every item."""
    from benchtoward.cli import _summary, open_session

    settings = get_settings()
    con = _connect()
    with open_session() as session:
        t, _, tiers = _task_and_tiers(session, task)
        for run in record_priors(session, t, tiers, con, data_dir=settings.data_dir):
            typer.echo(_summary(run))


def _reuse(values: list[str] | None) -> dict[str, Path]:
    out = {}
    for value in values or []:
        tier, sep, path = value.partition("=")
        if not sep or tier not in TIERS:
            raise typer.BadParameter(f"--reuse takes TIER=DIR with TIER one of {', '.join(TIERS)}, not {value!r}")
        out[tier] = Path(path)
    return out


def eval_text2sql(model: str = typer.Option(..., help="LLM slug, e.g. deepseek/deepseek-v4.1-flash-552b-a16b"),
                  task: str = typer.Option(TASK_REF, help="task, slug@version"),
                  reuse: list[str] = typer.Option(None, help="TIER=DIR (repeatable): a `text2sql dry-run` folder of "
                                                  "this model whose answers to the same prompts are reused; refused "
                                                  "if any answer there went over the output limit"),
                  force: bool = False, limit: int | None = typer.Option(None, help="first N instances (smoke run)"),
                  concurrency: int = typer.Option(8, help="requests in flight")) -> None:
    """Ask one LLM for SQL on every instance of both tiers and score it (run once; resumes unfinished runs)."""
    from benchtoward.cli import _summary, open_session
    from benchtoward.llm import MissingCredential, client_for

    sources = _reuse(reuse)
    settings = get_settings()
    con = _connect()
    with open_session() as session:
        row = session.get(Model, model)
        if row is None or row.type != "llm":
            raise typer.BadParameter(f"{model} is not an LLM in registry/models.yaml")
        t, spec, tiers = _task_and_tiers(session, task)
        try:
            client, why = client_for(row, settings), None
        except (MissingCredential, ValueError) as exc:  # fine while every answer comes from the cache
            client, why = None, str(exc)

        def progress(done: int, total: int) -> None:
            if done % 50 == 0 or done == total:
                typer.echo(f"  {row.slug}: {done}/{total} model calls")

        try:
            run = asyncio.run(run_text2sql_task(session, row, t, tiers, con, client, spec=spec, reuse=sources,
                                                concurrency=concurrency, force=force, limit=limit,
                                                data_dir=settings.data_dir, progress=progress))
        except ValueError as exc:
            typer.echo(f"{row.slug}: {exc}" + (f" ({why})" if why else ""))
            raise typer.Exit(1)
        typer.echo(_summary(run))
        if run.status == "failed":
            typer.echo("  some answers still need a model call; run the same command again to resume")
            raise typer.Exit(1)
        metrics = json.loads(run.metrics_json)
        typer.echo(f"  {metrics['answers_from_cache']} of {run.n_items} answers from the dry-run cache; cut off at "
                   f"the limit {metrics['cut_off_rate']['all']:.1%}, format failures "
                   f"{metrics['format_failure_rate']['all']:.1%}, API errors {metrics['api_error_rate']:.1%}")


def _latest_full_runs(session, task, slugs) -> dict:
    from benchtoward.db import Run

    out = {}
    for slug in slugs:
        runs = (session.query(Run).filter_by(task_id=task.id, model_slug=slug, status="done",
                                             task_spec_hash=task.spec_hash).order_by(Run.created_at.desc()))
        run = next((r for r in runs if not json.loads(r.config_json or "{}").get("partial")), None)
        if run is None:
            raise typer.BadParameter(f"no finished full run of {slug} on {task.slug}@{task.version} (current spec)")
        out[slug] = run
    return out


def _per_instance(session, run, value) -> dict[str, float]:
    from benchtoward.db import ItemResult

    return {r.instance_id: v for r in session.query(ItemResult).filter_by(run_id=run.id)
            if (v := value(r)) is not None}


def _ci(m: dict) -> str:
    return f"{m['value']:.3f} [{m['ci_low']:.3f}, {m['ci_high']:.3f}]"


def check_text2sql(task: str = typer.Option(TASK_REF, help="task, slug@version"),
                   json_out: Path | None = typer.Option(None, "--json", help="also write the full report here")) -> None:
    """The SOP check (stage 5) from the recorded runs of the spec's roster: priors, weak and strong reference."""
    from benchtoward.cli import open_session

    with open_session() as session:
        t, spec, _ = _task_and_tiers(session, task)
        check = spec["check"]
        weak, strong = check["roster"]["weak"], check["roster"]["strong"]
        runs = _latest_full_runs(session, t, [*check["priors"].values(), weak, strong])
        metrics = {slug: json.loads(r.metrics_json) for slug, r in runs.items()}
        report = sop_check(check, metrics)
        # paired differences (strong - weak) over the same questions, both languages together
        rs0 = lambda r: None if not r.instance_id.startswith("hard/") else max(r.score, 0.0)  # noqa: E731
        ex = lambda tier: lambda r: (float(bool(r.correct)) if r.instance_id.startswith(f"{tier}/")  # noqa: E731
                                     and r.label == "answerable" else None)
        values = {"rs0": rs0, "hard_ex": ex("hard"), "easy_ex": ex("easy")}
        report["paired"] = {k: paired_difference(_per_instance(session, runs[strong], f),
                                                 _per_instance(session, runs[weak], f)) for k, f in values.items()}
        report["runs"] = {slug: r.id for slug, r in runs.items()}
    typer.echo(f"SOP check of {t.slug}@{t.version} (roster fixed in the spec: weak {weak}, strong {strong}; priors "
               f"{check['priors']['rs']} for RS, {check['priors']['ex']} for EX)")
    for tier in TIERS:
        r = report["tiers"][tier]
        h, s, f = r["headroom"], r["separation"], r["format"]
        metric = h["metric"]
        typer.echo(f"{tier} tier, {metric}: prior {h['prior']:.3f} ({h['prior_rule']}); weak {_ci(metrics[weak][metric])}"
                   f"; strong {_ci(metrics[strong][metric])}; paired strong - weak {_ci(report['paired'][metric])}")
        typer.echo(f"  headroom H {h['H']:.3f} (best {h['best']}, its interval reaches {h['ci_high']:.3f}): "
                   f"{'pass' if h['pass'] else 'FAIL'}{' (warning: H < 0.10)' if h['warning'] else ''}")
        typer.echo(f"  separation (intervals disjoint, strong above weak): {'pass' if s['pass'] else 'FAIL'}")
        typer.echo("  format failures " + ", ".join(f"{k} {v:.1%}" for k, v in f["failure"].items())
                   + "; cut off at the limit (model failures) " + ", ".join(f"{k} {v:.1%}" for k, v in f["cut_off"].items())
                   + f": {'pass' if f['pass'] else 'FAIL'}")
        beyond = r["prior"]["references_beyond_prior"]
        typer.echo("  beats the prior: no baseline beyond the input-blind priors; references beyond the prior: "
                   + ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in beyond.items()))
        for m, extra in r["also"].items():
            typer.echo(f"  also {m}: weak {_ci(metrics[weak][m])}, strong {_ci(metrics[strong][m])}, H "
                       f"{extra['headroom']['H']:.3f}, separation {'pass' if extra['separation']['pass'] else 'FAIL'}")
        typer.echo(f"  verdict: {r['verdict'].upper()} (rests on: " + "; ".join(c["text"] for c in r["caveats"]) + ")")
    if json_out:
        Path(json_out).write_text(json.dumps(report, indent=1, ensure_ascii=False))


def register(app: typer.Typer, fetch_app: typer.Typer, baseline_app: typer.Typer) -> None:
    fetch_app.command(DATASET)(fetch_synthetic_snuh)
    baseline_app.command("synthetic-snuh-text2sql")(baseline_text2sql)
    app.command("eval-text2sql")(eval_text2sql)
    app.command("check-text2sql")(check_text2sql)
