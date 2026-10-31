"""`benchtoward text2sql ...`: the Synthetic-SNUH Text-to-SQL question files (fetch, check, review, prompt, dry-run) and
the frontier-model jobs that wrote, reviewed, fixed and voted on them (hard-generate, hard-review, llm-review,
llm-fix, llm-decide, send-back, spend), and the KHDP release of the frozen items (khdp-package).

None of these commands writes to the medbench database: dry-run results and job caches stay in files under
<data_dir>/drafts/. The frozen task synthetic-snuh-text2sql@0 runs through `benchtoward eval-text2sql`
(text2sql_task_cli.py). The frontier models are served by the route that registry/providers.yaml marks
`frontier: true` (`frontier_route`).
"""
import asyncio
import json
import statistics
from collections import Counter
from pathlib import Path

import typer
import yaml

from benchtoward.config import get_settings, provider_routes, registry_dir
from benchtoward.db import Model
from benchtoward.llm import MissingCredential, client_for

text2sql_app = typer.Typer(help="Synthetic-SNUH Text-to-SQL: question files, frontier-model jobs, KHDP release",
                           no_args_is_help=True)

DRAFT_NAME = "synthetic-snuh-text2sql-v0"
ROUTE_HELP = "serving route in registry/providers.yaml (default: the route marked frontier: true)"


def _draft(draft: Path | None) -> Path:
    return Path(draft) if draft else registry_dir() / "drafts" / f"{DRAFT_NAME}.yaml"


def _connect():
    from benchtoward.tasks.synthetic_snuh import connect_readonly, db_path

    path = db_path(get_settings().data_dir)
    if not path.exists():
        typer.echo(f"{path} not found: run `benchtoward text2sql fetch` first (needs KHDP_TOKEN)")
        raise typer.Exit(1)
    return connect_readonly(path)


def _questions(draft: Path | None):
    from benchtoward.tasks.text2sql import load_questions

    return load_questions(_draft(draft))


def _registry_model(slug: str) -> Model:
    """The model as registry/models.yaml defines it, without touching any database."""
    from benchtoward.registry import MODEL_FIELDS

    for m in yaml.safe_load((registry_dir() / "models.yaml").read_text(encoding="utf-8")):
        if slug == m["slug"] or slug in (m.get("aliases") or []):
            return Model(slug=m["slug"], digest=m.get("digest") or m["slug"], enabled=m.get("enabled", True),
                         **{f: m.get(f) for f in MODEL_FIELDS})
    typer.echo(f"unknown model {slug} (not in registry/models.yaml)")
    raise typer.Exit(2)


@text2sql_app.command("fetch")
def fetch_cmd(force: bool = typer.Option(False, help="rebuild the DuckDB file even if it is current")) -> None:
    """Download Synthetic-SNUH v4 OMOP CDM Parquet from KHDP, verify MD5 sums, unzip and load into DuckDB."""
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


@text2sql_app.command("check")
def check_cmd(draft: Path | None = typer.Option(None, help="question file (default: the v0 draft in registry/drafts)")
              ) -> None:
    """Execute every reference SQL on the real data; list problems and result rows per category. For the harder
    set, also the rule that no question quotes a concept name its reference uses (in quotes or parentheses)."""
    from dataclasses import replace

    from benchtoward.tasks.text2sql import ALL_CATEGORIES, DIFFICULTIES, P0, check_questions, prompt_spec

    meta, items = _questions(draft)
    con = _connect()
    checks = check_questions(con, items)
    if prompt_spec(meta) is not P0:  # v0 quotes concept names on purpose
        from benchtoward.tasks.text2sql_llm_fix import quoted_problem

        by_id = {i.id: i for i in items}
        checks = [c if c.problem else replace(c, problem=quoted_problem(con, by_id[c.id])) for c in checks]
    problems = [c for c in checks if not c.ok]
    for c in problems:
        typer.echo(f"PROBLEM {c.id} [{c.category}/{c.difficulty}]: {c.problem}")
    typer.echo(f"{'category':<19}{'items':>6}  {'easy/medium/hard':<17}{'rows total':>11}{'median':>8}{'max':>6}")
    for category in ALL_CATEGORIES:
        group = [c for c in checks if c.category == category]
        if not group:
            continue
        mix = Counter(c.difficulty for c in group)
        rows = [c.rows for c in group]
        typer.echo(f"{category:<19}{len(group):>6}  {'/'.join(str(mix[d]) for d in DIFFICULTIES):<17}"
                   f"{sum(rows):>11}{statistics.median(rows):>8g}{max(rows):>6}")
    typer.echo(f"{len(checks)} questions, {len(problems)} problems")
    if problems:
        raise typer.Exit(1)


def _md_cell(value) -> str:
    text = "NULL" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


@text2sql_app.command("review")
def review_cmd(draft: Path | None = typer.Option(None, help="question file (default: the v0 draft)"),
               out: Path | None = typer.Option(None, help="write the Markdown sheet here instead of stdout"),
               id: list[str] = typer.Option(None, help="only these question ids (repeatable)"),
               category: str | None = typer.Option(None, help="only this category"),
               rows: int = typer.Option(10, help="result rows to show per question")) -> None:
    """Review sheet for the owner: each question in Korean and English, its reference SQL and its result."""
    from benchtoward.tasks.text2sql import execute_sql

    meta, items = _questions(draft)
    items = [i for i in items if (not id or i.id in id) and (not category or i.category == category)]
    con = _connect()
    states = Counter(i.review for i in items)
    lines = ["# Synthetic-SNUH Text-to-SQL v0: review sheet", "", f"Status: {meta.get('status', '')}", "",
             f"How to review: in `{_draft(draft)}`, set each item's `review:` to approved, needs_edit or rejected "
             f"(add a `note:` saying why), then run `benchtoward text2sql check`.", "",
             "Review states: " + ", ".join(f"{k} {v}" for k, v in sorted(states.items())), ""]
    for item in items:
        result = execute_sql(con, item.sql, max_rows=1000)
        order = " · order matters" if item.order_matters else ""
        lines += [f"## {item.id}", "",
                  f"{item.category} · {item.difficulty} · scope {item.scope}{order} · review: {item.review}", "",
                  f"- KO: {item.question_ko}", f"- EN: {item.question_en}"]
        if item.note:
            lines.append(f"- Note: {item.note}")
        if item.why_hard:
            lines.append(f"- Why hard: {item.why_hard}")
        if not item.answerable:
            lines += ["", "Unanswerable: no reference SQL; the correct answer is to abstain.", ""]
            continue
        lines += ["", "```sql", item.sql, "```", ""]
        if result.error:
            lines += [f"**Reference fails: {result.error}: {result.detail}**", ""]
            continue
        lines += [f"Result: {len(result.rows)} row(s) x {len(result.columns)} column(s)", "",
                  "| " + " | ".join(_md_cell(c) for c in result.columns) + " |",
                  "|" + "---|" * len(result.columns)]
        lines += ["| " + " | ".join(_md_cell(v) for v in row) + " |" for row in result.rows[:rows]]
        if len(result.rows) > rows:
            lines.append(f"| ... {len(result.rows) - rows} more |" + " |" * (len(result.columns) - 1))
        lines.append("")
    text = "\n".join(lines)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text, encoding="utf-8")
        typer.echo(f"wrote {out} ({len(items)} questions)")
    else:
        typer.echo(text)


@text2sql_app.command("review-xlsx")
def review_xlsx_cmd(draft: Path | None = typer.Option(None, help="question file (default: the v0 draft)"),
                    out: Path = typer.Option(Path("data/drafts/text2sql-review-v0.xlsx"), help="workbook to write")) -> None:
    """Review workbook for the owner: one row per question with its result, a decision dropdown and a note."""
    from benchtoward.tasks.synthetic_snuh import db_path
    from benchtoward.tasks.text2sql_review import export_review

    _connect().close()  # fails early, with the fetch hint, when the database is missing
    path = export_review(_draft(draft), db_path(get_settings().data_dir), out)
    typer.echo(f"wrote {path}")


@text2sql_app.command("apply-review")
def apply_review_cmd(workbook: Path = typer.Argument(..., help="the filled review workbook"),
                     draft: Path | None = typer.Option(None, help="question file (default: the v0 draft)")) -> None:
    """Write the workbook's decisions (승인/수정 필요/제외) and notes into the draft as review / review_note."""
    from benchtoward.tasks.text2sql_review import apply_review

    counts = apply_review(workbook, _draft(draft))
    typer.echo("applied: " + (", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "nothing"))


@text2sql_app.command("llm-review")
def llm_review_cmd(models: str = typer.Option("openai/gpt-6-astra,google/gemini-4-argon,anthropic/claude-opus-5.5",
                                              help="reviewer model ids on the LLM route, comma separated"),
                   route: str | None = typer.Option(None, help=ROUTE_HELP),
                   draft: Path | None = typer.Option(None, help="question file (default: the v0 draft)"),
                   limit: int | None = typer.Option(None, help="only the first N questions (a pilot)"),
                   budget_usd: float = typer.Option(50.0, help="hard cap for the cache folder (its recorded spend "
                                                               "counts; every attempt reserves its worst case)"),
                   out: Path = typer.Option(Path("data/drafts/text2sql-llm-review"), help="cached answers per model"),
                   workbook: Path = typer.Option(Path("data/drafts/text2sql-review-v0-llm.xlsx"),
                                                 help="review workbook with the consensus filled in")) -> None:
    """Review the draft with frontier models (unanimous approval; two rejections reject; else needs_edit), then write
    a review workbook that `apply-review` can take as is."""
    from benchtoward.tasks.synthetic_snuh import db_path
    from benchtoward.tasks.text2sql_llm_review import run_review, write_review_workbook

    settings = get_settings()
    ids, clients, prices, extra = _frontier(models, route)
    _, items = _questions(draft)
    items = items[:limit] if limit else items
    _connect().close()
    database = db_path(settings.data_dir)
    summary = asyncio.run(run_review(items, database, clients, prices, out, budget_usd=budget_usd, extra=extra))
    typer.echo(f"asked {summary['asked']}, parsed {summary['parsed']}, cost ${summary['cost_usd']:.2f}"
               + (" (stopped at the budget)" if summary["stopped_for_budget"] else ""))
    path = write_review_workbook(_draft(draft), database, out, workbook, models=ids)
    typer.echo(f"wrote {path}")


FRONTIER = "anthropic/claude-opus-5.5,openai/gpt-6-astra,google/gemini-4-argon"
JOB_DIR = "text2sql-model-fixes"  # under <data_dir>/drafts: llm-fix and llm-decide caches; the budget counts both
HARD_JOB_DIR = "text2sql-hard"  # under <data_dir>/drafts: every cache of the harder set's job, one budget
JOB_CAP_USD = {JOB_DIR: 30.0, HARD_JOB_DIR: 25.0}  # each job's hard cap on paid calls (owner); others take v0's
FREE_DRY_RUN = "alibaba__qwen3.8-flash-next-180b-a6b"


def job_cap(job: Path, budget_usd: float | None = None) -> float:
    """The cap for a job folder: `--budget-usd` when given, else the job's own ($25 for the harder set's job, which
    also holds v0's 내원 fixes; $30 for v0's)."""
    return budget_usd if budget_usd is not None else JOB_CAP_USD.get(Path(job).name, JOB_CAP_USD[JOB_DIR])


def _catalog_prices(ids: list[str], url: str) -> dict[str, tuple[float, float]]:
    """USD per 1M tokens in and out from a route's public model list (`prices_url`: OpenAI-style `data`, each with
    `id` and `pricing.prompt` / `pricing.completion` in USD per token); exits on an unknown model id."""
    import httpx

    catalog = {m["id"]: m for m in httpx.get(url, timeout=30).json()["data"]}
    missing = [m for m in ids if m not in catalog]
    if missing:
        typer.echo(f"unknown model ids: {missing}")
        raise typer.Exit(1)
    return {m: (float(catalog[m]["pricing"]["prompt"]) * 1e6, float(catalog[m]["pricing"]["completion"]) * 1e6)
            for m in ids}


def frontier_route(route: str | None) -> tuple[str, dict]:
    """The frontier models' serving route and its registry/providers.yaml entry: `route`, or the one entry marked
    `frontier: true`. Optional fields: `prices_url` (its public model list with prices, for the budget) and
    `request.billing` (request fields that make it report what it billed in `usage.cost`)."""
    routes = provider_routes()
    if route is None:
        marked = [name for name, entry in routes.items() if (entry or {}).get("frontier")]
        if len(marked) != 1:
            typer.echo(f"no --route given, and registry/providers.yaml marks {len(marked) or 'no'} route(s) with "
                       "frontier: true (mark exactly one, or name the route)")
            raise typer.Exit(2)
        route = marked[0]
    if route not in routes:
        typer.echo(f"no route {route!r} in registry/providers.yaml")
        raise typer.Exit(2)
    return route, routes[route] or {}


def _frontier(models: str, route: str | None) -> tuple[list[str], dict, dict, dict | None]:
    settings = get_settings()
    name, entry = frontier_route(route)
    ids = [m.strip() for m in models.split(",") if m.strip()]
    prices = _catalog_prices(ids, entry["prices_url"]) if entry.get("prices_url") else {m: (0.0, 0.0) for m in ids}
    try:
        clients = {m: client_for(Model(slug=m, name=m, type="llm", provider=name, provider_model_id=m, digest=m),
                                 settings) for m in ids}
    except MissingCredential as exc:
        typer.echo(str(exc))
        raise typer.Exit(1)
    billing = (entry.get("request") or {}).get("billing")  # the route then reports what it billed
    return ids, clients, prices, dict(billing) if billing else None


def _job_dir(out: Path | None) -> Path:
    return Path(out) if out else get_settings().data_dir / "drafts" / JOB_DIR


def _spend(asked_from: float, budget) -> str:
    return (f"route-billed ${budget.spent_usd - asked_from:.3f} this run, ${budget.spent_usd:.2f} for the job so far, "
            f"cap ${budget.cap_usd:.2f}")


@text2sql_app.command("llm-fix")
def llm_fix_cmd(models: str = typer.Option(FRONTIER, help="proposers in rotation order; all of them re-review"),
                route: str | None = typer.Option(None, help=ROUTE_HELP),
                draft: Path | None = typer.Option(None, help="question file (default: the v0 draft)"),
                id: list[str] = typer.Option(None, help="only these needs_edit items (repeatable)"),
                budget_usd: float | None = typer.Option(None, help="hard cap for the whole job (every cache under "
                                                                   "the job folder counts; default: the job's, $25 "
                                                                   f"for {HARD_JOB_DIR}, $30 for {JOB_DIR})"),
                out: Path | None = typer.Option(None, help=f"job folder (default: <data_dir>/drafts/{JOB_DIR}, or "
                                                           f"{HARD_JOB_DIR} for the harder set)"),
                review_cache: Path | None = typer.Option(None, help="llm-review answers (default: "
                                                                    "<data_dir>/drafts/text2sql-llm-review; for the "
                                                                    "harder set its hard-review answers in the job)"),
                review_batch: int = typer.Option(5, help="valid proposals of a round re-reviewed per request (each "
                                                         "judged on its own; 1 = one request per item)"),
                apply: bool = typer.Option(True, help="write approved and rejected items into the draft")) -> None:
    """Fix needs_edit items with the frontier models: one proposes (rotating), the reference rules check the SQL,
    all re-review; unanimous approval writes the item, three rounds without it drop the item."""
    from benchtoward.tasks.synthetic_snuh import db_path
    from benchtoward.tasks.text2sql import prompt_spec
    from benchtoward.tasks.text2sql_hard import REVIEW_DIR, collect_reviews
    from benchtoward.tasks.text2sql_llm_fix import Budget, apply_fixes, run_fix, write_report
    from benchtoward.tasks.text2sql_llm_review import collect

    ids, clients, prices, extra = _frontier(models, route)
    meta, items = _questions(draft)
    spec = prompt_spec(meta)
    _connect().close()
    job = _hard_job(out) if spec.abstention and not out else _job_dir(out)  # the harder set has its own job
    raw = yaml.safe_load(_draft(draft).read_text(encoding="utf-8"))
    notes = {q["id"]: q.get("review_note") for q in raw.get("questions") or []}
    if spec.abstention and not review_cache:  # the harder set: its own review, kept in the job folder
        from benchtoward.tasks.text2sql import schema_summary
        from benchtoward.tasks.text2sql_llm_fix import _sha
        from benchtoward.tasks.text2sql_llm_review import render_batch_review_messages, render_review_messages

        con = _connect()
        schema = schema_summary(con, null_columns=spec.null_columns)

        def current(group):  # the first review's prompt for these items as they read now (alone or as a batch)
            return _sha(render_review_messages(group[0], con, schema, spec) if len(group) == 1 else
                        render_batch_review_messages(group, con, schema, spec))
        cached = collect_reviews(job / REVIEW_DIR, ids, items, current=current)
    else:
        from benchtoward.tasks.text2sql import schema_summary
        from benchtoward.tasks.text2sql_llm_review import _sha, render_review_messages

        cache = Path(review_cache) if review_cache else get_settings().data_dir / "drafts" / "text2sql-llm-review"
        con = _connect()
        schema = schema_summary(con)
        cached = collect(cache, ids, [i for i in items if i.review == "needs_edit"],
                         current=lambda i: _sha(render_review_messages(i, con, schema))) if cache.exists() else {}
    # Why each item needs an edit: the reviewers' verdicts on its current wording when one of them asked for it;
    # otherwise (an item sent back by a decision on the draft, perhaps after an earlier fix) its review note.
    asked = {i.id for i in items if any(r and r.get("verdict") != "approve" for r in (cached.get(i.id) or {}).values())}
    original = {i.id: cached[i.id] if i.id in asked else notes.get(i.id) for i in items}
    budget = Budget(job_cap(job, budget_usd), Budget.spent_in(job))
    before = budget.spent_usd
    outcomes = asyncio.run(run_fix(items, db_path(get_settings().data_dir), clients, prices, job / "fix", budget,
                                   proposers=ids, original_reviews=original, ids=id or None, extra=extra, spec=spec,
                                   review_batch=review_batch))
    write_report(job / f"fix-report-{_draft(draft).stem}.json", outcomes)
    for o in outcomes:
        where = f" in round {o.round} ({o.proposer.split('/')[-1]})" if o.status == "approved" else ""
        typer.echo(f"{o.id}: {o.status}{where}" + (f" after {len(o.rounds)} rounds" if o.status == "rejected" else ""))
    counts = Counter(o.status for o in outcomes)
    typer.echo(", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "no needs_edit items")
    typer.echo(_spend(before, budget) + (" (stopped at the budget)" if budget.stopped else ""))
    if apply and (counts["approved"] or counts["rejected"]):
        written = apply_fixes(_draft(draft), outcomes)
        typer.echo(f"wrote into {_draft(draft)}: " + ", ".join(f"{k} {v}" for k, v in sorted(written.items())))
    if counts["incomplete"]:
        typer.echo("incomplete items stay needs_edit; run the same command again to resume")
        raise typer.Exit(1)


@text2sql_app.command("llm-decide")
def llm_decide_cmd(models: str = typer.Option(FRONTIER, help="voter model ids on the route, comma separated"),
                   route: str | None = typer.Option(None, help=ROUTE_HELP),
                   draft: Path | None = typer.Option(None, help="question file (default: the v0 draft)"),
                   budget_usd: float | None = typer.Option(None, help="hard cap for the whole job (every cache "
                                                                      "under the job folder counts; default: the "
                                                                      f"job's, $25 for {HARD_JOB_DIR}, $30 for "
                                                                      f"{JOB_DIR})"),
                   out: Path | None = typer.Option(None, help=f"job folder (default: <data_dir>/drafts/{JOB_DIR}, or "
                                                              f"{HARD_JOB_DIR} for the harder set)"),
                   dry_run: Path | None = typer.Option(None, help="dry-run summary.json for the facts (default: the "
                                                                  "free model's; for the harder set its runs at "
                                                                  "16,384 and 32,768 output tokens)"),
                   apply: bool = typer.Option(True, help="record the decisions in the draft"),
                   freeze: bool = typer.Option(False, help="the freeze decisions about both tiers: the same prompt "
                                                           "from either draft, so the second draft's run is served "
                                                           "from the cache (job: the harder set's)"),
                   easy_draft: Path | None = typer.Option(None, help="--freeze: the easy tier's draft shown to the "
                                                                     "voters (default: v0)"),
                   hard_draft: Path | None = typer.Option(None, help="--freeze: the hard tier's draft shown to the "
                                                                     "voters (default: the harder set)"),
                   evidence: Path = typer.Option(Path("docs/text2sql/2026-10-30-freeze-evidence.md"),
                                                 help="--freeze: the measured evidence given to the voters"),
                   sop: Path = typer.Option(Path("docs/sop/benchmarks-and-models.md"),
                                            help="--freeze: the SOP whose Checks section is given to the voters")
                   ) -> None:
    """Vote on the draft's open decisions with the frontier models: two of three decide, else the draft's current
    behaviour stays. The harder set has its own decisions, job folder and facts; --freeze asks the decisions about
    both tiers."""
    from benchtoward.tasks.text2sql import load_questions, prompt_spec
    from benchtoward.tasks.text2sql_llm_decide import (FREEZE, HARD, SOP_CHECKS, V0, facts_text, freeze_facts,
                                                    hard_facts_text, run_decide, sop_section, write_decisions)
    from benchtoward.tasks.text2sql_llm_fix import FIX_DATE, Budget

    ids, clients, prices, extra = _frontier(models, route)
    meta, items = _questions(draft)
    hard = prompt_spec(meta).abstention
    setup = FREEZE if freeze else HARD if hard else V0
    if freeze:
        try:
            checks = sop_section(Path(sop).read_text(encoding="utf-8"), SOP_CHECKS)
        except ValueError as exc:
            typer.echo(str(exc))
            raise typer.Exit(2)
        facts = freeze_facts(load_questions(_draft(easy_draft))[0], load_questions(_hard_draft(hard_draft))[0],
                             checks, Path(evidence).read_text(encoding="utf-8"))
        job = _hard_job(out)
    elif hard:
        base = get_settings().data_dir / "drafts" / _draft(draft).stem
        labelled = [(Path(dry_run).parent, "the output limit of that run")] if dry_run else [
            (base / FREE_DRY_RUN, "the registry's 16,384 output tokens"),
            (base / f"{FREE_DRY_RUN}-max32k", "32,768 output tokens (diagnostic)")]
        runs = [(label, json.loads((d / "summary.json").read_text()),
                 [json.loads(line) for line in (d / "items.jsonl").read_text(encoding="utf-8").splitlines()])
                for d, label in labelled if (d / "summary.json").exists() and (d / "items.jsonl").exists()]
        facts = hard_facts_text(items, runs)
        job = _hard_job(out)
    else:
        summary_path = Path(dry_run) if dry_run else get_settings().data_dir / "drafts" / DRAFT_NAME / \
            FREE_DRY_RUN / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
        scored = summary_path.parent / "items.jsonl"
        failures = [r["instance_id"] for r in map(json.loads, scored.read_text(encoding="utf-8").splitlines())
                    if not r["correct"]] if summary and scored.exists() else []
        facts = facts_text(items, summary, failures)
        job = _job_dir(out)
    budget = Budget(job_cap(job, budget_usd), Budget.spent_in(job))
    before = budget.spent_usd
    results = asyncio.run(run_decide(_draft(draft), clients, prices, job / "decide", budget, facts=facts,
                                     extra=extra, setup=setup))
    for r in results:
        if r["status"] != "decided":
            typer.echo(f"{r['key']}: incomplete")
            continue
        n = sum(1 for v in r["votes"].values() if v and v["choice"] == r["outcome"])
        how = f"{n} of {len(r['votes'])}" if r["majority"] else f"no majority, {setup.fallback} stays"
        typer.echo(f"{r['key']}: {r['outcome']} ({how}; {setup.status[r['kind']]})")
    typer.echo(_spend(before, budget) + (" (stopped at the budget)" if budget.stopped else ""))
    if apply:
        written = write_decisions(_draft(draft), results, date=FIX_DATE, setup=setup)
        typer.echo(f"recorded {len(written)} decision(s) in {_draft(draft)}")
    if any(r["status"] != "decided" for r in results):
        raise typer.Exit(1)


@text2sql_app.command("send-back")
def send_back_cmd(id: list[str] = typer.Option(..., help="item ids to send back (repeatable)"),
                  reason: str = typer.Option(..., help="why, e.g. a decision on the draft; llm-fix gives it to the "
                                                       "proposer"),
                  draft: Path | None = typer.Option(None, help="question file (default: the v0 draft)")) -> None:
    """Set items back to needs_edit with a reason (e.g. a decision on the draft), so that `llm-fix` revises them."""
    from benchtoward.tasks.text2sql_hard import send_back

    try:
        sent = send_back(_draft(draft), id, reason)
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(2)
    typer.echo(f"sent back {len(sent)} item(s): {', '.join(sent)}")


def _hard_draft(draft: Path | None) -> Path:
    from benchtoward.tasks.text2sql_hard import HARD_DRAFT_NAME

    return Path(draft) if draft else registry_dir() / "drafts" / f"{HARD_DRAFT_NAME}.yaml"


def _hard_job(out: Path | None) -> Path:
    return Path(out) if out else get_settings().data_dir / "drafts" / HARD_JOB_DIR


@text2sql_app.command("hard-generate")
def hard_generate_cmd(models: str = typer.Option(FRONTIER, help="proposers; batches rotate over them by category"),
                      route: str | None = typer.Option(None, help=ROUTE_HELP),
                      draft: Path | None = typer.Option(None, help="the harder set's draft (default: "
                                                                   "registry/drafts/synthetic-snuh-text2sql-hard-v0)"),
                      plan: str | None = typer.Option(None, help="batch sizes, e.g. 'clinical_terms=8,8;unanswerable=7'"
                                                                 " (default: the module's PLAN)"),
                      gen_round: int = typer.Option(1, "--round", help="generation round (a later round sees the "
                                                                       "earlier ones' items)"),
                      hint: str = typer.Option("", help="extra instruction for this round's proposers"),
                      budget_usd: float = typer.Option(25.0, help="hard cap for the whole job folder"),
                      out: Path | None = typer.Option(None, help=f"job folder (default: <data_dir>/drafts/"
                                                                 f"{HARD_JOB_DIR})"),
                      apply: bool = typer.Option(True, help="append the new items to the draft")) -> None:
    """Write harder items with the frontier models (rotating by category), check every reference with v0's rules
    (failures go back to the proposer once), and append the items that pass to the draft as pending."""
    from benchtoward.tasks.synthetic_snuh import db_path
    from benchtoward.tasks.text2sql import load_questions
    from benchtoward.tasks.text2sql_hard import (GENERATE_DIR, PLAN, append_items, batch_round, draft_batches,
                                              parse_plan, run_generate)
    from benchtoward.tasks.text2sql_llm_fix import Budget

    ids, clients, prices, extra = _frontier(models, route)
    path = _hard_draft(draft)
    _, items = load_questions(path)
    batches = draft_batches(path)
    earlier = [i for i in items if (batch_round(batches.get(i.id)) or 0) < gen_round]
    _connect().close()
    job = _hard_job(out)
    budget = Budget(budget_usd, Budget.spent_in(job))
    before = budget.spent_usd
    result = asyncio.run(run_generate(db_path(get_settings().data_dir), clients, prices, job / GENERATE_DIR, budget,
                                      models=ids, existing=earlier, gen_round=gen_round,
                                      plan=parse_plan(plan) if plan else PLAN, hint=hint, extra=extra))
    for b in result.batches:
        typer.echo(f"{b['batch']} ({b['proposer'].split('/')[-1]}): asked {b['asked']}, proposed {b['proposed']}, "
                   f"passed {b['passed']}, sent back {b['sent_back']}, repaired {b['repaired']}")
    report = job / f"generate-report-g{gen_round}.json"
    report.write_text(json.dumps({"batches": result.batches, "sent_back": result.sent_back, "dropped": result.dropped,
                                  "incomplete": result.incomplete}, indent=1, ensure_ascii=False))
    present = set(batches.values())
    new = [r for r in result.items if r["batch"] not in present]
    written = append_items(path, new) if apply and new else []
    typer.echo(f"{len(result.items)} items passed, {len(result.dropped)} dropped after the repair; "
               f"{len(written)} items written into {path}")
    typer.echo(_spend(before, budget) + (" (stopped at the budget)" if budget.stopped else ""))
    if result.incomplete:
        typer.echo(f"incomplete batches: {', '.join(result.incomplete)}; run the same command again to resume")
        raise typer.Exit(1)


@text2sql_app.command("hard-review")
def hard_review_cmd(models: str = typer.Option(FRONTIER, help="reviewer model ids on the route, comma separated"),
                    route: str | None = typer.Option(None, help=ROUTE_HELP),
                    draft: Path | None = typer.Option(None, help="the harder set's draft"),
                    budget_usd: float = typer.Option(25.0, help="hard cap for the whole job folder"),
                    out: Path | None = typer.Option(None, help=f"job folder (default: <data_dir>/drafts/"
                                                               f"{HARD_JOB_DIR})"),
                    apply: bool = typer.Option(True, help="write the decisions into the draft")) -> None:
    """Review every pending item of the harder set with the frontier models (v0's rubric extended for the harder set
    and unanswerable items). Unanimous approval keeps an item; anything else marks it needs_edit for `llm-fix`."""
    from benchtoward.tasks.synthetic_snuh import db_path
    from benchtoward.tasks.text2sql import load_questions
    from benchtoward.tasks.text2sql_hard import REVIEW_DIR, apply_reviews, run_hard_review
    from benchtoward.tasks.text2sql_llm_fix import Budget

    ids, clients, prices, extra = _frontier(models, route)
    path = _hard_draft(draft)
    _, items = load_questions(path)
    _connect().close()
    job = _hard_job(out)
    budget = Budget(budget_usd, Budget.spent_in(job))
    before = budget.spent_usd
    reviews = asyncio.run(run_hard_review(items, db_path(get_settings().data_dir), clients, prices, job / REVIEW_DIR,
                                          budget, extra=extra))
    incomplete = sorted(k for k, v in reviews.items() if v is None)
    counts = apply_reviews(path, reviews) if apply else {}
    typer.echo(f"reviewed {len(reviews) - len(incomplete)} of {len(reviews)} pending item(s): "
               + (", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "nothing written"))
    typer.echo(_spend(before, budget) + (" (stopped at the budget)" if budget.stopped else ""))
    if incomplete:
        typer.echo(f"{len(incomplete)} item(s) still pending; run the same command again to resume")
        raise typer.Exit(1)


@text2sql_app.command("spend")
def spend_cmd(out: Path | None = typer.Option(None, help=f"job folder (default: <data_dir>/drafts/{HARD_JOB_DIR})")
              ) -> None:
    """What a job folder's paid calls cost: route-billed, estimated, and charged for failed attempts, by step."""
    from benchtoward.tasks.text2sql_hard import spend_report

    report = spend_report(_hard_job(out))
    typer.echo(f"route-billed ${report['route_usd']:.4f}, estimated ${report['estimate_usd']:.4f}, charged for "
               f"failed attempts ${report['failed_attempts_usd']:.4f}; {report['requests']} requests")
    for step, s in sorted(report["by_step"].items()):
        typer.echo(f"  {step}: ${s['usd']:.4f} ({s['requests']} requests)")


@text2sql_app.command("prompt")
def prompt_cmd(question_id: str = typer.Argument(..., help="e.g. syn-sql-001"),
               lang: str = typer.Option("en", help="ko or en"),
               draft: Path | None = typer.Option(None, help="question file (default: the v0 draft)")) -> None:
    """Print the exact messages a model receives for one question."""
    from benchtoward.tasks.text2sql import prompt_spec, render_messages, schema_summary

    meta, items = _questions(draft)
    spec = prompt_spec(meta)
    item = next((i for i in items if i.id == question_id), None)
    if item is None:
        typer.echo(f"no question {question_id}")
        raise typer.Exit(2)
    for message in render_messages(item, lang, schema_summary(_connect(), null_columns=spec.null_columns), spec):
        typer.echo(f"### {message['role']}\n{message['content']}\n")


def _load_predictions(path: Path) -> dict[str, str | None]:
    text = Path(path).read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
        records = [{"instance_id": k, "sql": v} for k, v in data.items()] if isinstance(data, dict) else data
    out = {}
    for r in records:
        key = r.get("instance_id") or f"{r['question_id']}-{r['lang']}"
        out[key] = r.get("sql")
    return out


def _fmt(rate: dict) -> str:
    return f"{rate['value']:.3f} (n={rate['n']})"


def _print_summary(summary: dict, out: Path) -> None:
    ex = summary["execution_accuracy"]
    ci = f" (95% CI {ex['ci_low']:.3f}-{ex['ci_high']:.3f})" if ex.get("ci_low") is not None else ""
    typer.echo(f"model: {summary.get('model', '-')} | prompt {summary['prompt']} | scorer {summary['scorer']}")
    if "headline" in summary:
        typer.echo(f"headline {summary['headline']['metric']} {summary['headline']['value']:.3f} (the draft's "
                   "protocol.headline)")
    scope = " on answerable items" if "reliability" in summary else ""
    typer.echo(f"execution accuracy{scope} {ex['value']:.3f}{ci}, n={ex['n']}")
    if "reliability" in summary:
        rs = summary["reliability"]
        typer.echo("reliability score (EHRSQL-2024): " + ", ".join(
            f"{k} {v:.3f}" for k, v in rs.items() if k.startswith("RS(")) + f" (N={rs['N']})")
        typer.echo("  by language: " + "; ".join(f"{lang} " + ", ".join(f"{k} {v:.3f}" for k, v in scores.items())
                                               for lang, scores in rs["by_language"].items()))
        ab = summary["abstention"]
        typer.echo(f"abstained on unanswerable {_fmt(ab['unanswerable_abstained'])}; "
                   f"on answerable {_fmt(ab['answerable_abstained'])}")
    for field in ("language", "category", "difficulty"):
        groups = summary[f"by_{field}"]
        typer.echo(f"by {field}: " + ", ".join(f"{k} {_fmt(v)}" for k, v in groups.items()))
    typer.echo(f"SQL found in {summary['sql_rate']:.1%} of answers; execution errors {summary['execution_error_rate']:.1%};"
               f" errors: {summary['errors'] or 'none'}")
    if "tokens_in" in summary:
        typer.echo(f"tokens: in {summary['tokens_in']:,}, out {summary['tokens_out']:,}")
    typer.echo(f"report: {out / 'items.jsonl'}")


@text2sql_app.command("dry-run")
def dry_run_cmd(model: str | None = typer.Option(None, help="registry model slug, e.g. alibaba/qwen3.8-flash-next-180b-a6b"),
                predictions: Path | None = typer.Option(None, help="score SQL from a file instead (.jsonl with "
                                                        "instance_id and sql, or a .json/.yaml mapping); for the "
                                                        "harder set null, 'null' or ABSTAIN (any case) abstains and "
                                                        "an empty string is no answer"),
                draft: Path | None = typer.Option(None, help="question file (default: the v0 draft)"),
                lang: str = typer.Option("both", help="ko, en or both"),
                limit: int | None = typer.Option(None, help="only the first N questions"),
                concurrency: int = typer.Option(8, help="parallel LLM calls"),
                max_tokens: int | None = typer.Option(None, help="output limit per answer (default: the registry's); "
                                                                 "for diagnostic runs only"),
                out: Path | None = typer.Option(None, help="output directory (default: <data_dir>/drafts/...)")) -> None:
    """Score a model's SQL against the draft questions. Results go to files only, never to the medbench database."""
    from benchtoward.evaluate import max_tokens_for
    from benchtoward.tasks.text2sql import LANGS, normalize_prediction, prompt_spec, run_dry_run, score_predictions

    if (model is None) == (predictions is None):
        typer.echo("give exactly one of --model or --predictions")
        raise typer.Exit(2)
    langs = LANGS if lang == "both" else (lang,)
    if any(x not in LANGS for x in langs):
        typer.echo(f"--lang must be ko, en or both, not {lang}")
        raise typer.Exit(2)
    meta, items = _questions(draft)
    spec = prompt_spec(meta)
    dropped = [i for i in items if i.review == "rejected"]
    if dropped:
        typer.echo(f"{len(dropped)} rejected item(s) left out: {', '.join(i.id for i in dropped)}")
    items = [i for i in items if i.review != "rejected"]
    items = items[:limit] if limit else items
    con = _connect()
    base = get_settings().data_dir / "drafts" / _draft(draft).stem
    if model is not None:
        row = _registry_model(model)
        out = Path(out) if out else base / row.slug.replace("/", "__")
        try:
            client = client_for(row, get_settings())
        except MissingCredential as exc:
            typer.echo(str(exc))
            raise typer.Exit(1)
        summary = asyncio.run(run_dry_run(items, con, client, row.provider_model_id, out, langs=langs,
                                          concurrency=concurrency, max_tokens=max_tokens or max_tokens_for(row),
                                          spec=spec))
        if summary["status"] != "done":
            typer.echo(f"incomplete: {summary['pending']} answer(s) still pending after transient errors; "
                       f"run the same command again to resume")
            raise typer.Exit(1)
        summary["model"] = row.slug
    else:
        wanted = {f"{i.id}-{x}" for i in items for x in langs}
        preds = {k: normalize_prediction(v, abstention=spec.abstention)
                 for k, v in _load_predictions(Path(predictions)).items() if k in wanted}
        out = Path(out) if out else base / f"predictions-{Path(predictions).stem}"
        out.mkdir(parents=True, exist_ok=True)
        rows, summary = score_predictions(con, items, preds, abstention=spec.abstention)
        summary["prompt"] = spec.id
        summary = {"status": "done", "model": f"file:{Path(predictions).name}", **summary}
        with (out / "items.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    headline = (meta.get("protocol") or {}).get("headline")  # the harder set: RS(0) (decision headline_metric)
    if headline:
        value = summary["execution_accuracy"]["value"] if headline == "EX" else \
            (summary.get("reliability") or {}).get(headline)
        if value is None:
            typer.echo(f"protocol.headline {headline} is not a score of this run (EX, RS(0), RS(5) or RS(N))")
            raise typer.Exit(2)
        summary["headline"] = {"metric": headline, "value": value}
        stored = json.loads((out / "summary.json").read_text())
        (out / "summary.json").write_text(json.dumps({**stored, "headline": summary["headline"]}, indent=1,
                                                     ensure_ascii=False))
    _print_summary(summary, out)


@text2sql_app.command("khdp-package")
def khdp_package_cmd(out: Path | None = typer.Option(None, help="release folder (default: <data_dir>/khdp/"
                                                                "synthetic-snuh-text2sql-1.0.0)"),
                     code_commit: str | None = typer.Option(None, help="the commit of github.com/medbench-kr/"
                                                                       "benchmarks that holds the evaluation code"),
                     force: bool = typer.Option(False, help="replace an earlier build in the folder")) -> None:
    """Build the KHDP dataset of the frozen items of both tiers (synthetic-snuh-text2sql 1.0.0) with the prompts,
    metadata, README, licence and SHA-256 sums; refused when the items, prompts or schema differ from what the task
    spec pins. Submitting it to KHDP is the maintainers' (SOP: KHDP datasets)."""
    from benchtoward.tasks.text2sql_release import RELEASE_DIR, build_release
    from benchtoward.tasks.text2sql_task import SLUG

    path = registry_dir() / "tasks" / f"{SLUG}.yaml"
    if not path.exists():
        typer.echo(f"{path} not found: the release is built from the frozen files the task spec pins")
        raise typer.Exit(1)
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    target = Path(out) if out else get_settings().data_dir / "khdp" / RELEASE_DIR
    if target.exists() and any(target.iterdir()) and not force:
        typer.echo(f"{target} holds an earlier build: pass --force to replace it")
        raise typer.Exit(1)
    con = _connect()
    try:
        meta = build_release(spec, registry_dir(), con, target, code_commit=code_commit)
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1)
    for name, tier in meta["tiers"].items():
        typer.echo(f"{name}: {tier['questions']} items ({tier['unanswerable']} unanswerable), items_sha256 "
                   f"{tier['items_sha256'][:12]}, prompt {tier['prompt']['id']}")
    typer.echo(f"wrote {target} ({meta['instances']} instances)")
    if not code_commit:
        typer.echo("no code commit in the README yet: after the public export, build again with --code-commit <sha> "
                   "--force before submitting")
