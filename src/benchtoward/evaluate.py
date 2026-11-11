import asyncio
import json
import uuid
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from benchtoward.db import ItemResult, Model, Run, Task, utcnow
from benchtoward.llm import ChatResult, cost_usd
from benchtoward.runkey import run_key
from benchtoward.scoring import accuracy_ci
from benchtoward.tasks.clinicalqa import PARSER_VERSION, PROMPT_ID, QAItem, parse_answer, render_messages

HARNESS_VERSION = "p0.1"
DEFAULT_MAX_TOKENS = 16384  # thinking models need room: 4096 truncated 3 of 10 ClinicalQA items
API_ERROR_FAIL_RATE = 0.01  # owner, 2026-11-05: inclusive; final call errors after harness retries


def max_tokens_for(model: Model) -> int:
    return model.max_tokens or DEFAULT_MAX_TOKENS


def _retry(result: ItemResult) -> bool:
    """On resume retry failed calls, including terminal HTTP refusals; keep all model answers."""
    return result.error is not None and result.error != "no_output"


def _model_results(session: Session, run: Run, instance_ids) -> dict[str, ItemResult]:
    """Include uncalled items for diagnostic scoring without storing fictitious API failures."""
    stored = {r.instance_id: r for r in session.query(ItemResult).filter_by(run_id=run.id)}
    results = {}
    for iid in instance_ids:
        result = stored.get(iid)
        if result is None:
            result = ItemResult(run_id=run.id, instance_id=iid, tokens_in=0, tokens_out=0)
            result._not_called = True
        results[iid] = result
    return results


def finish_model_run(run: Run, results) -> str | None:
    """Apply the shared API failure rule, retaining metrics and other internal diagnostics.

    ItemResult.error holds the final call failure, never a parser failure or a length cutoff. A successful
    retry overwrites that error. Budget stops and absent offline outputs are not API failures. The denominator
    is evaluated items/calls (two notes per error_pair), not necessarily Run.n_items, which can count pairs.
    Also keep interrupted/budget-stopped runs failed, so their missing items can be resumed.
    """
    results = list(results)
    errors = Counter(r.error for r in results if r.error and r.error not in ("budget", "no_output"))
    count, total = sum(errors.values()), len(results)
    reason = None
    if total and count / total >= API_ERROR_FAIL_RATE:
        common = sorted(errors, key=lambda e: (-errors[e], e))[0]
        reason = (f"API errors {count}/{total} ({count / total:.2%}) >= {API_ERROR_FAIL_RATE:.0%}; "
                  f"most common error: {common} ({errors[common]})")
    else:
        missing = sum(getattr(r, "_not_called", False) or r.error == "budget" for r in results)
        if missing:
            reason = f"Incomplete run: {missing}/{total} item(s) still need a model call"
    internal = json.loads(run.internal_json or "{}")
    if reason:
        internal["failed_reason"] = reason
    else:
        internal.pop("failed_reason", None)
    run.internal_json = json.dumps(internal, ensure_ascii=False)
    run.status = "failed" if reason else "done"
    return reason


def _new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def _find_run(session: Session, key: str) -> Run | None:
    return session.query(Run).filter_by(run_key=key).one_or_none()


def write_predictions(run: Run, rows: list[dict], data_dir: Path) -> Path:
    path = Path(data_dir) / "artifacts" / run.id / "predictions.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _start_run(session: Session, model: Model, task: Task, key: str, config: dict,
               harness_version: str = HARNESS_VERSION) -> Run:
    run = Run(id=_new_run_id(), run_key=key, model_slug=model.slug, task_id=task.id, harness_version=harness_version,
              config_json=json.dumps(config, sort_keys=True), tier="baseline", status="running",
              task_spec_hash=task.spec_hash)
    session.add(run)
    session.commit()
    return run


async def run_llm_task(session: Session, model: Model, task: Task, items: list[QAItem], client, *,
                       concurrency: int = 8, max_tokens: int = DEFAULT_MAX_TOKENS, force: bool = False,
                       limit: int | None = None, data_dir: Path | None = None) -> Run:
    if data_dir is None:
        from benchtoward.config import get_settings
        data_dir = get_settings().data_dir
    config = {"prompt": PROMPT_ID, "max_tokens": max_tokens, "temperature": 0, "parser": PARSER_VERSION}
    if limit is not None:
        config["limit"] = limit
        items = items[:limit]
    key = run_key(model.digest, task.spec_hash, HARNESS_VERSION, config)

    run = _find_run(session, key)
    if run is None and not force:   # a finished run from before the parser was in the config (p0.2) still counts:
        legacy = _find_run(session, run_key(model.digest, task.spec_hash, HARNESS_VERSION,
                                            {k: v for k, v in config.items() if k != "parser"}))
        if legacy is not None and legacy.status == "done":   # rescore-clinicalqa replaced every one p0.3 reads
            return legacy                                      # differently, so this one reads the same
    if run is not None and run.status == "done" and not force:
        return run
    if run is None:
        run = _start_run(session, model, task, key, config)
    else:
        if force:
            session.query(ItemResult).filter_by(run_id=run.id).delete()
        run.status = "running"
        session.commit()

    finished = {r.instance_id for r in session.query(ItemResult).filter_by(run_id=run.id) if not _retry(r)}
    todo = [item for item in items if item.instance_id not in finished]
    semaphore = asyncio.Semaphore(concurrency)

    async def one(item: QAItem) -> None:
        async with semaphore:
            try:
                result = await client.chat(model.provider_model_id or model.slug, render_messages(item),
                                                 max_tokens=max_tokens)
            except Exception as exc:  # one broken call must not abort the whole evaluation
                result = ChatResult(content="", reasoning=None, tokens_in=0, tokens_out=0,
                                    error=f"exception_{type(exc).__name__}", transient=True)
        prediction = None if result.error else parse_answer(result.content, set(item.options), item.options)
        session.merge(ItemResult(run_id=run.id, instance_id=item.instance_id, prediction=prediction,
                                 label=item.answer, correct=prediction == item.answer, error=result.error,
                                 transient=result.transient if result.error else None,
                                 raw=(result.content or "")[:20000] or None,
                                 tokens_in=result.tokens_in, tokens_out=result.tokens_out))
        session.commit()

    await asyncio.gather(*(one(item) for item in todo))
    return _finish_llm_run(session, model, run, items, data_dir)


def _finish_llm_run(session: Session, model: Model, run: Run, items: list[QAItem], data_dir: Path) -> Run:
    results = _model_results(session, run, (i.instance_id for i in items))
    run.n_items = len(items)
    run.tokens_in = sum(r.tokens_in for r in results.values())
    run.tokens_out = sum(r.tokens_out for r in results.values())
    run.cost_usd = cost_usd(model, run.tokens_in, run.tokens_out)
    run.eval_date = date.today()
    ordered = [results[item.instance_id] for item in items]
    run.metrics_json = json.dumps(_clinicalqa_metrics(items, ordered))
    finish_model_run(run, results.values())
    run.finished_at = utcnow()
    session.commit()
    write_predictions(run, [{"instance_id": r.instance_id, "prediction": r.prediction, "label": r.label,
                             "correct": r.correct} for r in ordered], data_dir)
    return run


def _clinicalqa_metrics(items: list[QAItem], ordered: list[ItemResult]) -> dict:
    correct = [None if r.prediction is None else bool(r.correct) for r in ordered]
    by_category: dict[str, list[bool]] = defaultdict(list)
    for item, r in zip(items, ordered):
        by_category[item.category].append(bool(r.correct))
    return {
        "accuracy": accuracy_ci(correct),
        "parse_rate": sum(c is not None for c in correct) / len(correct),
        "error_rate": sum(r.error is not None for r in ordered) / len(ordered),
        "parser": PARSER_VERSION,
        "by_category": {cat: {"value": sum(v) / len(v), "n": len(v)} for cat, v in sorted(by_category.items())},
    }


def rescore_clinicalqa(session: Session, task: Task, items: list[QAItem], *, data_dir: Path | None = None,
                       dry_run: bool = False) -> list[Run]:
    """Re-read every finished ClinicalQA run made with an older parser from its stored replies, without model calls
    (parser p0.3, owner 2026-11-03). Each becomes a new run whose config names the parser, so its run key differs and
    import-runs carries it to the site; the leaderboard lists a model's latest run, so the new one replaces the old,
    which stays as it was. Letters already read are kept (p0.3 applies p0.2 first); only unread replies that were
    stored are read again, and a run where nothing changes is left alone (its result URL too). dry_run returns
    the new runs, with metrics and `rescored_from`, without writing anything."""
    if data_dir is None:
        from benchtoward.config import get_settings
        data_dir = get_settings().data_dir
    by_id = {item.instance_id: item for item in items}
    position = {iid: n for n, iid in enumerate(by_id)}
    made = []
    for old in session.query(Run).filter(Run.task_id == task.id, Run.status.in_(("done", "held"))).order_by(Run.created_at):
        config = json.loads(old.config_json or "{}")
        if config.get("parser") == PARSER_VERSION or "prompt" not in config:
            continue
        model = session.get(Model, old.model_slug)
        olds = session.query(ItemResult).filter_by(run_id=old.id).all()
        if model is None or not olds or any(r.instance_id not in by_id for r in olds):
            continue
        config["parser"] = PARSER_VERSION
        key = run_key(model.digest, old.task_spec_hash or task.spec_hash, old.harness_version, config)
        if _find_run(session, key) is not None:
            continue
        # p0.3 runs p0.2 first, so a letter already read stays; only unread replies that were stored are read again.
        # Most runs keep the letter but not the reply (raw NULL); those items keep what they had.
        rereads = {r.instance_id: parse_answer(r.raw, set(by_id[r.instance_id].options), by_id[r.instance_id].options)
                   for r in olds if r.prediction is None and not r.error and r.raw}
        if not any(rereads.values()):
            continue   # nothing changes: keep the run (and its result URL) as it is
        internal = {k: v for k, v in json.loads(old.internal_json or "{}").items() if k != "checks"}
        internal["rescored_from"] = old.id   # the old run's checks described its numbers; they are not carried over
        new = Run(id=_new_run_id(), run_key=key, model_slug=old.model_slug, task_id=task.id,
                  harness_version=old.harness_version, config_json=json.dumps(config, sort_keys=True), tier=old.tier,
                  status=old.status, eval_date=old.eval_date, n_items=old.n_items, tokens_in=old.tokens_in,
                  tokens_out=old.tokens_out, gpu_seconds=old.gpu_seconds, cost_usd=old.cost_usd,
                  task_spec_hash=old.task_spec_hash, internal_json=json.dumps(internal, ensure_ascii=False))
        if not dry_run:
            session.add(new)
            session.flush()
        ordered_items, ordered = [], []
        for r in sorted(olds, key=lambda r: position[r.instance_id]):
            item = by_id[r.instance_id]
            prediction = r.prediction if r.prediction is not None else rereads.get(r.instance_id)
            copy = ItemResult(run_id=new.id, instance_id=r.instance_id, prediction=prediction, label=r.label,
                              correct=prediction == r.label, score=r.score, error=r.error, transient=r.transient,
                              raw=r.raw, tokens_in=r.tokens_in, tokens_out=r.tokens_out)
            if not dry_run:
                session.add(copy)
            ordered_items.append(item)
            ordered.append(copy)
        new.metrics_json = json.dumps(_clinicalqa_metrics(ordered_items, ordered))
        new.finished_at = utcnow()
        if dry_run:
            made.append(new)
            continue
        session.commit()
        write_predictions(new, [{"instance_id": r.instance_id, "prediction": r.prediction, "label": r.label,
                                 "correct": r.correct} for r in ordered], data_dir)
        made.append(new)
    return made


ASR_HARNESS_VERSION = "asr-p0.1"


async def run_asr_task(session: Session, model: Model, task: Task, clips: list, client, read_audio, *,
                       concurrency: int = 4, force: bool = False, limit: int | None = None,
                       data_dir: Path | None = None, language: str = "ko", progress=None,
                       max_consecutive_failures: int = 25) -> Run:
    """Transcribe every clip once (resumable, like run_llm_task) and score CER/WER on the control plane.

    `client.transcribe(model_id, audio, filename, language=, temperature=)` returns a Transcription; `read_audio(clip)`
    returns the WAV bytes. Failed calls count as empty transcripts for diagnostics; at least 1% fails the run,
    and resuming retries its failed calls. An account-level failure (token, credits, model) stops sending further clips,
    and so do `max_consecutive_failures` transient failures in a row (an endpoint that is down, e.g. a preempted pod).
    """
    from benchtoward.asr import Transcription
    from benchtoward.tasks.k_or_speech import score_item

    if data_dir is None:
        from benchtoward.config import get_settings
        data_dir = get_settings().data_dir
    config = {"language": language, "response_format": "json", "temperature": 0}
    if limit is not None:
        config["limit"] = limit
        clips = clips[:limit]
    key = run_key(model.digest, task.spec_hash, ASR_HARNESS_VERSION, config)

    run = _find_run(session, key)
    if run is not None and run.status == "done" and not force:
        return run
    if run is None:
        run = _start_run(session, model, task, key, config, harness_version=ASR_HARNESS_VERSION)
    else:
        if force:
            session.query(ItemResult).filter_by(run_id=run.id).delete()
        run.status = "running"
        session.commit()

    finished = {r.instance_id for r in session.query(ItemResult).filter_by(run_id=run.id) if not _retry(r)}
    todo = [clip for clip in clips if clip.instance_id not in finished]
    semaphore = asyncio.Semaphore(concurrency)
    stop = asyncio.Event()
    done_count = consecutive_failures = 0

    async def one(clip) -> None:
        nonlocal done_count, consecutive_failures
        async with semaphore:
            if stop.is_set():
                return
            try:
                result = await client.transcribe(model.provider_model_id, read_audio(clip), clip.filename,
                                                 language=language, temperature=0)
            except Exception as exc:  # one broken call must not abort the whole evaluation
                result = Transcription(text="", audio_seconds=0, raw=None, error=f"exception_{type(exc).__name__}",
                                       transient=True)
        consecutive_failures = consecutive_failures + 1 if result.error and result.transient else 0
        if result.fatal or consecutive_failures >= max_consecutive_failures:
            stop.set()
        hypothesis = None if result.error else result.text
        s = score_item(clip.reference, hypothesis)
        session.merge(ItemResult(run_id=run.id, instance_id=clip.instance_id, prediction=hypothesis,
                                 label=clip.reference, score=s.cer, correct=s.char_errors == 0, error=result.error,
                                 transient=result.transient if result.error else None,
                                 raw=(result.raw or "")[:20000] or None))
        session.commit()
        done_count += 1
        if progress is not None:
            progress(done_count, len(todo))

    await asyncio.gather(*(one(clip) for clip in todo))
    return _finish_asr_run(session, model, run, clips, data_dir)


def _audio_seconds(raw: str | None) -> float:
    try:
        usage = json.loads(raw or "{}").get("usage") or {}
        return float(usage.get("seconds") or 0)
    except (ValueError, AttributeError, TypeError):
        return 0.0


def _finish_asr_run(session: Session, model: Model, run: Run, clips: list, data_dir: Path) -> Run:
    from benchtoward.asr import audio_cost_usd, billed_minutes
    from benchtoward.tasks.k_or_speech import asr_metrics, item_rows

    results = _model_results(session, run, (c.instance_id for c in clips))
    seconds = [_audio_seconds(r.raw) for r in results.values() if r.error is None]
    run.n_items = len(clips)
    run.cost_usd = audio_cost_usd(model, seconds)
    run.eval_date = date.today()
    ordered = [results[clip.instance_id] for clip in clips]
    hypotheses, errors = [r.prediction for r in ordered], [r.error for r in ordered]
    rows = item_rows(clips, hypotheses, errors)
    for r, row in zip(ordered, rows):  # rescore from stored transcripts so resumed items use the current normaliser
        r.score, r.correct = row["cer"], row["char_errors"] == 0
    metrics = asr_metrics(clips, hypotheses, errors)
    metrics["audio_seconds"] = sum(seconds)
    metrics["billed_minutes"] = sum(billed_minutes(s) for s in seconds)
    run.metrics_json = json.dumps(metrics, ensure_ascii=False)
    finish_model_run(run, results.values())
    run.finished_at = utcnow()
    session.commit()
    write_predictions(run, rows, data_dir)
    return run


def record_baseline_run(session: Session, model: Model, task: Task, rows: list[dict], metrics: dict, config: dict,
                        *, data_dir: Path | None = None) -> Run:
    """Store a finished run computed outside the LLM runner (BenchToward baselines). Deduplicated by run key."""
    if data_dir is None:
        from benchtoward.config import get_settings
        data_dir = get_settings().data_dir
    key = run_key(model.digest, task.spec_hash, HARNESS_VERSION, config)
    run = _find_run(session, key)
    if run is not None and run.status == "done":
        return run
    if run is None:
        run = _start_run(session, model, task, key, config)
    for row in rows:
        session.merge(ItemResult(run_id=run.id, instance_id=row["instance_id"], prediction=row.get("prediction"),
                                 label=None if row.get("label") is None else str(row["label"]),
                                 correct=row.get("correct"), score=row.get("score")))
    run.n_items = len(rows)
    run.metrics_json = json.dumps(metrics)
    run.eval_date = date.today()
    run.status = "done"
    run.finished_at = utcnow()
    session.commit()
    write_predictions(run, rows, data_dir)
    return run


def constant_answer_baseline(items: list[QAItem], letter: str) -> tuple[list[dict], dict]:
    """Predict the same letter for every item (reference point for multiple-choice tasks)."""
    rows = [{"instance_id": it.instance_id, "prediction": letter, "label": it.answer, "correct": it.answer == letter}
            for it in items]
    metrics = {"accuracy": accuracy_ci([r["correct"] for r in rows]), "parse_rate": 1.0}
    return rows, metrics


class SyncConflict(RuntimeError):
    """import_bundle refuses a bundle that cannot be merged without losing or duplicating results."""


def rekey_runs(session: Session) -> int:
    """Recompute run keys after the digest or run-key definition changes, from each run's own task spec hash. A run's
    spec hash is never moved here (that would list stale runs and collide with the current ones): a rename-only spec
    edit moves hashes through a rename plan (benchtoward.rename). Returns runs changed; refuses colliding keys."""
    digests = dict(session.query(Model.slug, Model.digest))
    runs = session.query(Run).all()
    keys = {}
    for run in runs:
        if run.model_slug not in digests:
            raise ValueError(f"run {run.id} points at a missing model {run.model_slug}")
        keys[run.id] = run_key(digests[run.model_slug], run.task_spec_hash, run.harness_version,
                               json.loads(run.config_json or "{}"))
    seen: dict[str, str] = {}
    for run_id, key in sorted(keys.items()):
        if key in seen:
            raise ValueError(f"runs {seen[key]} and {run_id} would share run key {key[:12]}")
        seen[key] = run_id
    changed = 0
    for run in runs:
        if run.run_key != keys[run.id]:
            run.run_key = keys[run.id]
            changed += 1
    session.commit()
    return changed


def _pending_targets(session: Session) -> dict[str, set[str]]:
    """Entities a maintainer changed on the admin page that the paper agent's database has not applied yet: a sync must
    not undo them."""
    from benchtoward.db import OwnerAction, PaperAttempt

    pending = {"papers": set(), "tasks": set(), "runs": set(), "proposals": set()}
    for action in session.query(OwnerAction).filter_by(done=False):
        if action.kind in ("requeue", "hide_paper"):
            pending["papers"].add(action.target)
        elif action.kind == "hide_task":
            pending["tasks"].add(action.target)
        elif action.kind in ("dataset_confirmed", "dataset_rejected"):
            pending["proposals"].add(action.target)
        elif action.kind == "unpublish_run":
            pending["runs"].add(action.target)
        elif action.kind == "publish_run":
            pending["runs"].add(action.target)
            for attempt in session.query(PaperAttempt).filter_by(run_id=action.target):
                pending["papers"].add(attempt.paper_doi)
    return pending


def _copy(row, cls, skip: tuple = ()) -> dict:
    return {c.name: getattr(row, c.name) for c in cls.__table__.columns if c.name not in skip}


def _rename_guard(session: Session, source: Session, registry_dir: Path | None) -> None:
    """Both databases must have applied the same committed rename plans (benchtoward.rename): a bundle from a host
    that has not would bring back old model rows and runs on old spec hashes, and one migrated by a plan this code
    lacks would bring keys this side cannot reproduce."""
    from benchtoward.rename import applied_plan_ids, load_plans, rename_lag

    if registry_dir is None:
        from benchtoward.config import registry_dir as configured

        registry_dir = configured()
    plans = load_plans(registry_dir)
    known = {p.id for p in plans}
    unknown = applied_plan_ids(source) - known
    if unknown:
        raise SyncConflict(f"bundle was migrated by rename plan {', '.join(sorted(unknown))}, which this code does "
                           f"not have: deploy it here first, then sync again")
    if not plans:
        return
    ids = ", ".join(sorted(known))
    lag = rename_lag(source, plans)
    if lag:
        more = f" and {len(lag) - 3} more" if len(lag) > 3 else ""
        raise SyncConflict(f"bundle predates rename plan {ids} ({'; '.join(lag[:3])}{more}): on that host, git pull "
                           f"and benchtoward init, then sync again")
    lag = rename_lag(session, plans)
    if lag:
        more = f" and {len(lag) - 3} more" if len(lag) > 3 else ""
        raise SyncConflict(f"this database has not applied rename plan {ids} ({'; '.join(lag[:3])}{more}): run "
                           f"benchtoward rename-rekey here first")


def import_bundle(session: Session, source_db_url: str, *, registry_dir: Path | None = None) -> dict[str, int]:
    """Merge the paper agent's state and finished runs from another database (the paper agent's) into this one.

    Models upsert by slug; tasks match by (slug, version) and get this database's ids; papers upsert by DOI and
    attempts by id; runs with status done or held match by run key, with task ids remapped. Maintainer-only fields
    (task visibility, paper hidden) are never overwritten, nor anything with a maintainer action still pending.
    Events are not copied (the site receives them live). Returns counts of rows written per kind.
    """
    from benchtoward.db import Model, Paper, PaperAttempt, make_engine, session_factory

    counts = {"models": 0, "tasks": 0, "papers": 0, "attempts": 0, "proposals": 0, "needs": 0, "runs": 0}
    pending = _pending_targets(session)
    with session_factory(make_engine(source_db_url))() as source:
        _rename_guard(session, source, registry_dir)
        for model in source.query(Model):
            values = _copy(model, Model)
            target = session.get(Model, model.slug)
            if target is None:
                session.add(Model(**values))
                counts["models"] += 1
            elif model.provider == "agent" and any(getattr(target, k) != v for k, v in values.items()):
                for k, v in values.items():  # registry models belong to this database's own load_registry
                    setattr(target, k, v)
                counts["models"] += 1
        session.flush()

        from benchtoward.db import Dataset

        for dataset in source.query(Dataset).filter(Dataset.snuh_confirmed.is_(True)):  # before tasks that use them
            if session.get(Dataset, dataset.slug) is None:
                session.add(Dataset(**_copy(dataset, Dataset)))
        session.flush()

        task_ids: dict[int, int] = {}
        for task in source.query(Task):
            target = session.query(Task).filter_by(slug=task.slug, version=task.version).one_or_none()
            values = _copy(task, Task, skip=("id", "visibility"))
            if target is None:
                target = Task(**values)
                session.add(target)
                session.flush()
                counts["tasks"] += 1
            elif task.made_by == "agent" and f"{task.slug}@{task.version}" not in pending["tasks"] and any(
                    getattr(target, k) != v for k, v in values.items()):
                for k, v in values.items():
                    setattr(target, k, v)
                counts["tasks"] += 1
            task_ids[task.id] = target.id

        for paper in source.query(Paper):
            target = session.get(Paper, paper.doi)
            values = _copy(paper, Paper, skip=("hidden",))
            if target is None:
                session.add(Paper(**values))
                counts["papers"] += 1
            elif paper.doi not in pending["papers"] and any(
                    getattr(target, k) != v for k, v in values.items() if k != "updated_at"):
                for k, v in values.items():
                    setattr(target, k, v)
                counts["papers"] += 1
        for attempt in source.query(PaperAttempt):
            values = _copy(attempt, PaperAttempt)
            target = session.get(PaperAttempt, attempt.id)
            if target is None:
                session.add(PaperAttempt(**values))
                counts["attempts"] += 1
            elif any(getattr(target, k) != v for k, v in values.items()):
                for k, v in values.items():
                    setattr(target, k, v)
                counts["attempts"] += 1
        session.flush()

        from benchtoward.db import DataNeed, DatasetProposal

        for proposal in source.query(DatasetProposal):
            target = session.get(DatasetProposal, proposal.id)
            if target is None:
                session.add(DatasetProposal(**_copy(proposal, DatasetProposal)))
                counts["proposals"] += 1
            elif proposal.id not in pending["proposals"] and target.status == "pending" and proposal.status != "pending":
                target.status, target.decided_at = proposal.status, proposal.decided_at
                counts["proposals"] += 1
        for need in source.query(DataNeed):
            if session.get(DataNeed, need.id) is None:
                session.add(DataNeed(**_copy(need, DataNeed)))
                counts["needs"] += 1
        session.flush()

        counts["runs"] = _merge_runs(session, source, task_ids, pending)[0]
        session.commit()
    return counts


def _over_api_error_threshold(source: Session, run: Run) -> bool:
    """The shared API rule on the run's stored items (budget stops and missing offline outputs are not API errors)."""
    items = source.query(ItemResult).filter_by(run_id=run.id).all()
    errors = sum(1 for r in items if r.error and r.error not in ("budget", "no_output"))
    return bool(items) and errors / len(items) >= API_ERROR_FAIL_RATE


def _merge_runs(session: Session, source: Session, task_ids: dict[int, int], pending: dict[str, set[str]],
                skip: frozenset[str] = frozenset()) -> tuple[int, int]:
    """Runs with status done or held, matched by run key, with their item results; returns (runs, items) written.
    A run already done or unpublished here, or with a maintainer action pending, is kept; a held one is replaced
    only by the bundle's done run. Source task ids are remapped through `task_ids`; runs whose id is in `skip`
    are left out. A failed source run retracts an existing done/held copy; failed runs are never newly imported.
    This lets the work host's API-error audit remove a result already synced to the site."""
    written = items = 0
    for run in source.query(Run).filter_by(status="failed"):
        if run.task_id not in task_ids or run.id in skip:
            continue
        existing = session.query(Run).filter_by(run_key=run.run_key).one_or_none()
        if existing is not None and existing.status in ("done", "held"):
            existing.status = "failed"
            internal = json.loads(existing.internal_json or "{}")
            internal["failed_reason"] = json.loads(run.internal_json or "{}").get("failed_reason", "Source run failed")
            existing.internal_json = json.dumps(internal, ensure_ascii=False)
            if run.metrics_json is not None:   # keep the site's diagnostics when the source has none
                existing.metrics_json = run.metrics_json
            written += 1
    for run in source.query(Run).filter(Run.status.in_(("done", "held"))):
        if run.task_id not in task_ids or run.id in skip:
            continue
        if _over_api_error_threshold(source, run):   # e.g. a stale bundle of a run the audit already failed
            continue
        same_id = session.get(Run, run.id)
        if same_id is not None and same_id.run_key != run.run_key:
            session.rollback()
            raise SyncConflict(f"run {run.id} has key {same_id.run_key[:12]} here and {run.run_key[:12]} in the "
                               f"bundle; both databases must apply the same rename plans")
        existing = session.query(Run).filter_by(run_key=run.run_key).one_or_none()
        if existing is not None and (existing.status in ("done", "unpublished") or existing.id in pending["runs"]
                                     or (existing.status == "held" and run.status == "held")):
            continue
        if existing is not None:
            session.query(ItemResult).filter_by(run_id=existing.id).delete()
            session.delete(existing)
            session.flush()
        values = _copy(run, Run)
        values["task_id"] = task_ids[run.task_id]
        session.add(Run(**values))
        for item in source.query(ItemResult).filter_by(run_id=run.id):
            session.add(ItemResult(**_copy(item, ItemResult)))
            items += 1
        written += 1
    return written, items


def import_runs(session: Session, source_db_url: str, *, registry_dir: Path | None = None) -> dict:
    """The `import-runs` command: finished runs only, from another database (a review's scratch DB, or the
    Mac's snapshot that `SYNC_RESULTS=1 deploy/deploy.sh` sends). Runs match by run key and follow import-bundle's
    rules (done or held; never over a done, unpublished or maintainer-changed run); their item results come along,
    and a model this database lacks is added (one it has is never changed). Tasks match by (slug, version); a run
    whose task is not here is skipped and listed, since a task is not copied; so is a run whose model is in neither
    database (SQLite does not enforce the foreign key, and its result page would have no model). Papers, attempts,
    proposals, data needs, datasets and tasks are never written, so a scratch DB cannot restore old wiki paths or
    requeue a paper (audit 5.9). Returns {"runs", "items", "models": rows written, "skipped": one line per skipped run}."""
    from benchtoward.db import Model, make_engine, session_factory

    counts: dict = {"runs": 0, "items": 0, "models": 0, "skipped": []}
    pending = _pending_targets(session)
    with session_factory(make_engine(source_db_url))() as source:
        _rename_guard(session, source, registry_dir)
        task_ids: dict[int, int] = {}
        needed = {r.task_id for r in source.query(Run).filter(Run.status.in_(("done", "held", "failed")))}
        for task in source.query(Task).filter(Task.id.in_(needed)) if needed else ():
            target = session.query(Task).filter_by(slug=task.slug, version=task.version).one_or_none()
            if target is not None:
                task_ids[task.id] = target.id
        for run in source.query(Run).filter(Run.status.in_(("done", "held"))).order_by(Run.id):
            if run.task_id not in task_ids:
                task = source.get(Task, run.task_id)
                name = f"{task.slug}@{task.version}" if task is not None else f"id {run.task_id}"
                counts["skipped"].append(f"{run.id} (task {name} is not in this database)")
        orphans: set[str] = set()
        for run in source.query(Run).filter(Run.status.in_(("done", "held"))).order_by(Run.id):
            if run.task_id not in task_ids:
                continue
            model = source.get(Model, run.model_slug)
            if session.get(Model, run.model_slug) is not None:
                continue
            if model is None:
                orphans.add(run.id)
                counts["skipped"].append(f"{run.id} (model {run.model_slug} is in neither database)")
            else:
                session.add(Model(**_copy(model, Model)))
                session.flush()
                counts["models"] += 1
        counts["runs"], counts["items"] = _merge_runs(session, source, task_ids, pending, frozenset(orphans))
        session.commit()
    return counts
