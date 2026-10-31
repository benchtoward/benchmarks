import asyncio
import json
import uuid
from collections import defaultdict
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


def max_tokens_for(model: Model) -> int:
    return model.max_tokens or DEFAULT_MAX_TOKENS


def _retry(result: ItemResult) -> bool:
    """An item needs another call if its error was transient (or predates the transient flag)."""
    return result.error is not None and result.transient is not False


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
    config = {"prompt": PROMPT_ID, "max_tokens": max_tokens, "temperature": 0}
    if limit is not None:
        config["limit"] = limit
        items = items[:limit]
    key = run_key(model.digest, task.spec_hash, HARNESS_VERSION, config)

    run = _find_run(session, key)
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
        prediction = None if result.error else parse_answer(result.content, set(item.options))
        session.merge(ItemResult(run_id=run.id, instance_id=item.instance_id, prediction=prediction,
                                 label=item.answer, correct=prediction == item.answer, error=result.error,
                                 transient=result.transient if result.error else None,
                                 raw=(result.content or "")[:20000] or None,
                                 tokens_in=result.tokens_in, tokens_out=result.tokens_out))
        session.commit()

    await asyncio.gather(*(one(item) for item in todo))
    return _finish_llm_run(session, model, run, items, data_dir)


def _finish_llm_run(session: Session, model: Model, run: Run, items: list[QAItem], data_dir: Path) -> Run:
    results = {r.instance_id: r for r in session.query(ItemResult).filter_by(run_id=run.id)}
    run.n_items = len(items)
    run.tokens_in = sum(r.tokens_in for r in results.values())
    run.tokens_out = sum(r.tokens_out for r in results.values())
    run.cost_usd = cost_usd(model, run.tokens_in, run.tokens_out)
    run.eval_date = date.today()
    if any(_retry(r) for r in results.values()) or len(results) < len(items):
        run.status = "failed"
        run.metrics_json = None
        session.commit()
        return run

    ordered = [results[item.instance_id] for item in items]
    correct = [None if r.prediction is None else bool(r.correct) for r in ordered]
    by_category: dict[str, list[bool]] = defaultdict(list)
    for item, r in zip(items, ordered):
        by_category[item.category].append(bool(r.correct))
    metrics = {
        "accuracy": accuracy_ci(correct),
        "parse_rate": sum(c is not None for c in correct) / len(correct),
        "error_rate": sum(r.error is not None for r in ordered) / len(ordered),
        "parser": PARSER_VERSION,
        "by_category": {cat: {"value": sum(v) / len(v), "n": len(v)} for cat, v in sorted(by_category.items())},
    }
    run.metrics_json = json.dumps(metrics)
    run.status = "done"
    run.finished_at = utcnow()
    session.commit()
    write_predictions(run, [{"instance_id": r.instance_id, "prediction": r.prediction, "label": r.label,
                             "correct": r.correct} for r in ordered], data_dir)
    return run


ASR_HARNESS_VERSION = "asr-p0.1"


async def run_asr_task(session: Session, model: Model, task: Task, clips: list, client, read_audio, *,
                       concurrency: int = 4, force: bool = False, limit: int | None = None,
                       data_dir: Path | None = None, language: str = "ko", progress=None,
                       max_consecutive_failures: int = 25) -> Run:
    """Transcribe every clip once (resumable, like run_llm_task) and score CER/WER on the control plane.

    `client.transcribe(model_id, audio, filename, language=, temperature=)` returns a Transcription; `read_audio(clip)`
    returns the WAV bytes. Transient failures leave the run `failed` and are retried on the next call; final per-clip
    failures count as empty transcripts; an account-level failure (token, credits, model) stops sending further clips,
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

    results = {r.instance_id: r for r in session.query(ItemResult).filter_by(run_id=run.id)}
    seconds = [_audio_seconds(r.raw) for r in results.values() if r.error is None]
    run.n_items = len(clips)
    run.cost_usd = audio_cost_usd(model, seconds)
    run.eval_date = date.today()
    if any(_retry(r) for r in results.values()) or len(results) < len(clips):
        run.status = "failed"
        run.metrics_json = None
        session.commit()
        return run

    ordered = [results[clip.instance_id] for clip in clips]
    hypotheses, errors = [r.prediction for r in ordered], [r.error for r in ordered]
    rows = item_rows(clips, hypotheses, errors)
    for r, row in zip(ordered, rows):  # rescore from stored transcripts so resumed items use the current normaliser
        r.score, r.correct = row["cer"], row["char_errors"] == 0
    metrics = asr_metrics(clips, hypotheses, errors)
    metrics["audio_seconds"] = sum(seconds)
    metrics["billed_minutes"] = sum(billed_minutes(s) for s in seconds)
    run.metrics_json = json.dumps(metrics, ensure_ascii=False)
    run.status = "done"
    run.finished_at = utcnow()
    session.commit()
    write_predictions(run, rows, data_dir)
    return run


def record_baseline_run(session: Session, model: Model, task: Task, rows: list[dict], metrics: dict, config: dict,
                        *, data_dir: Path | None = None) -> Run:
    """Store a finished run computed outside the LLM runner (medbench baselines). Deduplicated by run key."""
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


def rekey_runs(session: Session) -> int:
    """Recompute run keys and task spec hashes after the spec-hash definition changes. Returns runs changed."""
    changed = 0
    for run in session.query(Run):
        model, task = session.get(Model, run.model_slug), session.get(Task, run.task_id)
        key = run_key(model.digest, task.spec_hash, run.harness_version, json.loads(run.config_json or "{}"))
        if run.run_key != key or run.task_spec_hash != task.spec_hash:
            run.run_key, run.task_spec_hash = key, task.spec_hash
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


def import_bundle(session: Session, source_db_url: str) -> dict[str, int]:
    """Merge the paper agent's state and finished runs from another medbench database (the paper agent's) into this one.

    Models upsert by slug; tasks match by (slug, version) and get this database's ids; papers upsert by DOI and
    attempts by id; runs with status done or held match by run key, with task ids remapped. Maintainer-only fields
    (task visibility, paper hidden) are never overwritten, nor anything with a maintainer action still pending.
    Events are not copied (the site receives them live). Returns counts of rows written per kind.
    """
    from benchtoward.db import Model, Paper, PaperAttempt, make_engine, session_factory

    counts = {"models": 0, "tasks": 0, "papers": 0, "attempts": 0, "proposals": 0, "needs": 0, "runs": 0}
    pending = _pending_targets(session)
    with session_factory(make_engine(source_db_url))() as source:
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

        for run in source.query(Run).filter(Run.status.in_(("done", "held"))):
            existing = session.query(Run).filter_by(run_key=run.run_key).one_or_none()
            if existing is not None and (existing.status in ("done", "unpublished") or existing.id in pending["runs"]
                                         or (existing.status == "held" and run.status == "held")):
                continue
            if existing is not None:
                session.query(ItemResult).filter_by(run_id=existing.id).delete()
                session.delete(existing)
                session.flush()
            values = _copy(run, Run)
            values["task_id"] = task_ids.get(run.task_id, run.task_id)
            session.add(Run(**values))
            for item in source.query(ItemResult).filter_by(run_id=run.id):
                session.add(ItemResult(**_copy(item, ItemResult)))
            counts["runs"] += 1
        session.commit()
    return counts


def import_runs(session: Session, source_db_url: str) -> int:
    """Backward-compatible name used by `benchtoward import-runs`: same merge, returns the number of runs."""
    return import_bundle(session, source_db_url)["runs"]
