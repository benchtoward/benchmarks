"""medbench commands for the K-OR speech task: fetch k-or-speech, eval-asr, eval-asr-all."""
import asyncio

import typer

from benchtoward.asr import asr_client_for
from benchtoward.config import get_settings
from benchtoward.db import Model
from benchtoward.llm import MissingCredential

DEFAULT_TASK = "k-or-speech-asr@0"


def fetch_k_or_speech() -> None:
    """Download Synthetic K-OR Speech Audio 1.0.0 from KHDP (resumable) and verify the archive checksum."""
    from benchtoward.khdp import KhdpClient
    from benchtoward.tasks.k_or_speech import fetch_audio, load_clips

    settings = get_settings()
    if not settings.khdp_token:
        typer.echo("KHDP_TOKEN is not set (KHDP account settings -> API Token)")
        raise typer.Exit(1)
    path = fetch_audio(settings.data_dir, KhdpClient(settings.khdp_token))
    clips = load_clips(settings.data_dir)
    typer.echo(f"{path}: {len(clips)} clips, {len({c.utterance_id for c in clips})} utterances")


def _eval_asr_one(session, model: Model, task_ref: str, force: bool, limit: int | None, concurrency: int) -> None:
    from benchtoward.cli import _summary
    from benchtoward.evaluate import run_asr_task
    from benchtoward.registry import resolve_task
    from benchtoward.tasks.k_or_speech import AudioReader, archive_path, load_clips

    settings = get_settings()
    if model.type != "asr":
        raise typer.BadParameter(f"{model.slug} is not an ASR model")
    if not archive_path(settings.data_dir).exists():
        raise typer.BadParameter(f"no audio under {archive_path(settings.data_dir)}; run `benchtoward fetch k-or-speech`")
    task = resolve_task(session, task_ref)
    clips = load_clips(settings.data_dir)
    client = asr_client_for(model, settings)

    def progress(done: int, total: int) -> None:
        if done % 500 == 0 or done == total:
            typer.echo(f"  {model.slug}: {done}/{total}")

    async def main():
        async with client:
            with AudioReader(settings.data_dir) as read_audio:
                return await run_asr_task(session, model, task, clips, client, read_audio, force=force, limit=limit,
                                          concurrency=concurrency, data_dir=settings.data_dir, progress=progress)

    run = asyncio.run(main())
    typer.echo(_summary(run))
    if run.status == "failed":
        from collections import Counter

        from benchtoward.db import ItemResult

        stored = session.query(ItemResult).filter_by(run_id=run.id).all()
        retry = Counter(r.error for r in stored if r.error is not None and r.transient is not False)
        typer.echo(f"  {len(clips if limit is None else clips[:limit]) - len(stored) + sum(retry.values())} clip(s) "
                   f"still need a transcript (errors: {dict(retry)}); run the same command again to resume")


def eval_asr(model: str = typer.Option(..., help="ASR model slug, e.g. alibaba/qwen3-asr-1.7b"),
             task: str = typer.Option(DEFAULT_TASK, help="speech task, slug@version"),
             force: bool = False, limit: int | None = None,
             concurrency: int = typer.Option(4, help="requests in flight")) -> None:
    """Transcribe the K-OR speech clips with one ASR model and score CER/WER (run once; resumes unfinished runs)."""
    from benchtoward.cli import open_session

    with open_session() as session:
        row = session.get(Model, model)
        if row is None:
            raise typer.BadParameter(f"unknown model {model}")
        _eval_asr_one(session, row, task, force, limit, concurrency)


def eval_asr_all(task: str = typer.Option(DEFAULT_TASK, help="speech task, slug@version"),
                 concurrency: int = typer.Option(4, help="requests in flight per model")) -> None:
    """Evaluate every enabled ASR model whose credentials are available, one after another."""
    from benchtoward.cli import open_session

    with open_session() as session:
        for model in session.query(Model).filter_by(type="asr", enabled=True).order_by(Model.slug):
            try:
                _eval_asr_one(session, model, task, False, None, concurrency)
            except MissingCredential as exc:
                typer.echo(f"skip {model.slug}: {exc}")
            except Exception as exc:  # one model's failure must not stop the others
                session.rollback()
                typer.echo(f"error {model.slug}: {type(exc).__name__}: {exc}")


def register(app: typer.Typer, fetch_app: typer.Typer) -> None:
    fetch_app.command("k-or-speech")(fetch_k_or_speech)
    app.command("eval-asr")(eval_asr)
    app.command("eval-asr-all")(eval_asr_all)
