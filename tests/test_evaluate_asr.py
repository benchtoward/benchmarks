import asyncio
import json

import pandas as pd
import pytest

from conftest import REGISTRY_DIR
from benchtoward.asr import Transcription
from benchtoward.db import ItemResult, Model, Run, Task
from benchtoward.evaluate import ASR_HARNESS_VERSION, run_asr_task
from benchtoward.registry import load_registry
from benchtoward.tasks.k_or_speech import ROOT_IN_ZIP, Clip

REFS = {"U1": "석션 주세요.", "U2": "BP 90 대로 떨어집니다.", "U3": "로큐 50 미리 들어갑니다."}
CLIPS = [Clip(instance_id=f"{u}-{v}", utterance_id=u, voice=v, reference=ref, intent="status_update",
              cs_style="none", speaker="surgeon", member=f"{ROOT_IN_ZIP}/qwen3m_{v}/{u}__qwen3m_{v}.wav")
         for u, ref in REFS.items() for v in ("chul", "maia")]


def ok(text, seconds=3):
    return Transcription(text=text, audio_seconds=seconds, raw=json.dumps({"text": text, "usage": {"seconds": seconds}}),
                         error=None)


def fail(error, transient=True, fatal=False):
    return Transcription(text="", audio_seconds=0, raw=None, error=error, transient=transient, fatal=fatal)


class FakeASR:
    """Answers by file name (<utterance>__qwen3m_<voice>.wav); counts calls."""

    def __init__(self, answers: dict[str, Transcription] | None = None):
        self.answers = answers or {}
        self.calls: list[str] = []

    async def transcribe(self, model_id, audio, filename, *, language="ko", temperature=0.0):
        assert language == "ko" and temperature == 0 and audio == f"audio:{filename}".encode()
        self.calls.append(filename)
        key = filename.split("__")[0] + "-" + filename.rsplit("_", 1)[1].removesuffix(".wav")
        return self.answers.get(key) or ok(REFS[key.split("-")[0]])


def read_audio(clip):
    return f"audio:{clip.filename}".encode()


@pytest.fixture
def ctx(tmp_session, tmp_path):
    load_registry(tmp_session, REGISTRY_DIR)
    model = tmp_session.get(Model, "alibaba/qwen3-asr-1.7b")
    model.price_usd_per_audio_min = 1 / 1500  # prices are deployment data; fixed here
    task = tmp_session.query(Task).filter_by(slug="k-or-speech-asr").one()
    return tmp_session, model, task, tmp_path


def run(ctx, client, **kw):
    session, model, task, data_dir = ctx
    kw.setdefault("concurrency", 2)
    return asyncio.run(run_asr_task(session, model, task, CLIPS, client, read_audio, data_dir=data_dir, **kw))


def test_metrics_and_items_stored(ctx):
    answers = {"U1-maia": ok("석션 줘요"), "U2-chul": ok("비피 구십 대로 떨어집니다")}
    r = run(ctx, FakeASR(answers))
    assert r.status == "done" and r.tier == "baseline" and r.harness_version == ASR_HARNESS_VERSION
    assert json.loads(r.config_json) == {"language": "ko", "response_format": "json", "temperature": 0}
    metrics = json.loads(r.metrics_json)
    # chars: 석션주세요 5, bp구십대로떨어집니다 11, 로큐오십미리들어갑니다 11; errors: 2 (U1-maia), 2 (U2-chul: bp)
    assert metrics["cer"]["value"] == pytest.approx(4 / 54) and metrics["cer"]["n_groups"] == 3
    assert metrics["parse_rate"] == 1.0 and metrics["error_rate"] == 0.0
    assert metrics["by_voice"]["chul"]["value"] == pytest.approx(2 / 27)
    item = ctx[0].get(ItemResult, (r.id, "U1-maia"))
    assert item.prediction == "석션 줘요" and item.label == "석션 주세요." and item.correct is False
    assert item.score == pytest.approx(2 / 5) and '"text"' in item.raw
    assert ctx[0].get(ItemResult, (r.id, "U3-chul")).score == 0 and ctx[0].get(ItemResult, (r.id, "U3-chul")).correct
    assert r.n_items == 6 and r.cost_usd == pytest.approx(6 * ctx[1].price_usd_per_audio_min)
    assert metrics["audio_seconds"] == 18 and metrics["billed_minutes"] == 6


def test_second_run_is_deduplicated(ctx):
    first = run(ctx, FakeASR())
    client = FakeASR()
    second = run(ctx, client)
    assert second.id == first.id and client.calls == [] and ctx[0].query(Run).count() == 1


def test_resume_only_transcribes_missing_clips(ctx):
    session = ctx[0]
    first = run(ctx, FakeASR())
    first.status = "running"
    session.query(ItemResult).filter(ItemResult.run_id == first.id,
                                     ItemResult.instance_id.in_(["U2-chul", "U3-maia"])).delete()
    session.commit()
    client = FakeASR()
    resumed = run(ctx, client)
    assert sorted(client.calls) == ["U2__qwen3m_chul.wav", "U3__qwen3m_maia.wav"]
    assert resumed.id == first.id and resumed.status == "done"
    assert json.loads(resumed.metrics_json)["cer"]["value"] == 0


def test_transient_errors_leave_run_failed_and_resumable(ctx):
    r = run(ctx, FakeASR({"U2-maia": fail("http_503")}))
    assert r.status == "failed" and r.metrics_json is None
    assert ctx[0].get(ItemResult, (r.id, "U2-maia")).error == "http_503"
    client = FakeASR()
    again = run(ctx, client)
    assert client.calls == ["U2__qwen3m_maia.wav"] and again.status == "done" and again.id == r.id


def test_final_error_counts_as_an_empty_transcript(ctx):
    r = run(ctx, FakeASR({"U1-chul": fail("http_400", transient=False)}))
    assert r.status == "done"
    metrics = json.loads(r.metrics_json)
    assert metrics["cer"]["value"] == pytest.approx(5 / 54) and metrics["cer"]["n_missing"] == 1
    assert metrics["error_rate"] == pytest.approx(1 / 6) and metrics["errors"] == {"http_400": 1}
    item = ctx[0].get(ItemResult, (r.id, "U1-chul"))
    assert item.prediction is None and item.score == 1.0 and item.correct is False


def test_account_error_stops_sending_the_rest(ctx):
    client = FakeASR({"U1-chul": fail("insufficient_credits", fatal=True)})
    r = run(ctx, client, concurrency=1)
    assert client.calls == ["U1__qwen3m_chul.wav"] and r.status == "failed"
    assert ctx[0].query(ItemResult).filter_by(run_id=r.id).count() == 1
    again = run(ctx, FakeASR(), concurrency=1)
    assert again.id == r.id and again.status == "done"


def test_endpoint_outage_trips_the_breaker(ctx):
    """An endpoint that fails every clip (e.g. its serving pod was preempted) is not hammered for hours."""
    client = FakeASR({c.instance_id: fail("http_500") for c in CLIPS})
    r = run(ctx, client, concurrency=1, max_consecutive_failures=3)
    assert len(client.calls) == 3 and r.status == "failed"
    again = run(ctx, FakeASR(), concurrency=1, max_consecutive_failures=3)
    assert again.id == r.id and again.status == "done"


def test_scattered_failures_do_not_trip_the_breaker(ctx):
    client = FakeASR({"U1-chul": fail("http_500"), "U1-maia": fail("http_500"), "U2-maia": fail("http_500"),
                      "U3-chul": fail("http_500")})
    run(ctx, client, concurrency=1, max_consecutive_failures=3)
    assert len(client.calls) == 6


class RaisingASR(FakeASR):
    async def transcribe(self, model_id, audio, filename, **kw):
        if filename.startswith("U3"):
            self.calls.append(filename)
            raise RuntimeError("boom")
        return await super().transcribe(model_id, audio, filename, **kw)


def test_client_exception_is_recorded_not_raised(ctx):
    r = run(ctx, RaisingASR())
    assert r.status == "failed" and ctx[0].get(ItemResult, (r.id, "U3-chul")).error == "exception_RuntimeError"
    assert run(ctx, FakeASR()).status == "done"


def test_predictions_parquet_written(ctx):
    r = run(ctx, FakeASR({"U1-maia": ok("석션 줘요")}))
    df = pd.read_parquet(ctx[3] / "artifacts" / r.id / "predictions.parquet")
    assert len(df) == 6 and {"instance_id", "utterance_id", "voice", "intent", "cs_style", "speaker", "reference",
                             "prediction", "cer", "char_errors", "ref_chars", "word_errors", "ref_words",
                             "error"} <= set(df.columns)
    row = df.set_index("instance_id").loc["U1-maia"]
    assert row["prediction"] == "석션 줘요" and row["char_errors"] == 2 and row["ref_chars"] == 5


def test_limit_changes_run_key(ctx):
    full = run(ctx, FakeASR())
    client = FakeASR()
    partial = run(ctx, client, limit=2)
    assert partial.id != full.id and len(client.calls) == 2 and partial.n_items == 2


def test_runner_with_real_client_over_mock_transport(ctx):
    """503s are retried inside the call, a 400 is final for its clip, and an outage leaves the run resumable."""
    import httpx

    from benchtoward.asr import ASRClient

    calls: dict[str, int] = {}
    outage = {"on": True}

    def handler(request: httpx.Request) -> httpx.Response:
        name = request.content.split(b'filename="', 1)[1].split(b'"', 1)[0].decode()
        calls[name] = calls.get(name, 0) + 1
        utt = name.split("__")[0]
        if name == "U2__qwen3m_maia.wav" and outage["on"]:
            return httpx.Response(503, text="pod pending")
        if name == "U1__qwen3m_chul.wav" and calls[name] == 1:
            return httpx.Response(502)
        if name == "U3__qwen3m_chul.wav":
            return httpx.Response(400, json={"detail": "could not decode audio"})
        return httpx.Response(200, json={"text": REFS[utt], "usage": {"type": "duration", "seconds": 3}})

    async def no_sleep(_seconds):
        return None

    client = ASRClient("https://asr.test/v1", "tok", transport=httpx.MockTransport(handler), sleep=no_sleep)
    first = run(ctx, client)
    assert first.status == "failed" and calls["U1__qwen3m_chul.wav"] == 2 and calls["U2__qwen3m_maia.wav"] == 5
    outage["on"] = False
    calls.clear()
    second = run(ctx, client)
    assert second.id == first.id and second.status == "done" and list(calls) == ["U2__qwen3m_maia.wav"]
    metrics = json.loads(second.metrics_json)
    assert metrics["errors"] == {"http_400": 1} and metrics["cer"]["value"] == pytest.approx(11 / 54)
    assert ctx[0].get(ItemResult, (second.id, "U3-chul")).raw == '{"detail":"could not decode audio"}'


def test_rescoring_on_finish_uses_current_normaliser(ctx):
    """Scores are recomputed from stored transcripts when a run finishes, so resumed items agree with fresh ones."""
    session = ctx[0]
    r = run(ctx, FakeASR())
    item = session.get(ItemResult, (r.id, "U1-chul"))
    item.score, r.status = 0.9, "running"
    session.commit()
    again = run(ctx, FakeASR())
    assert session.get(ItemResult, (again.id, "U1-chul")).score == 0
