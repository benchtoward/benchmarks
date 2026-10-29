import pytest
from typer.testing import CliRunner

from conftest import REGISTRY_DIR
from test_evaluate_asr import FakeASR
from test_k_or_speech import make_release


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDBENCH_DB_URL", f"sqlite:///{tmp_path}/m.db")
    monkeypatch.setenv("MEDBENCH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MEDBENCH_REGISTRY_DIR", str(REGISTRY_DIR))
    import medbench.cli as cli

    assert CliRunner().invoke(cli.app, ["init"]).exit_code == 0
    return tmp_path, cli


class ZipFakeASR(FakeASR):
    async def transcribe(self, model_id, audio, filename, *, language="ko", temperature=0.0):
        assert audio.startswith(b"RIFF-")
        self.calls.append(filename)
        from medbench.asr import Transcription

        return Transcription(text="석션 주세요", audio_seconds=3, raw='{"text": "석션 주세요"}', error=None)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None


def test_eval_asr_runs_from_the_release_archive(env, monkeypatch):
    tmp_path, cli = env
    make_release(tmp_path / "datasets" / "k-or-speech" / "1.0.0" / "synthetic_kor_speech_audio_v1.zip")
    fake = ZipFakeASR()
    monkeypatch.setattr("medbench.tasks.speech_cli.asr_client_for", lambda model, settings: fake)
    result = CliRunner().invoke(cli.app, ["eval-asr", "--model", "alibaba/qwen3-asr-1.7b", "--limit", "4"])
    assert result.exit_code == 0, result.output
    assert len(fake.calls) == 4 and "[done]" in result.output and "cer" in result.output


class DownASR(ZipFakeASR):
    async def transcribe(self, model_id, audio, filename, **kw):
        from medbench.asr import Transcription

        self.calls.append(filename)
        return Transcription(text="", audio_seconds=0, raw="Internal Server Error", error="http_500", transient=True)


def test_eval_asr_says_how_to_resume_after_an_outage(env, monkeypatch):
    tmp_path, cli = env
    make_release(tmp_path / "datasets" / "k-or-speech" / "1.0.0" / "synthetic_kor_speech_audio_v1.zip")
    monkeypatch.setattr("medbench.tasks.speech_cli.asr_client_for", lambda model, settings: DownASR())
    result = CliRunner().invoke(cli.app, ["eval-asr", "--model", "cohere/cohere-transcribe-2b", "--limit", "6"])
    assert result.exit_code == 0, result.output
    assert "[failed]" in result.output and "6 clip(s) still need a transcript" in result.output
    assert "http_500" in result.output and "run the same command again" in result.output


def test_eval_asr_refuses_non_asr_models_and_missing_data(env):
    tmp_path, cli = env
    result = CliRunner().invoke(cli.app, ["eval-asr", "--model", "google/medgemma-27b"])
    assert result.exit_code != 0 and "not an ASR model" in result.output
    result = CliRunner().invoke(cli.app, ["eval-asr", "--model", "alibaba/qwen3-asr-1.7b"])
    assert result.exit_code != 0 and "fetch k-or-speech" in result.output


def test_commands_are_registered(env):
    _, cli = env
    for args in (["fetch", "k-or-speech", "--help"], ["eval-asr", "--help"], ["eval-asr-all", "--help"]):
        assert CliRunner().invoke(cli.app, args).exit_code == 0, args


def test_eval_all_still_only_runs_llms(env, monkeypatch):
    _, cli = env
    seen = []
    monkeypatch.setattr(cli, "_eval_one", lambda session, model, *a: seen.append(model.type))
    assert CliRunner().invoke(cli.app, ["eval-all", "--task", "clinicalqa@0"]).exit_code == 0
    assert seen and set(seen) == {"llm"}
