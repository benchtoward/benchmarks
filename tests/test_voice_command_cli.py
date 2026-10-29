
import shutil
from pathlib import Path

import pytest
import yaml
from conftest import REGISTRY_DIR
from test_voice_command import ROUTE, FakeLLM, make_release, oracle
from typer.testing import CliRunner

from medbench.tasks import voice_command as vc


def registry_with_routes(tmp_path, request: dict | None) -> Path:
    """A copy of the registry whose providers.yaml has one route per LLM provider, with the given `request` block."""
    reg = tmp_path / "registry"
    shutil.copytree(REGISTRY_DIR, reg)
    llm = {m["provider"] for m in yaml.safe_load((reg / "models.yaml").read_text(encoding="utf-8")) if m["type"] == "llm"}
    routes = {name: {"base_url": "https://llm.example/v1", "key_env": "EXAMPLE_KEY", **({"request": request} if request
                                                                                      else {})} for name in llm}
    (reg / "providers.yaml").write_text(yaml.safe_dump(routes), encoding="utf-8")
    return reg


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDBENCH_DB_URL", f"sqlite:///{tmp_path}/m.db")
    monkeypatch.setenv("MEDBENCH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MEDBENCH_REGISTRY_DIR", str(registry_with_routes(tmp_path, ROUTE)))
    import medbench.cli as cli

    assert CliRunner().invoke(cli.app, ["init"]).exit_code == 0
    return tmp_path, cli


def test_commands_are_registered(env):
    _, cli = env
    for args in (["fetch", "kor-surgical-robot-voice-command", "--help"],
                 ["baseline", "kor-surgical-robot-voice-command", "--help"],
                 ["eval-voice-command", "--help"], ["compare-voice-command", "--help"]):
        result = CliRunner().invoke(cli.app, args)
        assert result.exit_code == 0, (args, result.output)


def test_fetch_needs_a_token_and_reports_counts(env, monkeypatch):
    tmp_path, cli = env
    monkeypatch.delenv("KHDP_TOKEN", raising=False)
    result = CliRunner().invoke(cli.app, ["fetch", "kor-surgical-robot-voice-command"])
    assert result.exit_code == 1 and "KHDP_TOKEN" in result.output
    monkeypatch.setenv("KHDP_TOKEN", "t")
    make_release(vc.dataset_dir(tmp_path))
    monkeypatch.setattr("medbench.tasks.voice_command_cli.fetch_dataset", lambda data_dir, client: vc.dataset_dir(data_dir))
    result = CliRunner().invoke(cli.app, ["fetch", "kor-surgical-robot-voice-command"])
    assert result.exit_code == 0, result.output
    assert "clean 19" in result.output and "overheard 6" in result.output


def test_eval_baseline_and_compare_on_the_release(env, monkeypatch):
    tmp_path, cli = env
    make_release(vc.dataset_dir(tmp_path))
    inst = vc.load_instances(tmp_path)
    fake = FakeLLM({i.item["utterance"]: oracle(i.item)["parsed"] for i in inst})
    monkeypatch.setattr("medbench.tasks.voice_command_cli.client_for", lambda model, settings: fake)
    result = CliRunner().invoke(cli.app, ["eval-voice-command", "--model", "alibaba/qwen3.8-flash-next-180b-a6b"])
    assert result.exit_code == 0, result.output
    assert "[done]" in result.output and "exact_match" in result.output and len(fake.calls) == len(inst)

    result = CliRunner().invoke(cli.app, ["baseline", "kor-surgical-robot-voice-command"])
    assert result.exit_code == 0, result.output
    assert "medbench/prior" in result.output and "[done]" in result.output

    result = CliRunner().invoke(cli.app, ["compare-voice-command", "--a", "medbench/prior",
                                          "--b", "alibaba/qwen3.8-flash-next-180b-a6b"])
    assert result.exit_code == 0, result.output
    line = next(l for l in result.output.splitlines() if l.startswith("clean"))
    assert "+" in line and "95% CI" in line and "McNemar" in line
    assert any(l.startswith("overheard") for l in result.output.splitlines())


def test_eval_refuses_missing_data_and_non_llm_models(env):
    _, cli = env
    result = CliRunner().invoke(cli.app, ["eval-voice-command", "--model", "alibaba/qwen3.8-flash-next-180b-a6b"])
    assert result.exit_code != 0 and "fetch kor-surgical-robot-voice-command" in result.output
    result = CliRunner().invoke(cli.app, ["eval-voice-command", "--model", "alibaba/qwen3-asr-1.7b"])
    assert result.exit_code != 0 and "not an LLM" in result.output


def test_eval_subset_option_limits_the_run(env, monkeypatch):
    tmp_path, cli = env
    make_release(vc.dataset_dir(tmp_path))
    inst = vc.load_instances(tmp_path, ["overheard"])
    fake = FakeLLM({i.item["utterance"]: oracle(i.item)["parsed"] for i in inst})
    monkeypatch.setattr("medbench.tasks.voice_command_cli.client_for", lambda model, settings: fake)
    result = CliRunner().invoke(cli.app, ["eval-voice-command", "--model", "alibaba/qwen3.8-flash-next-180b-a6b",
                                          "--subset", "overheard"])
    assert result.exit_code == 0, result.output
    assert len(fake.calls) == 6 and "false_activation" in result.output


def test_compare_uses_the_full_run_not_a_later_subset_or_smoke_run(env, monkeypatch):
    """A later `--subset` or `--limit` run must not stand in for a model's full run (review 2026-10-29)."""
    tmp_path, cli = env
    make_release(vc.dataset_dir(tmp_path))
    inst = vc.load_instances(tmp_path)
    good = FakeLLM({i.item["utterance"]: oracle(i.item)["parsed"] for i in inst})
    act = {"response": "execute", "actions": [{"action": "HOLD", "target": None, "instrument": None,
                                               "direction": None, "magnitude": None}], "message": None}
    bad = FakeLLM({i.item["utterance"]: act for i in inst})
    slug = "alibaba/qwen3.8-flash-next-180b-a6b"
    monkeypatch.setattr("medbench.tasks.voice_command_cli.client_for", lambda model, settings: good)
    assert CliRunner().invoke(cli.app, ["eval-voice-command", "--model", slug]).exit_code == 0
    monkeypatch.setattr("medbench.tasks.voice_command_cli.client_for", lambda model, settings: bad)
    for extra in (["--subset", "overheard"], ["--limit", "3"]):
        assert CliRunner().invoke(cli.app, ["eval-voice-command", "--model", slug, *extra]).exit_code == 0
    assert CliRunner().invoke(cli.app, ["baseline", "kor-surgical-robot-voice-command"]).exit_code == 0
    result = CliRunner().invoke(cli.app, ["compare-voice-command", "--a", "medbench/prior", "--b", slug])
    assert result.exit_code == 0, result.output
    line = next(l for l in result.output.splitlines() if l.startswith("overheard"))
    assert "100.0 -> 100.0" in line  # the full run: the oracle never acts on overheard talk


def test_compare_reports_false_activation_on_overheard_talk_not_only_exact(env, monkeypatch):
    """On overheard talk `exact` is the no-action rate (reject or clarify); false activation is an executed action
    (`unsafe_exec`). An `execute` reply with no actions fails exact but is no false activation (review 2026-10-29)."""
    tmp_path, cli = env
    make_release(vc.dataset_dir(tmp_path))
    inst = vc.load_instances(tmp_path)
    good = FakeLLM({i.item["utterance"]: oracle(i.item)["parsed"] for i in inst})
    heard = [i.item["utterance"] for i in inst if i.subset == "overheard"]
    act = {"response": "execute", "actions": [{"action": "HOLD", "target": None, "instrument": None,
                                               "direction": None, "magnitude": None}], "message": None}
    empty = {"response": "execute", "actions": [], "message": None}
    replies = {i.item["utterance"]: oracle(i.item)["parsed"] for i in inst}
    replies.update({u: (act if k % 2 == 0 else empty) for k, u in enumerate(heard)})
    mixed = FakeLLM(replies)
    for slug, fake in (("alibaba/qwen3.8-flash-next-180b-a6b", good), ("alibaba/qwen3.8-max-2.4t-a95b", mixed)):
        monkeypatch.setattr("medbench.tasks.voice_command_cli.client_for", lambda model, settings, f=fake: f)
        assert CliRunner().invoke(cli.app, ["eval-voice-command", "--model", slug]).exit_code == 0
    result = CliRunner().invoke(cli.app, ["compare-voice-command", "--a", "alibaba/qwen3.8-flash-next-180b-a6b",
                                          "--b", "alibaba/qwen3.8-max-2.4t-a95b"])
    assert result.exit_code == 0, result.output
    lines = [l for l in result.output.splitlines() if l.startswith("overheard")]
    no_action = next(l for l in lines if "no-action rate" in l)
    false_act = next(l for l in lines if "false activation" in l)
    assert "100.0 -> 0.0" in no_action
    assert "0.0 -> 50.0" in false_act and "+50.0 pp" in false_act


def test_eval_refuses_a_route_without_a_thinking_switch(env, monkeypatch):
    tmp_path, cli = env
    make_release(vc.dataset_dir(tmp_path))
    monkeypatch.setenv("MEDBENCH_REGISTRY_DIR", str(registry_with_routes(tmp_path / "bare", None)))
    fake = FakeLLM({})
    monkeypatch.setattr("medbench.tasks.voice_command_cli.client_for", lambda model, settings: fake)
    result = CliRunner().invoke(cli.app, ["eval-voice-command", "--model", "alibaba/qwen3.8-flash-next-180b-a6b"])
    assert result.exit_code != 0 and "thinking_off" in result.output and fake.calls == []
