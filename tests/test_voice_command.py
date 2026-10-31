"""kor-surgical-robot-voice-command@0: the port must reproduce the dataset author's evaluation code.

Fixtures in tests/fixtures/voice_command are real items of the KHDP release (CC BY 4.0) with the original code's
outputs on them (see the README there)."""
import asyncio
import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from benchtoward.llm import ChatResult
from benchtoward.tasks import voice_command as vc

FIX = Path(__file__).resolve().parent / "fixtures" / "voice_command"
EXPECTED = json.loads((FIX / "expected.json").read_text(encoding="utf-8"))
RAW = json.loads((FIX / "raw.json").read_text(encoding="utf-8"))
SCENES = json.loads((FIX / "scenes_v0.1.json").read_text(encoding="utf-8"))
RULES = (FIX / "rules_v0.1.txt").read_text(encoding="utf-8").rstrip("\n")


def items(subset: str) -> list[dict]:
    return [json.loads(line) for line in (FIX / f"{subset}.jsonl").read_text(encoding="utf-8").splitlines()]


def scored(key: str) -> list[dict]:
    subset, _ = key.split("/")
    raw = RAW[key]
    recs = {i: {"parsed": vc.parse_json(r["content"]), "error": r["error"]} for i, r in raw.items()}
    return [vc.score_item(it, recs.get(it["id"])) for it in items(subset)]


# --- prompt ----------------------------------------------------------------------------------------------------------

def test_system_prompt_matches_the_original_for_every_scene():
    assert set(EXPECTED["system_prompt"]) == set(SCENES)
    for scene_id, expected in EXPECTED["system_prompt"].items():
        assert vc.system_prompt(SCENES[scene_id], RULES) == expected, scene_id


def test_messages_match_the_original_including_robot_state():
    by_id = {it["id"]: it for it in items("clean")}
    assert any(by_id[i].get("context") for i in EXPECTED["messages"])
    for item_id, expected in EXPECTED["messages"].items():
        assert vc.messages(by_id[item_id], SCENES, RULES) == expected, item_id


def test_json_schema_matches_the_original():
    assert vc.json_schema() == EXPECTED["json_schema"]


def test_parse_json_matches_the_original():
    for text, expected in EXPECTED["parse_json"]:
        assert vc.parse_json(text) == expected, text


# --- scoring ---------------------------------------------------------------------------------------------------------

def test_score_item_reproduces_the_original_on_real_outputs():
    for key, expected in EXPECTED["score_item"].items():
        assert scored(key) == expected, key


def test_summarize_reproduces_the_original():
    for key, expected in EXPECTED["summarize"].items():
        got = vc.summarize(scored(key))
        assert {k: got[k] for k in expected} == expected, key


def test_paired_comparison_reproduces_the_original():
    for pair, expected in EXPECTED["compare"].items():
        a, b = pair.split("|")
        assert vc.compare(scored(a), scored(b)) == expected, pair
    assert vc.compare(scored("clean/mid"), scored("clean/weak"), n_boot=200, seed=3) == EXPECTED["compare_small_boot"]


def oracle(item):
    """Prediction built from the first acceptable value of every gold slot (as in the original tests)."""
    g = item["gold"]
    acts = [{"action": a["action"][0], "target": a["target"][0], "direction": a["direction"][0],
             "magnitude": a["magnitude"][0], "instrument": None if a["instrument"] == vc.ANY else a["instrument"][0]}
            for a in g["actions"]]
    return {"parsed": {"response": g["response"], "actions": acts, "message": None}}


def test_oracle_scores_everything():
    for subset in ("clean", "asr_generic", "overheard"):
        assert [it["id"] for it in items(subset) if not vc.score_item(it, oracle(it))["exact"]] == []


def test_wrong_target_keeps_the_sequence_but_fails_exact():
    it = next(i for i in items("clean") if i["gold"]["response"] == "execute" and i["gold"]["actions"][0]["target"] != [None])
    rec = oracle(it)
    rec["parsed"]["actions"][0]["target"] = "not in the scene"
    r = vc.score_item(it, rec)
    assert r["seq_ok"] and not r["exact"]


def test_executing_a_reject_is_unsafe_and_an_overheard_execution_is_a_false_activation():
    act = {"action": "RETRACT", "target": None, "direction": None, "magnitude": None, "instrument": None}
    rec = {"parsed": {"response": "execute", "actions": [act], "message": None}}
    rej = next(i for i in items("clean") if i["gold"]["response"] == "reject")
    assert vc.score_item(rej, rec)["unsafe_exec"] and not vc.score_item(rej, rec)["exact"]
    heard = items("overheard")[0]
    assert vc.score_item(heard, rec)["unsafe_exec"]
    clarify = {"parsed": {"response": "clarify", "actions": [], "message": "?"}}
    assert vc.score_item(heard, clarify)["exact"]  # overheard talk accepts reject or clarify


def test_alternative_sequence_accepted():
    it = next(i for i in items("clean") if i["gold"]["alts"])
    rec = copy.deepcopy(oracle(it))
    rec["parsed"]["actions"] = [{"action": a["action"][0], "target": a["target"][0], "direction": a["direction"][0],
                                 "magnitude": a["magnitude"][0],
                                 "instrument": None if a["instrument"] == vc.ANY else a["instrument"][0]}
                                for a in it["gold"]["alts"][0]]
    assert vc.score_item(it, rec)["exact"]


def test_missing_prediction_is_an_error_and_wrong():
    r = vc.score_item(items("clean")[0], None)
    assert r["error"] and not r["exact"] and not r["valid"]


# --- data ------------------------------------------------------------------------------------------------------------

def make_release(root: Path) -> Path:
    """A miniature release laid out like the KHDP files (fixture items; the OR-tuned ASR file reuses the generic one)."""
    files = {"benchmark/commands_v0.1.jsonl": FIX / "clean.jsonl",
             "benchmark/commands_v0.1_asr_generic.jsonl": FIX / "asr_generic.jsonl",
             "benchmark/commands_v0.1_asr_kor_or_stt.jsonl": FIX / "asr_generic.jsonl",
             "benchmark/kor_overheard_v0.1.jsonl": FIX / "overheard.jsonl",
             "benchmark/scenes_v0.1.json": FIX / "scenes_v0.1.json",
             "benchmark/rules_v0.1.txt": FIX / "rules_v0.1.txt"}
    for rel, src in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, root / rel)
    train = [{"id": f"t{i}", "gold": {"response": "reject", "actions": [], "alts": []}} for i in range(3)]
    train.append({"id": "t9", "gold": {"response": "execute", "alts": [], "actions": [
        {"action": ["SUCTION"], "target": [None], "direction": [None], "magnitude": [None, "medium"], "instrument": "*"}]}})
    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "train/train_v0.2.jsonl").write_text("".join(json.dumps(t) + "\n" for t in train), encoding="utf-8")
    return root


@pytest.fixture
def data_dir(tmp_path) -> Path:
    make_release(vc.dataset_dir(tmp_path))
    return tmp_path


def test_dataset_dir_follows_the_khdp_code_and_version(tmp_path):
    assert vc.dataset_dir(tmp_path) == tmp_path / "datasets" / "kor-surgical-robot-voice-command" / "1.0.0"


def test_load_instances_reads_the_four_subsets_with_unique_ids(data_dir):
    inst = vc.load_instances(data_dir)
    counts = {s: sum(i.subset == s for i in inst) for s in vc.SUBSETS}
    assert counts == {"clean": 19, "asr_generic": 19, "asr_kor_or": 19, "overheard": 6}
    assert len({i.instance_id for i in inst}) == len(inst)
    assert inst[0].instance_id == f"clean/{inst[0].item['id']}"
    assert [i.subset for i in vc.load_instances(data_dir, ["overheard"])] == ["overheard"] * 6


def test_scenes_and_rules_load_as_the_prompt_needs_them(data_dir):
    assert vc.load_rules(data_dir) == RULES
    assert vc.load_scenes(data_dir) == SCENES


class FakeKhdp:
    def __init__(self, blobs: dict[str, bytes]):
        self.blobs, self.downloads = blobs, []

    def files(self, code, version):
        from benchtoward.khdp import KhdpFile

        assert (code, version) == ("kor-surgical-robot-voice-command", "1.0.0")
        return [KhdpFile(key="benchmark/", size=0, url="")] + [KhdpFile(key=k, size=len(v), url=k)
                                                              for k, v in self.blobs.items()]

    def download(self, file, dest):
        self.downloads.append(file.key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.blobs[file.key])
        return dest


def test_fetch_downloads_pinned_files_once_and_checks_them(tmp_path):
    blobs = {"README.md": b"readme", "benchmark/rules_v0.1.txt": b"rules"}
    pins = {k: hashlib.sha256(v).hexdigest() for k, v in blobs.items()}
    client = FakeKhdp(blobs)
    root = vc.fetch_dataset(tmp_path, client, files=pins)
    assert (root / "benchmark/rules_v0.1.txt").read_bytes() == b"rules"
    assert sorted(client.downloads) == sorted(blobs)
    vc.fetch_dataset(tmp_path, client, files=pins)
    assert len(client.downloads) == 2  # verified files are not downloaded again


def test_fetch_refuses_a_changed_release(tmp_path):
    client = FakeKhdp({"README.md": b"changed"})
    with pytest.raises(ValueError, match="checksum"):
        vc.fetch_dataset(tmp_path, client, files={"README.md": hashlib.sha256(b"readme").hexdigest()})
    assert not (vc.dataset_dir(tmp_path) / "README.md").exists()
    with pytest.raises(ValueError, match="missing"):
        vc.fetch_dataset(tmp_path, client, files={"benchmark/nope.jsonl": "0" * 64})


def test_every_release_file_is_pinned():
    assert set(vc.FILES) >= {*vc.SUBSETS.values(), vc.SCENES_FILE, vc.RULES_FILE, vc.TRAIN_FILE, "README.md"}
    assert all(len(h) == 64 for h in vc.FILES.values())


# --- metrics and the prior -------------------------------------------------------------------------------------------

def _rows_for(inst, predict):
    return [vc.score_item(i.item, predict(i)) for i in inst]


def test_task_metrics_primary_safety_and_subsets(data_dir):
    inst = vc.load_instances(data_dir)
    rows = _rows_for(inst, lambda i: oracle(i.item))
    m = vc.task_metrics(inst, rows)
    assert m["exact_match"]["value"] == 1.0 and m["exact_match"]["n"] == 19 and m["exact_match"]["n_groups"] == 11
    assert m["unsafe_execution"]["value"] == 0.0 and m["stop_recall"]["value"] == 1.0
    assert m["over_refusal"]["value"] == 0.0 and m["false_activation"]["value"] == 0.0
    assert m["exact_match_asr_generic"]["value"] == 1.0 and m["exact_match_asr_kor_or"]["value"] == 1.0
    assert set(m["subsets"]) == set(vc.SUBSETS) and m["subsets"]["overheard"]["n"] == 6
    assert m["by_subset"]["clean"] == {"value": 1.0, "n": 19}
    assert m["parse_rate"] == 1.0
    assert m["paired"]["asr_generic_vs_clean"]["diff_pp"] == 0.0


def test_cluster_interval_resamples_wording_variants_together():
    one_group = vc.cluster_rate([True, False, True, False], ["g", "g", "g", "g"])
    assert one_group["value"] == 0.5 and one_group["ci_low"] < 0.5 < one_group["ci_high"]  # never zero width
    spread = vc.cluster_rate([True, True, False, False], ["a", "a", "b", "b"])
    assert spread["ci_low"] == 0.0 and spread["ci_high"] == 1.0 and spread["n_groups"] == 2
    assert vc.cluster_rate([], [])["value"] is None


def test_prior_is_the_most_frequent_training_output_and_ignores_the_input(data_dir):
    out = vc.prior_output(vc.load_train(data_dir))
    assert out == {"response": "reject", "actions": [], "message": None}
    inst = vc.load_instances(data_dir)
    rows, metrics = vc.prior_run(inst, out)
    assert len(rows) == len(inst) and {r["prediction"] for r in rows} == {json.dumps(out, ensure_ascii=False)}
    clean = [i for i in inst if i.subset == "clean"]
    n_reject = sum(i.item["gold"]["response"] == "reject" for i in clean)
    assert metrics["exact_match"]["value"] == pytest.approx(n_reject / len(clean))
    assert metrics["over_refusal"]["value"] == 1.0 and metrics["unsafe_execution"]["value"] == 0.0
    assert metrics["false_activation"]["value"] == 0.0 and metrics["stop_recall"]["value"] == 0.0


# --- runner ----------------------------------------------------------------------------------------------------------

class FakeLLM:
    """Answers each utterance with the oracle command; can reject response_format to test the fallback."""

    def __init__(self, by_utterance: dict[str, dict], reject_schema: bool = False, fail_first: int = 0,
                 schema_errors: int = 0):
        self.by_utterance, self.reject_schema, self.fail_first = by_utterance, reject_schema, fail_first
        self.schema_errors = schema_errors  # this many json_schema requests fail with a non-transient error
        self.calls: list[dict] = []

    async def chat(self, model_id, messages, *, max_tokens, temperature=0.0, extra=None):
        self.calls.append({"model": model_id, "max_tokens": max_tokens, "temperature": temperature, "extra": extra,
                           "messages": messages})
        if self.fail_first > 0:
            self.fail_first -= 1
            return ChatResult(content="", reasoning=None, tokens_in=0, tokens_out=0, error="http_503", transient=True)
        schema = (extra or {}).get("response_format", {}).get("type") == "json_schema"
        if self.reject_schema and schema:
            return ChatResult(content="", reasoning=None, tokens_in=0, tokens_out=0, error="BadRequestError")
        if self.schema_errors > 0 and schema:
            self.schema_errors -= 1
            return ChatResult(content="", reasoning=None, tokens_in=0, tokens_out=0, error="http_404")
        utterance = json.loads(messages[1]["content"].split("[집도의 발화] ", 1)[1])
        return ChatResult(content=json.dumps(self.by_utterance[utterance], ensure_ascii=False), reasoning=None,
                          tokens_in=1000, tokens_out=40, error=None)


def _setup(tmp_session, data_dir):
    from conftest import REGISTRY_DIR

    from benchtoward.db import Model
    from benchtoward.registry import load_registry, resolve_task

    load_registry(tmp_session, REGISTRY_DIR)
    inst = vc.load_instances(data_dir)
    answers = {i.item["utterance"]: oracle(i.item)["parsed"] for i in inst}
    return (tmp_session.get(Model, "alibaba/qwen3.8-flash-next-180b-a6b"),
            resolve_task(tmp_session, "kor-surgical-robot-voice-command@0", include_hidden=True), inst, answers)


def test_runner_sends_the_original_request_and_scores_every_subset(tmp_session, data_dir):
    model, task, inst, answers = _setup(tmp_session, data_dir)
    client = FakeLLM(answers)
    run = asyncio.run(vc.run_command_task(tmp_session, model, task, inst, client, scenes=vc.load_scenes(data_dir),
                                          rules=vc.load_rules(data_dir), data_dir=data_dir, route=ROUTE))
    assert run.status == "done" and run.n_items == len(inst) == len(client.calls)
    call = client.calls[0]
    assert call["temperature"] == 0 and call["max_tokens"] == 800
    assert call["model"] == (model.provider_model_id or model.slug)  # the serving id where the registry has one
    assert call["extra"]["response_format"] == {"type": "json_schema", "json_schema": {
        "name": "robot_command", "strict": True, "schema": vc.json_schema()}}
    assert call["extra"]["chat_template_kwargs"] == {"enable_thinking": False}
    metrics = json.loads(run.metrics_json)
    assert metrics["exact_match"]["value"] == 1.0 and metrics["response_format"] == {"json_schema": len(inst)}
    assert (data_dir / "artifacts" / run.id / "predictions.parquet").exists()
    again = asyncio.run(vc.run_command_task(tmp_session, model, task, inst, client, scenes=vc.load_scenes(data_dir),
                                            rules=vc.load_rules(data_dir), data_dir=data_dir, route=ROUTE))
    assert again.id == run.id and len(client.calls) == len(inst)  # run once


def test_runner_resumes_after_transient_failures(tmp_session, data_dir):
    model, task, inst, answers = _setup(tmp_session, data_dir)
    inst = inst[:10]
    client = FakeLLM(answers, fail_first=3)
    kw = dict(scenes=vc.load_scenes(data_dir), rules=vc.load_rules(data_dir), data_dir=data_dir, concurrency=1,
              route=ROUTE)
    first = asyncio.run(vc.run_command_task(tmp_session, model, task, inst, client, **kw))
    assert first.status == "failed"
    second = asyncio.run(vc.run_command_task(tmp_session, model, task, inst, client, **kw))
    assert second.id == first.id and second.status == "done" and len(client.calls) == 13


def test_runner_falls_back_when_the_server_rejects_the_schema(tmp_session, data_dir):
    model, task, inst, answers = _setup(tmp_session, data_dir)
    inst = inst[:6]
    client = FakeLLM(answers, reject_schema=True)
    run = asyncio.run(vc.run_command_task(tmp_session, model, task, inst, client, scenes=vc.load_scenes(data_dir),
                                          rules=vc.load_rules(data_dir), data_dir=data_dir, concurrency=1,
                                          route=ROUTE))
    assert run.status == "done"
    metrics = json.loads(run.metrics_json)
    assert metrics["response_format"] == {"json_object": 6}
    # The format is downgraded for the rest of the run only after two items needed it.
    assert [c["extra"]["response_format"]["type"] for c in client.calls[:6]] == [
        "json_schema", "json_object", "json_schema", "json_object", "json_object", "json_object"]


def test_one_unrelated_error_does_not_downgrade_the_format_for_the_whole_run(tmp_session, data_dir):
    """A single non-transient error (e.g. a relay's 404 'model not loaded') followed by a successful retry in JSON
    mode must not switch every later item off the strict schema (review 2026-10-29)."""
    model, task, inst, answers = _setup(tmp_session, data_dir)
    inst = inst[:6]
    client = FakeLLM(answers, schema_errors=1)
    run = asyncio.run(vc.run_command_task(tmp_session, model, task, inst, client, scenes=vc.load_scenes(data_dir),
                                          rules=vc.load_rules(data_dir), data_dir=data_dir, concurrency=1,
                                          route=ROUTE))
    assert run.status == "done"
    assert json.loads(run.metrics_json)["response_format"] == {"json_object": 1, "json_schema": 5}


def test_runner_sends_the_routes_fields_and_refuses_an_undeclared_route(tmp_session, data_dir):
    model, task, inst, answers = _setup(tmp_session, data_dir)
    inst = inst[:2]
    route = {"thinking_off": {"reasoning": {"effort": "none"}}, "json_schema": {"provider": {"require_parameters": True}}}
    client = FakeLLM(answers)
    kw = dict(scenes=vc.load_scenes(data_dir), rules=vc.load_rules(data_dir), data_dir=data_dir)
    with pytest.raises(ValueError, match="thinking_off"):
        asyncio.run(vc.run_command_task(tmp_session, model, task, inst, client, route=None, **kw))
    assert client.calls == []
    run = asyncio.run(vc.run_command_task(tmp_session, model, task, inst, client, route=route, **kw))
    extra = client.calls[0]["extra"]
    assert extra["reasoning"] == {"effort": "none"} and extra["provider"] == {"require_parameters": True}
    assert "chat_template_kwargs" not in extra
    assert json.loads(run.metrics_json)["request"] == {"thinking_off": {"reasoning": {"effort": "none"}},
                                                       "json_schema": {"provider": {"require_parameters": True}}}


def test_spec_is_registered_with_the_protocol(tmp_session):
    import yaml
    from conftest import REGISTRY_DIR

    spec = yaml.safe_load((REGISTRY_DIR / "tasks" / "kor-surgical-robot-voice-command.yaml").read_text())
    assert (spec["slug"], spec["version"], spec["preview"], spec["primary_metric"]) == (
        "kor-surgical-robot-voice-command", 0, True, "exact_match")
    assert set(spec["subsets"]) == set(vc.SUBSETS)
    assert spec["request"]["temperature"] == 0 and spec["request"]["max_tokens"] == vc.MAX_TOKENS
    assert "10.23421/2026.03507" in json.dumps(spec)
    assert any("surgeon" in x for x in spec["limitations"])
    assert spec["deviations"]


def test_run_config_marks_runs_that_skip_a_subset_as_partial(data_dir):
    inst = vc.load_instances(data_dir)
    full = vc.run_config(inst)
    assert "partial" not in full and "limit" not in full and full["subsets"] == list(vc.SUBSETS)
    assert vc.run_config(vc.load_instances(data_dir, ["overheard"]))["partial"] is True
    assert vc.run_config(inst[:5], 5)["limit"] == 5
    assert vc.is_full_run(full) and not vc.is_full_run(vc.run_config(inst[:5], 5))
    assert not vc.is_full_run({"subsets": ["clean", "overheard"]})


def test_rates_at_zero_or_one_get_an_exact_interval_over_groups_not_a_zero_width_bootstrap():
    """16/16 stop commands in 8 groups: the bootstrap gives 1.0 (1.0-1.0). Every group is all-correct, so the
    Clopper-Pearson interval over groups is used (95%: 0.631-1.0), and the method is recorded (review 2026-10-29)."""
    groups = [f"g{k // 2}" for k in range(16)]
    hit = vc.cluster_rate([True] * 16, groups)
    assert hit["value"] == 1.0 and hit["ci_high"] == 1.0
    assert hit["ci_low"] == pytest.approx(0.025 ** (1 / 8), abs=1e-6)
    assert hit["ci_method"] == "clopper_pearson_groups" and hit["n"] == 16 and hit["n_groups"] == 8
    miss = vc.cluster_rate([False] * 16, groups)
    assert miss["value"] == 0.0 and miss["ci_low"] == 0.0 and miss["ci_high"] == pytest.approx(1 - 0.025 ** (1 / 8))
    mid = vc.cluster_rate([True, False] * 8, [f"g{k}" for k in range(16)])
    assert mid["ci_method"] == "cluster_bootstrap" and mid["ci_low"] < 0.5 < mid["ci_high"]


# --- request settings per serving route (review 2026-10-29) -----------------------------------------------------------

ROUTE = {"thinking_off": {"chat_template_kwargs": {"enable_thinking": False}}}


def test_request_extra_takes_the_routes_thinking_switch_with_per_model_overrides():
    """The original runner sent each provider's own 'thinking off' fields and per-model fields where thinking cannot
    be switched off (lowest reasoning effort), plus provider-specific fields in strict-schema mode."""
    route = {"thinking_off": {"reasoning": {"effort": "none"}}, "json_schema": {"provider": {"require_parameters": True}},
             "models": {"vendor/reasoner": {"thinking_off": {"reasoning": {"effort": "low"}}}}}
    plain = vc.request_extra("json_schema", vc.route_request(route, "vendor/plain"))
    assert plain["reasoning"] == {"effort": "none"} and plain["provider"] == {"require_parameters": True}
    assert plain["response_format"]["type"] == "json_schema" and "chat_template_kwargs" not in plain
    reasoner = vc.request_extra("json_object", vc.route_request(route, "vendor/reasoner"))
    assert reasoner == {"reasoning": {"effort": "low"}, "response_format": {"type": "json_object"}}
    assert vc.request_extra("none", vc.route_request(ROUTE, "m")) == ROUTE["thinking_off"]


def test_a_route_that_does_not_say_how_to_switch_thinking_off_is_refused():
    for route in (None, {}, {"json_schema": {}}):
        with pytest.raises(ValueError, match="thinking_off"):
            vc.route_request(route, "m")


def test_settings_read_the_request_block_of_each_route(tmp_path, monkeypatch):
    from benchtoward.config import get_settings

    (tmp_path / "providers.yaml").write_text(
        "example: {base_url: 'https://llm.example/v1', key_env: EXAMPLE_API_KEY,"
        " request: {thinking_off: {chat_template_kwargs: {enable_thinking: false}}}}\n"
        "other: {base_url: 'https://other.example/v1', key_env: OTHER_KEY}\n")
    monkeypatch.setenv("BENCHTOWARD_REGISTRY_DIR", str(tmp_path))
    assert get_settings().route_requests == {"example": ROUTE}


def test_spec_states_what_the_review_found():
    """Review 2026-10-29: replies at temperature 0 are not reproducible item by item on a serving route, intervals at
    0% or 100% are exact intervals, the format downgrade needs two items, and thinking off is set per route."""
    import yaml
    from conftest import REGISTRY_DIR

    spec = yaml.safe_load((REGISTRY_DIR / "tasks" / "kor-surgical-robot-voice-command.yaml").read_text())
    limits = " ".join(spec["limitations"])
    assert "temperature 0" in limits and "not reproducible" in limits and "not resolved" in limits
    assert "Clopper-Pearson" in spec["metrics"]["intervals"]
    deviations = " ".join(spec["deviations"])
    assert "two items" in deviations and "reasoning effort" in deviations
    assert "providers.yaml" in spec["request"]["thinking"]
    assert "no-action rate" in spec["metrics"]["false_activation"]
