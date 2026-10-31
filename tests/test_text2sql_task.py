"""synthetic-snuh-text2sql@0: the frozen two-tier Text-to-SQL task (kind text2sql).

The spec pins both tiers' items and prompts; the runner asks each model once per instance (or reuses a dry run's
answer to the same prompt within the same output limit); the scorer reports RS(0) on the hard tier as the headline,
execution accuracy on both tiers, and 95% intervals from a bootstrap over questions (Korean and English together).
"""
import asyncio
import copy
import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from conftest import REGISTRY_DIR, make_omop_db, plain
from benchtoward.db import ItemResult, Model, Run, Task
from benchtoward.llm import ChatResult
from benchtoward.registry import load_registry
from benchtoward.tasks.synthetic_snuh import connect_readonly, db_path
from benchtoward.tasks.text2sql import ABSTAIN, HARD_PROMPT, P0, _prompt_sha, render_messages, schema_summary
from benchtoward.tasks import text2sql_task as t2s

SPEC_FILE = REGISTRY_DIR / "tasks" / "synthetic-snuh-text2sql.yaml"
DRAFTS = REGISTRY_DIR / "drafts"
needs_drafts = pytest.mark.skipif(not (DRAFTS / "synthetic-snuh-text2sql-hard-v0.yaml").exists(),
                                  reason="the question files are released on KHDP, not with the code")
# The code is public ahead of the task (owner, 2026-10-30); the spec goes out only when the task is published.
needs_spec = pytest.mark.skipif(not SPEC_FILE.exists(),
                                reason="the task's spec stays private while the task is in review")
# What these tests use of the spec, for the public repository, where the spec is not (yet) exported; the private
# repository checks that it agrees with the registered spec (test_the_stand_in_spec_agrees_with_the_registered_one).
STAND_IN_SPEC = {
    "slug": "synthetic-snuh-text2sql", "version": 0, "preview": True, "visibility": "open",
    "dataset": "synthetic-snuh", "kind": "text2sql", "primary_metric": "rs0",
    "title": "Synthetic-SNUH Text-to-SQL (OMOP CDM, Korean and English)",
    "tiers": {"hard": {"abstention": True}, "easy": {"abstention": False}},
    "request": {"temperature": 0, "max_tokens": 16384},
    "check": {"roster": {"weak": "openai/gpt-oss-20b", "strong": "deepseek/deepseek-v4.1-flash-552b-a16b"},
              "priors": {"rs": "benchtoward/prior", "ex": "benchtoward/text2sql-constant-query"}},
}


def base_spec() -> dict:
    return yaml.safe_load(SPEC_FILE.read_text(encoding="utf-8")) if SPEC_FILE.exists() else copy.deepcopy(STAND_IN_SPEC)

EASY = [
    {"id": "e-001", "question_ko": "여성 환자는 몇 명인가?", "question_en": "How many female patients are there?",
     "category": "cohort_count", "difficulty": "easy", "scope": "group", "review": "approved",
     "sql": "SELECT count(*) FROM person WHERE gender_concept_id = 8532"},
    {"id": "e-002", "question_ko": "진료과별 방문 수를 구하라.", "question_en": "Number of visits per department.",
     "category": "aggregation", "difficulty": "medium", "scope": "none", "review": "approved",
     "sql": "SELECT visit_source_value, count(*) FROM visit_occurrence GROUP BY 1"},
]
HARD = [
    {"id": "h-001", "question_ko": "당뇨병 환자는 몇 명인가?", "question_en": "How many patients have T2DM?",
     "category": "clinical_terms", "difficulty": "hard", "scope": "group", "review": "approved",
     "sql": "SELECT count(DISTINCT person_id) FROM condition_occurrence WHERE condition_concept_id = 201826"},
    {"id": "h-002", "question_ko": "검체의 채취 부위는?", "question_en": "Where were the specimens taken from?",
     "category": "unanswerable", "difficulty": "hard", "scope": "none", "review": "approved", "answerable": False,
     "sql": None},
]
RIGHT = {"e-001": "SELECT count(*) FROM person WHERE gender_concept_id = 8532",
         "e-002": "SELECT visit_source_value, count(*) FROM visit_occurrence GROUP BY 1",
         "h-001": "SELECT count(DISTINCT person_id) FROM condition_occurrence WHERE condition_concept_id = 201826",
         "h-002": ABSTAIN}


def _write_draft(path: Path, prompt_id: str, questions: list[dict]) -> Path:
    doc = {"status": "frozen", "prompt": {"id": prompt_id}, "questions": questions}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


class Fixture:
    """A registry with a two-tier text2sql spec over a miniature OMOP database."""

    def __init__(self, tmp_path: Path):
        self.registry = tmp_path / "registry"
        self.data_dir = tmp_path / "data"
        make_omop_db(db_path(self.data_dir))
        self.con = connect_readonly(db_path(self.data_dir))
        _write_draft(self.registry / "drafts" / "easy.yaml", P0.id, EASY)
        _write_draft(self.registry / "drafts" / "hard.yaml", HARD_PROMPT.id, HARD)
        self.spec = self.make_spec()

    def make_spec(self) -> dict:
        spec = base_spec()
        tiers = {}
        for name, file, prompt in (("hard", "drafts/hard.yaml", HARD_PROMPT), ("easy", "drafts/easy.yaml", P0)):
            meta, items = t2s.load_items(self.registry / file)
            tiers[name] = {**spec["tiers"][name],
                           "items": {"file": file, "questions": len(items), "sha256": t2s.items_sha256(items)},
                           "prompt": {"id": prompt.id, "sha256": t2s.prompt_sha256(prompt),
                                      "schema_sha256": t2s.schema_sha256(
                                          schema_summary(self.con, null_columns=prompt.null_columns))}}
        return {**spec, "tiers": tiers}

    def tiers(self):
        return t2s.load_tiers(self.spec, self.registry)

    def write_registry(self) -> Path:
        """The fixture registry with the repository's datasets and models and this spec as the task's."""
        for name in ("datasets.yaml", "models.yaml"):
            (self.registry / name).write_text((REGISTRY_DIR / name).read_text(encoding="utf-8"), encoding="utf-8")
        (self.registry / "tasks").mkdir(exist_ok=True)
        (self.registry / "tasks" / "synthetic-snuh-text2sql.yaml").write_text(
            yaml.safe_dump(self.spec, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return self.registry


@pytest.fixture
def fx(tmp_path):
    return Fixture(tmp_path)


# ---- the registered spec ----

@needs_spec
def test_the_stand_in_spec_agrees_with_the_registered_one():
    spec = yaml.safe_load(SPEC_FILE.read_text(encoding="utf-8"))
    for key, value in STAND_IN_SPEC.items():
        if key == "tiers":
            assert {t: {"abstention": v["abstention"]} for t, v in spec["tiers"].items()} == value
        elif isinstance(value, dict):
            assert {k: spec[key][k] for k in value} == value, key
        else:
            assert spec[key] == value, key


@needs_spec
def test_spec_is_one_hidden_preview_task_with_two_tiers():
    spec = yaml.safe_load(SPEC_FILE.read_text(encoding="utf-8"))
    assert (spec["slug"], spec["version"], spec["preview"]) == ("synthetic-snuh-text2sql", 0, True)
    assert (spec["kind"], spec["primary_metric"], spec["dataset"]) == ("text2sql", "rs0", "synthetic-snuh")
    assert list(spec["tiers"]) == ["hard", "easy"]  # decision easy_tier_publication: one task, two tiers
    assert spec["request"]["max_tokens"] == 16384 and spec["request"]["temperature"] == 0  # decision output_format
    assert spec["check"]["roster"] == {"weak": "openai/gpt-oss-20b",
                                       "strong": "deepseek/deepseek-v4.1-flash-552b-a16b"}  # reference_roster
    assert spec["check"]["priors"] == {"rs": "benchtoward/prior", "ex": "benchtoward/text2sql-constant-query"}


@needs_spec
@needs_drafts
def test_spec_pins_the_frozen_items_and_prompts():
    spec = yaml.safe_load(SPEC_FILE.read_text(encoding="utf-8"))
    tiers = {t.name: t for t in t2s.load_tiers(spec, REGISTRY_DIR)}
    assert len(tiers["hard"].items) == 101 and len(tiers["easy"].items) == 154
    assert sum(not q.answerable for q in tiers["hard"].items) == 18
    assert tiers["hard"].prompt is HARD_PROMPT and tiers["easy"].prompt is P0
    assert spec["tiers"]["hard"]["items"]["file"] == "drafts/synthetic-snuh-text2sql-hard-v0.yaml"


def test_an_edited_item_or_prompt_is_refused(fx):
    fx.tiers()  # as pinned
    edited = copy.deepcopy(HARD)
    edited[0]["question_en"] = "How many patients have type 2 diabetes?"
    _write_draft(fx.registry / "drafts" / "hard.yaml", HARD_PROMPT.id, edited)
    with pytest.raises(ValueError, match="items of tier hard"):
        fx.tiers()
    _write_draft(fx.registry / "drafts" / "hard.yaml", HARD_PROMPT.id, HARD)
    fx.spec["tiers"]["easy"]["prompt"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="prompt of tier easy"):
        fx.tiers()


def test_rejected_items_are_not_part_of_a_tier(fx):
    _write_draft(fx.registry / "drafts" / "easy.yaml", P0.id, [*EASY, {**EASY[0], "id": "e-009", "review": "rejected"}])
    assert [q.id for q in {t.name: t for t in fx.tiers()}["easy"].items] == ["e-001", "e-002"]  # same hash as before


def test_schema_must_match_the_pinned_prompt(fx):
    t2s.check_schemas(fx.spec, fx.tiers(), fx.con)
    fx.spec["tiers"]["hard"]["prompt"]["schema_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="schema of tier hard"):
        t2s.check_schemas(fx.spec, fx.tiers(), fx.con)


# the runs' (PR #109 review), moved by the rename-only plan registry/renames/2026-11-benchtoward.yaml (was abb0fa4a)
RUN_SPEC_HASH = "ac017b1c69ca1976ff65ebd4469eb681344d4db9dacf9f137f61347344340a25"


@needs_spec
def test_the_items_release_and_its_doi_never_change_the_spec_hash():
    """Owner, 2026-10-30: the items are released on KHDP as a BenchToward-built dataset. The spec says where, and gets
    the DOI once KHDP assigns it, in `items_release`, a presentational field: recording either leaves the spec hash,
    and so every recorded run, as it is."""
    from benchtoward.registry import PRESENTATIONAL_FIELDS
    from benchtoward.runkey import spec_hash

    spec = yaml.safe_load(SPEC_FILE.read_text(encoding="utf-8"))
    release = spec["items_release"]
    assert release["khdp"] == "synthetic-snuh-text2sql@1.0.0" and release["doi"] is None
    assert "docs/khdp/synthetic-snuh-text2sql.md" in release["package"]
    assert "items_release" in PRESENTATIONAL_FIELDS

    def hashed(s: dict) -> str:
        return spec_hash({k: v for k, v in s.items() if k not in PRESENTATIONAL_FIELDS})
    assert hashed(spec) == RUN_SPEC_HASH
    assert hashed({**spec, "items_release": {**release, "doi": "10.23421/2026.99999"}}) == RUN_SPEC_HASH


@needs_spec
def test_spec_is_a_protocol_transfer_with_every_benchmark_made_part_listed():
    """SOP A1 protocol transfers (PR #97): the authors' scoring code at a full commit, and one deviation starting with
    the name of each part BenchToward made."""
    spec = yaml.safe_load(SPEC_FILE.read_text(encoding="utf-8"))
    protocol = spec["protocol"]
    assert protocol["transfer"] is True and protocol["repository"] == "https://github.com/glee4810/ehrsql-2024"
    assert len(protocol["commit"]) == 40 and all(c in "0123456789abcdef" for c in protocol["commit"])
    heads = {" ".join(str(d).split()).partition(":")[0].strip().lower() for d in spec["deviations"] if ":" in str(d)}
    assert protocol["benchmark_made"] and all(part.lower() in heads for part in protocol["benchmark_made"])


@needs_spec
def test_registry_loads_the_task_with_its_dataset_and_priors(tmp_session):
    from benchtoward.registry import resolve_task

    load_registry(tmp_session, REGISTRY_DIR)
    task = tmp_session.query(Task).filter_by(slug="synthetic-snuh-text2sql", version=0).one()
    spec = yaml.safe_load(SPEC_FILE.read_text(encoding="utf-8"))
    assert (task.visibility, task.kind, task.dataset_slug) == (spec["visibility"], "text2sql", "synthetic-snuh")
    if task.visibility == "hidden":  # in review: commands reach it, the site does not
        with pytest.raises(LookupError):
            resolve_task(tmp_session, "synthetic-snuh-text2sql@0")
    assert resolve_task(tmp_session, "synthetic-snuh-text2sql@0", include_hidden=True).id == task.id
    for slug in ("benchtoward/prior", "benchtoward/text2sql-constant-query"):
        model = tmp_session.get(Model, slug)
        assert model.type == "baseline" and model.reference


# ---- intervals and the reliability score ----

def test_interval_resamples_both_languages_of_a_question_together():
    values = [1.0, 0.0] * 50  # every question: Korean right, English wrong
    groups = [f"q{i // 2}" for i in range(100)]
    ci = t2s.cluster_ci(values, groups)
    assert ci["value"] == 0.5 and ci["ci_low"] == ci["ci_high"] == 0.5  # each resample keeps both languages
    assert ci["n"] == 100 and ci["n_groups"] == 50 and "questions" in ci["ci_method"]
    apart = t2s.cluster_ci(values, [str(i) for i in range(100)])  # instances as their own clusters
    assert apart["ci_low"] < 0.5 < apart["ci_high"]


def test_reliability_score_is_the_ehrsql_2024_scorer():
    """RS(c) comes from the authors' scoring_utils.py (vendored unchanged): +1 right or abstaining on an
    unanswerable question, 0 abstaining on an answerable one, -c any other answer."""
    outcomes = [(True, "right"), (True, "abstain"), (True, "wrong"), (False, "abstain"), (False, "wrong")]
    scores = t2s.ehrsql_scores(outcomes)
    assert scores == [1, 0, -1, 1, -1]
    upstream = t2s.ehrsql_scorer()
    for c in (0, 5, len(scores)):
        assert t2s.rs(scores, c) == pytest.approx(float(upstream.penalize(scores, penalty=c)))
    assert t2s.rs(scores, 0) == pytest.approx(0.4) and t2s.rs(scores, 5) == pytest.approx(-1.6)


# ---- outcomes ----

@pytest.mark.parametrize("content, error, tokens, abstention, expected", [
    ("```sql\nSELECT 1\n```", None, 100, False, ("SELECT 1", "sql")),
    ("```sql\nABSTAIN\n```", None, 100, True, (ABSTAIN, "abstain")),
    ("", None, 16384, True, (None, "cut_off")),             # thinking used the whole limit: a model failure
    ("I cannot write this query.", None, 900, True, (None, "format_failure")),  # text without SQL or ABSTAIN
    ("", None, 300, False, (None, "format_failure")),       # empty before the limit: not a cut-off
    ("", "invalid_request", 0, False, (None, "api_error")),
])
def test_outcomes_tell_a_cut_off_from_a_format_failure(content, error, tokens, abstention, expected):
    assert t2s.read_answer(content, error, tokens, max_tokens=16384, abstention=abstention) == expected


# ---- scoring ----

def _answers(tiers, overrides=None):
    out = {}
    for inst in t2s.instances(tiers):
        pred = (overrides or {}).get(inst.instance_id, RIGHT[inst.question.id])
        out[inst.instance_id] = {"prediction": pred, "outcome": "abstain" if pred == ABSTAIN else "sql"}
    return out


def test_scores_both_tiers_with_rs0_as_the_headline(fx):
    tiers = fx.tiers()
    answers = _answers(tiers, {"hard/h-001-en": None, "easy/e-002-ko": "SELECT 1"})
    answers["hard/h-001-en"]["outcome"] = "cut_off"
    rows, m = t2s.score_answers(fx.con, tiers, answers)
    assert [r["instance_id"] for r in rows][:4] == ["hard/h-001-ko", "hard/h-001-en", "hard/h-002-ko",
                                                    "hard/h-002-en"]
    assert m["rs0"]["value"] == pytest.approx(3 / 4)       # right, cut off (-0), abstained twice on unanswerable
    assert m["hard_rs5"]["value"] == pytest.approx((3 - 5) / 4)
    assert m["hard_ex"]["value"] == pytest.approx(1 / 2) and m["hard_ex"]["n"] == 2
    assert m["easy_ex"]["value"] == pytest.approx(3 / 4) and m["easy_ex"]["n_groups"] == 2
    for key in ("rs0", "hard_ex", "easy_ex"):
        assert m[key]["ci_low"] <= m[key]["value"] <= m[key]["ci_high"]
    assert m["answers"]["hard"]["cut_off"] == 1 and m["cut_off_rate"]["hard"] == pytest.approx(1 / 4)
    assert m["format_failure_rate"]["all"] == 0 and m["api_error_rate"] == 0
    assert m["abstention"]["unanswerable_abstained"]["value"] == 1.0
    assert m["parse_rate"] == pytest.approx(7 / 8)
    assert m["scorer"]["reliability"].startswith("glee4810/ehrsql-2024@")
    by_id = {r["instance_id"]: r for r in rows}
    assert by_id["hard/h-002-ko"]["score"] == 1 and by_id["hard/h-001-en"]["score"] == -1
    assert by_id["easy/e-002-ko"]["correct"] is False and by_id["easy/e-002-ko"]["score"] == 0


def test_priors_are_fixed_rules(fx):
    """Input-blind rules: abstain on every item, and SELECT 0 for every item."""
    tiers = fx.tiers()
    _, abstain = t2s.score_answers(fx.con, tiers, t2s.prior_answers(tiers, "abstain"))
    assert abstain["rs0"]["value"] == pytest.approx(2 / 4)  # the unanswerable share of the hard tier
    assert abstain["hard_ex"]["value"] == 0 and abstain["easy_ex"]["value"] == 0
    _, constant = t2s.score_answers(fx.con, tiers, t2s.prior_answers(tiers, "constant"))
    assert constant["rs0"]["value"] == 0 and constant["hard_ex"]["value"] == 0 and constant["easy_ex"]["value"] == 0
    with pytest.raises(ValueError):
        t2s.prior_answers(tiers, "best_constant_on_test")


# ---- runner ----

class FakeClient:
    def __init__(self, answers: dict[str, str] | None = None, tokens_out: int = 50):
        self.calls = []
        self.answers = answers or {}
        self.tokens_out = tokens_out

    async def chat(self, model_id, messages, *, max_tokens, temperature=0.0):
        self.calls.append((model_id, messages[-1]["content"].rsplit("Question: ", 1)[1], max_tokens, temperature))
        question = messages[-1]["content"].rsplit("Question: ", 1)[1]
        qid = next(q["id"] for q in EASY + HARD if question in (q["question_ko"], q["question_en"]))
        return ChatResult(content=f"```sql\n{RIGHT[qid]}\n```", reasoning=None, tokens_in=10,
                          tokens_out=self.tokens_out, error=None)


def _served(model) -> str:
    """The id a route serves the model under (the public registry has none: the slug)."""
    return model.provider_model_id or model.slug


def _db(tmp_session, fx):
    load_registry(tmp_session, fx.write_registry())
    task = tmp_session.query(Task).filter_by(slug="synthetic-snuh-text2sql").one()
    model = tmp_session.get(Model, "deepseek/deepseek-v4.1-flash-552b-a16b")
    return task, model


def _cache(fx, tiers, model_id: str, tier: str, path: Path, *, wrong_sha_for=(), tokens_out=40, limit=None,
           empty=None) -> Path:
    """A dry run's responses.jsonl for one tier (instance ids without the tier). `limit` is the output limit the
    records carry (dry runs made before the review of PR #109 carry none); `empty` maps instance ids to the
    tokens_out of an answer with empty content (a thinking run that left no SQL)."""
    empty = empty or {}
    schema = schema_summary(fx.con, null_columns=next(t for t in tiers if t.name == tier).prompt.null_columns)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "responses.jsonl").open("w", encoding="utf-8") as fh:
        for inst in t2s.instances(tiers):
            if inst.tier != tier:
                continue
            sha = _prompt_sha(model_id, render_messages(inst.question, inst.lang, schema, inst.prompt))
            if inst.local_id in wrong_sha_for:
                sha = "0" * 16
            record = {"instance_id": inst.local_id, "prompt_sha": sha,
                      "content": "" if inst.local_id in empty else f"```sql\n{RIGHT[inst.question.id]}\n```",
                      "error": None, "transient": None, "tokens_in": 9,
                      "tokens_out": empty.get(inst.local_id, tokens_out)}
            if limit is not None:
                record["max_tokens"] = limit
            fh.write(json.dumps(record) + "\n")
    return path


def test_runner_reuses_a_dry_run_answer_only_for_the_same_prompt(fx, tmp_session):
    task, model = _db(tmp_session, fx)
    tiers = fx.tiers()
    cache = _cache(fx, tiers, _served(model), "hard", fx.data_dir / "dry-hard", wrong_sha_for={"h-001-ko"})
    client = FakeClient()
    run = asyncio.run(t2s.run_text2sql_task(tmp_session, model, task, tiers, fx.con, client, spec=fx.spec,
                                            reuse={"hard": cache}, data_dir=fx.data_dir))
    assert run.status == "done", run.metrics_json
    asked = {(c[1]) for c in client.calls}
    assert HARD[0]["question_ko"] in asked and HARD[0]["question_en"] not in asked  # only the changed prompt
    assert len(client.calls) == 1 + 4  # h-001-ko, and the easy tier (no cache given)
    assert all(c[2] == 16384 and c[3] == 0 for c in client.calls)
    m = json.loads(run.metrics_json)
    assert m["answers_from_cache"] == 3 and m["rs0"]["value"] == 1.0 and m["easy_ex"]["value"] == 1.0
    assert json.loads(run.config_json) == {"prompts": {"hard": HARD_PROMPT.id, "easy": P0.id}, "max_tokens": 16384,
                                           "temperature": 0, "tiers": ["hard", "easy"]}
    assert run.n_items == 8 and tmp_session.query(ItemResult).filter_by(run_id=run.id).count() == 8
    assert (fx.data_dir / "artifacts" / run.id / "predictions.parquet").exists()
    again = asyncio.run(t2s.run_text2sql_task(tmp_session, model, task, tiers, fx.con, FakeClient(), spec=fx.spec,
                                              data_dir=fx.data_dir))
    assert again.id == run.id and tmp_session.query(Run).count() == 1  # run once


def test_runner_refuses_a_cache_from_a_larger_output_limit(fx, tmp_session):
    task, model = _db(tmp_session, fx)
    tiers = fx.tiers()
    cache = _cache(fx, tiers, _served(model), "hard", fx.data_dir / "dry-32k", tokens_out=20000)
    with pytest.raises(ValueError, match="over the output limit"):
        asyncio.run(t2s.run_text2sql_task(tmp_session, model, task, tiers, fx.con, FakeClient(), spec=fx.spec,
                                          reuse={"hard": cache}, data_dir=fx.data_dir))


def test_runner_refuses_a_cache_recorded_under_another_output_limit(fx, tmp_session):
    """Review of PR #109: a dry run now records its limit, and a folder asked under another one is refused, even
    when no answer went over the task's limit."""
    task, model = _db(tmp_session, fx)
    tiers = fx.tiers()
    cache = _cache(fx, tiers, _served(model), "hard", fx.data_dir / "dry-8k", limit=8192)
    with pytest.raises(ValueError, match="limit of 8192"):
        asyncio.run(t2s.run_text2sql_task(tmp_session, model, task, tiers, fx.con, FakeClient(), spec=fx.spec,
                                          reuse={"hard": cache}, data_dir=fx.data_dir))
    same = _cache(fx, tiers, _served(model), "hard", fx.data_dir / "dry-16k", limit=16384)
    run = asyncio.run(t2s.run_text2sql_task(tmp_session, model, task, tiers, fx.con, FakeClient(), spec=fx.spec,
                                            reuse={"hard": same}, data_dir=fx.data_dir))
    assert json.loads(run.metrics_json)["answers_from_cache"] == 4


def test_runner_refuses_an_unrecorded_limit_with_an_answer_left_short_of_it(fx, tmp_session):
    """Older dry runs record no limit. An answer without SQL or ABSTAIN that stopped below the task's limit may have
    been cut off by a smaller one (it would be counted as a format failure), so such a folder is refused; answers
    without SQL that used exactly the task's limit are cut off at it and the folder is reused."""
    task, model = _db(tmp_session, fx)
    tiers = fx.tiers()
    short = _cache(fx, tiers, _served(model), "hard", fx.data_dir / "dry-short", empty={"h-001-en": 8192})
    with pytest.raises(ValueError, match="no output limit recorded"):
        asyncio.run(t2s.run_text2sql_task(tmp_session, model, task, tiers, fx.con, FakeClient(), spec=fx.spec,
                                          reuse={"hard": short}, data_dir=fx.data_dir))
    at_limit = _cache(fx, tiers, _served(model), "hard", fx.data_dir / "dry-at-limit", empty={"h-001-en": 16384})
    run = asyncio.run(t2s.run_text2sql_task(tmp_session, model, task, tiers, fx.con, FakeClient(), spec=fx.spec,
                                            reuse={"hard": at_limit}, data_dir=fx.data_dir))
    m = json.loads(run.metrics_json)
    assert m["answers_from_cache"] == 4 and m["answers"]["hard"]["cut_off"] == 1


def test_runner_refuses_a_model_that_cannot_answer_within_the_protocol_limit(fx, tmp_session):
    task, _ = _db(tmp_session, fx)
    short = tmp_session.get(Model, "google/medgemma-27b")  # 16K context, max_tokens 8192
    with pytest.raises(ValueError, match="16384"):
        asyncio.run(t2s.run_text2sql_task(tmp_session, short, task, fx.tiers(), fx.con, FakeClient(), spec=fx.spec,
                                          data_dir=fx.data_dir))


def test_runner_without_a_client_needs_every_answer_cached(fx, tmp_session):
    task, model = _db(tmp_session, fx)
    tiers = fx.tiers()
    reuse = {"hard": _cache(fx, tiers, _served(model), "hard", fx.data_dir / "h"),
             "easy": _cache(fx, tiers, _served(model), "easy", fx.data_dir / "e")}
    run = asyncio.run(t2s.run_text2sql_task(tmp_session, model, task, tiers, fx.con, None, spec=fx.spec,
                                            reuse=reuse, data_dir=fx.data_dir))
    assert run.status == "done" and json.loads(run.metrics_json)["answers_from_cache"] == 8
    other = tmp_session.get(Model, "openai/gpt-oss-20b")
    with pytest.raises(ValueError, match="need a model call"):
        asyncio.run(t2s.run_text2sql_task(tmp_session, other, task, tiers, fx.con, None, spec=fx.spec,
                                          reuse=reuse, data_dir=fx.data_dir))  # cached for another model


def test_cut_off_answers_are_wrong_and_counted_apart(fx, tmp_session):
    task, model = _db(tmp_session, fx)

    class Thinker(FakeClient):
        async def chat(self, model_id, messages, *, max_tokens, temperature=0.0):
            return ChatResult(content="", reasoning="...", tokens_in=10, tokens_out=max_tokens, error=None)

    run = asyncio.run(t2s.run_text2sql_task(tmp_session, model, task, fx.tiers(), fx.con, Thinker(), spec=fx.spec,
                                            data_dir=fx.data_dir))
    m = json.loads(run.metrics_json)
    assert m["cut_off_rate"]["all"] == 1.0 and m["format_failure_rate"]["all"] == 0.0
    assert m["rs0"]["value"] == 0.0 and m["hard_rs5"]["value"] == -5.0 and m["easy_ex"]["value"] == 0.0


def test_baselines_are_recorded_as_runs(fx, tmp_session):
    task, _ = _db(tmp_session, fx)
    runs = t2s.record_priors(tmp_session, task, fx.tiers(), fx.con, data_dir=fx.data_dir)
    by = {r.model_slug: json.loads(r.metrics_json) for r in runs}
    assert by["benchtoward/prior"]["rs0"]["value"] == 0.5
    assert by["benchtoward/text2sql-constant-query"]["rs0"]["value"] == 0.0
    assert json.loads(runs[1].config_json)["rule"] == "SELECT 0 for every item of both tiers"


# ---- the SOP check ----

def _m(rs0, rs0_ci, hard_ex, hard_ex_ci, easy_ex, easy_ex_ci, fmt=0.0, cut=0.0, api=0.0):
    entry = lambda v, ci: {"value": v, "ci_low": ci[0], "ci_high": ci[1]}  # noqa: E731
    return {"rs0": entry(rs0, rs0_ci), "hard_ex": entry(hard_ex, hard_ex_ci), "easy_ex": entry(easy_ex, easy_ex_ci),
            "format_failure_rate": {"hard": fmt, "easy": fmt, "all": fmt},
            "cut_off_rate": {"hard": cut, "easy": 0.0, "all": cut}, "api_error_rate": api}


CHECK_SPEC = {"roster": {"weak": "w", "strong": "s"}, "priors": {"rs": "p", "ex": "c"}}
PRIORS = {"p": _m(0.178, (0.178, 0.178), 0, (0, 0), 0, (0, 0)), "c": _m(0, (0, 0), 0, (0, 0), 0, (0, 0))}


def test_sop_check_passes_the_hard_tier_and_fails_the_saturated_easy_tier():
    runs = {**PRIORS, "w": _m(0.22, (0.16, 0.29), 0.19, (0.13, 0.27), 0.90, (0.86, 0.94), cut=0.12),
            "s": _m(0.66, (0.58, 0.74), 0.60, (0.50, 0.69), 0.994, (0.98, 1.0), cut=0.12)}
    report = t2s.sop_check(CHECK_SPEC, runs)
    hard, easy = report["tiers"]["hard"], report["tiers"]["easy"]
    assert hard["headroom"]["H"] == pytest.approx((1 - 0.66) / (1 - 0.178)) and hard["headroom"]["pass"]
    assert hard["separation"]["pass"] and hard["format"]["pass"] and hard["verdict"] == "pass"
    assert hard["format"]["cut_off"] == {"w": 0.12, "s": 0.12}  # reported, not counted as format failures
    assert not hard["prior"]["references_beyond_prior"]["w"]  # 0.16 is not beyond 0.178: shown, not a criterion
    assert easy["headroom"]["H"] == pytest.approx(0.006) and not easy["headroom"]["pass"]
    assert easy["verdict"] == "fail"


def test_sop_check_states_what_a_pass_rests_on():
    """Review of PR #109: the verdict carries its caveats. "Beats the prior" has nothing to test; a reference whose
    interval on the tier's metric includes the prior is named; and a format pass that holds only because answers cut
    off at the limit are model failures (decision output_format) says so."""
    runs = {**PRIORS, "w": _m(0.22, (0.16, 0.29), 0.19, (0.13, 0.27), 0.90, (0.86, 0.94), cut=0.124),
            "s": _m(0.66, (0.58, 0.74), 0.60, (0.50, 0.69), 0.994, (0.98, 1.0), cut=0.124)}
    report = t2s.sop_check(CHECK_SPEC, runs)
    hard, easy = report["tiers"]["hard"], report["tiers"]["easy"]
    assert hard["verdict"] == "pass"
    assert [c["kind"] for c in hard["caveats"]] == ["beats_prior_untested", "within_prior", "format_rests_on_cut_off"]
    within, fmt = hard["caveats"][1], hard["caveats"][2]
    assert "w" in within["text"] and "0.178" in within["text"] and "s" not in within["references"]
    assert fmt["references"] == {"w": pytest.approx(0.124), "s": pytest.approx(0.124)} and "12.4%" in fmt["text"]
    assert [c["kind"] for c in easy["caveats"]] == ["beats_prior_untested"]
    runs["w"] = _m(0.30, (0.20, 0.38), 0.19, (0.13, 0.27), 0.90, (0.86, 0.94), cut=0.05)
    runs["s"] = _m(0.66, (0.58, 0.74), 0.60, (0.50, 0.69), 0.994, (0.98, 1.0))
    assert [c["kind"] for c in t2s.sop_check(CHECK_SPEC, runs)["tiers"]["hard"]["caveats"]] == ["beats_prior_untested"]


def test_sop_check_format_and_api_errors():
    runs = {**PRIORS, "w": _m(0.22, (0.16, 0.29), 0.19, (0.13, 0.27), 0.90, (0.86, 0.94), fmt=0.11),
            "s": _m(0.66, (0.58, 0.74), 0.60, (0.50, 0.69), 0.95, (0.92, 0.97))}
    assert t2s.sop_check(CHECK_SPEC, runs)["tiers"]["hard"]["verdict"] == "fail"
    runs["w"] = _m(0.22, (0.16, 0.29), 0.19, (0.13, 0.27), 0.90, (0.86, 0.94), api=0.10)
    assert t2s.sop_check(CHECK_SPEC, runs)["tiers"]["hard"]["verdict"] == "incomplete"


def test_sop_check_separation_needs_disjoint_intervals_in_the_expected_direction():
    runs = {**PRIORS, "w": _m(0.40, (0.30, 0.50), 0.4, (0.3, 0.5), 0.9, (0.86, 0.94)),
            "s": _m(0.55, (0.45, 0.65), 0.5, (0.4, 0.6), 0.95, (0.92, 0.97))}
    assert not t2s.sop_check(CHECK_SPEC, runs)["tiers"]["hard"]["separation"]["pass"]
    runs["w"], runs["s"] = runs["s"], runs["w"]
    assert not t2s.sop_check(CHECK_SPEC, runs)["tiers"]["hard"]["separation"]["pass"]


@needs_spec
def test_the_spec_says_when_the_priors_were_set():
    """Review of PR #109: SELECT 0 came in with the frozen spec, after the dry runs were scored and the vote was
    taken, and was not voted on; the spec must not say the priors were written before any answer was scored."""
    spec = yaml.safe_load(SPEC_FILE.read_text(encoding="utf-8"))
    text = " ".join([spec["baselines"]["rule"], *spec["deviations"]])
    assert "before any answer" not in text
    assert "after the dry runs" in spec["baselines"]["rule"] and "not voted" in spec["baselines"]["rule"]


def test_paired_difference_resamples_questions():
    a = {"hard/q1-ko": 1.0, "hard/q1-en": 1.0, "hard/q2-ko": 0.0, "hard/q2-en": 0.0}
    b = {"hard/q1-ko": 0.0, "hard/q1-en": 0.0, "hard/q2-ko": 0.0, "hard/q2-en": 0.0}
    d = t2s.paired_difference(a, b)
    assert d["value"] == 0.5 and d["n_groups"] == 2 and 0.0 <= d["ci_low"] <= d["ci_high"] <= 1.0


# ---- commands ----

class CliEnv:
    def __init__(self, tmp_path, monkeypatch, fx: Fixture):
        self.fx = fx
        self.db = tmp_path / "medbench.db"
        reg = fx.write_registry()
        monkeypatch.setenv("BENCHTOWARD_REGISTRY_DIR", str(reg))
        monkeypatch.setenv("BENCHTOWARD_DATA_DIR", str(fx.data_dir))
        monkeypatch.setenv("BENCHTOWARD_DB_URL", f"sqlite:///{self.db}")
        from benchtoward.cli import app
        self.app, self.runner = app, CliRunner()

    def invoke(self, *args):
        return self.runner.invoke(self.app, list(args))


def test_commands_record_the_priors_and_the_references_and_print_the_check(tmp_path, monkeypatch, fx):
    env = CliEnv(tmp_path, monkeypatch, fx)
    assert env.invoke("init").exit_code == 0
    out = env.invoke("baseline", "synthetic-snuh-text2sql")
    assert out.exit_code == 0, out.stdout
    assert "benchtoward/prior" in out.stdout and "benchtoward/text2sql-constant-query" in out.stdout
    tiers = fx.tiers()
    served = {m["slug"]: m.get("provider_model_id") or m["slug"]
              for m in yaml.safe_load((REGISTRY_DIR / "models.yaml").read_text(encoding="utf-8"))}
    for slug in ("openai/gpt-oss-20b", "deepseek/deepseek-v4.1-flash-552b-a16b"):
        model_id, name = served[slug], slug.replace("/", "__")
        hard = _cache(fx, tiers, model_id, "hard", tmp_path / "dry" / "hard" / name)
        easy = _cache(fx, tiers, model_id, "easy", tmp_path / "dry" / "easy" / name)
        out = env.invoke("eval-text2sql", "--model", slug, "--reuse", f"hard={hard}", "--reuse", f"easy={easy}")
        assert out.exit_code == 0, out.stdout
        assert "8 of 8 answers from the dry-run cache" in plain(out.stdout)
    report = tmp_path / "check.json"
    out = env.invoke("check-text2sql", "--json", str(report))
    assert out.exit_code == 0, out.stdout
    text = plain(out.stdout)
    assert "hard" in text and "easy" in text and "headroom" in text.lower() and "separation" in text.lower()
    verdicts = [line for line in out.stdout.splitlines() if "verdict:" in line]
    assert len(verdicts) == 2 and all("beats the prior untested" in line for line in verdicts)
    assert set(json.loads(report.read_text())["tiers"]) == {"hard", "easy"}


def test_eval_refuses_an_unknown_tier_in_reuse(tmp_path, monkeypatch, fx):
    env = CliEnv(tmp_path, monkeypatch, fx)
    env.invoke("init")
    out = env.invoke("eval-text2sql", "--model", "openai/gpt-oss-20b", "--reuse", f"medium={tmp_path}")
    assert out.exit_code != 0 and "medium" in plain(out.output)
