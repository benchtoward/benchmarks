import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from conftest import make_omop_db
from benchtoward.cli import app
from benchtoward.llm import ChatResult
from benchtoward.tasks import text2sql_cli
from benchtoward.tasks.synthetic_snuh import db_path
from benchtoward.tasks.text2sql import load_questions

QUESTIONS = [
    {"id": "t-001", "question_ko": "여성 환자는 몇 명인가?", "question_en": "How many female patients are there?",
     "category": "cohort_count", "difficulty": "easy", "scope": "group",
     "sql": "SELECT count(*) FROM person WHERE gender_concept_id = 8532"},
    {"id": "t-002", "question_ko": "진료과별 방문 수를 구하라.", "question_en": "Number of visits per department.",
     "category": "aggregation", "difficulty": "medium", "scope": "none",
     "sql": "SELECT visit_source_value, count(*) FROM visit_occurrence GROUP BY 1"},
]


class Env:
    def __init__(self, tmp_path: Path, monkeypatch):
        self.data_dir = tmp_path / "data"
        self.medbench_db = tmp_path / "medbench.db"
        monkeypatch.setenv("BENCHTOWARD_DATA_DIR", str(self.data_dir))
        monkeypatch.setenv("BENCHTOWARD_DB_URL", f"sqlite:///{self.medbench_db}")
        make_omop_db(db_path(self.data_dir))
        self.draft = self.write_draft(tmp_path / "draft.yaml", QUESTIONS)
        self.runner = CliRunner()

    @staticmethod
    def write_draft(path: Path, questions: list[dict]) -> Path:
        doc = {"status": "DRAFT - awaiting owner review, not published", "dataset": "synthetic-snuh@4.0.0",
               "questions": questions}
        path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return path

    def invoke(self, *args: str):
        return self.runner.invoke(app, ["text2sql", *args])


@pytest.fixture
def env(tmp_path, monkeypatch):
    return Env(tmp_path, monkeypatch)


def test_check_reports_rows_per_category(env):
    result = env.invoke("check", "--draft", str(env.draft))
    assert result.exit_code == 0, result.stdout
    assert "2 questions, 0 problems" in result.stdout
    assert "cohort_count" in result.stdout and "aggregation" in result.stdout
    assert not env.medbench_db.exists()


def test_check_fails_on_a_broken_reference(env, tmp_path):
    draft = env.write_draft(tmp_path / "bad.yaml", [*QUESTIONS, {**QUESTIONS[0], "id": "t-003",
                                                               "sql": "SELECT count(*) FROM person WHERE 1 = 0"}])
    result = env.invoke("check", "--draft", str(draft))
    assert result.exit_code == 1
    assert "t-003" in result.stdout and "trivial result" in result.stdout


def test_review_sheet_shows_both_languages_sql_and_results(env, tmp_path):
    out = tmp_path / "review.md"
    result = env.invoke("review", "--draft", str(env.draft), "--out", str(out))
    assert result.exit_code == 0, result.stdout
    sheet = out.read_text(encoding="utf-8")
    assert "## t-001" in sheet and "여성 환자는 몇 명인가?" in sheet and "How many female patients" in sheet
    assert "SELECT count(*) FROM person WHERE gender_concept_id = 8532" in sheet
    assert "| 2 |" in sheet  # the reference result
    assert "review: pending" in sheet


def test_prompt_prints_rendered_messages(env):
    result = env.invoke("prompt", "t-001", "--lang", "ko", "--draft", str(env.draft))
    assert result.exit_code == 0, result.stdout
    assert "CREATE TABLE person" in result.stdout and "Question: 여성 환자는 몇 명인가?" in result.stdout


def test_dry_run_scores_a_predictions_file_without_touching_the_medbench_db(env, tmp_path):
    preds = tmp_path / "preds.jsonl"
    preds.write_text("\n".join(json.dumps(r) for r in [
        {"instance_id": "t-001-en", "sql": "SELECT count(*) FROM person p JOIN concept c "
                                           "ON c.concept_id = p.gender_concept_id WHERE c.concept_name = 'FEMALE'"},
        {"instance_id": "t-002-en", "sql": "SELECT count(*), visit_source_value FROM visit_occurrence GROUP BY 2"},
        {"instance_id": "t-001-ko", "sql": "SELECT 1"},
    ]))
    out = tmp_path / "out"
    result = env.invoke("dry-run", "--predictions", str(preds), "--draft", str(env.draft), "--out", str(out))
    assert result.exit_code == 0, result.stdout
    assert "execution accuracy 0.667" in result.stdout
    assert json.loads((out / "summary.json").read_text())["execution_accuracy"]["n"] == 3
    assert not env.medbench_db.exists()


def test_dry_run_with_a_model_uses_the_registry_and_resumes(env, tmp_path, monkeypatch):
    calls = []

    class Fake:
        async def chat(self, model_id, messages, *, max_tokens, temperature=0.0):
            calls.append((model_id, max_tokens))
            question = messages[-1]["content"].rsplit("Question: ", 1)[1]
            sql = "SELECT count(*) FROM person WHERE gender_concept_id = 8532" \
                if "female" in question or "여성" in question else "SELECT 1"
            return ChatResult(content=f"```sql\n{sql}\n```", reasoning="thinking", tokens_in=100, tokens_out=20,
                              error=None)

    seen = {}

    def fake_client_for(model, settings):
        seen["model"] = model
        return Fake()

    monkeypatch.setattr(text2sql_cli, "client_for", fake_client_for)
    registry = tmp_path / "registry"
    registry.mkdir()
    (registry / "models.yaml").write_text("- {slug: example/sql-model, name: SQL model, type: llm, provider: example, "
                                          "provider_model_id: sql-model-7b, developer: Example}\n")
    monkeypatch.setenv("BENCHTOWARD_REGISTRY_DIR", str(registry))
    out = tmp_path / "dry"
    args = ("dry-run", "--model", "example/sql-model", "--draft", str(env.draft), "--out", str(out))
    result = env.invoke(*args)
    assert result.exit_code == 0, result.stdout
    assert seen["model"].provider == "example" and seen["model"].provider_model_id == "sql-model-7b"
    assert len(calls) == 4 and calls[0][0] == "sql-model-7b"
    assert "execution accuracy 0.500" in result.stdout and "by language: ko 0.500 (n=2), en 0.500 (n=2)" in result.stdout
    assert env.invoke(*args).exit_code == 0 and len(calls) == 4  # finished answers are not requested again
    assert not env.medbench_db.exists()


def test_dry_run_can_raise_the_output_limit_for_a_diagnostic_run(env, tmp_path, monkeypatch):
    seen = []

    class Fake:
        async def chat(self, model_id, messages, *, max_tokens, temperature=0.0):
            seen.append(max_tokens)
            return ChatResult(content="```sql\nSELECT 1\n```", reasoning=None, tokens_in=1, tokens_out=1, error=None)

    monkeypatch.setattr(text2sql_cli, "client_for", lambda model, settings: Fake())
    registry = tmp_path / "registry"
    registry.mkdir()
    (registry / "models.yaml").write_text("- {slug: example/sql-model, name: SQL model, type: llm, provider: example, "
                                          "provider_model_id: sql-model-7b, developer: Example}\n")
    monkeypatch.setenv("BENCHTOWARD_REGISTRY_DIR", str(registry))
    result = env.invoke("dry-run", "--model", "example/sql-model", "--draft", str(env.draft), "--max-tokens", "32768",
                        "--out", str(tmp_path / "dry"))
    assert result.exit_code == 0, result.stdout
    assert set(seen) == {32768}


def test_dry_run_rejects_unknown_model(env):
    result = env.invoke("dry-run", "--model", "nobody/none", "--draft", str(env.draft))
    assert result.exit_code != 0 and "unknown model" in result.stdout


def test_missing_database_explains_how_to_fetch(env, tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHTOWARD_DATA_DIR", str(tmp_path / "empty"))
    result = env.invoke("check", "--draft", str(env.draft))
    assert result.exit_code == 1 and "benchtoward text2sql fetch" in result.stdout


def test_dry_run_leaves_out_rejected_items(env, tmp_path):
    draft = env.write_draft(tmp_path / "rejected.yaml", [QUESTIONS[0], {**QUESTIONS[1], "review": "rejected"}])
    preds = tmp_path / "preds.jsonl"
    preds.write_text("\n".join(json.dumps({"instance_id": f"t-001-{lang}", "sql": QUESTIONS[0]["sql"]})
                               for lang in ("ko", "en")))
    out = tmp_path / "out"
    result = env.invoke("dry-run", "--predictions", str(preds), "--draft", str(draft), "--out", str(out))
    assert result.exit_code == 0, result.stdout
    assert "1 rejected item(s) left out" in result.stdout and "execution accuracy 1.000" in result.stdout
    assert json.loads((out / "summary.json").read_text())["execution_accuracy"]["n"] == 2


FIX_DRAFT = """status: "DRAFT - not published"
open_decisions:
  - Unanswerable questions and abstention scoring (EHRSQL reliability score RS(c)); not in v0.
  - Relative time questions ("in the last 6 months") would need a fixed "now"; v0 uses absolute dates only.

questions:

  - id: t-001
    category: cohort_count
    difficulty: easy
    scope: group
    question_ko: 여성 환자는 몇 명인가?
    question_en: How many female patients are there?
    sql: |
      SELECT count(*) FROM person WHERE gender_concept_id = 8532
    review: approved

  - id: t-002
    category: aggregation
    difficulty: medium
    scope: none
    question_ko: 가장 많이 방문한 진료과는?
    question_en: Which department has the most visits?
    sql: |
      SELECT visit_source_value FROM visit_occurrence GROUP BY 1 ORDER BY count(*) DESC, 1 LIMIT 1
    review: needs_edit
    review_note: "[gpt-6-astra] needs_edit: ties at the top"
"""


FRONTIER = ["anthropic/claude-opus-5.5", "openai/gpt-6-astra", "google/gemini-4-argon"]


def _text(message: dict) -> str:
    """A message's text; requests to Anthropic models carry it as parts with a prompt-cache mark."""
    content = message["content"]
    return content if isinstance(content, str) else "".join(part["text"] for part in content)


class Frontier:
    """Fake frontier models answering proposals, reviews and votes by their prompt."""

    def __init__(self):
        self.calls, self.models = [], []

    def client_for(self, model, settings):
        fake = self
        self.models.append(model)

        class Client:
            async def chat(self, model_id, messages, *, max_tokens, temperature=0.0, extra=None, before_attempt=None):
                user = _text(messages[-1])
                fake.calls.append((model_id, extra))
                if "Decision key: " in user:
                    key = user.split("Decision key: ", 1)[1].split("\n", 1)[0]
                    body = {"choice": {"unanswerable": "later_set", "relative_time": "absolute_only"}[key],
                            "reason": "votes"}
                elif "Revise this item" in user:
                    body = {"question_ko": "방문 수가 가장 많은 진료과(동률이면 모두)는?",
                            "question_en": "Which department(s) have the most visits (all ties)?",
                            "sql": "WITH n AS (SELECT visit_source_value AS d, count(*) AS k FROM visit_occurrence "
                                   "GROUP BY 1) SELECT d FROM n WHERE k = (SELECT max(k) FROM n)",
                            "order_matters": False, "change": "all ties"}
                else:
                    body = {"verdict": "approve", "reason": "fine", "fix": ""}
                return ChatResult(content=json.dumps(body), reasoning=None, tokens_in=100, tokens_out=10, error=None,
                                  cost_usd=0.001)
        return Client()


# The frontier models' route as registry/providers.yaml declares it (a private file of each deployment): marked
# `frontier: true`, with the public price list of its models and the request fields that make it report its bill.
ROUTES = {"local": {"base_url": "http://127.0.0.1:8000/v1", "key_env": "LOCAL_KEY"},
          "frontier-api": {"base_url": "https://llm.example.org/v1", "key_env": "EXAMPLE_KEY", "frontier": True,
                           "prices_url": "https://llm.example.org/v1/models",
                           "request": {"billing": {"usage": {"include": True}}}}}


def fake_prices(ids, url):
    assert url == "https://llm.example.org/v1/models"
    return {m: (10.0, 50.0) for m in ids}


@pytest.fixture
def frontier(env, monkeypatch):
    fake = Frontier()
    monkeypatch.setattr(text2sql_cli, "client_for", fake.client_for)
    monkeypatch.setattr(text2sql_cli, "provider_routes", lambda: ROUTES)
    monkeypatch.setattr(text2sql_cli, "_catalog_prices", fake_prices)
    return fake


def test_llm_fix_revises_needs_edit_items_into_the_draft_within_the_budget(env, frontier, tmp_path):
    draft = tmp_path / "fix.yaml"
    draft.write_text(FIX_DRAFT, encoding="utf-8")
    result = env.invoke("llm-fix", "--draft", str(draft), "--budget-usd", "5")
    assert result.exit_code == 0, result.stdout
    assert "t-002: approved in round 1 (claude-opus-5.5)" in result.stdout and "approved 1" in result.stdout
    assert "route-billed $0.004" in result.stdout and "cap $5.00" in result.stdout
    _, items = load_questions(draft)
    assert items[1].review == "approved" and "all ties" in items[1].question_en
    assert all(extra == {"usage": {"include": True}} for _, extra in frontier.calls)   # the route reports its bill
    assert {m.provider for m in frontier.models} == {"frontier-api"}                    # the route marked frontier
    job = env.data_dir / "drafts" / "text2sql-model-fixes"
    assert (job / "fix" / "anthropic__claude-opus-5.5.jsonl").exists() and (job / "fix-report-fix.json").exists()
    assert not env.medbench_db.exists()


def test_the_frontier_route_is_the_one_providers_yaml_marks(tmp_path, monkeypatch):
    """No deployment's route name is in the code: the frontier jobs default to the route that registry/providers.yaml
    marks `frontier: true`; `--route` names another. A route without `prices_url` prices nothing (the budget then
    counts only what the route bills), and one without `request.billing` sends no billing fields."""
    import typer

    registry = tmp_path / "registry"
    registry.mkdir()
    (registry / "providers.yaml").write_text(yaml.safe_dump(ROUTES), encoding="utf-8")
    monkeypatch.setenv("BENCHTOWARD_REGISTRY_DIR", str(registry))
    monkeypatch.setenv("EXAMPLE_KEY", "k")
    monkeypatch.setenv("LOCAL_KEY", "k")
    monkeypatch.setattr(text2sql_cli, "_catalog_prices", fake_prices)
    assert text2sql_cli.frontier_route(None) == ("frontier-api", ROUTES["frontier-api"])
    assert text2sql_cli.frontier_route("local") == ("local", ROUTES["local"])
    ids, clients, prices, extra = text2sql_cli._frontier("a/one,b/two", None)
    assert ids == ["a/one", "b/two"] and prices == {"a/one": (10.0, 50.0), "b/two": (10.0, 50.0)}
    assert extra == {"usage": {"include": True}} and set(clients) == set(ids)
    assert text2sql_cli._frontier("a/one", "local")[2:] == ({"a/one": (0.0, 0.0)}, None)
    for routes in ({"local": ROUTES["local"]}, {"x": ROUTES["frontier-api"], "y": ROUTES["frontier-api"]}):
        (registry / "providers.yaml").write_text(yaml.safe_dump(routes), encoding="utf-8")
        with pytest.raises(typer.Exit):
            text2sql_cli.frontier_route(None)
    with pytest.raises(typer.Exit):
        text2sql_cli.frontier_route("nowhere")


def test_llm_fix_gives_the_note_as_the_reason_when_the_reviewers_had_approved(env, frontier, tmp_path, monkeypatch):
    """An item set back to needs_edit by a decision (e.g. rewording 내원) was approved in the review: the reason
    for the edit is the item's review_note, not the earlier approvals."""
    draft = tmp_path / "fix.yaml"
    draft.write_text(FIX_DRAFT.replace("[gpt-6-astra] needs_edit: ties at the top", "Decision naewon: reword"),
                     encoding="utf-8")
    cache = env.data_dir / "drafts" / "text2sql-llm-review"
    cache.mkdir(parents=True)
    for m in FRONTIER:
        (cache / f"{m.replace('/', '__')}.jsonl").write_text(json.dumps(
            {"id": "t-002", "review": {"verdict": "approve", "reason": "looked fine", "fix": ""}}) + "\n")
    prompts = []
    original = text2sql_cli.client_for

    def spy(model, settings):
        client = original(model, settings)
        chat = client.chat

        async def recording(model_id, messages, **kw):
            prompts.append(_text(messages[-1]))
            return await chat(model_id, messages, **kw)
        client.chat = recording
        return client
    monkeypatch.setattr(text2sql_cli, "client_for", spy)
    assert env.invoke("llm-fix", "--draft", str(draft)).exit_code == 0
    assert "Decision naewon: reword" in prompts[0] and "looked fine" not in prompts[0]


def test_llm_fix_ignores_reviews_of_an_earlier_version_of_the_item(env, frontier, tmp_path, monkeypatch):
    """An item revised after its review and later sent back by a decision (e.g. 내원 in a fixed item): the old
    reviewers' reasons were about the earlier wording, so the reason is the review note."""
    from benchtoward.tasks.synthetic_snuh import connect_readonly
    from benchtoward.tasks.text2sql import schema_summary
    from benchtoward.tasks.text2sql_llm_review import _sha, render_review_messages
    draft = tmp_path / "fix.yaml"
    draft.write_text(FIX_DRAFT.replace("[gpt-6-astra] needs_edit: ties at the top", "Decision naewon: reword"),
                     encoding="utf-8")
    _, items = load_questions(draft)
    con = connect_readonly(db_path(env.data_dir))
    current = _sha(render_review_messages(items[1], con, schema_summary(con)))
    cache = env.data_dir / "drafts" / "text2sql-llm-review"
    cache.mkdir(parents=True)
    for m in FRONTIER:
        (cache / f"{m.replace('/', '__')}.jsonl").write_text(json.dumps(
            {"id": "t-002", "prompt_sha": "an-older-wording", "review": {"verdict": "needs_edit",
                                                                          "reason": "stale tie complaint"}}) + "\n")
    prompts = []
    original = text2sql_cli.client_for

    def spy(model, settings):
        client = original(model, settings)
        chat = client.chat

        async def recording(model_id, messages, **kw):
            prompts.append(_text(messages[-1]))
            return await chat(model_id, messages, **kw)
        client.chat = recording
        return client
    monkeypatch.setattr(text2sql_cli, "client_for", spy)
    assert env.invoke("llm-fix", "--draft", str(draft)).exit_code == 0
    assert "Decision naewon: reword" in prompts[0] and "stale tie complaint" not in prompts[0]
    for m in FRONTIER:   # reviews of the current wording are still the reasons
        (cache / f"{m.replace('/', '__')}.jsonl").write_text(json.dumps(
            {"id": "t-002", "prompt_sha": current, "review": {"verdict": "needs_edit", "reason": "current tie"}}) + "\n")
    draft.write_text(FIX_DRAFT, encoding="utf-8")
    prompts.clear()
    assert env.invoke("llm-fix", "--draft", str(draft), "--out", str(tmp_path / "job2")).exit_code == 0
    assert "current tie" in prompts[0]


def test_llm_fix_refuses_to_start_when_the_job_has_spent_its_cap(env, frontier, tmp_path):
    draft = tmp_path / "fix.yaml"
    draft.write_text(FIX_DRAFT, encoding="utf-8")
    job = env.data_dir / "drafts" / "text2sql-model-fixes" / "decide"
    job.mkdir(parents=True)
    (job / "x.jsonl").write_text(json.dumps({"cost_usd": 5.0}) + "\n")
    result = env.invoke("llm-fix", "--draft", str(draft), "--budget-usd", "5")
    assert "stopped at the budget" in result.stdout and not frontier.calls
    assert load_questions(draft)[1][1].review == "needs_edit"


def test_llm_decide_records_the_votes_in_the_draft(env, frontier, tmp_path):
    draft = tmp_path / "decide.yaml"
    draft.write_text(FIX_DRAFT, encoding="utf-8")
    result = env.invoke("llm-decide", "--draft", str(draft), "--budget-usd", "5")
    assert result.exit_code == 0, result.stdout
    assert "unanswerable: later_set (3 of 3; next work)" in result.stdout
    assert "relative_time: absolute_only (3 of 3; v0 unchanged)" in result.stdout
    meta, items = load_questions(draft)
    assert meta["open_decisions"] == [] and [d["key"] for d in meta["decisions"]] == ["unanswerable", "relative_time"]
    assert len(items) == 2 and len(frontier.calls) == 6


# ---- sending items back, and the harder set ----

def test_send_back_marks_items_needs_edit_with_the_reason(env, tmp_path):
    draft = tmp_path / "fix.yaml"
    draft.write_text(FIX_DRAFT, encoding="utf-8")
    result = env.invoke("send-back", "--draft", str(draft), "--id", "t-001", "--reason", "Decision naewon: reword 내원")
    assert result.exit_code == 0, result.stdout
    _, items = load_questions(draft)
    raw = yaml.safe_load(draft.read_text(encoding="utf-8"))["questions"]
    assert items[0].review == "needs_edit" and raw[0]["review_note"] == "Decision naewon: reword 내원"
    assert items[1].review == "needs_edit" and raw[1]["review_note"] == "[gpt-6-astra] needs_edit: ties at the top"
    assert env.invoke("send-back", "--draft", str(draft), "--id", "t-404", "--reason", "x").exit_code != 0


HARD_DRAFT = """# Harder set (test)
status: "DRAFT - not published"
prompt:
  id: text2sql-omop-hard-p0
questions: []
"""


class HardFrontier:
    """Fake frontier models for the harder set: a few valid items per batch, reviews that approve all but one."""

    def __init__(self):
        self.calls = []

    def client_for(self, model, settings):
        fake = self

        class Client:
            async def chat(self, model_id, messages, *, max_tokens, temperature=0.0, extra=None, before_attempt=None):
                user = _text(messages[-1])
                fake.calls.append(model_id)
                if "new items of category" in user:
                    category = user.rsplit('of category "', 1)[1].split('"', 1)[0]
                    if category == "unanswerable":
                        rows = [{"question_ko": "각 방문의 진료비는?", "question_en": "What did each visit cost?",
                                 "sql": None, "order_matters": False, "why_hard": "no cost table"}]
                    else:
                        rows = [{"question_ko": f"{category} {model_id} 질문", "question_en": f"{category} question",
                                 "sql": f"SELECT count(*) FROM visit_occurrence WHERE visit_source_value <> "
                                        f"'{category}-{model_id}'", "order_matters": False, "why_hard": "joins"}]
                    body = {"items": rows}
                elif "Revise this item" in user:
                    body = {"question_ko": "2024년 입원의 진료비 총액은?", "question_en": "Total bill of 2024 stays?",
                            "sql": None, "order_matters": False, "why_hard": "no cost table", "change": "scoped"}
                else:
                    verdict = "needs_edit" if "cost" in user and "Total bill" not in user and model_id.startswith(
                        "openai") else "approve"
                    body = {"verdict": verdict, "reason": "which visits?" if verdict != "approve" else "fine",
                            "fix": ""}
                return ChatResult(content=json.dumps(body), reasoning=None, tokens_in=100, tokens_out=10,
                                  error=None, cost_usd=0.001)
        return Client()


@pytest.fixture
def hard(env, monkeypatch, tmp_path):
    from test_text2sql_hard import make_hard_db
    path = db_path(env.data_dir)
    path.unlink()
    make_hard_db(path)
    fake = HardFrontier()
    monkeypatch.setattr(text2sql_cli, "client_for", fake.client_for)
    monkeypatch.setattr(text2sql_cli, "provider_routes", lambda: ROUTES)
    monkeypatch.setattr(text2sql_cli, "_catalog_prices", fake_prices)
    draft = tmp_path / "hard.yaml"
    draft.write_text(HARD_DRAFT, encoding="utf-8")
    return fake, draft


def test_the_harder_set_is_generated_reviewed_and_fixed_by_the_frontier_models(env, hard):
    fake, draft = hard
    plan = "clinical_terms=1,1;unanswerable=1"
    result = env.invoke("hard-generate", "--draft", str(draft), "--plan", plan, "--budget-usd", "5")
    assert result.exit_code == 0, result.stdout
    assert "3 items written" in result.stdout and "cap $5.00" in result.stdout
    _, items = load_questions(draft)
    assert [i.category for i in items] == ["clinical_terms", "clinical_terms", "unanswerable"]
    assert {i.review for i in items} == {"pending"} and items[2].sql is None
    calls = len(fake.calls)
    again = env.invoke("hard-generate", "--draft", str(draft), "--plan", plan, "--budget-usd", "5")
    assert again.exit_code == 0 and "0 items written" in again.stdout and len(fake.calls) == calls   # cached

    result = env.invoke("hard-review", "--draft", str(draft), "--budget-usd", "5")
    assert result.exit_code == 0, result.stdout
    assert "approved 2, needs_edit 1" in result.stdout
    assert [i.review for i in load_questions(draft)[1]] == ["approved", "approved", "needs_edit"]

    result = env.invoke("llm-fix", "--draft", str(draft), "--budget-usd", "5")
    assert result.exit_code == 0, result.stdout
    assert "syn-hard-003: approved in round 1" in result.stdout
    fixed = load_questions(draft)[1][2]
    assert fixed.review == "approved" and fixed.sql is None and fixed.question_en == "Total bill of 2024 stays?"
    job = env.data_dir / "drafts" / "text2sql-hard"
    assert {p.name for p in job.iterdir()} >= {"generate", "hard-review", "fix", "fix-report-hard.json"}

    result = env.invoke("spend")
    assert result.exit_code == 0 and "route-billed $0.0" in result.stdout and "generate" in result.stdout
    assert not env.medbench_db.exists()


def test_a_hard_dry_run_reports_the_reliability_score(env, hard, tmp_path):
    _, draft = hard
    assert env.invoke("hard-generate", "--draft", str(draft), "--plan", "clinical_terms=1;unanswerable=1",
                      "--budget-usd", "5").exit_code == 0
    preds = tmp_path / "preds.jsonl"
    preds.write_text("\n".join(json.dumps(r) for r in [
        {"instance_id": "syn-hard-001-en", "sql": "SELECT count(*) FROM visit_occurrence"},
        {"instance_id": "syn-hard-001-ko", "sql": "ABSTAIN"},
        {"instance_id": "syn-hard-002-en", "sql": "ABSTAIN"},
        {"instance_id": "syn-hard-002-ko", "sql": "SELECT 1"}]))
    out = tmp_path / "out"
    result = env.invoke("dry-run", "--predictions", str(preds), "--draft", str(draft), "--out", str(out))
    assert result.exit_code == 0, result.stdout
    assert "execution accuracy on answerable items 0.500" in result.stdout
    assert "RS(0) 0.500" in result.stdout and "RS(5) -0.750" in result.stdout     # +1, 0, +1, -c
    check = env.invoke("check", "--draft", str(draft))
    assert check.exit_code == 0 and "2 questions, 0 problems" in check.stdout and "unanswerable" in check.stdout


def test_a_hard_predictions_file_abstains_with_null_or_abstain_and_an_empty_string_is_no_answer(env, hard, tmp_path):
    _, draft = hard
    assert env.invoke("hard-generate", "--draft", str(draft), "--plan", "clinical_terms=1;unanswerable=1",
                      "--budget-usd", "5").exit_code == 0
    preds = tmp_path / "preds.json"                    # 001 answerable, 002 unanswerable
    preds.write_text(json.dumps({"syn-hard-001-en": "", "syn-hard-001-ko": " abstain ",
                                 "syn-hard-002-en": None, "syn-hard-002-ko": "null"}))
    out = tmp_path / "out"
    result = env.invoke("dry-run", "--predictions", str(preds), "--draft", str(draft), "--out", str(out))
    assert result.exit_code == 0, result.stdout
    rows = {r["instance_id"]: r for r in map(json.loads, (out / "items.jsonl").read_text().splitlines())}
    assert rows["syn-hard-001-en"]["error"] == "no_sql" and rows["syn-hard-001-en"]["rs"] == -1     # empty: -c
    assert rows["syn-hard-001-ko"]["abstained"] and rows["syn-hard-001-ko"]["rs"] == 0
    assert rows["syn-hard-002-en"]["abstained"] and rows["syn-hard-002-ko"]["abstained"]
    assert "RS(0) 0.500" in result.stdout                                                         # (-1 + 0 + 1 + 1) / 4


def test_check_applies_the_quoted_concept_rule_to_the_harder_set_only(env, hard, tmp_path):
    _, draft = hard
    item = {"id": "syn-hard-001", "category": "clinical_terms", "difficulty": "hard", "scope": "group",
            "question_ko": "당화혈색소(Hemoglobin A1c/Hemoglobin.total in Blood) 결과는 몇 건인가?",
            "question_en": "How many HbA1c results are there?",
            "sql": "SELECT count(*) FROM measurement WHERE measurement_concept_id = 3004410", "review": "approved"}
    doc = {"status": "DRAFT", "prompt": {"id": "text2sql-omop-hard-p0"}, "questions": [item]}
    draft.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    result = env.invoke("check", "--draft", str(draft))
    assert result.exit_code == 1 and "PROBLEM syn-hard-001" in result.stdout
    assert "quotes concept names (Hemoglobin A1c/Hemoglobin.total in Blood)" in result.stdout
    v0 = env.write_draft(tmp_path / "v0.yaml", [{k: v for k, v in item.items() if k != "review"} |
                                                {"category": "cohort_count", "difficulty": "easy"}])
    assert env.invoke("check", "--draft", str(v0)).exit_code == 0          # v0 quotes concept names on purpose
    item["question_ko"] = "당화혈색소 결과는 몇 건인가?"
    draft.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    assert "1 questions, 0 problems" in env.invoke("check", "--draft", str(draft)).stdout


def test_a_fixed_hard_item_sent_back_later_is_fixed_for_the_new_reason_not_its_first_review(env, hard, monkeypatch):
    """syn-hard-038 was revised by llm-fix after its first review and later sent back by the naewon decision: the
    first review was about an earlier wording, so the reason given to the proposer is the send-back note."""
    fake, draft = hard
    assert env.invoke("hard-generate", "--draft", str(draft), "--plan", "clinical_terms=1;unanswerable=1",
                      "--budget-usd", "5").exit_code == 0
    assert env.invoke("hard-review", "--draft", str(draft), "--budget-usd", "5").exit_code == 0
    prompts = []
    original = fake.client_for

    def spy(model, settings):
        client = original(model, settings)
        chat = client.chat

        async def recording(model_id, messages, **kw):
            prompts.append(_text(messages[-1]))
            return await chat(model_id, messages, **kw)
        client.chat = recording
        return client
    monkeypatch.setattr(text2sql_cli, "client_for", spy)
    assert env.invoke("llm-fix", "--draft", str(draft), "--budget-usd", "5").exit_code == 0
    assert "which visits?" in next(p for p in prompts if "Revise this item" in p)      # the first review's reason
    assert env.invoke("send-back", "--draft", str(draft), "--id", "syn-hard-002",
                      "--reason", "Decision naewon: reword 내원").exit_code == 0
    prompts.clear()
    result = env.invoke("llm-fix", "--draft", str(draft), "--budget-usd", "5")
    assert result.exit_code == 0, result.stdout
    propose = next(p for p in prompts if "Revise this item" in p)
    assert "Decision naewon: reword 내원" in propose and "which visits?" not in propose


def test_the_default_cap_follows_the_job_folder(env, frontier, tmp_path):
    """llm-fix and llm-decide default to the job's cap: $25 for the harder set's job, $30 for v0's."""
    draft = tmp_path / "fix.yaml"
    draft.write_text(FIX_DRAFT, encoding="utf-8")
    assert "cap $30.00" in env.invoke("llm-decide", "--draft", str(draft), "--no-apply").stdout
    assert "cap $30.00" in env.invoke("llm-fix", "--draft", str(draft), "--no-apply").stdout
    hard_job = env.data_dir / "drafts" / "text2sql-hard"
    assert "cap $25.00" in env.invoke("llm-fix", "--draft", str(draft), "--no-apply", "--out", str(hard_job)).stdout
    assert "cap $25.00" in env.invoke("llm-decide", "--draft", str(draft), "--no-apply", "--out",
                                      str(hard_job)).stdout
    assert "cap $7.00" in env.invoke("llm-fix", "--draft", str(draft), "--budget-usd", "7").stdout


def test_llm_fix_on_the_harder_set_defaults_to_its_job_and_cap(env, hard):
    _, draft = hard
    assert env.invoke("hard-generate", "--draft", str(draft), "--plan", "unanswerable=1").exit_code == 0
    assert env.invoke("hard-review", "--draft", str(draft)).exit_code == 0
    result = env.invoke("llm-fix", "--draft", str(draft))
    assert result.exit_code == 0, result.stdout
    assert "cap $25.00" in result.stdout and (env.data_dir / "drafts" / "text2sql-hard" / "fix").exists()


def test_llm_review_asks_the_route_for_its_bill_and_counts_the_folder_spend(env, frontier, tmp_path):
    out = tmp_path / "review-cache"
    out.mkdir()
    (out / "earlier.jsonl").write_text(json.dumps({"id": "x", "cost_usd": 4.99}) + "\n")
    book = tmp_path / "book.xlsx"
    args = ("llm-review", "--draft", str(env.draft), "--out", str(out), "--workbook", str(book))
    result = env.invoke(*args, "--budget-usd", "5")
    assert result.exit_code == 0, result.stdout
    assert "asked 0" in result.stdout and "stopped at the budget" in result.stdout and not frontier.calls
    result = env.invoke(*args, "--budget-usd", "50")
    assert "asked 6, parsed 6" in result.stdout and book.exists()
    assert all(extra == {"usage": {"include": True}} for _, extra in frontier.calls)


def test_llm_decide_on_the_harder_set_votes_on_its_questions_in_its_job_with_its_dry_runs(env, hard, monkeypatch,
                                                                                         tmp_path):
    from benchtoward.tasks.text2sql import HARD_CONVENTIONS
    _, draft = hard
    assert env.invoke("hard-generate", "--draft", str(draft), "--plan", "clinical_terms=1;unanswerable=1").exit_code == 0
    text = draft.read_text(encoding="utf-8").replace(
        "questions:\n", "open_decisions:\n  - Output limit. The free model thinks for long.\n"
                        "  - Second round. Little money is left.\n\nquestions:\n", 1)
    draft.write_text(text, encoding="utf-8")
    runs = env.data_dir / "drafts" / draft.stem
    for name, n in (("alibaba__qwen3.8-flash-next-180b-a6b", 1), ("alibaba__qwen3.8-flash-next-180b-a6b-max32k", 2)):
        (runs / name).mkdir(parents=True)
        preds = tmp_path / f"{name}.jsonl"
        preds.write_text("\n".join(json.dumps({"instance_id": f"syn-hard-001-{lang}",
                                               "sql": "SELECT count(*) FROM visit_occurrence" if k < n else ""})
                                   for k, lang in enumerate(("ko", "en"))))
        assert env.invoke("dry-run", "--predictions", str(preds), "--draft", str(draft),
                          "--out", str(runs / name)).exit_code == 0
    seen = []

    class Voter:
        async def chat(self, model_id, messages, *, max_tokens, temperature=0.0, extra=None, before_attempt=None):
            user = _text(messages[-1])
            seen.append(user)
            key = user.split("Decision key: ", 1)[1].split("\n", 1)[0]
            choice = {"output_limit": "limit_32k", "second_round": "now"}[key]
            return ChatResult(json.dumps({"choice": choice, "reason": "r"}), None, 100, 10, None, cost_usd=0.001)
    monkeypatch.setattr(text2sql_cli, "client_for", lambda model, settings: Voter())
    result = env.invoke("llm-decide", "--draft", str(draft), "--budget-usd", "25")
    assert result.exit_code == 0, result.stdout
    assert "output_limit: limit_32k (3 of 3; to implement)" in result.stdout
    assert "second_round: now (3 of 3; needs the owner's budget approval" in result.stdout
    assert HARD_CONVENTIONS in seen[0] and "16,384 output tokens" in seen[0] and "32,768 output tokens" in seen[0]
    assert "1 of 2 answerable answers have no SQL" in seen[0]
    assert (env.data_dir / "drafts" / "text2sql-hard" / "decide").exists()
    meta, _ = load_questions(draft)
    assert meta["open_decisions"] == [] and [d["key"] for d in meta["decisions"]] == ["output_limit", "second_round"]


def test_a_draft_that_names_a_headline_gets_it_first_in_the_dry_run(env, hard, tmp_path):
    _, draft = hard
    assert env.invoke("hard-generate", "--draft", str(draft), "--plan", "clinical_terms=1;unanswerable=1").exit_code == 0
    text = draft.read_text(encoding="utf-8").replace("prompt:\n", "protocol:\n  headline: RS(0)\nprompt:\n", 1)
    draft.write_text(text, encoding="utf-8")
    preds = tmp_path / "preds.jsonl"
    preds.write_text("\n".join(json.dumps(r) for r in [
        {"instance_id": "syn-hard-001-en", "sql": "SELECT count(*) FROM visit_occurrence"},
        {"instance_id": "syn-hard-001-ko", "sql": "ABSTAIN"},
        {"instance_id": "syn-hard-002-en", "sql": "ABSTAIN"},
        {"instance_id": "syn-hard-002-ko", "sql": "SELECT 1"}]))
    out = tmp_path / "out"
    result = env.invoke("dry-run", "--predictions", str(preds), "--draft", str(draft), "--out", str(out))
    assert result.exit_code == 0, result.stdout
    assert result.stdout.splitlines()[1] == "headline RS(0) 0.500 (the draft's protocol.headline)"
    assert json.loads((out / "summary.json").read_text())["headline"] == {"metric": "RS(0)", "value": 0.5}


def test_llm_decide_freeze_asks_once_and_records_the_vote_in_both_tiers(env, frontier, monkeypatch, tmp_path):
    from benchtoward.tasks.text2sql_llm_decide import FREEZE, SOP_CHECKS
    first = {d.key: d.options[0].id for d in FREEZE.table}
    shared = [d for d in FREEZE.table if d.key != "output_format"]

    def draft(name, prompt_id, decisions):
        entries = "".join(f"  - {d.match} Background of {d.key}.\n" for d in decisions)
        path = tmp_path / name
        path.write_text(FIX_DRAFT.replace("open_decisions:\n", f"prompt:\n  id: {prompt_id}\nopen_decisions:\n" + entries)
                        .split("  - Unanswerable")[0] + "\nquestions:" + FIX_DRAFT.split("questions:")[1],
                        encoding="utf-8")
        return path
    easy = draft("easy.yaml", "text2sql-omop-p0", shared)
    hard = draft("hard.yaml", "text2sql-omop-hard-p0", FREEZE.table)
    evidence, sop = tmp_path / "evidence.md", tmp_path / "sop.md"
    evidence.write_text("EVIDENCE: deepseek-v4.1-flash 0.596.\n", encoding="utf-8")
    sop.write_text(f"# SOP\n\n{SOP_CHECKS}\n\nReference roster per benchmark.\n\n### KHDP datasets\n\nnot this\n",
                   encoding="utf-8")
    seen = []

    class Voter:
        async def chat(self, model_id, messages, *, max_tokens, temperature=0.0, extra=None, before_attempt=None):
            user = _text(messages[-1])
            seen.append(user)
            key = user.split("Decision key: ", 1)[1].split("\n", 1)[0]
            return ChatResult(json.dumps({"choice": first[key], "reason": "r"}), None, 100, 10, None, cost_usd=0.001)
    monkeypatch.setattr(text2sql_cli, "client_for", lambda model, settings: Voter())
    job = env.data_dir / "drafts" / "text2sql-hard"
    common = ("--freeze", "--easy-draft", str(easy), "--hard-draft", str(hard), "--evidence", str(evidence),
              "--sop", str(sop), "--budget-usd", "25", "--out", str(job))
    result = env.invoke("llm-decide", "--draft", str(hard), *common)
    assert result.exit_code == 0, result.stdout
    assert f"output_format: {first['output_format']} (3 of 3;" in result.stdout and "cap $25.00" in result.stdout
    assert len(seen) == 9 and "EVIDENCE: deepseek-v4.1-flash 0.596." in seen[0]
    assert "Reference roster per benchmark." in seen[0] and "not this" not in seen[0]
    result = env.invoke("llm-decide", "--draft", str(easy), *common)
    assert result.exit_code == 0, result.stdout
    assert len(seen) == 9 and "route-billed $0.000 this run" in result.stdout     # served from the cache
    assert [d["key"] for d in load_questions(easy)[0]["decisions"]] == ["easy_tier_publication", "reference_roster"]
    assert [d["key"] for d in load_questions(hard)[0]["decisions"]] == [d.key for d in FREEZE.table]
