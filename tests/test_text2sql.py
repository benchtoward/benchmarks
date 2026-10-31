import asyncio
import datetime as dt
import json
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from benchtoward.llm import ChatResult
from benchtoward.tasks.synthetic_snuh import connect_readonly
from benchtoward.tasks.text2sql import (CATEGORIES, Question, canonical_value, check_questions,
                                     execute_sql, extract_sql, load_questions, render_messages, results_match,
                                     run_dry_run, schema_summary, score_predictions)

REPO = Path(__file__).resolve().parents[1]
DRAFT = REPO / "registry" / "drafts" / "synthetic-snuh-text2sql-v0.yaml"
HARD_DRAFT = REPO / "registry" / "drafts" / "synthetic-snuh-text2sql-hard-v0.yaml"
# The question files are released on KHDP, not with the code (tests/test_public_export.py), and the task's spec goes
# out only when the task is published; tests that read them skip in the public repository.
needs_drafts = pytest.mark.skipif(not (DRAFT.exists() and HARD_DRAFT.exists()),
                                  reason="the question files are released on KHDP, not with the code")
SPEC = REPO / "registry" / "tasks" / "synthetic-snuh-text2sql.yaml"
needs_spec = pytest.mark.skipif(not SPEC.exists(), reason="the task's spec stays private while the task is in review")


@pytest.fixture
def con(omop_db):
    return connect_readonly(omop_db)


def q(**kw) -> Question:
    base = dict(id="t-001", question_ko="질문", question_en="Question", category="cohort_count",
                difficulty="easy", sql="SELECT 1")
    return Question(**{**base, **kw})


# ---- canonical values and result matching (scorer rules) ----

def test_numbers_compare_by_value_across_types():
    assert canonical_value(3) == canonical_value(3.0) == canonical_value(Decimal("3.00"))
    assert canonical_value(True) == canonical_value(1)
    assert canonical_value(0.1 + 0.2) == canonical_value(0.3)  # rounded to 3 decimals (EHRSQL-2024)
    assert canonical_value(2.0001) == canonical_value(2)
    assert canonical_value(0.97) != canonical_value(0.9712)  # rounding to fewer places is the question's business
    assert canonical_value("3") == canonical_value(3) and canonical_value(" 0.50") == canonical_value(0.5)
    assert canonical_value("E11.9") == "E11.9" and canonical_value("4548-4") == "4548-4"
    assert canonical_value("nan") == "nan" and canonical_value("None") != canonical_value(None)
    assert canonical_value(float("nan")) is None


def test_dates_strings_and_nulls():
    assert canonical_value(dt.date(2021, 1, 1)) == canonical_value(dt.datetime(2021, 1, 1)) == "2021-01-01"
    assert canonical_value("2021-01-01 ") == "2021-01-01"
    assert canonical_value(dt.datetime(2021, 1, 1, 8, 30)) == "2021-01-01 08:30:00"
    assert canonical_value(dt.timedelta(days=3)) == canonical_value(3)
    assert canonical_value(None) is None and canonical_value(None) != canonical_value(0)
    assert canonical_value("MALE") != canonical_value("male")  # strings are case-sensitive


def test_row_order_ignored_unless_order_matters():
    ref = [("a", 1), ("b", 2)]
    assert results_match(ref, [("b", 2), ("a", 1)])
    assert not results_match(ref, [("b", 2), ("a", 1)], order_matters=True)
    assert results_match(ref, [("a", 1), ("b", 2)], order_matters=True)


def test_duplicates_count():
    assert not results_match([(1,), (2,)], [(1,), (1,), (2,)])
    assert not results_match([(1,), (1,), (2,)], [(1,), (2,)])


def test_column_order_and_names_ignored_but_column_count_matters():
    assert results_match([("FEMALE", 2), ("MALE", 2)], [(2, "FEMALE"), (2, "MALE")])
    assert results_match([("x", 1, 2.0)], [(2, "x", 1)])
    assert not results_match([("FEMALE", 2)], [("FEMALE", 2, 8532)])
    assert not results_match([(1, 2)], [(2, 1), (1, 2)])


def test_permutation_must_hold_for_every_row():
    # swapping the columns fixes row 1 but breaks row 2: not a match
    assert not results_match([(1, 2), (3, 4)], [(2, 1), (3, 4)])


def test_empty_and_null_results():
    assert results_match([], [])
    assert not results_match([(1,)], [])
    assert results_match([(None,)], [(None,)])
    assert not results_match([(None,)], [(0,)])


# ---- SQL extraction and safe execution ----

def test_extract_sql_from_code_fences_and_bare_text():
    assert extract_sql("Here:\n```sql\nSELECT 1;\n```") == "SELECT 1"
    assert extract_sql("```sql\nSELECT 1\n```\nbetter:\n```sql\nSELECT 2\n```") == "SELECT 2"
    assert extract_sql("```\nWITH a AS (SELECT 1) SELECT * FROM a\n```") == "WITH a AS (SELECT 1) SELECT * FROM a"
    assert extract_sql("SELECT count(*) FROM person;") == "SELECT count(*) FROM person"
    assert extract_sql("I think the answer is:\nSELECT count(*)\nFROM person") == "SELECT count(*)\nFROM person"
    assert extract_sql("I cannot answer this.") is None
    assert extract_sql("") is None


def test_execute_returns_rows_and_columns(con):
    result = execute_sql(con, "SELECT gender_concept_id, count(*) AS n FROM person GROUP BY 1 ORDER BY 1")
    assert result.error is None and result.columns == ["gender_concept_id", "n"]
    assert result.rows == [(8507, 2), (8532, 2)]


@pytest.mark.parametrize("sql,error", [
    ("DROP TABLE person", "not_select"),
    ("SELECT 1; SELECT 2", "multiple_statements"),
    ("PRAGMA version", "not_select"),
    ("SELEC 1", "syntax_error"),
    ("SELECT * FROM no_such_table", "execution_error"),
    ("SELECT * FROM read_csv('/etc/hosts')", "execution_error"),
    ("", "empty"),
])
def test_execute_rejects_unsafe_or_broken_sql(con, sql, error):
    assert execute_sql(con, sql).error == error


def test_execute_times_out(con):
    result = execute_sql(con, "SELECT count(*) FROM range(1000000000) a, range(1000000) b", timeout_s=0.5)
    assert result.error == "timeout"
    assert execute_sql(con, "SELECT 42").rows == [(42,)]  # the connection stays usable


def test_execute_caps_rows(con):
    assert execute_sql(con, "SELECT * FROM range(20)", max_rows=10).error == "too_many_rows"
    assert len(execute_sql(con, "SELECT * FROM range(10)", max_rows=10).rows) == 10


# ---- prompt ----

def test_schema_summary_lists_nonempty_tables_with_types_and_samples(con):
    text = schema_summary(con, sample_rows=2)
    assert "CREATE TABLE person (" in text and "year_of_birth INTEGER" in text
    assert "CREATE TABLE specimen" not in text and "specimen" in text  # empty tables are named, not described
    assert "SYN-000001" in text and "SYN-000003" not in text  # first 2 rows by the first column
    assert "Gender | Gender | 2" in text  # concept vocabulary overview, not the concept rows themselves


def test_render_messages_language_and_format(con):
    schema = schema_summary(con)
    item = q(question_ko="여성 환자는 몇 명인가?", question_en="How many female patients are there?")
    ko = render_messages(item, "ko", schema)
    en = render_messages(item, "en", schema)
    assert ko[0]["role"] == "system" and "DuckDB" in ko[0]["content"] and "OMOP CDM v5.4" in ko[0]["content"]
    assert "```sql" in ko[0]["content"]
    assert ko[1]["content"].endswith("Question: 여성 환자는 몇 명인가?")
    assert en[1]["content"].endswith("Question: How many female patients are there?")
    assert schema in en[1]["content"]
    with pytest.raises(ValueError):
        render_messages(item, "fr", schema)


# ---- question files and draft checks ----

def _write(tmp_path, items, meta=None) -> Path:
    path = tmp_path / "draft.yaml"
    path.write_text(yaml.safe_dump({"status": "DRAFT", **(meta or {}), "questions": items}, allow_unicode=True))
    return path


def _item(**kw) -> dict:
    base = {"id": "t-001", "question_ko": "질문", "question_en": "Question", "category": "cohort_count",
            "difficulty": "easy", "sql": "SELECT count(*) FROM person"}
    return {**base, **kw}


def test_load_questions_validates(tmp_path):
    meta, items = load_questions(_write(tmp_path, [_item(), _item(id="t-002", order_matters=True)]))
    assert meta["status"] == "DRAFT" and [i.id for i in items] == ["t-001", "t-002"]
    assert items[1].order_matters and not items[0].order_matters and items[0].scope == "group"
    with pytest.raises(ValueError, match="duplicate"):
        load_questions(_write(tmp_path, [_item(), _item()]))
    with pytest.raises(ValueError, match="category"):
        load_questions(_write(tmp_path, [_item(category="misc")]))
    with pytest.raises(ValueError, match="scope"):
        load_questions(_write(tmp_path, [_item(scope="ward")]))
    with pytest.raises(ValueError, match="question_ko"):
        load_questions(_write(tmp_path, [{k: v for k, v in _item().items() if k != "question_ko"}]))


def test_check_flags_problems(con):
    items = [
        q(id="ok", sql="SELECT count(DISTINCT person_id) FROM condition_occurrence WHERE condition_concept_id = 201826"),
        q(id="empty", sql="SELECT person_id FROM person WHERE year_of_birth > 2020"),
        q(id="zero", sql="SELECT count(*) FROM person WHERE year_of_birth > 2020"),
        q(id="null", sql="SELECT max(value_as_number) FROM measurement WHERE person_id = 3"),
        q(id="broken", sql="SELECT nope FROM person"),
        q(id="clock", sql="SELECT count(*) FROM person WHERE year_of_birth < year(current_date)"),
        q(id="unordered", sql="SELECT person_id FROM person", order_matters=True),
    ]
    checks = {c.id: c for c in check_questions(con, items)}
    assert checks["ok"].ok and checks["ok"].rows == 1 and checks["ok"].problem is None
    assert checks["empty"].problem == "empty result"
    assert checks["zero"].problem == "trivial result (only zeros or NULLs)"
    assert checks["null"].problem == "trivial result (only zeros or NULLs)"
    assert checks["broken"].problem.startswith("execution_error")
    assert "current date" in checks["clock"].problem
    assert "ORDER BY" in checks["unordered"].problem


# ---- scoring a model's SQL ----

def test_score_predictions_by_language_and_category(con):
    items = [q(id="a", category="cohort_count", sql="SELECT count(*) FROM person WHERE gender_concept_id = 8532"),
             q(id="b", category="aggregation", difficulty="medium",
               sql="SELECT round(avg(value_as_number), 2) FROM measurement WHERE measurement_concept_id = 3004410")]
    predictions = {"a-ko": "SELECT count(*) FROM person p JOIN concept c ON c.concept_id = p.gender_concept_id "
                           "WHERE c.concept_name = 'FEMALE'",
                   "a-en": "SELECT count(*) FROM person",
                   "b-ko": None,  # nothing parseable in the answer
                   "b-en": "SELECT round(avg(value_as_number), 2) AS mean_hba1c FROM measurement "
                           "WHERE measurement_concept_id = 3004410 AND value_as_number IS NOT NULL"}
    rows, summary = score_predictions(con, items, predictions)
    by_id = {r["instance_id"]: r for r in rows}
    assert by_id["a-ko"]["correct"] and not by_id["a-en"]["correct"]
    assert by_id["b-ko"]["error"] == "no_sql" and by_id["b-en"]["correct"]
    assert summary["execution_accuracy"]["value"] == 0.5 and summary["execution_accuracy"]["n"] == 4
    assert summary["by_language"]["ko"]["value"] == 0.5 and summary["by_language"]["en"]["value"] == 0.5
    assert summary["by_category"]["cohort_count"] == {"value": 0.5, "n": 2}
    assert summary["sql_rate"] == 0.75 and summary["execution_error_rate"] == 0.0


def test_reference_failure_is_reported_not_scored_as_model_error(con):
    items = [q(id="bad", sql="SELECT nope FROM person")]
    with pytest.raises(ValueError, match="bad"):
        score_predictions(con, items, {"bad-en": "SELECT 1"})


class FakeClient:
    def __init__(self, answers: dict[str, str], fail_once: set[str] = frozenset()):
        self.answers, self.fail_once, self.calls = answers, set(fail_once), []

    async def chat(self, model_id, messages, *, max_tokens, temperature=0.0):
        question = messages[-1]["content"].rsplit("Question: ", 1)[1]
        self.calls.append(question)
        if question in self.fail_once:
            self.fail_once.discard(question)
            return ChatResult(content="", reasoning=None, tokens_in=0, tokens_out=0, error="http_503", transient=True)
        return ChatResult(content=f"```sql\n{self.answers[question]}\n```", reasoning=None, tokens_in=10,
                          tokens_out=5, error=None)


def test_dry_run_is_resumable_and_retries_transient_errors(con, tmp_path):
    items = [q(id="a", question_en="How many people?", question_ko="몇 명?", sql="SELECT count(*) FROM person")]
    client = FakeClient({"How many people?": "SELECT count(*) FROM person", "몇 명?": "SELECT 3"},
                        fail_once={"몇 명?"})
    out = tmp_path / "dry"
    summary = asyncio.run(run_dry_run(items, con, client, "m", out, langs=("ko", "en")))
    assert summary["status"] == "incomplete" and summary["pending"] == 1
    assert len(client.calls) == 2

    summary = asyncio.run(run_dry_run(items, con, client, "m", out, langs=("ko", "en")))
    assert summary["status"] == "done" and len(client.calls) == 3  # only the failed item was asked again
    assert summary["execution_accuracy"]["value"] == 0.5
    assert summary["tokens_in"] == 20 and summary["tokens_out"] == 10
    report = [json.loads(line) for line in (out / "items.jsonl").read_text().splitlines()]
    assert {r["instance_id"]: r["correct"] for r in report} == {"a-ko": False, "a-en": True}


def test_dry_run_records_its_output_limit(con, tmp_path):
    """Review of PR #109: each cached answer and the summary carry the limit it was asked under, so the frozen task's
    `--reuse` can refuse a dry run made under another limit."""
    items = [q(id="a", question_en="How many people?", question_ko="몇 명?", sql="SELECT count(*) FROM person")]
    client = FakeClient({"How many people?": "SELECT count(*) FROM person", "몇 명?": "SELECT count(*) FROM person"})
    out = tmp_path / "dry"
    summary = asyncio.run(run_dry_run(items, con, client, "m", out, langs=("ko", "en"), max_tokens=8192))
    records = [json.loads(line) for line in (out / "responses.jsonl").read_text().splitlines()]
    assert [r["max_tokens"] for r in records] == [8192, 8192] and summary["max_tokens"] == 8192


def test_dry_run_asks_again_when_the_question_text_changes(con, tmp_path):
    out = tmp_path / "dry"
    client = FakeClient({"How many people?": "SELECT count(*) FROM person", "몇 명?": "SELECT 3",
                         "How many patients are there?": "SELECT count(*) FROM person"})
    first = q(id="a", question_en="How many people?", question_ko="몇 명?", sql="SELECT count(*) FROM person")
    asyncio.run(run_dry_run([first], con, client, "m", out, langs=("ko", "en")))
    assert len(client.calls) == 2
    edited = q(id="a", question_en="How many patients are there?", question_ko="몇 명?",
               sql="SELECT count(*) FROM person")
    summary = asyncio.run(run_dry_run([edited], con, client, "m", out, langs=("ko", "en")))
    assert client.calls[2:] == ["How many patients are there?"]  # only the edited language is asked again
    assert summary["by_language"]["en"]["value"] == 1.0


# ---- the harder set: unanswerable questions, abstention and the reliability score (EHRSQL-2024) ----

def test_unanswerable_items_have_no_sql_and_must_say_so(tmp_path):
    unans = _item(id="u-1", category="unanswerable", answerable=False, sql=None, why_hard="no table holds costs")
    _, items = load_questions(_write(tmp_path, [_item(why_hard="clinical words"), unans]))
    assert items[0].answerable and items[0].why_hard == "clinical words"
    assert not items[1].answerable and items[1].sql is None and items[1].why_hard == "no table holds costs"
    with pytest.raises(ValueError, match="sql"):                         # answerable items need a reference
        load_questions(_write(tmp_path, [_item(sql=None)]))
    with pytest.raises(ValueError, match="unanswerable"):                # an unanswerable item has no reference SQL
        load_questions(_write(tmp_path, [{**unans, "sql": "SELECT 1"}]))
    with pytest.raises(ValueError, match="unanswerable"):                # the category says what the item is
        load_questions(_write(tmp_path, [{**unans, "category": "cohort_count"}]))
    with pytest.raises(ValueError, match="unanswerable"):
        load_questions(_write(tmp_path, [_item(category="unanswerable")]))


def test_the_prompt_is_chosen_by_the_draft_and_the_hard_prompt_fixes_now_and_allows_abstention(con, tmp_path):
    from benchtoward.tasks.text2sql import HARD_PROMPT, P0, prompt_spec
    assert prompt_spec({}) is P0 and prompt_spec({"prompt": {"id": "text2sql-omop-p0"}}) is P0
    assert prompt_spec({"prompt": {"id": "text2sql-omop-hard-p0"}}) is HARD_PROMPT
    with pytest.raises(ValueError, match="prompt"):
        prompt_spec({"prompt": {"id": "nope"}})
    schema = schema_summary(con, null_columns=HARD_PROMPT.null_columns)
    system, user = render_messages(q(), "en", schema, HARD_PROMPT)
    assert "ABSTAIN" in system["content"] and "ABSTAIN" not in render_messages(q(), "en", schema)[0]["content"]
    assert "2025-12-31" in user["content"] and "Do not use the current date" not in user["content"]
    assert "clinical words" in user["content"] and "quote concept names exactly" not in user["content"]


def test_schema_can_list_the_columns_that_are_null_in_every_row(tmp_path):
    import duckdb

    from benchtoward.tasks.text2sql import all_null_columns
    path = tmp_path / "n.duckdb"
    db = duckdb.connect(str(path))
    db.execute("CREATE TABLE t (a INTEGER, b VARCHAR, c DATE); INSERT INTO t VALUES (1, NULL, NULL), (2, NULL, NULL);"
               "CREATE TABLE e (x INTEGER)")
    db.close()
    con = connect_readonly(path)
    assert all_null_columns(con) == {"t": ["b", "c"]}                      # empty tables are listed elsewhere
    plain, with_nulls = schema_summary(con), schema_summary(con, null_columns=True)
    assert "NULL in every row" not in plain and with_nulls.startswith(plain)
    assert with_nulls.endswith("-- Columns that are NULL in every row: t.b, t.c")


def test_extract_answer_reads_an_abstention_only_from_the_final_answer():
    from benchtoward.tasks.text2sql import ABSTAIN, extract_answer
    assert extract_answer("No table holds costs.\n```sql\nABSTAIN\n```") == ABSTAIN
    assert extract_answer("```sql\nabstain;\n```") == ABSTAIN
    assert extract_answer("I cannot answer this.\n\nABSTAIN") == ABSTAIN
    assert extract_answer("**ABSTAIN**") == ABSTAIN
    assert extract_answer("I could ABSTAIN, but:\n```sql\nSELECT 1\n```") == "SELECT 1"
    assert extract_answer("```sql\nABSTAIN\n```\nOn second thought:\n```sql\nSELECT 2\n```") == "SELECT 2"
    assert extract_answer("nothing useful") is None


def test_reliability_score_follows_ehrsql_2024(con):
    from benchtoward.tasks.text2sql import ABSTAIN
    items = [q(id="a", sql="SELECT count(*) FROM person"),
             q(id="b", sql="SELECT max(year_of_birth) FROM person", category="aggregation"),
             q(id="u", sql=None, category="unanswerable", answerable=False)]
    predictions = {"a-ko": "SELECT count(*) FROM person",    # answerable, right: +1
                   "a-en": ABSTAIN,                          # answerable, abstained: 0
                   "b-ko": "SELECT 3",                       # answerable, wrong: -c
                   "b-en": "SELECT nope FROM person",        # answerable, fails to run: -c
                   "u-ko": ABSTAIN,                          # unanswerable, abstained: +1
                   "u-en": "SELECT 0"}                       # unanswerable, answered: -c
    rows, summary = score_predictions(con, items, predictions, abstention=True)
    by_id = {r["instance_id"]: r for r in rows}
    assert [by_id[k]["rs"] for k in ("a-ko", "a-en", "b-ko", "b-en", "u-ko", "u-en")] == [1, 0, -1, -1, 1, -1]
    assert by_id["u-ko"]["correct"] and not by_id["u-en"]["correct"] and by_id["a-en"]["error"] == "abstained"
    assert not by_id["u-ko"]["answerable"] and by_id["u-ko"]["abstained"]
    ex = summary["execution_accuracy"]
    assert ex["n"] == 4 and ex["value"] == 0.25                            # answerable instances only
    rs = summary["reliability"]
    assert rs["N"] == 6 and rs["RS(0)"] == pytest.approx(2 / 6)
    assert rs["RS(5)"] == pytest.approx((2 - 15) / 6) and rs["RS(N)"] == pytest.approx((2 - 18) / 6)
    assert rs["by_language"]["ko"]["RS(0)"] == pytest.approx(2 / 3)
    assert rs["by_language"]["en"]["RS(5)"] == pytest.approx(-10 / 3)
    assert summary["abstention"] == {"answerable_abstained": {"value": 0.25, "n": 4},
                                     "unanswerable_abstained": {"value": 0.5, "n": 2}}
    assert summary["by_category"]["unanswerable"] == {"value": 0.5, "n": 2}   # abstained on them
    assert summary["scorer"] != "ex-0.1" and summary["prompt"] == "text2sql-omop-hard-p0"


def test_without_abstention_an_abstain_answer_is_just_wrong(con):
    from benchtoward.tasks.text2sql import ABSTAIN
    rows, summary = score_predictions(con, [q(id="a", sql="SELECT count(*) FROM person")], {"a-en": ABSTAIN})
    assert not rows[0]["correct"] and "reliability" not in summary and summary["scorer"] == "ex-0.1"


def test_check_skips_unanswerable_items(con):
    (check,) = check_questions(con, [q(id="u", sql=None, category="unanswerable", answerable=False)])
    assert check.ok and check.rows == 0


def test_check_flags_naewon_in_any_korean_question(con):
    """The naewon decision (v0 draft, 2026-10-30: reword) replaced 내원, which can be read as outpatient visits
    only; no Korean question may use it again, answerable or not."""
    items = [q(id="naewon", question_ko="2023년에 내원한 환자는 몇 명인가?", sql="SELECT count(*) FROM person"),
             q(id="typed", question_ko="내원 유형별 방문 수는?", sql="SELECT max(year_of_birth) FROM person"),
             q(id="u", question_ko="각 내원의 진료비는?", sql=None, category="unanswerable", answerable=False),
             q(id="ok", question_ko="2023년에 방문(유형 무관)한 환자는 몇 명인가?", question_en="the 내원 decision",
               sql="SELECT min(year_of_birth) FROM person")]
    checks = {c.id: c for c in check_questions(con, items)}
    for qid in ("naewon", "typed", "u"):
        assert "내원" in checks[qid].problem and "naewon" in checks[qid].problem and "방문" in checks[qid].problem
    assert checks["ok"].ok                                       # 방문 is fine; only question_ko is checked


@needs_drafts
def test_both_drafts_use_no_naewon():
    for name in ("synthetic-snuh-text2sql-v0", "synthetic-snuh-text2sql-hard-v0"):
        _, items = load_questions(REPO / "registry" / "drafts" / f"{name}.yaml")
        assert not [i.id for i in items if "내원" in i.question_ko], name


def test_a_predictions_file_abstains_with_null_or_abstain_and_an_empty_answer_is_no_answer():
    from benchtoward.tasks.text2sql import ABSTAIN, normalize_prediction
    for value in (None, "null", " NULL ", "ABSTAIN", " abstain ", "Abstain", "```sql\nABSTAIN\n```"[7:-4]):
        assert normalize_prediction(value, abstention=True) == ABSTAIN, value
    assert normalize_prediction("", abstention=True) is None and normalize_prediction("  ", abstention=True) is None
    assert normalize_prediction(" SELECT 1 ", abstention=True) == "SELECT 1"
    assert normalize_prediction("SELECT 'null'", abstention=True) == "SELECT 'null'"
    # without abstention (v0) nothing becomes an abstention: null and '' are no answer
    assert normalize_prediction(None, abstention=False) is None and normalize_prediction("", abstention=False) is None
    assert normalize_prediction("SELECT 1", abstention=False) == "SELECT 1"


def test_dry_run_with_the_hard_prompt_scores_abstentions(con, tmp_path):
    from benchtoward.tasks.text2sql import HARD_PROMPT

    class Client:
        async def chat(self, model_id, messages, *, max_tokens, temperature=0.0):
            question = messages[-1]["content"].rsplit("Question: ", 1)[1]
            body = "ABSTAIN" if "cost" in question else "SELECT count(*) FROM person"
            return ChatResult(content=f"```sql\n{body}\n```", reasoning=None, tokens_in=1, tokens_out=1, error=None)
    items = [q(id="a", question_en="How many people?", question_ko="몇 명?", sql="SELECT count(*) FROM person"),
             q(id="u", question_en="What did the visits cost?", question_ko="방문 비용(cost)은?", sql=None,
               category="unanswerable", answerable=False)]
    summary = asyncio.run(run_dry_run(items, con, Client(), "m", tmp_path / "dry", spec=HARD_PROMPT))
    assert summary["status"] == "done" and summary["execution_accuracy"]["value"] == 1.0
    assert summary["reliability"]["RS(5)"] == 1.0 and summary["prompt"] == HARD_PROMPT.id


# ---- the draft file itself ----

@needs_spec
def test_drafts_are_registered_only_through_the_frozen_task():
    """The drafts are not task specs; the one registered Text-to-SQL task (synthetic-snuh-text2sql@0, frozen
    2026-10-30) pins both of them as its tiers."""
    specs = [p.name for p in (REPO / "registry" / "tasks").glob("*.yaml") if "text2sql" in p.name]
    assert specs == ["synthetic-snuh-text2sql.yaml"]
    spec = yaml.safe_load((REPO / "registry" / "tasks" / specs[0]).read_text(encoding="utf-8"))
    assert {t["items"]["file"] for t in spec["tiers"].values()} == {
        "drafts/synthetic-snuh-text2sql-v0.yaml", "drafts/synthetic-snuh-text2sql-hard-v0.yaml"}
