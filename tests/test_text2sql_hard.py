"""The harder Text-to-SQL set (owner, 2026-10-30: the three frontier models resolve everything and make the fixes):
proposers rotate by category, every reference passes v0's rules, all three review every item, and only unanimous
approval keeps an item without a fix."""
import asyncio
import json
from pathlib import Path

import pytest

from benchtoward.llm import ChatResult

CLAUDE, GPT, GEMINI = "anthropic/claude-opus-5.5", "openai/gpt-6-astra", "google/gemini-4-argon"
MODELS = [CLAUDE, GPT, GEMINI]


def make_hard_db(path: Path) -> Path:
    """A miniature OMOP CDM with the tables and columns the harder set's profile and traps use."""
    import duckdb

    con = duckdb.connect(str(path))
    con.execute("""
        CREATE TABLE concept (concept_id INTEGER, concept_name VARCHAR, domain_id VARCHAR, vocabulary_id VARCHAR,
                              concept_class_id VARCHAR, concept_code VARCHAR);
        INSERT INTO concept VALUES (0, 'No matching concept', 'Metadata', 'None', 'Undefined', 'No matching concept'),
            (8507, 'MALE', 'Gender', 'Gender', 'Gender', 'M'), (8532, 'FEMALE', 'Gender', 'Gender', 'Gender', 'F'),
            (9201, 'Inpatient Visit', 'Visit', 'Visit', 'Visit', 'IP'),
            (9202, 'Outpatient Visit', 'Visit', 'Visit', 'Visit', 'OP'),
            (201826, 'Type 2 diabetes mellitus', 'Condition', 'SNOMED', 'Clinical Finding', '44054006'),
            (313217, 'Atrial fibrillation', 'Condition', 'SNOMED', 'Clinical Finding', '49436004'),
            (4086923, 'Screening procedure', 'Measurement', 'SNOMED', 'Procedure', '20135006'),
            (1503297, 'metformin', 'Drug', 'RxNorm', 'Ingredient', '6809'),
            (21600744, 'Biguanides', 'Drug', 'ATC', 'ATC 4th', 'A10BA'),
            (3004410, 'Hemoglobin A1c/Hemoglobin.total in Blood', 'Measurement', 'LOINC', 'Lab Test', '4548-4'),
            (4249893, 'Colonoscopy', 'Procedure', 'SNOMED', 'Procedure', '73761001'),
            (4167217, 'Family history of clinical finding', 'Observation', 'SNOMED', 'Context-dependent', '416471007');
        CREATE TABLE person (person_id INTEGER, gender_concept_id INTEGER, year_of_birth INTEGER,
                             person_source_value VARCHAR, race_source_value VARCHAR);
        INSERT INTO person VALUES (1, 8507, 1950, 'SYN-000001', NULL), (2, 8532, 1980, 'SYN-000002', NULL),
            (3, 8532, 2001, 'SYN-000003', NULL);
        CREATE TABLE visit_occurrence (visit_occurrence_id INTEGER, person_id INTEGER, visit_concept_id INTEGER,
                                       visit_start_date DATE, visit_end_date DATE, visit_source_value VARCHAR,
                                       admitted_from_concept_id INTEGER, admitted_from_source_value VARCHAR);
        INSERT INTO visit_occurrence VALUES (1, 1, 9202, '2025-03-01', '2025-03-01', 'IME', 9202, 'Outpatient Visit'),
            (2, 2, 9201, '2025-11-02', '2025-11-09', 'IMC', 0, NULL),
            (3, 2, 9202, '2024-07-07', '2024-07-07', 'IME', 9202, 'Outpatient Visit'),
            (4, 3, 9202, '2023-01-10', '2023-01-10', 'PED', 9202, 'Outpatient Visit');
        CREATE TABLE condition_occurrence (condition_occurrence_id INTEGER, person_id INTEGER,
                                           condition_concept_id INTEGER, condition_start_date DATE,
                                           condition_type_concept_id INTEGER, visit_occurrence_id INTEGER,
                                           condition_source_value VARCHAR);
        INSERT INTO condition_occurrence VALUES (10, 1, 201826, '2025-03-01', 44786627, 1, 'E11.9'),
            (11, 2, 313217, '2025-11-02', 44786627, 2, 'I48.9'), (12, 2, 201826, '2024-07-07', 44786627, 3, 'E11.9'),
            (13, 2, 201826, '2024-07-07', 44786627, 3, 'E11.9'), (14, 3, 4086923, '2023-01-10', 44786627, 4, 'Z13.1');
        CREATE TABLE drug_exposure (drug_exposure_id INTEGER, person_id INTEGER, drug_concept_id INTEGER,
                                    drug_exposure_start_date DATE, drug_exposure_end_date DATE, quantity DOUBLE,
                                    days_supply INTEGER, route_concept_id INTEGER, visit_occurrence_id INTEGER,
                                    atc_cd VARCHAR);
        INSERT INTO drug_exposure VALUES (20, 1, 1503297, '2025-03-01', '2025-03-30', 30, 30, 4132161, 1, 'A10BA02'),
            (21, 2, 1503297, '2024-07-07', '2024-08-05', 30, 30, 4132161, 3, NULL),
            (22, 2, 21600744, '2025-11-03', '2025-11-03', 1, 1, NULL, 2, 'A10BA'),
            (23, 3, 0, '2023-01-10', '2023-01-10', NULL, NULL, NULL, 4, NULL);
        CREATE TABLE measurement (measurement_id INTEGER, person_id INTEGER, measurement_concept_id INTEGER,
                                  measurement_date DATE, value_as_number DOUBLE, range_low DOUBLE, range_high DOUBLE,
                                  unit_source_value VARCHAR, visit_occurrence_id INTEGER,
                                  measurement_source_value VARCHAR, measurement_datetime TIMESTAMP);
        INSERT INTO measurement VALUES (30, 1, 3004410, '2025-03-01', 7.1, 4.0, 6.0, '%', 1, 'L3068', NULL),
            (31, 2, 3004410, '2025-11-02', 6.4, 4.0, NULL, '%', 2, 'L3068', NULL),
            (32, 2, 3004410, '2024-07-07', NULL, NULL, NULL, NULL, 3, 'L3068', NULL),
            (33, 3, 0, '2023-01-10', -1.5, NULL, NULL, NULL, 4, 'L9999', NULL);
        CREATE TABLE procedure_occurrence (procedure_occurrence_id INTEGER, person_id INTEGER,
                                           procedure_concept_id INTEGER, procedure_date DATE,
                                           visit_occurrence_id INTEGER);
        INSERT INTO procedure_occurrence VALUES (40, 1, 4249893, '2025-03-01', 1), (41, 2, 4249893, '2025-11-04', 2);
        CREATE TABLE observation (observation_id INTEGER, person_id INTEGER, observation_concept_id INTEGER,
                                  observation_date DATE);
        INSERT INTO observation VALUES (50, 2, 4167217, '2024-07-07');
        CREATE TABLE death (person_id INTEGER, death_date DATE, cause_concept_id INTEGER);
        INSERT INTO death VALUES (1, '2025-04-01', NULL);
        CREATE TABLE note (note_id INTEGER, person_id INTEGER);
    """)
    con.close()
    return path


@pytest.fixture
def hard_db(tmp_path) -> Path:
    return make_hard_db(tmp_path / "hard.duckdb")


def _text(message: dict) -> str:
    content = message["content"]
    return content if isinstance(content, str) else "".join(part["text"] for part in content)


def item(ko, en, sql, *, category=None, why="why", order=False):
    return {"question_ko": ko, "question_en": en, "sql": sql, "order_matters": order, "category": category,
            "scope": "group", "why_hard": why}


class Models:
    """Fake proposers and reviewers: `generate(model, category, batch_no, user)` and `repair(...)` give item lists,
    `review(model, user)` a verdict."""

    def __init__(self, generate, repair=None, review=None):
        self.generate, self.repair, self.review, self.calls = generate, repair, review, []

    def clients(self):
        fake = self

        class Client:
            def __init__(self, model):
                self.model = model

            async def chat(self, model_id, messages, *, max_tokens, temperature=0.0, extra=None, before_attempt=None):
                user = _text(messages[-1])
                first = _text(messages[1])
                if "break the rules" in user:
                    category = first.rsplit('of category "', 1)[1].split('"', 1)[0]
                    fake.calls.append(("repair", model_id, category, user))
                    body = {"items": fake.repair(model_id, category, user)}
                elif "new items of category" in user:
                    category = user.rsplit('of category "', 1)[1].split('"', 1)[0]
                    fake.calls.append(("generate", model_id, category, user))
                    body = {"items": fake.generate(model_id, category, user)}
                else:
                    fake.calls.append(("review", model_id, user))
                    body = fake.review(model_id, user)
                return ChatResult(content=json.dumps(body), reasoning=None, tokens_in=1000, tokens_out=200,
                                  error=None, cost_usd=0.01)
        return {m: Client(m) for m in MODELS}


def _verdict(v="approve", reason=None, fix=""):
    return {"natural": True, "unambiguous": True, "ko_en_match": True, "sql_correct": True, "result_plausible": True,
            "difficulty": "hard", "verdict": v, "reason": reason or f"because {v}", "fix": fix, "harder_variant": ""}


# ---- generation ----

def test_proposers_rotate_by_category_so_every_category_has_every_model():
    from benchtoward.tasks.text2sql_hard import PLAN, plan_batches
    batches = plan_batches(MODELS)
    by_category = {}
    for b in batches:
        by_category.setdefault(b.category, []).append(b.proposer)
    assert all(set(proposers) == set(MODELS) for proposers in by_category.values())
    assert by_category["clinical_terms"] == [CLAUDE, GPT, GEMINI]
    assert by_category["cohort_definition"] == [GPT, GEMINI, CLAUDE]           # the next category starts one later
    counts = {m: sum(b.size for b in batches if b.proposer == m) for m in MODELS}
    assert sum(counts.values()) == sum(sum(v) for v in PLAN.values())
    assert max(counts.values()) - min(counts.values()) <= 3
    unanswerable = sum(PLAN["unanswerable"]) / sum(sum(v) for v in PLAN.values())
    assert 0.15 <= unanswerable <= 0.20                                       # EHRSQL-style share of abstentions


def test_parse_items_takes_the_batch_category_and_keeps_answerability():
    from benchtoward.tasks.text2sql_hard import parse_items
    text = json.dumps({"items": [item("가", "a", "SELECT 1;", category="other"), item("나", "b", None),
                                 {"question_ko": "다"}, item("라", "d", "SELECT 2", order=True)]})
    rows = parse_items("Here:\n```json\n" + text + "\n```", "clinical_terms")
    assert [r["question_en"] for r in rows] == ["a", "d"] and {r["category"] for r in rows} == {"clinical_terms"}
    assert rows[0]["sql"] == "SELECT 1" and rows[1]["order_matters"] is True
    rows = parse_items(json.dumps([item("가", "a", None), item("나", "b", "SELECT 1")]), "unanswerable")
    assert [r["question_en"] for r in rows] == ["a"] and rows[0]["sql"] is None and not rows[0]["order_matters"]
    assert parse_items("no json", "clinical_terms") is None


def test_a_question_may_not_quote_a_concept_name_its_reference_uses(hard_db):
    from benchtoward.tasks.synthetic_snuh import connect_readonly
    from benchtoward.tasks.text2sql_hard import as_question, item_problem
    con = connect_readonly(hard_db)
    sql = ("SELECT count(DISTINCT co.person_id) FROM condition_occurrence co JOIN concept c ON c.concept_id = "
           "co.condition_concept_id WHERE c.concept_name = 'Type 2 diabetes mellitus'")
    quoted = as_question("h", item("\"Type 2 diabetes mellitus\" 환자 수", "How many T2DM patients?", sql,
                                   category="clinical_terms"))
    assert "quotes concept names (Type 2 diabetes mellitus)" in item_problem(con, quoted, [])
    plain = as_question("h", item("제2형 당뇨병 환자 수는?", "How many patients have T2DM?", sql,
                                  category="clinical_terms"))
    assert item_problem(con, plain, []) is None
    assert "trivial" in item_problem(con, as_question("h", item("가", "a", "SELECT 0", category="data_quality")), [])


def test_the_profile_lists_concepts_in_use_and_the_data_quality_facts(hard_db):
    from benchtoward.tasks.synthetic_snuh import connect_readonly
    from benchtoward.tasks.text2sql_hard import data_profile
    text = data_profile(connect_readonly(hard_db))
    assert "201826 | Type 2 diabetes mellitus | 2 | 3 | E11.9" in text                 # patients, records, KCD
    assert "Screening procedure (Measurement) 1" in text                               # non-Condition domain
    assert "drug_exposure.drug_concept_id = 0 (No matching concept): 1 records" in text
    assert "condition_occurrence: 1 groups of duplicate records (1 extra rows)" in text
    assert "ATC/ATC 4th 1 records" in text and "value_as_number < 0: 1 results" in text


def _good(category, n):
    """n valid items of a category for the fixture database."""
    if category == "unanswerable":
        return [item(f"방문 {k}의 진료비는?", f"What did visit {k} cost?", None, why="no cost table") for k in range(n)]
    return [item(f"{category} 질문 {k}", f"{category} question {k}",
                 f"SELECT count(*) + {k} FROM visit_occurrence WHERE visit_source_value <> '{category}'")
            for k in range(n)]


def test_generation_checks_every_reference_and_sends_failures_back_once(hard_db, tmp_path):
    from benchtoward.tasks.text2sql_hard import run_generate
    from benchtoward.tasks.text2sql_llm_fix import Budget
    plan = {"clinical_terms": [3, 2], "unanswerable": [2]}

    def generate(model, category, user):
        rows = _good(category, 3 if category == "clinical_terms" else 2)
        if category == "clinical_terms" and "question 0" not in user:              # the first batch
            rows[1] = item("깨진 질문", "broken question", "SELECT nope FROM person")
            rows[2] = item("빈 질문", "empty question", "SELECT person_id FROM person WHERE year_of_birth > 2020")
        elif category == "clinical_terms":
            rows = [item("두번째", "second batch item", "SELECT max(year_of_birth) FROM person")]
        return rows

    def repair(model, category, user):
        assert "execution_error" in user and "empty result" in user
        return [item("고친 질문", "repaired question", "SELECT min(year_of_birth) FROM person"),
                item("또 깨짐", "still broken", "SELECT nope2 FROM person")]
    fake = Models(generate, repair)
    budget = Budget(10.0)
    result = asyncio.run(run_generate(hard_db, fake.clients(), {m: (1.0, 1.0) for m in MODELS}, tmp_path / "gen",
                                      budget, models=MODELS, existing=[], plan=plan))
    english = [r["question_en"] for r in result.items]
    assert english == ["clinical_terms question 0", "repaired question", "second batch item",
                       "What did visit 0 cost?", "What did visit 1 cost?"]
    assert [r["proposer"] for r in result.items] == [CLAUDE, CLAUDE, GPT, GPT, GPT]   # rotation by category
    assert [b["sent_back"] for b in result.batches if b["batch"].startswith("clinical")][0] == 2
    assert [d["question_en"] for d in result.dropped] == ["still broken"]
    second = [c for c in fake.calls if c[0] == "generate" and c[2] == "clinical_terms"][1][3]
    assert "clinical_terms question 0" in second and "repaired question" in second    # sees the items written so far
    assert "2025-12-31" in second and "ABSTAIN" in second and "Data profile" in second
    assert budget.spent_usd == pytest.approx(0.04) and not result.incomplete
    again = asyncio.run(run_generate(hard_db, fake.clients(), {m: (1.0, 1.0) for m in MODELS}, tmp_path / "gen",
                                     Budget(10.0), models=MODELS, existing=[], plan=plan))
    assert len(fake.calls) == 4 and [r["question_en"] for r in again.items] == english   # cached


def test_a_duplicate_across_categories_is_dropped_from_the_later_one(hard_db, tmp_path):
    from benchtoward.tasks.text2sql_hard import run_generate
    from benchtoward.tasks.text2sql_llm_fix import Budget
    same = "SELECT count(*) FROM drug_exposure WHERE atc_cd IS NULL"
    fake = Models(lambda model, category, user: [item(f"{category}", f"{category} q", same)],
                  repair=lambda *a: [])
    result = asyncio.run(run_generate(hard_db, fake.clients(), {}, tmp_path / "gen", Budget(10.0), models=MODELS,
                                      existing=[], plan={"clinical_terms": [1], "data_quality": [1]}))
    assert [r["category"] for r in result.items] == ["clinical_terms"]
    caught = result.sent_back + result.dropped                                # in its batch or in the final check
    assert caught[0]["category"] == "data_quality" and "same reference SQL" in caught[0]["problem"]


def test_the_budget_stops_generation_and_reports_the_batch(hard_db, tmp_path):
    from benchtoward.tasks.text2sql_hard import run_generate
    from benchtoward.tasks.text2sql_llm_fix import Budget
    fake = Models(lambda *a: _good("clinical_terms", 2))
    result = asyncio.run(run_generate(hard_db, fake.clients(), {m: (10.0, 50.0) for m in MODELS}, tmp_path / "gen",
                                      Budget(0.5), models=MODELS, existing=[], plan={"clinical_terms": [2]}))
    assert result.incomplete == ["clinical_terms/g1/b1"] and not fake.calls and not result.items


# ---- the draft ----

HARD_HEADER = """# DRAFT header
status: "DRAFT"
prompt:
  id: text2sql-omop-hard-p0
questions: []
"""


def test_append_items_writes_pending_items_that_read_back(tmp_path):
    from benchtoward.tasks.text2sql import load_questions
    from benchtoward.tasks.text2sql_hard import append_items
    draft = tmp_path / "hard.yaml"
    draft.write_text(HARD_HEADER, encoding="utf-8")
    rows = [{**item("제2형 당뇨병: 환자 수", "T2DM: how many?", "SELECT 1\n\nFROM person", category="clinical_terms",
                    order=True), "proposer": CLAUDE, "gen_round": 1, "batch": "clinical_terms/g1/b1"},
            {**item("진료비는?", "What did it cost?", None, category="unanswerable", why="no cost table"),
             "proposer": GPT, "gen_round": 1, "batch": "unanswerable/g1/b1"}]
    assert append_items(draft, rows) == ["syn-hard-001", "syn-hard-002"]
    assert append_items(draft, [{**rows[0], "gen_round": 2, "batch": "clinical_terms/g2/b1"}]) == ["syn-hard-003"]
    _, items = load_questions(draft)
    assert [i.id for i in items] == ["syn-hard-001", "syn-hard-002", "syn-hard-003"]
    one, two = items[0], items[1]
    assert one.sql == "SELECT 1\n\nFROM person" and one.order_matters and one.difficulty == "hard"
    assert one.question_ko == "제2형 당뇨병: 환자 수" and one.review == "pending" and one.why_hard == "why"
    assert two.sql is None and not two.answerable and two.why_hard == "no cost table"
    text = draft.read_text(encoding="utf-8")
    assert text.startswith("# DRAFT header") and "proposed_by: openai/gpt-6-astra" in text
    assert "# ---------------------------------------------------------------- unanswerable" in text
    from benchtoward.tasks.text2sql_hard import batch_round, draft_batches, parse_plan
    assert draft_batches(draft) == {"syn-hard-001": "clinical_terms/g1/b1", "syn-hard-002": "unanswerable/g1/b1",
                                    "syn-hard-003": "clinical_terms/g2/b1"}
    assert batch_round("clinical_terms/g2/b1") == 2 and batch_round(None) is None
    assert parse_plan("clinical_terms=8,8; unanswerable=7") == {"clinical_terms": [8, 8], "unanswerable": [7]}
    with pytest.raises(ValueError, match="unknown category"):
        parse_plan("misc=3")


# ---- review ----

def _draft_with_items(tmp_path, hard_db):
    from benchtoward.tasks.text2sql_hard import append_items
    draft = tmp_path / "hard.yaml"
    draft.write_text(HARD_HEADER, encoding="utf-8")
    rows = [{**r, "category": c, "proposer": CLAUDE, "gen_round": 1, "batch": f"{c}/g1/b1"}
            for c, r in [("clinical_terms", _good("clinical_terms", 3)[0]),
                         ("clinical_terms", _good("clinical_terms", 3)[1]),
                         ("clinical_terms", _good("clinical_terms", 3)[2]),
                         ("unanswerable", _good("unanswerable", 1)[0])]]
    append_items(draft, rows)
    return draft


def test_only_unanimous_approval_keeps_an_item_and_everything_else_goes_to_the_fix(hard_db, tmp_path):
    from benchtoward.tasks.text2sql import load_questions
    from benchtoward.tasks.text2sql_hard import apply_reviews, collect_reviews, run_hard_review
    from benchtoward.tasks.text2sql_llm_fix import Budget
    draft = _draft_with_items(tmp_path, hard_db)
    _, items = load_questions(draft)

    def review(model, user):
        if "question 1" in user and model == GPT:
            return _verdict("needs_edit", reason="which visits?", fix="say outpatient")
        if "question 2" in user and model != CLAUDE:
            return _verdict("reject", reason="trivial")
        if "cost" in user:
            assert "Reference SQL: none" in user and "abstain" in user
        return _verdict()
    fake = Models(None, review=review)
    out = tmp_path / "rev"
    reviews = asyncio.run(run_hard_review(items, hard_db, fake.clients(), {}, out, Budget(10.0)))
    assert len(reviews) == 4 and all(len(r) == 3 for r in reviews.values())
    assert all("2025-12-31" in c[2] for c in fake.calls)                           # the harder set's note
    counts = apply_reviews(draft, reviews)
    assert counts == {"approved": 2, "needs_edit": 2}
    _, items = load_questions(draft)
    assert [i.review for i in items] == ["approved", "needs_edit", "needs_edit", "approved"]
    import yaml
    raw = {q["id"]: q for q in yaml.safe_load(draft.read_text(encoding="utf-8"))["questions"]}
    note = raw["syn-hard-002"]["review_note"]
    assert "2 of 3 approved" in note and "say outpatient" in note
    assert "2 rejected" in raw["syn-hard-003"]["review_note"] and "review_note" not in raw["syn-hard-001"]
    collected = collect_reviews(out, MODELS, items)
    assert collected["syn-hard-002"][GPT]["fix"] == "say outpatient" and set(collected) == {i.id for i in items}
    again = asyncio.run(run_hard_review(items, hard_db, fake.clients(), {}, out, Budget(10.0)))
    assert again == {}                                                          # nothing pending any more


def test_the_first_review_asks_about_several_items_at_once_with_the_same_rubric(hard_db, tmp_path):
    from benchtoward.tasks.text2sql import load_questions
    from benchtoward.tasks.text2sql_hard import collect_reviews, run_hard_review
    from benchtoward.tasks.text2sql_llm_fix import Budget
    draft = _draft_with_items(tmp_path, hard_db)
    _, items = load_questions(draft)
    seen = []

    def review(model, user):
        ids = [line.split("Question id: ", 1)[1] for line in user.splitlines() if line.startswith("Question id: ")]
        seen.append((model, ids))
        if len(ids) == 1:                                                     # the fallback for a missing verdict
            return _verdict()
        rows = [{"id": qid, **_verdict("needs_edit" if qid == "syn-hard-002" else "approve")} for qid in ids]
        return {"reviews": [r for r in rows if not (model == GEMINI and r["id"] == "syn-hard-004")]}
    fake = Models(None, review=review)
    out = tmp_path / "rev"
    reviews = asyncio.run(run_hard_review(items, hard_db, fake.clients(), {}, out, Budget(10.0), batch_size=2))
    batches = sorted((m, ids) for m, ids in seen if len(ids) > 1)
    assert batches == sorted((m, ids) for m in MODELS for ids in (["syn-hard-001", "syn-hard-002"],
                                                                   ["syn-hard-003", "syn-hard-004"]))
    assert [(m, ids) for m, ids in seen if len(ids) == 1] == [(GEMINI, ["syn-hard-004"])]   # its missing verdict
    assert reviews["syn-hard-002"][GPT]["verdict"] == "needs_edit"
    assert reviews["syn-hard-004"][GEMINI]["verdict"] == "approve"
    batch_prompt = next(c[2] for c in fake.calls if "syn-hard-004" in c[2] and "syn-hard-003" in c[2])
    assert "Reference SQL: none" in batch_prompt and "2025-12-31" in batch_prompt
    assert collect_reviews(out, MODELS, items)["syn-hard-002"][GPT]["verdict"] == "needs_edit"
    assert collect_reviews(out, MODELS, items)["syn-hard-004"][GEMINI]["verdict"] == "approve"


def test_parse_batch_review_keeps_only_valid_verdicts_for_the_asked_ids():
    from benchtoward.tasks.text2sql_llm_review import parse_batch_review
    body = {"reviews": [{"id": "a", **_verdict()}, {"id": "b", "verdict": "maybe"}, {"id": "z", **_verdict()}]}
    assert parse_batch_review(json.dumps(body), ["a", "b"]) == {"a": {"id": "a", **_verdict()}}
    assert parse_batch_review("nothing", ["a"]) is None


def test_review_incomplete_under_the_budget_leaves_items_pending(hard_db, tmp_path):
    from benchtoward.tasks.text2sql import load_questions
    from benchtoward.tasks.text2sql_hard import apply_reviews, run_hard_review
    from benchtoward.tasks.text2sql_llm_fix import Budget
    draft = _draft_with_items(tmp_path, hard_db)
    _, items = load_questions(draft)
    fake = Models(None, review=lambda m, u: _verdict())
    reviews = asyncio.run(run_hard_review(items, hard_db, fake.clients(), {m: (10.0, 50.0) for m in MODELS},
                                          tmp_path / "rev", Budget(0.01)))
    assert set(reviews.values()) == {None}
    assert apply_reviews(draft, reviews) == {}
    assert {i.review for i in load_questions(draft)[1]} == {"pending"}


def test_spend_report_separates_route_billed_from_estimates(tmp_path):
    from benchtoward.tasks.text2sql_hard import spend_report
    (tmp_path / "generate").mkdir()
    (tmp_path / "fix").mkdir()
    (tmp_path / "generate" / "m.jsonl").write_text(
        json.dumps({"cost_usd": 0.5, "cost_source": "route"}) + "\n"
        + json.dumps({"cost_usd": 0.0, "cost_source": "estimate", "failed_attempts_usd": 0.2}) + "\n")
    (tmp_path / "fix" / "m.jsonl").write_text(json.dumps({"cost_usd": 0.25, "cost_source": "route"}) + "\n")
    report = spend_report(tmp_path)
    assert report["route_usd"] == pytest.approx(0.75) and report["failed_attempts_usd"] == pytest.approx(0.2)
    assert report["by_step"]["generate"] == {"usd": pytest.approx(0.7), "requests": 2} and report["requests"] == 3
