"""Text-to-SQL draft items that need an edit are fixed by the three frontier reviewers themselves (owner, 2026-10-30):
one model proposes a revised item, the reference rules check its SQL, all three re-review it; proposers rotate
claude -> gpt -> gemini over at most three rounds; unanimous approval writes the item, otherwise it is dropped."""
import asyncio
import json

import pytest

from medbench.llm import ChatResult

CLAUDE, GPT, GEMINI = "anthropic/claude-opus-5.5", "openai/gpt-6-astra", "google/gemini-4-argon"
MODELS = [CLAUDE, GPT, GEMINI]

DRAFT = """# DRAFT header comment that must survive
status: "DRAFT"
questions:

# --- a section comment

  - id: q-001
    category: patient_lookup
    difficulty: easy
    scope: individual
    question_ko: 환자 SYN-000001은 몇 년생인가?
    question_en: In which year was patient SYN-000001 born?
    sql: |
      SELECT year_of_birth AS birth_year FROM person WHERE person_source_value = 'SYN-000001'
    review: approved

  - id: q-002
    category: aggregation
    difficulty: medium
    scope: none
    question_ko: 가장 많이 방문한 진료과는?
    question_en: Which department has the most visits?
    sql: |
      SELECT visit_source_value FROM visit_occurrence GROUP BY 1 ORDER BY count(*) DESC, 1 LIMIT 1
    note: author note stays
    review: needs_edit
    review_note: "[gpt-6-astra] needs_edit: ties at the top. Fix: return all tied departments"

# --- another section

  - id: q-003
    category: cohort_count
    difficulty: medium
    scope: group
    question_ko: 1960년 이후 출생 환자는 몇 명인가?
    question_en: How many patients were born after 1960?
    sql: |
      SELECT count(*) FROM person WHERE year_of_birth > 1960
    review: needs_edit
    review_note: "[claude-opus-5.5] needs_edit: after = strictly after?"
"""

GOOD_002 = ("WITH n AS (SELECT visit_source_value, count(*) AS k FROM visit_occurrence GROUP BY 1)\n"
            "SELECT visit_source_value, k FROM n WHERE k = (SELECT max(k) FROM n)")


def _proposal(sql, ko="수정된 질문", en="Revised question", order_matters=False, change="tie rule added"):
    return json.dumps({"question_ko": ko, "question_en": en, "sql": sql, "order_matters": order_matters,
                       "change": change})


def _verdict(v, reason=None, fix=""):
    return json.dumps({"natural": True, "unambiguous": True, "ko_en_match": True, "sql_correct": True,
                       "result_plausible": True, "difficulty": "medium", "verdict": v,
                       "reason": reason or f"because {v}", "fix": fix, "harder_variant": ""})


def text_of(message: dict) -> str:
    """A message's text; requests to Anthropic models carry it as parts with a prompt-cache mark."""
    content = message["content"]
    return content if isinstance(content, str) else "".join(part["text"] for part in content)


class Script:
    """Fake models: proposals come from `proposals[(qid, round)]`, reviews from `reviews(model, user_text)`."""

    def __init__(self, proposals, reviews):
        self.proposals, self.reviews, self.calls = proposals, reviews, []

    def client(self, model):
        script = self

        class Client:
            async def chat(self, model_id, messages, *, max_tokens, temperature=0.0, extra=None, before_attempt=None):
                user = text_of(messages[-1])
                if "Revise this item" in user:
                    qid = user.split("Question id: ", 1)[1].split("\n", 1)[0]
                    rnd = int(user.split("This is round ", 1)[1].split(" ", 1)[0])
                    script.calls.append(("propose", model_id, qid, rnd, user))
                    content = script.proposals[(qid, rnd)]
                else:
                    script.calls.append(("review", model_id, user))
                    content = script.reviews(model_id, user)
                return ChatResult(content=content, reasoning=None, tokens_in=1000, tokens_out=500, error=None,
                                  cost_usd=0.01)
        return Client()

    def clients(self):
        return {m: self.client(m) for m in MODELS}


@pytest.fixture
def draft(tmp_path):
    p = tmp_path / "draft.yaml"
    p.write_text(DRAFT, encoding="utf-8")
    return p


def _run(draft, omop_db, script, out, *, cap=30.0, prices=None, ids=None):
    from medbench.tasks.text2sql import load_questions
    from medbench.tasks.text2sql_llm_fix import Budget, run_fix
    _, items = load_questions(draft)
    original = {"q-002": {GPT: {"verdict": "needs_edit", "reason": "ties at the top", "fix": "return all ties"},
                          GEMINI: {"verdict": "approve", "reason": "fine", "fix": ""},
                          CLAUDE: {"verdict": "needs_edit", "reason": "nondeterministic", "fix": "WITH ..."}},
                "q-003": {GPT: None, GEMINI: None, CLAUDE: None}}
    budget = Budget(cap, Budget.spent_in(out))
    return asyncio.run(run_fix(items, omop_db, script.clients(), prices or {m: (10.0, 50.0) for m in MODELS}, out,
                               budget, proposers=MODELS, original_reviews=original, ids=ids)), budget


def test_parse_proposal_needs_both_questions_sql_and_a_boolean_order():
    from medbench.tasks.text2sql_llm_fix import parse_proposal
    ok = parse_proposal("Sure:\n```json\n" + _proposal("SELECT 1") + "\n```")
    assert ok["sql"] == "SELECT 1" and ok["order_matters"] is False and ok["change"] == "tie rule added"
    assert parse_proposal(json.dumps({"question_ko": "가", "question_en": "a", "sql": ""})) is None
    assert parse_proposal(json.dumps({"question_ko": "가", "question_en": "a", "sql": "SELECT 1",
                                      "order_matters": "yes"})) is None
    assert parse_proposal("no json here") is None
    assert parse_proposal(json.dumps({"question_ko": "가", "question_en": "a", "sql": "SELECT 1;"}))["sql"] == "SELECT 1"


def test_v0_fix_prompt_is_unchanged(omop_db, tmp_path):
    from test_text2sql_review import DRAFT as REVIEW_DRAFT

    from medbench.tasks.synthetic_snuh import connect_readonly
    from medbench.tasks.text2sql import load_questions
    from medbench.tasks.text2sql_llm_fix import _sha, render_fix_messages
    path = tmp_path / "d.yaml"
    path.write_text(REVIEW_DRAFT, encoding="utf-8")
    _, items = load_questions(path)
    messages = render_fix_messages(items[0], connect_readonly(omop_db), "SCHEMA-TEXT", "NOTE", [], 1)
    # 1857128ef9c8e4b6 until 2026-10-30, when the fixture item stopped quoting a real v0 item (public export); the
    # code gave that same hash for the old fixture before the hash here was updated
    assert _sha(messages) == "7bbcfc11e70fecdb"


def test_an_unanswerable_item_is_proposed_without_sql_and_an_answerable_one_with_sql():
    from medbench.tasks.text2sql_llm_fix import parse_proposal
    null = json.dumps({"question_ko": "가", "question_en": "a", "sql": None, "order_matters": False,
                       "why_hard": "no cost table", "change": "c"})
    assert parse_proposal(null) is None                                       # answerable items need SQL
    p = parse_proposal(null, answerable=False)
    assert p["sql"] is None and p["why_hard"] == "no cost table"
    assert parse_proposal(_proposal("SELECT 1"), answerable=False) is None    # it stays unanswerable


HARD_DRAFT = """status: "DRAFT"
prompt:
  id: text2sql-omop-hard-p0
questions:

  - id: h-001
    category: unanswerable
    difficulty: hard
    scope: group
    answerable: false
    question_ko: 각 방문의 진료비는 얼마인가?
    question_en: How much did each visit cost?
    sql: null
    why_hard: no cost table
    review: needs_edit
    review_note: "[gpt-6-astra] needs_edit: say which visits"

  - id: h-002
    category: clinical_terms
    difficulty: hard
    scope: group
    question_ko: 제2형 당뇨병 환자는 몇 명인가?
    question_en: How many patients have T2DM?
    sql: |
      SELECT count(DISTINCT person_id) FROM condition_occurrence WHERE condition_concept_id = 201826
    why_hard: T2DM must be mapped to its concept
    review: approved
"""


def test_an_unanswerable_hard_item_is_fixed_with_the_hard_prompt_and_written_back(omop_db, tmp_path):
    from medbench.tasks.text2sql import HARD_CONVENTIONS, HARD_PROMPT, load_questions
    from medbench.tasks.text2sql_llm_fix import Budget, apply_fixes, run_fix
    draft = tmp_path / "hard.yaml"
    draft.write_text(HARD_DRAFT, encoding="utf-8")
    _, items = load_questions(draft)
    proposal = json.dumps({"question_ko": "2024년 각 입원의 진료비 총액은?", "question_en": "What was the total bill of "
                           "each 2024 inpatient stay?", "sql": None, "order_matters": False,
                           "why_hard": "no cost or payer table; visit rows hold no amounts", "change": "scoped"})
    script = Script({("h-001", 1): proposal}, lambda model, user: _verdict("approve"))
    (o,) = asyncio.run(run_fix(items, omop_db, script.clients(), {m: (1.0, 1.0) for m in MODELS}, tmp_path / "fix",
                               Budget(10.0), proposers=MODELS, original_reviews={"h-001": "say which visits"},
                               spec=HARD_PROMPT))
    assert o.status == "approved" and o.revised.sql is None and not o.revised.answerable
    propose = [c for c in script.calls if c[0] == "propose"][0][4]
    assert HARD_CONVENTIONS in propose and "no cost table" in propose and "abstain" in propose
    review = [c for c in script.calls if c[0] == "review"][0][2]
    assert "Reference SQL: none" in review and "no cost or payer table" in review
    apply_fixes(draft, [o])
    _, items = load_questions(draft)
    assert items[0].question_en.startswith("What was the total bill") and items[0].sql is None
    assert items[0].why_hard == "no cost or payer table; visit rows hold no amounts" and items[0].review == "approved"
    assert items[1].review == "approved" and items[1].sql.startswith("SELECT count(DISTINCT")


def test_a_later_json_object_such_as_a_correction_supersedes_an_earlier_one():
    from medbench.tasks.text2sql_llm_fix import parse_json_object
    text = ('{"items": [1]}\n\nCorrection: the first one had a typo. The corrected answer:\n{"items": [2]}\n'
            'and {not json}')
    assert parse_json_object(text, lambda b: b.get("items")) == [2]
    fenced = 'Draft:\n```json\n{"items": [3]}\n```\nNotes {"other": 1}'
    assert parse_json_object(fenced, lambda b: b.get("items")) == [3]       # objects that are not accepted are skipped
    assert parse_json_object('{"a": "line\nbreak", "items": [4]}', lambda b: b.get("items")) == [4]
    assert parse_json_object("no json", lambda b: b) is None


def test_a_cached_answer_that_did_not_parse_is_parsed_again_before_asking(tmp_path):
    from medbench.tasks.text2sql_llm_fix import Asker, Budget
    client = Retrying(0, answer='{"ok": 1} {"ok": 2}')
    asker = Asker({"m/x": client}, {"m/x": (1.0, 1.0)}, tmp_path, Budget(10.0))
    messages = [{"role": "user", "content": "q"}]
    first = asyncio.run(asker.ask("m/x", "k", "q-1", messages, lambda t: None))      # an old parser
    assert first.status == "unparsed" and client.sent == 1
    again = Asker({"m/x": client}, {"m/x": (1.0, 1.0)}, tmp_path, Budget(10.0))
    from medbench.tasks.text2sql_llm_fix import parse_json_object
    answer = asyncio.run(again.ask("m/x", "k", "q-1", messages, lambda t: parse_json_object(t, lambda b: b)))
    assert answer.status == "ok" and answer.value == {"ok": 2} and client.sent == 1     # no new request


def test_a_hard_proposal_that_quotes_a_concept_name_goes_back_as_a_problem(omop_db, tmp_path):
    from medbench.tasks.text2sql import HARD_PROMPT, load_questions
    from medbench.tasks.text2sql_llm_fix import Budget, run_fix
    draft = tmp_path / "hard.yaml"
    draft.write_text(HARD_DRAFT.replace("review: approved", "review: needs_edit"), encoding="utf-8")
    _, items = load_questions(draft)
    sql = ("SELECT count(DISTINCT co.person_id) FROM condition_occurrence co JOIN concept c ON c.concept_id = "
           "co.condition_concept_id WHERE c.concept_name = 'Type 2 diabetes mellitus'")
    script = Script({("h-002", 1): _proposal(sql, en='How many patients have "Type 2 diabetes mellitus"?'),
                     ("h-002", 2): _proposal(sql, en="How many patients have T2DM?")},
                    lambda model, user: _verdict("approve"))
    outcomes = asyncio.run(run_fix(items, omop_db, script.clients(), {}, tmp_path / "fix", Budget(10.0),
                                   proposers=MODELS, original_reviews={}, ids=["h-002"], spec=HARD_PROMPT))
    (o,) = outcomes
    assert o.status == "approved" and o.round == 2
    assert "quotes concept names (Type 2 diabetes mellitus)" in o.rounds[0]["problem"]


def test_reference_rules_apply_to_a_proposed_sql(omop_db):
    from dataclasses import replace

    from medbench.tasks.synthetic_snuh import connect_readonly
    from medbench.tasks.text2sql import Question
    from medbench.tasks.text2sql_llm_fix import reference_problem
    con = connect_readonly(omop_db)
    base = Question(id="x", question_ko="가", question_en="a", sql="SELECT count(*) FROM person",
                    category="cohort_count", difficulty="easy")
    other = replace(base, id="y", sql="SELECT max(year_of_birth) FROM person")
    assert reference_problem(con, base, [other]) is None
    assert "syntax" in reference_problem(con, replace(base, sql="SELEC 1"), [])
    assert "empty result" in reference_problem(con, replace(base, sql="SELECT 1 FROM person WHERE 1 = 0"), [])
    assert "trivial" in reference_problem(con, replace(base, sql="SELECT 0"), [])
    assert "current date" in reference_problem(con, replace(base, sql="SELECT current_date"), [])
    assert "not_select" in reference_problem(con, replace(base, sql="CREATE TABLE t (a INT)"), [])
    assert "ORDER BY" in reference_problem(con, replace(base, order_matters=True), [])
    assert "too many rows" in reference_problem(con, replace(base, sql="SELECT * FROM range(2000)"), [])
    assert "y" in reference_problem(con, replace(base, sql=other.sql), [other])   # duplicates another item


def test_budget_is_a_hard_cap_on_spend_and_requests_in_flight(tmp_path):
    from medbench.tasks.text2sql_llm_fix import Budget, worst_case_usd
    (tmp_path / "decide").mkdir()
    (tmp_path / "decide" / "m.jsonl").write_text(json.dumps({"cost_usd": 1.5}) + "\n" + json.dumps({"cost_usd": 0.5}))
    assert Budget.spent_in(tmp_path) == pytest.approx(2.0)
    budget = Budget(3.0, Budget.spent_in(tmp_path))
    assert budget.try_reserve(0.6) and budget.try_reserve(0.3)
    assert not budget.try_reserve(0.2) and budget.stopped                    # 2.0 + 0.9 in flight + 0.2 > 3.0
    budget.settle(0.6, 0.1)
    assert budget.spent_usd == pytest.approx(2.1) and budget.try_reserve(0.5)
    messages = [{"role": "user", "content": "x" * 1000}]
    assert worst_case_usd(messages, 100, (10.0, 50.0)) >= (1000 * 10 + 100 * 50) / 1e6


class Retrying:
    """A route client that sends `fail` failed attempts (each one asked of the budget first) before an answer."""

    def __init__(self, fail, *, billed=True, answer='{"ok": 1}'):
        self.fail, self.billed, self.answer, self.sent = fail, billed, answer, 0

    async def chat(self, model_id, messages, *, max_tokens, temperature=0.0, extra=None, before_attempt=None):
        for n in range(self.fail + 1):
            if before_attempt is not None and not await before_attempt(n):
                return ChatResult("", None, 0, 0, "budget", transient=True, attempts=self.sent,
                                  failed_attempts=self.sent if self.billed else 0)
            self.sent += 1
        return ChatResult(self.answer, None, 1000, 100, None, cost_usd=0.02, attempts=self.sent,
                          failed_attempts=self.fail if self.billed else 0)


def _ask(tmp_path, client, cap, model="m/x", messages=None):
    from medbench.tasks.text2sql_llm_fix import Asker, Budget
    budget = Budget(cap, Budget.spent_in(tmp_path))
    asker = Asker({model: client}, {model: (10.0, 50.0)}, tmp_path / "c", budget, max_tokens=1000)
    answer = asyncio.run(asker.ask(model, "k", "q-1", messages or [{"role": "user", "content": "x" * 800}],
                                   lambda t: json.loads(t)))
    return answer, budget


def test_failed_attempts_that_may_have_been_billed_are_charged_their_estimate_and_kept(tmp_path):
    from medbench.tasks.text2sql_llm_fix import Budget, worst_case_usd
    worst = worst_case_usd([{"role": "user", "content": "x" * 800}], 1000, (10.0, 50.0))
    answer, budget = _ask(tmp_path, Retrying(2), cap=10.0)
    assert answer.status == "ok" and budget.reserved_usd == pytest.approx(0)
    assert budget.spent_usd == pytest.approx(0.02 + 2 * worst)               # two retries charged at their estimate
    record = json.loads((tmp_path / "c" / "m__x.jsonl").read_text().splitlines()[0])
    assert record["attempts"] == 3 and record["failed_attempts"] == 2
    assert record["failed_attempts_usd"] == pytest.approx(2 * worst, abs=1e-6) and record["cost_usd"] == 0.02
    assert Budget.spent_in(tmp_path) == pytest.approx(0.02 + 2 * worst, abs=1e-5)   # a rerun counts them too
    _, budget = _ask(tmp_path / "unbilled", Retrying(2, billed=False), cap=10.0)
    assert budget.spent_usd == pytest.approx(0.02)                           # e.g. 429s: nothing ran, nothing charged


def test_a_retry_that_does_not_fit_the_budget_is_not_sent(tmp_path):
    from medbench.tasks.text2sql_llm_fix import worst_case_usd
    worst = worst_case_usd([{"role": "user", "content": "x" * 800}], 1000, (10.0, 50.0))
    client = Retrying(5)
    answer, budget = _ask(tmp_path, client, cap=2.5 * worst)                 # room for two attempts
    assert client.sent == 2 and answer.status == "budget" and budget.stopped
    assert budget.spent_usd == pytest.approx(2 * worst) and budget.reserved_usd == pytest.approx(0)


def test_anthropic_requests_mark_the_shared_prefix_for_prompt_caching(tmp_path):
    from medbench.tasks.text2sql_llm_fix import Asker, Budget
    seen = {}

    class Client:
        async def chat(self, model_id, messages, *, max_tokens, temperature=0.0, extra=None, before_attempt=None):
            seen[model_id] = messages
            return ChatResult('{"ok": 1}', None, 10, 10, None, cost_usd=0.0)
    messages = [{"role": "system", "content": "rubric"},
                {"role": "user", "content": "Database schema: big\n\nNotes\n\nQuestion id: q-1\nKorean: ..."}]
    models = ["anthropic/claude-opus-5.5", "openai/gpt-6-astra"]
    asker = Asker({m: Client() for m in models}, {}, tmp_path, Budget(10.0), cache_split="\n\nQuestion id:")
    for m in models:
        asyncio.run(asker.ask(m, "k", "q-1", messages, lambda t: json.loads(t)))
    parts = seen["anthropic/claude-opus-5.5"][1]["content"]
    assert parts[0] == {"type": "text", "text": "Database schema: big\n\nNotes", "cache_control": {"type": "ephemeral"}}
    assert parts[1] == {"type": "text", "text": "\n\nQuestion id: q-1\nKorean: ..."}
    assert seen["openai/gpt-6-astra"] == messages and messages[1]["content"].startswith("Database schema")
    from medbench.tasks.text2sql_llm_fix import _sha
    record = json.loads((tmp_path / "anthropic__claude-opus-5.5.jsonl").read_text())
    assert record["prompt_sha"] == _sha(messages)                            # cached by the logical prompt


def test_an_invalid_sql_goes_back_as_feedback_and_the_next_proposer_fixes_it(draft, omop_db, tmp_path):
    from medbench.tasks.text2sql_llm_fix import run_fix  # noqa: F401  (import check)
    proposals = {("q-002", 1): _proposal("SELECT visit_source_value FROM visit_occurence"),   # typo: fails
                 ("q-002", 2): _proposal(GOOD_002, ko="방문 수가 가장 많은 진료과(동률이면 모두)와 방문 수는?",
                                         en="Which department(s) have the most visits (all ties), and how many?")}
    script = Script(proposals, lambda model, user: _verdict("approve"))
    outcomes, budget = _run(draft, omop_db, script, tmp_path / "fix", ids=["q-002"])
    (o,) = outcomes
    assert o.status == "approved" and o.round == 2 and o.proposer == GPT
    assert o.revised.sql == GOOD_002 and o.revised.question_en.startswith("Which department(s)")
    proposes = [c for c in script.calls if c[0] == "propose"]
    assert [(c[1], c[3]) for c in proposes] == [(CLAUDE, 1), (GPT, 2)]
    assert "visit_occurence" in proposes[1][4] and "execution_error" in proposes[1][4]   # round 1's failure fed back
    assert "ties at the top" in proposes[0][4] and "nondeterministic" in proposes[0][4]  # every reviewer's reason
    assert "author note stays" in proposes[0][4]
    reviews = [c for c in script.calls if c[0] == "review"]
    assert len(reviews) == 3 and all("Which department(s)" in c[2] for c in reviews)   # only the valid one re-reviewed
    assert budget.spent_usd == pytest.approx(5 * 0.01)                                  # the route's billed cost


def test_three_rounds_without_approval_reject_the_item_and_proposers_rotate(draft, omop_db, tmp_path):
    proposals = {("q-003", r): _proposal(f"SELECT count(*) FROM person WHERE year_of_birth > {1950 + r}",
                                         ko=f"{1950 + r}년 이후", en=f"after {1950 + r}") for r in (1, 2, 3)}

    def reviews(model, user):
        return _verdict("needs_edit", reason="still says after", fix="say 'later than'") if model == GEMINI \
            else _verdict("approve")
    script = Script(proposals, reviews)
    (o,), _ = _run(draft, omop_db, script, tmp_path / "fix", ids=["q-003"])
    assert o.status == "rejected" and o.revised is None and len(o.rounds) == 3
    assert [c[1] for c in script.calls if c[0] == "propose"] == [CLAUDE, GPT, GEMINI]
    third = [c for c in script.calls if c[0] == "propose"][2][4]
    assert "still says after" in third and "after 1952" in third              # earlier rounds and their reviews
    from medbench.tasks.text2sql_llm_fix import fix_note
    assert "3 rounds" in fix_note(o) and "still says after" in fix_note(o)


def test_answers_are_cached_and_a_rerun_asks_nothing(draft, omop_db, tmp_path):
    proposals = {("q-002", 1): _proposal(GOOD_002), ("q-003", 1): _proposal("SELECT count(*) FROM person "
                                                                            "WHERE year_of_birth >= 1961")}
    script = Script(proposals, lambda model, user: _verdict("approve"))
    first, _ = _run(draft, omop_db, script, tmp_path / "fix")
    assert [o.status for o in first] == ["approved", "approved"] and len(script.calls) == 8
    again, budget = _run(draft, omop_db, script, tmp_path / "fix")
    assert len(script.calls) == 8 and [o.status for o in again] == ["approved", "approved"]
    assert budget.spent_usd == pytest.approx(8 * 0.01)                        # earlier spend counts against the cap


def test_the_budget_stops_new_requests_and_leaves_items_incomplete(draft, omop_db, tmp_path):
    proposals = {("q-002", 1): _proposal(GOOD_002), ("q-003", 1): _proposal("SELECT 1")}
    script = Script(proposals, lambda model, user: _verdict("approve"))
    outcomes, budget = _run(draft, omop_db, script, tmp_path / "fix", cap=0.05)   # worst case of one call is ~0.4
    assert {o.status for o in outcomes} == {"incomplete"} and not script.calls and budget.stopped


def test_re_reviews_of_a_round_can_go_several_items_per_request(draft, omop_db, tmp_path):
    from medbench.tasks.text2sql import load_questions
    from medbench.tasks.text2sql_llm_fix import Budget, run_fix
    proposals = {("q-002", 1): _proposal(GOOD_002, en="Which department(s) have the most visits (ties)?"),
                 ("q-003", 1): _proposal("SELECT count(*) FROM person WHERE year_of_birth >= 1961",
                                         en="How many patients were born in 1961 or later?"),
                 ("q-003", 2): _proposal("SELECT count(*) FROM person WHERE year_of_birth >= 1962",
                                         en="How many patients were born in 1962 or later?")}
    batches = []

    def reviews(model, user):
        ids = [line.split("Question id: ", 1)[1] for line in user.splitlines() if line.startswith("Question id: ")]
        if len(ids) == 1:
            return _verdict("approve")
        batches.append((model, ids))
        rows = [{"id": "q-002", **json.loads(_verdict("approve"))},
                {"id": "q-003", **json.loads(_verdict("needs_edit" if model == GPT else "approve",
                                                      reason="1961 or later is odd"))}]
        return json.dumps({"reviews": rows if model != GEMINI else rows[:1]})   # Gemini forgets q-003
    script = Script(proposals, reviews)
    _, items = load_questions(draft)
    original = {"q-002": "ties", "q-003": "after?"}
    outcomes = asyncio.run(run_fix(items, omop_db, script.clients(), {m: (10.0, 50.0) for m in MODELS},
                                   tmp_path / "fix", Budget(30.0), proposers=MODELS, original_reviews=original,
                                   review_batch=5))
    assert [(o.id, o.status, o.round) for o in outcomes] == [("q-002", "approved", 1), ("q-003", "approved", 2)]
    assert sorted(batches) == sorted((m, ["q-002", "q-003"]) for m in MODELS)      # round 1: one request each
    singles = [c for c in script.calls if c[0] == "review" and c[2].count("Question id: ") == 1]
    assert len(singles) == 1 + 3                     # Gemini's missing verdict, then round 2 has a single item
    assert "1961 or later is odd" in [c for c in script.calls if c[0] == "propose"][-1][4]


def test_apply_writes_only_the_fixed_items_lines(draft, omop_db, tmp_path):
    from medbench.tasks.text2sql import load_questions
    from medbench.tasks.text2sql_llm_fix import apply_fixes
    proposals = {("q-002", 1): _proposal(GOOD_002, ko='방문 수가 가장 많은 진료과는? "동률" 포함',
                                         en="Which department(s) have the most visits: all ties?",
                                         order_matters=False, change="returns all tied departments"),
                 **{("q-003", r): _proposal("SELECT count(*) FROM person WHERE year_of_birth >= 1961")
                    for r in (1, 2, 3)}}
    script = Script(proposals, lambda model, user: _verdict("reject", reason="needs a birth-year range")
                    if "1961" in user and model != CLAUDE else _verdict("approve"))
    outcomes, _ = _run(draft, omop_db, script, tmp_path / "fix")
    before = draft.read_text(encoding="utf-8")
    counts = apply_fixes(draft, outcomes)
    assert counts == {"approved": 1, "rejected": 1}
    text = draft.read_text(encoding="utf-8")
    assert "# DRAFT header comment that must survive" in text and "# --- another section" in text
    assert before.split("  - id: q-002")[0] == text.split("  - id: q-002")[0]          # q-001 untouched
    _, items = load_questions(draft)
    q2, q3 = items[1], items[2]
    assert q2.sql == GOOD_002 and q2.review == "approved" and q2.question_ko.endswith('"동률" 포함')
    assert q2.question_en == "Which department(s) have the most visits: all ties?" and q2.note == "author note stays"
    assert q3.review == "rejected" and q3.sql == "SELECT count(*) FROM person WHERE year_of_birth > 1960"
    import yaml
    raw = {q["id"]: q for q in yaml.safe_load(text)["questions"]}
    assert "round 1" in raw["q-002"]["review_note"] and "claude-opus-5.5" in raw["q-002"]["review_note"]
    assert "returns all tied departments" in raw["q-002"]["review_note"]
    assert "needs a birth-year range" in raw["q-003"]["review_note"] and "dropped" in raw["q-003"]["review_note"]


def test_apply_sets_order_matters_when_the_revision_asks_for_an_order(draft, omop_db, tmp_path):
    from medbench.tasks.text2sql import load_questions
    from medbench.tasks.text2sql_llm_fix import apply_fixes
    sql = "SELECT visit_source_value, count(*) AS n FROM visit_occurrence GROUP BY 1 ORDER BY n DESC, 1"
    script = Script({("q-002", 1): _proposal(sql, order_matters=True)}, lambda model, user: _verdict("approve"))
    outcomes, _ = _run(draft, omop_db, script, tmp_path / "fix", ids=["q-002"])
    apply_fixes(draft, outcomes)
    _, items = load_questions(draft)
    assert items[1].order_matters is True and items[1].sql == sql and items[2].review == "needs_edit"


def test_a_concept_the_reference_filters_by_id_may_not_be_named_in_quotes_or_parentheses(omop_db):
    """Like syn-hard-041: the reference filters by concept_id only and the question glosses the clinical terms with
    the vocabulary names in parentheses. Ids compared with a *_concept_id column (= N, IN (...)) resolve to names."""
    from medbench.tasks.synthetic_snuh import connect_readonly
    from medbench.tasks.text2sql import Question
    from dataclasses import replace

    from medbench.tasks.text2sql_llm_fix import quoted_concepts, quoted_problem
    con = connect_readonly(omop_db)
    sql = ("SELECT c.concept_name, count(*) FROM measurement m JOIN concept c ON c.concept_id = "
           "m.measurement_concept_id\nWHERE m.measurement_concept_id IN (3004410,\n 9999999)\n"
           "  AND m.person_id IN (SELECT person_id FROM condition_occurrence WHERE condition_concept_id = 201826)\n"
           "GROUP BY 1")

    def item(ko, en):
        return Question(id="h", question_ko=ko, question_en=en, sql=sql, category="data_quality", difficulty="hard")
    glossed = item("당화혈색소(Hemoglobin A1c/Hemoglobin.total in Blood) 결과 수는?",
                   "How many HbA1c results do patients with type 2 diabetes mellitus have?")
    assert quoted_concepts(con, glossed) == ["Hemoglobin A1c/Hemoglobin.total in Blood"]
    assert "Hemoglobin A1c/Hemoglobin.total in Blood" in quoted_problem(con, glossed)
    lower = item("당화혈색소 결과 수는?", "How many HbA1c (hemoglobin a1c/hemoglobin.total in blood) results?")
    assert quoted_concepts(con, lower) == ["Hemoglobin A1c/Hemoglobin.total in Blood"]     # case-insensitive
    quoted = item("'Type 2 diabetes mellitus' 환자의 당화혈색소 결과 수는?", "How many HbA1c results in T2DM?")
    assert quoted_concepts(con, quoted) == ["Type 2 diabetes mellitus"]
    inside = item("당화혈색소 결과 수는?", "How many HbA1c results (patients with type 2 diabetes mellitus only)?")
    assert quoted_concepts(con, inside) == ["Type 2 diabetes mellitus"]                     # anywhere in the brackets
    plain = item("제2형 당뇨병 환자의 당화혈색소 결과 수는?",
                 "How many HbA1c results do patients with type 2 diabetes mellitus have?")
    assert quoted_concepts(con, plain) == [] and quoted_problem(con, plain) is None        # a plain clinical word
    men = Question(id="m", question_ko="남성 환자 수를 'male_count'로 반환하라.", question_en="How many (female) patients?",
                   sql="SELECT count(*) AS male_count FROM person WHERE gender_concept_id = 8507",
                   category="data_quality", difficulty="hard")
    assert quoted_concepts(con, men) == []                     # MALE only inside an identifier or a longer word
    assert quoted_concepts(con, replace(men, question_ko="남성(male) 환자 수는?")) == ["MALE"]


def test_the_concept_ids_of_a_reference_are_read_from_comparisons_with_concept_id_columns():
    from medbench.tasks.text2sql_llm_fix import concept_ids
    sql = ("SELECT * FROM x WHERE a.condition_concept_id = 201826 AND drug_concept_id IN (1, 2,\n 3) "
           "AND visit_concept_id NOT IN (9201) AND c.concept_id = 7 AND person_id = 5 AND quantity IN (8, 9) "
           "AND measurement_concept_id IN (SELECT 4) AND gender_concept_id=8507")
    assert concept_ids(sql) == {201826, 1, 2, 3, 9201, 7, 8507}


class Raising:
    """A route client whose request fails with an exception after it was sent, or hangs until cancelled."""

    def __init__(self, exc=None):
        self.exc, self.started = exc, asyncio.Event()

    async def chat(self, model_id, messages, *, max_tokens, temperature=0.0, extra=None, before_attempt=None):
        assert await before_attempt(0)
        self.started.set()
        if self.exc is not None:
            raise self.exc
        await asyncio.Event().wait()


def test_an_attempt_that_raises_is_charged_its_estimate_and_written_to_the_cache(tmp_path):
    from medbench.tasks.text2sql_llm_fix import Budget, worst_case_usd
    messages = [{"role": "user", "content": "x" * 800}]
    worst = worst_case_usd(messages, 1000, (10.0, 50.0))
    with pytest.raises(RuntimeError):
        _ask(tmp_path, Raising(RuntimeError("connection reset")), cap=10.0, messages=messages)
    (record,) = [json.loads(line) for line in (tmp_path / "c" / "m__x.jsonl").read_text().splitlines()]
    assert record["error"] == "exception_RuntimeError" and record["answer"] is None
    assert record["failed_attempts"] == 1 and record["failed_attempts_usd"] == pytest.approx(worst, abs=1e-6)
    assert Budget.spent_in(tmp_path) == pytest.approx(worst, abs=1e-5)       # the next run counts it


def test_a_cancelled_request_is_charged_its_estimate_and_written_to_the_cache(tmp_path):
    from medbench.tasks.text2sql_llm_fix import Asker, Budget, worst_case_usd
    messages = [{"role": "user", "content": "x" * 800}]
    worst = worst_case_usd(messages, 1000, (10.0, 50.0))
    client = Raising()
    budget = Budget(10.0)
    asker = Asker({"m/x": client}, {"m/x": (10.0, 50.0)}, tmp_path / "c", budget, max_tokens=1000)

    async def main():
        task = asyncio.create_task(asker.ask("m/x", "k", "q-1", messages, json.loads))
        await client.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(main())
    assert budget.spent_usd == pytest.approx(worst) and budget.reserved_usd == pytest.approx(0)
    (record,) = [json.loads(line) for line in (tmp_path / "c" / "m__x.jsonl").read_text().splitlines()]
    assert record["error"] == "exception_CancelledError" and record["failed_attempts_usd"] == pytest.approx(worst,
                                                                                                          abs=1e-6)
    assert Budget.spent_in(tmp_path) == pytest.approx(worst, abs=1e-5)
    again = Asker({"m/x": Retrying(0)}, {"m/x": (10.0, 50.0)}, tmp_path / "c", Budget(10.0), max_tokens=1000)
    assert asyncio.run(again.ask("m/x", "k", "q-1", messages, json.loads)).status == "ok"   # asked again next run


def test_the_fix_report_keeps_earlier_runs_and_an_item_fixed_again_keeps_its_earlier_fix(tmp_path):
    from medbench.tasks.text2sql_llm_fix import FixOutcome, write_report
    path = tmp_path / "fix-report.json"
    write_report(path, [FixOutcome("a", "approved", 2, GPT), FixOutcome("b", "approved", 1, CLAUDE)])
    write_report(path, [FixOutcome("b", "incomplete")])                     # a resumable run: not an earlier fix
    write_report(path, [FixOutcome("b", "approved", 3, GEMINI), FixOutcome("c", "approved", 1, CLAUDE)])
    rows = {r["id"]: r for r in json.loads(path.read_text())}
    assert list(rows) == ["a", "b", "c"] and rows["a"]["round"] == 2 and "earlier" not in rows["a"]
    assert rows["b"]["round"] == 3 and [e["round"] for e in rows["b"]["earlier"]] == [1]
