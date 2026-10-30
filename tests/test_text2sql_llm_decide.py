"""The Text-to-SQL draft's open decisions are voted on by the three frontier reviewers (owner, 2026-10-30): each
model picks one explicit option with a reason; two of three win; with no majority v0's current behaviour stays."""
import asyncio
import json
from pathlib import Path

import pytest
import yaml

from medbench.llm import ChatResult

CLAUDE, GPT, GEMINI = "anthropic/claude-opus-5.5", "openai/gpt-6-astra", "google/gemini-4-argon"
MODELS = [GPT, GEMINI, CLAUDE]
REPO = Path(__file__).resolve().parents[1]
# The question files are released on KHDP, and the SOP and the freeze evidence are documents of the deployment: none
# of them is in the public code repository, so the tests that read them skip there.
needs_drafts = pytest.mark.skipif(not (REPO / "registry/drafts/synthetic-snuh-text2sql-v0.yaml").exists(),
                                  reason="the question files are released on KHDP, not with the code")
needs_docs = pytest.mark.skipif(not (REPO / "docs/sop/benchmarks-and-models.md").exists(),
                                reason="the SOP and the freeze evidence are not part of the published code")

DRAFT = """# DRAFT header comment that must survive
status: "DRAFT"
protocol:
  metric: execution accuracy (EX)
prompt:
  id: text2sql-omop-p0

open_decisions:
  - >-
    Korean "내원". The model read 내원 as outpatient-only. Either add a line to the database
    note or reword the items.
  - Unanswerable questions and abstention scoring; not in v0.
  - Split into dev and test, and whether to keep both languages as separate scored instances.

questions:

  - id: q-001
    category: patient_lookup
    difficulty: easy
    scope: individual
    question_ko: 환자 SYN-000001은 몇 년생인가?
    question_en: In which year was patient SYN-000001 born?
    sql: |
      SELECT year_of_birth AS birth_year FROM person WHERE person_source_value = 'SYN-000001'
    review: approved
"""


def _decisions():
    from medbench.tasks.text2sql_llm_decide import Decision, Option
    return [
        Decision("naewon", 'Korean "내원"', "How should Korean 내원 be read?", "keep", [
            Option("keep", "Leave everything as it is.", "none"),
            Option("note", "Add a line to the database note.", "implement"),
            Option("reword", "Reword the two items.", "implement")]),
        Decision("unanswerable", "Unanswerable questions", "Add unanswerable questions?", "not_in_v0", [
            Option("not_in_v0", "No unanswerable questions in v0.", "none"),
            Option("later", "Add them in a later version.", "next_work")]),
        Decision("split", "Split into dev and test", "Hold out a dev set?", "none", [
            Option("none", "No split.", "none"), Option("dev_test", "A small dev set.", "implement")]),
        Decision("languages", "Split into dev and test", "How are languages reported?", "both_scored", [
            Option("both_scored", "Both scored.", "none"), Option("separate", "Separately.", "implement")]),
    ]


class Voters:
    def __init__(self, votes):
        self.votes, self.calls = votes, []

    def clients(self):
        voters = self

        class Client:
            async def chat(self, model_id, messages, *, max_tokens, temperature=0.0, extra=None, before_attempt=None):
                user = messages[-1]["content"]
                key = user.split("Decision key: ", 1)[1].split("\n", 1)[0]
                voters.calls.append((model_id, key, user))
                content = voters.votes[key][model_id]
                return ChatResult(content=content, reasoning=None, tokens_in=2000, tokens_out=300, error=None,
                                  cost_usd=0.02)
        return {m: Client() for m in MODELS}


def _vote(choice, reason="r"):
    return json.dumps({"choice": choice, "reason": f"{reason} {choice}"})


@pytest.fixture
def draft(tmp_path):
    p = tmp_path / "draft.yaml"
    p.write_text(DRAFT, encoding="utf-8")
    return p


def test_parse_choice_accepts_only_offered_options():
    from medbench.tasks.text2sql_llm_decide import parse_choice
    assert parse_choice("```json\n" + _vote("note") + "\n```", ["keep", "note"]) == {"choice": "note",
                                                                                      "reason": "r note"}
    assert parse_choice(_vote("something_else"), ["keep", "note"]) is None
    assert parse_choice("I would add a note.", ["keep", "note"]) is None


def test_two_of_three_win_and_no_majority_keeps_v0():
    from medbench.tasks.text2sql_llm_decide import tally
    def v(c):
        return {"choice": c, "reason": ""}
    assert tally({"a": v("note"), "b": v("note"), "c": v("keep")}, "keep") == ("note", True)
    assert tally({"a": v("note"), "b": v("reword"), "c": v("keep")}, "keep") == ("keep", False)
    assert tally({"a": v("reword"), "b": v("reword"), "c": None}, "keep") == ("reword", True)
    assert tally({"a": v("reword"), "b": None, "c": None}, "keep") == ("keep", False)


def test_run_decide_asks_every_model_about_every_decision_with_protocol_options_and_facts(draft, tmp_path):
    from medbench.tasks.text2sql_llm_decide import run_decide
    from medbench.tasks.text2sql_llm_fix import Budget
    votes = {"naewon": {GPT: _vote("note"), GEMINI: _vote("note"), CLAUDE: _vote("reword")},
             "unanswerable": {GPT: _vote("later"), GEMINI: _vote("not_in_v0"), CLAUDE: _vote("later")},
             "split": {GPT: _vote("none"), GEMINI: _vote("dev_test"), CLAUDE: "no idea"},
             "languages": {GPT: _vote("separate"), GEMINI: _vote("separate"), CLAUDE: _vote("both_scored")}}
    voters = Voters(votes)
    out = tmp_path / "decide"
    results = asyncio.run(run_decide(draft, voters.clients(), {m: (10.0, 50.0) for m in MODELS}, out,
                                     Budget(30.0), facts="A free model scored 0.994 (306 of 308).",
                                     decisions=_decisions()))
    assert len(voters.calls) == 12
    user = next(u for m, k, u in voters.calls if k == "naewon")
    assert "execution accuracy (EX)" in user and "0.994" in user and "Add a line to the database note." in user
    assert "Unanswerable questions" in user                         # the other open decisions are context
    by = {r["key"]: r for r in results}
    assert by["naewon"]["outcome"] == "note" and by["naewon"]["majority"]
    assert by["naewon"]["votes"][CLAUDE] == {"choice": "reword", "reason": "r reword"}
    assert by["unanswerable"]["outcome"] == "later" and by["unanswerable"]["kind"] == "next_work"
    assert by["split"]["outcome"] == "none" and not by["split"]["majority"]   # one vote unparsed, no majority
    assert by["split"]["votes"][CLAUDE] is None
    asyncio.run(run_decide(draft, voters.clients(), {m: (10.0, 50.0) for m in MODELS}, out, Budget(30.0),
                           facts="A free model scored 0.994 (306 of 308).", decisions=_decisions()))
    assert len(voters.calls) == 13                                   # only the unparsed answer is asked again


def test_write_decisions_records_votes_and_removes_resolved_entries(draft, tmp_path):
    from medbench.tasks.text2sql import load_questions
    from medbench.tasks.text2sql_llm_decide import run_decide, write_decisions
    from medbench.tasks.text2sql_llm_fix import Budget
    votes = {"naewon": {GPT: _vote("note"), GEMINI: _vote("note"), CLAUDE: _vote("reword")},
             "unanswerable": {GPT: _vote("later"), GEMINI: _vote("later"), CLAUDE: _vote("later")},
             "split": {GPT: _vote("none"), GEMINI: _vote("none"), CLAUDE: _vote("none")}}
    decisions = _decisions()[:3]                                      # "languages" is not voted on
    results = asyncio.run(run_decide(draft, Voters(votes).clients(), {m: (0, 0) for m in MODELS},
                                     tmp_path / "decide", Budget(30.0), facts="f", decisions=decisions))
    write_decisions(draft, results, table=_decisions(), date="2026-10-30")
    text = draft.read_text(encoding="utf-8")
    assert "# DRAFT header comment that must survive" in text
    meta, items = load_questions(draft)
    assert [i.id for i in items] == ["q-001"]
    # the split entry also holds the languages decision, which is still open
    split = "Split into dev and test, and whether to keep both languages as separate scored instances."
    assert meta["open_decisions"] == [split]
    recorded = {d["key"]: d for d in meta["decisions"]}
    assert set(recorded) == {"naewon", "unanswerable", "split"}
    naewon = recorded["naewon"]
    assert naewon["outcome"] == "note" and naewon["majority"] is True and naewon["date"] == "2026-10-30"
    assert naewon["options"]["reword"] == "Reword the two items."
    assert naewon["votes"]["anthropic/claude-opus-5.5"] == {"choice": "reword", "reason": "r reword"}
    assert recorded["unanswerable"]["status"] == "next work"
    assert naewon["decision"] == "How should Korean 내원 be read?"
    assert naewon["background"].startswith('Korean "내원". The model read 내원 as outpatient-only.')   # the evidence


def test_write_decisions_is_idempotent_and_keeps_earlier_records(draft, tmp_path):
    from medbench.tasks.text2sql import load_questions
    from medbench.tasks.text2sql_llm_decide import run_decide, write_decisions
    from medbench.tasks.text2sql_llm_fix import Budget
    votes = {"naewon": {m: _vote("keep") for m in MODELS}, "unanswerable": {m: _vote("later") for m in MODELS}}
    first, second = _decisions()[:1], _decisions()[1:2]
    for decisions in (first, second, second):
        results = asyncio.run(run_decide(draft, Voters(votes).clients(), {m: (0, 0) for m in MODELS},
                                         tmp_path / "decide", Budget(30.0), facts="f", decisions=decisions))
        write_decisions(draft, results, table=_decisions(), date="2026-10-30")
    meta, _ = load_questions(draft)
    assert [d["key"] for d in meta["decisions"]] == ["naewon", "unanswerable"]
    assert len(meta["open_decisions"]) == 1


def test_facts_give_the_dry_run_score_and_the_failed_questions(draft):
    from medbench.tasks.text2sql import load_questions
    from medbench.tasks.text2sql_llm_decide import facts_text
    _, items = load_questions(draft)
    summary = {"model": "free-model", "execution_accuracy": {"value": 306 / 308, "n": 308},
               "by_language": {"ko": {"value": 0.987}, "en": {"value": 1.0}}, "by_difficulty": {}}
    text = facts_text(items, summary, failures=["q-001-ko"])
    assert "0.994 (306 of 308" in text and "ko 0.987" in text and "approved 1" in text
    assert "q-001-ko: 환자 SYN-000001은 몇 년생인가?" in text


@needs_drafts
def test_every_decision_in_the_table_matches_the_real_draft():
    from medbench.tasks.text2sql_llm_decide import DECISIONS, FREEZE
    doc = yaml.safe_load((REPO / "registry/drafts/synthetic-snuh-text2sql-v0.yaml").read_text(encoding="utf-8"))
    entries = [" ".join(e.split()) for e in doc.get("open_decisions") or []]
    recorded = {d["key"] for d in doc.get("decisions") or []}
    for d in DECISIONS:
        assert d.current in {o.id for o in d.options} and {o.kind for o in d.options} <= {"none", "implement",
                                                                                           "next_work"}
        assert d.key in recorded or sum(e.startswith(d.match) for e in entries) == 1, d.key
    assert all(any(e.startswith(d.match) for d in [*DECISIONS, *FREEZE.table]) for e in entries)  # none left out


def test_v0_decide_prompt_is_unchanged_so_its_cached_votes_stay_valid(draft):
    from medbench.tasks.text2sql import load_questions
    from medbench.tasks.text2sql_llm_decide import _entries, render_decide_messages
    from medbench.tasks.text2sql_llm_fix import _sha
    meta, _ = load_questions(draft)
    assert _sha(render_decide_messages(_decisions()[0], meta, "FACTS", _entries(meta))) == "f9132f85c4238751"


@needs_drafts
def test_every_hard_decision_matches_the_hard_draft():
    from medbench.tasks.text2sql_llm_decide import FREEZE, HARD
    doc = yaml.safe_load((REPO / "registry/drafts/synthetic-snuh-text2sql-hard-v0.yaml").read_text(encoding="utf-8"))
    entries = [" ".join(e.split()) for e in doc.get("open_decisions") or []]
    recorded = {d["key"] for d in doc.get("decisions") or []}
    assert {d.key for d in HARD.table} == {"headline_metric", "output_limit", "review_batch", "second_round"}
    for d in HARD.table:
        assert d.current in {o.id for o in d.options} and {o.kind for o in d.options} <= set(HARD.status)
        assert d.key in recorded or sum(e.startswith(d.match) for e in entries) == 1, d.key
    assert all(any(e.startswith(d.match) for d in [*HARD.table, *FREEZE.table]) for e in entries)
    second = next(d for d in HARD.table if d.key == "second_round")
    assert "$25" in second.question and "left" in second.question                    # the budget left is stated
    assert any(o.kind == "needs_budget" for o in second.options)


def test_the_harder_sets_vote_uses_its_note_and_records_its_own_statuses(tmp_path):
    from medbench.tasks.text2sql import CONVENTIONS, HARD_CONVENTIONS, load_questions
    from medbench.tasks.text2sql_llm_decide import HARD, Decision, Option, run_decide, write_decisions
    from medbench.tasks.text2sql_llm_fix import Budget
    draft = tmp_path / "hard.yaml"
    draft.write_text(DRAFT.replace("id: text2sql-omop-p0", "id: text2sql-omop-hard-p0"), encoding="utf-8")
    table = [Decision("naewon", 'Korean "내원"', "Round two?", "no", [
        Option("no", "No.", "none"), Option("now", "Now, over the cap.", "needs_budget")])]
    votes = {"naewon": {GPT: _vote("now"), GEMINI: _vote("now"), CLAUDE: _vote("no")}}
    voters = Voters(votes)
    results = asyncio.run(run_decide(draft, voters.clients(), {m: (0, 0) for m in MODELS}, tmp_path / "decide",
                                     Budget(30.0), facts="f", decisions=table, setup=HARD))
    user = voters.calls[0][2]
    assert HARD_CONVENTIONS in user and CONVENTIONS not in user and "wrote, reviewed and fixed" in user
    assert "[the draft's current behaviour]" in user and "v0's current behaviour" not in user
    assert results[0]["outcome"] == "now"
    write_decisions(draft, results, table=table, date="2026-10-30", setup=HARD)
    meta, _ = load_questions(draft)
    (record,) = meta["decisions"]
    assert record["status"] == HARD.status["needs_budget"] and "budget approval" in record["status"]
    assert "v0's current behaviour" not in draft.read_text(encoding="utf-8")


def test_hard_facts_give_each_runs_scores_answers_without_sql_and_right_answers_with_sql():
    from medbench.tasks.text2sql import Question
    from medbench.tasks.text2sql_llm_decide import hard_facts_text
    items = [Question("h-1", "가", "a", "SELECT 1", "clinical_terms", "hard", review="approved"),
             Question("h-2", "나", "b", None, "unanswerable", "hard", answerable=False, review="approved")]
    summary = {"model": "free", "execution_accuracy": {"value": 0.5, "n": 2, "ci_low": 0.1, "ci_high": 0.9},
               "reliability": {"RS(0)": 0.5, "RS(5)": -1.0, "RS(N)": -2.0, "N": 4},
               "abstention": {"unanswerable_abstained": {"value": 0.5, "n": 2},
                              "answerable_abstained": {"value": 0.0, "n": 2}},
               "by_category": {"clinical_terms": {"value": 0.5, "n": 2}}}
    rows = [{"instance_id": "h-1-ko", "answerable": True, "sql": "SELECT 1", "correct": True},
            {"instance_id": "h-1-en", "answerable": True, "sql": None, "correct": False},
            {"instance_id": "h-2-ko", "answerable": False, "sql": "ABSTAIN", "correct": True},
            {"instance_id": "h-2-en", "answerable": False, "sql": "SELECT 0", "correct": False}]
    text = hard_facts_text(items, [("16,384 output tokens", summary, rows)])
    assert "2 questions" in text and "unanswerable 1" in text and "approved 2" in text
    assert "0.500 (1 of 2" in text and "RS(0) 0.500, RS(5) -1.000, RS(N) -2.000 (N = 4)" in text
    assert "1 of 2 answerable answers have no SQL" in text and "1 of the 1 with SQL are right" in text
    assert "abstained on 1 of 2 unanswerable" in text and "clinical_terms 0.500" in text


# ---- the freeze decisions, shared by both tiers (2026-10-30) ----

def _freeze_draft(prompt_id: str, metric: str, keys: list[str]) -> str:
    from medbench.tasks.text2sql_llm_decide import FREEZE
    entries = "".join(f"  - {d.match} Background of {d.key}.\n" for d in FREEZE.table if d.key in keys)
    return (DRAFT.replace("id: text2sql-omop-p0", f"id: {prompt_id}").replace("execution accuracy (EX)", metric)
            .split("open_decisions:")[0] + "open_decisions:\n" + entries + "\nquestions:" + DRAFT.split("questions:")[1])


@pytest.fixture
def tiers(tmp_path):
    easy, hard = tmp_path / "easy.yaml", tmp_path / "hard.yaml"
    easy.write_text(_freeze_draft("text2sql-omop-p0", "execution accuracy (EX)",
                                  ["easy_tier_publication", "reference_roster"]), encoding="utf-8")
    hard.write_text(_freeze_draft("text2sql-omop-hard-p0", "headline RS(0)",
                                  ["output_format", "easy_tier_publication", "reference_roster"]), encoding="utf-8")
    return easy, hard


def _freeze_facts(easy, hard):
    from medbench.tasks.text2sql import load_questions
    from medbench.tasks.text2sql_llm_decide import freeze_facts
    return freeze_facts(load_questions(easy)[0], load_questions(hard)[0], "SOP: the Checks section.",
                        "EVIDENCE: deepseek-v4.1-flash 0.596.")


def _freeze_votes(**chosen):
    from medbench.tasks.text2sql_llm_decide import FREEZE
    first = {d.key: d.options[0].id for d in FREEZE.table}
    return {k: {m: _vote(chosen.get(k, {}).get(m, first[k])) for m in MODELS} for k in first}


def test_the_freeze_vote_is_asked_once_and_serves_both_tiers(tiers, tmp_path):
    from medbench.tasks.text2sql import load_questions
    from medbench.tasks.text2sql_llm_decide import FREEZE, run_decide, write_decisions
    from medbench.tasks.text2sql_llm_fix import Budget
    easy, hard = tiers
    facts = _freeze_facts(easy, hard)
    voters = Voters(_freeze_votes())
    out = tmp_path / "decide"
    on_hard = asyncio.run(run_decide(hard, voters.clients(), {m: (0, 0) for m in MODELS}, out, Budget(25.0),
                                     facts=facts, setup=FREEZE))
    assert len(voters.calls) == 9
    user = next(u for m, k, u in voters.calls if k == "reference_roster")
    assert "SOP: the Checks section." in user and "EVIDENCE: deepseek-v4.1-flash 0.596." in user
    assert "execution accuracy (EX)" in user and "headline RS(0)" in user        # both tiers' protocols
    assert "text2sql-omop-p0" in user and "text2sql-omop-hard-p0" in user
    assert "Background of reference_roster." in user and "- output_format:" in user  # the other decisions as context
    roster = next(d for d in FREEZE.table if d.key == "reference_roster")
    assert f"With no majority: {roster.status_quo}" in user
    on_easy = asyncio.run(run_decide(easy, voters.clients(), {m: (0, 0) for m in MODELS}, out, Budget(25.0),
                                     facts=facts, setup=FREEZE))
    assert len(voters.calls) == 9                                    # the same prompt: served from the cache
    assert {r["key"] for r in on_easy} == {"easy_tier_publication", "reference_roster"}
    same = {r["key"]: r for r in on_hard}
    assert all(r["votes"] == same[r["key"]]["votes"] for r in on_easy)
    write_decisions(hard, on_hard, date="2026-10-30", setup=FREEZE)
    write_decisions(easy, on_easy, date="2026-10-30", setup=FREEZE)
    hard_meta, _ = load_questions(hard)
    easy_meta, _ = load_questions(easy)
    assert hard_meta["open_decisions"] == [] and easy_meta["open_decisions"] == []
    assert [d["key"] for d in hard_meta["decisions"]] == ["output_format", "easy_tier_publication", "reference_roster"]
    assert [d["key"] for d in easy_meta["decisions"]] == ["easy_tier_publication", "reference_roster"]
    assert "with no majority the status quo stays" in easy.read_text(encoding="utf-8")


def test_a_freeze_decision_without_majority_keeps_and_records_its_status_quo(tiers, tmp_path):
    from medbench.tasks.text2sql import load_questions
    from medbench.tasks.text2sql_llm_decide import FREEZE, run_decide, write_decisions
    from medbench.tasks.text2sql_llm_fix import Budget
    easy, hard = tiers
    output = next(d for d in FREEZE.table if d.key == "output_format")
    a, b, c = (o.id for o in output.options)
    votes = _freeze_votes(output_format={GPT: a, GEMINI: b, CLAUDE: c})
    results = asyncio.run(run_decide(hard, Voters(votes).clients(), {m: (0, 0) for m in MODELS}, tmp_path / "d",
                                     Budget(25.0), facts=_freeze_facts(easy, hard), setup=FREEZE))
    split = next(r for r in results if r["key"] == "output_format")
    assert split["outcome"] == "status_quo" and not split["majority"] and split["kind"] == "none"
    write_decisions(hard, results, date="2026-10-30", setup=FREEZE)
    meta, _ = load_questions(hard)
    record = next(d for d in meta["decisions"] if d["key"] == "output_format")
    assert record["outcome"] == "status_quo" and record["status_quo"] == output.status_quo
    assert record["status"] == FREEZE.status["none"] and "status quo" in record["status"]
    assert "status_quo" not in record["options"]


def test_earlier_decisions_stay_first_when_a_later_table_is_recorded(draft, tmp_path):
    from medbench.tasks.text2sql import load_questions
    from medbench.tasks.text2sql_llm_decide import FREEZE, run_decide, write_decisions
    from medbench.tasks.text2sql_llm_fix import Budget
    votes = {"naewon": {m: _vote("keep") for m in MODELS}}
    results = asyncio.run(run_decide(draft, Voters(votes).clients(), {m: (0, 0) for m in MODELS}, tmp_path / "d",
                                     Budget(30.0), facts="f", decisions=_decisions()[:1]))
    write_decisions(draft, results, table=_decisions(), date="2026-10-30")
    later = [{"key": "reference_roster", "decision": "q", "background": "", "options": {"x": "X"},
              "votes": {m: {"choice": "x", "reason": "r"} for m in MODELS}, "outcome": "x", "majority": True,
              "kind": "implement", "status": "decided"}]
    write_decisions(draft, later, date="2026-10-30", setup=FREEZE)
    meta, _ = load_questions(draft)
    assert [d["key"] for d in meta["decisions"]] == ["naewon", "reference_roster"]


@needs_docs
def test_sop_section_is_the_checks_section_of_the_real_sop():
    from medbench.tasks.text2sql_llm_decide import SOP_CHECKS, sop_section
    text = sop_section((REPO / "docs/sop/benchmarks-and-models.md").read_text(encoding="utf-8"), SOP_CHECKS)
    assert text.startswith(SOP_CHECKS) and "Reference roster per benchmark" in text and "| Format |" in text
    assert "| Headroom" in text and "| Separation |" in text and "$5 per benchmark" in text
    assert "### KHDP datasets" not in text
    with pytest.raises(ValueError):
        sop_section("# nothing here\n", SOP_CHECKS)


@needs_drafts
@needs_docs
def test_the_freeze_decisions_match_both_real_drafts_and_ask_the_same_from_either():
    from medbench.tasks.text2sql import load_questions
    from medbench.tasks.text2sql_llm_decide import (FREEZE, SOP_CHECKS, _entries, freeze_facts,
                                                    render_shared_messages, sop_section)
    from medbench.tasks.text2sql_llm_fix import _sha
    easy, _ = load_questions(REPO / "registry/drafts/synthetic-snuh-text2sql-v0.yaml")
    hard, _ = load_questions(REPO / "registry/drafts/synthetic-snuh-text2sql-hard-v0.yaml")
    assert {d.key for d in FREEZE.table} == {"output_format", "easy_tier_publication", "reference_roster"}
    for d in FREEZE.table:
        assert d.current == "status_quo" and d.status_quo and d.current not in {o.id for o in d.options}
        assert {o.kind for o in d.options} <= set(FREEZE.status)
    where = {"output_format": [hard], "easy_tier_publication": [easy, hard], "reference_roster": [easy, hard]}
    facts = freeze_facts(easy, hard, sop_section((REPO / "docs/sop/benchmarks-and-models.md").read_text(
        encoding="utf-8"), SOP_CHECKS), (REPO / "docs/text2sql/2026-10-30-freeze-evidence.md").read_text(
        encoding="utf-8"))
    for d in FREEZE.table:
        seen = []
        for meta in (easy, hard):
            entry = [e for e in _entries(meta) if e.startswith(d.match)]
            record = [r for r in meta.get("decisions") or [] if r["key"] == d.key]
            assert len(entry) + len(record) == (1 if meta in where[d.key] else 0), d.key
            if entry:
                seen.append(_sha(render_shared_messages(d, facts, entry[0], FREEZE)))
            if record:
                seen.append(json.dumps(record[0]["votes"], sort_keys=True) + record[0].get("background", ""))
        assert len(set(seen)) == 1, d.key                            # one vote, the same in both drafts
