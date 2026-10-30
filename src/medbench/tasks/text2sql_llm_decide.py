"""Open decisions of the Text-to-SQL draft, voted on by the three frontier reviewers (owner, 2026-10-30).

The owner asked that what waited on the owner's review be settled by the three models instead. Each model gets
the draft's protocol and prompt, the database note, the facts of the dry runs and review, and one decision with
explicit options, and answers {"choice", "reason"} as JSON. Two of three decide; with no majority v0's current
behaviour stays. `write_decisions` records each decision (options, every model's choice and reason, outcome) in a
`decisions:` block of the draft (with the open_decisions entry the voters saw as its background) and removes the
settled entries from `open_decisions`. The outcomes themselves are
carried out separately: small local ones in the draft and prompt, new items as next work.

The freeze decisions (`FREEZE`, 2026-10-30) concern both tiers at once: their prompt carries both drafts' protocols,
the SOP text they touch and the measured evidence, and nothing of the draft it is run from, so the same vote is
asked once and the run from the other draft is served from the cache. With no majority each keeps a status quo
stated with the decision, which is not one of the options. Public with the evaluation code since 2026-10-30
(owner); the decisions with every vote and reason are recorded in the question files.
"""
import asyncio
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from medbench.tasks.text2sql import (ABSTAIN, ALL_CATEGORIES, CATEGORIES, CONVENTIONS, DIFFICULTIES, HARD_CONVENTIONS,
                                     load_questions)
from medbench.tasks.text2sql_llm_fix import Asker, Budget, parse_json_object

STATUS = {"none": "v0 unchanged", "implement": "to implement", "next_work": "next work"}


@dataclass(frozen=True)
class Option:
    id: str
    text: str
    kind: str  # none (v0 stays), implement (a small change to the draft or prompt), next_work (new items or sets)


@dataclass(frozen=True)
class Decision:
    key: str
    match: str  # the start of its entry in the draft's open_decisions (whitespace normalised)
    question: str
    current: str  # the option that is v0's current behaviour, or "status_quo" when that is none of the options
    options: list[Option]
    status_quo: str = ""  # what stays with no majority when `current` is not an option (shown to the voters)


DECISIONS = [
    Decision("headroom", "Headroom.", "v0 is close to saturated: a free open model scored 0.994 execution accuracy. "
             "What should happen about headroom before the task is frozen?", "keep_v0", [
                 Option("keep_v0", "Freeze v0 as it is; headroom is not addressed in this version.", "none"),
                 Option("easy_tier_plus_harder_set", "Keep v0 as an easy tier and build a separate harder set next to "
                        "it (concepts named in clinical words rather than quoted, cohort definitions over several "
                        "criteria and time windows, data-quality traps such as concept 0 or non-Condition concepts in "
                        "condition_occurrence).", "next_work"),
                 Option("rework_v0_harder", "Rework v0 into harder items before it is frozen, so there is one set.",
                        "next_work"),
                 Option("shorter_note", "Keep the items but shorten the database note, so models must infer more "
                        "conventions from the schema and data.", "next_work")]),
    Decision("naewon", 'Korean "내원".', 'How should the Korean word 내원 ("visit") be disambiguated?', "keep", [
        Option("keep", "Change nothing; reading 내원 as outpatient-only stays a model error.", "none"),
        Option("note", "Add a line to the database note: 내원 (visit) means any visit_occurrence row unless the "
               "question names a visit type. This changes every prompt.", "implement"),
        Option("reword", "Reword the Korean of items 115 and 127 (방문, or 내원(유형 무관)); the note stays as it is.",
               "implement"),
        Option("both", "Add the note line and reword the two items.", "implement")]),
    Decision("unanswerable", "Unanswerable questions", "Should the benchmark have unanswerable questions with "
             "abstention scoring (the EHRSQL reliability score RS(c))?", "not_in_v0", [
                 Option("not_in_v0", "No: v0 is frozen without unanswerable questions or abstention scoring.", "none"),
                 Option("later_set", "Yes, in a later version or the harder set, not in v0.", "next_work"),
                 Option("before_freeze", "Yes, added to v0 before it is frozen.", "next_work")]),
    Decision("relative_time", "Relative time questions", 'Should there be relative-time questions ("in the last 6 '
             'months") with a fixed "now", as EHRSQL fixes one?', "absolute_only", [
                 Option("absolute_only", "No: absolute dates only, as in v0.", "none"),
                 Option("later_set", 'Yes, with a fixed "now" stated in the prompt, in a later version or the harder '
                        "set, not in v0.", "next_work"),
                 Option("before_freeze", 'Yes, with a fixed "now", added to v0 before it is frozen.', "next_work")]),
    Decision("concept_id_or_name", "Whether a question that names a concept may be answered with concept_id",
             "When a question names a concept, may an answer give concept_id instead of concept_name (or the "
             "reverse)?", "explicit", [
                 Option("explicit", "No: each question says which one to return, and only that is correct (v0).",
                        "none"),
                 Option("accept_either", "Yes: the scorer maps concept ids to names, so either is correct.",
                        "implement"),
                 Option("explicit_plus_note", "No, as in v0, and the database note also states the default: return "
                        "concept names unless the question asks for ids.", "implement")]),
    Decision("database_note", "Whether the prompt keeps the database note", "Should the prompt keep the database "
             "note (conventions such as age = event year minus year of birth), or should the conventions move into "
             "each question?", "keep_note", [
                 Option("keep_note", "Keep the note in every prompt (BIRD's external knowledge); questions stay "
                        "short.", "none"),
                 Option("move_into_questions", "Remove the note and state each convention in the questions that "
                        "need it.", "next_work"),
                 Option("keep_and_restate", "Keep the note and also restate conventions in the questions that "
                        "depend on them.", "next_work")]),
    Decision("split", "Split into dev and test", "Should the items be split into a development set and a test set?",
             "no_split", [
                 Option("no_split", "No: every item is a test item; the dry runs serve as development (v0).", "none"),
                 Option("dev_test", "Yes: a small dev set (about 10% of items, stratified by category, fixed seed) "
                        "published with its answers for prompt development; the rest is the test set.",
                        "implement")]),
    Decision("languages", "Split into dev and test", "How are the Korean and English versions reported?",
             "both_scored", [
                 Option("both_scored", "Both are scored instances; the headline is execution accuracy over all "
                        "question-language pairs, with each language also shown (v0).", "none"),
                 Option("separate_only", "Korean and English are reported as two separate scores, with no combined "
                        "headline.", "implement"),
                 Option("korean_headline", "Korean is the headline score; English is reported as a secondary "
                        "score.", "implement")]),
]

SYSTEM = (
    "You are one of three voters deciding open design questions for a draft Text-to-SQL benchmark over an OMOP CDM "
    "v5.4 database (DuckDB) of synthetic patients modelled on a Korean university hospital. The benchmark is "
    "published on a public leaderboard that ranks AI models; it is meant for open models. Pick exactly one of the "
    "offered options for the decision you are asked about, weighing validity of the ranking, headroom, cost of the "
    "change and consistency with the published protocols the draft adopts. Two of three votes decide; with no "
    "majority the draft keeps its current (v0) behaviour.\n"
    'Answer with one JSON object and nothing else: {"choice": "<option id>", "reason": "<two to four sentences>"}.'
)

# The harder set's open decisions (review of PR #91, 2026-10-30; the owner delegated them to the three models).
HARD_BUDGET_LEFT_USD = 1.67  # $25 cap minus the job's route-billed $23.33 after the 13 fixes, before this vote
HARD_DECISIONS = [
    Decision("headline_metric", "Headline metric.", "Which number is the harder set's headline on the leaderboard: "
             "execution accuracy on the answerable items, or one of the reliability scores RS(0), RS(5), RS(N) over "
             "all items?", "no_headline", [
                 Option("no_headline", "Name no headline: execution accuracy on the answerable items and RS(0), "
                        "RS(5), RS(N) are reported side by side, as the protocol now says.", "none"),
                 Option("ex_answerable", "Execution accuracy on the answerable items is the headline; RS(0), RS(5), "
                        "RS(N) and the abstention rates are shown next to it.", "implement"),
                 Option("rs0", "RS(0) is the headline (a wrong answer costs no more than an abstention); EX, RS(5) "
                        "and RS(N) are shown next to it.", "implement"),
                 Option("rs5", "RS(5) is the headline (a wrong answer costs as much as five right ones earn); EX, "
                        "RS(0) and RS(N) are shown next to it.", "implement"),
                 Option("rsN", "RS(N) is the headline (one wrong answer outweighs every right one); EX, RS(0) and "
                        "RS(5) are shown next to it.", "implement")]),
    Decision("output_limit", "Output limit.", "What output limit does the harder set give an evaluated model? The "
             "registry default is 16,384 output tokens, thinking included.", "keep_16k", [
                 Option("keep_16k", "Keep the registry default of 16,384 output tokens; an answer that runs out of "
                        "tokens without SQL is scored as an answer (-c), as now.", "none"),
                 Option("limit_32k", "Set 32,768 output tokens for the harder set (in the draft's prompt block, used "
                        "by its dry runs and its evaluation).", "implement"),
                 Option("thinking_budget", "Keep the output limit but give reasoning models a thinking budget that "
                        "leaves room for the final query (needs support on each route and its own diagnostic runs).",
                        "next_work")]),
    Decision("review_batch", "Review batch.", "Should reviewing five items per request (each judged on its own; an "
             "item missing from the answer is asked again alone) become the standard procedure for this benchmark's "
             "model reviews and re-reviews?", "as_is", [
                 Option("as_is", "Keep the current defaults without a stated standard: the harder set's first review "
                        "and llm-fix re-reviews take five items per request, v0's llm-review one.", "none"),
                 Option("standard_5", "Yes: five items per request is the standard for every first review and "
                        "re-review, stated in the SOP and the drafts.", "implement"),
                 Option("per_item", "No: one item per request is the standard; batches only when a budget needs "
                        "them, recorded with the review.", "implement")]),
    Decision("second_round", "Second round.", "Should a second, harder generation round be written for the harder "
             f"set now? Paid calls for this job are capped at $25 and ${HARD_BUDGET_LEFT_USD:.2f} is left before this "
             "vote; round 1 cost $16.25 for 101 items written, reviewed and fixed (about $0.16 per item).",
             "no_second_round", [
                 Option("no_second_round", "No second round: the harder set stays at its 101 items.", "none"),
                 Option("after_open_models", "Not now: first run more of the free open models on the current set, "
                        "and write a second round only if one of them reaches 0.85 execution accuracy on the "
                        "answerable items.", "next_work"),
                 Option("now", "Yes, now: a second round of about 30 harder items with the same review and fix "
                        "(about $5 at round 1's cost), which does not fit what is left under the $25 cap and so "
                        "waits for the owner's budget approval.", "needs_budget")]),
]

HARD_SYSTEM = (
    "You are one of three voters deciding open design questions for the harder set of a draft Text-to-SQL benchmark "
    "over an OMOP CDM v5.4 database (DuckDB) of synthetic patients modelled on a Korean university hospital. The "
    "harder set sits next to an easy tier and adds unanswerable questions scored by abstention (the EHRSQL-2024 "
    "reliability score). The benchmark is published on a public leaderboard that ranks AI models; it is meant for "
    "open models. Pick exactly one of the offered options for the decision you are asked about, weighing validity "
    "of the ranking, headroom, cost of the change and consistency with the published protocols the draft adopts. "
    "The owner caps paid model calls for this set at $25; an option that needs more cannot run without the owner's "
    "budget approval. Two of three votes decide; with no majority the draft keeps its current behaviour.\n"
    'Answer with one JSON object and nothing else: {"choice": "<option id>", "reason": "<two to four sentences>"}.'
)


@dataclass(frozen=True)
class DecideSetup:
    """What a draft's vote uses: its decisions, the voters' instructions, the database note evaluated models get,
    what each outcome kind means for the draft, what the voters did with the items, and what stays without a
    majority."""
    table: list[Decision]
    system: str
    notes: str
    status: dict[str, str]
    authors: str
    fallback: str
    shared: bool = False  # the decisions concern both tiers: the prompt holds no draft-specific text


V0 = DecideSetup(DECISIONS, SYSTEM, CONVENTIONS, STATUS, "also reviewed and fixed the items", "v0's current behaviour")
HARD = DecideSetup(HARD_DECISIONS, HARD_SYSTEM, HARD_CONVENTIONS,
                   {"none": "unchanged", "implement": "to implement", "next_work": "next work",
                    "needs_budget": "needs the owner's budget approval (does not fit the $25 cap); not run"},
                   "also wrote, reviewed and fixed the items", "the draft's current behaviour")


# The freeze decisions (2026-10-30): asked about both tiers at once, after the free open models' dry runs and the
# SOP check measured on them (docs/text2sql/2026-10-30-freeze-evidence.md). The owner delegated every Text-to-SQL
# decision to the three models; nothing here is reviewed by the owner.
SOP_CHECKS = "### Checks (stage 5, by the model evaluator)"
FREEZE_DECISIONS = [
    Decision("output_format", "Hard tier output limit and format rule.", "What output limit does the hard tier give "
             "an evaluated model, and how does the SOP's format check (parse or format failures < 10% for the "
             "roster) count an answer whose thinking used the whole limit and left no SQL?", "status_quo", [
                 Option("keep_16k_model_failure", "Keep the registry's 16,384 output tokens. An answer whose thinking "
                        "used the whole limit and left no SQL stays scored as a wrong answer (wrong in EX, -c in "
                        "RS(c)) and counts as a model failure, not a format failure, in the format check; only an "
                        "answer with text but no parseable SQL or ABSTAIN is a format failure. Consequences: the "
                        "format check passes for every measured run (0 such answers); the references' 16,384-token "
                        "runs stand without reruns; models that think long are capped by the limit (qwen3.8-max "
                        "0.355 at 16,384, 0.711 at 32,768), so the task page states the limit and each model's share "
                        "of answers cut off.", "implement"),
                 Option("limit_32k", "Raise the hard tier's limit to 32,768 output tokens, with a client read timeout "
                        "long enough for such answers on slow routes, or streaming; an answer without SQL still "
                        "counts as a format failure. Consequences: the client change comes first, then the reference "
                        "runs and the format check are repeated at 32,768 (free route; gpt-oss-20b on the paid route "
                        "for a few cents); qwen3.8-max had 4.5% answers without SQL at 32,768, the references are "
                        "unmeasured there; every run is slower (qwen3.8-max wrote 3.19 million output tokens for 202 "
                        "answers).", "next_work"),
                 Option("thinking_budget", "Keep 16,384 output tokens and give reasoning models a thinking budget "
                        "that leaves room for the final query. Consequences: needs a thinking-budget field the client "
                        "does not send today, support on every route and model, and its own diagnostic runs; the "
                        "format check and the reference runs wait for them.", "next_work")],
             status_quo="The limit stays 16,384 and the format rule stays unsettled: counting every answer without "
                        "SQL as a format failure, the hard tier fails the format check at 16,384 (both measured "
                        "references 12.4%), so the check is not passed."),
    Decision("easy_tier_publication", "Easy tier publication.", "The easy tier v0 fails the SOP headroom check on its "
             "own (best EX 0.997, H about 0.003, the best model's interval reaches 1.0). How is it published?",
             "status_quo", [
                 Option("easy_tier_of_one", "Publish v0 only as the easy tier of one two-tier Text-to-SQL benchmark "
                        "whose SOP check (headroom, separation, format) rests on the hard tier; v0's EX is shown next "
                        "to the hard tier's headline and is never ranked on its own. Consequences: one task with two "
                        "tiers and the hard tier's RS(0) as its headline; the page states that v0 is saturated; v0 "
                        "still separates the measured weak reference from the strong one (paired +0.094 [0.058, "
                        "0.136]).", "implement"),
                 Option("saturated_diagnostic", "Publish v0 as its own task, marked as a saturated diagnostic with no "
                        "leaderboard ranking or claim; the hard tier is a separate benchmark with its own check. "
                        "Consequences: the SOP has no such status today (building, in review, published), so a new "
                        "label and page treatment come first; two tasks to maintain.", "next_work"),
                 Option("do_not_publish", "Do not publish v0; it stays an internal draft (development and smoke "
                        "tests) and only the hard tier goes through the check. Consequences: the 154 reviewed items "
                        "stay private; the hard tier keeps v0's prompt and scoring rules for its answerable items.",
                        "implement")],
             status_quo="v0 stays an unfrozen draft proposed as its own task (synthetic-snuh-text2sql); on its own it "
                        "fails the check and goes back to Build, so nothing of v0 is published until a later "
                        "decision."),
    Decision("reference_roster", "Reference roster.", "Which weak and strong reference models make the roster of the "
             "Text-to-SQL benchmark's SOP check (separation, and the roster of the format check)? The SOP fixes the "
             "roster before the runs; these runs came first, so any pair chosen now is chosen after seeing them.",
             "status_quo", [
                 Option("oss20b_dsflash", "Weak openai/gpt-oss-20b, strong deepseek/deepseek-v4.1-flash-552b-a16b "
                        "(the measured pair): hard EX 0.193 against 0.596, paired +0.404 [0.319, 0.494], no overlap; "
                        "v0 0.899 against 0.994, paired +0.094 [0.058, 0.136], no overlap; 12.4% answers without SQL "
                        "each on the hard tier at 16,384. The weak model runs on the paid route (a few cents), the "
                        "strong one free.", "implement"),
                 Option("oss20b_qwenmax", "Weak openai/gpt-oss-20b, strong alibaba/qwen3.8-max-2.4t-a95b (best on v0, "
                        "best on the hard tier at 32,768): hard EX at 16,384 0.355, paired +0.163 [0.099, 0.236] but "
                        "the two intervals touch; at 32,768 0.711, paired +0.518 [0.427, 0.610]; v0 0.997, paired "
                        "+0.097 [0.058, 0.140], no overlap; 52.5% answers without SQL at 16,384, 4.5% at 32,768.",
                        "implement"),
                 Option("qwenflash_dsflash", "Weak alibaba/qwen3.8-flash-next-180b-a6b, strong "
                        "deepseek/deepseek-v4.1-flash-552b-a16b, both free: hard EX 0.217 against 0.596, paired "
                        "+0.380 [0.290, 0.476], no overlap; v0 0.984 against 0.994, paired +0.010 [-0.007, 0.029], "
                        "the intervals overlap (fails on v0); 61.9% answers without SQL for the weak model on the "
                        "hard tier at 16,384.", "implement")],
             status_quo="No roster is recorded: the separation measured with gpt-oss-20b and deepseek-v4.1-flash "
                        "stays a measurement, and the check has no fixed roster until a later decision."),
]

FREEZE_SYSTEM = (
    "You are one of three voters deciding how a draft Text-to-SQL benchmark is frozen for a public leaderboard. It "
    "runs over an OMOP CDM v5.4 database (DuckDB) of synthetic patients modelled on a Korean university hospital and "
    "has two drafted tiers: an easy tier v0 scored by execution accuracy, and a hard tier with unanswerable "
    "questions whose headline is the EHRSQL-2024 reliability score RS(0). The leaderboard ranks AI models and is "
    "meant for open models; a benchmark is published only after it passes the site's SOP checks, quoted below. "
    "Pick exactly one of the offered options for the decision you are asked about, weighing validity of the "
    "ranking, the SOP's rules, headroom, cost of the change and consistency with the published protocols the drafts "
    "adopt. Paid model calls for this job are capped at $25; runs on the free route cost nothing. Two of three votes "
    "decide; with no majority the status quo stated with the decision stays.\n"
    'Answer with one JSON object and nothing else: {"choice": "<option id>", "reason": "<two to four sentences>"}.'
)

FREEZE = DecideSetup(FREEZE_DECISIONS, FREEZE_SYSTEM, "",
                     {"none": "unchanged: no majority, so the status quo stays", "implement": "to implement",
                      "next_work": "next work"},
                     "also wrote, reviewed and fixed the items", "the status quo", shared=True)


def sop_section(text: str, heading: str) -> str:
    """The SOP section under `heading` (an exact heading line), up to the next heading of the same or a higher
    level."""
    lines = text.split("\n")
    if heading not in lines:
        raise ValueError(f"no section {heading!r} in the SOP")
    start = lines.index(heading)
    level = len(heading) - len(heading.lstrip("#"))
    end = next((n for n in range(start + 1, len(lines)) if lines[n].startswith("#")
                and len(lines[n]) - len(lines[n].lstrip("#")) <= level), len(lines))
    return "\n".join(lines[start:end]).strip()


def _tier(meta: dict) -> str:
    keys = ("proposed_task", "tier", "protocol", "prompt")
    return yaml.safe_dump({k: meta[k] for k in keys if meta.get(k) is not None}, allow_unicode=True, sort_keys=False,
                          width=120)


def freeze_facts(easy: dict, hard: dict, sop: str, evidence: str) -> str:
    """What the freeze voters see about both tiers, the same from either draft: the two drafts' protocol and prompt
    blocks, the SOP text the decisions touch and the measured evidence."""
    return (f"The easy tier's draft (protocol and prompt):\n```yaml\n{_tier(easy)}```\n\n"
            f"The hard tier's draft (protocol and prompt):\n```yaml\n{_tier(hard)}```\n\n"
            f"The SOP text these decisions touch (docs/sop/benchmarks-and-models.md):\n{sop.strip()}\n\n"
            f"Evidence:\n{evidence.strip()}")


def render_shared_messages(decision: Decision, facts: str, background: str, setup: DecideSetup) -> list[dict]:
    """A decision about both tiers: no draft-specific text, so either draft asks the same."""
    options = "\n".join(f"- {o.id}: {o.text}" + (f"  [{setup.fallback}]" if o.id == decision.current else "")
                        for o in decision.options)
    others = "\n".join(f"- {d.key}: {d.question}" for d in setup.table if d.key != decision.key) or "- none"
    stays = f"\nWith no majority: {decision.status_quo}" if decision.status_quo else ""
    user = (f"The three frontier models voting here {setup.authors}, so they would be marked as reviewers if the "
            f"benchmark ever ranked them.\n\n{facts}\n\n"
            f"The other decisions of this vote, each voted on separately:\n{others}\n\n"
            f"Decision key: {decision.key}\nDecision: {decision.question}\nBackground from the drafts: {background}\n"
            f"Options:\n{options}{stays}\n\nChoose one option id.")
    return [{"role": "system", "content": setup.system}, {"role": "user", "content": user}]


def _norm(text: str) -> str:
    return " ".join(str(text).split())


def facts_text(items, summary: dict | None, failures: list[str] | None = None) -> str:
    """The draft's state and the free model's dry run (with the instance ids it got wrong), as given to the voters."""
    cats = Counter(i.category for i in items)
    diffs = Counter(i.difficulty for i in items)
    states = Counter(i.review for i in items)
    lines = [f"- {len(items)} questions, each in Korean and English: "
             + ", ".join(f"{c} {cats[c]}" for c in CATEGORIES if cats[c])
             + "; difficulty " + ", ".join(f"{d} {diffs[d]}" for d in DIFFICULTIES if diffs[d]) + ".",
             "- Review by the three frontier models (unanimous approval): "
             + ", ".join(f"{k} {v}" for k, v in sorted(states.items()))
             + ". Items that need an edit are revised by the same three models in a separate step."]
    if summary:
        ex = summary["execution_accuracy"]
        correct = round(ex["value"] * ex["n"])
        langs = ", ".join(f"{k} {v['value']:.3f}" for k, v in (summary.get("by_language") or {}).items())
        diff = ", ".join(f"{k} {v['value']:.3f}" for k, v in (summary.get("by_difficulty") or {}).items())
        lines.append(f"- Dry run with a free open model ({summary.get('model', 'the free model')}, internal, not "
                     f"published): execution accuracy {ex['value']:.3f} ({correct} of {ex['n']} question-language "
                     f"pairs); by language {langs}; by difficulty {diff}.")
    by_id = {i.id: i for i in items}
    for iid in failures or []:
        qid, _, lang = iid.rpartition("-")
        if qid in by_id:
            lines.append(f"- Answered wrongly by the free model: {iid}: {by_id[qid].text(lang)}")
    return "\n".join(lines)


def hard_facts_text(items, runs: list[tuple[str, dict, list[dict]]]) -> str:
    """The harder set's state and its dry runs (label, summary, scored rows): scores, abstentions, and how many
    answerable answers had no SQL and how many of those with SQL were right."""
    cats = Counter(i.category for i in items)
    states = Counter(i.review for i in items)
    lines = [f"- {len(items)} questions, each in Korean and English: "
             + ", ".join(f"{c} {cats[c]}" for c in ALL_CATEGORIES if cats[c]) + ".",
             "- Review by the three frontier models (unanimous approval): "
             + ", ".join(f"{k} {v}" for k, v in sorted(states.items())) + "."]
    for label, s, rows in runs:
        ex, rs, ab = s["execution_accuracy"], s["reliability"], s["abstention"]
        ci = f", 95% CI {ex['ci_low']:.3f}-{ex['ci_high']:.3f}" if ex.get("ci_low") is not None else ""
        answerable = [r for r in rows if r.get("answerable")]
        no_sql = sum(r.get("sql") is None for r in answerable)
        with_sql = [r for r in answerable if r.get("sql") not in (None, ABSTAIN)]
        un, an = ab["unanswerable_abstained"], ab["answerable_abstained"]
        cats_ex = ", ".join(f"{k} {v['value']:.3f}" for k, v in (s.get("by_category") or {}).items()
                            if k != "unanswerable")
        lines.append(
            f"- Dry run, {s.get('model', 'free model')}, {label} (internal, not published): execution accuracy on "
            f"answerable items {ex['value']:.3f} ({round(ex['value'] * ex['n'])} of {ex['n']} question-language "
            f"pairs{ci}); RS(0) {rs['RS(0)']:.3f}, RS(5) {rs['RS(5)']:.3f}, RS(N) {rs['RS(N)']:.3f} (N = {rs['N']}); "
            f"abstained on {round(un['value'] * un['n'])} of {un['n']} unanswerable and "
            f"{round(an['value'] * an['n'])} of {an['n']} answerable instances; {no_sql} of {len(answerable)} "
            f"answerable answers have no SQL, and {sum(r['correct'] for r in with_sql)} of the {len(with_sql)} with "
            f"SQL are right; by category {cats_ex}.")
    return "\n".join(lines)


def _entries(meta: dict) -> list[str]:
    return [_norm(e) for e in meta.get("open_decisions") or []]


def render_decide_messages(decision: Decision, meta: dict, facts: str, others: list[str],
                           setup: DecideSetup = V0) -> list[dict]:
    entry = next((e for e in _entries(meta) if e.startswith(decision.match)), "")
    protocol = yaml.safe_dump({"protocol": meta.get("protocol"), "prompt": meta.get("prompt")}, allow_unicode=True,
                              sort_keys=False, width=120)
    options = "\n".join(f"- {o.id}: {o.text}" + (f"  [{setup.fallback}]" if o.id == decision.current else "")
                        for o in decision.options)
    other = "\n".join(f"- {e}" for e in others if e != entry) or "- none"
    user = (f"Draft: {meta.get('proposed_task', 'synthetic-snuh-text2sql')} (v{meta.get('draft_version', 0)}), "
            f"dataset {meta.get('dataset', '')}.\nIntended use: {_norm(meta.get('intended_use', ''))}\n"
            f"The three frontier models voting here {setup.authors}, so they would be marked as "
            "reviewers if the benchmark ever ranked them.\n\n"
            f"The draft's protocol and prompt:\n```yaml\n{protocol}```\n\n"
            f"The database note every evaluated model receives:\n{setup.notes}\n\n"
            f"Facts:\n{facts}\n\n"
            f"Other open decisions, voted on separately:\n{other}\n\n"
            f"Decision key: {decision.key}\nDecision: {decision.question}\nBackground from the draft: {entry}\n"
            f"Options:\n{options}\n\nChoose one option id.")
    return [{"role": "system", "content": setup.system}, {"role": "user", "content": user}]


def parse_choice(text: str | None, option_ids) -> dict | None:
    """{"choice", "reason"} when the choice is one of `option_ids`, else None."""
    ids = set(option_ids)

    def accept(body: dict):
        choice = str(body.get("choice") or "").strip()
        return {"choice": choice, "reason": _norm(body.get("reason") or "")} if choice in ids else None
    return parse_json_object(text, accept)


def tally(votes: dict[str, dict | None], current: str) -> tuple[str, bool]:
    """(outcome, majority): the option more than half of the voters chose, else the current behaviour."""
    counts = Counter(v["choice"] for v in votes.values() if v)
    if counts:
        choice, n = counts.most_common(1)[0]
        if n * 2 > len(votes):
            return choice, True
    return current, False


def _recorded(meta: dict) -> dict[str, dict]:
    return {d["key"]: d for d in meta.get("decisions") or [] if isinstance(d, dict) and "key" in d}


async def run_decide(draft: Path, clients: dict, prices: dict[str, tuple[float, float]], out: Path, budget: Budget,
                     *, facts: str, decisions: list[Decision] | None = None, extra: dict | None = None,
                     setup: DecideSetup = V0) -> list[dict]:
    """Ask every model about every decision that is still open in the draft (`decisions`, default the setup's
    table). Returns one result per decision; status 'incomplete' when a request failed or the budget stopped it
    (asked again on the next run)."""
    meta, _ = load_questions(draft)
    entries, recorded = _entries(meta), _recorded(meta)
    decisions = setup.table if decisions is None else decisions
    todo = [d for d in decisions if d.key not in recorded and any(e.startswith(d.match) for e in entries)]
    asker = Asker(clients, prices, out, budget, extra=extra)

    async def one(decision: Decision) -> dict:
        background = next((e for e in entries if e.startswith(decision.match)), "")
        messages = (render_shared_messages(decision, facts, background, setup) if setup.shared else
                    render_decide_messages(decision, meta, facts, entries, setup))
        ids = [o.id for o in decision.options]
        answers = await asyncio.gather(*(asker.ask(m, "decide", decision.key, messages,
                                                   lambda text: parse_choice(text, ids)) for m in clients))
        votes = {m: a.value for m, a in zip(clients, answers)}
        outcome, majority = tally(votes, decision.current)
        kind = next((o.kind for o in decision.options if o.id == outcome), "none")  # a status quo is no option
        status = "incomplete" if any(a.status in ("error", "budget") for a in answers) else "decided"
        return {"key": decision.key, "decision": decision.question, "background": background,
                "options": {o.id: o.text for o in decision.options},
                **({"status_quo": decision.status_quo} if decision.status_quo else {}), "votes": votes,
                "outcome": outcome, "majority": majority, "kind": kind, "status": status}

    return list(await asyncio.gather(*(one(d) for d in todo)))


_TOP = re.compile(r"^\S")


def write_decisions(draft: Path, results: list[dict], *, table: list[Decision] | None = None, date: str,
                    setup: DecideSetup = V0) -> list[str]:
    """Record decided results in the draft's `decisions:` block (earlier records stay) and drop each open_decisions
    entry whose every decision in `table` (default the setup's) is recorded. Only the text between
    `open_decisions:` and `questions:` changes. Returns the keys recorded now."""
    table = setup.table if table is None else table
    text = Path(draft).read_text(encoding="utf-8")
    meta, _ = load_questions(draft)
    records = _recorded(meta)
    new = [r for r in results if r["status"] == "decided"]
    for r in new:
        records[r["key"]] = {"key": r["key"], "date": date, "decision": r["decision"],
                             **({"background": r["background"]} if r.get("background") else {}),
                             "options": r["options"],
                             **({"status_quo": r["status_quo"]} if r.get("status_quo") else {}),
                             "votes": r["votes"], "outcome": r["outcome"], "majority": r["majority"],
                             "status": setup.status[r["kind"]]}
    start = re.search(r"^open_decisions:", text, re.M)
    end = re.search(r"^questions:", text, re.M)
    if not start or not end or end.start() < start.start():
        raise ValueError("the draft needs open_decisions: before questions:")
    span = text[start.start():end.start()].split("\n")
    chunks, current, in_open = [], None, True
    for line in span[1:]:
        if _TOP.match(line):
            in_open = False  # the rest is the comment and the earlier decisions block, written again below
            if not line.startswith(("#", "decisions:")):
                raise ValueError(f"unexpected key between open_decisions and questions: {line}")
            continue
        if not in_open:
            continue
        if line.startswith("  - "):
            current = [line]
            chunks.append(current)
        elif current is not None and (line.startswith("    ") or not line.strip()):
            current.append(line)
    keep = []
    for chunk in chunks:
        while chunk and not chunk[-1].strip():
            chunk.pop()
        entry = _norm(yaml.safe_load("\n".join(chunk))[0])
        deciders = [d for d in table if entry.startswith(d.match)]
        if not deciders or not all(d.key in records for d in deciders):
            keep.append("\n".join(chunk))
    block = "open_decisions:" + ("\n" + "\n".join(keep) if keep else " []") + "\n\n"
    if records:
        order = {d.key: n for n, d in enumerate(table)}
        listed = sorted(records.values(), key=lambda d: order.get(d["key"], -1))  # earlier tables' records first
        dumped = yaml.safe_dump(listed, allow_unicode=True, sort_keys=False, width=116, default_flow_style=False)
        block += ("# Open decisions settled by the three frontier reviewers (owner, 2026-10-30): each model chose one "
                  f"option\n# with a reason; two of three decide, and with no majority {setup.fallback} stays. "
                  "`status` says\n# what the draft does about the outcome.\ndecisions:\n"
                  + "\n".join(f"  {line}" if line else "" for line in dumped.rstrip("\n").split("\n")) + "\n\n")
    text = text[:start.start()] + block + text[end.start():]
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "draft.yaml"
        probe.write_text(text, encoding="utf-8")
        check, _ = load_questions(probe)
    if set(_recorded(check)) != set(records):
        raise ValueError("the decisions block does not read back")
    Path(draft).write_text(text, encoding="utf-8")
    return [r["key"] for r in new]
