"""The harder Synthetic-SNUH Text-to-SQL set next to v0, written, checked and reviewed by three frontier models.

The owner handed the Text-to-SQL work to the three frontier models, fixes included (2026-10-30), and the models
voted (v0 draft, decisions headroom, unanswerable and relative_time) to keep v0 as an easy tier and to build a harder
set next to it, with unanswerable questions scored by abstention (the EHRSQL-2024 reliability score RS(c)) and
relative-time questions against a fixed "now". This module builds that set in three steps:

1. Generate. Each category of `HARD_CATEGORIES` is written in batches that rotate over the models (batch k of
   category j goes to models[(j + k) % 3]), so every category has items from every model. A proposer gets the
   schema and the database note the evaluated model gets (`HARD_PROMPT`), a data profile (concepts in use with
   patient counts, data-quality facts), the category's guide and the items already written in that category, and
   answers JSON items: question_ko, question_en, sql (null for an unanswerable item), order_matters, category,
   scope, why_hard.
2. Check. Every reference runs under v0's reference rules (`reference_problem`: one SELECT, runs, 1-1,000 rows not
   all zero or NULL, no clock, ORDER BY when order matters, not another item's SQL, no 내원 in the Korean), and no
   question may put a concept name its reference uses (as a literal or by concept_id) in quotation marks or
   parentheses. Items that fail go back to their proposer once with the problems; items that fail again are
   dropped.
3. Review. All three models judge every new item with v0's rubric extended for the harder set and for
   unanswerable items (`text2sql_llm_review.HARD_SYSTEM`). Only unanimous approval keeps an item as it is; every
   other item is marked needs_edit with the reviewers' reasons and goes to `text2sql llm-fix` (three rounds with
   rotating proposers and unanimous re-review, else dropped).

Paid calls go through `text2sql_llm_fix.Asker` and its `Budget`: one hard cap for the whole job folder, retries
included. Public with the evaluation code since 2026-10-30 (owner); the items it wrote are released on KHDP.
"""
import asyncio
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import yaml

from benchtoward.tasks.synthetic_snuh import connect_readonly
from benchtoward.tasks.text2sql import (HARD_CATEGORIES, HARD_NOW, HARD_PROMPT, MAX_REFERENCE_ROWS, Question,
                                     load_questions, schema_summary)
from benchtoward.tasks.text2sql_llm_fix import (Asker, Budget, _item_span, _rewrite_item, _scalar, clean_sql,
                                             parse_json_object, quoted_problem, reference_problem, review_askers,
                                             review_items)
from benchtoward.tasks.text2sql_llm_review import aggregate, parse_review
from benchtoward.tasks.text2sql_review import _yaml_str

HARD_DRAFT_NAME = "synthetic-snuh-text2sql-hard-v0"
JOB_DIR = "text2sql-hard"  # under <data_dir>/drafts: every cache of this job, so one budget covers them all
GENERATE_DIR, REVIEW_DIR = "generate", "hard-review"
GEN_MAX_TOKENS = 32000  # a batch of items, with room for the proposer's reasoning
REVIEW_DATE = "2026-10-30"
ID_PREFIX = "syn-hard-"
# Items per batch for each category; the batches of a category rotate over the models.
PLAN = {"clinical_terms": [7, 7, 7], "cohort_definition": [7, 7, 7], "data_quality": [7, 7, 7],
        "relative_time": [7, 7, 7], "unanswerable": [6, 6, 6]}  # 102 items, 18 of them (17.6%) unanswerable
GUIDE_START = "\n\nCategory "  # generation prompts share everything before the category guide (prompt caching)

GEN_SYSTEM = (
    "You write items for the harder set of a Text-to-SQL benchmark over an OMOP CDM v5.4 database (DuckDB) of "
    "synthetic patients modelled on Seoul National University Hospital (SNUH). An easy tier already exists: short "
    "questions that quote concept names exactly and need one or two joins; a free open model answers 99% of them. "
    "The harder set must separate strong models without becoming ambiguous. Each item has exactly one reasonable "
    "reading given the schema and the database notes that the evaluated model receives (both shown below), and its "
    "reference SQL, executed on the real data, is the answer key. The benchmark ranks AI models, so an ambiguous "
    "question or a wrong reference makes the ranking wrong.\n"
    "Rules for every item:\n"
    "- The Korean and English versions ask exactly the same thing; the Korean reads as a Korean clinician or "
    "researcher would ask it, the English as an English-speaking one would. The Korean never uses 내원 (write 방문, "
    "and say that every visit type counts where it could matter, or name the type).\n"
    "- Name concepts in clinical words as the database notes define them: Korean clinical terms in the Korean "
    "question; in the English question a common synonym or abbreviation (AFib, T2DM, HbA1c, CKD, ...) where one "
    "exists, else the plain clinical term. Never put a concept name in quotation marks or parentheses, as a gloss or "
    "as an output label. Use only concepts listed in "
    "the data profile, and only terms that identify one concept there (not ones that could equally mean a broader "
    "or narrower concept that is also in the data).\n"
    "- Say exactly what to return (which columns, rounding, sort order) and state every criterion, time window and "
    "tie rule, so that the answer is deterministic.\n"
    "- Reference SQL: one read-only DuckDB SELECT (WITH allowed); no current_date, now() or other clock functions "
    f"(write today as DATE '{HARD_NOW}'); 1 to {MAX_REFERENCE_ROWS:,} result rows that are not all zero or NULL; "
    "when order_matters is true, a full, tie-free ORDER BY. Filter concepts by concept_id or by exact concept_name. "
    "Write the SQL over several lines (a line per clause or CTE) so that reviewers can read it.\n"
    "- Do not repeat or lightly vary an item listed as already written.\n"
    'Answer with one JSON object and nothing else: {"items": [...]}, each item an object with the keys '
    '"question_ko", "question_en", "sql" (null only for an unanswerable item), "order_matters" (true only if the '
    'question asks for a sorted list), "category", "scope" ("individual" for one named patient, "group" for '
    'patients meeting criteria, "none" for no patient filter) and "why_hard" (one or two sentences: what a strong '
    "model must get right; for an unanswerable item, which table or column is missing or empty)."
)

GUIDES = {
    "clinical_terms": (
        "Category clinical_terms: every concept is named in clinical words that map to one concept of the data "
        "profile (Korean terms such as 심방세동, 제2형 당뇨병, 당화혈색소, 대장내시경; English synonyms or "
        "abbreviations such as AFib, T2DM, HbA1c). Use two or more concepts, possibly from different domains "
        "(condition, drug, laboratory test, procedure), combined with a comparison, a ratio, a grouping or a "
        "per-patient condition, so that the query needs real joins. The difficulty is finding the right concepts "
        "from clinical words and combining them correctly, not guessing a convention."),
    "cohort_definition": (
        "Category cohort_definition: a cohort defined by several criteria across tables, with an index event and "
        "time windows: for example patients whose first diagnosis of one disease is followed within 90 days by a "
        "prescription of a drug, aged 40 to 69 at the index date, without another disease recorded before it; a "
        "laboratory value before and after a drug start; a washout period; exclusion criteria. State the index "
        "event (first or any), the bounds of every window and whether they are inclusive, and what to return (a "
        "count, a rate, a mean, a list)."),
    "data_quality": (
        "Category data_quality: the correct answer requires handling a data-quality issue of this database that the "
        "obvious query gets wrong. Issues present (see the data profile): records with concept_id 0; "
        "non-Condition concepts in condition_occurrence; NULLs (value_as_number, range_high, unit_source_value, "
        "atc_cd, days_supply, route_concept_id, admitted_from_source_value), including NULLs under NOT IN, NOT LIKE, "
        "<>, COUNT(column) and AVG; exact duplicate records; negative laboratory values; drug records mapped to ATC "
        "class concepts next to RxNorm ingredients. The question, or the database notes, must say how the issue is "
        "handled (for example 'count duplicate prescriptions once', 'results without a value do not count', "
        "'prescriptions without an ATC code count as not starting with C10'), so the answer is unambiguous; the "
        "trap is that a natural query silently gives a different result. why_hard names that query's mistake."),
    "relative_time": (
        f"Category relative_time: time stated relative to today, which the database notes fix at {HARD_NOW} "
        "(오늘, 올해, 작년, 최근 90일, 지난 6개월, 3년 이상 전, 이번 분기, 현재 나이, ...). Follow the notes' rules "
        "for calendar units and for 'in the last N days, months or years'; the reference SQL writes today as "
        f"DATE '{HARD_NOW}' and never uses a clock function. Combine the relative window with other criteria (a "
        "diagnosis, a laboratory threshold, a visit type, a department) so that it is more than a date filter."),
    "unanswerable": (
        "Category unanswerable: questions a clinician or analyst could plausibly ask of this database, phrased as "
        "naturally as answerable ones, that the data cannot answer because no table or column holds what they "
        "need: information OMOP keeps in tables that are absent here (costs, insurance or payers); tables that are "
        "present but empty (clinical notes, specimens, devices, providers, care sites, locations); columns that are "
        "NULL in every row (times of day, causes of death, race and ethnicity, prescription doses and instructions, "
        "condition end dates, and the others the schema lists). A careful model must be able to tell from the "
        "schema and the notes that the data is missing, so do not ask about a clinical concept that might simply be "
        "absent from the vocabulary (such as a vital sign), and never ask something whose honest answer is 0, an "
        "empty list or a date outside 2021-2025: those are answerable. Vary the topics. sql is null; why_hard names "
        "the missing or empty table or column."),
}


# ---- the data profile shown to proposers ----

def _rows(con, sql: str) -> list[tuple]:
    return con.execute(sql).fetchall()


def _table(header: str, rows) -> str:
    return header + "\n" + "\n".join(" | ".join("" if v is None else str(v) for v in r) for r in rows)


def data_profile(con: duckdb.DuckDBPyConnection, *, top: int = 120) -> str:
    """Concepts in use (by number of patients) and data-quality facts, for proposers only: evaluated models see the
    schema with three example rows per table and the database note, never this list."""
    parts = [
        _table(f"Conditions in condition_occurrence (Condition domain), top {top} by patients:\n"
               "concept_id | concept_name | patients | records | KCD codes",
               _rows(con, f"""SELECT c.concept_id, c.concept_name, count(DISTINCT co.person_id) AS p, count(*),
                   array_to_string(list_slice(list(DISTINCT co.condition_source_value
                   ORDER BY co.condition_source_value), 1, 4), ',') FROM condition_occurrence co JOIN concept c ON c.concept_id = co.condition_concept_id
                   WHERE c.domain_id = 'Condition' GROUP BY 1, 2 ORDER BY p DESC, 1 LIMIT {top}""")),
        _table(f"Drugs in drug_exposure, top {top} by patients:\n"
               "concept_id | concept_name | vocabulary_id | concept_class_id | patients | records | ATC codes",
               _rows(con, f"""SELECT c.concept_id, c.concept_name, c.vocabulary_id, c.concept_class_id,
                   count(DISTINCT d.person_id) AS p, count(*),
                   array_to_string(list_slice(list(DISTINCT d.atc_cd ORDER BY d.atc_cd), 1, 3), ',')
                   FROM drug_exposure d JOIN concept c ON c.concept_id = d.drug_concept_id
                   GROUP BY 1, 2, 3, 4 ORDER BY p DESC, 1 LIMIT {top}""")),
        _table(f"Laboratory tests in measurement, top {top} by patients:\n"
               "concept_id | concept_name | patients | results | with value_as_number | most common unit",
               _rows(con, f"""WITH units AS (SELECT measurement_concept_id, unit_source_value, row_number() OVER
                   (PARTITION BY measurement_concept_id ORDER BY count(*) DESC, unit_source_value NULLS LAST) AS rk
                   FROM measurement GROUP BY 1, 2)
                   SELECT c.concept_id, c.concept_name, count(DISTINCT m.person_id) AS p, count(*),
                   count(m.value_as_number), u.unit_source_value FROM measurement m
                   JOIN concept c ON c.concept_id = m.measurement_concept_id
                   LEFT JOIN units u ON u.measurement_concept_id = m.measurement_concept_id AND u.rk = 1
                   GROUP BY 1, 2, 6 ORDER BY p DESC, 1 LIMIT {top}""")),
        _table(f"Procedures in procedure_occurrence, top {top} by patients:\n"
               "concept_id | concept_name | vocabulary_id | patients | records",
               _rows(con, f"""SELECT c.concept_id, c.concept_name, c.vocabulary_id, count(DISTINCT po.person_id) AS p,
                   count(*) FROM procedure_occurrence po JOIN concept c ON c.concept_id = po.procedure_concept_id
                   GROUP BY 1, 2, 3 ORDER BY p DESC, 1 LIMIT {top}""")),
        _table("Observations:\nconcept_id | concept_name | patients | records",
               _rows(con, """SELECT c.concept_id, c.concept_name, count(DISTINCT o.person_id) AS p, count(*)
                   FROM observation o JOIN concept c ON c.concept_id = o.observation_concept_id
                   GROUP BY 1, 2 ORDER BY p DESC, 1""")),
        _table("Visits by type:\nvisit_concept_id | patients | visits",
               _rows(con, "SELECT visit_concept_id, count(DISTINCT person_id), count(*) FROM visit_occurrence "
                          "GROUP BY 1 ORDER BY 1")),
        _table("Departments (visit_source_value), top 30 by visits:\ncode | visits | inpatient | emergency",
               _rows(con, "SELECT visit_source_value, count(*) AS n, count(*) FILTER (WHERE visit_concept_id = 9201), "
                          "count(*) FILTER (WHERE visit_concept_id = 9203) FROM visit_occurrence "
                          "GROUP BY 1 ORDER BY n DESC, 1 LIMIT 30")),
        "Data-quality facts:\n" + "\n".join(f"- {line}" for line in quality_facts(con)),
    ]
    return "\n\n".join(parts)


def quality_facts(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Counts behind the data-quality traps the harder set may use."""
    def one(sql: str) -> tuple:
        return con.execute(sql).fetchone()

    facts = []
    for table, column in (("drug_exposure", "drug_concept_id"), ("measurement", "measurement_concept_id"),
                          ("condition_occurrence", "condition_concept_id"),
                          ("procedure_occurrence", "procedure_concept_id")):
        n, k = one(f"SELECT count(*) FILTER (WHERE {column} = 0), count(DISTINCT person_id) FILTER (WHERE "
                   f"{column} = 0) FROM {table}")
        facts.append(f"{table}.{column} = 0 (No matching concept): {n} records of {k} patients")
    other = _rows(con, """SELECT c.domain_id, c.concept_name, count(*) FROM condition_occurrence co
                          JOIN concept c ON c.concept_id = co.condition_concept_id WHERE c.domain_id <> 'Condition'
                          GROUP BY 1, 2 ORDER BY 3 DESC, 2, 1""")
    facts.append("condition_occurrence records whose concept is not in the Condition domain: "
                 + "; ".join(f"{name} ({domain}) {n}" for domain, name, n in other))
    levels = _rows(con, """SELECT c.vocabulary_id, c.concept_class_id, count(*), count(DISTINCT c.concept_id)
                           FROM drug_exposure d JOIN concept c ON c.concept_id = d.drug_concept_id
                           GROUP BY 1, 2 ORDER BY 3 DESC, 1, 2""")
    facts.append("drug_exposure records by drug concept level: "
                 + "; ".join(f"{v}/{cls} {n} records, {k} concepts" for v, cls, n, k in levels))
    duplicates = {
        "drug_exposure": "person_id, drug_concept_id, drug_exposure_start_date, drug_exposure_end_date, quantity, "
                         "days_supply, visit_occurrence_id, atc_cd",
        "condition_occurrence": "person_id, condition_concept_id, condition_start_date, condition_type_concept_id, "
                                "visit_occurrence_id, condition_source_value",
        "visit_occurrence": "person_id, visit_concept_id, visit_start_date, visit_end_date, visit_source_value",
        "measurement": "person_id, measurement_concept_id, measurement_date, value_as_number, "
                       "measurement_source_value, visit_occurrence_id",
        "procedure_occurrence": "person_id, procedure_concept_id, procedure_date, visit_occurrence_id",
    }
    for table, cols in duplicates.items():
        groups, extra = one(f"SELECT count(*), coalesce(sum(k - 1), 0) FROM (SELECT count(*) AS k FROM {table} "
                            f"GROUP BY {cols} HAVING count(*) > 1)")
        facts.append(f"{table}: {groups} groups of duplicate records ({extra} extra rows) with equal {cols}")
    nulls = {"measurement": ["value_as_number", "range_low", "range_high", "unit_source_value"],
             "drug_exposure": ["atc_cd", "days_supply", "quantity", "route_concept_id"],
             "visit_occurrence": ["admitted_from_source_value"]}
    for table, cols in nulls.items():
        counts = one(f"SELECT count(*), {', '.join(f'count(*) FILTER (WHERE {c} IS NULL)' for c in cols)} FROM {table}")
        facts.append(f"{table} ({counts[0]} rows), NULL in: "
                     + ", ".join(f"{c} {n}" for c, n in zip(cols, counts[1:])))
    neg = one("SELECT count(*), count(DISTINCT measurement_concept_id) FROM measurement WHERE value_as_number < 0")
    facts.append(f"measurement.value_as_number < 0: {neg[0]} results of {neg[1]} tests")
    zero_from = one("SELECT count(*) FROM visit_occurrence WHERE admitted_from_concept_id = 0")[0]
    facts.append(f"visit_occurrence.admitted_from_concept_id = 0: {zero_from} visits")
    deaths = one("SELECT count(*) FROM death")[0]
    facts.append(f"death: {deaths} patients; cause_concept_id and cause_source_value are NULL for all of them")
    return facts


# ---- generation ----

@dataclass
class Batch:
    category: str
    index: int  # k-th batch of its category
    size: int
    proposer: str


def plan_batches(models: list[str], plan: dict[str, list[int]] = PLAN) -> list[Batch]:
    """Batch k of the j-th category (in HARD_CATEGORIES order) goes to models[(j + k) % len(models)]."""
    batches = []
    for j, category in enumerate(c for c in HARD_CATEGORIES if c in plan):
        for k, size in enumerate(plan[category]):
            batches.append(Batch(category, k, size, models[(j + k) % len(models)]))
    return batches


def _accept_item(body: dict, category: str) -> dict | None:
    ko, en, sql, order = body.get("question_ko"), body.get("question_en"), body.get("sql"), body.get("order_matters",
                                                                                                   False)
    if not all(isinstance(v, str) and v.strip() for v in (ko, en)) or not isinstance(order, bool):
        return None
    answerable = category != "unanswerable"
    if answerable != (isinstance(sql, str) and bool(sql.strip())):
        return None
    scope = body.get("scope") if body.get("scope") in ("individual", "group", "none") else "group"
    return {"category": category, "question_ko": " ".join(ko.split()), "question_en": " ".join(en.split()),
            "sql": clean_sql(sql) if answerable else None, "order_matters": order and answerable, "scope": scope,
            "why_hard": " ".join(str(body.get("why_hard") or "").split())}


def parse_items(text: str | None, category: str) -> list[dict] | None:
    """The items of a proposer's JSON answer ({"items": [...]}, or a bare list); malformed items are left out and
    every item takes the batch's category. None when there is no such JSON."""
    def accept(body):
        rows = body.get("items") if isinstance(body, dict) else None
        if not isinstance(rows, list):
            return None
        return [a for a in (_accept_item(r, category) for r in rows if isinstance(r, dict)) if a is not None]
    found = parse_json_object(text, accept)
    if found is not None:
        return found
    match = re.search(r"\[.*\]", text or "", re.S)
    try:
        rows = json.loads(match.group(0), strict=False) if match else None
    except json.JSONDecodeError:
        return None
    return accept({"items": rows}) if isinstance(rows, list) else None


def as_question(qid: str, raw: dict) -> Question:
    return Question(id=qid, question_ko=raw["question_ko"], question_en=raw["question_en"], sql=raw["sql"],
                    category=raw["category"], difficulty="hard", scope=raw.get("scope", "group"),
                    order_matters=raw.get("order_matters", False), answerable=raw["category"] != "unanswerable",
                    why_hard=raw.get("why_hard") or None)


def item_problem(con, item: Question, others: list[Question]) -> str | None:
    """Why a generated item cannot enter the draft: v0's reference rules, or a quoted concept name."""
    return reference_problem(con, item, others) or quoted_problem(con, item)


def _brief(raw: dict) -> str:
    return raw["question_en"]


def render_generate_messages(batch: Batch, schema: str, profile: str, written: list[dict], *,
                             hint: str = "") -> list[dict]:
    already = "\n".join(f"- {_brief(r)}" for r in written) or "- none yet"
    user = (f"Database schema (DuckDB), as the evaluated model sees it:\n\n{schema}\n\n"
            f"The database notes the evaluated model receives:\n{HARD_PROMPT.conventions}\n\n"
            f"The evaluated model is also told: {HARD_PROMPT.system}\n\n"
            f"Data profile (for you, not shown to evaluated models):\n\n{profile}"
            f"{GUIDE_START}{GUIDES[batch.category][len('Category '):]}\n\n"
            f"Items already written in this category:\n{already}\n\n" + (f"{hint}\n\n" if hint else "")
            + f'Write {batch.size} new items of category "{batch.category}".')
    return [{"role": "system", "content": GEN_SYSTEM}, {"role": "user", "content": user}]


def render_repair_messages(batch: Batch, generated: list[dict], failures: list[tuple[dict, str]]) -> list[dict]:
    """The proposer's own generation prompt and answer, then the items that broke the rules, to fix or replace."""
    listing = "\n\n".join(
        f"{n}. Korean: {raw['question_ko']}\nEnglish: {raw['question_en']}\nSQL:\n```sql\n{raw['sql']}\n```\n"
        f"Problem: {problem}" for n, (raw, problem) in enumerate(failures, start=1))
    return generated + [{"role": "user", "content": (
        f"These {len(failures)} items break the rules and cannot enter the benchmark:\n\n{listing}\n\n"
        f"Return exactly {len(failures)} corrected items (or new items of the same category that keep to every "
        'rule) in the same JSON format: {"items": [...]}.')}]


@dataclass
class GenerationResult:
    items: list[dict] = field(default_factory=list)  # accepted, in plan order, each with proposer and gen_round
    batches: list[dict] = field(default_factory=list)  # per batch: proposed, passed, sent back, repaired
    sent_back: list[dict] = field(default_factory=list)  # items that failed the first check, with the problem
    dropped: list[dict] = field(default_factory=list)  # items that failed after the repair, with the problem
    incomplete: list[str] = field(default_factory=list)  # batches the budget or an error stopped


async def run_generate(database: Path, clients: dict, prices: dict[str, tuple[float, float]], out: Path,
                       budget: Budget, *, models: list[str], existing: list[Question], gen_round: int = 1,
                       plan: dict[str, list[int]] = PLAN, hint: str = "", extra: dict | None = None
                       ) -> GenerationResult:
    """Generate, check and repair every batch of the plan. Categories run in parallel, a category's batches one
    after another (each sees the items accepted before it). Answers are cached per model, batch and prompt."""
    con = connect_readonly(database)
    schema = schema_summary(con, null_columns=HARD_PROMPT.null_columns)
    profile = data_profile(con)
    asker = Asker(clients, prices, out, budget, max_tokens=GEN_MAX_TOKENS, concurrency=2, extra=extra,
                  cache_split=GUIDE_START)
    result = GenerationResult()
    accepted: dict[str, list[dict]] = {c: [] for c in plan}
    batches = plan_batches(models, plan)

    def problems(raws: list[dict], taken: list[Question]) -> tuple[list[dict], list[tuple[dict, str]]]:
        ok, bad = [], []
        for raw in raws:
            q = as_question("new", raw)
            problem = item_problem(con, q, taken + [as_question(f"#{n}", r) for n, r in enumerate(ok)])
            (bad.append((raw, problem)) if problem else ok.append(raw))
        return ok, bad

    def taken_now() -> list[Question]:
        return existing + [as_question(f"~{c}{n}", r) for c, rows in accepted.items() for n, r in enumerate(rows)]

    async def category(name: str) -> None:
        written = [{"question_en": q.question_en} for q in existing if q.category == name]
        for batch in (b for b in batches if b.category == name):
            key = f"{name}/g{gen_round}/b{batch.index + 1}"
            messages = render_generate_messages(batch, schema, profile, written + accepted[name], hint=hint)
            answer = await asker.ask(batch.proposer, "generate", key, messages, lambda t: parse_items(t, name))
            if answer.status in ("error", "budget"):
                result.incomplete.append(key)
                return
            raws = answer.value or []
            ok, bad = problems(raws, taken_now())
            log = {"batch": key, "proposer": batch.proposer, "asked": batch.size, "proposed": len(raws),
                   "passed": len(ok), "sent_back": len(bad), "repaired": 0}
            tag = {"proposer": batch.proposer, "batch": key}
            if bad:
                result.sent_back += [{**raw, "problem": p, **tag} for raw, p in bad]
                followup = messages + [{"role": "assistant", "content": answer.raw}]
                fix = await asker.ask(batch.proposer, "repair", key, render_repair_messages(batch, followup, bad),
                                      lambda t: parse_items(t, name))
                if fix.status in ("error", "budget"):
                    result.incomplete.append(key + "/repair")
                fixed_ok, fixed_bad = problems(fix.value or [], taken_now() + [as_question(f"!{n}", r)
                                                                               for n, r in enumerate(ok)])
                log["repaired"] = len(fixed_ok)
                ok += fixed_ok
                result.dropped += [{**raw, "problem": p, **tag} for raw, p in fixed_bad]
            result.batches.append(log)
            for raw in ok:
                raw.update({"proposer": batch.proposer, "gen_round": gen_round, "batch": key})
            accepted[name] += ok

    await asyncio.gather(*(category(c) for c in dict.fromkeys(b.category for b in batches)))
    # Duplicates across categories (written in parallel) are dropped from the later category.
    final, taken = [], list(existing)
    for name in (c for c in HARD_CATEGORIES if c in accepted):
        for raw in accepted[name]:
            q = as_question(f"#{len(final)}", raw)
            problem = item_problem(con, q, taken)
            if problem:
                result.dropped.append({**raw, "problem": problem, "batch": "final check"})
                continue
            final.append(raw)
            taken.append(q)
    result.items = final
    return result


# ---- the draft file ----

def next_ids(existing: list[Question], n: int) -> list[str]:
    numbers = [int(q.id[len(ID_PREFIX):]) for q in existing if q.id.startswith(ID_PREFIX)
               and q.id[len(ID_PREFIX):].isdigit()]
    start = max(numbers, default=0) + 1
    return [f"{ID_PREFIX}{k:03d}" for k in range(start, start + n)]


def render_item(qid: str, raw: dict) -> str:
    lines = [f"  - id: {qid}", f"    category: {raw['category']}", "    difficulty: hard",
             f"    scope: {raw.get('scope', 'group')}"]
    if raw["category"] == "unanswerable":
        lines.append("    answerable: false")
    lines += [f"    question_ko: {_scalar(raw['question_ko'])}", f"    question_en: {_scalar(raw['question_en'])}"]
    if raw["sql"]:
        lines += ["    sql: |", *(f"      {line}" if line.strip() else "" for line in raw["sql"].split("\n"))]
    else:
        lines.append("    sql: null")
    if raw.get("order_matters"):
        lines.append("    order_matters: true")
    if raw.get("why_hard"):
        lines.append(f"    why_hard: {_scalar(raw['why_hard'])}")
    lines += [f"    proposed_by: {raw['proposer']}", f"    generation_batch: {raw['batch']}", "    review: pending"]
    return "\n".join(lines)


def batch_round(key: str | None) -> int | None:
    """The generation round of a batch key such as 'clinical_terms/g1/b2'."""
    match = re.fullmatch(r"[a-z_]+/g(\d+)/b\d+", key or "")
    return int(match.group(1)) if match else None


def draft_batches(draft: Path) -> dict[str, str]:
    """item id -> the generation batch it came from, for the items of the draft that record one."""
    doc = yaml.safe_load(Path(draft).read_text(encoding="utf-8")) or {}
    return {q["id"]: q["generation_batch"] for q in doc.get("questions") or [] if q.get("generation_batch")}


def parse_plan(text: str) -> dict[str, list[int]]:
    """'clinical_terms=8,8,8;unanswerable=7,7' -> {category: batch sizes}."""
    plan = {}
    for part in filter(None, (p.strip() for p in text.split(";"))):
        name, _, sizes = part.partition("=")
        if name.strip() not in HARD_CATEGORIES:
            raise ValueError(f"unknown category {name.strip()!r} (one of {', '.join(HARD_CATEGORIES)})")
        plan[name.strip()] = [int(s) for s in sizes.split(",") if s.strip()]
    return plan


def append_items(draft: Path, raws: list[dict]) -> list[str]:
    """Append generated items (review: pending) at the end of the draft's questions, one comment line per category,
    and check that the file still reads back with them. Returns the new ids."""
    text = Path(draft).read_text(encoding="utf-8")
    _, existing = load_questions(draft)
    ids = next_ids(existing, len(raws))
    if re.search(r"^questions:\s*\[\]\s*$", text, re.M):
        text = re.sub(r"^questions:\s*\[\]\s*$", "questions:", text, flags=re.M)
    elif not re.search(r"^questions:\s*$", text, re.M):
        raise ValueError("the draft has no questions: key")
    blocks, last = [], None
    for qid, raw in zip(ids, raws):
        if raw["category"] != last:
            blocks.append(f"# ---------------------------------------------------------------- {raw['category']} "
                          f"(generation round {raw['gen_round']})")
            last = raw["category"]
        blocks.append(render_item(qid, raw))
    text = text.rstrip("\n") + "\n\n" + "\n\n".join(blocks) + "\n"
    _check_and_write(draft, text, expect=len(existing) + len(raws))
    return ids


def _check_and_write(draft: Path, text: str, *, expect: int | None = None) -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "draft.yaml"
        probe.write_text(text, encoding="utf-8")
        _, items = load_questions(probe)
    if expect is not None and len(items) != expect:
        raise ValueError(f"the draft would read back {len(items)} items, not {expect}")
    Path(draft).write_text(text, encoding="utf-8")


# ---- review ----

REVIEW_BATCH = 5  # items per review request: the shared schema prompt is paid once per batch, not per item


async def run_hard_review(items: list[Question], database: Path, clients: dict, prices: dict[str, tuple[float, float]],
                          out: Path, budget: Budget, *, extra: dict | None = None, batch_size: int = REVIEW_BATCH
                          ) -> dict[str, dict[str, dict | None] | None]:
    """Every model reviews every pending item with the harder set's rubric, `batch_size` items per request; an item
    whose verdict is missing from a batch answer is asked again on its own. item id -> {model: review or None}, or
    None when a request failed or the budget stopped it (asked again on the next run)."""
    con = connect_readonly(database)
    schema = schema_summary(con, null_columns=HARD_PROMPT.null_columns)
    asker, batch_asker = review_askers(clients, prices, out, budget, extra)
    pending = [i for i in items if i.review == "pending"]
    return await review_items(asker, batch_asker, list(clients), pending, con, schema, HARD_PROMPT, "r0", batch_size)


def review_decision(reviews: dict[str, dict | None]) -> tuple[str, str]:
    """(review state, review_note) for the draft: approved only when all approve; otherwise needs_edit (for
    `llm-fix`), with every non-approving reviewer's reason and fix."""
    decision, note = aggregate(reviews)
    if decision == "approved":
        return "approved", ""
    verdicts = Counter((r or {}).get("verdict") for r in reviews.values())
    head = (f"Three-model review ({REVIEW_DATE}): {verdicts['approve']} of {len(reviews)} approved"
            + (f", {verdicts['reject']} rejected" if verdicts["reject"] else "") + "; sent to llm-fix.")
    return "needs_edit", head + "\n" + note


def apply_reviews(draft: Path, reviews: dict[str, dict[str, dict | None] | None]) -> dict[str, int]:
    """Write review (approved or needs_edit) and review_note for every completely reviewed item, editing only those
    items' lines."""
    text = Path(draft).read_text(encoding="utf-8")
    counts: Counter = Counter()
    for qid, item_reviews in reviews.items():
        if item_reviews is None:
            continue
        state, note = review_decision(item_reviews)
        values = {"review": [f"    review: {state}"],
                  "review_note": [f"    review_note: {_yaml_str(note)}"] if note else None}
        start, end = _item_span(text, qid)
        text = text[:start] + _rewrite_item(text[start:end], values) + text[end:]
        counts[state] += 1
    _check_and_write(draft, text)
    return dict(counts)


def send_back(draft: Path, ids: list[str], reason: str) -> list[str]:
    """Set items to needs_edit with `reason` as their review_note (e.g. a decision on the draft that `llm-fix` is to
    carry out), editing only those lines. Refuses unknown ids."""
    _, items = load_questions(draft)
    known = {i.id for i in items}
    unknown = [i for i in ids if i not in known]
    if unknown:
        raise ValueError(f"no such items: {', '.join(unknown)}")
    text = Path(draft).read_text(encoding="utf-8")
    for qid in dict.fromkeys(ids):
        start, end = _item_span(text, qid)
        text = text[:start] + _rewrite_item(text[start:end], {"review": ["    review: needs_edit"],
                                                              "review_note": [f"    review_note: {_yaml_str(reason)}"]}
                                            ) + text[end:]
    _check_and_write(draft, text, expect=len(items))
    return list(dict.fromkeys(ids))


def collect_reviews(out: Path, models: list[str], items: list[Question], current=None
                    ) -> dict[str, dict[str, dict | None]]:
    """item id -> {model: its review in the harder set's review step}, from the job's review caches. With
    `current(group)` (the hash of the review prompt for those items: one item alone, or several as a batch), a
    review given on another wording counts as none: a batch answer only if every item in it is unchanged, a single
    answer only if its item is."""
    by_id = {i.id: i for i in items}
    shas: dict[tuple[str, ...], str | None] = {}

    def unchanged(record: dict, ids: list[str]) -> bool:
        if current is None:
            return True
        if any(i not in by_id for i in ids):
            return False
        if tuple(ids) not in shas:
            shas[tuple(ids)] = current([by_id[i] for i in ids])
        return record.get("prompt_sha") == shas[tuple(ids)]

    found: dict[str, dict[str, dict | None]] = {}
    for model in models:
        path = Path(out) / f"{model.replace('/', '__')}.jsonl"
        if not path.exists():
            continue
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for r in records:  # batch answers first; an item asked again on its own overrides its batch entry
            if r.get("kind") == "review-batch" and isinstance(r.get("answer"), dict) and \
                    unchanged(r, r["key"].split("/", 1)[1].split(",")):
                for qid, review in r["answer"].items():
                    found.setdefault(qid, {})[model] = review
        for r in records:
            answer = r.get("answer") if r.get("answer") is not None else parse_review(r.get("raw"))
            if r.get("kind") == "review" and r["key"].endswith("/r0") and answer is not None and \
                    unchanged(r, [r["key"][:-3]]):
                found.setdefault(r["key"][:-3], {})[model] = answer
    ids = {i.id for i in items}
    return {qid: {m: reviews.get(m) for m in models} for qid, reviews in found.items() if qid in ids}


def spend_report(root: Path) -> dict:
    """Spend recorded under the job folder: what the route billed, estimates where it did not say, and the
    estimates charged for failed attempts, by step."""
    out: dict = {"route_usd": 0.0, "estimate_usd": 0.0, "failed_attempts_usd": 0.0, "requests": 0, "by_step": {}}
    for path in sorted(Path(root).rglob("*.jsonl")) if Path(root).exists() else []:
        step = path.parent.name
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            cost = float(r.get("cost_usd") or 0)
            key = "route_usd" if r.get("cost_source") == "route" else "estimate_usd"
            out[key] += cost
            out["failed_attempts_usd"] += float(r.get("failed_attempts_usd") or 0)
            out["requests"] += 1
            s = out["by_step"].setdefault(step, {"usd": 0.0, "requests": 0})
            s["usd"] += cost + float(r.get("failed_attempts_usd") or 0)
            s["requests"] += 1
    return out
