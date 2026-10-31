"""Text-to-SQL over OMOP CDM v5.4 (Synthetic-SNUH v4, DuckDB): question set, prompt, execution-accuracy scorer.

The items, prompts and result comparison of the frozen task synthetic-snuh-text2sql@0
(registry/tasks/synthetic-snuh-text2sql.yaml; its runner and metrics are in text2sql_task.py). The question files
are in registry/drafts/, frozen and pinned by sha256 in the task spec; they are released on KHDP as the dataset
synthetic-snuh-text2sql (text2sql_release.py), not with the code.
`run_dry_run` asks a model outside the medbench database and caches its answers in a folder, which the task's runner
can reuse; nothing here writes to the medbench database.

Scoring is execution accuracy: the model's SQL and the reference SQL run on the same read-only database and their
results are compared under the rules in `results_match` and `canonical_value` (adapted from the Spider test-suite
evaluation and EHRSQL; see the draft file's `protocol` block for citations). The harder set's prompt
(`HARD_PROMPT`) fixes "today" and lets a model abstain (ABSTAIN) on questions the data cannot answer; it is scored
with execution accuracy on the answerable items and the EHRSQL-2024 reliability score RS(c) over all of them.
"""
import asyncio
import datetime as dt
import hashlib
import itertools
import json
import math
import re
import threading
import uuid
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import duckdb
import yaml

from benchtoward.llm import ChatResult
from benchtoward.scoring import accuracy_ci

PROMPT_ID = "text2sql-omop-p0"
SCORER_VERSION = "ex-0.1"
RS_SCORER_VERSION = "ex-rs-0.1"  # execution accuracy on answerable items plus the EHRSQL-2024 reliability score
LANGS = ("ko", "en")
CATEGORIES = ("patient_lookup", "cohort_count", "aggregation", "temporal", "multi_domain_join", "vocabulary_lookup")
# The harder set (draft synthetic-snuh-text2sql-hard-v0): what makes each item hard.
HARD_CATEGORIES = ("clinical_terms", "cohort_definition", "data_quality", "relative_time", "unanswerable")
ALL_CATEGORIES = CATEGORIES + HARD_CATEGORIES
ABSTAIN = "ABSTAIN"  # a model's answer that it cannot answer the question from this database (EHRSQL's null)
RS_PENALTIES = (0, 5)  # RS(c) for these c, and RS(N) with N = the number of scored instances (EHRSQL-2024)
HARD_NOW = "2025-12-31"  # the fixed "today" of the harder set's relative-time questions
DIFFICULTIES = ("easy", "medium", "hard")
REVIEW_STATES = ("pending", "approved", "needs_edit", "rejected")
SCOPES = ("individual", "group", "none")  # EHRSQL patient scope: one patient, a cohort, no patient filter
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_ROWS = 10_000  # a prediction returning more rows than this is wrong (references stay under 1,000)
MAX_REFERENCE_ROWS = 1_000
FLOAT_DECIMALS = 3  # EHRSQL-2024 scorer: round(float(x), 3)

SYSTEM_PROMPT = (
    "You translate clinical questions into SQL for an OMOP CDM v5.4 database stored in DuckDB. "
    "Write one read-only DuckDB query (SELECT, optionally with WITH clauses) that answers the question. "
    "Return exactly the columns the question asks for; column order and names do not matter, but extra or missing "
    "columns make the answer wrong. Think as needed, then give the final query in a ```sql code block at the end."
)

CONVENTIONS = """Notes on this database:
- Synthetic data for 10,000 patients modelled on Seoul National University Hospital (SNUH); dates fall between 2021-01-01 and 2025-12-31. Do not use the current date.
- Concept names and codes are in table concept: join a *_concept_id column to concept.concept_id. Questions quote concept names exactly as they appear in concept.concept_name.
- "Patients" means distinct person_id. A patient's identifier is person.person_source_value (e.g. 'SYN-000123').
- Age in years at an event = the year of the event date minus person.year_of_birth.
- Date ranges are inclusive. "In 2023" refers to the event's own date (visit_start_date for visits, condition_start_date for diagnoses, drug_exposure_start_date for prescriptions, measurement_date for laboratory results, procedure_date for procedures).
- visit_occurrence.visit_source_value is the SNUH department code of the visit (e.g. 'IMC'). Every visit has exactly one diagnosis with condition_type_concept_id 44786627 (Primary Condition); other diagnoses are 44786629 (Secondary Condition).
- condition_occurrence.condition_source_value holds the KCD-7 code of a diagnosis (e.g. 'E11.9'); drug_exposure.atc_cd holds the ATC code of a prescription when known; measurement.measurement_source_value is the SNUH laboratory code.
- When a question asks for a rounded number, use ROUND(x, n). Percentages are on a 0-100 scale."""

HARD_SYSTEM_PROMPT = SYSTEM_PROMPT + (
    " Some questions cannot be answered from this database, because the data they need is not recorded in it. "
    "For such a question do not guess and do not write a query: make your final code block a ```sql block that "
    f"contains only the word {ABSTAIN}.")

# The harder set's note: v0's, with a fixed "now", concepts in clinical words and the data-quality conventions its
# traps rely on (v0's lines from "Patients" on are kept as they are).
HARD_CONVENTIONS = "\n".join([
    "Notes on this database:",
    "- Synthetic data for 10,000 patients modelled on Seoul National University Hospital (SNUH); dates fall between "
    "2021-01-01 and 2025-12-31.",
    f'- Today is {HARD_NOW} for every question (a fixed "now"; never use the system clock, such as current_date or '
    'now()). "This year" is 2025 and "last year" is 2024 (calendar years; months and quarters likewise). "In the '
    'last N days, months or years" means on or after today minus N days, months or years (e.g. '
    f"DATE '{HARD_NOW}' - INTERVAL 6 MONTH) up to today; \"more than N months ago\" means before that date. A "
    "patient's age now = 2025 minus person.year_of_birth.",
    "- Questions name concepts in clinical words (Korean terms, synonyms, abbreviations) instead of quoting concept "
    "names. Such a term means the concept whose concept_name is its standard name in the vocabulary (SNOMED CT for "
    "conditions and procedures, RxNorm ingredients for drugs, LOINC for laboratory tests), not broader or more "
    "specific concepts, unless the question says otherwise (for example a KCD code range). Concept names and codes "
    "are in table concept: join a *_concept_id column to concept.concept_id.",
    "- concept_id 0 ('No matching concept') marks a record that is not mapped to a concept; such a record never "
    "counts as a drug, laboratory test or diagnosis.",
    "- condition_occurrence also holds a few records whose concept is not in the Condition domain (concept.domain_id "
    "Measurement, Observation or Procedure); a diagnosis or disease means a record whose concept is in the Condition "
    "domain.",
    "- Some tables hold duplicate records that differ only in their id. Count records as they are stored unless the "
    "question says to count duplicates once.",
    *CONVENTIONS.split("\n")[3:],
])


@dataclass(frozen=True)
class PromptSpec:
    """What an evaluated model is told: the system prompt, the database note, whether it may abstain (scored with
    the reliability score) and whether the schema lists the columns that are NULL in every row."""
    id: str
    system: str
    conventions: str
    abstention: bool = False
    null_columns: bool = False


P0 = PromptSpec(PROMPT_ID, SYSTEM_PROMPT, CONVENTIONS)
HARD_PROMPT = PromptSpec("text2sql-omop-hard-p0", HARD_SYSTEM_PROMPT, HARD_CONVENTIONS, abstention=True,
                         null_columns=True)
PROMPTS = {p.id: p for p in (P0, HARD_PROMPT)}


def prompt_spec(meta: dict) -> PromptSpec:
    """The prompt a question file names in `prompt.id` (v0's when it names none)."""
    pid = ((meta or {}).get("prompt") or {}).get("id") or PROMPT_ID
    if pid not in PROMPTS:
        raise ValueError(f"unknown prompt id {pid!r} (known: {', '.join(PROMPTS)})")
    return PROMPTS[pid]


_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.S)
_SQL_START = re.compile(r"^\s*(WITH|SELECT|FROM)\b", re.I | re.M)
_NUMERIC_STRING = re.compile(r"[+-]?\d+(\.\d+)?")
_CLOCK = re.compile(r"\b(current_date|current_timestamp|current_time|now\s*\(|today\s*\(|get_current_\w+)", re.I)
EXEC_ERRORS = ("empty", "syntax_error", "not_select", "multiple_statements", "execution_error", "timeout",
               "too_many_rows")


@dataclass(frozen=True)
class Question:
    id: str
    question_ko: str
    question_en: str
    sql: str | None  # None only for an unanswerable item, whose correct answer is to abstain
    category: str
    difficulty: str
    scope: str = "group"
    order_matters: bool = False
    note: str | None = None
    review: str = "pending"
    answerable: bool = True
    why_hard: str | None = None  # the harder set: what makes the item hard (or why the data cannot answer it)

    def text(self, lang: str) -> str:
        if lang not in LANGS:
            raise ValueError(f"language must be one of {LANGS}, not {lang!r}")
        return self.question_ko if lang == "ko" else self.question_en


def instance_id(question_id: str, lang: str) -> str:
    return f"{question_id}-{lang}"


def load_questions(path: Path) -> tuple[dict, list[Question]]:
    """Read a question file: top-level metadata plus `questions`. Raises ValueError on malformed items."""
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    meta = {k: v for k, v in doc.items() if k != "questions"}
    items, seen = [], set()
    for raw in doc.get("questions") or []:
        qid = raw.get("id", "?")
        answerable = raw.get("answerable", True)
        if not isinstance(answerable, bool):
            raise ValueError(f"{qid}: answerable must be true or false")
        required = ("id", "question_ko", "question_en", "sql", "category", "difficulty") if answerable else \
            ("id", "question_ko", "question_en", "category", "difficulty")
        for field in required:
            if not str(raw.get(field) or "").strip():
                raise ValueError(f"{qid}: missing {field}")
        if qid in seen:
            raise ValueError(f"duplicate question id {qid}")
        seen.add(qid)
        if raw["category"] not in ALL_CATEGORIES:
            raise ValueError(f"{qid}: category {raw['category']!r} is not one of {ALL_CATEGORIES}")
        if not answerable and str(raw.get("sql") or "").strip():
            raise ValueError(f"{qid}: an unanswerable item has no reference sql")
        if (raw["category"] == "unanswerable") == answerable:
            raise ValueError(f"{qid}: an item is in category unanswerable exactly when answerable is false")
        if raw["difficulty"] not in DIFFICULTIES:
            raise ValueError(f"{qid}: difficulty {raw['difficulty']!r} is not one of {DIFFICULTIES}")
        if raw.get("scope", "group") not in SCOPES:
            raise ValueError(f"{qid}: scope {raw.get('scope')!r} is not one of {SCOPES}")
        review = raw.get("review", "pending")
        if review not in REVIEW_STATES:
            raise ValueError(f"{qid}: review {review!r} is not one of {REVIEW_STATES}")
        items.append(Question(id=qid, question_ko=raw["question_ko"].strip(), question_en=raw["question_en"].strip(),
                              sql=raw["sql"].strip() if answerable else None, category=raw["category"],
                              difficulty=raw["difficulty"], scope=raw.get("scope", "group"),
                              order_matters=bool(raw.get("order_matters", False)), note=raw.get("note"),
                              review=review, answerable=answerable, why_hard=raw.get("why_hard")))
    return meta, items


# ---- execution ----

@dataclass
class ExecResult:
    columns: list[str]
    rows: list[tuple]
    error: str | None = None  # one of EXEC_ERRORS
    detail: str | None = None


def execute_sql(con: duckdb.DuckDBPyConnection, sql: str | None, *, timeout_s: float = DEFAULT_TIMEOUT_S,
                max_rows: int = DEFAULT_MAX_ROWS) -> ExecResult:
    """Run one SELECT on a read-only connection with a wall-clock limit. Never raises for bad SQL."""
    sql = (sql or "").strip().rstrip(";").strip()
    if not sql:
        return ExecResult([], [], "empty")
    try:
        statements = con.extract_statements(sql)
    except duckdb.Error as exc:
        return ExecResult([], [], "syntax_error", str(exc)[:500])
    if len(statements) != 1:
        return ExecResult([], [], "multiple_statements" if statements else "empty")
    if statements[0].type != duckdb.StatementType.SELECT:
        return ExecResult([], [], "not_select", str(statements[0].type))

    lock, state = threading.Lock(), {"running": True, "fired": False}

    def stop() -> None:
        with lock:
            if state["running"]:
                state["fired"] = True
                con.interrupt()

    timer = threading.Timer(timeout_s, stop)
    timer.start()
    try:
        cursor = con.execute(sql)
        columns = [d[0] for d in cursor.description or []]
        rows = cursor.fetchmany(max_rows + 1)
    except duckdb.Error as exc:
        with lock:
            state["running"] = False
        if state["fired"] or isinstance(exc, duckdb.InterruptException):
            return ExecResult([], [], "timeout", f"over {timeout_s:g} s")
        return ExecResult([], [], "execution_error", str(exc)[:500])
    finally:
        with lock:
            state["running"] = False
        timer.cancel()
    if len(rows) > max_rows:
        return ExecResult(columns, [], "too_many_rows", f"more than {max_rows} rows")
    return ExecResult(columns, [tuple(r) for r in rows])


# ---- result comparison ----

def canonical_value(value):
    """Normalise one cell so that equal answers compare equal regardless of SQL type.

    Numbers compare by value after rounding to 3 decimals, as in the EHRSQL-2024 scorer (3 == 3.0 == DECIMAL '3.00'
    == '3'; booleans are 1/0; NaN is NULL); dates and midnight timestamps become 'YYYY-MM-DD'; intervals become days;
    other strings are stripped but case-sensitive; NULL equals only NULL (never the string 'None').
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, (float, Decimal)):
        f = float(value)
        if math.isnan(f):
            return None
        if math.isinf(f):
            return f
        r = round(f, FLOAT_DECIMALS)
        return int(r) if r == int(r) and abs(r) < 2 ** 53 else r
    if isinstance(value, dt.datetime):
        if value.tzinfo is None and value.time() == dt.time(0):
            return value.date().isoformat()
        return value.isoformat(sep=" ")
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, dt.timedelta):
        return canonical_value(value.total_seconds() / 86400)
    if isinstance(value, str):
        text = value.strip()
        return canonical_value(Decimal(text)) if _NUMERIC_STRING.fullmatch(text) else text
    if isinstance(value, (list, tuple)):
        return tuple(canonical_value(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((str(k), canonical_value(v)) for k, v in value.items()))
    if isinstance(value, uuid.UUID):
        return str(value)
    return value if isinstance(value, bytes) else str(value)


def _canonical_rows(rows) -> list[tuple]:
    return [tuple(canonical_value(v) for v in row) for row in rows]


def _permutations(ref: list[tuple], pred: list[tuple], ncol: int, cap: int = 5040):
    """Column mappings worth trying: predicted column i may stand for reference column j only if their value
    multisets agree (Spider test-suite pruning)."""
    ref_cols = [Counter(r[j] for r in ref) for j in range(ncol)]
    pred_cols = [Counter(r[i] for r in pred) for i in range(ncol)]
    candidates = [[i for i in range(ncol) if pred_cols[i] == ref_cols[j]] for j in range(ncol)]
    for n, perm in enumerate(p for p in itertools.product(*candidates) if len(set(p)) == ncol):
        if n >= cap:
            return
        yield perm


def results_match(reference, predicted, *, order_matters: bool = False) -> bool:
    """Execution match: same number of rows and columns; rows compared as multisets (duplicates count) unless the
    question asks for an order; column order and names ignored; values compared after `canonical_value`."""
    ref, pred = _canonical_rows(reference), _canonical_rows(predicted)
    if len(ref) != len(pred):
        return False
    if not ref:
        return True
    ncol = len(ref[0])
    if any(len(r) != ncol for r in ref) or any(len(r) != ncol for r in pred):
        return False
    target = ref if order_matters else Counter(ref)
    for perm in _permutations(ref, pred, ncol):
        permuted = [tuple(r[i] for i in perm) for r in pred]
        if (permuted if order_matters else Counter(permuted)) == target:
            return True
    return False


# ---- prompt ----

def extract_sql(text: str | None) -> str | None:
    """The last fenced code block (```sql preferred), else the text from the first SELECT/WITH line on."""
    text = text or ""
    blocks = _FENCE.findall(text)
    sql_blocks = [body for lang, body in blocks if lang.lower() in ("sql", "duckdb", "postgresql", "")]
    body = (sql_blocks or [b for _, b in blocks] or [None])[-1]
    if body is None:
        match = _SQL_START.search(text)
        body = text[match.start():] if match else None
    if body is None:
        return None
    body = body.strip().rstrip(";").strip()
    return body or None


def _is_abstain(text: str | None) -> bool:
    return (text or "").strip().strip("*`_.;: \t").upper() == ABSTAIN


def normalize_prediction(value: str | None, *, abstention: bool) -> str | None:
    """A prediction read from a file, as the scorer takes it. With abstention (the harder set), JSON null, the string
    'null' and ABSTAIN in any case (surrounding spaces ignored, as in a model's answer) abstain; an empty string is no
    answer (scored -c). Without abstention (v0) null and an empty string are no answer."""
    if value is None:
        return ABSTAIN if abstention else None
    text = str(value).strip()
    if not text:
        return None
    if abstention and (text.lower() == "null" or _is_abstain(text)):
        return ABSTAIN
    return text


def extract_answer(text: str | None) -> str | None:
    """For prompts that allow abstention: ABSTAIN when the final answer abstains (the last code block holds only
    the word, or with no code block the last non-empty line is the word), else `extract_sql`."""
    text = text or ""
    blocks = _FENCE.findall(text)
    if blocks:
        return ABSTAIN if _is_abstain(blocks[-1][1]) else extract_sql(text)
    lines = [line for line in text.splitlines() if line.strip()]
    if lines and _is_abstain(lines[-1]):
        return ABSTAIN
    return extract_sql(text)


def _cell(value) -> str:
    text = "NULL" if value is None else str(value)
    return text if len(text) <= 60 else text[:57] + "..."


def all_null_columns(con: duckdb.DuckDBPyConnection) -> dict[str, list[str]]:
    """table -> its columns that are NULL in every row, for non-empty tables (in table and column order)."""
    out = {}
    tables = [r[0] for r in con.execute("SELECT table_name FROM information_schema.tables "
                                        "WHERE table_schema = 'main' ORDER BY table_name").fetchall()]
    for table in tables:
        cols = [r[0] for r in con.execute("SELECT column_name FROM information_schema.columns WHERE table_schema = "
                                          "'main' AND table_name = ? ORDER BY ordinal_position", [table]).fetchall()]
        counts = con.execute(f'SELECT count(*), {", ".join(f"count({_ident(c)})" for c in cols)} '
                             f'FROM "{table}"').fetchone()
        if counts[0]:
            empty = [c for c, n in zip(cols, counts[1:]) if n == 0]
            if empty:
                out[table] = empty
    return out


def _ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def schema_summary(con: duckdb.DuckDBPyConnection, *, sample_rows: int = 3, null_columns: bool = False) -> str:
    """CREATE TABLE statements with `sample_rows` example rows per non-empty table (Rajkumar et al. 2022), the
    names of empty tables, and a per-vocabulary count of concepts instead of concept rows. With `null_columns`, a
    last line names the columns that are NULL in every row (the harder set's prompt, so that a model can tell when
    the data cannot answer a question)."""
    tables = [r[0] for r in con.execute("SELECT table_name FROM information_schema.tables "
                                        "WHERE table_schema = 'main' ORDER BY table_name").fetchall()]
    parts, empty = [], []
    for table in tables:
        if con.execute(f'SELECT count(*) FROM "{table}"').fetchall()[0][0] == 0:
            empty.append(table)
            continue
        cols = con.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = 'main' "
                           "AND table_name = ? ORDER BY ordinal_position", [table]).fetchall()
        create = f"CREATE TABLE {table} (\n" + ",\n".join(f"  {c} {t}" for c, t in cols) + "\n);"
        if sample_rows:
            rows = con.execute(f'SELECT * FROM "{table}" ORDER BY 1 LIMIT {int(sample_rows)}').fetchall()
            lines = [" | ".join(c for c, _ in cols)] + [" | ".join(_cell(v) for v in row) for row in rows]
            create += f"\n/*\n{len(rows)} example rows from {table}:\n" + "\n".join(lines) + "\n*/"
        parts.append(create)
    if "concept" in tables:
        rows = con.execute("SELECT domain_id, vocabulary_id, count(*) FROM concept GROUP BY ALL ORDER BY 1, 2").fetchall()
        parts.append("/*\nConcepts by domain and vocabulary:\ndomain_id | vocabulary_id | concepts\n"
                     + "\n".join(" | ".join(str(v) for v in r) for r in rows) + "\n*/")
    if empty:
        parts.append(f"-- Tables present but empty: {', '.join(empty)}")
    text = "\n\n".join(parts)
    if null_columns:
        nulls = all_null_columns(con)
        if nulls:
            text += "\n-- Columns that are NULL in every row: " + ", ".join(
                f"{t}.{c}" for t, cols in nulls.items() for c in cols)
    return text


def render_messages(question: Question, lang: str, schema: str, spec: PromptSpec = P0) -> list[dict]:
    text = question.text(lang)
    user = f"Database schema (DuckDB):\n\n{schema}\n\n{spec.conventions}\n\nQuestion: {text}"
    return [{"role": "system", "content": spec.system}, {"role": "user", "content": user}]


# ---- reference checks ----

@dataclass
class Check:
    id: str
    category: str
    difficulty: str
    rows: int
    columns: int
    problem: str | None

    @property
    def ok(self) -> bool:
        return self.problem is None


def _trivial(rows: list[tuple]) -> bool:
    return all(canonical_value(v) in (None, 0) for row in rows for v in row)


NAEWON = "내원"
NAEWON_PROBLEM = (
    "question_ko uses 내원, which can be read as outpatient visits only; the naewon decision (v0 draft, 2026-10-30: "
    "reword, 3 of 3) writes 방문 instead and, where the visit type could matter, says that every type counts "
    "(외래·입원·응급 등 방문 유형 무관) or names the type")


def wording_problem(item: Question) -> str | None:
    """Why a question's wording cannot stay, whatever its SQL: 내원 in the Korean (decision naewon)."""
    return NAEWON_PROBLEM if NAEWON in item.question_ko else None


def check_questions(con: duckdb.DuckDBPyConnection, items: list[Question]) -> list[Check]:
    """Execute every reference query and flag ones a frozen task must not contain, and wording that a decision on
    the draft ruled out (`wording_problem`)."""
    checks, seen_sql = [], {}
    for item in items:
        if not item.answerable:  # nothing to execute: the correct answer is to abstain
            checks.append(Check(item.id, item.category, item.difficulty, 0, 0, wording_problem(item)))
            continue
        result = execute_sql(con, item.sql, max_rows=MAX_REFERENCE_ROWS)
        problem = None
        if _CLOCK.search(item.sql):
            problem = "uses the current date or time (results would change)"
        elif result.error == "too_many_rows":
            problem = f"too many rows (over {MAX_REFERENCE_ROWS})"
        elif result.error:
            problem = f"{result.error}: {result.detail}"
        elif not result.rows:
            problem = "empty result"
        elif _trivial(result.rows):
            problem = "trivial result (only zeros or NULLs)"
        elif item.order_matters and "order by" not in item.sql.lower():
            problem = "order_matters is set but the reference has no ORDER BY"
        elif " ".join(item.sql.split()).lower() in seen_sql:
            problem = f"same reference SQL as {seen_sql[' '.join(item.sql.split()).lower()]}"
        seen_sql.setdefault(" ".join(item.sql.split()).lower(), item.id)
        checks.append(Check(item.id, item.category, item.difficulty, len(result.rows),
                            len(result.rows[0]) if result.rows else 0, problem or wording_problem(item)))
    return checks


# ---- scoring ----

def _rate(values: list[bool]) -> dict:
    return {"value": sum(values) / len(values) if values else 0.0, "n": len(values)}


def reliability_score(scores: list[int], penalty: float) -> float:
    """EHRSQL-2024 reliability score RS(c): the mean of the per-instance scores (+1 a correct answer or a correct
    abstention, 0 an abstention on an answerable question, -1 any other answer) with -1 counted as -c."""
    return sum(-penalty if s == -1 else s for s in scores) / len(scores) if scores else 0.0


def _reliability(rows: list[dict]) -> dict:
    scores = [r["rs"] for r in rows]
    out = {f"RS({c})": reliability_score(scores, c) for c in RS_PENALTIES}
    out.update({"RS(N)": reliability_score(scores, len(scores)), "N": len(scores)})
    out["by_language"] = {lang: {f"RS({c})": reliability_score([r["rs"] for r in rows if r["lang"] == lang], c)
                                 for c in RS_PENALTIES}
                          for lang in LANGS if any(r["lang"] == lang for r in rows)}
    return out


def score_predictions(con: duckdb.DuckDBPyConnection, items: list[Question], predictions: dict[str, str | None], *,
                      timeout_s: float = DEFAULT_TIMEOUT_S, max_rows: int = DEFAULT_MAX_ROWS,
                      abstention: bool = False) -> tuple[list[dict], dict]:
    """Score predicted SQL (keyed by instance id '<question id>-<lang>'; None = no SQL in the answer, ABSTAIN = the
    model abstained). With `abstention` (or any unanswerable item), execution accuracy is over answerable instances
    (an abstention there is wrong) and the summary adds the EHRSQL-2024 reliability score RS(0), RS(5), RS(N):
    +1 for a correct answer and for abstaining on an unanswerable question, 0 for abstaining on an answerable one,
    -c for any other answer (a wrong or failing query, no SQL, or a query for an unanswerable question)."""
    by_id = {item.id: item for item in items}
    keys = []
    for key in predictions:
        qid, _, lang = key.rpartition("-")
        if qid not in by_id or lang not in LANGS:
            raise ValueError(f"prediction {key!r} matches no question and language")
        keys.append((qid, lang))
    order = {item.id: n for n, item in enumerate(items)}
    keys.sort(key=lambda k: (order[k[0]], LANGS.index(k[1])))

    references: dict[str, ExecResult] = {}
    for qid in dict.fromkeys(k[0] for k in keys):
        if not by_id[qid].answerable:
            continue
        ref = execute_sql(con, by_id[qid].sql, timeout_s=max(timeout_s, 120), max_rows=MAX_REFERENCE_ROWS)
        if ref.error or not ref.rows:
            raise ValueError(f"reference SQL of {qid} does not give a usable result: {ref.error or 'empty'}")
        references[qid] = ref
    abstention = abstention or any(not by_id[qid].answerable for qid, _ in keys)

    rows = []
    for qid, lang in keys:
        item, sql = by_id[qid], predictions[instance_id(qid, lang)]
        ref = references.get(qid)
        abstained = sql == ABSTAIN
        if not item.answerable:
            correct, error, n_pred = abstained, None if abstained or sql is not None else "no_sql", None
            rs = 1 if abstained else -1
        elif abstained:
            correct, error, n_pred, rs = False, "abstained", None, 0
        elif sql is None:
            correct, error, n_pred, rs = False, "no_sql", None, -1
        else:
            pred = execute_sql(con, sql, timeout_s=timeout_s, max_rows=max_rows)
            error, n_pred = pred.error, len(pred.rows)
            correct = error is None and results_match(ref.rows, pred.rows, order_matters=item.order_matters)
            rs = 1 if correct else -1
        row = {"instance_id": instance_id(qid, lang), "question_id": qid, "lang": lang,
               "category": item.category, "difficulty": item.difficulty, "correct": bool(correct),
               "error": error, "sql": sql, "pred_rows": n_pred, "ref_rows": len(ref.rows) if ref else None}
        if abstention:
            row.update({"answerable": item.answerable, "abstained": abstained, "rs": rs})
        rows.append(row)

    scored = [r for r in rows if not abstention or r["answerable"]]  # execution accuracy: answerable instances

    def group(field: str, values, among) -> dict:
        return {v: _rate([r["correct"] for r in among if r[field] == v]) for v in values
                if any(r[field] == v for r in among)}

    summary = {
        "execution_accuracy": accuracy_ci([r["correct"] for r in scored]),
        "by_language": group("lang", LANGS, scored),
        "by_category": group("category", ALL_CATEGORIES, rows),
        "by_difficulty": group("difficulty", DIFFICULTIES, scored),
        "sql_rate": sum(r["sql"] not in (None, ABSTAIN) for r in rows) / len(rows) if rows else 0.0,
        "execution_error_rate": sum(r["error"] in EXEC_ERRORS for r in rows) / len(rows) if rows else 0.0,
        "errors": dict(Counter(r["error"] for r in rows if r["error"])),
        "prompt": PROMPT_ID,
        "scorer": SCORER_VERSION,
    }
    if abstention:
        answerable = [r for r in rows if r["answerable"]]
        unanswerable = [r for r in rows if not r["answerable"]]
        summary.update({
            "reliability": _reliability(rows),
            "abstention": {"answerable_abstained": _rate([r["abstained"] for r in answerable]),
                           "unanswerable_abstained": _rate([r["abstained"] for r in unanswerable])},
            "prompt": HARD_PROMPT.id,
            "scorer": RS_SCORER_VERSION,
        })
    return rows, summary


# ---- dry run with an LLM (no medbench database involved) ----

def _load_responses(path: Path) -> dict[str, dict]:
    latest = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                latest[record["instance_id"]] = record
    return latest


def _prompt_sha(model_id: str, messages: list[dict]) -> str:
    """Identifies what was asked: an answer cached for an older wording of the question is not reused."""
    return hashlib.sha256(json.dumps([model_id, messages], ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def _needs_call(record: dict | None, prompt_sha: str) -> bool:
    if record is None or record.get("prompt_sha") != prompt_sha:
        return True
    return record.get("error") is not None and record.get("transient", True) is not False


async def run_dry_run(items: list[Question], con: duckdb.DuckDBPyConnection, client, model_id: str, out_dir: Path, *,
                      langs=LANGS, concurrency: int = 8, max_tokens: int = 16384, schema: str | None = None,
                      timeout_s: float = DEFAULT_TIMEOUT_S, spec: PromptSpec = P0) -> dict:
    """Ask the model for SQL for every (question, language), caching answers in out_dir/responses.jsonl so an
    interrupted run resumes, then score. Writes items.jsonl and summary.json next to the cache. With a prompt that
    allows abstention, an ABSTAIN answer is read as such and the reliability score is reported. Each cached answer
    and the summary record the output limit (`max_tokens`), which the frozen task's runner checks before reusing it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "responses.jsonl"
    schema = schema if schema is not None else schema_summary(con, null_columns=spec.null_columns)
    responses = _load_responses(cache)
    messages = {instance_id(item.id, lang): render_messages(item, lang, schema, spec)
                for item in items for lang in langs}
    shas = {iid: _prompt_sha(model_id, m) for iid, m in messages.items()}
    todo = [(item, lang) for item in items for lang in langs
            if _needs_call(responses.get(instance_id(item.id, lang)), shas[instance_id(item.id, lang)])]
    semaphore = asyncio.Semaphore(concurrency)

    async def one(item: Question, lang: str) -> None:
        iid = instance_id(item.id, lang)
        async with semaphore:
            try:
                result = await client.chat(model_id, messages[iid], max_tokens=max_tokens)
            except Exception as exc:  # one broken call must not abort the run
                result = ChatResult(content="", reasoning=None, tokens_in=0, tokens_out=0,
                                    error=f"exception_{type(exc).__name__}", transient=True)
        record = {"instance_id": iid, "prompt_sha": shas[iid], "content": result.content, "error": result.error,
                  "transient": result.transient if result.error else None, "tokens_in": result.tokens_in,
                  "tokens_out": result.tokens_out, "reasoning_chars": len(result.reasoning or ""),
                  "max_tokens": max_tokens}
        responses[record["instance_id"]] = record
        with cache.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    await asyncio.gather(*(one(item, lang) for item, lang in todo))

    wanted = [instance_id(item.id, lang) for item in items for lang in langs]
    pending = [iid for iid in wanted if _needs_call(responses.get(iid), shas[iid])]
    tokens_in = sum(responses[i]["tokens_in"] for i in wanted if i in responses)
    tokens_out = sum(responses[i]["tokens_out"] for i in wanted if i in responses)
    if pending:
        summary = {"status": "incomplete", "pending": len(pending), "model": model_id,
                   "tokens_in": tokens_in, "tokens_out": tokens_out}
    else:
        extract = extract_answer if spec.abstention else extract_sql
        predictions = {iid: None if responses[iid]["error"] else extract(responses[iid]["content"]) for iid in wanted}
        rows, summary = score_predictions(con, items, predictions, timeout_s=timeout_s, abstention=spec.abstention)
        summary["prompt"] = spec.id
        for row in rows:
            if responses[row["instance_id"]]["error"]:
                row["error"] = f"llm_{responses[row['instance_id']]['error']}"
        with (out_dir / "items.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        summary = {"status": "done", "model": model_id, "langs": list(langs), "n_questions": len(items),
                   **summary, "tokens_in": tokens_in, "tokens_out": tokens_out, "max_tokens": max_tokens}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    return summary
