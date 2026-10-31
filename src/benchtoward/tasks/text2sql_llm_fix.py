"""Fixes to the Text-to-SQL draft made by the three frontier reviewers themselves (owner, 2026-10-30).

The owner asked that the items the three-model review left at needs_edit be resolved by the same three models,
including the edits, instead of waiting for the owner's own review. For each such item one model proposes a revised
item (Korean and English question, reference SQL, whether row order matters) as JSON, given the item, the schema,
the database notes, the reference result and every reviewer's reason and suggested fix. Proposers rotate
claude -> gpt -> gemini over at most three rounds. A proposed SQL must pass the reference rules (`reference_problem`);
one that fails goes back to the next proposer as feedback. A valid proposal is re-reviewed by all three with the
rubric and aggregation of `text2sql_llm_review`; unanimous approval writes it into the draft, editing only that
item's lines. An item still not approved after three rounds is marked rejected (dropped) with the reasons.

Paid calls go through `Budget`, a hard cap: a request starts only if its worst-case cost fits under the cap
together with the spend already recorded in the job's caches and the requests in flight. Answers are cached per
model, kind, item, round and prompt, so a rerun asks only what is missing. Public with the evaluation code since
2026-10-30 (owner).
"""
import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from benchtoward.tasks.synthetic_snuh import connect_readonly
from benchtoward.tasks.text2sql import (CONVENTIONS, MAX_REFERENCE_ROWS, P0, PromptSpec, Question, check_questions,
                                     execute_sql, load_questions, schema_summary)
from benchtoward.tasks.text2sql_llm_review import (ITEM_START, MAX_TOKENS, _preview, aggregate, json_objects,
                                                parse_batch_review, parse_review, reference_text,
                                                render_batch_review_messages, render_review_messages)
from benchtoward.tasks.text2sql_review import _yaml_str

MAX_ROUNDS = 3
CONCURRENCY = 4  # requests in flight per model
FIX_DATE = "2026-10-30"

SYSTEM = (
    "You fix a draft item of a Text-to-SQL benchmark over an OMOP CDM v5.4 database (DuckDB) of synthetic patients "
    "modelled on a Korean university hospital. Each item is a question in Korean and English plus a reference SQL "
    "query whose result is the answer key; the benchmark ranks AI models, so an ambiguous question or a wrong "
    "reference makes the ranking wrong. This item was sent back for an edit, by its three reviewers or by a decision "
    "on the draft; the reasons follow. Write a revised item that resolves every concern while keeping the item's "
    "intent, category and difficulty as far as possible:\n"
    "- the question has exactly one reasonable reading given the schema and the database notes;\n"
    "- the Korean and English versions ask exactly the same thing;\n"
    "- the reference SQL answers it exactly (filters, joins, distinct counting, NULLs, time boundaries);\n"
    "- the answer is deterministic: break ties with an explicit rule stated in the question, or return all tied rows;\n"
    "- questions quote concept names exactly as they appear in concept.concept_name and say which columns to return.\n"
    "Rules for the reference SQL: one read-only DuckDB SELECT (WITH allowed); no current date or time; 1 to "
    f"{MAX_REFERENCE_ROWS:,} result rows that are not all zero or NULL; when order_matters is true, a full, tie-free "
    "ORDER BY.\n"
    "Answer with one JSON object and nothing else, with these keys:\n"
    '  "question_ko": the Korean question;\n'
    '  "question_en": the English question;\n'
    '  "sql": the reference SQL;\n'
    '  "order_matters": true only if the question asks for a sorted list, else false;\n'
    '  "change": one or two sentences on what you changed and why.'
)

# The harder set: concepts in clinical words, the author's note on why the item is hard, unanswerable items.
HARD_SYSTEM = (
    "You fix a draft item of the harder set of a Text-to-SQL benchmark over an OMOP CDM v5.4 database (DuckDB) of "
    "synthetic patients modelled on a Korean university hospital. Each item is a question in Korean and English "
    "plus a reference SQL query whose result is the answer key, or, for an unanswerable item, no SQL: its correct "
    "answer is to abstain. The benchmark ranks AI models, so an ambiguous question or a wrong reference makes the "
    "ranking wrong. This item was sent back for an edit by its three reviewers, or by a decision or rule on the "
    "draft; the reasons follow. Write a revised item that resolves every concern while keeping the item's intent, "
    "category and difficulty as far as possible:\n"
    "- the question has exactly one reasonable reading given the schema and the database notes;\n"
    "- the Korean and English versions ask exactly the same thing; the Korean never uses 내원 (write 방문, and say "
    "that every visit type counts where it could matter, or name the type);\n"
    "- concepts are named in clinical words (Korean terms, synonyms, abbreviations) as the database notes define "
    "them, never as a concept name in quotation marks or parentheses (as a gloss or as an output label), and each "
    "term identifies exactly the concepts the reference uses;\n"
    "- the reference SQL answers it exactly (filters, joins, distinct counting, NULLs, time boundaries, the fixed "
    "today of the notes) and the answer is deterministic; the question says which columns to return;\n"
    "- the item stays hard for the reason in the author's note, which you may correct;\n"
    "- an unanswerable item stays unanswerable: no table or column of this database holds what it asks for, a "
    "careful model can tell this from the schema and the notes, and an honest answer is not 0 or no rows.\n"
    "Rules for the reference SQL of an answerable item: one read-only DuckDB SELECT (WITH allowed); no current date "
    f"or time; 1 to {MAX_REFERENCE_ROWS:,} result rows that are not all zero or NULL; when order_matters is true, a "
    "full, tie-free ORDER BY.\n"
    "Answer with one JSON object and nothing else, with these keys:\n"
    '  "question_ko": the Korean question;\n'
    '  "question_en": the English question;\n'
    '  "sql": the reference SQL, or null for an unanswerable item;\n'
    '  "order_matters": true only if the question asks for a sorted list, else false;\n'
    '  "why_hard": one or two sentences on what makes the item hard (for an unanswerable item: which data is '
    "missing);\n"
    '  "change": one or two sentences on what you changed and why.'
)


# ---- budget and cached requests ----

class Budget:
    """A hard cap on paid calls for one job. `spent_usd` starts with what the job's caches already record (earlier
    runs count); a request starts only if its worst-case cost fits under the cap with the requests in flight."""

    def __init__(self, cap_usd: float, spent_usd: float = 0.0):
        self.cap_usd, self.spent_usd, self.reserved_usd, self.stopped = cap_usd, spent_usd, 0.0, False

    @staticmethod
    def spent_in(root: Path) -> float:
        """Sum of cost_usd, plus the estimates charged for failed attempts, over every cached answer under `root`."""
        total = 0.0
        for path in sorted(Path(root).rglob("*.jsonl")) if Path(root).exists() else []:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    total += float(record.get("cost_usd") or 0) + float(record.get("failed_attempts_usd") or 0)
        return total

    def try_reserve(self, usd: float) -> bool:
        if self.spent_usd + self.reserved_usd + usd > self.cap_usd:
            self.stopped = True
            return False
        self.reserved_usd += usd
        return True

    def settle(self, reserved: float, actual: float) -> None:
        self.reserved_usd -= reserved
        self.spent_usd += actual


def worst_case_usd(messages: list[dict], max_tokens: int, price: tuple[float, float]) -> float:
    """An upper bound on one request's cost: every UTF-8 byte of the prompt as one token (no tokenizer uses fewer
    bytes per token) plus a template allowance, and the full output allowance."""
    tokens_in = len(json.dumps(messages, ensure_ascii=False).encode("utf-8")) + 200
    return (tokens_in * price[0] + max_tokens * price[1]) / 1_000_000


class Attempts:
    """The budget side of one request, retries included: every attempt reserves the worst case before it is sent
    (`start` for the first, `before_attempt` as the route client's hook for each retry), and `settle` turns the
    reservations into spend. Pass it to `LLMClient.chat(before_attempt=...)`."""

    def __init__(self, budget: Budget, worst: float):
        self.budget, self.worst, self.reserved = budget, worst, []

    def start(self) -> bool:
        if not self.budget.try_reserve(self.worst):
            return False
        self.reserved.append(self.worst)
        return True

    async def before_attempt(self, n: int) -> bool:  # the first attempt is reserved by start()
        return True if n == 0 else self.start()

    def finish(self, cost: float, failed_attempts: int) -> float:
        """Settle a request that returned: its cost plus each failed attempt that may have been billed, charged its
        worst case. Returns that charge for the failed attempts."""
        failed = self.worst * failed_attempts
        self.budget.settle(sum(self.reserved), cost + failed)
        return failed

    def charge_all(self) -> float:
        """A request that raised or was cancelled: every attempt sent may have been billed, so each is charged its
        worst case. Returns the charge."""
        charged = sum(self.reserved)
        self.budget.settle(charged, charged)
        return charged


def _sha(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class Answer:
    value: dict | None
    status: str  # ok (parsed), unparsed (answered, but no usable JSON), error (route failed), budget (not asked)
    raw: str = ""


def cache_breakpoint(messages: list[dict], split: str) -> list[dict]:
    """The messages with the first user message cut at `split` into two text parts, the first marked for prompt
    caching (a route that passes `cache_control` on to Anthropic caches everything up to the mark). Unchanged when
    `split` is absent."""
    out = [dict(m) for m in messages]
    for m in out:
        content = m.get("content")
        if m.get("role") == "user" and isinstance(content, str) and split in content:
            at = content.index(split)
            m["content"] = [{"type": "text", "text": content[:at], "cache_control": {"type": "ephemeral"}},
                            {"type": "text", "text": content[at:]}]
            break
    return out


class Asker:
    """Asks a model once per (kind, key) and prompt. Answers go to <cache_dir>/<model>.jsonl and are reused while the
    prompt is unchanged; unusable answers are asked again on the next run. Every request passes the budget, retries
    included (`Attempts`): each attempt reserves its worst-case cost first, and a failed attempt that may have been
    billed keeps that estimate as spend (recorded as failed_attempts_usd, so later runs count it). A request that
    raises or is cancelled is charged its reserved attempts the same way and recorded before the exception goes on,
    so it is asked again next run and its cost still counts. With `cache_split`, requests to
    Anthropic models mark the prompt up to that text for prompt caching (the cache key stays the logical prompt)."""

    def __init__(self, clients: dict, prices: dict[str, tuple[float, float]], cache_dir: Path, budget: Budget, *,
                 max_tokens: int = MAX_TOKENS, concurrency: int = CONCURRENCY, extra: dict | None = None,
                 cache_split: str | None = None):
        self.clients, self.prices, self.budget = clients, prices, budget
        self.cache_dir, self.max_tokens, self.extra = Path(cache_dir), max_tokens, extra
        self.cache_split = cache_split
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sems = {m: asyncio.Semaphore(concurrency) for m in clients}
        self.cached = {m: self._load(m) for m in clients}
        self.asked = 0

    def _path(self, model: str) -> Path:
        return self.cache_dir / f"{model.replace('/', '__')}.jsonl"

    def _write(self, model: str, record: dict) -> None:
        """Cache a request's record. No await: the event loop runs nothing else meanwhile, and a cancelled task can
        still write it."""
        self.cached[model][(record["kind"], record["key"])] = record
        with self._path(model).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load(self, model: str) -> dict[tuple[str, str], dict]:
        path, records = self._path(model), {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    records[(r["kind"], r["key"])] = r
        return records

    async def ask(self, model: str, kind: str, key: str, messages: list[dict], parse) -> Answer:
        sha = _sha(messages)
        hit = self.cached[model].get((kind, key))
        if hit and hit.get("prompt_sha") == sha and hit.get("answer") is not None:
            return Answer(hit["answer"], "ok", hit.get("raw") or "")
        if hit and hit.get("prompt_sha") == sha and hit.get("raw") and not hit.get("error"):
            value = parse(hit["raw"])  # an answer an earlier parser could not read: read it again, do not re-ask
            if value is not None:
                return Answer(value, "ok", hit["raw"])
        price = self.prices.get(model, (0.0, 0.0))
        attempts = Attempts(self.budget, worst_case_usd(messages, self.max_tokens, price))
        sent = cache_breakpoint(messages, self.cache_split) if self.cache_split and model.startswith("anthropic/") \
            else messages
        async with self.sems[model]:
            if not attempts.start():
                return Answer(None, "budget")
            kwargs = {"extra": self.extra} if self.extra else {}
            try:
                result = await self.clients[model].chat(model, sent, max_tokens=self.max_tokens,
                                                        before_attempt=attempts.before_attempt, **kwargs)
            except BaseException as exc:  # cancellation included: what was sent may have been billed
                charged = attempts.charge_all()
                self._write(model, {"kind": kind, "key": key, "prompt_sha": sha, "answer": None, "raw": "",
                                    "error": f"exception_{type(exc).__name__}", "tokens_in": 0, "tokens_out": 0,
                                    "cost_usd": 0.0, "cost_estimate_usd": 0.0, "cost_source": "estimate",
                                    "attempts": len(attempts.reserved), "failed_attempts": len(attempts.reserved),
                                    "failed_attempts_usd": round(charged, 6)})
                raise
        estimate = (result.tokens_in * price[0] + result.tokens_out * price[1]) / 1_000_000
        cost = result.cost_usd if result.cost_usd is not None else estimate
        failed_usd = attempts.finish(cost, result.failed_attempts)
        answer = None if result.error else parse(result.content)
        self.asked += 1
        self._write(model, {"kind": kind, "key": key, "prompt_sha": sha, "answer": answer, "raw": result.content,
                            "error": result.error, "tokens_in": result.tokens_in, "tokens_out": result.tokens_out,
                            "cached_tokens": result.cached_tokens, "reasoning_tokens": result.reasoning_tokens,
                            "cost_usd": round(cost, 6), "cost_estimate_usd": round(estimate, 6),
                            "cost_source": "route" if result.cost_usd is not None else "estimate",
                            "attempts": result.attempts, "failed_attempts": result.failed_attempts,
                            "failed_attempts_usd": round(failed_usd, 6)})
        if result.error:
            return Answer(None, "budget" if result.error == "budget" else "error", result.content)
        return Answer(answer, "ok" if answer is not None else "unparsed", result.content)


def parse_json_object(text: str | None, accept):
    """`accept(obj)` of the last JSON object in `text` (fenced or bare) for which it returns a value: a later object,
    such as a correction the model appends, supersedes an earlier one."""
    values = [v for v in (accept(body) for body in json_objects(text)) if v is not None]
    return values[-1] if values else None


# ---- proposals ----

def clean_sql(sql: str) -> str:
    """Reference SQL as the draft keeps it: LF line ends, no tabs, trailing spaces or final semicolon."""
    lines = sql.replace("\r\n", "\n").expandtabs(4).strip().rstrip(";").strip().split("\n")
    return "\n".join(line.rstrip() for line in lines)


def _accept_proposal(body: dict, answerable: bool = True) -> dict | None:
    ko, en, sql = (body.get(k) for k in ("question_ko", "question_en", "sql"))
    order = body.get("order_matters", False)
    if not all(isinstance(v, str) and v.strip() for v in (ko, en)) or not isinstance(order, bool):
        return None
    if answerable != (isinstance(sql, str) and bool(sql.strip())) or (not answerable and sql not in (None, "")):
        return None  # an answerable item needs SQL; an unanswerable one stays without
    proposal = {"question_ko": " ".join(ko.split()), "question_en": " ".join(en.split()),
                "sql": clean_sql(sql) if answerable else None, "order_matters": order,
                "change": " ".join(str(body.get("change") or "").split())}
    if isinstance(body.get("why_hard"), str) and body["why_hard"].strip():
        proposal["why_hard"] = " ".join(body["why_hard"].split())
    return proposal


def parse_proposal(text: str | None, answerable: bool = True) -> dict | None:
    """A revised item: question_ko, question_en, sql (no trailing semicolon; None for an unanswerable item),
    order_matters, change and, when given, why_hard; or None."""
    return parse_json_object(text, lambda body: _accept_proposal(body, answerable))


def _normal(sql: str | None) -> str | None:
    return " ".join(sql.split()).lower() if sql else None


def reference_problem(con, item: Question, others: list[Question]) -> str | None:
    """Why `item`'s SQL cannot be a reference (the rules `text2sql check` applies), or None if it can."""
    (check,) = check_questions(con, [item])
    if check.problem:
        return check.problem
    if not item.answerable:
        return None
    same = next((o.id for o in others if o.id != item.id and _normal(o.sql) == _normal(item.sql)), None)
    return f"same reference SQL as {same}" if same else None


_LITERAL = re.compile(r"'((?:[^']|'')+)'")
# A *_concept_id column (or concept.concept_id) compared with integers: `= N`, `IN (N, ...)`, `NOT IN (...)`.
_CONCEPT_IDS = re.compile(r"\b\w*concept_id\s*(?:=\s*(\d+)\b|(?:NOT\s+)?IN\s*\(([\d\s,]+)\))", re.I)
# Text a question sets apart: in quotation marks (an apostrophe inside a word does not open a quote) or brackets.
_QUOTED = [re.compile(p) for p in (r"(?<![A-Za-z0-9])'([^'\n]+)'(?![A-Za-z0-9])", r'"([^"\n]+)"', r"“([^”]+)”",
                                   r"‘([^’]+)’", r"「([^」]+)」", r"『([^』]+)』")]
_WORD = re.compile(r"[A-Za-z0-9_]")


def concept_ids(sql: str) -> set[int]:
    """The integers a SQL compares with a concept id column (= N, IN (...), NOT IN (...))."""
    found = set()
    for one, many in _CONCEPT_IDS.findall(sql or ""):
        found |= {int(n) for n in re.findall(r"\d+", one or many)}
    return found


def _set_apart(text: str) -> list[str]:
    """The spans of `text` inside quotation marks or parentheses (nested parentheses give each level)."""
    spans, opened = [], []
    for at, ch in enumerate(text):
        if ch in "(（":
            opened.append(at)
        elif ch in ")）" and opened:
            spans.append(text[opened.pop() + 1:at])
    return spans + [m for pattern in _QUOTED for m in pattern.findall(text)]


def _names_in(name: str, spans: list[str]) -> bool:
    """`name` (case-insensitive) in one of `spans` as a whole term: not part of a longer word or identifier."""
    low = name.lower()
    for span in spans:
        span = span.lower()
        for m in re.finditer(re.escape(low), span):
            before, after = span[m.start() - 1:m.start()], span[m.end():m.end() + 1]
            if not _WORD.fullmatch(before or " ") and not _WORD.fullmatch(after or " "):
                return True
    return False


def quoted_concepts(con, item: Question) -> list[str]:
    """Names of the concepts a reference uses (string literals that are concept names, and the concept ids it
    compares with a *_concept_id column) that a question, in either language, sets apart in quotation marks or
    parentheses, as a gloss or a label (case-insensitive). The same words written plainly are allowed."""
    if not item.sql:
        return []
    literals = sorted({m.replace("''", "'") for m in _LITERAL.findall(item.sql)})
    ids = sorted(concept_ids(item.sql))
    known = set()
    if literals:
        known |= {r[0] for r in con.execute("SELECT DISTINCT concept_name FROM concept WHERE concept_name IN "
                                            "(SELECT unnest(?))", [literals]).fetchall()}
    if ids:
        known |= {r[0] for r in con.execute("SELECT DISTINCT concept_name FROM concept WHERE concept_id IN "
                                            "(SELECT unnest(?))", [ids]).fetchall() if r[0]}
    spans = _set_apart(item.question_ko) + _set_apart(item.question_en)
    return sorted(n for n in known if _names_in(n, spans))


QUOTED_RULE = ("no question may quote a concept name its reference uses, in quotation marks or parentheses (a name "
               "the reference filters by concept_id counts too)")


def quoted_problem(con, item: Question) -> str | None:
    """The harder set names concepts in clinical words: a question may not quote a concept name its reference uses."""
    quoted = quoted_concepts(con, item)
    return (f"the question quotes concept names ({', '.join(quoted)}) in quotation marks or parentheses; name them in "
            "clinical words only") if quoted else None


def _short(model: str) -> str:
    return model.split("/")[-1]


def _reviews_text(reviews: dict[str, dict | None] | str | None) -> str:
    if isinstance(reviews, str):
        return reviews
    lines = []
    for model, r in (reviews or {}).items():
        if r is None:
            lines.append(f"- {_short(model)}: no usable review")
            continue
        fix = f"\n  Suggested fix: {r['fix']}" if r.get("fix") else ""
        lines.append(f"- {_short(model)}: {r.get('verdict')}. {r.get('reason', '')}{fix}")
    return "\n".join(lines) or "(no reviews recorded)"


def _round_text(entry: dict) -> str:
    head = f"Round {entry['round']}, proposed by {_short(entry['proposer'])}:"
    if entry.get("proposal") is None:
        return f"{head} no usable JSON proposal ({entry['problem']})."
    p = entry["proposal"]
    sql = f"SQL:\n```sql\n{p['sql']}\n```" if p.get("sql") else "SQL: none (unanswerable)"
    why = f"\nWhy hard: {p['why_hard']}" if p.get("why_hard") else ""
    body = (f"{head}\nKorean: {p['question_ko']}\nEnglish: {p['question_en']}\norder_matters: {p['order_matters']}"
            f"{why}\n{sql}")
    if entry.get("problem"):
        return f"{body}\nRejected before review: the SQL breaks the reference rules ({entry['problem']})."
    return (f"{body}\nResult: {entry['result']}\nRe-review ({entry['decision']}):\n"
            f"{_reviews_text(entry['reviews'])}")


def render_fix_messages(item: Question, con, schema: str, reviews, history: list[dict], round_no: int,
                        max_rounds: int = MAX_ROUNDS, spec: PromptSpec = P0) -> list[dict]:
    note = f"Author's note: {item.note}\n" if item.note else ""
    earlier = ("\n\nEarlier rounds of this fix (none was approved by all three reviewers):\n\n"
               + "\n\n".join(_round_text(e) for e in history)) if history else ""
    if spec is not P0:
        why = f"Author's note on why it is hard: {item.why_hard}\n" if item.why_hard else ""
        user = (f"Database schema (DuckDB):\n\n{schema}\n\n{spec.conventions}{ITEM_START} {item.id}\n"
                f"Category: {item.category}; intended difficulty: {item.difficulty}; patient scope: {item.scope}; "
                f"answerable: {item.answerable}; result order matters: {item.order_matters}\n{why}{note}"
                f"Korean: {item.question_ko}\nEnglish: {item.question_en}\n\n{reference_text(item, con)}\n\n"
                f"Why this item needs an edit:\n{_reviews_text(reviews)}{earlier}\n\n"
                f"This is round {round_no} of {max_rounds}. Revise this item so that all three reviewers can "
                "approve it" + ("." if item.answerable else "; it stays unanswerable (sql null), so the correct "
                                "answer is still to abstain."))
        return [{"role": "system", "content": HARD_SYSTEM}, {"role": "user", "content": user}]
    result = execute_sql(con, item.sql, max_rows=MAX_REFERENCE_ROWS)  # v0's prompt, byte for byte
    user = (f"Database schema (DuckDB):\n\n{schema}\n\n{CONVENTIONS}\n\n"
            f"Question id: {item.id}\nCategory: {item.category}; intended difficulty: {item.difficulty}; "
            f"patient scope: {item.scope}; result order matters: {item.order_matters}\n"
            f"Korean: {item.question_ko}\nEnglish: {item.question_en}\n{note}\n"
            f"Reference SQL:\n```sql\n{item.sql}\n```\n\n"
            f"Reference result: {_preview(result, order_matters=item.order_matters)}\n\n"
            f"Why this item needs an edit:\n{_reviews_text(reviews)}{earlier}\n\n"
            f"This is round {round_no} of {max_rounds}. Revise this item so that all three reviewers can approve it.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


# ---- the fix loop ----

@dataclass
class FixOutcome:
    id: str
    status: str  # approved, rejected, incomplete
    round: int | None = None
    proposer: str | None = None
    revised: Question | None = None
    change: str = ""
    rounds: list[dict] = field(default_factory=list)


BATCH_MAX_TOKENS = 12000  # a review request about several items


async def review_items(asker: Asker, batch_asker: Asker, models: list[str], items: list[Question], con, schema: str,
                       spec: PromptSpec, round_key: str, batch_size: int = 1) -> dict[str, dict | None]:
    """Every model reviews every item with the rubric of `spec`, `batch_size` items per request (each judged on its
    own; an item missing from a batch answer is asked again alone; a group of one is asked alone). Cache keys:
    '<id>/<round_key>' alone, '<round_key>/<ids>' in a batch. item id -> {model: review or None}, or None when a
    request failed or the budget stopped it."""
    results: dict[str, dict[str, dict | None]] = {i.id: {} for i in items}
    failed: set[str] = set()

    async def single(model: str, item: Question) -> None:
        answer = await asker.ask(model, "review", f"{item.id}/{round_key}",
                                 render_review_messages(item, con, schema, spec), parse_review)
        if answer.status in ("error", "budget"):
            failed.add(item.id)
        results[item.id][model] = answer.value

    async def group(model: str, chunk: list[Question]) -> None:
        if len(chunk) == 1:
            return await single(model, chunk[0])
        ids = [i.id for i in chunk]
        answer = await batch_asker.ask(model, "review-batch", f"{round_key}/" + ",".join(ids),
                                       render_batch_review_messages(chunk, con, schema, spec),
                                       lambda text: parse_batch_review(text, ids))
        if answer.status in ("error", "budget"):
            failed.update(ids)
            return
        for item in chunk:
            if item.id in (answer.value or {}):
                results[item.id][model] = answer.value[item.id]
            else:
                await single(model, item)

    chunks = [items[k:k + max(1, batch_size)] for k in range(0, len(items), max(1, batch_size))]
    await asyncio.gather(*(group(m, c) for c in chunks for m in models))
    return {i.id: None if i.id in failed else {m: results[i.id].get(m) for m in models} for i in items}


def review_askers(clients: dict, prices: dict, out: Path, budget: Budget, extra: dict | None) -> tuple[Asker, Asker]:
    """An Asker for single-item reviews and one for batches, sharing the per-model concurrency limits."""
    asker = Asker(clients, prices, out, budget, extra=extra, cache_split=ITEM_START)
    batch_asker = Asker(clients, prices, out, budget, max_tokens=BATCH_MAX_TOKENS, extra=extra,
                        cache_split=ITEM_START)
    batch_asker.sems = asker.sems
    return asker, batch_asker


async def run_fix(items: list[Question], database: Path, clients: dict, prices: dict[str, tuple[float, float]],
                  out: Path, budget: Budget, *, proposers: list[str], original_reviews: dict[str, object],
                  ids: list[str] | None = None, max_rounds: int = MAX_ROUNDS, extra: dict | None = None,
                  spec: PromptSpec = P0, review_batch: int = 1) -> list[FixOutcome]:
    """Fix every needs_edit item (or the needs_edit items among `ids`), round by round: every open item gets a
    proposal from the round's proposer, then all three re-review the valid proposals (`review_batch` items per
    request). `original_reviews`: item id -> {model: review} from the three-model review, or the review note text.
    `spec`: the draft's prompt (the harder set's adds the author's note on why an item is hard and keeps
    unanswerable items without SQL). Returns one outcome per item, in draft order."""
    con = connect_readonly(database)
    schema = schema_summary(con, null_columns=spec.null_columns)
    asker, batch_asker = review_askers(clients, prices, out, budget, extra)
    reviewers = list(clients)
    todo = [i for i in items if i.review == "needs_edit" and (not ids or i.id in ids)]
    histories: dict[str, list[dict]] = {i.id: [] for i in todo}
    outcomes: dict[str, FixOutcome] = {}

    async def propose(item: Question, round_no: int, proposer: str) -> Question | None:
        history = histories[item.id]
        messages = render_fix_messages(item, con, schema, original_reviews.get(item.id), history, round_no,
                                       max_rounds, spec)
        answer = await asker.ask(proposer, "propose", f"{item.id}/r{round_no}", messages,
                                 lambda text: parse_proposal(text, item.answerable))
        if answer.status in ("error", "budget"):
            outcomes[item.id] = FixOutcome(item.id, "incomplete", rounds=history)
            return None
        entry = {"round": round_no, "proposer": proposer, "proposal": answer.value}
        history.append(entry)
        if answer.value is None:
            entry["problem"] = ("the answer had no JSON object with question_ko, question_en, sql and order_matters"
                                + ("" if item.answerable else " (sql null: the item stays unanswerable)"))
            return None
        p = answer.value
        revised = replace(item, question_ko=p["question_ko"], question_en=p["question_en"], sql=p["sql"],
                          order_matters=p["order_matters"])
        if spec is not P0 and p.get("why_hard"):
            revised = replace(revised, why_hard=p["why_hard"])
        problem = reference_problem(con, revised, items) or (quoted_problem(con, revised) if spec is not P0 else None)
        if problem:
            entry["problem"] = problem
            return None
        entry["result"] = _preview(execute_sql(con, revised.sql, max_rows=MAX_REFERENCE_ROWS),
                                   order_matters=revised.order_matters) if revised.answerable else \
            "none (unanswerable: the correct answer is to abstain)"
        return revised

    for round_no in range(1, max_rounds + 1):
        proposer = proposers[(round_no - 1) % len(proposers)]
        open_items = [i for i in todo if i.id not in outcomes]
        revised = await asyncio.gather(*(propose(i, round_no, proposer) for i in open_items))
        valid = [(item, r) for item, r in zip(open_items, revised) if r is not None]
        reviews = await review_items(asker, batch_asker, reviewers, [r for _, r in valid], con, schema, spec,
                                     f"r{round_no}", review_batch)
        for item, r in valid:
            entry = histories[item.id][-1]
            if reviews[item.id] is None:
                outcomes[item.id] = FixOutcome(item.id, "incomplete", rounds=histories[item.id])
                continue
            entry["reviews"] = reviews[item.id]
            entry["decision"], entry["note"] = aggregate(entry["reviews"])
            if entry["decision"] == "approved":
                outcomes[item.id] = FixOutcome(item.id, "approved", round_no, proposer, r,
                                               entry["proposal"]["change"], histories[item.id])
    for item in todo:
        outcomes.setdefault(item.id, FixOutcome(item.id, "rejected", rounds=histories[item.id]))
    return [outcomes[i.id] for i in todo]


def fix_note(outcome: FixOutcome) -> str:
    """The review_note an outcome writes into the draft."""
    if outcome.status == "approved":
        change = f" Change: {outcome.change}" if outcome.change else ""
        return (f"Fixed by the three-model review ({FIX_DATE}): revised in round {outcome.round} by "
                f"{_short(outcome.proposer)} and approved by all three reviewers.{change}")
    parts = []
    for e in outcome.rounds:
        who = f"round {e['round']} ({_short(e['proposer'])})"
        if e.get("proposal") is None or e.get("problem"):
            parts.append(f"{who}: {e.get('problem')}")
        else:
            parts.append(f"{who}: {e['decision']}. {e['note']}")
    return (f"Not approved by all three reviewers after {len(outcome.rounds)} rounds of the three-model fix "
            f"({FIX_DATE}); dropped. " + " | ".join(parts))


# ---- writing into the draft ----

_FIELD = re.compile(r"^    ([A-Za-z_]+):")


def _scalar(text: str) -> str:
    """A plain YAML scalar when it reads back unchanged, else a double-quoted one."""
    import yaml
    try:
        if yaml.safe_load(f"k: {text}") == {"k": text} and not text.startswith(("'", '"')):
            return text
    except yaml.YAMLError:
        pass
    return _yaml_str(text)


def _fields(lines: list[str]) -> tuple[list[list[str]], list[str]]:
    """An item's lines after `- id:` as [key line + continuation lines] groups, and the trailing blank and comment
    lines that belong to whatever follows the item."""
    last = max((i for i, line in enumerate(lines) if line.startswith("    ") and line.strip()), default=-1)
    groups: list[list[str]] = []
    for line in lines[:last + 1]:
        if _FIELD.match(line) or not groups:
            groups.append([line])
        else:
            groups[-1].append(line)
    return groups, lines[last + 1:]


def _rewrite_item(block: str, values: dict[str, list[str] | None]) -> str:
    """Replace (or add, or with None remove) the item's fields named in `values`; the rest stays byte for byte."""
    lines = block.split("\n")
    groups, tail = _fields(lines[1:])
    keys = [(_FIELD.match(g[0]) or [None, None])[1] for g in groups]
    out = [lines[0]]
    done = set()
    for key, group in zip(keys, groups):
        if key in values:
            done.add(key)
            if values[key] is not None:
                out += values[key]
            continue
        out += group
    for key, rendered in values.items():
        if key in done or rendered is None:
            continue
        anchor = {"order_matters": "scope", "review_note": "review"}.get(key, "review")
        at = next((i for i, line in enumerate(out) if line.startswith(f"    {anchor}:")), len(out) - 1)
        out[at + 1:at + 1] = rendered
    return "\n".join(out + tail)


def _item_span(text: str, qid: str) -> tuple[int, int]:
    start = re.search(rf"^  - id: {re.escape(qid)}[ \t]*$", text, re.M)
    if start is None:
        raise ValueError(f"no item {qid} in the draft")
    nxt = re.search(r"^  - id: ", text[start.end():], re.M)
    return start.start(), start.end() + nxt.start() if nxt else len(text)


def render_fields(outcome: FixOutcome) -> dict[str, list[str] | None]:
    values: dict[str, list[str] | None] = {
        "review": [f"    review: {outcome.status}"],
        "review_note": [f"    review_note: {_yaml_str(fix_note(outcome))}"],
    }
    if outcome.status == "approved":
        r = outcome.revised
        sql_lines = ["    sql: |", *(f"      {line}" if line.strip() else "" for line in r.sql.split("\n"))] \
            if r.answerable else ["    sql: null"]
        values.update({"question_ko": [f"    question_ko: {_scalar(r.question_ko)}"],
                       "question_en": [f"    question_en: {_scalar(r.question_en)}"],
                       "sql": sql_lines,
                       "order_matters": ["    order_matters: true"] if r.order_matters else None})
        if r.why_hard:
            values["why_hard"] = [f"    why_hard: {_scalar(r.why_hard)}"]
    return values


def apply_fixes(draft: Path, outcomes: list[FixOutcome]) -> dict[str, int]:
    """Write approved and rejected outcomes into the draft (incomplete ones change nothing), editing only those
    items' lines. Checks that the file still parses and that every approved item reads back as revised."""
    text = Path(draft).read_text(encoding="utf-8")
    counts: dict[str, int] = {}
    done = [o for o in outcomes if o.status in ("approved", "rejected")]
    for o in done:
        start, end = _item_span(text, o.id)
        text = text[:start] + _rewrite_item(text[start:end], render_fields(o)) + text[end:]
        counts[o.status] = counts.get(o.status, 0) + 1
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "draft.yaml"
        probe.write_text(text, encoding="utf-8")
        _, items = load_questions(probe)
    by_id = {i.id: i for i in items}
    for o in done:
        if o.status == "approved" and by_id[o.id] != replace(o.revised, review="approved"):
            raise ValueError(f"{o.id}: the revised item does not read back as proposed")
    Path(draft).write_text(text, encoding="utf-8")
    return counts


def write_report(path: Path, outcomes: list[FixOutcome]) -> None:
    """Every round of every item (proposal, rule problem, re-review) as JSON, for the record. Items of earlier runs
    stay; an item fixed again keeps its earlier approved or rejected fixes under "earlier" (an incomplete row of a
    run that is resumed is replaced)."""
    path = Path(path)
    rows = {r["id"]: r for r in json.loads(path.read_text(encoding="utf-8"))} if path.exists() else {}
    for o in outcomes:
        before = rows.get(o.id)
        earlier = (before or {}).get("earlier", [])
        if before and before["status"] in ("approved", "rejected"):
            earlier = earlier + [{k: v for k, v in before.items() if k != "earlier"}]
        rows[o.id] = {"id": o.id, "status": o.status, "round": o.round, "proposer": o.proposer, "change": o.change,
                      "rounds": o.rounds, **({"earlier": earlier} if earlier else {})}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(rows.values()), indent=1, ensure_ascii=False, default=str), encoding="utf-8")
