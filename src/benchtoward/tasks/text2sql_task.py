"""synthetic-snuh-text2sql@0: Synthetic-SNUH Text-to-SQL as a registered task (kind `text2sql`).

One task with two tiers (decision easy_tier_publication, 2026-10-30). The hard tier
(registry/drafts/synthetic-snuh-text2sql-hard-v0.yaml: 101 questions, 18 of them unanswerable) gives the headline, the
EHRSQL-2024 reliability score RS(0) (decision headline_metric). The easy tier (registry/drafts/synthetic-snuh-text2sql-
v0.yaml: 154 questions) is reported next to it as execution accuracy, marked saturated, and never ranked on its own.
The spec pins each tier's items, prompt and schema by sha256, so an edited draft never changes the frozen task: an
item change is a new task version.

A run asks the model once per question and language with the tier's prompt (`text2sql.render_messages`) at the
protocol's output limit (16,384 tokens, thinking included; decision output_format), or reuses a dry run's answer to
the same prompt from the same model, provided that dry run was asked under the same limit (`load_cache`). An answer
whose thinking used the whole limit and left no SQL is a wrong answer and, in the SOP format check, a model failure
(`cut_off`); only text with neither SQL nor ABSTAIN is a format failure.

Scoring: `text2sql.score_predictions` for each tier (DuckDB execution and medbench's result comparison); RS(c) through
the EHRSQL-2024 authors' `scoring_utils.py` (third_party/ehrsql-2024, unchanged); 95% intervals from a bootstrap over
questions, the Korean and English versions of a question resampled together. The priors are input-blind rules:
abstain on every item (`medbench/prior`, EHRSQL-2024's simplest baseline) and `SELECT 0` for every item, which came
in with the frozen spec after the dry runs were scored and the vote was taken (it was not voted on).
"""
import asyncio
import functools
import hashlib
import importlib.util
import json
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from benchtoward.tasks.text2sql import (ABSTAIN, LANGS, RS_SCORER_VERSION, SCORER_VERSION, PromptSpec, Question,
                                     _load_responses, _prompt_sha, extract_answer, extract_sql, instance_id,
                                     load_questions, prompt_spec, render_messages, schema_summary, score_predictions)

SLUG = "synthetic-snuh-text2sql"
TASK_REF = f"{SLUG}@0"
HARNESS_VERSION = "text2sql-p0"
TIERS = ("hard", "easy")  # the hard tier gives the headline; instances run in this order
HEADLINE_TIER = "hard"
HEADLINE = {"hard": "rs0", "easy": "easy_ex"}  # each tier's metric in the SOP check
ALSO_CHECKED = {"hard": ("hard_ex",), "easy": ()}
N_BOOT = 2000
SEED = 0
CI_METHOD = ("95% percentile interval of a bootstrap over questions (the Korean and English versions of a question "
             "resampled together), 2,000 resamples, seed 0")
PRIOR = "medbench/prior"
CONSTANT = "medbench/text2sql-constant-query"
CONSTANT_QUERY = "SELECT 0"
PRIOR_RULES = {
    "abstain": (PRIOR, "abstain on every item of both tiers (the easy tier has no abstention: no answer there)"),
    "constant": (CONSTANT, f"{CONSTANT_QUERY} for every item of both tiers"),
}
OUTCOMES = ("sql", "abstain", "cut_off", "format_failure", "api_error")
EHRSQL_DIR = Path(__file__).resolve().parents[3] / "third_party" / "ehrsql-2024"
EHRSQL_COMMIT = "f9e1aa02160d39e3f8df52bf5c69c5cf2e472499"
# SOP Checks (owner decisions 2026-10-29)
H_MIN, H_WARN, FORMAT_MAX, API_MAX = 0.05, 0.10, 0.10, 0.10
ITEM_FIELDS = ("id", "question_ko", "question_en", "sql", "category", "difficulty", "scope", "order_matters",
               "answerable")


@dataclass(frozen=True)
class Tier:
    name: str
    items: list[Question]
    prompt: PromptSpec


@dataclass(frozen=True)
class Instance:
    tier: str
    question: Question
    lang: str
    prompt: PromptSpec

    @property
    def local_id(self) -> str:
        """The id in the tier's own files (dry runs): '<question id>-<lang>'."""
        return instance_id(self.question.id, self.lang)

    @property
    def instance_id(self) -> str:
        return f"{self.tier}/{self.local_id}"


# ---- the frozen items and prompts ----

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_items(path: Path) -> tuple[dict, list[Question]]:
    """A question file's metadata and its kept items (rejected items are not part of a tier)."""
    meta, items = load_questions(Path(path))
    return meta, [q for q in items if q.review != "rejected"]


def items_sha256(items: list[Question]) -> str:
    """What fixes a tier's instances and their scoring: every item's id, wording, reference SQL and labels (not its
    review notes)."""
    payload = [{f: getattr(q, f) for f in ITEM_FIELDS} for q in items]
    return _sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def prompt_sha256(spec: PromptSpec) -> str:
    return _sha256(json.dumps([spec.id, spec.system, spec.conventions, spec.abstention, spec.null_columns],
                              ensure_ascii=False))


def schema_sha256(schema: str) -> str:
    return _sha256(schema)


def load_tiers(spec: dict, registry_root: Path) -> list[Tier]:
    """Both tiers as the spec pins them (files relative to the registry folder). Raises ValueError when a file's
    items or the code's prompt differ from the pinned hashes."""
    pinned = spec.get("tiers") or {}
    if set(pinned) != set(TIERS):
        raise ValueError(f"a text2sql spec has the tiers {', '.join(TIERS)}, not {', '.join(pinned) or 'none'}")
    tiers = []
    for name in TIERS:
        t = pinned[name]
        meta, items = load_items(Path(registry_root) / t["items"]["file"])
        prompt = prompt_spec(meta)
        if prompt.id != t["prompt"]["id"] or prompt_sha256(prompt) != t["prompt"]["sha256"]:
            raise ValueError(f"the prompt of tier {name} ({prompt.id}) differs from the one the spec pins "
                             f"({t['prompt']['id']}): a prompt change is a new task version")
        if items_sha256(items) != t["items"]["sha256"]:
            raise ValueError(f"the items of tier {name} ({t['items']['file']}) changed since the freeze: an item "
                             "change is a new task version")
        tiers.append(Tier(name, items, prompt))
    return tiers


def check_schemas(spec: dict, tiers: list[Tier], con) -> dict[str, str]:
    """Each tier's schema text from the database, checked against the spec (it is part of every prompt)."""
    schemas = {}
    for tier in tiers:
        schema = schema_summary(con, null_columns=tier.prompt.null_columns)
        if schema_sha256(schema) != spec["tiers"][tier.name]["prompt"]["schema_sha256"]:
            raise ValueError(f"the schema of tier {tier.name} built from this database differs from the one the spec "
                             "pins: check the Synthetic-SNUH 4.0.0 build (`benchtoward fetch synthetic-snuh`)")
        schemas[tier.name] = schema
    return schemas


def instances(tiers: list[Tier]) -> list[Instance]:
    return [Instance(t.name, q, lang, t.prompt) for t in tiers for q in t.items for lang in LANGS]


# ---- answers ----

def read_answer(content: str | None, error: str | None, tokens_out: int, *, max_tokens: int,
                abstention: bool) -> tuple[str | None, str]:
    """(prediction, outcome) of one answer. Outcomes: sql, abstain, cut_off (no SQL or ABSTAIN and the output used
    the whole limit: a model failure, decision output_format), format_failure (any other answer without SQL or
    ABSTAIN), api_error (a final error of the route)."""
    if error:
        return None, "api_error"
    prediction = (extract_answer if abstention else extract_sql)(content)
    if prediction is None:
        return None, "cut_off" if (tokens_out or 0) >= max_tokens else "format_failure"
    return prediction, "abstain" if abstention and prediction == ABSTAIN else "sql"


def prior_answers(tiers: list[Tier], rule: str) -> dict[str, dict]:
    """The input-blind priors: `abstain` (every item; on the easy tier, which has no abstention, that is no answer)
    and `constant` (SELECT 0 for every item, which no reference returns: every reference has a row that is not all
    zero or NULL). Neither looks at the question or at any model's answers."""
    if rule not in PRIOR_RULES:
        raise ValueError(f"unknown prior rule {rule!r} (fixed rules: {', '.join(PRIOR_RULES)})")
    out = {}
    for inst in instances(tiers):
        if rule == "abstain":
            out[inst.instance_id] = {"prediction": ABSTAIN if inst.prompt.abstention else None, "outcome": "abstain"}
        else:
            out[inst.instance_id] = {"prediction": CONSTANT_QUERY, "outcome": "sql"}
    return out


# ---- the EHRSQL-2024 reliability score (the authors' code) and intervals ----

@functools.cache
def ehrsql_scorer():
    """The EHRSQL-2024 shared task's scoring_program/scoring_utils.py at the pinned commit, vendored unchanged."""
    spec = importlib.util.spec_from_file_location("ehrsql_2024_scoring_utils", EHRSQL_DIR / "scoring_utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ehrsql_scores(outcomes: list[tuple[bool, str]]) -> list[int]:
    """Per-instance reliability scores from the authors' `reliability_score`. Each outcome is (answerable, kind) with
    kind right, wrong or abstain; it reaches their function as one token per side ('null' marks an unanswerable
    question and an abstention, as in the shared task's files)."""
    real, pred = {}, {}
    for n, (answerable, kind) in enumerate(outcomes):
        if kind not in ("right", "wrong", "abstain") or (kind == "right" and not answerable):
            raise ValueError(f"outcome {kind!r} for an {'answerable' if answerable else 'unanswerable'} question")
        real[str(n)] = "ANSWER" if answerable else "null"
        pred[str(n)] = "null" if kind == "abstain" else ("ANSWER" if kind == "right" else "WRONG")
    return [int(s) for s in ehrsql_scorer().reliability_score(real, pred)]


def rs(scores: list[int], penalty: float) -> float:
    """RS(c) with the authors' `penalize`: the mean score with every -1 counted as -c."""
    return float(ehrsql_scorer().penalize(scores, penalty=penalty)) if scores else 0.0


def cluster_ci(values, groups, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Mean of `values` with a 95% interval from a bootstrap over `groups` (questions): every resample draws whole
    questions, so both languages of a question stay together."""
    v = np.asarray(values, dtype=float)
    if not len(v):
        return {"value": None, "ci_low": None, "ci_high": None, "n": 0, "n_groups": 0, "ci_method": CI_METHOD}
    unique, inverse = np.unique(np.asarray(groups), return_inverse=True)
    sums = np.bincount(inverse, weights=v, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique)).astype(float)
    picked = np.random.default_rng(seed).integers(0, len(unique), (n_boot, len(unique)))
    boots = sums[picked].sum(axis=1) / counts[picked].sum(axis=1)
    low, high = np.percentile(boots, [2.5, 97.5])
    return {"value": float(v.mean()), "ci_low": float(low), "ci_high": float(high), "n": int(len(v)),
            "n_groups": int(len(unique)), "ci_method": CI_METHOD}


def question_of(instance_id_: str) -> str:
    """'hard/syn-hard-001-ko' -> 'hard/syn-hard-001': the cluster of an instance."""
    return instance_id_.rsplit("-", 1)[0]


def paired_difference(a: dict[str, float], b: dict[str, float], n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Mean of a - b over the instances both have, with the question bootstrap's 95% interval."""
    ids = sorted(set(a) & set(b))
    return cluster_ci([a[i] - b[i] for i in ids], [question_of(i) for i in ids], n_boot=n_boot, seed=seed)


# ---- scoring ----

def _rates(counts: dict[str, dict[str, int]], outcome: str) -> dict:
    total = {t: sum(c.values()) for t, c in counts.items()}
    out = {t: counts[t][outcome] / total[t] if total[t] else 0.0 for t in counts}
    n = sum(total.values())
    out["all"] = sum(c[outcome] for c in counts.values()) / n if n else 0.0
    return out


def score_answers(con, tiers: list[Tier], answers: dict[str, dict]) -> tuple[list[dict], dict]:
    """Score the answers given (instance id -> {prediction, outcome}) tier by tier. Returns one row per instance and
    the run's metrics: `rs0` (the headline, hard tier), `hard_ex`, `hard_rs5`, `hard_rsN`, `easy_ex`, each with its
    question-bootstrap interval, plus abstention, outcome counts and rates (cut_off, format failures, API errors)."""
    rows, metrics, counts, summaries = [], {}, {}, {}
    for tier in tiers:
        insts = [i for i in instances([tier]) if i.instance_id in answers]
        if not insts:
            continue
        preds = {i.local_id: answers[i.instance_id]["prediction"] for i in insts}
        scored, summary = score_predictions(con, tier.items, preds, abstention=tier.prompt.abstention)
        by_local = {r["instance_id"]: r for r in scored}
        tier_rows = []
        for inst in insts:
            r = by_local[inst.local_id]
            q = inst.question
            if tier.prompt.abstention:
                kind = "abstain" if r["abstained"] else ("right" if r["correct"] and q.answerable else "wrong")
            else:
                kind = "right" if r["correct"] else "wrong"
            tier_rows.append({"instance_id": inst.instance_id, "tier": tier.name, "question_id": q.id,
                              "lang": inst.lang, "category": q.category, "difficulty": q.difficulty,
                              "answerable": q.answerable, "label": "answerable" if q.answerable else "unanswerable",
                              "prediction": r["sql"], "outcome": answers[inst.instance_id]["outcome"],
                              "error": r["error"], "correct": bool(r["correct"]), "kind": kind,
                              "pred_rows": r["pred_rows"]})
        groups = [f"{tier.name}/{r['question_id']}" for r in tier_rows]
        answerable = [n for n, r in enumerate(tier_rows) if r["answerable"]]
        metrics[f"{tier.name}_ex"] = cluster_ci([float(tier_rows[n]["correct"]) for n in answerable],
                                                [groups[n] for n in answerable])
        if tier.prompt.abstention:
            scores = ehrsql_scores([(r["answerable"], r["kind"]) for r in tier_rows])
            if scores != [by_local[i.local_id]["rs"] for i in insts]:
                raise RuntimeError("the EHRSQL-2024 scorer and text2sql.score_predictions disagree")
            n = len(scores)
            for penalty, key in ((0, "rs0" if tier.name == HEADLINE_TIER else f"{tier.name}_rs0"),
                                 (5, f"{tier.name}_rs5"), (n, f"{tier.name}_rsN")):
                entry = cluster_ci([s if s != -1 else -penalty for s in scores], groups)
                entry["value"] = rs(scores, penalty)
                if penalty == n:
                    entry["N"] = n
                metrics[key] = entry
            for row, s in zip(tier_rows, scores):
                row["score"] = s
            metrics["abstention"] = summary["abstention"]
        else:
            for row in tier_rows:
                row["score"] = 1 if row["correct"] else 0
        seen = Counter(r["outcome"] for r in tier_rows)
        counts[tier.name] = {o: seen.get(o, 0) for o in OUTCOMES}
        summaries[tier.name] = {k: v for k, v in summary.items() if k not in ("execution_accuracy", "abstention")}
        summaries[tier.name].update({"n_questions": len({r["question_id"] for r in tier_rows}),
                                     "n_instances": len(tier_rows)})
        rows += tier_rows
    for row in rows:
        row.pop("kind")
    metrics["answers"] = {t: {o: c[o] for o in OUTCOMES} for t, c in counts.items()}
    metrics["cut_off_rate"] = _rates(counts, "cut_off")
    metrics["format_failure_rate"] = _rates(counts, "format_failure")
    metrics["api_error_rate"] = _rates(counts, "api_error")["all"]
    metrics["parse_rate"] = sum(r["outcome"] in ("sql", "abstain") for r in rows) / len(rows) if rows else None
    metrics["tiers"] = summaries
    metrics["scorer"] = {"execution": f"{SCORER_VERSION} ({RS_SCORER_VERSION} with abstention)",
                         "reliability": f"glee4810/ehrsql-2024@{EHRSQL_COMMIT[:7]} scoring_utils.reliability_score "
                                        "and penalize",
                         "intervals": CI_METHOD}
    metrics["prompts"] = {t.name: t.prompt.id for t in tiers}
    return rows, metrics


# ---- runs ----

def run_config(tiers: list[Tier], max_tokens: int, limit: int | None = None) -> dict:
    """What a run asked; a run over part of the instances (`limit`) is marked partial and never listed."""
    config = {"prompts": {t.name: t.prompt.id for t in tiers}, "max_tokens": max_tokens, "temperature": 0,
              "tiers": [t.name for t in tiers]}
    if limit is not None:
        config.update({"limit": limit, "partial": True})
    return config


def load_cache(path: Path, max_tokens: int) -> dict[str, dict]:
    """A dry run's answers (`text2sql dry-run`: responses.jsonl, latest record per instance), refused as a whole
    when they were not all asked under `max_tokens`:

    - an answer that records its limit (dry runs made after the review of PR #109) must record this one;
    - any answer over `max_tokens` comes from a run with a larger limit;
    - an answer that records no limit, has neither SQL nor ABSTAIN and stopped below `max_tokens` may have been cut
      off by a smaller limit (it would count as a format failure, not as cut off), so it cannot stand in for this
      run's. Answers without SQL that used exactly `max_tokens` were cut off at it and are reused.

    For an answer without a recorded limit that has SQL, a smaller limit cannot be told apart from a short answer;
    such an answer is reused."""
    path = Path(path)
    file = path / "responses.jsonl" if path.is_dir() else path
    records = _load_responses(file)
    if not records:
        raise ValueError(f"{file}: no cached answers")
    other = sorted({r["max_tokens"] for r in records.values() if r.get("max_tokens") not in (None, max_tokens)})
    if other:
        raise ValueError(f"{file}: cached answers asked under an output limit of {', '.join(map(str, other))} "
                         f"tokens, not {max_tokens}, so they are not reused")
    over = sum((r.get("tokens_out") or 0) > max_tokens for r in records.values())
    if over:
        raise ValueError(f"{file}: {over} cached answer(s) over the output limit of {max_tokens} tokens, so it comes "
                         "from a run with a larger limit and is not reused")
    short = sorted(iid for iid, r in records.items()
                   if r.get("max_tokens") is None and not r.get("error")
                   and extract_answer(r.get("content")) is None and (r.get("tokens_out") or 0) < max_tokens)
    if short:
        raise ValueError(f"{file}: no output limit recorded, and {len(short)} answer(s) without SQL or ABSTAIN stopped "
                         f"below {max_tokens} tokens ({', '.join(short[:3])}{', ...' if len(short) > 3 else ''}), "
                         "so a smaller limit may have cut them off; run the dry run again (it now records its limit)")
    return records


async def run_text2sql_task(session, model, task, tiers: list[Tier], con, client, *, spec: dict,
                            reuse: dict[str, Path] | None = None, concurrency: int = 8, force: bool = False,
                            limit: int | None = None, data_dir: Path | None = None, progress=None):
    """Answer every instance once (resumable, like the other LLM runners) and score on the control plane.

    `reuse` maps a tier to a dry run's folder: an answer there is used when it came from the same model for the same
    prompt (`text2sql` prompt hash: model id and messages) without error. `client` may be None when every answer
    is cached. The request is the protocol's: temperature 0 and the spec's output limit, which a model must be able
    to give (a model whose registry limit is lower is refused)."""
    from benchtoward.db import ItemResult
    from benchtoward.evaluate import _find_run, _retry, _start_run
    from benchtoward.llm import ChatResult
    from benchtoward.runkey import run_key

    max_tokens = int(spec["request"]["max_tokens"])
    if model.max_tokens and model.max_tokens < max_tokens:
        raise ValueError(f"{model.slug} answers in at most {model.max_tokens} tokens; the protocol gives every model "
                         f"{max_tokens} (decision output_format), so this model cannot be run on the task")
    if data_dir is None:
        from benchtoward.config import get_settings
        data_dir = get_settings().data_dir
    schemas = check_schemas(spec, tiers, con)
    model_id = model.provider_model_id or model.slug
    insts = instances(tiers)
    insts = insts[:limit] if limit is not None else insts
    config = run_config(tiers, max_tokens, limit)
    key = run_key(model.digest, task.spec_hash, HARNESS_VERSION, config)
    run = _find_run(session, key)
    if run is not None and run.status == "done" and not force:
        return run

    messages = {i.instance_id: render_messages(i.question, i.lang, schemas[i.tier], i.prompt) for i in insts}
    shas = {iid: _prompt_sha(model_id, m) for iid, m in messages.items()}
    cached: dict[str, dict] = {}
    for tier_name, path in (reuse or {}).items():
        if tier_name not in TIERS:
            raise ValueError(f"unknown tier {tier_name!r} to reuse answers for (tiers: {', '.join(TIERS)})")
        records = load_cache(path, max_tokens)
        for inst in insts:
            record = records.get(inst.local_id)
            if inst.tier == tier_name and record and not record.get("error") \
                    and record.get("prompt_sha") == shas[inst.instance_id]:
                cached[inst.instance_id] = record

    finished = set()
    if run is not None and not force:
        finished = {r.instance_id for r in session.query(ItemResult).filter_by(run_id=run.id) if not _retry(r)}
    from_cache = [i for i in insts if i.instance_id not in finished and i.instance_id in cached]
    to_call = [i for i in insts if i.instance_id not in finished and i.instance_id not in cached]
    if to_call and client is None:
        raise ValueError(f"{len(to_call)} answer(s) need a model call (no cached answer to the same prompt) and no "
                         "client for this model's route is available")
    if run is None:
        run = _start_run(session, model, task, key, config, harness_version=HARNESS_VERSION)
    else:
        if force:
            session.query(ItemResult).filter_by(run_id=run.id).delete()
        run.status = "running"
        session.commit()

    for inst in from_cache:
        record = cached[inst.instance_id]
        session.merge(ItemResult(run_id=run.id, instance_id=inst.instance_id, error=None, transient=None,
                                 raw=json.dumps({"content": (record.get("content") or "")[:20000], "source": "dry_run"},
                                                ensure_ascii=False),
                                 tokens_in=int(record.get("tokens_in") or 0),
                                 tokens_out=int(record.get("tokens_out") or 0)))
    session.commit()

    semaphore = asyncio.Semaphore(concurrency)
    state = {"done": 0}

    async def one(inst: Instance) -> None:
        async with semaphore:
            try:
                result = await client.chat(model_id, messages[inst.instance_id], max_tokens=max_tokens,
                                           temperature=0)
            except Exception as exc:  # one broken call must not abort the whole evaluation
                result = ChatResult(content="", reasoning=None, tokens_in=0, tokens_out=0,
                                    error=f"exception_{type(exc).__name__}", transient=True)
        session.merge(ItemResult(run_id=run.id, instance_id=inst.instance_id, error=result.error,
                                 transient=result.transient if result.error else None,
                                 raw=json.dumps({"content": (result.content or "")[:20000], "source": "model"},
                                                ensure_ascii=False),
                                 tokens_in=result.tokens_in, tokens_out=result.tokens_out))
        session.commit()
        state["done"] += 1
        if progress is not None:
            progress(state["done"], len(to_call))

    await asyncio.gather(*(one(i) for i in to_call))
    sources = {t: str(p) for t, p in (reuse or {}).items()}
    return finish_text2sql_run(session, model, run, insts, tiers, con, max_tokens, data_dir, sources=sources)


def finish_text2sql_run(session, model, run, insts: list[Instance], tiers: list[Tier], con, max_tokens: int,
                        data_dir: Path, sources: dict | None = None):
    """Score every stored answer (re-read, so parser fixes apply to resumed runs) and write the predictions."""
    from benchtoward.db import ItemResult, utcnow
    from benchtoward.evaluate import _retry, write_predictions
    from benchtoward.llm import cost_usd

    results = {r.instance_id: r for r in session.query(ItemResult).filter_by(run_id=run.id)}
    run.n_items = len(insts)
    run.tokens_in = sum(r.tokens_in for r in results.values())
    run.tokens_out = sum(r.tokens_out for r in results.values())
    run.cost_usd = cost_usd(model, run.tokens_in, run.tokens_out)
    run.eval_date = date.today()
    if any(_retry(r) for r in results.values()) or any(i.instance_id not in results for i in insts):
        run.status, run.metrics_json = "failed", None
        session.commit()
        return run

    answers = {}
    for inst in insts:
        r = results[inst.instance_id]
        content = json.loads(r.raw or "{}").get("content")
        prediction, outcome = read_answer(content, r.error, r.tokens_out, max_tokens=max_tokens,
                                          abstention=inst.prompt.abstention)
        answers[inst.instance_id] = {"prediction": prediction, "outcome": outcome}
    rows, metrics = score_answers(con, tiers, answers)
    from_cache = sum(json.loads(r.raw or "{}").get("source") == "dry_run" for r in results.values())
    metrics.update({"max_tokens": max_tokens, "answers_from_cache": from_cache})
    for row in rows:
        r = results[row["instance_id"]]
        r.prediction, r.label, r.correct, r.score = row["prediction"], row["label"], row["correct"], row["score"]
        row.update({"tokens_out": r.tokens_out})
    run.metrics_json = json.dumps(metrics, ensure_ascii=False)
    run.internal_json = json.dumps({"answers_from_cache": from_cache, "cache": sources or {}})
    run.status = "done"
    run.finished_at = utcnow()
    session.commit()
    write_predictions(run, rows, data_dir)
    return run


def record_priors(session, task, tiers: list[Tier], con, *, data_dir: Path | None = None) -> list:
    """Record both fixed-rule priors as medbench baseline runs (always abstain; SELECT 0)."""
    from benchtoward.db import Model
    from benchtoward.evaluate import record_baseline_run

    runs = []
    for rule, (slug, description) in PRIOR_RULES.items():
        rows, metrics = score_answers(con, tiers, prior_answers(tiers, rule))
        config = {"rule": description, "tiers": list(TIERS), "prompts": {t.name: t.prompt.id for t in tiers}}
        runs.append(record_baseline_run(session, session.get(Model, slug), task, rows, metrics, config,
                                        data_dir=data_dir))
    return runs


# ---- the SOP check (stage 5) ----

def _prior_key(metric: str) -> str:
    return "rs" if metric == "rs0" or "_rs" in metric else "ex"


def _metric_check(check: dict, runs: dict[str, dict], metric: str) -> dict:
    weak, strong = check["roster"]["weak"], check["roster"]["strong"]
    prior_slug = check["priors"][_prior_key(metric)]
    prior = runs[prior_slug][metric]["value"]
    refs = {slug: runs[slug][metric] for slug in (weak, strong)}
    best = max(refs, key=lambda s: refs[s]["value"])
    h = (1.0 - refs[best]["value"]) / (1.0 - prior) if prior < 1.0 else None
    headroom = {"metric": metric, "best": best, "value": refs[best]["value"], "ci_high": refs[best]["ci_high"],
                "prior": prior, "prior_rule": prior_slug, "H": h,
                "pass": h is not None and h >= H_MIN and refs[best]["ci_high"] < 1.0,
                "warning": h is not None and H_MIN <= h < H_WARN}
    separation = {"metric": metric, "weak": [refs[weak]["ci_low"], refs[weak]["ci_high"]],
                  "strong": [refs[strong]["ci_low"], refs[strong]["ci_high"]],
                  "pass": refs[strong]["ci_low"] > refs[weak]["ci_high"]}
    beyond = {slug: refs[slug]["ci_low"] > prior for slug in refs}
    return {"headroom": headroom, "separation": separation,
            "prior": {"metric": metric, "value": prior, "rule": prior_slug, "references_beyond_prior": beyond}}


def sop_check(check: dict, runs: dict[str, dict]) -> dict:
    """The SOP's stage-5 check for each tier from the runs' metrics (slug -> metrics): headroom and separation on the
    tier's metric (hard: RS(0); easy: EX), format failures under 10% for both references (answers cut off at the
    limit reported apart, decision output_format), and API errors of 10% or more making the check incomplete.
    "Beats the prior" has no baseline to test: the only baselines are the two input-blind priors; whether the
    references' intervals lie beyond the prior is reported. Each tier's `caveats` say what its verdict rests on
    (`_caveats`)."""
    weak, strong = check["roster"]["weak"], check["roster"]["strong"]
    report = {"roster": dict(check["roster"]), "priors": dict(check["priors"]), "tiers": {}}
    for tier in TIERS:
        main = _metric_check(check, runs, HEADLINE[tier])
        fmt = {"failure": {s: runs[s]["format_failure_rate"][tier] for s in (weak, strong)},
               "cut_off": {s: runs[s]["cut_off_rate"][tier] for s in (weak, strong)}}
        fmt["pass"] = all(v < FORMAT_MAX for v in fmt["failure"].values())
        api = {s: runs[s]["api_error_rate"] for s in (weak, strong)}
        if any(v >= API_MAX for v in api.values()):
            verdict = "incomplete"
        else:
            verdict = "pass" if main["headroom"]["pass"] and main["separation"]["pass"] and fmt["pass"] else "fail"
        report["tiers"][tier] = {
            **main, "format": fmt, "api_errors": api,
            "beats_prior": {"baselines": [], "note": "no baseline beyond the two input-blind priors"},
            "also": {m: _metric_check(check, runs, m) for m in ALSO_CHECKED[tier]},
            "verdict": verdict, "caveats": _caveats(main, fmt, runs)}
    return report


def _caveats(main: dict, fmt: dict, runs: dict[str, dict]) -> list[dict]:
    """What a tier's verdict rests on beyond its three criteria (review of PR #109): "beats the prior" is not tested
    (no baseline beyond the priors); a reference whose interval on the tier's metric does not lie above the prior; and
    a format pass that holds only because answers cut off at the limit count as model failures (decision
    output_format, taken after the dry runs), with each reference's share if they counted as format failures."""
    out = [{"kind": "beats_prior_untested", "references": {},
            "text": "beats the prior untested (no baseline beyond the two input-blind priors)"}]
    p = main["prior"]
    within = {slug: [runs[slug][p["metric"]]["ci_low"], runs[slug][p["metric"]]["ci_high"]]
              for slug, beyond in p["references_beyond_prior"].items() if not beyond}
    if within:
        out.append({"kind": "within_prior", "references": within, "text": "; ".join(
            f"{slug}'s {p['metric']} interval [{lo:.3f}, {hi:.3f}] includes the prior {p['value']:.3f} ({p['rule']})"
            for slug, (lo, hi) in within.items())})
    counted = {slug: fmt["failure"][slug] + fmt["cut_off"][slug] for slug in fmt["failure"]}
    if fmt["pass"] and any(v >= FORMAT_MAX for v in counted.values()):
        over = {slug: v for slug, v in counted.items() if v >= FORMAT_MAX}
        out.append({"kind": "format_rests_on_cut_off", "references": over, "text": (
            "format passes only because answers cut off at the limit are model failures (decision output_format, "
            "taken after the dry runs); counted as format failures: "
            + ", ".join(f"{slug} {v:.1%}" for slug, v in over.items()) + f", over {FORMAT_MAX:.0%}")})
    return out
