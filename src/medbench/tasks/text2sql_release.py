"""The Synthetic-SNUH Text-to-SQL items as a KHDP dataset: synthetic-snuh-text2sql 1.0.0 (owner, 2026-10-30).

The owner decided that the items of both tiers (questions and reference SQL) are published on KHDP as a dataset
built by medbench, submitted by the maintainers (SOP: KHDP datasets), while the code that wrote, reviewed and scores
them is public in github.com/medbench-kr/benchmarks. `build_release` turns the frozen question files that the task
spec pins into that dataset:

- items/<tier>.yaml: the kept items of a tier with the fields that fix them and their scoring (and, on the hard tier,
  the author's note on why an item is hard), readable by `text2sql.load_questions`; their hash is the one the spec
  pins (`text2sql_task.items_sha256`). Review notes, proposers and the decisions blocks stay in the question files.
- items/items.jsonl: both tiers, one JSON object per item.
- prompts/<tier>/: the system prompt, the database note, the schema text built from Synthetic-SNUH 4.0.0 and the
  layout of the user message, exactly as every evaluated model gets them.
- metadata.json, README.md, LICENSE.txt and SHA256SUMS.

Nothing is written when the items, a prompt or a schema differ from what the spec pins.
"""
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml

from medbench.tasks.text2sql import LANGS, PromptSpec, Question, render_messages
from medbench.tasks.text2sql_task import (TASK_REF, Tier, check_schemas, items_sha256, load_tiers, prompt_sha256,
                                          schema_sha256)

CODE, VERSION = "synthetic-snuh-text2sql", "1.0.0"
RELEASE_DIR = f"{CODE}-{VERSION}"  # under <data_dir>/khdp/ (git-ignored)
TITLE = "Synthetic-SNUH Text-to-SQL: Korean and English questions with reference SQL over OMOP CDM"
LICENCE = "CC BY-NC 4.0"
LICENCE_URL = "https://creativecommons.org/licenses/by-nc/4.0/"
PUBLIC_CODE = "https://github.com/medbench-kr/benchmarks"
PENDING = "the commit is added once the code is exported"
SOURCE = {
    "dataset": "synthetic-snuh@4.0.0",
    "title": "Synthetic-SNUH v4.0.0: synthetic OMOP CDM and KR Core FHIR records for 10,000 virtual patients",
    "doi": "10.23421/2026.03183",
    "licence": LICENCE,
    "copyright": "Copyright (c) 2026 Seoul National University Hospital",
    "citation": ("Lee H-C, Seoul National University Hospital. Synthetic-SNUH v4.0.0: synthetic OMOP CDM and KR Core "
                 "FHIR records for 10,000 virtual patients. KHDP. 2026. doi:10.23421/2026.03183"),
}
# Every item's fields in the release, in this order (`why_hard`: hard tier only, null on the easy tier).
RELEASE_FIELDS = ("id", "tier", "category", "difficulty", "scope", "question_ko", "question_en", "answerable", "sql",
                  "order_matters", "why_hard")
CODE_FILES = ("src/medbench/tasks/text2sql.py", "src/medbench/tasks/text2sql_task.py",
              "src/medbench/tasks/text2sql_cli.py", "third_party/ehrsql-2024/scoring_utils.py")
TIER_METRIC = {
    "hard": ("RS(0), the EHRSQL-2024 reliability score (headline); execution accuracy on the answerable questions, "
             "RS(5) and RS(N) next to it"),
    "easy": "execution accuracy (reported next to the hard tier, never ranked on its own)",
}


def release_item(tier: str, q: Question) -> dict:
    return {"id": q.id, "tier": tier, "category": q.category, "difficulty": q.difficulty, "scope": q.scope,
            "question_ko": q.question_ko, "question_en": q.question_en, "answerable": q.answerable, "sql": q.sql,
            "order_matters": q.order_matters, "why_hard": q.why_hard}


class _Dumper(yaml.SafeDumper):
    """Multi-line strings (reference SQL) as literal blocks, so the files read like the question files."""


_Dumper.add_representer(str, lambda dumper, value: dumper.represent_scalar(
    "tag:yaml.org,2002:str", value, style="|" if "\n" in value else None))


def write_items(path: Path, tier: Tier) -> str:
    """A tier's kept items as a YAML question file; returns their hash (the one the spec pins)."""
    sha = items_sha256(tier.items)
    rows = [{k: v for k, v in release_item(tier.name, q).items() if k != "why_hard" or v} for q in tier.items]
    head = {"dataset": f"{CODE}@{VERSION}", "tier": tier.name, "task": TASK_REF, "source": SOURCE["dataset"],
            "licence": LICENCE, "prompt": {"id": tier.prompt.id}, "items_sha256": sha, "questions": rows}
    text = (f"# {TITLE}, {tier.name} tier ({len(tier.items)} items). {LICENCE}: see README.md and LICENSE.txt.\n"
            + yaml.dump(head, Dumper=_Dumper, allow_unicode=True, sort_keys=False, width=120))
    Path(path).write_text(text, encoding="utf-8")
    return sha


def user_template(prompt: PromptSpec) -> str:
    """The user message with {schema}, {database_note} and {question} in place of the text (`render_messages`)."""
    q = Question(id="x", question_ko="{question}", question_en="{question}", sql=None, category="unanswerable",
                 difficulty="hard", answerable=False)
    stand_in = PromptSpec(prompt.id, prompt.system, "{database_note}", prompt.abstention, prompt.null_columns)
    return render_messages(q, "en", "{schema}", stand_in)[1]["content"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_release(spec: dict, registry_root: Path, con, out: Path, *, code_commit: str | None = None) -> dict:
    """Write the release into `out` (replacing an earlier build there) and return its metadata. Raises ValueError,
    writing nothing, when the items, a prompt or a schema differ from what `spec` pins."""
    tiers = load_tiers(spec, registry_root)
    schemas = check_schemas(spec, tiers, con)
    out = Path(out)
    work = out.with_name(out.name + ".partial")
    shutil.rmtree(work, ignore_errors=True)
    (work / "items").mkdir(parents=True)
    meta = {"dataset": CODE, "version": VERSION, "title": TITLE, "licence": LICENCE, "licence_url": LICENCE_URL,
            "doi": None, "source": SOURCE, "task": TASK_REF, "site": "https://medbench.kr",
            "code": {"repository": PUBLIC_CODE, "commit": code_commit, "licence": "Apache-2.0",
                     "files": list(CODE_FILES)},
            "languages": list(LANGS), "request": {"temperature": spec["request"].get("temperature", 0),
                                                  "max_tokens": spec["request"]["max_tokens"]},
            "tiers": {}, "instances": 0}
    rows = []
    for tier in tiers:
        sha = write_items(work / "items" / f"{tier.name}.yaml", tier)
        rows += [release_item(tier.name, q) for q in tier.items]
        folder = work / "prompts" / tier.name
        folder.mkdir(parents=True)
        for name, text in (("system.txt", tier.prompt.system), ("database_note.txt", tier.prompt.conventions),
                           ("schema.txt", schemas[tier.name]), ("user_template.txt", user_template(tier.prompt))):
            (folder / name).write_text(text, encoding="utf-8")
        answerable = sum(q.answerable for q in tier.items)
        meta["tiers"][tier.name] = {
            "file": f"items/{tier.name}.yaml", "questions": len(tier.items), "answerable": answerable,
            "unanswerable": len(tier.items) - answerable, "instances": len(tier.items) * len(LANGS),
            "categories": dict(Counter(q.category for q in tier.items)), "items_sha256": sha,
            "prompt": {"id": tier.prompt.id, "sha256": prompt_sha256(tier.prompt),
                       "schema_sha256": schema_sha256(schemas[tier.name]), "folder": f"prompts/{tier.name}"},
            "abstention": tier.prompt.abstention, "metric": TIER_METRIC.get(tier.name, "execution accuracy")}
        meta["instances"] += len(tier.items) * len(LANGS)
    (work / "items" / "items.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (work / "metadata.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (work / "README.md").write_text(readme(meta), encoding="utf-8")
    (work / "LICENSE.txt").write_text(licence_text(), encoding="utf-8")
    files = sorted(str(p.relative_to(work)) for p in work.rglob("*") if p.is_file())
    (work / "SHA256SUMS").write_text("".join(f"{_sha256(work / f)}  {f}\n" for f in files), encoding="utf-8")
    if out.exists():
        shutil.rmtree(out)
    work.rename(out)
    return meta


def _counts(categories: dict) -> str:
    return ", ".join(f"{name} {n}" for name, n in categories.items())


def readme(meta: dict) -> str:
    hard, easy = meta["tiers"]["hard"], meta["tiers"]["easy"]
    n = hard["questions"] + easy["questions"]
    commit = meta["code"]["commit"]
    at = f"commit {commit}" if commit else PENDING
    return f"""# Synthetic-SNUH Text-to-SQL {meta['version']}

Korean and English questions about the Synthetic-SNUH 4.0.0 database (OMOP CDM v5.4), each with a reference SQL
query, for evaluating Text-to-SQL models. There are {n} questions in two tiers, each asked in Korean and in English
({meta['instances']} scored instances):

- a hard tier of {hard['questions']} questions, {hard['unanswerable']} of which the database cannot answer;
- an easy tier of {easy['questions']} questions.

medbench.kr built the items for its benchmark `{meta['task']}`. They were written for this database; none was copied
from another benchmark.

The data behind the questions are synthetic, so the answers say nothing about real SNUH patients.

## Files

| File | Content |
|---|---|
| `items/hard.yaml` | the hard tier: {hard['questions']} items, its prompt id and the items' hash |
| `items/easy.yaml` | the easy tier: {easy['questions']} items, likewise |
| `items/items.jsonl` | both tiers, one JSON object per item |
| `prompts/<tier>/system.txt` | the tier's system prompt |
| `prompts/<tier>/database_note.txt` | the note on the database's conventions that every prompt carries |
| `prompts/<tier>/schema.txt` | the schema text built from Synthetic-SNUH 4.0.0: CREATE TABLE statements, 3 example rows per non-empty table, the empty tables, concepts counted by domain and vocabulary, and on the hard tier the columns that are NULL in every row |
| `prompts/<tier>/user_template.txt` | how the user message is put together from the schema, the note and the question |
| `metadata.json` | counts, hashes, prompt ids, the source dataset and the evaluation code |
| `LICENSE.txt` | licence and attribution |
| `SHA256SUMS` | SHA-256 of every other file |

Item fields: `id`, `tier`, `category`, `difficulty`, `scope` (patient scope as in EHRSQL: individual, group or none),
`question_ko`, `question_en`, `answerable` (false when the database cannot answer the question, so the correct answer
is to abstain), `sql` (the reference query; null for an unanswerable item), `order_matters` (true when the question
asks for a sorted list, so row order is compared) and, on the hard tier, `why_hard` (the writer's note on what a model
must get right).

Categories: hard tier {_counts(hard['categories'])}; easy tier {_counts(easy['categories'])}.

## How the items were made

- Easy tier: drafted by the medbench agent (Claude) from the data on 2026-10-29.
- Hard tier: written on 2026-10-30 by GPT-6 Astra, Gemini 4 Argon and Claude Opus 5.5. Each category was written in
  batches that rotated over the three models. A writer saw the schema and database note that evaluated models get,
  plus a profile of the concepts in use and of the data's quality problems.
- Checks: every reference query ran on the database. It had to be one read-only SELECT returning 1 to 1,000 rows that
  are not all zero or NULL, use no clock function, have a full ORDER BY when row order matters, and differ from every
  other item's query. Korean questions do not use 내원, which can be read as outpatient visits only. Hard-tier
  questions name concepts in clinical words and never quote a concept name.
- Review: GPT-6 Astra, Gemini 4 Argon and Claude Opus 5.5 judged every item with a JSON rubric (a natural question,
  one reasonable reading, the Korean and English asking the same thing, a correct reference query, a plausible
  result). An item was kept only when all three approved it. The same models revised the items that needed an edit:
  one model proposed a revision (the proposer rotating over up to three rounds), the revised query had to pass the
  checks, and all three reviewed it again. An item not approved after three rounds was dropped.
- Design questions, such as the headline metric, the output limit and how the easy tier is published, were decided by
  a vote of the three models, two of three deciding.
- No clinician reviewed the items.

## Tiers and scoring

The model's query and the reference query run on the same read-only DuckDB copy of Synthetic-SNUH 4.0.0, and their
results are compared:

- rows as multisets, and in order only for items with `order_matters`;
- column names and column order ignored (a permutation search, as in the Spider test-suite evaluation), but the number
  of columns must match;
- numbers compared after rounding to 3 decimals (EHRSQL-2024); NULL equals only NULL; dates compared as YYYY-MM-DD;
- a query that fails, is not a single SELECT, runs over 30 s or returns more than 10,000 rows is wrong.

Hard tier ({hard['questions']} questions, prompt `{hard['prompt']['id']}`): today is fixed at 2025-12-31, and a model
answers ABSTAIN when the database cannot answer. The headline is the EHRSQL-2024 reliability score RS(0): +1 for a
right answer to an answerable question and for an abstention on an unanswerable one, 0 for an abstention on an
answerable one, and -c for any other answer (c = 0 for RS(0)). The medbench.kr task computes it with the shared task's
own `scoring_utils.py`. Execution accuracy on the answerable questions, RS(5) and RS(N) are reported next to it.

Easy tier ({easy['questions']} questions, prompt `{easy['prompt']['id']}`): execution accuracy. Open models answered
nearly all of it in medbench's trial runs, so it is reported next to the hard tier and never ranked on its own.

Request: temperature {meta['request']['temperature']} and {meta['request']['max_tokens']:,} output tokens, thinking
included. An answer whose thinking uses the whole limit and leaves no SQL is wrong. The prompt follows Rajkumar et al.
2022 (CREATE TABLE statements with 3 example rows) with a database note in the role of BIRD's external knowledge.

## Evaluate a model

The evaluation code is in {meta['code']['repository']} ({at}; Apache-2.0): `text2sql.py` (prompt, execution and result
comparison), `text2sql_task.py` (the task: reliability scores through the EHRSQL-2024 scorer and intervals from a
bootstrap over questions) and `text2sql_cli.py` (commands). Put the item files where you like, then:

```bash
uv sync
export KHDP_TOKEN=...                     # KHDP account settings -> API Token
uv run medbench fetch synthetic-snuh      # Synthetic-SNUH 4.0.0 from KHDP, MD5-checked, loaded into DuckDB
uv run medbench text2sql prompt syn-hard-001 --lang ko --draft items/hard.yaml    # the messages of one instance
uv run medbench text2sql dry-run --draft items/hard.yaml --predictions predictions.jsonl
```

A predictions file has one JSON object per line with `instance_id` (`<id>-ko` or `<id>-en`) and `sql`; on the hard tier
null, the string `null` or `ABSTAIN` abstains. `--model <slug>` asks a model through an OpenAI-compatible route
instead (see the repository's README). The items' hash (`items_sha256`, over every item's id, questions, reference
query and labels) is the one the medbench.kr task pins; a changed item makes a new version.

## Licence and attribution

{LICENCE} ({LICENCE_URL}), the licence of the source data; non-commercial use only. See LICENSE.txt.

This dataset is built on Synthetic-SNUH v4.0.0 by Hyung-Chul Lee, Seoul National University Hospital ({SOURCE['copyright']};
{LICENCE}; doi:{SOURCE['doi']}). medbench.kr wrote the questions, reference queries, prompts and metadata. The schema
files hold 3 example rows of each non-empty table and counts of concepts from that release, unchanged. The questions
and queries name its concepts, which come from OHDSI Athena (SNOMED CT, RxNorm, LOINC, ATC) and keep their own
licences; the concept table is not included.

## Citation

- medbench.kr. Synthetic-SNUH Text-to-SQL {meta['version']}: Korean and English questions with reference SQL over
  Synthetic-SNUH 4.0.0. KHDP. 2026. (DOI assigned by KHDP)
- {SOURCE['citation']}

Scoring and design follow:

- Lee G, Kweon S, Bae S, Choi E. Overview of the EHRSQL 2024 Shared Task on Reliable Text-to-SQL Modeling on
  Electronic Health Records. ClinicalNLP @ NAACL 2024. doi:10.18653/v1/2024.clinicalnlp-1.62
- Lee G, Hwang H, Bae S, Kwon Y, Shin W, Yang S, Seo M, Kim J-Y, Choi E. EHRSQL: A Practical Text-to-SQL Benchmark for
  Electronic Health Records. NeurIPS 35 Datasets and Benchmarks, 2022. arXiv:2301.07695
- Zhong R, Yu T, Klein D. Semantic Evaluation for Text-to-SQL with Distilled Test Suites. EMNLP 2020.
- Legrand P, Noor K, Bhagwanani S, Dobson RJ. An OMOP-Based Open-Source Text-to-SQL Benchmark Dataset. ClinicalNLP @
  LREC 2026, 381-393.
- Rajkumar N, Li R, Bahdanau D. Evaluating the Text-to-SQL Capabilities of Large Language Models. arXiv:2204.00498,
  2022.

## Limitations

- The data are synthetic and meant for building queries and agents, not for clinical inference.
- GPT-6 Astra, Gemini 4 Argon and Claude Opus 5.5 wrote the hard items and reviewed both tiers, and the medbench agent
  (Claude) drafted the easy items. No clinician reviewed them, and a question may allow a reading the reviewers missed.
  These models have seen the items and their answers, so their scores are not comparable with other models'.
- Each item has one reference query. An answer is judged by its result, so an equivalent query counts as right; one
  that returns extra or missing columns does not.
- The unanswerable questions ask only for data that has no place in this database (tables present but empty, columns
  NULL in every row, both visible in the prompt), not for requests outside its scope.
- There is no training or development split: every item is a test item. Models trained after this release may have
  seen the items and their answers.
- The easy tier is close to saturated for current open models, and the output limit caps models that think for long.
"""


def licence_text() -> str:
    return f"""{TITLE} {VERSION}

Licence: Creative Commons Attribution-NonCommercial 4.0 International ({LICENCE})
{LICENCE_URL} (legal code: {LICENCE_URL}legalcode)

The questions, reference SQL queries, prompts and metadata were written by medbench.kr (2026) for Synthetic-SNUH
v4.0.0.

Source data: Synthetic-SNUH v4.0.0 by Hyung-Chul Lee, Seoul National University Hospital.
{SOURCE['copyright']}. Licensed under {LICENCE}. doi:{SOURCE['doi']}
Apart from the schema files, this dataset contains no source records. The schema files (prompts/*/schema.txt) hold 3
example rows of each non-empty table and counts of concepts by domain and vocabulary, taken from the release unchanged,
because the evaluated prompts show them; they are redistributed under the source's licence with this attribution.

The vocabulary concepts named in the questions and queries and shown in the schema files come from OHDSI Athena
(SNOMED CT, RxNorm, LOINC, ATC and OMOP structural vocabularies) and remain subject to their own licences. The concept
table is not included.

The evaluation code is published separately under the Apache License 2.0 ({PUBLIC_CODE}); the EHRSQL-2024 scorer it
includes is licensed CC BY 4.0 by the shared task's organisers.
"""
