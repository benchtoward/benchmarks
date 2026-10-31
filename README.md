# BenchToward benchmarks

The evaluation code behind [BenchToward](https://benchtoward.com): how each benchmark is defined, how its instances
and splits are built from open SNUH datasets, how predictions are scored, and the baselines. Results, per-item
predictions and leaderboards are on the site; this repository is what produced them, so anyone can check it and
report problems.

Where the field has already published an evaluation (a paper with code), a benchmark adopts it and lists every
deviation in its specification (`deviations`). Fixes never change a released version silently: they ship as a new
task version.

## Benchmarks

| Task | What it measures | Data | Adopted from | Code |
|---|---|---|---|---|
| [`clinicalqa@0`](https://benchtoward.com/tasks/clinicalqa) | Korean clinical knowledge (KMLE-level MCQ) | [ClinicalQA: Korean Medical Knowledge QA Benchmark](https://doi.org/10.23421/2026.0800) | BenchToward definition (no published protocol) | [spec](registry/tasks/clinicalqa.yaml), [`clinicalqa.py`](src/benchtoward/tasks/clinicalqa.py), [`evaluate.py`](src/benchtoward/evaluate.py), [`scoring.py`](src/benchtoward/scoring.py) |
| [`inspire-mortality@0`](https://benchtoward.com/tasks/inspire-mortality) | 30-day postoperative in-hospital mortality | [INSPIRE](https://khdp.net/database/data-search-detail/660/inspire/1.3) | [INSPIRE authors' 30-day in-hospital mortality task (gbm_mortality.py)](https://github.com/vitaldb/inspire/blob/04a249028b9290729a1f8027322bfcc1101739ed/gbm_mortality.py) | [spec](registry/tasks/inspire-mortality.yaml), [`inspire_mortality.py`](src/benchtoward/tasks/inspire_mortality.py), [`inspire.py`](src/benchtoward/baselines/inspire.py) |
| [`k-or-speech-asr@0`](https://benchtoward.com/tasks/k-or-speech-asr) | Korean operating-room speech recognition | [Synthetic K-OR Speech Audio](https://khdp.net/database/data-search-detail/774/Synthetic-K-OR-Speech-Audio/1.0.0) | BenchToward definition (no published protocol) | [spec](registry/tasks/k-or-speech-asr.yaml) |
| [`kor-surgical-robot-voice-command@0`](https://benchtoward.com/tasks/kor-surgical-robot-voice-command) | Korean surgical robot voice commands | [Korean Surgical Robot Voice-Command Benchmark](https://khdp.net/database/data-search-detail/3507/kor-surgical-robot-voice-command/1.0.0) | Dataset author's evaluation code (prompt, scorer, paired comparison), ported unchanged | [spec](registry/tasks/kor-surgical-robot-voice-command.yaml), [`voice_command.py`](src/benchtoward/tasks/voice_command.py), [`voice_command_cli.py`](src/benchtoward/tasks/voice_command_cli.py) |
| [`scope-arrest@0`](https://benchtoward.com/tasks/scope-arrest) | ICU cardiac arrest within 0.5-24 h from a 5-min ECG epoch | [SCOPE: ICU ECG and PPG before cardiac arrest, death or discharge](https://khdp.net/database/data-search-detail/698/icu-cardiac-arrest/1.0) | BenchToward definition | [spec](registry/tasks/scope-arrest.yaml), [`scope_arrest.py`](src/benchtoward/tasks/scope_arrest.py), [`scope_sampling.py`](src/benchtoward/tasks/scope_sampling.py), [`scope_hrv.py`](src/benchtoward/tasks/scope_hrv.py), [`scope_arrest.py`](src/benchtoward/baselines/scope_arrest.py) |
| [`synthetic-snuh-text2sql@0`](https://benchtoward.com/tasks/synthetic-snuh-text2sql) | Synthetic-SNUH Text-to-SQL (OMOP CDM, Korean and English) | [Synthetic-SNUH v4](https://doi.org/10.23421/2026.03183) | [EHRSQL 2024 shared task scoring (reliability score) with Spider test-suite execution matching](https://github.com/glee4810/ehrsql-2024) | [spec](registry/tasks/synthetic-snuh-text2sql.yaml), [`text2sql_task.py`](src/benchtoward/tasks/text2sql_task.py), [`text2sql.py`](src/benchtoward/tasks/text2sql.py), [`text2sql_task_cli.py`](src/benchtoward/tasks/text2sql_task_cli.py), [`synthetic_snuh.py`](src/benchtoward/tasks/synthetic_snuh.py), [`README.md`](third_party/ehrsql-2024/README.md) |
| [`vitaldb-arrhythmia-abnormal@0`](https://benchtoward.com/tasks/vitaldb-arrhythmia-abnormal) | Abnormal rhythm in 5-second intraoperative ECG (lead II) | [VitalDB Arrhythmia Database](https://khdp.net/database/data-search-detail/738/vitaldb-arrhythmia/1.0.0) | [Eun DI, Shim K, Lee H, Lim Y, Lim H, Lee H, Lee J, Lee HC](https://github.com/vitaldb/arrdb) | [spec](registry/tasks/vitaldb-arrhythmia-abnormal.yaml), [`vitaldb_arrhythmia_abnormal.py`](src/benchtoward/tasks/vitaldb_arrhythmia_abnormal.py), [`vitaldb_arrhythmia.py`](src/benchtoward/tasks/vitaldb_arrhythmia.py), [`arrhythmia.py`](src/benchtoward/baselines/arrhythmia.py) |
| [`vitaldb-ioh@0`](https://benchtoward.com/tasks/vitaldb-ioh) | Intraoperative hypotension 5 min ahead (numeric MAP) | [VitalDB](https://khdp.net/database/data-search-detail/658/vitaldb_open/1.0.0) | BenchToward definition (no published protocol) | [spec](registry/tasks/vitaldb-ioh.yaml), [`vitaldb_ioh.py`](src/benchtoward/tasks/vitaldb_ioh.py), [`vitaldb_split.py`](src/benchtoward/tasks/vitaldb_split.py), [`ioh.py`](src/benchtoward/baselines/ioh.py) |

## Code published ahead of its task

The maintainers published this code before its benchmark. Its tests run here without the benchmark's items or specification (those that need them skip, saying why).

| Task | Why | Code |
|---|---|---|
| `synthetic-snuh-text2sql@0` (in review) | owner, 2026-10-30: the code that wrote, reviewed, fixed and voted on the Synthetic-SNUH Text-to-SQL items (`benchtoward text2sql ...`) and the code that scores them are public while the task is in review. The items themselves (questions and reference SQL of both tiers) are released on KHDP as the BenchToward-built dataset synthetic-snuh-text2sql, not in this repository. | [`text2sql.py`](src/benchtoward/tasks/text2sql.py), [`text2sql_task.py`](src/benchtoward/tasks/text2sql_task.py), [`text2sql_task_cli.py`](src/benchtoward/tasks/text2sql_task_cli.py), [`text2sql_cli.py`](src/benchtoward/tasks/text2sql_cli.py), [`text2sql_review.py`](src/benchtoward/tasks/text2sql_review.py), [`text2sql_llm_review.py`](src/benchtoward/tasks/text2sql_llm_review.py), [`text2sql_llm_fix.py`](src/benchtoward/tasks/text2sql_llm_fix.py), [`text2sql_llm_decide.py`](src/benchtoward/tasks/text2sql_llm_decide.py), [`text2sql_hard.py`](src/benchtoward/tasks/text2sql_hard.py), [`text2sql_release.py`](src/benchtoward/tasks/text2sql_release.py) |

`synthetic-snuh-text2sql@0`: The frontier-model jobs (`text2sql hard-generate`, `hard-review`, `llm-review`, `llm-fix`, `llm-decide`) ask the models through the route that `registry/providers.yaml` marks `frontier: true`; its optional `prices_url` (a public model list with prices) and `request.billing` (request fields that make the route report its bill) feed their budget. With the items from KHDP, `benchtoward text2sql dry-run --draft items/hard.yaml --predictions FILE` scores a model's answers.

## Reproduce a benchmark

```bash
uv sync
export KHDP_TOKEN=...            # KHDP account settings -> API Token (open datasets on khdp.net)
uv run benchtoward init          # local SQLite database and the registry
uv run benchtoward fetch inspire && uv run benchtoward build inspire-mortality
uv run benchtoward baseline inspire-mortality
uv run pytest -q
```

Data is never included here: each builder downloads its dataset from its source under that dataset's license.
VitalDB waveforms come from the `vitaldb` package. LLM tasks need an OpenAI-compatible endpoint: add a route to
`registry/providers.yaml` (`name: {base_url: ..., key_env: ...}`) and a model entry in `registry/models.yaml`.
A task whose protocol fixes request settings says so in its spec; `kor-surgical-robot-voice-command` needs the route
to declare how it switches thinking off (`request: {thinking_off: {...}}`, see `route_request` in
`src/benchtoward/tasks/voice_command.py`).

## Report a problem

Found something wrong with a benchmark (a label rule, a split, a metric, a deviation from the adopted protocol)?
Open an issue with the "Evaluation problem" template, naming the task and version. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

Code: Apache License 2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). Datasets keep their own licenses (see each task
page). Results published on BenchToward are CC BY 4.0.

BenchToward is an independent project using open SNUH datasets. It is not an official service of Seoul National
University Hospital.
