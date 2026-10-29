# medbench.kr benchmarks

The evaluation code behind [medbench.kr](https://medbench.kr): how each benchmark is defined, how its instances
and splits are built from open SNUH datasets, how predictions are scored, and the baselines. Results, per-item
predictions and leaderboards are on the site; this repository is what produced them, so anyone can check it and
report problems.

Where the field has already published an evaluation (a paper with code), a benchmark adopts it and lists every
deviation in its specification (`deviations`). Fixes never change a released version silently: they ship as a new
task version.

## Benchmarks

| Task | What it measures | Data | Adopted from | Code |
|---|---|---|---|---|
| [`clinicalqa@0`](https://medbench.kr/tasks/clinicalqa) | Korean clinical knowledge (KMLE-level MCQ) | [ClinicalQA: Korean Medical Knowledge QA Benchmark](https://doi.org/10.23421/2026.0800) | medbench.kr definition (no published protocol) | [spec](registry/tasks/clinicalqa.yaml), [`clinicalqa.py`](src/medbench/tasks/clinicalqa.py), [`evaluate.py`](src/medbench/evaluate.py), [`scoring.py`](src/medbench/scoring.py) |
| [`inspire-mortality@0`](https://medbench.kr/tasks/inspire-mortality) | 30-day postoperative in-hospital mortality | [INSPIRE](https://khdp.net/database/data-search-detail/660/inspire/1.3) | [INSPIRE authors' 30-day in-hospital mortality task (gbm_mortality.py)](https://github.com/vitaldb/inspire/blob/04a249028b9290729a1f8027322bfcc1101739ed/gbm_mortality.py) | [spec](registry/tasks/inspire-mortality.yaml), [`inspire_mortality.py`](src/medbench/tasks/inspire_mortality.py), [`inspire.py`](src/medbench/baselines/inspire.py) |
| [`k-or-speech-asr@0`](https://medbench.kr/tasks/k-or-speech-asr) | Korean operating-room speech recognition | [Synthetic K-OR Speech Audio](https://khdp.net/database/data-search-detail/774/Synthetic-K-OR-Speech-Audio/1.0.0) | medbench.kr definition (no published protocol) | [spec](registry/tasks/k-or-speech-asr.yaml) |
| [`scope-arrest@0`](https://medbench.kr/tasks/scope-arrest) | ICU cardiac arrest within 0.5-24 h from a 5-min ECG epoch | [SCOPE: ICU ECG and PPG before cardiac arrest, death or discharge](https://khdp.net/database/data-search-detail/698/icu-cardiac-arrest/1.0) | Lee H, Yang HL, Ryu HG, Jung CW, Cho YJ, Yoon SB, Yoon HK, Lee HC | [spec](registry/tasks/scope-arrest.yaml), [`scope_arrest.py`](src/medbench/tasks/scope_arrest.py), [`scope_sampling.py`](src/medbench/tasks/scope_sampling.py), [`scope_hrv.py`](src/medbench/tasks/scope_hrv.py), [`scope_arrest.py`](src/medbench/baselines/scope_arrest.py) |
| [`vitaldb-arrhythmia-abnormal@0`](https://medbench.kr/tasks/vitaldb-arrhythmia-abnormal) | Abnormal rhythm in 5-second intraoperative ECG (lead II) | [VitalDB Arrhythmia Database](https://khdp.net/database/data-search-detail/738/vitaldb-arrhythmia/1.0.0) | [Eun DI, Shim K, Lee H, Lim Y, Lim H, Lee H, Lee J, Lee HC](https://github.com/vitaldb/arrdb) | [spec](registry/tasks/vitaldb-arrhythmia-abnormal.yaml), [`vitaldb_arrhythmia_abnormal.py`](src/medbench/tasks/vitaldb_arrhythmia_abnormal.py), [`vitaldb_arrhythmia.py`](src/medbench/tasks/vitaldb_arrhythmia.py), [`arrhythmia.py`](src/medbench/baselines/arrhythmia.py) |
| [`vitaldb-ioh@0`](https://medbench.kr/tasks/vitaldb-ioh) | Intraoperative hypotension 5 min ahead (numeric MAP) | [VitalDB](https://khdp.net/database/data-search-detail/658/vitaldb_open/1.0.0) | medbench.kr definition (no published protocol) | [spec](registry/tasks/vitaldb-ioh.yaml), [`vitaldb_ioh.py`](src/medbench/tasks/vitaldb_ioh.py), [`vitaldb_split.py`](src/medbench/tasks/vitaldb_split.py), [`ioh.py`](src/medbench/baselines/ioh.py) |

## Reproduce a benchmark

```bash
uv sync
export KHDP_TOKEN=...            # KHDP account settings -> API Token (open datasets on khdp.net)
uv run medbench init             # local SQLite database and the registry
uv run medbench fetch inspire && uv run medbench build inspire-mortality && uv run medbench baseline inspire-mortality
uv run pytest -q
```

Data is never included here: each builder downloads its dataset from its source under that dataset's license.
VitalDB waveforms come from the `vitaldb` package. LLM tasks need an OpenAI-compatible endpoint: add a route to
`registry/providers.yaml` (`name: {base_url: ..., key_env: ...}`) and a model entry in `registry/models.yaml`.

## Report a problem

Found something wrong with a benchmark (a label rule, a split, a metric, a deviation from the adopted protocol)?
Open an issue with the "Evaluation problem" template, naming the task and version. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

Code: Apache License 2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). Datasets keep their own licenses (see each task
page). Results published on medbench.kr are CC BY 4.0.

medbench.kr is an independent project using open SNUH datasets. It is not an official service of Seoul National
University Hospital.
