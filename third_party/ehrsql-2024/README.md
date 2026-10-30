# ehrsql-2024 (CC BY 4.0)

Source: https://github.com/glee4810/ehrsql-2024 at commit f9e1aa02160d39e3f8df52bf5c69c5cf2e472499 (2026-06-25,
the repository's latest commit when the Text-to-SQL task was frozen on 2026-10-30). `scoring_program/scoring_utils.py`
was last changed at e207f57a2d0af32b273d7ed69c37e30fa9862b68 ("scoring update", 2024-03-20) and is the same file at
both commits. The EHRSQL 2024 shared task on reliable Text-to-SQL for electronic health records (Clinical NLP Workshop
@ NAACL 2024) was organised by Gyubok Lee, Sunjun Kweon, Seongsu Bae and Edward Choi (EdLab, KAIST). Licence:
Creative Commons Attribution 4.0 International ([LICENSE](LICENSE), copied from the repository).

Cite: Lee G, Kweon S, Bae S, Choi E. Overview of the EHRSQL 2024 Shared Task on Reliable Text-to-SQL Modeling on
Electronic Health Records. Proceedings of the 6th Clinical Natural Language Processing Workshop (ClinicalNLP @ NAACL
2024). doi:10.18653/v1/2024.clinicalnlp-1.62.

`synthetic-snuh-text2sql@0` uses it as follows.

- `scoring_utils.py` is the file at the pinned commit, unchanged (sha256
  `cfd8eb05bfcfad05e6dcb7003a9508d8034180236af80153039ae58a12c159ad`). `src/medbench/tasks/text2sql_task.py` loads it
  and calls its `reliability_score` (the per-question score: +1, 0 or -1) and `penalize` (RS(c), the mean with -1
  counted as -c) for every reliability score the task reports, as the shared task's `scoring_program/scoring.py` does
  for RS(0), RS(5), RS(10) and RS(N).
- Not used: `execute_sql`, `execute_all` and `process_answer`, which run the queries on the shared task's SQLite file
  and compare the first 100 sorted rows. medbench.kr runs both queries on DuckDB and compares results with its own
  rules (`src/medbench/tasks/text2sql.py`, `results_match`); `reliability_score` receives one token per question that
  says whether the answer was right, wrong or an abstention. Each difference is a listed deviation in
  `registry/tasks/synthetic-snuh-text2sql.yaml`.
- The abstain-all prior (`medbench/prior`) is the shared task's simplest baseline (abstain on every question), named
  in its README.
