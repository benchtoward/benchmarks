# Inspire-AKI-prediction (MIT)

Source: https://github.com/StealthX051/Inspire-AKI-prediction at tag `v1.0.0-jamia-open`, commit
4dc799c47b839664276b6fcd7890c55f9ac9cd00, the analysis code of Do J, Shah KH, Xu MC, Kim AH, Suresh V, Guggilla N,
Li M, Kothari R. Integration of intraoperative data in interpretable machine learning models to predict
postoperative AKI in noncardiac surgery patients. JAMIA Open. 2026. doi:10.1093/jamiaopen/ooag092. The authors ran
it on INSPIRE 1.3. Licence: MIT, Copyright (c) 2026 Justin Do, Karan H. Shah, Melissa Chen Xu, Andrew H. Kim,
Vivaswat Suresh, Nidhir Guggilla, Michael Li, and Rishi Kothari ([LICENSE](LICENSE)).

`inspire-aki@0` follows the paper as this code computes it (owner decision 2026-10-30), and uses it as follows
(configuration `configs/aki/default.yaml`).

- `src/benchtoward/tasks/inspire_aki.py` ports `cohort/filters.py` and `cohort/preop.py` (operation filters, derived
  inputs, the pre-operative laboratory and ward values), `cohort/labels.py` (`derive_aki_labels`, with its CRRT rule:
  any CRRT record of the patient), and `clinical_baselines/gs_aki.py` (the adapted GS-AKI count). The inputs are the
  columns the pre-operative model reads (`datasets/tabular.py`, `models/tabular.py`), `op_len` and
  `booking_case_length` included.
- `registry/inspire-aki/intraperitoneal_proxy_map_5char.csv` is
  `configs/clinical_baselines/intraperitoneal_proxy_map_5char.csv`, byte for byte.
- `registry/inspire-aki/procedure-audit-inspire-1.3.csv` holds the keep or exclude decision that
  `reporting/procedure_audit.py` (`annotate_operations_with_procedure_audit`, run unchanged with the CMS ICD-10-PCS
  order file of 1 April 2026) made for the candidate operations of INSPIRE 1.3, keyed by the three values its rules
  read (5-character ICD-10-PCS code, CTS department, recorded CPB times). Its rules are not ported.
- `registry/splits/inspire-aki-1.3-patients.csv` is outer fold 0 of `evaluation/split_manager.py`
  (`build_grouped_nested_cv_manifest`), run unchanged with the authors' scikit-learn 1.5.2;
  `registry/inspire-aki/outer-folds-inspire-1.3.csv` holds all five outer folds of the same run (report only).
- `src/benchtoward/baselines/inspire_aki.py` ports the pre-processing (`features/normalization.py`,
  `features/missingness.py`), the ridge logistic regression and XGBoost of `models/tabular.py`, the class weights
  of `models/weighting.py` and the isotonic calibration of `evaluation/calibration.py`, with the seeds of
  `pipelines/train.py`. Their hyperparameters come from the authors' Optuna search (`models/hpo.py`, run unchanged on
  the train part of each outer fold and of each report variant).

What differs, and why, is listed in the task specification (`registry/tasks/inspire-aki.yaml`, `deviations` and
`differences`). The intraoperative features, sequence models, AutoGluon, SVM, KNN, MLP and random forest are not
ported.
