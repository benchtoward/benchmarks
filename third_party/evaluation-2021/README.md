# evaluation-2021 (BSD-2-Clause)

Source: https://github.com/physionetchallenges/evaluation-2021 at commit e2a75fc01f729cb74cc4e853e054ce81e28381fc,
the evaluation code of the PhysioNet/Computing in Cardiology Challenge 2021 (Reyna MA et al. Will Two Do? Varying
Dimensions in Electrocardiography: The PhysioNet/Computing in Cardiology Challenge 2021. Comput Cardiol. 2021;48).
Licence: BSD-2-Clause, Copyright (c) 2020, 2021 PhysioNet/Computing in Cardiology Challenges ([LICENSE](LICENSE)).

Files here are byte for byte those of the pinned commit: `evaluate_model.py`, `helper_code.py`, `weights.csv`,
`weights_abbreviations.csv`, `dx_mapping_scored.csv`, `dx_mapping_unscored.csv` and the upstream README
(`README.upstream.md`). `evaluate_model.m` is not copied.

`lydus-ecg12-dx@0` uses them as follows.

- `src/benchtoward/metrics_challenge2021.py` copies the scoring functions of `evaluate_model.py` unchanged (only
  `load_weights` gets a default path) and adds vectorised forms that take per-record bootstrap counts.
  `tests/test_metrics_challenge2021.py` compares the port with this copy, function by function and end to end on
  Challenge-format label and output files.
- `weights.csv` defines the 26 scored classes and the Challenge metric; the two dx mapping tables are the SNOMED CT
  vocabulary of `registry/maps/lydus-dx-snomed-v0.csv`.
