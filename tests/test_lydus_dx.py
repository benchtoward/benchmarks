"""LYDUS free-text diagnosis (machine interpretation statements) to SNOMED CT codes and Challenge 2021 classes."""
import csv

import numpy as np
import pandas as pd
import pytest

from conftest import REGISTRY_DIR
from benchtoward import metrics_challenge2021 as cm
from benchtoward.tasks import lydus_dx as dx

TABLE = dx.load_map(REGISTRY_DIR / "maps" / dx.MAP_FILE)


def parse(text):
    return dx.parse_dx(text, TABLE)


def test_map_file_is_a_reviewable_table_with_known_roles():
    with (REGISTRY_DIR / "maps" / dx.MAP_FILE).open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == ["statement", "n_records", "role", "codes", "hedged_codes", "note"]
    assert {r["role"] for r in rows} <= set(dx.ROLES)
    assert len({r["statement"] for r in rows}) == len(rows)
    known = set(dx.vocabulary())
    for r in rows:
        for code in filter(None, (r["codes"] + ";" + r["hedged_codes"]).split(";")):
            assert code in known, (r["statement"], code)   # every code is in the Challenge 2021 tables


def test_vocabulary_holds_the_scored_and_unscored_challenge_codes():
    vocab = dx.vocabulary()
    assert vocab["426783006"] == "sinus rhythm" and vocab["164873001"] == "left ventricular hypertrophy"
    assert len(vocab) > 100


def test_ge_statements_with_continuation_lines():
    p = parse("Sinus rhythm\nwith 1st degree A-V block\nwith\nPremature atrial complexes\nAbnormal ECG\n")
    assert p.codes == {"426783006", "270492004", "284470004"} and not p.hedged and not p.no_analysis
    p = parse("Atrial fibrillation\nwith premature ventricular or aberrantly conducted complexes\n"
              "Right bundle branch block\nAbnormal ECG\n")
    assert p.codes == {"164889003", "59118001"} and p.hedged == {"427172004"}   # "or": hedged (owner, 2026-11-03)


def test_hedges_apply_to_the_next_finding_only():
    p = parse("Normal sinus rhythm\nPossible\nLeft atrial enlargement\nLeft axis deviation\nBorderline ECG\n")
    assert p.codes == {"426783006", "39732003"} and p.hedged == {"67741000119109"}
    p = parse("Normal sinus rhythm\nCannot rule out\n, age undetermined\nInferior infarct\n")
    assert "164865005" in p.hedged and "164865005" not in p.codes
    p = parse("Unusual P axis, possible ectopic atrial rhythm\nOtherwise normal ECG\n")
    assert p.codes == set() and p.hedged == {"106068003"}


def test_modifiers_and_paced_complexes():
    p = parse("Ventricular-paced rhythm\nwith occasional\nsupraventricular complexes\nAbnormal ECG\n")
    assert p.codes == {"10370003"}
    p = parse("Sinus rhythm\nwith frequent\nventricular-paced complexes\nAbnormal ECG\n")
    assert p.codes == {"426783006", "251266004"}       # pacing pattern (unscored), not a paced rhythm
    p = parse("Atrial-paced rhythm\nwith prolonged AV conduction\nAbnormal ECG\n")
    assert p.codes == {"10370003"}                     # no prolonged PR interval


def test_schiller_statements_and_explanations():
    p = parse(" WPW syndrome\n  because PR duration <120 ms and QRS duration >=120 ms and Rp duration in\n"
              "  [I|II|aVL|V4|V5|V6] >=60 ms and PR duration + QRS duration <300 ms\n tachycardia\n"
              "  because HR >100 per minute\n prolonged QT\n  because QTc (Bazett) >470 ms\n probably abnormal ECG\n")
    assert p.codes == {"74390002", "111975006"}


def test_no_analysis_and_unknown_lines():
    assert parse("*** No QRS complexes found, no ECG analysis possible\n").no_analysis
    p = parse("Sinus rhythm\nA statement nobody has seen\n")
    assert p.unknown == ["A statement nobody has seen"]
    with pytest.raises(ValueError, match="not in the map"):
        dx.parse_all(pd.Series(["A statement nobody has seen\n"]), TABLE)


def test_challenge_labels_use_equivalent_classes():
    y = dx.challenge_labels({"59118001", "63593006", "164873001"})
    assert y.dtype == bool and y.shape == (26,)
    assert y[cm.class_index("713427006")] and y[cm.class_index("284470004")] and y.sum() == 2


def test_statement_counts_count_records_not_lines():
    counts = dx.statement_counts(pd.Series(["with\nSinus rhythm\nwith\n", "Sinus rhythm\n"]))
    assert counts == {"with": 1, "Sinus rhythm": 2}


def test_parse_all_returns_one_row_per_record():
    parsed = dx.parse_all(pd.Series(["Sinus rhythm\n", "Atrial fibrillation\nPossible\nAtrial flutter\n"]), TABLE)
    assert parsed.labels.shape == (2, 26)
    assert parsed.labels[1, cm.class_index("164889003")] and not parsed.labels[1, cm.class_index("164890007")]
    assert parsed.hedged_labels[1, cm.class_index("164890007")]
    assert parsed.codes[0] == ["426783006"]
    assert np.array_equal(parsed.no_analysis, [False, False])


def test_alternatives_joined_by_or_are_hedged_not_labels():
    """Owner decision 2026-11-03 (option B): a statement that offers alternatives is not a definite finding."""
    p = parse("Atrial fibrillation\nwith premature ventricular or aberrantly conducted complexes\nAbnormal ECG\n")
    assert "427172004" not in p.codes and "427172004" in p.hedged      # PVC: hedged, not a label
    assert "164889003" in p.codes                                       # AF stays definite
    p = parse("Sinus rhythm\nwith\nPremature ventricular complexes\nor\nFusion complexes\nRight bundle branch block\n")
    assert {"427172004", "13640000"} <= p.hedged and not {"427172004", "13640000"} & p.codes
    assert "59118001" in p.codes                                        # RBBB after the alternative stays definite


def test_a_code_stated_definitely_elsewhere_stays_a_label():
    p = parse("Premature ventricular complexes\nAtrial fibrillation\nwith premature ventricular or aberrantly "
              "conducted complexes\n")
    assert "427172004" in p.codes and "427172004" not in p.hedged


def test_an_alternative_never_reaches_back_across_a_summary_line():
    """Codex review of the 'or' rule: a summary line ends the statement group."""
    p = parse("Sinus rhythm\nAbnormal ECG\nor\nFusion complexes\n")
    assert "426783006" in p.codes and "13640000" in p.hedged
