"""lydus_labels: the dataset-provided LYDUS label columns as binary targets, and their metrics."""
import numpy as np
import pandas as pd
import pytest

from benchtoward.tasks import lydus_labels as L


def meta(*rows):
    return pd.DataFrame(rows, columns=list(L.COLUMNS))


def col(name):
    return L.TARGETS.index(name)


def test_targets_are_25_and_primary_is_a_fixed_subset_of_15():
    assert len(L.TARGETS) == 25 and len(set(L.TARGETS)) == 25
    assert len(L.PRIMARY) == 15 and set(L.PRIMARY) <= set(L.TARGETS)


def test_definite_values_are_positive_and_other_values_of_the_column_known_negative():
    y, known = L.encode(meta(("A.fib", "VPC", "RBBB", "1AVB")))
    for t in ("rhythm:A.fib", "premature beat:VPC", "bbb:RBBB", "avb:1AVB"):
        assert y[0, col(t)] and known[0, col(t)]
    for t in ("rhythm:NSR", "premature beat:APC", "bbb:LBBB", "avb:IVB"):
        assert not y[0, col(t)] and known[0, col(t)]


def test_nsp_is_a_known_negative_except_in_rhythm_where_it_masks_the_column():
    y, known = L.encode(meta(("NSP", "NSP", "NSP", "NSP")))
    assert not y.any()
    rhythm = [col(t) for t in L.TARGETS if t.startswith("rhythm:")]
    others = [col(t) for t in L.TARGETS if not t.startswith("rhythm:")]
    assert not known[0, rhythm].any() and known[0, others].all()


@pytest.mark.parametrize("value", ["Possible pacing", "Undetermined", "Others"])
def test_uncertain_rhythm_values_mask_every_rhythm_target(value):
    y, known = L.encode(meta((value, "NSP", "NSP", "NSP")))
    assert not known[0, [col(t) for t in L.TARGETS if t.startswith("rhythm:")]].any()


def test_apc_vpc_makes_both_premature_beat_targets_positive():
    y, known = L.encode(meta(("NSR", "APC,VPC", "NSP", "NSP")))
    assert y[0, col("premature beat:APC")] and y[0, col("premature beat:VPC")]


def test_composite_bundle_branch_values_do_not_switch_on_their_parts():
    y, _ = L.encode(meta(("NSR", "NSP", "BFB(RBBB+LAFB)", "NSP")))
    assert y[0, col("bbb:BFB(RBBB+LAFB)")]
    assert not y[0, col("bbb:RBBB")] and not y[0, col("bbb:LAFB")]


def test_value_outside_the_vocabulary_masks_its_column_and_is_audited():
    m = meta(("NSR", "NSP", "NSP", "Mobitz"))
    _, known = L.encode(m)
    assert not known[0, [col(t) for t in L.TARGETS if t.startswith("avb:")]].any()
    assert L.audit(m) == {"avb": {"Mobitz": 1}}
    assert L.audit(meta(("Possible pacing", "APC,VPC", "BFB", "NSP"))) == {}


def test_instances_lists_round_trip_to_the_same_matrices():
    y, known = L.encode(meta(("A.fib", "VPC", "NSP", "NSP"), ("Undetermined", "NSP", "LBBB", "IVB")))
    lists = [L.target_names(y[r], known[r]) for r in range(2)]
    y2, known2 = L.matrices([p for p, _ in lists], [u for _, u in lists])
    assert (known2 == known).all() and ((y2 & known2) == (y & known)).all()


def test_eligible_needs_positives_negatives_and_positive_groups():
    n = 60
    y = np.zeros((n, len(L.TARGETS)), dtype=bool)
    known = np.ones_like(y)
    y[:20, col("rhythm:A.fib")] = True              # 20 positives in 20 groups: eligible
    y[:20, col("rhythm:NSR")] = True                # 20 positives but all in 5 groups: not
    y[:19, col("bbb:RBBB")] = True                  # 19 positives: not
    groups = np.array([f"g{i}" for i in range(n)], dtype=object)
    groups_nsr = groups.copy()
    groups_nsr[:20] = [f"h{i % 5}" for i in range(20)]
    assert "rhythm:A.fib" in L.eligible(y, known, groups)
    assert "bbb:RBBB" not in L.eligible(y, known, groups)
    y_only_nsr = np.zeros_like(y)
    y_only_nsr[:, col("rhythm:NSR")] = y[:, col("rhythm:NSR")]
    assert "rhythm:NSR" not in L.eligible(y_only_nsr, known, groups_nsr)


def test_auroc_matches_sklearn_and_counts_ties_half():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    lab = rng.random(200) < 0.3
    sc = np.round(rng.random(200), 1)               # many ties
    assert L.auroc(lab, sc) == pytest.approx(roc_auc_score(lab, sc))
    assert L.auroc(lab, np.zeros(200)) == pytest.approx(0.5)
    assert np.isnan(L.auroc(np.zeros(5, bool), np.arange(5)))


def test_metrics_ignore_unknown_records_and_give_perfect_scores_for_a_perfect_model():
    rng = np.random.default_rng(0)
    n = 400
    y = rng.random((n, len(L.TARGETS))) < 0.3
    known = rng.random(y.shape) < 0.9
    scores = np.where(y, 0.9, 0.1)
    scores[~known] = 1 - scores[~known]             # wrong on unknown entries: must not count
    out = L.metrics(y, known, scores, groups=np.arange(n), n_boot=50)
    assert out["macro_auroc"]["value"] == pytest.approx(1.0)
    assert out["macro_f1"]["value"] == pytest.approx(1.0)
    assert out["macro_auroc"]["ci_low"] == pytest.approx(1.0)
    assert set(out["per_target"]) == set(L.PRIMARY)


def test_constant_scores_give_macro_auroc_one_half():
    n = 300
    rng = np.random.default_rng(2)
    y = rng.random((n, len(L.TARGETS))) < 0.3
    out = L.metrics(y, np.ones_like(y), np.full(y.shape, 0.2), groups=np.arange(n), n_boot=20)
    assert out["macro_auroc"]["value"] == pytest.approx(0.5)


def test_bootstrap_resamples_whole_groups_and_is_seeded():
    n = 200
    rng = np.random.default_rng(3)
    y = rng.random((n, len(L.TARGETS))) < 0.4
    scores = np.clip(y + rng.normal(0, 0.6, y.shape), 0, 1)
    groups = np.repeat(np.arange(50), 4)
    a = L.metrics(y, np.ones_like(y), scores, groups, n_boot=100)
    b = L.metrics(y, np.ones_like(y), scores, groups, n_boot=100)
    assert a == b
    assert a["macro_auroc"]["n_groups"] == 50
    assert a["macro_auroc"]["ci_low"] < a["macro_auroc"]["value"] < a["macro_auroc"]["ci_high"]


def test_thresholds_fall_back_to_one_half_with_few_validation_positives():
    n = 100
    y = np.zeros((n, len(L.TARGETS)), dtype=bool)
    y[:5, col("rhythm:A.fib")] = True
    y[:30, col("bbb:RBBB")] = True
    scores = np.where(y, 0.3, 0.1)
    th = L.thresholds_from_validation(y, np.ones_like(y), scores)
    assert th["rhythm:A.fib"] == 0.5
    assert th["bbb:RBBB"] == pytest.approx(0.3)


def test_metrics_are_strict_json_even_when_a_target_has_no_positives():
    import json
    n = 120
    rng = np.random.default_rng(4)
    y = rng.random((n, len(L.TARGETS))) < 0.3
    y[:, col("avb:AV dissociation")] = False                 # as in the real test split
    scores = rng.random(y.shape)
    out = L.metrics(y, np.ones_like(y), scores, np.arange(n), n_boot=20, targets=L.TARGETS)
    json.dumps(out, allow_nan=False)
    assert out["per_target"]["avb:AV dissociation"]["auroc"] is None
