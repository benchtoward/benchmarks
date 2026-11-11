"""LYDUS split: likely repeat ECGs of one patient (no patient id in the release) are linked and kept on one side."""
import numpy as np
import pandas as pd
import pytest

from benchtoward.tasks import lydus_split as ls

HZ = 500


def ecg(rr_s=0.8, n=5000, shape=1.0, seed=0, noise=2.0):
    """8 stored leads with a QRS-like spike train; `shape` scales the leads differently (a patient's morphology)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / HZ
    beat = np.zeros(n)
    for c in np.arange(0.3, n / HZ - 0.3, rr_s):
        beat += np.exp(-((t - c) / 0.012) ** 2) * 800 - np.exp(-((t - c - 0.25) / 0.06) ** 2) * 120 * shape
    gains = np.array([0.6, 1.0, -0.5 * shape, -0.2, 0.3 * shape, 0.8, 1.1 * shape, 0.9])
    return (gains[:, None] * beat[None, :] + rng.normal(0, noise, (8, n))).astype(np.int16)


def test_beat_template_is_eight_leads_by_sixty_points():
    t = ls.beat_template(ecg())
    assert t.shape == (8, 60) and np.isfinite(t).all()
    assert np.isnan(ls.beat_template(np.zeros((8, 5000), np.int16))).all()   # flat: no beats


def test_templates_correlate_for_the_same_morphology_not_another():
    a, b, c = (ls.beat_template(ecg(shape=1.0, seed=s, rr_s=r)) for s, r in ((1, 0.8), (2, 0.9), (3, 0.8)))
    other = ls.beat_template(ecg(shape=-2.5, seed=4))
    Z, ok = ls.unit_vectors(np.stack([a, b, c, other]))
    corr = Z @ Z.T
    assert ok.all() and corr[0, 1] > 0.96 and corr[0, 2] > 0.96 and corr[0, 3] < 0.9


def test_template_pairs_blocked_search_finds_all_pairs_above_the_threshold():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(50, 6))
    Z[10] = Z[3] + 0.01
    Z[40] = Z[3] + 0.02
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    ok = np.ones(50, bool)
    i, j, corr = ls.template_pairs(Z, ok, 0.99, block=7)
    assert set(zip(i.tolist(), j.tolist())) == {(3, 10), (3, 40), (10, 40)} and (corr >= 0.99).all()
    ok[40] = False
    i, j, _ = ls.template_pairs(Z, ok, 0.99, block=7)
    assert set(zip(i.tolist(), j.tolist())) == {(3, 10)}


def meta(rows):
    return pd.DataFrame(rows, columns=["new_id", "sex", "age"])


def test_link_rules_need_matching_sex_and_age():
    m = meta([("000001", "M", 70), ("000004", "M", 71), ("000020", "M", 75), ("000030", "F", 70),
              ("000031", "F", 70), ("000090", None, 70), ("000095", "M", None)])
    i = np.array([0, 0, 0, 5, 6])
    j = np.array([1, 2, 3, 6, 1])
    corr = np.full(5, 0.99)
    edges = ls.link_pairs(m, (i, j, corr))
    pairs = set(zip(edges["i"], edges["j"], edges["rule"]))
    assert (0, 1, "template") in pairs                 # same sex, 1 year apart
    assert not any(p[:2] == (0, 2) for p in pairs)      # 5 years apart
    assert not any(p[:2] == (0, 3) for p in pairs)      # other sex
    assert (3, 4, "index") in pairs                     # adjacent ids, same sex and age, no template needed
    assert (0, 1, "index") in pairs                     # ids 3 apart, ages 1 year apart
    assert not any(5 in p[:2] or 6 in p[:2] for p in pairs)   # unknown sex or age: never linked


def test_groups_are_components_named_by_their_smallest_id():
    ids = np.array(["000005", "000002", "000009", "000001"])
    edges = pd.DataFrame({"i": [0, 1], "j": [1, 2], "rule": ["template", "index"]})
    groups = ls.groups_from_links(ids, edges)
    assert groups.tolist() == ["000002", "000002", "000002", "000001"]


def test_assign_split_keeps_groups_whole_and_is_deterministic():
    groups = np.array([f"g{k // 3:04d}" for k in range(3000)])
    a = ls.assign_split(groups, seed=20261030, fractions=(0.7, 0.1, 0.2))
    b = ls.assign_split(groups[::-1].copy(), seed=20261030, fractions=(0.7, 0.1, 0.2))[::-1]
    assert np.array_equal(a, b)                         # independent of row order
    df = pd.DataFrame({"g": groups, "s": a})
    assert (df.groupby("g")["s"].nunique() == 1).all()
    share = df["s"].value_counts(normalize=True)
    assert abs(share["train"] - 0.7) < 0.01 and abs(share["validation"] - 0.1) < 0.01


def test_leakage_estimate_counts_cross_split_neighbours_against_the_other_sex_null():
    m = meta([("000001", "M", 70), ("000002", "M", 70), ("000003", "F", 70), ("000100", "M", 70)])
    Z = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    ok = np.ones(4, bool)
    split = np.array(["test", "train", "train", "train"])
    out = ls.leakage_estimate(split, m, Z, ok, thresholds=(0.95,))
    row = out["template"]["0.95"]
    assert row == {"test_records_with_same_sex_neighbour": 1, "with_other_sex_neighbour": 1, "excess": 0}
    assert out["index"]["10"]["test_records_with_same_sex_age_neighbour"] == 1


def test_public_split_file_drops_the_group_column_only():
    full = "new_id,group,split\n000000,000000,train\n000001,000000,train\n000002,000002,test\n"
    assert ls.public_split_bytes(full.encode()) == b"new_id,split\n000000,train\n000001,train\n000002,test\n"
    with pytest.raises(ValueError):
        ls.public_split_bytes(b"new_id,split\n000000,train\n")   # only a full split file is reduced
