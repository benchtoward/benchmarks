import numpy as np

from medbench.scoring import accuracy_ci, binary_metrics_ci


def test_accuracy_counts_none_as_wrong():
    r = accuracy_ci([True, False, None, True])
    assert r["value"] == 0.5 and r["n"] == 4 and r["n_unparsed"] == 1


def test_accuracy_ci_brackets_value_and_is_seeded():
    r1 = accuracy_ci([True] * 70 + [False] * 30, seed=1)
    r2 = accuracy_ci([True] * 70 + [False] * 30, seed=1)
    assert r1 == r2
    assert r1["ci_low"] <= 0.7 <= r1["ci_high"]
    assert r1["ci_low"] < r1["ci_high"]


def test_binary_metrics_perfect_and_grouped():
    y = np.array([0, 0, 1, 1] * 25)
    s = y.astype(float)
    g = np.repeat(np.arange(50), 2)
    r = binary_metrics_ci(y, s, g, n_boot=200)
    assert r["auroc"]["value"] == 1.0
    assert r["n_groups"] == 50 and r["n"] == 100
    assert r["prevalence"] == 0.5
    assert r["brier"]["value"] == 0.0


def test_binary_metrics_ci_brackets_value():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    s = np.clip(y * 0.3 + rng.random(400) * 0.7, 0, 1)
    g = np.repeat(np.arange(100), 4)
    r = binary_metrics_ci(y, s, g, n_boot=200, seed=3)
    for name in ("auroc", "auprc", "brier"):
        assert r[name]["ci_low"] <= r[name]["value"] <= r[name]["ci_high"]


def test_paired_rank_ranges_separated_and_tied():
    from medbench.scoring import paired_rank_ranges

    rng = np.random.default_rng(0)
    strong = rng.random(1000) < 0.9
    weak = rng.random(1000) < 0.5
    ranges = paired_rank_ranges(lambda idx: [strong[idx].mean(), weak[idx].mean()], n_items=1000, n_boot=200)
    assert ranges == [(1, 1), (2, 2)]

    twin = strong.copy()  # same answers except 3 items it gets wrong and 3 others it gets right
    right, wrong = np.flatnonzero(strong)[:3], np.flatnonzero(~strong)[:3]
    twin[right], twin[wrong] = False, True
    ranges = paired_rank_ranges(lambda idx: [strong[idx].mean(), twin[idx].mean()], n_items=1000, n_boot=200)
    assert ranges == [(1, 2), (1, 2)]


def test_paired_rank_ranges_lower_is_better_and_groups():
    from medbench.scoring import paired_rank_ranges

    groups = np.repeat(np.arange(100), 3)
    ranges = paired_rank_ranges(lambda idx: [0.1, 0.3], n_items=300, groups=groups, higher_is_better=False, n_boot=50)
    assert ranges == [(1, 1), (2, 2)]


def test_shared_rank_counts_only_clearly_better_models():
    from medbench.scoring import paired_rank_stats

    rng = np.random.default_rng(0)
    a = rng.random(1000) < 0.90
    b = a.copy()                         # tied with a: differs on 6 items, 3 each way
    right, wrong = np.flatnonzero(a)[:3], np.flatnonzero(~a)[:3]
    b[right], b[wrong] = False, True
    c = rng.random(1000) < 0.60          # clearly worse than both
    stats = paired_rank_stats(lambda idx: [a[idx].mean(), b[idx].mean(), c[idx].mean()], n_items=1000, n_boot=200)
    assert [s["shared"] for s in stats] == [1, 1, 3]
    assert stats[0]["range"] == (1, 2) and stats[2]["range"] == (3, 3)


def test_threshold_metrics_known_confusion():
    from medbench.scoring import threshold_metrics_ci

    y = np.array([1, 1, 0, 0, 1, 0])
    s = np.array([0.9, 0.2, 0.6, 0.1, 0.5, 0.49])   # 0.5 counts as positive
    r = threshold_metrics_ci(y, s, np.arange(6), threshold=0.5, n_boot=50)
    assert r["accuracy"]["value"] == 4 / 6
    assert r["sensitivity"]["value"] == 2 / 3 and r["specificity"]["value"] == 2 / 3
    assert r["f1"]["value"] == 2 / 3 and r["threshold"] == 0.5


def test_threshold_metrics_perfect_and_cluster_bootstrap():
    from medbench.scoring import threshold_metrics_ci

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    g = np.repeat(np.arange(100), 4)
    perfect = threshold_metrics_ci(y, y.astype(float), g, n_boot=100)
    assert all(perfect[k]["value"] == 1.0 and perfect[k]["ci_low"] == 1.0 for k in
               ("accuracy", "sensitivity", "specificity", "f1"))
    noisy = np.clip(y * 0.4 + rng.random(400) * 0.6, 0, 1)
    r = threshold_metrics_ci(y, noisy, g, n_boot=200, seed=2)
    for k in ("accuracy", "sensitivity", "specificity", "f1"):
        assert r[k]["ci_low"] <= r[k]["value"] <= r[k]["ci_high"] and r[k]["ci_low"] < r[k]["ci_high"]
    assert r == threshold_metrics_ci(y, noisy, g, n_boot=200, seed=2)


def test_threshold_metrics_skip_resamples_without_a_class():
    from medbench.scoring import threshold_metrics_ci

    y = np.array([1, 0, 0, 0])
    r = threshold_metrics_ci(y, np.array([0.9, 0.1, 0.1, 0.1]), np.array([0, 1, 2, 3]), n_boot=200)
    assert r["sensitivity"]["value"] == 1.0 and r["sensitivity"]["ci_low"] == 1.0   # resamples without a positive skipped
    none = threshold_metrics_ci(np.zeros(3, int), np.zeros(3), np.arange(3), n_boot=10)
    assert none["sensitivity"]["value"] is None and none["specificity"]["value"] == 1.0
