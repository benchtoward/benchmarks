import numpy as np
import pytest

from benchtoward.scoring import accuracy_ci, binary_metrics_ci


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
    from benchtoward.scoring import paired_rank_ranges

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
    from benchtoward.scoring import paired_rank_ranges

    groups = np.repeat(np.arange(100), 3)
    ranges = paired_rank_ranges(lambda idx: [0.1, 0.3], n_items=300, groups=groups, higher_is_better=False, n_boot=50)
    assert ranges == [(1, 1), (2, 2)]


def test_shared_rank_counts_only_clearly_better_models():
    from benchtoward.scoring import paired_rank_stats

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
    from benchtoward.scoring import threshold_metrics_ci

    y = np.array([1, 1, 0, 0, 1, 0])
    s = np.array([0.9, 0.2, 0.6, 0.1, 0.5, 0.49])   # 0.5 counts as positive
    r = threshold_metrics_ci(y, s, np.arange(6), threshold=0.5, n_boot=50)
    assert r["accuracy"]["value"] == 4 / 6
    assert r["sensitivity"]["value"] == 2 / 3 and r["specificity"]["value"] == 2 / 3
    assert r["f1"]["value"] == 2 / 3 and r["threshold"] == 0.5


def test_threshold_metrics_perfect_and_cluster_bootstrap():
    from benchtoward.scoring import threshold_metrics_ci

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
    from benchtoward.scoring import threshold_metrics_ci

    y = np.array([1, 0, 0, 0])
    r = threshold_metrics_ci(y, np.array([0.9, 0.1, 0.1, 0.1]), np.array([0, 1, 2, 3]), n_boot=200)
    assert r["sensitivity"]["value"] == 1.0 and r["sensitivity"]["ci_low"] == 1.0   # resamples without a positive skipped
    none = threshold_metrics_ci(np.zeros(3, int), np.zeros(3), np.arange(3), n_boot=10)
    assert none["sensitivity"]["value"] is None and none["specificity"]["value"] == 1.0


# --- multi-class (rhythm and beat classes) ----------------------------------------------------------------------


def test_multiclass_metrics_known_confusion():
    from benchtoward.scoring import multiclass_metrics_ci

    y = np.array(["N"] * 6 + ["S"] * 3 + ["V"] * 1)
    p = np.array(["N", "N", "N", "N", "S", "Q", "S", "S", "N", "V"])   # Q: a valid answer that is no class here
    r = multiclass_metrics_ci(y, p, np.arange(10), classes=["N", "S", "V"], n_boot=50)
    f1 = {"N": 2 * 4 / (2 * 4 + 1 + 2), "S": 2 * 2 / (2 * 2 + 1 + 1), "V": 1.0}
    assert r["macro_f1"]["value"] == pytest.approx(np.mean(list(f1.values())))
    assert r["accuracy"]["value"] == 0.7 and r["n"] == 10 and r["n_groups"] == 10
    n, s = r["by_class"]["N"], r["by_class"]["S"]
    assert n["n"] == 6 and n["sensitivity"]["value"] == 4 / 6 and n["ppv"]["value"] == 4 / 5
    assert n["fpr"]["value"] == 1 / 4                      # one S beat called N, out of the four non-N beats
    assert s["sensitivity"]["value"] == 2 / 3 and s["ppv"]["value"] == 2 / 3
    assert s["f1"]["value"] == pytest.approx(f1["S"])
    assert r["confusion"]["labels"] == ["N", "S", "V"] and r["confusion"]["predicted"] == ["N", "S", "V", "other"]
    assert r["confusion"]["matrix"] == [[4, 1, 0, 1], [1, 2, 0, 0], [0, 0, 1, 0]]   # the Q answer is a miss only
    assert r["classes_scored"] == ["N", "S", "V"]


def test_multiclass_macro_f1_averages_classes_present_in_the_reference():
    from benchtoward.scoring import multiclass_metrics_ci

    y = np.array(["N", "N", "S", "S"])
    r = multiclass_metrics_ci(y, y.copy(), np.arange(4), classes=["N", "S", "V"], n_boot=50)
    assert r["macro_f1"]["value"] == 1.0 and r["classes_scored"] == ["N", "S"]
    assert r["by_class"]["V"]["n"] == 0 and r["by_class"]["V"]["f1"]["value"] is None
    wrong = multiclass_metrics_ci(y, np.array(["N", "V", "S", "S"]), np.arange(4), classes=["N", "S", "V"], n_boot=50)
    assert wrong["macro_f1"]["value"] == pytest.approx(np.mean([2 / 3, 1.0]))   # a false V lowers N's recall only
    assert wrong["by_class"]["V"]["ppv"]["value"] == 0.0


def test_multiclass_bootstrap_is_by_group_seeded_and_brackets_the_value():
    from benchtoward.scoring import multiclass_metrics_ci

    rng = np.random.default_rng(0)
    y = rng.choice(["A", "B", "C"], 600)
    p = np.where(rng.random(600) < 0.7, y, rng.choice(["A", "B", "C"], 600))
    g = np.repeat(np.arange(60), 10)
    r1 = multiclass_metrics_ci(y, p, g, classes=["A", "B", "C"], n_boot=200, seed=2)
    assert r1 == multiclass_metrics_ci(y, p, g, classes=["A", "B", "C"], n_boot=200, seed=2)
    for name in ("macro_f1", "accuracy"):
        assert r1[name]["ci_low"] <= r1[name]["value"] <= r1[name]["ci_high"]
        assert r1[name]["ci_low"] < r1[name]["ci_high"]
    one = multiclass_metrics_ci(y, p, np.zeros(600), classes=["A", "B", "C"], n_boot=50)
    assert one["macro_f1"]["ci_low"] == pytest.approx(one["macro_f1"]["value"])   # one group: every resample equal
    assert one["macro_f1"]["ci_high"] == pytest.approx(one["macro_f1"]["value"])


def test_multiclass_bootstrap_keeps_the_full_reference_classes_and_redraws_resamples_without_one():
    from sklearn.metrics import f1_score

    from benchtoward.scoring import multiclass_metrics_ci

    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(12), 20)
    y = rng.choice(["A", "B"], len(groups))
    y[0] = "C"                                          # C only in group 0 (one instance), answered right
    p = np.where(rng.random(len(groups)) < 0.6, y, rng.choice(["A", "B"], len(groups)))
    p[0] = "C"
    r = multiclass_metrics_ci(y, p, groups, classes=["A", "B", "C"], n_boot=200, seed=4)
    # the same bootstrap by hand: draw whole groups; a resample without a class of the full reference is drawn again
    gen, values, redraws = np.random.default_rng(4), [], 0
    while len(values) < 200:
        pick = gen.integers(0, 12, 12)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in pick])
        if not {"A", "B", "C"} <= set(y[idx]):
            redraws += 1
            continue
        values.append(f1_score(y[idx], p[idx], labels=["A", "B", "C"], average="macro"))
    assert r["bootstrap"] == {"n": 200, "by": "group", "redraws": redraws} and redraws > 0
    assert r["macro_f1"]["ci_low"] == pytest.approx(np.percentile(values, 2.5))
    assert r["macro_f1"]["ci_high"] == pytest.approx(np.percentile(values, 97.5))
    assert r["by_class"]["C"]["f1"]["ci_low"] == 1.0   # every resample kept C
    none = multiclass_metrics_ci(y, p, groups, classes=["A", "B", "C"], n_boot=0)
    assert none["bootstrap"] == {"n": 0, "by": "group", "redraws": 0} and none["macro_f1"]["ci_low"] is None


def test_multiclass_bootstrap_gives_up_on_a_reference_it_cannot_resample():
    from benchtoward.scoring import multiclass_metrics_ci

    groups = np.arange(400)
    y = np.array(["A"] * 360 + [f"C{i}" for i in range(40)])   # 40 classes in one group each
    classes = ["A", *[f"C{i}" for i in range(40)]]
    with pytest.raises(RuntimeError, match="redraw"):
        multiclass_metrics_ci(y, y.copy(), groups, classes=classes, n_boot=5)


def test_multiclass_bootstrap_caps_redraws_per_resample_not_in_total(monkeypatch):
    """The bootstrap gives up when one resample needs more than MAX_REDRAWS_PER_RESAMPLE redraws in a row, however
    few redraws the other resamples needed (review of PR #89)."""
    from benchtoward import scoring

    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(12), 20)
    y = rng.choice(["A", "B"], len(groups))
    y[0] = "C"                                          # C only in group 0: about a third of resamples lack it
    classes = ["A", "B", "C"]
    # the bootstrap's draws by hand: the longest run of redraws for one resample, and the redraws in total
    gen, accepted, run, longest, total = np.random.default_rng(4), 0, 0, 0, 0
    while accepted < 200:
        pick = gen.integers(0, 12, 12)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in pick])
        if not set(classes) <= set(y[idx]):
            run, total = run + 1, total + 1
            longest = max(longest, run)
            continue
        run, accepted = 0, accepted + 1
    assert 2 <= longest < total <= (longest - 1) * 200   # a cap on the total would never be reached here
    monkeypatch.setattr(scoring, "MAX_REDRAWS_PER_RESAMPLE", longest)
    r = scoring.multiclass_metrics_ci(y, y.copy(), groups, classes=classes, n_boot=200, seed=4)
    assert r["bootstrap"] == {"n": 200, "by": "group", "redraws": total}
    monkeypatch.setattr(scoring, "MAX_REDRAWS_PER_RESAMPLE", longest - 1)
    with pytest.raises(RuntimeError, match=f"more than {longest - 1} redraws"):
        scoring.multiclass_metrics_ci(y, y.copy(), groups, classes=classes, n_boot=200, seed=4)
