"""SCOPE in-unit cardiac arrest: the leakage-safe window rule, labels, instance ids and the stay split.

Times are seconds relative to the outcome anchor (2100-01-01 00:00:00 UTC); negative means before the outcome.
"""
import numpy as np
import pandas as pd
import pytest

from medbench.tasks.scope_sampling import (ANCHOR_UTC, BLACKOUT_S, CASE_EPOCHS, CONTROL_EPOCHS, EPOCH_S,
                                           case_epoch_starts, control_block_fraction, control_epoch_starts,
                                           epoch_starts, instance_id, label_for)

H = 3600
FULL = (-48 * H, 10.0)  # a 48 h record whose file ends 10 s after the anchor (typical discharge file)


def test_anchor_is_2100_utc():
    assert ANCHOR_UTC == 4102444800.0 and EPOCH_S == 300 and BLACKOUT_S == 1800


def test_case_epochs_cover_0_5_to_24_h_before_the_arrest():
    starts = case_epoch_starts(*FULL)
    assert len(starts) == CASE_EPOCHS == 282
    assert starts[0] == -24 * H and starts[-1] + EPOCH_S == -BLACKOUT_S
    assert np.all(np.diff(starts) == EPOCH_S)  # contiguous: each epoch begins where the previous one ends


def test_case_epochs_clip_to_a_short_record():
    starts = case_epoch_starts(-10 * H, 10.0)
    assert starts[0] == -10 * H and len(starts) == (10 * H - BLACKOUT_S) // EPOCH_S


def test_file_end_never_moves_a_window():
    """Most arrest files run 1 h past the outcome, most other files stop earlier: the rule must ignore the file end."""
    for rec_end in (-BLACKOUT_S, 10.0, H + 0.07):
        assert case_epoch_starts(-48 * H, rec_end) == case_epoch_starts(*FULL)
        assert control_epoch_starts(123456789, -48 * H, rec_end) == control_epoch_starts(123456789, *FULL)


def test_no_window_touches_the_last_half_hour_or_after_the_outcome():
    for case_id in range(100000000, 100000300):
        for starts in (case_epoch_starts(-48 * H, H), control_epoch_starts(case_id, -48 * H, H),
                       control_epoch_starts(case_id, -25 * H, H)):
            assert max(starts) + EPOCH_S <= -BLACKOUT_S


def test_control_block_is_24_h_and_deterministic():
    a = control_epoch_starts(100014944, *FULL)
    assert len(a) == CONTROL_EPOCHS == 288 and np.all(np.diff(a) == EPOCH_S)
    assert a == control_epoch_starts(100014944, *FULL)
    assert a != control_epoch_starts(100014945, *FULL)


def test_control_position_is_drawn_independently_of_the_record():
    """The draw u ~ U(0,1) comes from (seed, stay id) only; across stays the block lands anywhere in the record."""
    ids = np.arange(100000000, 100003000)
    u = np.array([control_block_fraction(int(i)) for i in ids])
    assert abs(u.mean() - 0.5) < 0.02 and abs(np.quantile(u, 0.25) - 0.25) < 0.03 and abs(np.quantile(u, 0.75) - 0.75) < 0.03
    ends_h = np.array([-(control_epoch_starts(int(i), *FULL)[-1] + EPOCH_S) / H for i in ids])
    assert ends_h.min() >= 0.5 and ends_h.max() <= 24.0
    assert np.mean(ends_h < 0.6) < 0.01   # not pinned to the end of the record, unlike the arrest windows
    assert abs(np.median(ends_h) - 12.25) < 0.75


def test_control_in_a_record_shorter_than_24_h_uses_what_there_is():
    starts = control_epoch_starts(100000001, -20 * H, 10.0)
    assert starts[0] == -20 * H and starts[-1] + EPOCH_S <= -BLACKOUT_S
    assert len(starts) == (20 * H - BLACKOUT_S) // EPOCH_S
    assert control_epoch_starts(100000001, -BLACKOUT_S - 100, 10.0) == []


def test_labels():
    assert label_for("Cardiac Arrest") == 1
    assert label_for("Discharge") == 0 and label_for("Death") == 0
    with pytest.raises(ValueError):
        label_for("Transfer")


def test_epoch_starts_dispatches_on_label():
    assert epoch_starts(5, 1, *FULL) == case_epoch_starts(*FULL)
    assert epoch_starts(5, 0, *FULL) == control_epoch_starts(5, *FULL)


def test_instance_ids_are_stable_opaque_and_unique():
    ids = {instance_id(c, s) for c in range(100000000, 100000050) for s in case_epoch_starts(*FULL)}
    assert len(ids) == 50 * 282
    one = instance_id(100014944, -86400)
    assert one == instance_id(100014944, -86400) and one.startswith("scope-") and len(one) == 22
    assert "100014944" not in one and "86400" not in one


def test_registry_spec_matches_the_code_and_the_published_split():
    import hashlib

    import yaml

    from conftest import REGISTRY_DIR
    from medbench.tasks.scope_hrv import FEATURES
    from medbench.tasks.scope_sampling import SEED

    spec = yaml.safe_load((REGISTRY_DIR / "tasks" / "scope-arrest.yaml").read_text(encoding="utf-8"))
    assert spec["features"] == FEATURES and spec["controls"]["seed"] == SEED == spec["split"]["seed"]
    assert spec["blackout_h"] * 3600 == BLACKOUT_S and spec["signal"]["epoch_s"] == EPOCH_S
    path = REGISTRY_DIR.parent / spec["split"]["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == spec["split"]["sha256"]
    split = pd.read_csv(path)
    assert len(split) == 4517 and split["case_id"].is_unique
    assert split["split"].value_counts().to_dict() == {"train": 3162, "test": 903, "validation": 452}


def test_stay_split_is_stratified_deterministic_and_complete():
    from medbench.tasks.scope_arrest import stay_split

    rng = np.random.default_rng(0)
    ids = rng.choice(np.arange(100000000, 999999999), 4517, replace=False)
    outcome = np.array(["Discharge"] * 4095 + ["Death"] * 348 + ["Cardiac Arrest"] * 74)
    meta = pd.DataFrame({"case_id": ids, "outcome": outcome})
    split = stay_split(meta)
    assert split == stay_split(meta.sample(frac=1, random_state=1))  # row order does not matter
    assert set(split) == set(int(i) for i in ids)
    df = meta.assign(split=meta["case_id"].map(split))
    arrests = df[df.outcome == "Cardiac Arrest"]["split"].value_counts().to_dict()
    assert arrests == {"train": 52, "validation": 7, "test": 15}
    share = df["split"].value_counts(normalize=True)
    assert abs(share["train"] - 0.7) < 0.01 and abs(share["test"] - 0.2) < 0.01
