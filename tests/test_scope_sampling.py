"""Window rules of scope-arrest@0, which follow the dataset authors' SCOPE processing script."""
import random

import pytest

from benchtoward.tasks.scope_sampling import (ANCHOR_UTC, CASE_EPOCHS, EPOCH_SAMPLES, HZ, SEED, case_epochs,
                                           control_block, control_epochs, instance_id, label_for, stay_seeds)

DAY = 24 * 3600 * HZ  # samples in 24 h at 250 Hz


def test_anchor_is_2100_utc():
    import datetime

    assert datetime.datetime.fromtimestamp(ANCHOR_UTC, datetime.UTC).isoformat() == "2100-01-01T00:00:00+00:00"


def test_cohort_is_arrest_against_discharge_and_deaths_are_left_out():
    assert label_for("Cardiac Arrest") == 1 and label_for("Discharge") == 0
    assert label_for("Death") is None
    with pytest.raises(ValueError):
        label_for("Transfer")


def test_case_epochs_are_the_282_epochs_from_24_h_to_0_5_h_before_the_arrest():
    epochs = case_epochs(DAY)  # the cropped day before the arrest
    assert len(epochs) == CASE_EPOCHS == 282
    assert epochs[0] == (0, 1435.0) and epochs[-1] == (281 * EPOCH_SAMPLES, 30.0)
    assert all(b[0] - a[0] == EPOCH_SAMPLES for a, b in zip(epochs, epochs[1:]))


def test_case_epochs_tolerate_one_extra_sample_from_rounding():
    epochs = case_epochs(DAY + 1)
    assert [s for s, _ in epochs] == [s for s, _ in case_epochs(DAY)]
    assert epochs[-1][1] == pytest.approx(30.0 + 1 / (HZ * 60))


def test_control_block_is_the_authors_random_draw():
    total = DAY + 3_000_000
    random.seed(SEED + 7)  # what the authors' worker does before drawing
    expected = random.randint(0, total - DAY)
    assert control_block(total, SEED + 7) == (expected, expected + DAY)
    assert control_block(total, SEED + 7) == control_block(total, SEED + 7)
    assert control_block(total, SEED + 8) != control_block(total, SEED + 7)


def test_control_shorter_than_24_h_uses_the_whole_record():
    assert control_block(DAY - 5, SEED) == (0, DAY - 5)
    assert control_block(DAY, SEED) == (0, DAY)


def test_control_epochs_tile_the_block_from_its_start():
    epochs = control_epochs(DAY)
    assert len(epochs) == 288 and epochs[0] == (0, 1435.0) and epochs[-1] == (287 * EPOCH_SAMPLES, 0.0)
    short = control_epochs(3 * EPOCH_SAMPLES + 100)
    assert [s for s, _ in short] == [0, EPOCH_SAMPLES, 2 * EPOCH_SAMPLES]
    assert control_epochs(EPOCH_SAMPLES - 1) == []


def test_seeds_are_42_plus_the_rank_within_each_outcome_in_metadata_order():
    ids = [11, 12, 13, 14, 15, 16]
    outcomes = ["Discharge", "Cardiac Arrest", "Death", "Discharge", "Cardiac Arrest", "Discharge"]
    assert stay_seeds(ids, outcomes) == {11: 42, 14: 43, 16: 44, 12: 42, 15: 43}


def test_instance_ids_are_stable_opaque_and_unique():
    ids = {instance_id(c, t) for c in range(100000000, 100000050) for t in range(-DAY, 0, EPOCH_SAMPLES)}
    assert len(ids) == 50 * 288
    one = instance_id(100014944, -DAY)
    assert one == instance_id(100014944, -DAY) and one.startswith("scope-") and len(one) == 22
    assert "100014944" not in one and str(DAY) not in one


def test_registry_spec_matches_the_code_and_the_published_split():
    import hashlib

    import pandas as pd
    import yaml

    from conftest import REGISTRY_DIR
    from benchtoward.baselines.scope_arrest import PARAM_BOUNDS
    from benchtoward.tasks.scope_arrest import SPLIT_SEED, TEST_SIZE
    from benchtoward.tasks.scope_hrv import FEATURES, FLAT_S, QUALITY_RATIO

    spec = yaml.safe_load((REGISTRY_DIR / "tasks" / "scope-arrest.yaml").read_text(encoding="utf-8"))
    assert spec["protocol"]["name"] == "BenchToward definition" and spec["features"] == FEATURES
    assert spec["controls"]["seed"] == SEED == spec["split"]["seed"] == SPLIT_SEED
    assert spec["split"]["fractions"] == [1 - TEST_SIZE, TEST_SIZE] and spec["epoch_rules"]["flat_s"] == FLAT_S
    assert "80%" in spec["epoch_rules"]["quality"] and QUALITY_RATIO == 0.8
    assert spec["cohort"] == {"label_1": "Cardiac Arrest", "label_0": "Discharge", "excluded": ["Death"]}
    assert spec["signal"]["hz"] == HZ and spec["signal"]["epoch_s"] * HZ == EPOCH_SAMPLES
    lgbm = " ".join(spec["baselines"]["benchtoward/scope-hrv-lgbm"].split())
    assert all(f"{name} {lo}-{hi}" in lgbm for name, (lo, hi) in PARAM_BOUNDS.items())
    path = REGISTRY_DIR.parent / spec["split"]["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == spec["split"]["sha256"]
    split = pd.read_csv(path)
    assert split["case_id"].is_unique and set(split["split"]) == {"train", "test"}
    assert abs((split["split"] == "test").mean() - TEST_SIZE) < 0.005
