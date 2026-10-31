"""The per-epoch pipeline of scope-arrest@0 (the dataset authors' screening, filtering, R peaks, quality rule and
HRV measures). NeuroKit2 pins pandas<3, so it is not a project dependency: a fake module checks the exact calls,
and the real one runs when it is installed."""
import types

import numpy as np
import pandas as pd
import pytest

from benchtoward.tasks.scope_hrv import (CANDIDATES, FEATURES, QUALITY_RATIO, keep_quality, process_epoch,
                                      quality_ratio, screen_epoch)

FS = 250


def noisy(n=75_000, seed=0):
    return np.random.default_rng(seed).normal(0, 0.1, n).astype(np.float32)


def test_feature_lists_are_the_authors():
    assert len(CANDIDATES) == 44 and len(FEATURES) == 33 and set(FEATURES) < set(CANDIDATES)
    # the order of the authors' training script and of the released model file
    assert FEATURES[:6] == ["IALS", "pNN50", "TINN", "HTI", "PI", "Prc20NN"] and FEATURES[-1] == "SDNN"
    assert {"SD2", "SDNN"} < set(FEATURES) and not {"SI", "AI", "GI", "ULF", "LnHF"} & set(FEATURES)


def test_screening_drops_more_than_10_s_of_missing_or_flat_signal():
    x = noisy()
    assert screen_epoch(x) is None
    gap = x.copy()
    gap[1000:1000 + 10 * FS] = np.nan  # exactly 10 s is allowed
    assert screen_epoch(gap) is None
    gap[1000:1001 + 10 * FS] = np.nan
    assert screen_epoch(gap) == "flat"
    flat = x.copy()
    flat[5000:5001 + 10 * FS] = 0.25
    assert screen_epoch(flat) == "flat"
    flat[5000 + 10 * FS] = 0.3  # a 10 s plateau is allowed
    assert screen_epoch(flat) is None


def test_quality_rule_counts_samples_and_keeps_80_percent():
    q = np.r_[np.full(80, 0.95), np.full(20, 0.5)]
    assert quality_ratio(q) == pytest.approx(0.8) and QUALITY_RATIO == 0.8 and keep_quality(quality_ratio(q))
    assert not keep_quality(quality_ratio(np.r_[np.full(79, 0.9), np.full(21, 0.89)]))


class FakeNk:
    """Records the calls and returns a fixed rhythm."""

    def __init__(self, quality=0.95, hrv=None, fail=None):
        self.calls, self.quality, self.fail = [], quality, fail
        self.hrv_frame = hrv if hrv is not None else pd.DataFrame(
            {f"HRV_{name}": [float(k)] for k, name in enumerate(CANDIDATES) if name != "ULF"})
        self.signal = types.SimpleNamespace(signal_filter=self.signal_filter)

    def signal_filter(self, signal, **kw):
        self.calls.append(("signal_filter", kw))
        return np.asarray(signal, dtype=float)

    def ecg_peaks(self, cleaned, **kw):
        self.calls.append(("ecg_peaks", kw))
        if self.fail == "peaks":
            raise ValueError("no peaks")
        return None, {"ECG_R_Peaks": np.arange(100, 75_000, 250)}

    def ecg_quality(self, cleaned, **kw):
        self.calls.append(("ecg_quality", {k: v for k, v in kw.items() if k != "rpeaks"}))
        return np.full(len(cleaned), self.quality)

    def hrv(self, peaks, **kw):
        self.calls.append(("hrv", kw))
        return self.hrv_frame


def test_process_epoch_makes_the_authors_calls_in_order():
    nk = FakeNk()
    result = process_epoch(noisy(), nk=nk)
    assert result.status == "ok" and result.quality_ratio == 1.0 and result.n_beats == 300
    assert nk.calls == [
        ("signal_filter", {"sampling_rate": 250, "lowcut": 0.5, "method": "butterworth", "order": 5}),
        ("signal_filter", {"sampling_rate": 250, "method": "powerline", "powerline": 60}),
        ("ecg_peaks", {"sampling_rate": 250, "method": "neurokit", "correct_artifacts": True}),
        ("ecg_quality", {"sampling_rate": 250, "method": "averageQRS"}),
        ("hrv", {"sampling_rate": 250, "show": False}),
    ]
    assert result.features["IALS"] == float(CANDIDATES.index("IALS")) and np.isnan(result.features["ULF"])
    assert result.ecg.dtype == np.float16 and len(result.ecg) == 75_000


def test_process_epoch_statuses():
    assert process_epoch(np.full(75_000, 0.1, dtype=np.float32), nk=FakeNk()).status == "flat"
    low = process_epoch(noisy(), nk=FakeNk(quality=0.5))
    assert low.status == "quality" and low.quality_ratio == 0.0 and low.ecg is None
    assert process_epoch(noisy(), nk=FakeNk(fail="peaks")).status == "hrv_error"
    assert process_epoch(noisy(), nk=FakeNk(hrv=pd.DataFrame())).status == "hrv_error"


def test_process_epoch_with_neurokit2():
    nk = pytest.importorskip("neurokit2")
    ecg = nk.ecg_simulate(duration=300, sampling_rate=FS, heart_rate=75, noise=0.02, random_state=1)
    result = process_epoch(np.asarray(ecg, dtype=np.float32))
    assert result.status == "ok" and 350 < result.n_beats < 400 and result.quality_ratio >= 0.8
    assert 780 < result.features["MeanNN"] < 820 and all(name in result.features for name in CANDIDATES)
