"""HRV measures for scope-arrest@0 on synthetic RR series, and the epoch screening rules."""
import numpy as np
import pytest

from medbench.tasks.scope_hrv import (CANDIDATES, FEATURES, check_epoch, hrv_from_rri, peaks_to_rri)

FS = 250


def peaks_from_rri(rri_ms) -> np.ndarray:
    return np.concatenate([[100], 100 + np.cumsum(np.asarray(rri_ms) * FS / 1000.0)]).round().astype(int)


def test_feature_lists_match_the_paper():
    assert len(FEATURES) == 33 and len(CANDIDATES) == 43 and set(FEATURES) < set(CANDIDATES)
    assert FEATURES[:6] == ["IALS", "pNN50", "TINN", "HTI", "PI", "Prc20NN"]  # BorutaShap order (Suppl. Fig. 1)
    assert {"SDNN", "SD1", "SD2", "HF", "LnHF", "CVI", "S", "GI", "SDSD", "MadNN"} == set(CANDIDATES) - set(FEATURES)


def test_peaks_to_rri_in_ms():
    assert peaks_to_rri(np.array([0, 250, 450]), 250).tolist() == [1000.0, 800.0]


def test_constant_rhythm():
    f = hrv_from_rri(np.full(300, 1000.0))
    assert f["MeanNN"] == 1000 and f["MedianNN"] == 1000 and f["MinNN"] == f["MaxNN"] == 1000
    assert f["SDNN"] == 0 and f["RMSSD"] == 0 and f["pNN50"] == 0 and f["pNN20"] == 0 and f["HTI"] == 1


def test_alternating_rhythm():
    rri = np.tile([800.0, 1000.0], 150)
    f = hrv_from_rri(rri)
    assert f["MeanNN"] == 900 and f["RMSSD"] == pytest.approx(200)
    assert f["pNN50"] == pytest.approx(299 / 300 * 100)
    assert f["PIP"] == pytest.approx(298 / 300) and f["IALS"] == 1 and f["PSS"] == 1 and f["PAS"] == 1
    assert f["PI"] == pytest.approx(50, abs=0.5)
    assert f["SD1"] == pytest.approx(200 / np.sqrt(2), rel=0.01)


def test_respiratory_modulation_shows_in_hf_and_slow_modulation_in_lf():
    t = np.arange(0, 300, 1.0)
    fast = hrv_from_rri(1000 + 40 * np.sin(2 * np.pi * 0.25 * t))
    slow = hrv_from_rri(1000 + 40 * np.sin(2 * np.pi * 0.08 * t))
    assert fast["HFn"] > 0.8 and fast["LFn"] < 0.1
    assert slow["LFn"] > 0.8 and slow["LFHF"] > 10


def test_realistic_series_gives_all_33_features():
    rng = np.random.default_rng(0)
    rri = 850 + np.cumsum(rng.normal(0, 8, 350)).clip(-150, 150) + rng.normal(0, 15, 350)
    f = hrv_from_rri(rri)
    assert set(f) == set(CANDIDATES)
    assert all(np.isfinite(f[name]) for name in FEATURES)
    assert f["SDANN1"] > 0 and f["SDNNI1"] > 0 and f["TINN"] >= 0 and 0 < f["LFn"] < 1


def test_screening_drops_long_gaps_and_much_missing_data():
    x = np.sin(np.linspace(0, 600, 150_000))
    status, filled, nan_frac = check_epoch(x)
    assert status is None and nan_frac == 0 and filled.shape == (150_000,)
    short = x.copy()
    short[[10, 5000, 90000]] = np.nan  # single dropped samples are interpolated
    status, filled, _ = check_epoch(short)
    assert status is None and np.isfinite(filled).all()
    gap = x.copy()
    gap[1000:1060] = np.nan  # 0.12 s at 500 Hz
    assert check_epoch(gap)[0] == "gap"
    sparse = x.copy()
    sparse[::60] = np.nan  # 1.7% missing, no long gap
    assert check_epoch(sparse)[0] == "missing"
    assert check_epoch(np.zeros(150_000))[0] == "flat"
    assert check_epoch(np.full(150_000, np.nan))[0] == "missing"


def _synthetic_ecg(rri_ms, fs=500, seconds=300):
    t = np.arange(int(seconds * fs)) / fs
    x = 0.05 * np.sin(2 * np.pi * 0.3 * t)
    beat = 0.2
    for r in np.asarray(rri_ms) / 1000.0:
        beat += r
        if beat >= seconds:
            break
        x += 1.2 * np.exp(-0.5 * ((t - beat) / 0.012) ** 2) - 0.25 * np.exp(-0.5 * ((t - beat - 0.03) / 0.015) ** 2)
        x += 0.25 * np.exp(-0.5 * ((t - beat - 0.25) / 0.04) ** 2)
    return x


def test_process_epoch_on_a_synthetic_ecg():
    pytest.importorskip("neurokit2")  # the pod pipeline; neurokit2 pins pandas<3, so it is not a project dependency
    from medbench.tasks.scope_hrv import process_epoch

    rng = np.random.default_rng(1)
    x = _synthetic_ecg(800 + rng.normal(0, 20, 400))
    x[60_000:60_150] += np.hanning(150) * 1.5  # one distorted beat: averageQRS rescales distances to [0, 1]
    result = process_epoch(x)
    assert result.status == "ok" and result.ecg250.shape == (75_000,) and result.ecg250.dtype == np.float16
    assert abs(result.features["MeanNN"] - 800) < 15 and result.n_beats > 350 and result.good_fraction > 0.8
    assert process_epoch(np.zeros(150_000)).status == "flat"


def test_matches_neurokit2_formulas():
    nk = pytest.importorskip("neurokit2")
    from neurokit2.hrv.hrv_nonlinear import (_hrv_nonlinear_fragmentation, _hrv_nonlinear_poincare,
                                             _hrv_nonlinear_poincare_hra)

    rng = np.random.default_rng(7)
    for trial in range(5):
        rri = 750 + np.cumsum(rng.normal(0, 6, 380)).clip(-120, 120) + rng.normal(0, 25, 380)
        if trial == 4:
            rri[50:53] = [400, 1300, 600]  # ectopic beats
        peaks = peaks_from_rri(rri)
        ours = hrv_from_rri(peaks_to_rri(peaks, FS))
        ref = {}
        ref.update(nk.hrv_time(peaks, sampling_rate=FS).iloc[0].to_dict())
        ref.update(nk.hrv_frequency(peaks, sampling_rate=FS).iloc[0].to_dict())
        ref = {k.removeprefix("HRV_"): v for k, v in ref.items()}
        r = peaks_to_rri(peaks, FS)
        for fn in (_hrv_nonlinear_poincare, _hrv_nonlinear_poincare_hra, _hrv_nonlinear_fragmentation):
            ref.update(fn(r, out={}))
        for name in CANDIDATES:
            assert ours[name] == pytest.approx(ref[name], rel=1e-6, abs=1e-9, nan_ok=True), name
