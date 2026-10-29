"""HRV measures of a 5-min ECG epoch for scope-arrest@0, after Lee et al. 2023 (npj Digit Med 6:215).

The paper computed 43 candidate HRV measures with NeuroKit2 and kept 33 by BorutaShap (Supplementary Fig. 1). The
measures below are ports of NeuroKit2 0.2.13's formulas (hrv_time, hrv_frequency with its defaults, and the Poincare,
asymmetry and fragmentation parts of hrv_nonlinear), so they run without NeuroKit2 (which pins pandas<3) and without
its slow entropy measures; tests compare them with NeuroKit2 when it is installed.

`process_epoch` is the pod pipeline: screening for missing data, 2:1 decimation to 250 Hz, then NeuroKit2's cleaning,
R-peak detection and averageQRS quality index (imported lazily; installed only in the pod), then the measures here.
Pure numpy/scipy otherwise, so the same file runs in compute pods.
"""
import itertools
from dataclasses import dataclass, field

import numpy as np
import scipy.interpolate
import scipy.signal
import scipy.stats

# The 33 measures BorutaShap accepted, in the paper's order of importance (Supplementary Fig. 1).
FEATURES = ["IALS", "pNN50", "TINN", "HTI", "PI", "Prc20NN", "pNN20", "CVNN", "MedianNN", "MCVNN", "SDNNI1",
            "MinNN", "CVSD", "VHF", "RMSSD", "PSS", "LFn", "CSI", "MeanNN", "PIP", "IQRNN", "HFn", "SD1SD2", "LF",
            "SI", "MaxNN", "VLF", "SDANN1", "Prc80NN", "PAS", "CSI_Modified", "LFHF", "AI"]
REJECTED = ["S", "GI", "SDNN", "HF", "SD2", "SDSD", "MadNN", "SD1", "LnHF", "CVI"]
CANDIDATES = FEATURES + REJECTED

SOURCE_HZ = 500
HZ = 250
EPOCH_SAMPLES = 300 * SOURCE_HZ
MAX_MISSING_FRACTION = 0.01
MAX_GAP_S = 0.1
MIN_BEATS = 30
QUALITY_BEAT = 0.9  # a beat is good if its averageQRS quality index is >= 0.9
MIN_GOOD_FRACTION = 0.8  # the epoch is kept only if more than 80% of its beats are good (paper)
FLAT_STD_MV = 1e-3

BANDS = {"ULF": (0, 0.0033), "VLF": (0.0033, 0.04), "LF": (0.04, 0.15), "HF": (0.15, 0.4), "VHF": (0.4, 0.5)}


def peaks_to_rri(peaks: np.ndarray, fs: int = HZ) -> np.ndarray:
    return np.diff(np.asarray(peaks, dtype=float)) / fs * 1000.0


# ---- time domain (neurokit2 hrv_time) -----------------------------------------------------------------------

def _mad(x: np.ndarray) -> float:
    return float(np.nanmedian(np.abs(x - np.nanmedian(x))) * 1.4826)


def _windows(rri: np.ndarray, minutes: int = 1):
    size = minutes * 60 * 1000
    time = np.nancumsum(rri / 1000)
    cum = (time - time[0]) * 1000 + rri[0]
    n = int(np.round(cum[-1] / size))
    if n < 3:
        return None
    out = []
    for i in range(n):
        start = i * size
        out.append(rri[np.where(cum >= start)[0][0]:np.where(cum < start + size)[0][-1]])
    return out


def _tinn(rri: np.ndarray, bar_x: np.ndarray, bar_y: np.ndarray, binsize: float) -> float:
    """neurokit2's TINN search, kept as is (its inner bound m is not reset per n, so only the first n is tried)."""
    min_error = 2 ** 14
    X, Y = bar_x[np.argmax(bar_y)], np.max(bar_y)
    above = np.where(bar_x - np.min(rri) > 0)[0]
    if len(above) == 0:
        return np.nan
    n, m, N, M = bar_x[above[0]], X + binsize, 0, 0
    while n < X:
        while m < np.max(rri):
            n_start, n_end = np.where(bar_x == n)[0][0], np.where(bar_x == X)[0][0]
            qn = np.polyval(np.polyfit([n, X], [0, Y], deg=1), bar_x[n_start:n_end + 1])
            m_start, m_end = n_end, np.where(bar_x == m)[0][0]
            qm = np.polyval(np.polyfit([X, m], [Y, 0], deg=1), bar_x[m_start:m_end + 1])
            q = np.zeros(len(bar_x))
            q[n_start:n_end + 1] = qn
            q[m_start:m_end + 1] = qm
            error = np.sum((bar_y[n_start:m_end + 1] - q[n_start:m_end + 1]) ** 2)
            if error < min_error:
                N, M, min_error = n, m, error
            m += binsize
        n += binsize
    return float(M - N)


def hrv_time(rri: np.ndarray) -> dict[str, float]:
    diff = np.diff(rri)
    out = {"MeanNN": np.nanmean(rri), "SDNN": np.nanstd(rri, ddof=1)}
    windows = _windows(rri)
    out["SDANN1"] = np.nan if windows is None else np.nanstd([np.nanmean(w) for w in windows], ddof=1)
    out["SDNNI1"] = np.nan if windows is None else np.nanmean([np.nanstd(w, ddof=1) for w in windows])
    out["RMSSD"] = np.sqrt(np.nanmean(diff ** 2))
    out["SDSD"] = np.nanstd(diff, ddof=1)
    out["CVNN"] = out["SDNN"] / out["MeanNN"]
    out["CVSD"] = out["RMSSD"] / out["MeanNN"]
    out["MedianNN"] = np.nanmedian(rri)
    out["MadNN"] = _mad(rri)
    out["MCVNN"] = out["MadNN"] / out["MedianNN"]
    out["IQRNN"] = scipy.stats.iqr(rri)
    out["Prc20NN"] = np.nanpercentile(rri, q=20)
    out["Prc80NN"] = np.nanpercentile(rri, q=80)
    out["pNN50"] = np.sum(np.abs(diff) > 50) / (len(diff) + 1) * 100
    out["pNN20"] = np.sum(np.abs(diff) > 20) / (len(diff) + 1) * 100
    out["MinNN"] = np.nanmin(rri)
    out["MaxNN"] = np.nanmax(rri)
    binsize = 1000 / 128
    bar_y, bar_x = np.histogram(rri, bins=np.arange(0, np.max(rri) + binsize, binsize))
    out["HTI"] = len(rri) / np.max(bar_y)
    out["TINN"] = _tinn(rri, bar_x, bar_y, binsize)
    return {k: float(v) for k, v in out.items()}


# ---- frequency domain (neurokit2 hrv_frequency defaults: quadratic interpolation at 100 Hz, Welch, normalized) --

def hrv_frequency(rri: np.ndarray, interpolation_rate: int = 100) -> dict[str, float]:
    time = np.nancumsum(rri / 1000)
    x_new = np.arange(time[0], time[-1] + 1 / interpolation_rate, 1 / interpolation_rate)
    signal = scipy.interpolate.interp1d(time, rri, kind="quadratic", bounds_error=False,
                                        fill_value=([rri[0]], [rri[-1]]))(x_new)
    signal = signal - np.mean(signal)
    n = len(signal)
    min_frequency = (2 * interpolation_rate) / (n / 2)
    nperseg = int((2 / min_frequency) * interpolation_rate)
    if nperseg > n / 2:
        nperseg = int(n / 2)
    frequency, power = scipy.signal.welch(signal, fs=interpolation_rate, scaling="density", detrend=False,
                                          nfft=int(nperseg * 2), average="mean", nperseg=nperseg, window="hann")
    power = power / np.max(power)
    keep = (frequency >= min_frequency) & (frequency <= 0.5)
    frequency, power = frequency[keep], power[keep]
    out = {}
    for name, (lo, hi) in BANDS.items():
        where = (frequency >= lo) & (frequency < hi)
        value = np.trapezoid(y=power[where], x=frequency[where])
        out[name] = np.nan if value == 0.0 else float(value)
    total = np.nansum(list(out.values()))
    with np.errstate(divide="ignore", invalid="ignore"):
        out["LFHF"] = out["LF"] / out["HF"]
        out["LFn"] = out["LF"] / total
        out["HFn"] = out["HF"] / total
        out["LnHF"] = float(np.log(out["HF"]))
    return {k: float(v) for k, v in out.items() if k != "ULF"}


# ---- nonlinear: Poincare, heart rate asymmetry, fragmentation (neurokit2 hrv_nonlinear parts) ----------------

def hrv_poincare(rri: np.ndarray) -> dict[str, float]:
    x1 = (rri[:-1] - rri[1:]) / np.sqrt(2)
    x2 = (rri[:-1] + rri[1:]) / np.sqrt(2)
    sd1, sd2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
    T, L = 4 * sd1, 4 * sd2
    return {"SD1": sd1, "SD2": sd2, "SD1SD2": sd1 / sd2, "S": np.pi * sd1 * sd2, "CSI": L / T,
            "CVI": np.log10(L * T), "CSI_Modified": L ** 2 / T}


def hrv_asymmetry(rri: np.ndarray) -> dict[str, float]:
    N = len(rri) - 1
    x, y = rri[:-1], rri[1:]
    diff = y - x
    dec, acc, same = np.where(diff > 0)[0], np.where(diff < 0)[0], np.where(diff == 0)[0]
    dist = abs(y - x) / np.sqrt(2)
    theta = abs(np.arctan(1) - np.arctan(y / x))
    area = 0.5 * theta * (x ** 2 + y ** 2)
    off_line = N - len(same)  # points not on the line of identity
    return {"GI": np.sum(dist[dec]) / np.sum(dist) * 100, "SI": np.sum(theta[dec]) / np.sum(theta) * 100,
            "AI": np.sum(area[dec]) / np.sum(area) * 100, "PI": len(acc) / off_line * 100 if off_line else np.nan}


def _consecutive(x) -> list[tuple]:
    return [tuple(g) for _, g in itertools.groupby(x, lambda n, c=itertools.count(): n - next(c))]


def hrv_fragmentation(rri: np.ndarray) -> dict[str, float]:
    diff = np.diff(rri)
    crossings = np.nonzero(np.abs(np.diff(np.sign(diff))) > 0)[0]
    lengths = [len(g) for g in _consecutive(np.where(diff > 0)[0]) + _consecutive(np.where(diff < 0)[0])]
    alternations = [len(g) for g in _consecutive(crossings)]
    return {"PIP": len(crossings) / (len(diff) + 1), "IALS": 1 / np.average(lengths),
            "PSS": np.sum(np.asarray(lengths) < 3) / len(lengths),
            "PAS": np.sum(np.asarray(alternations) >= 4) / len(alternations) if alternations else np.nan}


def hrv_from_rri(rri_ms) -> dict[str, float]:
    """All 43 candidate measures (the paper's 33 are FEATURES) from RR intervals in ms."""
    rri = np.asarray(rri_ms, dtype=float)
    out: dict[str, float] = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        out.update(hrv_time(rri))
        out.update(hrv_frequency(rri))
        out.update(hrv_poincare(rri))
        out.update(hrv_asymmetry(rri))
        out.update(hrv_fragmentation(rri))
    return {name: float(out[name]) for name in CANDIDATES}


# ---- one epoch, as in the pod ---------------------------------------------------------------------------------

@dataclass
class EpochResult:
    status: str  # ok | missing | gap | flat | few_beats | quality | error
    nan_fraction: float = 0.0
    n_beats: int = 0
    good_fraction: float = float("nan")
    features: dict[str, float] = field(default_factory=dict)
    ecg250: np.ndarray | None = None  # raw 250 Hz epoch (float16), only when status == "ok"


def check_epoch(x500: np.ndarray) -> tuple[str | None, np.ndarray | None, float]:
    """Screen a 300 s, 500 Hz epoch: too much missing data, a gap over 0.1 s, or no signal. Short gaps are filled
    linearly. Returns (reason or None, filled signal, missing fraction)."""
    x = np.asarray(x500, dtype=float)
    missing = np.isnan(x)
    nan_fraction = float(missing.mean())
    if nan_fraction > MAX_MISSING_FRACTION or missing.all():
        return "missing", None, nan_fraction
    if missing.any():
        edges = np.diff(np.r_[0, missing.astype(np.int8), 0])
        runs = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
        if runs.max() > MAX_GAP_S * SOURCE_HZ:
            return "gap", None, nan_fraction
        idx = np.arange(len(x))
        x = x.copy()
        x[missing] = np.interp(idx[missing], idx[~missing], x[~missing])
    if np.std(x) < FLAT_STD_MV:
        return "flat", None, nan_fraction
    return None, x, nan_fraction


def downsample(x500: np.ndarray) -> np.ndarray:
    """500 Hz to 250 Hz by polyphase decimation (anti-aliasing FIR), as the paper's 250 Hz input."""
    return scipy.signal.resample_poly(np.asarray(x500, dtype=float), 1, 2)


def process_epoch(x500: np.ndarray) -> EpochResult:
    import neurokit2 as nk

    reason, x, nan_fraction = check_epoch(x500)
    if reason:
        return EpochResult(reason, nan_fraction)
    ecg = downsample(x)
    try:
        cleaned = nk.ecg_clean(ecg, sampling_rate=HZ, method="neurokit")
        peaks = np.asarray(nk.ecg_peaks(cleaned, sampling_rate=HZ, method="neurokit")[1]["ECG_R_Peaks"])
        if len(peaks) < MIN_BEATS:
            return EpochResult("few_beats", nan_fraction, len(peaks))
        quality = nk.ecg_quality(cleaned, rpeaks=peaks, sampling_rate=HZ, method="averageQRS")
        good = float(np.mean(quality[peaks] >= QUALITY_BEAT))
        if not good > MIN_GOOD_FRACTION:
            return EpochResult("quality", nan_fraction, len(peaks), good)
        features = hrv_from_rri(peaks_to_rri(peaks, HZ))
    except Exception:  # a broken epoch must not stop the stay
        return EpochResult("error", nan_fraction)
    return EpochResult("ok", nan_fraction, len(peaks), good, features, ecg.astype(np.float16))
