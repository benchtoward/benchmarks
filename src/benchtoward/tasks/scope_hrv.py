"""One 5-min, 250 Hz ECG epoch of scope-arrest@0, processed as in the dataset authors' SCOPE script (Lee et al. 2023
setup): screening for long missing or flat stretches, a 0.5 Hz Butterworth high-pass (order 5) and a 60 Hz powerline
filter, NeuroKit2 R peaks with artifact correction, the averageQRS quality rule, then NeuroKit2's HRV measures.

NeuroKit2 is imported only when an epoch is processed (it is installed in the compute pods, not in the project,
because it pins pandas<3). Everything else is numpy, so the same file runs in the pods and on a laptop.
"""
from dataclasses import dataclass, field

import numpy as np

HZ = 250
FLAT_S = 10  # a missing or constant stretch longer than this drops the epoch
QUALITY_THRESHOLD = 0.9  # a sample is good if its averageQRS quality index is >= 0.9
QUALITY_RATIO = 0.8  # the epoch is kept if at least 80% of its samples are good

# The 44 HRV measures the authors' script extracts (its order), and the 33 their model uses (the order of their
# training script and of the released model file).
CANDIDATES = ["MeanNN", "SDNN", "SDANN1", "SDNNI1", "RMSSD", "SDSD", "CVNN", "CVSD", "MedianNN", "MadNN", "MCVNN",
              "IQRNN", "Prc20NN", "Prc80NN", "pNN50", "pNN20", "MinNN", "MaxNN", "HTI", "TINN",
              "ULF", "VLF", "LF", "HF", "VHF", "LFHF", "LFn", "HFn", "LnHF",
              "SD1", "SD2", "SD1SD2", "S", "CSI", "CVI", "CSI_Modified", "PIP", "IALS", "PSS", "PAS",
              "GI", "SI", "AI", "PI"]
FEATURES = ["IALS", "pNN50", "TINN", "HTI", "PI", "Prc20NN", "MedianNN", "CVSD", "PIP", "MCVNN", "LFn", "CSI",
            "pNN20", "MinNN", "MeanNN", "SDNNI1", "MaxNN", "HFn", "LF", "LFHF", "PAS", "PSS", "SDANN1", "VHF",
            "RMSSD", "VLF", "IQRNN", "CVNN", "CSI_Modified", "SD2", "Prc80NN", "SD1SD2", "SDNN"]


@dataclass
class EpochResult:
    status: str  # ok | flat | quality | hrv_error
    quality_ratio: float = float("nan")
    n_beats: int = 0
    nan_fraction: float = 0.0
    features: dict[str, float] = field(default_factory=dict)
    ecg: np.ndarray | None = None  # the raw 250 Hz epoch (float16), only when status == "ok"


def _longest_run(starts_of_runs: np.ndarray, n: int) -> int:
    bounds = np.r_[0, starts_of_runs, n]
    return int(np.diff(bounds).max()) if n else 0


def screen_epoch(x: np.ndarray, fs: int = HZ) -> str | None:
    """'flat' if more than 10 s in a row are missing or constant, else None."""
    x = np.asarray(x, dtype=float)
    limit = fs * FLAT_S
    missing = np.isnan(x)
    if missing.any():
        edges = np.diff(np.r_[0, missing.astype(np.int8), 0])
        if (np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).max() > limit:
            return "flat"
    # runs of equal consecutive values; a missing sample never equals its neighbour
    if _longest_run(np.flatnonzero(np.diff(x) != 0) + 1, len(x)) > limit:
        return "flat"
    return None


def quality_ratio(quality_index: np.ndarray) -> float:
    q = np.asarray(quality_index, dtype=float)
    return float((q >= QUALITY_THRESHOLD).sum() / len(q))


def keep_quality(ratio: float) -> bool:
    return ratio >= QUALITY_RATIO


def process_epoch(segment: np.ndarray, nk=None, fs: int = HZ) -> EpochResult:
    x = np.asarray(segment).squeeze()
    nan_fraction = float(np.isnan(x).mean()) if len(x) else 1.0
    if screen_epoch(x, fs):
        return EpochResult("flat", nan_fraction=nan_fraction)
    if nk is None:
        import neurokit2 as nk
    try:
        cleaned = nk.signal.signal_filter(signal=x, sampling_rate=fs, lowcut=0.5, method="butterworth", order=5)
        cleaned = nk.signal.signal_filter(signal=cleaned, sampling_rate=fs, method="powerline", powerline=60)
        _, peaks = nk.ecg_peaks(cleaned, sampling_rate=fs, method="neurokit", correct_artifacts=True)
        rpeaks = peaks["ECG_R_Peaks"]
        quality = nk.ecg_quality(cleaned, rpeaks=rpeaks, sampling_rate=fs, method="averageQRS")
        ratio = quality_ratio(quality)
        if not keep_quality(ratio):
            return EpochResult("quality", ratio, len(rpeaks), nan_fraction)
        hrv = nk.hrv(rpeaks, sampling_rate=fs, show=False)
        if hrv.empty:
            return EpochResult("hrv_error", ratio, len(rpeaks), nan_fraction)
        features = {name: float(hrv[f"HRV_{name}"].iloc[0]) if f"HRV_{name}" in hrv.columns else np.nan
                    for name in CANDIDATES}
    except Exception:  # as the authors: an epoch that fails anywhere in the pipeline is dropped
        return EpochResult("hrv_error", nan_fraction=nan_fraction)
    return EpochResult("ok", ratio, len(rpeaks), nan_fraction, features, x.astype(np.float16))
