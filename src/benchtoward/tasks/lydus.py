"""LYDUS-SNUH-ECG-50K 1.1.1 (KHDP, doi:10.23421/2025.0704, CC BY 4.0): download, metadata and signals.

The shared LYDUS loader for every BenchToward task on this dataset (lydus-ecg12-dx@0 first). The release holds one
metadata CSV (one row per ECG: `new_id`, the reviewed rhythm, premature-beat, bundle-branch and AV-block columns, the
machine interpretation `dx`, age, sex and the machine's measurements) and the ECGs as NPZ files at 100 and 500 Hz.
The 500 Hz file stores 8 independent leads (I, II, V1-V6) as int16, 5,000 samples each; the other four limb leads
are derived (`to_12_leads`), as 12-lead electrocardiographs do. The dataset page says 12 channels; the file holds 8.
The NPZ is stored uncompressed, so `open_signals` memory-maps it instead of loading 4 GB.
"""
import hashlib
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_SLUG = "lydus-snuh-ecg-50k"
KHDP_CODE = "LYDUS-SNUH-ECG-50K"
VERSION = "1.1.1"
DOI = "10.23421/2025.0704"
METADATA_FILE = "LYDUS_SNUH_ECG_ANONYMIZED_50K_0512.csv"
SIGNAL_FILE = "LYDUS_SNUH_ECG_ANONYMIZED_50k_500hz_1219.npz"
# Measured on the download of 2026-10-30 (KHDP gives sizes, not checksums).
SHA256 = {
    METADATA_FILE: "841fd752cdb7420d6f1e70e39c364dd21fc89120d2506421f0a40cd896ad5045",
    SIGNAL_FILE: "c53419da2e875da1db145f975915954eff9f05778acae2945ae25373b9b47154",
}
HZ = 500
SAMPLES = 5000
STORED_LEADS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
LEADS12 = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
# PhysioNet/CinC Challenge 2021 lead sets (Reyna et al. 2021).
LEAD_SETS = {
    "12": LEADS12,
    "6": ("I", "II", "III", "aVR", "aVL", "aVF"),
    "4": ("I", "II", "III", "V2"),
    "3": ("I", "II", "V2"),
    "2": ("I", "II"),
}
MEASUREMENTS = ("vrate", "arate", "pri", "qrsd", "qti", "qtc", "paxis", "raxis", "taxis")


def dataset_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "datasets" / DATASET_SLUG / VERSION


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def fetch(client, data_dir: Path) -> list[Path]:
    """Download the metadata CSV and the 500 Hz NPZ from KHDP (`client`: benchtoward.khdp.KhdpClient) and verify their
    sha256. Files already present with the right checksum are kept."""
    root = dataset_dir(data_dir)
    wanted = {f.key.split("/")[-1]: f for f in client.files(KHDP_CODE, VERSION)}
    missing = set(SHA256) - set(wanted)
    if missing:
        raise ValueError(f"KHDP {KHDP_CODE} {VERSION} lacks {sorted(missing)}")
    paths = []
    for name, expected in SHA256.items():
        dest = root / name
        if not (dest.exists() and sha256_file(dest) == expected):
            client.download(wanted[name], dest)
            if sha256_file(dest) != expected:
                bad = dest.with_name(dest.name + ".bad")
                dest.replace(bad)
                raise ValueError(f"{name}: sha256 differs from the release's ({bad} kept for inspection)")
        paths.append(dest)
    return paths


def read_metadata(data_dir: Path) -> pd.DataFrame:
    """The metadata CSV with `new_id` as text (six digits, the record's index in the 160K release)."""
    df = pd.read_csv(dataset_dir(data_dir) / METADATA_FILE, dtype={"new_id": str}, keep_default_na=False,
                     na_values={c: [""] for c in ("age", *MEASUREMENTS)})
    df["new_id"] = df["new_id"].astype(object)
    df["sex"] = df["sex"].replace("", None)
    return df


@dataclass
class Signals:
    keys: np.ndarray            # new_id per row
    values: np.memmap           # (records, 8 stored leads, 5000) int16

    def index_of(self, new_id: str) -> int:
        return int(np.flatnonzero(self.keys == new_id)[0])


def _member_offset(path: Path, member: str) -> tuple[int, tuple, np.dtype]:
    """Byte offset, shape and dtype of an uncompressed .npy member of an NPZ (for np.memmap)."""
    with zipfile.ZipFile(path) as z:
        info = z.getinfo(member)
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"{path.name}: {member} is not stored uncompressed; cannot memory-map it")
    with Path(path).open("rb") as fh:
        fh.seek(info.header_offset)
        local = fh.read(30)
        name_len, extra_len = struct.unpack("<HH", local[26:30])
        fh.seek(info.header_offset + 30 + name_len + extra_len)
        version = np.lib.format.read_magic(fh)
        read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else \
            np.lib.format.read_array_header_2_0
        shape, fortran, dtype = read_header(fh)
        if fortran:
            raise ValueError(f"{path.name}: {member} is Fortran-ordered")
        return fh.tell(), shape, dtype


def open_signals(data_dir: Path) -> Signals:
    return open_signals_file(dataset_dir(data_dir) / SIGNAL_FILE)


def open_signals_file(path: Path) -> Signals:
    path = Path(path)
    with np.load(path) as npz:
        keys = np.asarray(npz["keys"]).astype(object)
    offset, shape, dtype = _member_offset(path, "vals.npy")
    if len(shape) != 3 or shape[1:] != (len(STORED_LEADS), SAMPLES) or shape[0] != len(keys):
        raise ValueError(f"{path.name}: unexpected signal array {shape}")
    return Signals(keys=keys, values=np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape))


def check_alignment(meta: pd.DataFrame, signals: Signals) -> None:
    """The CSV rows and the NPZ rows must be the same records in the same order (release note of 1.1.1)."""
    if len(meta) != len(signals.keys) or not np.array_equal(meta["new_id"].to_numpy(object), signals.keys):
        raise ValueError("metadata and signal records are not in the same order")


def to_12_leads(x8: np.ndarray) -> np.ndarray:
    """(..., 8, n) stored leads I, II, V1-V6 -> (..., 12, n) float32 in LEADS12 order; III = II - I,
    aVR = -(I + II) / 2, aVL = I - II / 2, aVF = II - I / 2."""
    x = np.asarray(x8, dtype=np.float32)
    i, ii = x[..., 0, :], x[..., 1, :]
    limb = np.stack([i, ii, ii - i, -(i + ii) / 2, i - ii / 2, ii - i / 2], axis=-2)
    return np.concatenate([limb, x[..., 2:, :]], axis=-2)


def select_leads(x12: np.ndarray, lead_set: str) -> np.ndarray:
    """The Challenge lead set '12', '6', '4', '3' or '2' from a 12-lead array (lead axis -2)."""
    names = LEAD_SETS[lead_set]
    return np.asarray(x12)[..., [LEADS12.index(n) for n in names], :]
