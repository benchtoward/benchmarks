"""VitalDB Arrhythmia Database (KHDP `vitaldb-arrhythmia` 1.0.0, CC BY 4.0): anesthesiologist-labeled beats and
rhythms on VitalDB ECG lead II. Annotations come from KHDP; waveforms from VitalDB (500 Hz)."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from benchtoward.khdp import KhdpClient

CODE, VERSION = "vitaldb-arrhythmia", "1.0.0"
ECG_TRACK = "SNUADC/ECG_II"
ECG_HZ = 500


def dataset_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "datasets" / CODE / VERSION


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _checksums(path: Path) -> dict[str, str]:
    sums = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            sums[parts[1].lstrip("*").strip()] = parts[0].lower()
    return sums


def fetch_annotations(data_dir: Path, client: KhdpClient) -> Path:
    """Download every release file into data/datasets/vitaldb-arrhythmia/1.0.0/ and verify SHA-256 (resumable)."""
    root = dataset_dir(data_dir)
    files = {f.key.removeprefix(f"{VERSION}/"): f for f in client.files(CODE, VERSION)}
    sums_file = files.get("SHA256SUMS.txt")
    if sums_file is None:
        raise ValueError("SHA256SUMS.txt missing from the KHDP release")
    client.download(sums_file, root / "SHA256SUMS.txt")
    sums = _checksums(root / "SHA256SUMS.txt")
    for rel, f in sorted(files.items()):
        if rel == "SHA256SUMS.txt":
            continue
        dest = root / rel
        expected = sums.get(rel)
        if dest.exists() and (expected is None or _sha256(dest) == expected):
            continue
        client.download(f, dest)
        if expected is not None and _sha256(dest) != expected:
            dest.unlink()
            raise ValueError(f"checksum mismatch for {rel}")
    missing = [rel for rel in sums if not (root / rel).exists()]
    if missing:
        raise ValueError(f"{len(missing)} file(s) listed in SHA256SUMS.txt were not downloaded, e.g. {missing[:3]}")
    return root


def load_metadata(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(dataset_dir(data_dir) / "metadata.csv").rename(columns={"case_id": "caseid"})


def load_annotations(data_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted((dataset_dir(data_dir) / "Annotation_Files").glob("Annotation_file_*.csv")):
        caseid = int(path.stem.rsplit("_", 1)[1])
        df = pd.read_csv(path)
        if "caseid" in df.columns:  # one release file (2453) carries its own caseid column, in another column order
            if not (df["caseid"].dropna().astype(float) == caseid).all():  # a few rows leave it empty
                raise ValueError(f"{path.name}: caseid column disagrees with the file name")
            df = df.drop(columns="caseid")
        df.insert(0, "caseid", caseid)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["bad_signal_quality"] = out["bad_signal_quality"].astype(str).str.lower().eq("true")
    return out


def load_ecg(caseid: int, start_s: float, duration_s: float) -> np.ndarray:
    import vitaldb

    values = vitaldb.load_case(int(caseid), [ECG_TRACK], 1 / ECG_HZ)[:, 0]
    a = int(round(start_s * ECG_HZ))
    return values[a:a + int(round(duration_s * ECG_HZ))]
