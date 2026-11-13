"""PTB-XL 1.0.3 (PhysioNet, doi:10.13026/kfzx-aw45, CC BY 4.0): download, metadata, labels and 500 Hz records.

PTB-XL holds 21,799 ten-second 12-lead ECGs of 18,869 patients from the Physikalisch-Technische Bundesanstalt
(Germany), with SCP-ECG statements, a ready aggregation of the diagnostic statements into five superclasses
(scp_statements.csv) and ten patient-safe folds (`strat_fold`; the dataset page recommends folds 1-8 for training,
9 for validation and 10 for testing, and says folds 9 and 10 had at least one human evaluation).

Citation requests (PhysioNet page, read 2026-11-12): the release (Wagner et al. 2022, doi:10.13026/kfzx-aw45), the
paper (Wagner et al., Sci Data 2020;7:154, doi:10.1038/s41597-020-0495-6) and the standard PhysioNet citation
(Pollard et al., Nature Health 2026, doi:10.1038/s44360-026-00096-z).
"""
import ast
import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_SLUG = "ptb-xl"
VERSION = "1.0.3"
DOI = "10.13026/kfzx-aw45"
ZIP_URL = ("https://physionet.org/static/published-projects/ptb-xl/"
           "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip")
SUMS_URL = "https://physionet.org/files/ptb-xl/1.0.3/SHA256SUMS.txt"
ZIP_ROOT = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
DATABASE_FILE = "ptbxl_database.csv"
STATEMENTS_FILE = "scp_statements.csv"
HZ = 500
SAMPLES = 5000
GAIN = 1000.0                      # ADC units per mV in every 500 Hz header, i.e. 1 uV per unit
HEADER_LEADS = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
LEADS12 = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
STORED_EIGHT = (0, 1, 6, 7, 8, 9, 10, 11)    # I, II, V1-V6: the independent leads (the others are derived from I, II)
# The diagnostic superclasses of scp_statements.csv; the authors' MultiLabelBinarizer orders them alphabetically.
SUPERCLASSES = ("CD", "HYP", "MI", "NORM", "STTC")
SEX = {0: "M", 1: "F"}             # the release codes sex as 0 male, 1 female


def dataset_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "datasets" / DATASET_SLUG / VERSION


def release_dir(data_dir: Path) -> Path:
    """Where the release's files are, once the ZIP is extracted."""
    return dataset_dir(data_dir) / ZIP_ROOT


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def parse_listing(text: str) -> dict[str, str]:
    """PhysioNet's SHA256SUMS.txt as {relative path: sha256}."""
    out = {}
    for line in text.splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            out[name.strip()] = digest
    return out


def verify_listing(root: Path, listing: str, names=None) -> dict[str, str]:
    """Check the files `names` (every listed file when None) under `root` against the listing; raise on a mismatch."""
    sums = parse_listing(listing)
    names = list(sums) if names is None else list(names)
    bad = [n for n in names if n not in sums or not (Path(root) / n).exists() or sha256_file(Path(root) / n) != sums[n]]
    if bad:
        raise ValueError(f"{len(bad)} files differ from SHA256SUMS.txt, e.g. {bad[:3]}")
    return {n: "ok" for n in names}


def fetch(data_dir: Path, get=None) -> Path:
    """Download the release ZIP and PhysioNet's checksum listing, extract, and verify every listed file. Files
    already extracted and verified are kept. `get(url, dest)` downloads (httpx streaming by default)."""
    root = dataset_dir(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    get = get or _download
    sums = root / "SHA256SUMS.txt"
    if not sums.exists():
        get(SUMS_URL, sums)
    archive = root / "ptb-xl-1.0.3.zip"
    if not (release_dir(data_dir) / DATABASE_FILE).exists():
        if not archive.exists():
            get(ZIP_URL, archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(root)
    verify_listing(release_dir(data_dir), sums.read_text())
    return release_dir(data_dir)


def _download(url: str, dest: Path) -> None:
    import httpx

    tmp = Path(str(dest) + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120,
                      headers={"User-Agent": "BenchToward dataset fetch"}) as r:
        r.raise_for_status()
        with tmp.open("wb") as fh:
            for block in r.iter_bytes(1 << 20):
                fh.write(block)
    tmp.replace(dest)


def read_database(root: Path) -> pd.DataFrame:
    """ptbxl_database.csv indexed by ecg_id: scp_codes as dicts, patient_id and strat_fold as integers, sex as M/F."""
    df = pd.read_csv(Path(root) / DATABASE_FILE, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    df["patient_id"] = df["patient_id"].astype(float).astype(int)
    df["strat_fold"] = df["strat_fold"].astype(int)
    df["sex"] = df["sex"].astype(int).map(SEX)
    return df


def read_statements(root: Path) -> pd.DataFrame:
    return pd.read_csv(Path(root) / STATEMENTS_FILE, index_col=0)


def superdiagnostic(scp_codes: dict, statements: pd.DataFrame) -> list[str]:
    """The record's diagnostic superclasses, as Strodthoff et al.'s aggregate_diagnostic computes them: every key of
    scp_codes that is a diagnostic statement (diagnostic == 1), whatever its likelihood, mapped to its
    diagnostic_class; sorted, without repeats."""
    diagnostic = statements[statements["diagnostic"] == 1.0]
    out = set()
    for code in scp_codes:
        if code in diagnostic.index:
            c = diagnostic.loc[code, "diagnostic_class"]
            if str(c) != "nan" and c is not None:
                out.add(str(c))
    return sorted(out)


def split_of(fold: int) -> str:
    """The dataset page's recommendation: folds 1-8 train, 9 validation, 10 test."""
    fold = int(fold)
    if not 1 <= fold <= 10:
        raise ValueError(f"strat_fold {fold} is not 1-10")
    return "train" if fold <= 8 else ("validation" if fold == 9 else "test")


def read_header(path: Path) -> dict:
    lines = [l for l in Path(path).read_text().splitlines() if l.strip() and not l.startswith("#")]
    name, n_sig, fs, n = lines[0].split()[:4]
    signals = []
    for line in lines[1:1 + int(n_sig)]:
        # file format gain[(baseline)]/units adc_resolution adc_zero initial_value checksum block_size name; the
        # baseline defaults to adc_zero when the gain field carries none (WFDB header format)
        parts = line.split()
        gain_field = parts[2].split("/")[0]
        gain, _, base = gain_field.partition("(")
        baseline = int(base.rstrip(")")) if base else int(parts[4])
        signals.append({"file": parts[0], "fmt": parts[1], "gain": float(gain), "baseline": baseline,
                        "name": parts[-1]})
    return {"name": name, "n_sig": int(n_sig), "fs": float(fs.split("/")[0]), "samples": int(n), "signals": signals}


def read_record(root: Path, filename: str) -> np.ndarray:
    """(12, 5000) int16 in the release's ADC units (1 uV, baseline 0) of one 500 Hz record, leads in HEADER_LEADS
    order. Refuses a header with another rate, length, format, gain, baseline or lead order."""
    base = Path(root) / filename
    h = read_header(base.with_suffix(".hea"))
    if h["fs"] != HZ or h["samples"] != SAMPLES or h["n_sig"] != len(HEADER_LEADS):
        raise ValueError(f"{filename}: expected 12 leads at {HZ} Hz x {SAMPLES}, got {h['n_sig']} at {h['fs']} x "
                         f"{h['samples']}")
    for s, lead in zip(h["signals"], HEADER_LEADS):
        if s["fmt"] != "16" or s["name"].upper() != lead:
            raise ValueError(f"{filename}: unexpected format or lead order ({s['fmt']}, {s['name']})")
        if s["gain"] != GAIN:
            raise ValueError(f"{filename}: gain {s['gain']} is not {GAIN}")
        if s["baseline"] != 0:
            raise ValueError(f"{filename}: baseline {s['baseline']} is not 0")
    raw = np.fromfile(base.parent / h["signals"][0]["file"], dtype="<i2")
    if raw.size != SAMPLES * len(HEADER_LEADS):
        raise ValueError(f"{filename}: {raw.size} samples, expected {SAMPLES * len(HEADER_LEADS)}")
    return raw.reshape(SAMPLES, len(HEADER_LEADS)).T.astype(np.int16)


def stored_eight(x12: np.ndarray) -> np.ndarray:
    """The independent leads I, II, V1-V6 of a 12-lead array (lead axis -2), the layout the LYDUS feature and network
    code reads."""
    return np.asarray(x12)[..., list(STORED_EIGHT), :]
