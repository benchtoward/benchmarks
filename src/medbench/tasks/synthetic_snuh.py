"""Synthetic-SNUH v4 (KHDP `synthetic-snuh` 4.0.0, CC BY-NC 4.0; vocabularies keep their own licences).

10,000 synthetic patients modelled on SNUH 2021-2025 as OMOP CDM v5.4. The release says it is for building pipelines,
queries and AI agents, not for clinical inference. This module downloads the Parquet release from KHDP, checks MD5
sums, unzips it into data/datasets/synthetic-snuh/4.0.0/cdm/ and loads it into one DuckDB file whose tables carry
the OMOP CDM names. Model-written SQL only ever runs on `connect_readonly` connections.
"""
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath

import duckdb

from medbench.khdp import KhdpClient

CODE, VERSION = "synthetic-snuh", "4.0.0"
CDM_ZIP = "Synthetic-SNUH_v4_CDM_Parquet.zip"
DOC_FILES = ("README.md", "LICENSE.txt")
SUMS_FILE = "MD5SUMS.txt"
DB_NAME = "omop_cdm.duckdb"
_MARKER = ".source-md5"

# OMOP CDM v5.4 tables (https://ohdsi.github.io/CommonDataModel/cdm54.html); anything else in the zip is refused.
OMOP_TABLES = frozenset({
    "person", "observation_period", "visit_occurrence", "visit_detail", "condition_occurrence", "drug_exposure",
    "procedure_occurrence", "device_exposure", "measurement", "observation", "death", "note", "note_nlp",
    "specimen", "fact_relationship", "location", "care_site", "provider", "payer_plan_period", "cost",
    "drug_era", "dose_era", "condition_era", "episode", "episode_event", "metadata", "cdm_source", "concept",
    "vocabulary", "domain", "concept_class", "concept_relationship", "relationship", "concept_synonym",
    "concept_ancestor", "source_to_concept_map", "drug_strength", "cohort", "cohort_definition",
})

# Row counts published in the release README (4.0.0); a build that disagrees is refused.
EXPECTED_ROWS = {
    "care_site": 0, "concept": 5192, "condition_occurrence": 81253, "death": 48, "device_exposure": 0,
    "drug_exposure": 428321, "location": 0, "measurement": 1170252, "note": 0, "observation": 4216,
    "observation_period": 10000, "person": 10000, "procedure_occurrence": 73044, "provider": 0, "specimen": 0,
    "visit_occurrence": 55934,
}


def dataset_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "datasets" / CODE / VERSION


def db_path(data_dir: Path) -> Path:
    return dataset_dir(data_dir) / DB_NAME


def _md5(path: Path) -> str:
    h = hashlib.md5()
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


def _safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = []
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "cdm" \
                or "\\" in info.filename:
            raise ValueError(f"unexpected path in {CDM_ZIP}: {info.filename!r}")
        members.append(info)
    return members


def _extract(zip_path: Path, root: Path, md5: str) -> Path:
    """Unzip into root/cdm (replacing an older extraction); a marker records which archive it came from."""
    target = root / "cdm"
    if (target / _MARKER).exists() and (target / _MARKER).read_text().strip() == md5:
        return target
    staging = root / "cdm.part"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for info in _safe_members(archive):
                rel = PurePosixPath(info.filename).relative_to("cdm")
                dest = staging.joinpath(*rel.parts)
                if info.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as src, dest.open("wb") as out:
                    shutil.copyfileobj(src, out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    (staging / _MARKER).write_text(md5)
    shutil.rmtree(target, ignore_errors=True)
    staging.replace(target)
    return target


def fetch_cdm(data_dir: Path, client: KhdpClient) -> Path:
    """Download the OMOP CDM Parquet release, its README and licence, verify MD5 sums and unzip (resumable)."""
    root = dataset_dir(data_dir)
    files = {f.key.removeprefix(f"{VERSION}/"): f for f in client.files(CODE, VERSION)}
    if SUMS_FILE not in files:
        raise ValueError(f"{SUMS_FILE} missing from the KHDP release")
    client.download(files[SUMS_FILE], root / SUMS_FILE)
    sums = _checksums(root / SUMS_FILE)
    for name in (*DOC_FILES, CDM_ZIP):
        if name not in files or name not in sums:
            raise ValueError(f"{name} missing from the KHDP release or from {SUMS_FILE}")
        dest = root / name
        if dest.exists() and _md5(dest) == sums[name]:
            continue
        client.download(files[name], dest)
        if _md5(dest) != sums[name]:
            dest.unlink()
            raise ValueError(f"checksum mismatch for {name}")
    return _extract(root / CDM_ZIP, root, sums[CDM_ZIP])


def _table_files(cdm: Path) -> dict[str, list[str]]:
    tables = {}
    for table_dir in sorted(p for p in cdm.iterdir() if p.is_dir()):
        name = table_dir.name
        if name not in OMOP_TABLES or not re.fullmatch(r"[a-z_]+", name):
            raise ValueError(f"{name!r} is not an OMOP CDM v5.4 table")
        tables[name] = sorted(str(p) for p in table_dir.rglob("*.parquet"))
    return tables


def build_duckdb(data_dir: Path, *, expected_rows: dict[str, int] | None = EXPECTED_ROWS,
                 force: bool = False) -> Path:
    """Load every cdm/<table>/**/*.parquet into table <table> of omop_cdm.duckdb (partition paths are not columns)."""
    root = dataset_dir(data_dir)
    cdm = root / "cdm"
    source = (cdm / _MARKER).read_text().strip()
    path, sidecar = db_path(data_dir), root / (DB_NAME + ".json")
    if not force and path.exists() and sidecar.exists() and json.loads(sidecar.read_text()).get("source_md5") == source:
        return path
    tables = _table_files(cdm)
    staging = path.with_name(path.name + ".part")
    staging.unlink(missing_ok=True)
    try:
        con = duckdb.connect(str(staging))
        try:
            for name, parquet in tables.items():
                con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM read_parquet(?, hive_partitioning = false, '
                            f'union_by_name = true)', [parquet])
            counts = table_counts(con)
        finally:
            con.close()
        if expected_rows is not None and counts != expected_rows:
            diff = sorted(k for k in set(counts) | set(expected_rows) if counts.get(k) != expected_rows.get(k))
            raise ValueError(f"row counts differ from the release README for: {', '.join(diff)}")
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    staging.replace(path)
    sidecar.write_text(json.dumps({"source_md5": source, "rows": counts}, indent=1, sort_keys=True))
    return path


def connect_readonly(path: Path) -> duckdb.DuckDBPyConnection:
    """Read-only, no file or network access, configuration locked: safe for SQL written by a model."""
    return duckdb.connect(str(path), read_only=True, config={
        "enable_external_access": False, "autoinstall_known_extensions": False,
        "autoload_known_extensions": False, "lock_configuration": True})


def table_counts(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    names = [r[0] for r in con.execute("SELECT table_name FROM information_schema.tables "
                                       "WHERE table_schema = 'main' ORDER BY table_name").fetchall()]
    return {n: con.execute(f'SELECT count(*) FROM "{n}"').fetchall()[0][0] for n in names}
