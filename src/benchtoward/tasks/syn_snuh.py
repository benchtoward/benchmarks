"""SYN-SNUH 1.0.0 (KHDP `SYN-SNUH`, CC BY 4.0, doi:10.23421/2026.0770): OMOP CDM v5.4 synthetic EHR of 10,000
patients with one LLM-written Korean note per visit (`note_llm_log.text`; `note.note_text` is a template rendered
from the structured rows and is not used).

This module only reads the release file, read-only. Builders download it themselves from KHDP; it is never
redistributed.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

CODE, VERSION = "SYN-SNUH", "1.0.0"
DB_NAME = "synthetic_snuh.duckdb"
SHA256 = "fd2056b31cc74c171c3906a7e48482979b993d7a52471dbe5ddd3f3df21a87c8"


def db_path(data_dir: Path) -> Path:
    return Path(data_dir) / "datasets" / "syn-snuh" / VERSION / DB_NAME


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(data_dir: Path, client) -> Path:
    """Download `synthetic_snuh.duckdb` (open access) from KHDP with `client` (a KhdpClient) into
    data/datasets/syn-snuh/1.0.0/, unless a copy with the expected SHA-256 is already there. KHDP publishes no
    checksum for this release, so the SHA-256 is BenchToward's, measured on the downloaded file; a file that differs is
    removed and refused."""
    dest = db_path(data_dir)
    if dest.exists() and sha256(dest) == SHA256:
        return dest
    files = [f for f in client.files(CODE, VERSION) if f.key.rsplit("/", 1)[-1] == DB_NAME]
    if not files:
        raise ValueError(f"{DB_NAME} is not among the files of KHDP {CODE} {VERSION}")
    client.download(files[0], dest)
    if sha256(dest) != SHA256:
        dest.unlink()
        raise ValueError(f"SHA-256 of the downloaded {DB_NAME} differs from {SHA256}: not SYN-SNUH {VERSION}")
    return dest


def connect_readonly(path: Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(path), read_only=True)


@dataclass(frozen=True)
class Visit:
    """One visit with its LLM note and the OMOP rows the note was written from."""
    vid: int
    pid: int
    cond: str | None
    ntype: str
    text: str | None
    drugs: tuple[str, ...] = ()
    labs: dict = field(default_factory=dict, hash=False, compare=False)
    hsdn: frozenset = frozenset()

    @property
    def ed(self) -> bool:
        return self.ntype.startswith("ED")


def load_visits(path: Path) -> list[Visit]:
    """Every visit, joined on visit_occurrence_id (note_id = visit_occurrence_id = note_llm_log.note_id)."""
    con = connect_readonly(path)
    try:
        notes = con.sql("""select n.visit_occurrence_id vid, n.person_id pid, n.note_source_value ntype, l.text,
                                  c.condition_source_value cond
                           from note n left join note_llm_log l using (note_id)
                           left join condition_occurrence c on c.visit_occurrence_id = n.visit_occurrence_id
                           order by n.visit_occurrence_id""").fetchall()
        drugs: dict[int, list[str]] = {}
        for vid, drug in con.sql("select visit_occurrence_id, drug_source_value from drug_exposure "
                                 "order by visit_occurrence_id, drug_exposure_id").fetchall():
            drugs.setdefault(vid, []).append(drug)
        labs: dict[int, dict] = {}
        for vid, name, val in con.sql("select visit_occurrence_id, measurement_source_value, value_as_number "
                                      "from measurement where value_as_number is not null").fetchall():
            labs.setdefault(vid, {})[name] = float(val)
        hsdn: dict[int, set] = {}
        for vid, o in con.sql("select visit_occurrence_id, observation_source_value from observation "
                              "where observation_source_value like 'HSDN:%'").fetchall():
            parts = o.split(":")
            if len(parts) >= 3:
                hsdn.setdefault(vid, set()).add(parts[2])
    finally:
        con.close()
    return [Visit(vid=v, pid=p, cond=c, ntype=t, text=x, drugs=tuple(drugs.get(v, ())), labs=labs.get(v, {}),
                  hsdn=frozenset(hsdn.get(v, ()))) for v, p, t, x, c in notes]


def all_drug_names(path: Path) -> list[str]:
    """Distinct `drug_source_value` strings, lower-cased (the known-drug vocabulary of the release)."""
    con = connect_readonly(path)
    try:
        return sorted({r[0].lower() for r in con.sql("select distinct drug_source_value from drug_exposure "
                                                    "where drug_source_value is not null").fetchall()})
    finally:
        con.close()


def release_texts(path: Path) -> set[str]:
    """Every LLM note text in the release (for the lookup gate)."""
    con = connect_readonly(path)
    try:
        return {r[0] for r in con.sql("select text from note_llm_log where text is not null").fetchall()}
    finally:
        con.close()
