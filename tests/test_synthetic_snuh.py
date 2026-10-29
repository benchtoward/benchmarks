import hashlib
import io
import zipfile

import duckdb
import httpx
import pandas as pd
import pytest

from medbench.khdp import KhdpClient
from medbench.tasks.synthetic_snuh import (CDM_ZIP, build_duckdb, connect_readonly, db_path, dataset_dir, fetch_cdm,
                                           table_counts)

ROWS = {"person": 3, "concept": 4, "condition_occurrence": 3, "death": 0}


def _parquet(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def cdm_zip(extra: dict[str, bytes] | None = None) -> bytes:
    """A miniature release: unpartitioned person and concept, condition_occurrence partitioned by year, empty death."""
    person = pd.DataFrame({"person_id": [1, 2, 3], "gender_concept_id": [8507, 8532, 8532],
                           "year_of_birth": [1950, 1980, 2001]})
    concept = pd.DataFrame({"concept_id": [8507, 8532, 201826, 320128],
                            "concept_name": ["MALE", "FEMALE", "Type 2 diabetes mellitus", "Essential hypertension"],
                            "domain_id": ["Gender", "Gender", "Condition", "Condition"]})
    cond = pd.DataFrame({"condition_occurrence_id": [10, 11, 12], "person_id": [1, 2, 2],
                         "condition_concept_id": [201826, 201826, 320128],
                         "condition_start_date": pd.to_datetime(["2021-03-01", "2022-05-02", "2022-07-07"]).date})
    death = pd.DataFrame({"person_id": pd.Series([], dtype="int32"), "death_date": pd.Series([], dtype="object")})
    files = {
        "cdm/person/data_0.parquet": _parquet(person),
        "cdm/concept/data_0.parquet": _parquet(concept),
        "cdm/condition_occurrence/year=2021/data_0.parquet": _parquet(cond.iloc[:1]),
        "cdm/condition_occurrence/year=2022/data_0.parquet": _parquet(cond.iloc[1:]),
        "cdm/death/data_0.parquet": _parquet(death),
        **(extra or {}),
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


class KhdpMock:
    """Gateway file listing (bare keys, as Synthetic-SNUH lists them) plus presigned storage."""

    def __init__(self, zip_bytes: bytes | None = None):
        self.requests: list[httpx.Request] = []
        self.files = {"README.md": b"# Synthetic-SNUH v4\n", "LICENSE.txt": b"CC BY-NC 4.0\n",
                      CDM_ZIP: zip_bytes if zip_bytes is not None else cdm_zip(),
                      "Synthetic-SNUH_v4_CDM_CSV.zip": b"not needed"}
        self.sums = {k: hashlib.md5(v).hexdigest() for k, v in self.files.items()}
        self.client = KhdpClient("secret-token", http=httpx.Client(transport=httpx.MockTransport(self._handle)))

    def tamper(self, key: str) -> None:
        self.files[key] = self.files[key] + b"tampered"

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if request.url.host == "api.khdp.net" and path == "/datasets/synthetic-snuh/4.0.0/files":
            listing = {**self.files, "MD5SUMS.txt": self._sums_text()}
            items = [{"key": k, "size": len(v), "url": f"https://storage.test/3183/{k}?X-Amz-Signature=x"}
                     for k, v in sorted(listing.items())]
            return httpx.Response(200, json={"items": items, "continueToken": None})
        if request.url.host == "storage.test":
            key = path.removeprefix("/3183/")
            return httpx.Response(200, content=self._sums_text() if key == "MD5SUMS.txt" else self.files[key])
        return httpx.Response(404, json={})

    def _sums_text(self) -> bytes:
        return "".join(f"{md5}  {key}\n" for key, md5 in self.sums.items()).encode()

    def downloads(self) -> list[str]:
        return [r.url.path.removeprefix("/3183/") for r in self.requests if r.url.host == "storage.test"]


def test_fetch_downloads_verifies_and_extracts(tmp_path):
    khdp = KhdpMock()
    root = fetch_cdm(tmp_path, khdp.client)
    assert root == dataset_dir(tmp_path) / "cdm"
    assert dataset_dir(tmp_path) == tmp_path / "datasets" / "synthetic-snuh" / "4.0.0"
    assert (root / "condition_occurrence" / "year=2022" / "data_0.parquet").exists()
    assert (dataset_dir(tmp_path) / "README.md").exists() and (dataset_dir(tmp_path) / "LICENSE.txt").exists()
    assert "Synthetic-SNUH_v4_CDM_CSV.zip" not in khdp.downloads()  # only the Parquet release is needed


def test_token_goes_only_to_the_gateway(tmp_path):
    khdp = KhdpMock()
    fetch_cdm(tmp_path, khdp.client)
    for request in khdp.requests:
        has_token = request.headers.get("Authorization") == "Bearer secret-token"
        assert has_token == (request.url.host == "api.khdp.net")


def test_fetch_is_resumable(tmp_path):
    khdp = KhdpMock()
    fetch_cdm(tmp_path, khdp.client)
    first = len(khdp.downloads())
    fetch_cdm(tmp_path, khdp.client)
    assert khdp.downloads()[first:] == ["MD5SUMS.txt"]  # verified files are neither downloaded nor re-extracted


def test_fetch_rejects_checksum_mismatch(tmp_path):
    khdp = KhdpMock()
    khdp.tamper(CDM_ZIP)
    with pytest.raises(ValueError, match="checksum"):
        fetch_cdm(tmp_path, khdp.client)
    assert not (dataset_dir(tmp_path) / CDM_ZIP).exists()
    assert not (dataset_dir(tmp_path) / "cdm").exists()


@pytest.mark.parametrize("member", ["../evil.parquet", "cdm/../../evil.parquet", "/abs/evil.parquet", "other/x.parquet"])
def test_fetch_rejects_members_outside_cdm(tmp_path, member):
    khdp = KhdpMock(cdm_zip({member: b"x"}))
    with pytest.raises(ValueError, match="unexpected path"):
        fetch_cdm(tmp_path, khdp.client)
    assert not (dataset_dir(tmp_path) / "cdm").exists()
    assert not (tmp_path / "evil.parquet").exists()


def test_build_uses_omop_table_names_without_partition_column(tmp_path):
    fetch_cdm(tmp_path, KhdpMock().client)
    path = build_duckdb(tmp_path, expected_rows=ROWS)
    assert path == db_path(tmp_path) and path.name == "omop_cdm.duckdb"
    con = connect_readonly(path)
    assert table_counts(con) == ROWS
    columns = [r[0] for r in con.execute("DESCRIBE condition_occurrence").fetchall()]
    assert columns == ["condition_occurrence_id", "person_id", "condition_concept_id", "condition_start_date"]
    assert con.execute("SELECT count(DISTINCT person_id) FROM condition_occurrence "
                       "WHERE condition_concept_id = 201826").fetchall() == [(2,)]


def test_build_checks_expected_row_counts(tmp_path):
    fetch_cdm(tmp_path, KhdpMock().client)
    with pytest.raises(ValueError, match="person"):
        build_duckdb(tmp_path, expected_rows={**ROWS, "person": 10000})
    assert not db_path(tmp_path).exists()


def test_build_is_skipped_when_current_and_redone_when_forced(tmp_path):
    fetch_cdm(tmp_path, KhdpMock().client)
    path = build_duckdb(tmp_path, expected_rows=ROWS)
    stamp = path.stat().st_mtime_ns
    assert build_duckdb(tmp_path, expected_rows=ROWS).stat().st_mtime_ns == stamp
    assert build_duckdb(tmp_path, expected_rows=ROWS, force=True).stat().st_mtime_ns != stamp


def test_build_rejects_non_omop_table_directories(tmp_path):
    fetch_cdm(tmp_path, KhdpMock(cdm_zip({"cdm/person; DROP TABLE x/data_0.parquet": b"x"})).client)
    with pytest.raises(ValueError, match="not an OMOP CDM"):
        build_duckdb(tmp_path, expected_rows=ROWS)


def test_readonly_connection_blocks_writes_and_file_access(tmp_path):
    fetch_cdm(tmp_path, KhdpMock().client)
    con = connect_readonly(build_duckdb(tmp_path, expected_rows=ROWS))
    secret = tmp_path / "secret.csv"
    secret.write_text("a\n1\n")
    for sql in ["DROP TABLE person", "CREATE TABLE x AS SELECT 1", f"SELECT * FROM read_csv('{secret}')",
                f"COPY person TO '{tmp_path / 'out.csv'}'", f"ATTACH '{tmp_path / 'o.duckdb'}' AS o",
                "SET enable_external_access = true"]:
        with pytest.raises(duckdb.Error):
            con.execute(sql)
    assert not (tmp_path / "out.csv").exists()
    assert con.execute("SELECT count(*) FROM person").fetchall() == [(3,)]
