"""SYN-SNUH release file: fetch from KHDP with the SHA-256 check (invented bytes, fake client)."""
import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest

from benchtoward.tasks import syn_snuh


@dataclass
class _File:
    key: str
    size: int
    url: str


class _Client:
    def __init__(self, payload: bytes, key: str = "1.0.0/synthetic_snuh.duckdb"):
        self.payload, self.key, self.downloads = payload, key, 0

    def files(self, code, version):
        assert (code, version) == ("SYN-SNUH", "1.0.0")
        return [_File("1.0.0/README.txt", 3, "u0"), _File(self.key, len(self.payload), "u1")]

    def download(self, file, dest: Path):
        self.downloads += 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.payload)
        return dest


def test_fetch_downloads_once_and_checks_the_sha256(tmp_path, monkeypatch):
    payload = b"invented release bytes"
    monkeypatch.setattr(syn_snuh, "SHA256", hashlib.sha256(payload).hexdigest())
    client = _Client(payload)
    path = syn_snuh.fetch(tmp_path, client)
    assert path == tmp_path / "datasets" / "syn-snuh" / "1.0.0" / "synthetic_snuh.duckdb"
    assert path.read_bytes() == payload and client.downloads == 1
    assert syn_snuh.fetch(tmp_path, client) == path and client.downloads == 1   # a verified copy is reused


def test_fetch_refuses_a_file_whose_sha256_differs(tmp_path, monkeypatch):
    monkeypatch.setattr(syn_snuh, "SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        syn_snuh.fetch(tmp_path, _Client(b"something else"))
    assert not syn_snuh.db_path(tmp_path).exists()


def test_fetch_needs_the_release_file(tmp_path):
    with pytest.raises(ValueError, match="synthetic_snuh.duckdb"):
        syn_snuh.fetch(tmp_path, _Client(b"x", key="1.0.0/other.duckdb"))
