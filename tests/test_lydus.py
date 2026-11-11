"""LYDUS-SNUH-ECG-50K loader: files, metadata, memory-mapped signals, derived limb leads and the Challenge 2021 lead
sets (no network)."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from benchtoward.tasks import lydus


def write_release(root, n=6, seed=0, ids=None):
    """A miniature LYDUS release in the real layout: the metadata CSV and an uncompressed 500 Hz NPZ."""
    rng = np.random.default_rng(seed)
    ids = ids or [f"{i:06d}" for i in range(0, 3 * n, 3)]
    d = root / "datasets" / lydus.DATASET_SLUG / lydus.VERSION
    d.mkdir(parents=True, exist_ok=True)
    vals = rng.integers(-300, 300, (len(ids), 8, 5000)).astype(np.int16)
    np.savez(d / lydus.SIGNAL_FILE, keys=np.array(ids), vals=vals)
    rows = [{"new_id": i, "rhythm": "NSR", "premature beat": "NSP", "bbb": "NSP", "avb": "NSP",
             "dx": "Normal sinus rhythm\nNormal ECG\n", "age": 60.0 + k, "sex": "M" if k % 2 else "F",
             "vrate": 70.0, "arate": 70.0, "pri": 150.0, "qrsd": 90.0, "qti": 380.0, "qtc": 410.0, "paxis": 50.0,
             "raxis": 40.0, "taxis": 30.0} for k, i in enumerate(ids)]
    pd.DataFrame(rows).to_csv(d / lydus.METADATA_FILE, index=False)
    return d, vals


def test_release_constants():
    assert lydus.KHDP_CODE == "LYDUS-SNUH-ECG-50K" and lydus.VERSION == "1.1.1"
    assert lydus.DOI == "10.23421/2025.0704"
    assert set(lydus.SHA256) == {lydus.METADATA_FILE, lydus.SIGNAL_FILE}
    assert all(len(v) == 64 for v in lydus.SHA256.values())
    assert lydus.STORED_LEADS == ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")


def test_lead_sets_are_the_challenge_2021_sets():
    assert lydus.LEADS12 == ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
    assert lydus.LEAD_SETS == {
        "12": lydus.LEADS12,
        "6": ("I", "II", "III", "aVR", "aVL", "aVF"),
        "4": ("I", "II", "III", "V2"),
        "3": ("I", "II", "V2"),
        "2": ("I", "II"),
    }


def test_derived_limb_leads_follow_einthoven_and_goldberger():
    rng = np.random.default_rng(1)
    x8 = rng.integers(-500, 500, (2, 8, 50)).astype(np.int16)
    x12 = lydus.to_12_leads(x8)
    lead = {name: x12[:, k] for k, name in enumerate(lydus.LEADS12)}
    i, ii = x8[:, 0].astype(np.float32), x8[:, 1].astype(np.float32)
    assert x12.shape == (2, 12, 50) and x12.dtype == np.float32
    assert np.allclose(lead["III"], ii - i) and np.allclose(lead["aVR"], -(i + ii) / 2)
    assert np.allclose(lead["aVL"], i - ii / 2) and np.allclose(lead["aVF"], ii - i / 2)
    assert np.allclose(lead["V4"], x8[:, 5])
    two = lydus.select_leads(x12, "2")
    assert two.shape == (2, 2, 50) and np.allclose(two[:, 1], ii)
    with pytest.raises(KeyError):
        lydus.select_leads(x12, "5")


def test_read_metadata_keeps_ids_as_text_and_signals_match(tmp_path):
    d, vals = write_release(tmp_path)
    meta = lydus.read_metadata(tmp_path)
    assert meta["new_id"].tolist()[:2] == ["000000", "000003"] and meta["new_id"].dtype == object
    signals = lydus.open_signals(tmp_path)
    assert isinstance(signals.values, np.memmap) and signals.values.shape == (6, 8, 5000)
    assert np.array_equal(np.asarray(signals.values[2]), vals[2])
    assert signals.keys.tolist() == meta["new_id"].tolist()
    assert signals.index_of("000006") == 2


def test_open_signals_refuses_compressed_or_misordered_files(tmp_path):
    d, vals = write_release(tmp_path)
    np.savez_compressed(d / lydus.SIGNAL_FILE, keys=np.array(["000000"]), vals=vals[:1])
    with pytest.raises(ValueError, match="uncompressed"):
        lydus.open_signals(tmp_path)
    write_release(tmp_path)
    meta = lydus.read_metadata(tmp_path)
    with pytest.raises(ValueError, match="order"):
        lydus.check_alignment(meta.iloc[::-1], lydus.open_signals(tmp_path))


class FakeClient:
    def __init__(self, payloads):
        self.payloads = payloads
        self.downloaded = []

    def files(self, code, version):
        assert (code, version) == (lydus.KHDP_CODE, lydus.VERSION)
        from benchtoward.khdp import KhdpFile
        return [KhdpFile(f"{version}/{name}", len(data), f"https://storage.example/{name}")
                for name, data in self.payloads.items()]

    def download(self, file, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.payloads[file.key.split("/")[-1]])
        self.downloaded.append(file.key)
        return dest


def test_fetch_downloads_the_500hz_release_and_checks_sha256(tmp_path, monkeypatch):
    payloads = {lydus.METADATA_FILE: b"new_id\n000000\n", lydus.SIGNAL_FILE: b"npz",
                "LYDUS_SNUH_ECG_ANONYMIZED_50k_100hz_1219.npz": b"skip"}
    monkeypatch.setattr(lydus, "SHA256", {k: hashlib.sha256(v).hexdigest() for k, v in payloads.items()
                                          if k != "LYDUS_SNUH_ECG_ANONYMIZED_50k_100hz_1219.npz"})
    client = FakeClient(payloads)
    paths = lydus.fetch(client, tmp_path)
    assert sorted(p.name for p in paths) == sorted([lydus.METADATA_FILE, lydus.SIGNAL_FILE])
    assert len(client.downloaded) == 2                       # the 100 Hz file is not needed
    assert lydus.fetch(client, tmp_path) and len(client.downloaded) == 2   # present and verified: no download
    monkeypatch.setattr(lydus, "SHA256", {lydus.METADATA_FILE: "0" * 64, lydus.SIGNAL_FILE: "0" * 64})
    (lydus.dataset_dir(tmp_path) / lydus.METADATA_FILE).unlink()
    with pytest.raises(ValueError, match="sha256"):
        lydus.fetch(client, tmp_path)
