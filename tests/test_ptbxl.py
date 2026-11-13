"""PTB-XL 1.0.3 loader: WFDB records, the release's diagnostic aggregation and its patient-safe folds (no network)."""
import numpy as np
import pandas as pd
import pytest

from benchtoward.tasks import ptbxl

STATEMENTS = pd.DataFrame(
    # A slice of the release's scp_statements.csv: diagnostic rows carry a superclass, form/rhythm rows do not.
    {"description": ["normal ECG", "inferior myocardial infarction", "non-diagnostic T abnormalities",
                     "complete right bundle branch block", "left ventricular hypertrophy", "sinus rhythm",
                     "premature ventricular complex"],
     "diagnostic": [1.0, 1.0, 1.0, 1.0, 1.0, None, None],
     "form": [None, None, 1.0, None, None, None, 1.0],
     "rhythm": [None, None, None, None, None, 1.0, None],
     "diagnostic_class": ["NORM", "MI", "STTC", "CD", "HYP", None, None],
     "diagnostic_subclass": ["NORM", "IMI", "STTC", "CRBBB", "LVH", None, None]},
    index=pd.Index(["NORM", "IMI", "NDT", "CRBBB", "LVH", "SR", "PVC"], name=None))


def test_superclasses_are_the_five_diagnostic_classes_in_sorted_order():
    # MultiLabelBinarizer orders the authors' classes alphabetically
    assert ptbxl.SUPERCLASSES == ("CD", "HYP", "MI", "NORM", "STTC")


def test_superdiagnostic_takes_every_diagnostic_code_whatever_its_likelihood():
    # Strodthoff et al. aggregate_diagnostic: every key of scp_codes that is a diagnostic statement, no threshold
    assert ptbxl.superdiagnostic({"IMI": 15.0, "SR": 0.0}, STATEMENTS) == ["MI"]
    assert ptbxl.superdiagnostic({"NORM": 100.0, "LVH": 0.0, "SR": 0.0}, STATEMENTS) == ["HYP", "NORM"]
    assert ptbxl.superdiagnostic({"NDT": 100.0, "CRBBB": 100.0, "PVC": 0.0}, STATEMENTS) == ["CD", "STTC"]
    assert ptbxl.superdiagnostic({"SR": 0.0, "PVC": 100.0}, STATEMENTS) == []   # no diagnostic statement
    assert ptbxl.superdiagnostic({"UNKNOWN": 50.0}, STATEMENTS) == []


def test_folds_map_to_the_recommended_split():
    assert [ptbxl.split_of(f) for f in range(1, 11)] == ["train"] * 8 + ["validation", "test"]
    with pytest.raises(ValueError):
        ptbxl.split_of(0)


def write_record(directory, name, signal, gain=1000.0, baseline=0, fs=500):
    """A WFDB format-16 record: the header and interleaved little-endian int16 samples (leads x samples)."""
    directory.mkdir(parents=True, exist_ok=True)
    n_leads, n = signal.shape
    lines = [f"{name} {n_leads} {fs} {n}"]
    for k, lead in enumerate(ptbxl.HEADER_LEADS):
        lines.append(f"{name}.dat 16 {gain:g}/mV 16 {baseline} {int(signal[k, 0])} 0 0 {lead}")
    (directory / f"{name}.hea").write_text("\n".join(lines) + "\n")
    np.asarray(signal.T, dtype="<i2").tofile(directory / f"{name}.dat")


def test_read_record_returns_the_twelve_leads_in_header_order(tmp_path):
    raw = (np.arange(12 * 5000).reshape(12, 5000) % 3000 - 1500).astype(np.int16)
    write_record(tmp_path / "records500" / "00000", "00001_hr", raw)
    out = ptbxl.read_record(tmp_path, "records500/00000/00001_hr")
    assert out.dtype == np.int16 and out.shape == (12, 5000)
    assert np.array_equal(out, raw)


def test_read_record_reads_the_baseline_in_the_gain_field(tmp_path):
    # PTB-XL's own headers write "1000.0(0)/mV 16 0": the baseline in parentheses
    raw = np.ones((12, 5000), dtype=np.int16)
    write_record(tmp_path / "r", "x_hr", raw)
    hea = tmp_path / "r" / "x_hr.hea"
    hea.write_text(hea.read_text().replace("1000/mV 16 0", "1000.0(0)/mV 16 0"))
    assert np.array_equal(ptbxl.read_record(tmp_path, "r/x_hr"), raw)
    hea.write_text(hea.read_text().replace("1000.0(0)/mV", "1000.0(7)/mV"))
    with pytest.raises(ValueError, match="baseline"):
        ptbxl.read_record(tmp_path, "r/x_hr")


def test_read_record_refuses_another_gain_rate_or_lead_order(tmp_path):
    raw = np.zeros((12, 5000), dtype=np.int16)
    write_record(tmp_path / "a", "x_hr", raw, gain=200.0)
    with pytest.raises(ValueError, match="gain"):
        ptbxl.read_record(tmp_path, "a/x_hr")
    write_record(tmp_path / "b", "x_hr", raw, fs=100)
    with pytest.raises(ValueError, match="500"):
        ptbxl.read_record(tmp_path, "b/x_hr")
    write_record(tmp_path / "c", "x_hr", raw, baseline=5)
    with pytest.raises(ValueError, match="baseline"):
        ptbxl.read_record(tmp_path, "c/x_hr")


def test_stored_eight_leads_are_the_independent_ones():
    x12 = np.arange(12)[:, None] * np.ones((1, 4))
    x8 = ptbxl.stored_eight(x12)
    assert x8.shape == (8, 4) and list(x8[:, 0]) == [0, 1, 6, 7, 8, 9, 10, 11]   # I, II, V1-V6


def test_read_database_parses_codes_and_types(tmp_path):
    (tmp_path / ptbxl.DATABASE_FILE).write_text(
        "ecg_id,patient_id,age,sex,scp_codes,strat_fold,filename_hr\n"
        "1,15709.0,56.0,1,\"{'NORM': 100.0, 'SR': 0.0}\",3,records500/00000/00001_hr\n"
        "2,13243.0,300.0,0,\"{'IMI': 15.0}\",10,records500/00000/00002_hr\n")
    db = ptbxl.read_database(tmp_path)
    assert list(db.index) == [1, 2] and db.loc[1, "scp_codes"] == {"NORM": 100.0, "SR": 0.0}
    assert db["patient_id"].tolist() == [15709, 13243] and db["strat_fold"].tolist() == [3, 10]
    assert db["sex"].tolist() == ["F", "M"]   # PTB-XL: 0 male, 1 female


def test_checksums_match_the_physionet_listing(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"x")
    listing = "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881 a.txt\n"
    assert ptbxl.verify_listing(tmp_path, listing, names=["a.txt"]) == {"a.txt": "ok"}
    (tmp_path / "a.txt").write_bytes(b"y")
    with pytest.raises(ValueError, match="a.txt"):
        ptbxl.verify_listing(tmp_path, listing, names=["a.txt"])
