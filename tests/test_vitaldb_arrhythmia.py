from pathlib import Path

import httpx
import pandas as pd
import pytest

from benchtoward.khdp import KhdpClient, KhdpEntry, page_url
from benchtoward.tasks.vitaldb_arrhythmia import fetch_annotations, load_annotations
from benchtoward.tasks.vitaldb_split import caseid_split, vitaldb_subject_split

FIXTURE = Path(__file__).parent / "fixtures" / "arrhythmia"


class KhdpMock:
    """Gateway file listing with presigned storage URLs, plus the storage itself."""

    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.overrides: dict[str, tuple[int, object]] = {}
        self.files = {f"1.0.0/{p.relative_to(FIXTURE).as_posix()}": p.read_bytes()
                      for p in FIXTURE.rglob("*") if p.is_file()}
        self.client = httpx.Client(transport=httpx.MockTransport(self._handle))

    def corrupt(self, key: str) -> None:
        self.files[key] = self.files[key] + b"tampered"

    def respond(self, path: str, status: int, body: object) -> None:
        self.overrides[path] = (status, body)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path in self.overrides:
            status, body = self.overrides[path]
            return httpx.Response(status, json=body)
        if request.url.host == "api.khdp.net" and path == "/datasets/vitaldb-arrhythmia/1.0.0/files":
            keys = sorted(self.files)
            token = request.url.params.get("continueToken")
            page = keys[3:] if token else keys[:3]
            items = [{"key": k, "size": len(self.files[k]), "url": f"https://storage.test/738/{k}?X-Amz-Signature=x"}
                     for k in page]
            return httpx.Response(200, json={"items": items, "continueToken": None if token else "next"})
        if request.url.host == "api.khdp.net" and path == "/datasets":
            return httpx.Response(200, json={"totalCnt": 2, "data": [
                {"code": "vitaldb_open", "title": "Open VitalDB", "type": "SNUH", "accessPolicy": "open", "version": "1.0.0"},
                {"code": "vitaldb", "title": "Extended VitalDB", "type": "SNUH", "accessPolicy": "credentialed", "version": "1.0.0"}]})
        if request.url.host == "storage.test":
            return httpx.Response(200, content=self.files[path.removeprefix("/738/")])
        if request.url.host == "khdp.net":
            return httpx.Response(200, json={"cvId": 738, "ciCode": "vitaldb-arrhythmia"})
        return httpx.Response(404, json={})


@pytest.fixture
def khdp_mock():
    return KhdpMock()


def test_split_is_independent_of_cohort():
    cases = pd.DataFrame({"caseid": range(1, 101), "subjectid": [i // 2 for i in range(100)]})
    full = vitaldb_subject_split(cases)
    assert full == vitaldb_subject_split(cases.sample(frac=1, random_state=3))   # row order does not matter
    assert set(full.values()) == {"train", "validation", "test"}
    cmap = caseid_split(cases)
    assert cmap[1] == full[0] and cmap[3] == full[1]


def test_fetch_verifies_checksums(tmp_path, khdp_mock):
    client = KhdpClient("khdp_pat_test", http=khdp_mock.client)
    root = fetch_annotations(tmp_path, client)
    assert (root / "metadata.csv").exists()
    df = load_annotations(tmp_path)
    assert set(df.columns) >= {"caseid", "time_second", "beat_type", "rhythm_label", "bad_signal_quality"}
    assert sorted(df["caseid"].unique()) == [337, 1001, 2453] and df["bad_signal_quality"].dtype == bool
    assert df["caseid"].dtype.kind == "i" and df.loc[df.caseid == 2453, "time_second"].tolist() == [300.0, 301.0, 302.0]
    khdp_mock.corrupt("1.0.0/Annotation_Files/Annotation_file_337.csv")
    with pytest.raises(ValueError, match="checksum"):
        fetch_annotations(tmp_path / "again", KhdpClient("khdp_pat_test", http=khdp_mock.client))


def test_fetch_is_resumable(tmp_path, khdp_mock):
    client = KhdpClient("khdp_pat_test", http=khdp_mock.client)
    fetch_annotations(tmp_path, client)
    data_gets = lambda: [r for r in khdp_mock.requests  # noqa: E731
                         if r.url.host == "storage.test" and not r.url.path.endswith("SHA256SUMS.txt")]
    before = len(data_gets())
    fetch_annotations(tmp_path, client)
    assert len(data_gets()) == before   # verified files are not downloaded again (the sums file is re-read)


def test_token_only_sent_to_gateway(tmp_path, khdp_mock):
    fetch_annotations(tmp_path, KhdpClient("khdp_pat_test", http=khdp_mock.client))
    page_url("vitaldb-arrhythmia", "1.0.0", http=khdp_mock.client)
    for req in khdp_mock.requests:
        assert ("authorization" in req.headers) == (req.url.host == "api.khdp.net"), req.url


def test_files_of_credentialed_dataset_raise(khdp_mock):
    khdp_mock.respond("/datasets/vitaldb/1.0.0/files", 400, {"detail": "Is Not Open Access Dataset"})
    with pytest.raises(PermissionError):
        KhdpClient("khdp_pat_test", http=khdp_mock.client).files("vitaldb", "1.0.0")


def test_datasets_and_page_url(khdp_mock):
    entries = KhdpClient("khdp_pat_test", http=khdp_mock.client).datasets("vitaldb")
    assert [(e.code, e.is_open) for e in entries] == [("vitaldb_open", True), ("vitaldb", False)]
    assert isinstance(entries[0], KhdpEntry)
    assert page_url("vitaldb-arrhythmia", "1.0.0", http=khdp_mock.client) == \
        "https://khdp.net/database/data-search-detail/738/vitaldb-arrhythmia/1.0.0"
