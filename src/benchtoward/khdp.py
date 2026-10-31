"""KHDP Gateway client (https://api.khdp.net/AGENTS.md) for open SNUH datasets.

The personal access token goes only to api.khdp.net: never to presigned storage URLs, compute pods or a
web server. File access works for `open` datasets only; stricter ones answer 400 by design.
"""
from dataclasses import dataclass
from pathlib import Path

import httpx

KHDP_GATEWAY = "https://api.khdp.net"
KHDP_SITE_API = "https://khdp.net/_api"
PAGE_URL = "https://khdp.net/database/data-search-detail/{cv_id}/{code}/{version}"


@dataclass
class KhdpFile:
    key: str  # e.g. "1.0.0/Annotation_Files/Annotation_file_1001.csv"
    size: int
    url: str  # presigned, expires (24 h)


@dataclass
class KhdpEntry:
    code: str
    version: str
    title: str
    summary: str
    access_policy: str  # open | restricted | credentialed | link
    license: str | None
    type: str | None  # uploader affiliation (e.g. "SNUH"), NOT the patients' hospital

    @property
    def is_open(self) -> bool:
        return self.access_policy == "open"


def _entry(d: dict) -> KhdpEntry:
    return KhdpEntry(code=d.get("code") or d.get("datasetCode") or "", version=str(d.get("version") or d.get("latestVersion") or ""),
                     title=d.get("title") or "", summary=d.get("summary") or "", access_policy=str(d.get("accessPolicy") or ""),
                     license=d.get("license") if isinstance(d.get("license"), str) else (d.get("license") or {}).get("code"),
                     type=d.get("type"))


@dataclass(frozen=True)
class KhdpSubmission:
    """A dataset submission of this account (draft, under review or published)."""
    code: str
    version: str
    status: str
    title: str
    doi: str | None


def submission_label(status: str) -> str:
    """KHDP submission status as shown on BenchToward."""
    s = (status or "").lower()
    if "review" in s:
        return "under review"
    if s in ("published", "public", "open"):
        return "published"
    return s.replace("_", " ") or "unknown"


class KhdpClient:
    def __init__(self, token: str, http: httpx.Client | None = None):
        self.token = token
        self.http = http or httpx.Client(timeout=httpx.Timeout(connect=10, read=120, write=60, pool=30),
                                         follow_redirects=True)

    def _get(self, path: str, **params) -> httpx.Response:
        return self.http.get(f"{KHDP_GATEWAY}{path}", params={k: v for k, v in params.items() if v is not None},
                             headers={"Authorization": f"Bearer {self.token}"})

    def datasets(self, query: str, access_policy: str | None = None, page: int = 1) -> list[KhdpEntry]:
        response = self._get("/datasets", query=query, accessPolicy=access_policy, page=page)
        response.raise_for_status()
        body = response.json()
        rows = body.get("data") or body.get("items") or []
        return [_entry(d) for d in rows if d.get("code") or d.get("datasetCode")]

    def metadata(self, code: str, version: str) -> KhdpEntry:
        response = self._get(f"/datasets/{code}/{version}")
        if response.status_code == 404:
            raise LookupError(f"no KHDP dataset {code} {version}")
        response.raise_for_status()
        return _entry(response.json() | {"code": code})

    def files(self, code: str, version: str) -> list[KhdpFile]:
        out, token = [], None
        while True:
            response = self._get(f"/datasets/{code}/{version}/files", continueToken=token)
            if response.status_code == 400 and "open access" in response.text.lower():
                raise PermissionError(f"{code} {version} is not an open-access dataset")
            response.raise_for_status()
            body = response.json()
            out += [KhdpFile(i["key"].lstrip("/"), int(i.get("size") or 0), i["url"]) for i in body.get("items", [])]
            token = body.get("continueToken")
            if not token:
                return out

    def submission(self, code: str, version: str) -> KhdpSubmission | None:
        """This account's submission of code/version, or None if there is none (deleted or never made)."""
        response = self._get(f"/submissions/{code}/{version}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        d = response.json()
        doi = (d.get("doiLink") or "").removeprefix("https://doi.org/").removeprefix("http://doi.org/") or None
        return KhdpSubmission(code=d.get("code", code), version=d.get("version", version), status=d.get("status", ""),
                              title=d.get("title", ""), doi=doi)

    def download(self, file: KhdpFile, dest: Path) -> Path:
        """Stream a presigned URL to `dest` (no bearer header: the URL carries its own signature).

        An interrupted download leaves `<dest>.part`; the next call continues it with a Range request (or starts over
        if storage ignores the range). Callers verify checksums."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        have = tmp.stat().st_size if tmp.exists() else 0
        if file.size and have >= file.size:
            have = 0  # a stale or oversized partial file: start over
        headers = {"Range": f"bytes={have}-"} if have else {}
        with self.http.stream("GET", file.url, headers=headers) as response:
            response.raise_for_status()
            mode = "ab" if have and response.status_code == 206 else "wb"
            with tmp.open(mode) as fh:
                for chunk in response.iter_bytes():
                    fh.write(chunk)
        tmp.replace(dest)
        return dest


def page_url(code: str, version: str, http: httpx.Client | None = None) -> str:
    """Public KHDP page; the cvId it needs comes only from the unauthenticated site API."""
    client = http or httpx.Client(timeout=30)
    response = client.get(f"{KHDP_SITE_API}/dataset/code/{code}", params={"version": version})
    response.raise_for_status()
    return PAGE_URL.format(cv_id=response.json()["cvId"], code=code, version=version)
