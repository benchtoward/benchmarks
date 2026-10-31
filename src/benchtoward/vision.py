"""Image input for OpenAI-compatible chat routes: content parts, a per-request cost bound and a hard spend cap.

Shared by image tasks (lydus-ecg-image@0 first). Images go as base64 data URLs in `image_url` content parts, the
form OpenAI-compatible servers accept (vLLM and hosted routes); `LLMClient.chat` sends the message list unchanged.
"""
from __future__ import annotations

import base64
import contextlib
import json
import math
import os
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
# Paid reference runs of a benchmark check (SOP "Checks"): $5 per benchmark version, and up to $20 for an image task
# (owner, 2026-10-30). The $20 is a ceiling on the task's cumulative paid spend: a caller may ask for a lower one or
# for 0 (no paid call), never a higher one (image_check_cap_usd), and the spend counts every earlier paid run.
CHECK_CAP_USD = 5.0
IMAGE_CHECK_CAP_USD = 20.0
# Upper bound on the prompt tokens one image costs on the routes BenchToward uses (they resize large images first:
# OpenAI high detail at most 765 + 170 per 512 px tile of a 2048 x 768 fit, Anthropic about 1,600 at 1568 px,
# Gemini 258 per 768 px tile); 8,000 leaves room for routes that tile finer.
IMAGE_TOKENS_MAX = 8000


def mime_for(name: str) -> str:
    suffix = name[name.rfind("."):].lower() if "." in name else ""
    if suffix not in MIME:
        raise ValueError(f"no image type for {name!r}")
    return MIME[suffix]


def image_part(data: bytes, mime: str = "image/png") -> dict:
    if not data:
        raise ValueError("empty image")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(data).decode()}"}}


def image_message(text: str, images: list[tuple[bytes, str]], *, text_first: bool = True) -> dict:
    """One user message with `text` and the images ((bytes, mime) pairs) as content parts."""
    parts = [image_part(data, mime) for data, mime in images]
    text_part = {"type": "text", "text": text}
    return {"role": "user", "content": [text_part, *parts] if text_first else [*parts, text_part]}


def count_images(messages: list[dict]) -> int:
    return sum(1 for m in messages if isinstance(m.get("content"), list)
               for p in m["content"] if isinstance(p, dict) and p.get("type") == "image_url")


def worst_case_usd(messages: list[dict], max_tokens: int, price_in_per_m: float, price_out_per_m: float) -> float:
    """An upper bound on one request's cost: every UTF-8 byte of the text as a token plus 200 for the template,
    IMAGE_TOKENS_MAX per image (its base64 bytes are not text tokens), and the full output allowance."""
    text_bytes = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            text_bytes += sum(len(json.dumps(p.get("text", ""), ensure_ascii=False).encode()) for p in content
                              if isinstance(p, dict) and p.get("type") == "text")
        else:
            text_bytes += len(json.dumps(content, ensure_ascii=False).encode())
    tokens_in = text_bytes + 200 + IMAGE_TOKENS_MAX * count_images(messages)
    return (tokens_in * price_in_per_m + max_tokens * price_out_per_m) / 1_000_000


class SpendCapReached(RuntimeError):
    """The recorded spend leaves no room for one more request under the ceiling: nothing was sent."""


class PaidRunInProgress(RuntimeError):
    """Another paid run of the task holds its lock."""


class OverrunRecorded(RuntimeError):
    """The task's ledger records a request that cost more than its worst case, and the code has not acknowledged it
    (a raised worst-case estimate and a new harness version): nothing was sent."""


class SpendCap:
    """A hard cap on paid requests: an attempt is sent only if its worst case fits under the cap with the money
    already spent (earlier runs included) and the attempts in flight. Use one `Request` per item as the route
    client's `before_attempt` hook; every attempt that may have been billed is charged its worst case unless the
    request returns a cost, and an answer whose cost is unknown is charged its worst case too. With a `ledger`,
    every reservation and charge is also written there (under `run`), each reservation before its attempt is sent.
    A request that costs more than its worst case stops all further sending, because the bound the cap relies on did
    not hold; with a ledger it is also written there as an overrun line under `harness` (the caller's harness
    version), so that later processes can refuse to send too."""

    def __init__(self, cap_usd: float, spent_usd: float = 0.0, *, ledger: SpendLedger | None = None,
                 run: str = "", harness: str = ""):
        if not cap_usd >= 0:
            raise ValueError("cap must be >= 0")
        self.cap_usd, self.spent_usd, self.reserved_usd, self.refused = cap_usd, float(spent_usd), 0.0, 0
        self.ledger, self.run, self.harness, self.overrun = ledger, run, harness, False

    def try_reserve(self, usd: float) -> bool:
        if self.overrun or self.spent_usd + self.reserved_usd + usd > self.cap_usd + 1e-12:
            self.refused += 1
            return False
        self.reserved_usd += usd
        return True

    def settle(self, reserved: float, charged: float) -> None:
        self.reserved_usd -= reserved
        self.spent_usd += charged

    def request(self, worst_usd: float, **note) -> Request:
        """`note`: fields written with the request's ledger lines (e.g. the model and instance)."""
        return Request(self, worst_usd, note)


class Request:
    def __init__(self, cap: SpendCap, worst_usd: float, note: dict | None = None):
        self.cap, self.worst, self.reserved, self.note = cap, worst_usd, 0.0, dict(note or {})
        self.id = uuid.uuid4().hex

    async def before_attempt(self, n: int) -> bool:
        if not self.cap.try_reserve(self.worst):
            return False
        self.reserved += self.worst
        self._record("reserve", self.worst)   # written before the attempt is sent
        return True

    def finish(self, cost_usd: float | None, failed_attempts: int) -> float:
        """A request that returned: what its answer cost plus each failed attempt (that may have been billed) at its
        worst case. `cost_usd` is 0 for a request that returned no answer (an error), and None when the answer's
        cost is unknown (the route reported neither its cost nor its usage). An unknown cost, or one that is not a
        finite number >= 0, is charged at the worst case: a paid answer is never free, and one recorded at $0 could
        be sent again without limit. A known cost above the worst case is an overrun: it stops all further sending
        and is written to the ledger before the charge. Returns the charge."""
        unknown = cost_usd is None or not math.isfinite(cost_usd) or cost_usd < 0
        cost = self.worst if unknown else float(cost_usd)
        if not unknown and cost > self.worst + 1e-12:
            self.cap.overrun = True
            self._record("overrun", cost, worst_usd=self.worst, harness=self.cap.harness)
        return self._settle(cost + self.worst * failed_attempts, **({"unknown_cost": True} if unknown else {}))

    def charge_all(self) -> float:
        """A request that raised: every reserved attempt is charged in full. Returns the charge."""
        return self._settle(self.reserved)

    def _settle(self, charged: float, **extra) -> float:
        had_reserved = self.reserved > 0
        self.cap.settle(self.reserved, charged)
        self.reserved = 0.0
        if had_reserved or charged:
            self._record("settle", charged, **extra)
        return charged

    def _record(self, op: str, usd: float, **extra) -> None:
        if self.cap.ledger is not None:
            self.cap.ledger.append(**self.note, **extra, op=op, run=self.cap.run, request=self.id, usd=usd)


class SpendLedger:
    """The paid spend of one task, as JSON lines in a file that run options never delete (`--force` deletes a run's
    item rows, not this file). Each attempt of a request is written as a reservation of its worst case before it
    is sent; when the request ends, one settlement replaces its reservations by its charge. A request without a
    settlement (the process was killed with the attempt in flight) stays charged at its worst case. `carry` lines
    hold spend recorded only in the database, written before `--force` deletes it. `overrun` lines record requests
    that cost more than their worst case; they are records, not charges.

    The file lives in one data folder, so the ceiling it keeps holds per data folder: a run pointed at another data
    folder (another checkout or host) keeps its own ledger. The spending limit on the route's key is the backstop
    across them."""

    def __init__(self, path: Path, task: str):
        self.path, self.task = Path(path), task

    def append(self, **record) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"task": self.task, "at": datetime.now(UTC).isoformat(timespec="seconds"),
                           **record}, sort_keys=True)
        with open(self.path, "a+b") as f:
            f.seek(0, os.SEEK_END)
            cut_short = False
            if f.tell():
                f.seek(-1, os.SEEK_END)
                cut_short = f.read(1) != b"\n"   # an earlier write cut short: start a new line after it
            f.write((b"\n" if cut_short else b"") + line.encode("utf-8") + b"\n")
            f.flush()
            os.fsync(f.fileno())

    def _records(self) -> list[dict]:
        """This task's lines. Lines of other tasks and unreadable lines (a write cut short, so its attempt was never
        sent) are skipped."""
        out = []
        lines = self.path.read_text(encoding="utf-8").splitlines() if self.path.exists() else []
        for line in lines:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict) and r.get("task") == self.task:
                out.append(r)
        return out

    def by_run(self) -> dict[str, float]:
        """Spend per run key: each request's settlement, or its reservations if it has none, plus carried spend."""
        reserved, settled, run_of, out = defaultdict(float), {}, {}, defaultdict(float)
        for r in self._records():
            try:
                usd = float(r["usd"])
            except (ValueError, KeyError, TypeError):
                continue
            if not math.isfinite(usd) or usd < 0:
                continue
            run, op, req = str(r.get("run", "")), r.get("op"), str(r.get("request", ""))
            if op == "carry":
                out[run] += usd
            elif op == "reserve":
                reserved[req] += usd
                run_of.setdefault(req, run)
            elif op == "settle":
                settled[req] = settled.get(req, 0.0) + usd
                run_of.setdefault(req, run)
        for req, run in run_of.items():
            out[run] += settled[req] if req in settled else reserved[req]
        return dict(out)

    def total(self) -> float:
        return sum(self.by_run().values())

    def overruns(self) -> list[dict]:
        """The overrun lines: requests that cost more than their worst case (`usd` against `worst_usd`), each with
        the harness version it was recorded under."""
        return [r for r in self._records() if r.get("op") == "overrun"]

    @contextlib.contextmanager
    def lock(self):
        """Hold the task's paid-run lock, so two runs never start from the same recorded spend. Raises
        PaidRunInProgress at once if another process (or another holder in this one) has it."""
        import fcntl

        lock_path = self.path.with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise PaidRunInProgress(f"another paid run of {self.task} is in progress ({lock_path})") from None
            try:
                yield self
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def image_check_cap_usd(asked: float | None = None) -> float:
    """The ceiling for an image task's paid check: IMAGE_CHECK_CAP_USD, or a lower one the caller asks for (0: no
    paid call). Anything above it, negative or not a number is refused: more needs the owner's approval and a change
    of the SOP."""
    if asked is None:
        return IMAGE_CHECK_CAP_USD
    if not 0 <= asked <= IMAGE_CHECK_CAP_USD:
        raise ValueError(f"spend cap ${asked} is not between $0 and the image-task check cap "
                         f"${IMAGE_CHECK_CAP_USD:.2f} (SOP)")
    return float(asked)
