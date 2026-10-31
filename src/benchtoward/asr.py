"""OpenAI-compatible speech-to-text client (`POST {base}/audio/transcriptions`, multipart) with retries.

Audio goes only to routes marked `audio: true` in registry/providers.yaml, i.e. the deployment's own speech service:
the K-OR speech audio is never sent to a third-party API (the dataset forbids TTS and voice-cloning training, and
third parties may train on inputs).
"""
import asyncio
import math
from dataclasses import dataclass

import httpx

from benchtoward.config import Settings
from benchtoward.db import Model
from benchtoward.llm import MissingCredential

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}  # 409: the serving pod is not ready yet
ACCOUNT_STATUS = {401, 402, 403, 404}  # token, credits, scope or model: the same for every clip, so stop the run
DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=60.0, pool=60.0)
MAX_READ_TIMEOUT_RETRIES = 2
MAX_RETRY_AFTER_S = 60.0


@dataclass
class Transcription:
    text: str
    audio_seconds: float
    raw: str | None  # response body, kept so the transcript can be re-read
    error: str | None
    transient: bool = False  # True: worth retrying later (5xx, 429, network, account); False: final for this clip
    fatal: bool = False  # account- or model-level failure: stop sending the remaining clips of the run


class _BadResponse(Exception):
    pass


class ASRClient:
    """Never raises for HTTP or payload errors. Use `async with` to share one connection across many clips."""

    def __init__(self, base_url: str, api_key: str, *, transport: httpx.AsyncBaseTransport | None = None,
                 max_retries: int = 4, timeout: httpx.Timeout = DEFAULT_TIMEOUT, sleep=asyncio.sleep):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.transport = transport
        self.max_retries = max_retries
        self.timeout = timeout
        self.sleep = sleep
        self._http: httpx.AsyncClient | None = None

    def _new_http(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self.transport, timeout=self.timeout)

    async def __aenter__(self) -> "ASRClient":
        self._http = self._new_http()
        return self

    async def __aexit__(self, *exc) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def transcribe(self, model_id: str, audio: bytes, filename: str, *, language: str | None = "ko",
                         temperature: float = 0.0) -> Transcription:
        if self._http is not None:
            return await self._transcribe(self._http, model_id, audio, filename, language, temperature)
        async with self._new_http() as http:
            return await self._transcribe(http, model_id, audio, filename, language, temperature)

    async def _transcribe(self, http: httpx.AsyncClient, model_id: str, audio: bytes, filename: str,
                          language: str | None, temperature: float) -> Transcription:
        data = {"model": model_id, "response_format": "json", "temperature": f"{temperature:g}"}
        if language:
            data["language"] = language
        headers = {"Authorization": f"Bearer {self.api_key}"}
        error, transient, raw, wait, read_timeouts = "unknown", True, None, None, 0
        for attempt in range(self.max_retries + 1):
            if attempt:
                await self.sleep(wait if wait is not None else 2 ** (attempt - 1))
            wait = None
            try:
                response = await http.post(f"{self.base_url}/audio/transcriptions", data=data, headers=headers,
                                           files={"file": (filename, audio, "audio/wav")})
            except httpx.ReadTimeout:
                error, transient = "transport_ReadTimeout", True
                read_timeouts += 1
                if read_timeouts > MAX_READ_TIMEOUT_RETRIES:
                    break
                continue
            except httpx.TransportError as exc:
                error, transient = f"transport_{type(exc).__name__}", True
                continue
            raw = response.text[:20000] or None
            if response.status_code == 200:
                try:
                    return _parse(response)
                except _BadResponse:
                    error, transient = "bad_response", True
                    continue
            error = _error_code(response)
            if response.status_code in ACCOUNT_STATUS:
                return Transcription(text="", audio_seconds=0, raw=raw, error=error, transient=True, fatal=True)
            transient = response.status_code in RETRY_STATUS
            if not transient:
                break
            wait = _retry_after(response)
        return Transcription(text="", audio_seconds=0, raw=raw, error=error, transient=transient)


def _parse(response: httpx.Response) -> Transcription:
    try:
        body = response.json()
    except ValueError as exc:
        raise _BadResponse from exc
    if not isinstance(body, dict) or not isinstance(body.get("text"), str):
        raise _BadResponse
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    try:
        seconds = float(usage.get("seconds") or 0)
    except (TypeError, ValueError):
        seconds = 0.0
    return Transcription(text=body["text"], audio_seconds=seconds, raw=response.text[:20000], error=None)


def _error_code(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict):
        code = body.get("error")
        if isinstance(code, dict):
            code = code.get("code") or code.get("type")
        if isinstance(code, str) and code:
            return code
    return f"http_{response.status_code}"


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return min(float(response.headers["retry-after"]), MAX_RETRY_AFTER_S)
    except (KeyError, ValueError):
        return None


def asr_client_for(model: Model, settings: Settings) -> ASRClient:
    if model.provider not in settings.audio_providers or model.provider not in settings.providers:
        raise ValueError(f"{model.slug}: speech audio is only sent to routes marked `audio: true` in "
                         "registry/providers.yaml, never to a third-party API")
    base, key = settings.providers[model.provider]
    if not key:
        raise MissingCredential(f"no API key for provider {model.provider}")
    return ASRClient(base, key)


def billed_minutes(seconds: float) -> int:
    """Started audio minutes of one request (the speech service bills each request by the minute, rounded up)."""
    return math.ceil(seconds / 60) if seconds > 0 else 0


def audio_cost_usd(model: Model, seconds_per_request) -> float:
    price = model.price_usd_per_audio_min or 0.0
    return sum(billed_minutes(s) for s in seconds_per_request) * price
