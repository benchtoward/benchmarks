import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx

from benchtoward.config import Settings
from benchtoward.db import Model

RETRY_STATUS = {429, 500, 502, 503, 504}
UNSENT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout)  # the request never reached the route
# Non-streaming completions of up to 16K tokens can take minutes; a slow answer is not a dead connection.
DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=900.0, write=60.0, pool=60.0)
MAX_READ_TIMEOUT_RETRIES = 1  # each retry regenerates the whole answer, so retry a timeout only once


@dataclass
class ChatResult:
    content: str
    reasoning: str | None
    tokens_in: int
    tokens_out: int
    error: str | None
    transient: bool = False  # True: worth retrying later (5xx, 429, network, malformed); False: final for this item
    cost_usd: float | None = None  # what the route billed, when its response says (usage.cost)
    attempts: int = 1  # requests sent, retries included
    failed_attempts: int = 0  # sent requests that failed but may have been billed (timeouts, 5xx, malformed 200s)
    cached_tokens: int = 0  # prompt tokens the route served from its prompt cache, when it says
    reasoning_tokens: int = 0  # completion tokens spent on thinking, when the route says
    finish_reason: str | None = None  # e.g. "stop" or "length" (cut at max_tokens), when the route says


class MissingCredential(Exception):
    pass


class _BadResponse(Exception):
    pass


class LLMClient:
    """Minimal OpenAI-compatible chat client. Never raises for HTTP or payload errors."""

    def __init__(self, base_url: str, api_key: str, *, transport: httpx.AsyncBaseTransport | None = None,
                 max_retries: int = 4, timeout: httpx.Timeout = DEFAULT_TIMEOUT, sleep=asyncio.sleep):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.transport = transport
        self.max_retries = max_retries
        self.timeout = timeout
        self.sleep = sleep

    async def chat(self, model_id: str, messages: list[dict], *, max_tokens: int,
                   temperature: float = 0.0, extra: dict | None = None,
                   before_attempt: Callable[[int], Awaitable[bool]] | None = None) -> ChatResult:
        """`extra`: further request fields a task's protocol needs (e.g. response_format); they never replace the
        model, messages, max_tokens or temperature. `before_attempt(n)` is awaited before request n (0 is the
        first, then each retry); when it returns False nothing more is sent and the error is 'budget'. The result
        counts the requests sent and the failed ones that may have been billed (read timeouts and other transport
        errors after sending, 5xx answers, malformed 200s; not 429 or 4xx refusals, nor refused connections), so a
        budget can charge retries too."""
        payload = {**(extra or {}), "model": model_id, "messages": messages, "max_tokens": max_tokens,
                   "temperature": temperature}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        error, transient, read_timeouts = "unknown", True, 0
        sent = billed = 0
        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout) as http:
            for attempt in range(self.max_retries + 1):
                if attempt:
                    await self.sleep(2 ** (attempt - 1))
                if before_attempt is not None and not await before_attempt(attempt):
                    error, transient = "budget", True
                    break
                sent += 1
                try:
                    response = await http.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
                except httpx.ReadTimeout:
                    error, transient = "transport_ReadTimeout", True
                    billed += 1
                    read_timeouts += 1
                    if read_timeouts > MAX_READ_TIMEOUT_RETRIES:
                        break
                    continue
                except httpx.TransportError as exc:
                    error, transient = f"transport_{type(exc).__name__}", True
                    billed += not isinstance(exc, UNSENT_ERRORS)
                    continue
                if response.status_code == 200:
                    try:
                        result = _parse(response)
                    except _BadResponse:
                        error, transient = "bad_response", True
                        billed += 1
                        continue
                    result.attempts, result.failed_attempts = sent, billed
                    return result
                error = _error_code(response)
                transient = response.status_code in RETRY_STATUS
                billed += response.status_code >= 500
                if not transient:
                    break
        return ChatResult(content="", reasoning=None, tokens_in=0, tokens_out=0, error=error, transient=transient,
                          attempts=sent, failed_attempts=billed)


def _text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def _count(details, key: str) -> int:
    value = details.get(key) if isinstance(details, dict) else None
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def _parse(response: httpx.Response) -> ChatResult:
    try:
        body = response.json()
    except ValueError as exc:
        raise _BadResponse from exc
    if not isinstance(body, dict) or body.get("error") or not isinstance(body.get("choices"), list) \
            or not body["choices"] or not isinstance(body["choices"][0], dict) \
            or not isinstance(body["choices"][0].get("message"), dict):
        raise _BadResponse
    message = body["choices"][0]["message"]
    finish = body["choices"][0].get("finish_reason")
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    cost = usage.get("cost")
    return ChatResult(
        content=_text(message.get("content")),
        # some servers name the field reasoning_content, others (e.g. for gpt-oss) name it reasoning
        reasoning=_text(message.get("reasoning_content")) or _text(message.get("reasoning")) or None,
        tokens_in=int(usage.get("prompt_tokens") or 0),
        tokens_out=int(usage.get("completion_tokens") or 0),
        error=None,
        cost_usd=float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else None,
        cached_tokens=_count(usage.get("prompt_tokens_details"), "cached_tokens"),
        reasoning_tokens=_count(usage.get("completion_tokens_details"), "reasoning_tokens"),
        finish_reason=finish if isinstance(finish, str) else None,
    )


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


def client_for(model: Model, settings: Settings) -> LLMClient:
    route = settings.providers.get(model.provider)
    if route is None:
        raise ValueError(f"model {model.slug} is not served by an LLM API (no route for provider {model.provider})")
    base, key = route
    if not key:
        raise MissingCredential(f"no API key for provider {model.provider}")
    return LLMClient(base, key)


def cost_usd(model: Model, tokens_in: int, tokens_out: int) -> float:
    price_in = model.price_in_usd_per_m or 0.0
    price_out = model.price_out_usd_per_m or 0.0
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000
