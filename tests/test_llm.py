import asyncio
import json

import httpx
import pytest

from medbench.config import Settings
from medbench.db import Model
from medbench.llm import LLMClient, MissingCredential, client_for, cost_usd


async def _no_sleep(_seconds):
    return None


def _ok(content="B", reasoning=None, usage=True):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    body = {"choices": [{"message": message}]}
    if usage:
        body["usage"] = {"prompt_tokens": 11, "completion_tokens": 7}
    return httpx.Response(200, json=body)


def _client(responses, calls):
    def handler(request: httpx.Request):
        calls.append(json.loads(request.content))
        return responses.pop(0)

    return LLMClient("https://llm.test/v1", "tok", transport=httpx.MockTransport(handler), sleep=_no_sleep)


def _chat(client):
    return asyncio.run(client.chat("m-1", [{"role": "user", "content": "q"}], max_tokens=64))


def test_parses_content_reasoning_usage():
    calls = []
    r = _chat(_client([_ok("C", reasoning="thinking")], calls))
    assert (r.content, r.reasoning, r.tokens_in, r.tokens_out, r.error) == ("C", "thinking", 11, 7, None)
    assert calls[0]["model"] == "m-1" and calls[0]["max_tokens"] == 64 and calls[0]["temperature"] == 0.0
    assert r.cost_usd is None                                    # the route reported no cost


def test_keeps_the_cost_the_route_reports():
    body = {"choices": [{"message": {"content": "C"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "cost": 0.0123}}
    r = _chat(_client([httpx.Response(200, json=body)], []))
    assert r.cost_usd == pytest.approx(0.0123)


def test_retries_then_succeeds():
    calls = []
    responses = [httpx.Response(503, text="busy"), httpx.Response(503, text="busy"), _ok("A")]
    r = _chat(_client(responses, calls))
    assert len(calls) == 3 and r.error is None and r.content == "A"


def test_404_not_retried():
    calls = []
    r = _chat(_client([httpx.Response(404, json={"error": "model_not_found", "message": "x"})], calls))
    assert len(calls) == 1 and r.error == "model_not_found" and r.content == ""


def test_gives_up_after_max_retries():
    calls = []
    r = _chat(_client([httpx.Response(503, text="busy")] * 5, calls))
    assert len(calls) == 5 and r.error == "http_503"


def test_missing_usage_is_zero():
    r = _chat(_client([_ok("A", usage=False)], []))
    assert (r.tokens_in, r.tokens_out) == (0, 0)


def test_client_for_missing_key_raises():
    model = Model(slug="example/x", name="x", type="llm", provider="example", provider_model_id="x/y", digest="example/x")
    with pytest.raises(MissingCredential):
        client_for(model, Settings(providers={"example": ("https://llm.example/v1", None)}))


def test_client_for_uses_the_provider_table():
    model = Model(slug="example/x", name="x", type="llm", provider="example", provider_model_id="x/y", digest="example/x")
    client = client_for(model, Settings(providers={"example": ("https://llm.example/v1", "k")}))
    assert client.base_url == "https://llm.example/v1"
    with pytest.raises(ValueError):
        client_for(Model(slug="b/x", name="x", type="baseline", provider="medbench", digest="b/x"), Settings())


def test_providers_come_from_the_registry_file_and_environment(tmp_path, monkeypatch):
    from medbench.config import get_settings
    (tmp_path / "providers.yaml").write_text(
        "example: {base_url: 'https://llm.example/v1', key_env: EXAMPLE_API_KEY, base_env: EXAMPLE_API_BASE}\n"
        "other: {base_url: 'https://other.example/v1/', key_env: OTHER_KEY}\n")
    monkeypatch.setenv("MEDBENCH_REGISTRY_DIR", str(tmp_path))
    monkeypatch.setenv("EXAMPLE_API_KEY", "k1")
    monkeypatch.setenv("EXAMPLE_API_BASE", "https://override.example/v1")
    monkeypatch.delenv("OTHER_KEY", raising=False)
    assert get_settings().providers == {"example": ("https://override.example/v1", "k1"),
                                        "other": ("https://other.example/v1", None)}
    monkeypatch.setenv("MEDBENCH_REGISTRY_DIR", str(tmp_path / "missing"))
    assert get_settings().providers == {}


def test_cost_usd():
    model = Model(slug="example/m", name="m", type="llm", provider="example", provider_model_id="m", digest="example/m",
                  price_in_usd_per_m=0.12, price_out_usd_per_m=0.35)
    assert cost_usd(model, 1_000_000, 2_000_000) == pytest.approx(0.82)
    free = Model(slug="example/f", name="f", type="llm", provider="example", provider_model_id="f", digest="example/f")
    assert cost_usd(free, 10, 10) == 0.0


def test_read_timeout_retried_only_once():
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ReadTimeout("slow", request=request)

    client = LLMClient("https://llm.test/v1", "tok", transport=httpx.MockTransport(handler), sleep=_no_sleep)
    r = _chat(client)
    assert len(calls) == 2 and r.error == "transport_ReadTimeout" and r.transient is True


def test_client_read_timeout_is_long():
    client = LLMClient("https://llm.test/v1", "tok")
    assert client.timeout.read >= 900 and client.timeout.connect <= 30


def test_4xx_is_terminal_and_5xx_transient():
    r = _chat(_client([httpx.Response(400, json={"error": "context_length_exceeded"})], []))
    assert r.error == "context_length_exceeded" and r.transient is False
    r = _chat(_client([httpx.Response(503, text="busy")] * 5, []))
    assert r.transient is True


def test_malformed_200_is_retried_as_bad_response():
    calls = []
    responses = [httpx.Response(200, text="<html>oops</html>"), httpx.Response(200, json={"error": {"code": 502}}),
                 httpx.Response(200, json=[1, 2]), _ok("D")]
    r = _chat(_client(responses, calls))
    assert len(calls) == 4 and r.error is None and r.content == "D"
    r = _chat(_client([httpx.Response(200, json={"choices": [None]})] * 5, []))
    assert r.error == "bad_response" and r.transient is True


def test_list_content_is_joined():
    body = {"choices": [{"message": {"content": [{"type": "text", "text": "정답: "}, {"type": "text", "text": "C"}]}}]}
    r = _chat(_client([httpx.Response(200, json=body)], []))
    assert r.content == "정답: C"


def test_extra_request_fields_are_sent_without_overriding_the_core_ones():
    calls = []
    client = _client([_ok("{}")], calls)
    extra = {"response_format": {"type": "json_object"}, "chat_template_kwargs": {"enable_thinking": False},
             "model": "other", "max_tokens": 1}
    asyncio.run(client.chat("m-1", [{"role": "user", "content": "q"}], max_tokens=64, extra=extra))
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert calls[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert calls[0]["model"] == "m-1" and calls[0]["max_tokens"] == 64


# ---- attempts, for budgets that must count retries ----

def test_counts_attempts_and_the_failed_ones_that_may_have_been_billed():
    responses = [httpx.Response(429, text="slow down"), httpx.Response(503, text="busy"),
                 httpx.Response(200, text="<html>oops</html>"), _ok("A")]
    r = _chat(_client(responses, []))
    assert r.error is None and r.attempts == 4
    assert r.failed_attempts == 2                       # the 503 and the malformed 200; a 429 ran no model
    r = _chat(_client([httpx.Response(404, json={"error": "model_not_found"})], []))
    assert r.attempts == 1 and r.failed_attempts == 0   # refused by the route: nothing generated


def test_a_read_timeout_may_have_been_billed_but_a_refused_connection_not():
    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    r = _chat(LLMClient("https://llm.test/v1", "tok", transport=httpx.MockTransport(timeout), sleep=_no_sleep))
    assert r.attempts == 2 and r.failed_attempts == 2

    def refused(request):
        raise httpx.ConnectError("refused", request=request)

    r = _chat(LLMClient("https://llm.test/v1", "tok", transport=httpx.MockTransport(refused), sleep=_no_sleep,
                        max_retries=2))
    assert r.attempts == 3 and r.failed_attempts == 0


def test_before_attempt_is_asked_before_every_request_and_can_stop_the_retries():
    calls, asked = [], []

    async def before_attempt(n):
        asked.append(n)
        return n < 2                                    # the budget allows two requests

    client = _client([httpx.Response(503, text="busy")] * 5, calls)
    r = asyncio.run(client.chat("m-1", [{"role": "user", "content": "q"}], max_tokens=64,
                                before_attempt=before_attempt))
    assert asked == [0, 1, 2] and len(calls) == 2
    assert r.error == "budget" and r.transient is True and r.attempts == 2 and r.failed_attempts == 2


def test_cached_and_reasoning_tokens_are_kept_when_the_route_reports_them():
    body = {"choices": [{"message": {"content": "C"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 40, "prompt_tokens_details": {"cached_tokens": 80},
                      "completion_tokens_details": {"reasoning_tokens": 30}}}
    r = _chat(_client([httpx.Response(200, json=body)], []))
    assert (r.cached_tokens, r.reasoning_tokens) == (80, 30)
    r = _chat(_client([_ok("A")], []))
    assert (r.cached_tokens, r.reasoning_tokens) == (0, 0)
