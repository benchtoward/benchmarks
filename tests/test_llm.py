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
