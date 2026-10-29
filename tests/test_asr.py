import asyncio

import httpx
import pytest

from medbench.asr import ASRClient, asr_client_for, audio_cost_usd, billed_minutes
from medbench.config import Settings
from medbench.db import Model
from medbench.llm import MissingCredential


class Sleeps(list):
    async def __call__(self, seconds):
        self.append(seconds)


def _ok(text="석션 주세요.", seconds=3):
    return httpx.Response(200, json={"text": text, "usage": {"type": "duration", "seconds": seconds}})


def _client(responses, requests, sleeps=None, **kw):
    def handler(request: httpx.Request):
        requests.append(request)
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    return ASRClient("https://asr.test/v1", "tok", transport=httpx.MockTransport(handler),
                     sleep=sleeps if sleeps is not None else Sleeps(), **kw)


def _transcribe(client, **kw):
    return asyncio.run(client.transcribe("qwen3-asr-1.7b", b"RIFFdata", "U1__chul.wav", **kw))


def test_sends_multipart_request_and_parses_text_and_duration():
    requests = []
    r = _transcribe(_client([_ok()], requests))
    assert (r.text, r.audio_seconds, r.error, r.transient, r.fatal) == ("석션 주세요.", 3, None, False, False)
    assert '"text"' in r.raw
    [req] = requests
    assert req.method == "POST" and str(req.url) == "https://asr.test/v1/audio/transcriptions"
    assert req.headers["authorization"] == "Bearer tok"
    assert req.headers["content-type"].startswith("multipart/form-data")
    body = req.content
    for name, value in (("model", b"qwen3-asr-1.7b"), ("language", b"ko"), ("response_format", b"json"),
                        ("temperature", b"0")):
        assert f'name="{name}"\r\n\r\n'.encode() + value + b"\r\n" in body
    assert b'name="file"; filename="U1__chul.wav"' in body and b"RIFFdata" in body


def test_language_can_be_omitted():
    requests = []
    _transcribe(_client([_ok()], requests), language=None)
    assert b'name="language"' not in requests[0].content


def test_retries_5xx_then_succeeds():
    requests, sleeps = [], Sleeps()
    r = _transcribe(_client([httpx.Response(503, text="busy"), httpx.Response(502), _ok()], requests, sleeps))
    assert len(requests) == 3 and r.error is None and sleeps == [1, 2]


def test_429_honours_retry_after_with_a_cap():
    requests, sleeps = [], Sleeps()
    responses = [httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(429, headers={"Retry-After": "900"}),
                 _ok()]
    r = _transcribe(_client(responses, requests, sleeps))
    assert r.error is None and sleeps == [7, 60]


def test_gives_up_after_max_retries_and_stays_transient():
    requests = []
    r = _transcribe(_client([httpx.Response(503)] * 5, requests))
    assert len(requests) == 5 and r.error == "http_503" and r.transient is True and r.fatal is False


def test_bad_request_is_final_for_the_clip():
    requests = []
    r = _transcribe(_client([httpx.Response(400, json={"detail": "could not decode audio"})], requests))
    assert len(requests) == 1 and r.error == "http_400" and r.transient is False and r.fatal is False
    assert "could not decode audio" in r.raw


def test_account_errors_stop_without_retrying():
    for status, body, code in ((402, {"error": "insufficient_credits"}, "insufficient_credits"),
                               (401, {"error": {"code": "token_revoked"}}, "token_revoked"),
                               (404, {"error": "model_not_found"}, "model_not_found")):
        requests = []
        r = _transcribe(_client([httpx.Response(status, json=body)], requests))
        assert len(requests) == 1 and r.error == code and r.transient is True and r.fatal is True


def test_malformed_body_is_retried():
    requests = []
    responses = [httpx.Response(200, text="not json"), httpx.Response(200, json={"usage": {}}), _ok("네")]
    r = _transcribe(_client(responses, requests))
    assert len(requests) == 3 and r.text == "네"
    r = _transcribe(_client([httpx.Response(200, text="<html>")] * 5, []))
    assert r.error == "bad_response" and r.transient is True


def test_read_timeouts_are_retried_twice_at_most():
    requests = []
    timeout = httpx.ReadTimeout("slow")
    r = _transcribe(_client([timeout, timeout, timeout, _ok()], requests))
    assert len(requests) == 3 and r.error == "transport_ReadTimeout" and r.transient is True


def test_shared_connection_inside_context_manager():
    requests = []
    client = _client([_ok("a"), _ok("b")], requests)

    async def both():
        async with client:
            return [await client.transcribe("m", b"x", "a.wav"), await client.transcribe("m", b"x", "b.wav")]

    assert [r.text for r in asyncio.run(both())] == ["a", "b"]


def test_audio_goes_only_to_routes_marked_for_audio():
    """The K-OR audio stays with the deployment's own speech service: routes need `audio: true`."""
    own = Model(slug="alibaba/qwen3-asr-1.7b", name="q", type="asr", provider="lab",
                provider_model_id="qwen3-asr-1.7b", digest="alibaba/qwen3-asr-1.7b")
    routes = {"lab": ("https://asr.example/v1", "t"), "cloud": ("https://cloud.example/v1", "k")}
    client = asr_client_for(own, Settings(providers=routes, audio_providers=("lab",)))
    assert client.base_url == "https://asr.example/v1"
    with pytest.raises(MissingCredential):
        asr_client_for(own, Settings(providers={"lab": ("https://asr.example/v1", None)}, audio_providers=("lab",)))
    other = Model(slug="openai/whisper", name="w", type="asr", provider="cloud", provider_model_id="w",
                  digest="openai/whisper")
    with pytest.raises(ValueError, match="audio"):
        asr_client_for(other, Settings(providers=routes, audio_providers=("lab",)))


def test_billing_by_started_audio_minute():
    assert [billed_minutes(s) for s in (0, 1, 3, 60, 61, 125)] == [0, 1, 1, 1, 2, 3]
    model = Model(slug="a/b", name="b", type="asr", provider="lab", digest="a/b", price_usd_per_audio_min=0.001)
    assert audio_cost_usd(model, [3, 3, 61]) == pytest.approx(0.004)
    free = Model(slug="a/c", name="c", type="asr", provider="lab", digest="a/c")
    assert audio_cost_usd(free, [3]) == 0.0
