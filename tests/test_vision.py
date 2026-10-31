import asyncio
import base64
import json

import httpx
import pytest

from benchtoward import vision
from benchtoward.llm import LLMClient

PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 40


async def _no_sleep(_seconds):
    return None


def _client(responses, calls):
    def handler(request: httpx.Request):
        calls.append(json.loads(request.content))
        return responses.pop(0)

    return LLMClient("https://llm.test/v1", "tok", transport=httpx.MockTransport(handler), sleep=_no_sleep)


def _ok(content="{}"):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}],
                                     "usage": {"prompt_tokens": 1000, "completion_tokens": 10}})


def test_image_part_is_a_base64_data_url():
    part = vision.image_part(PNG, "image/png")
    assert part["type"] == "image_url"
    url = part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,") and base64.b64decode(url.split(",", 1)[1]) == PNG
    with pytest.raises(ValueError):
        vision.image_part(b"")


def test_mime_from_file_name():
    assert vision.mime_for("a.jpg") == vision.mime_for("B.JPEG") == "image/jpeg"
    assert vision.mime_for("x.png") == "image/png"
    with pytest.raises(ValueError):
        vision.mime_for("x.gif")


def test_message_with_text_and_images_goes_out_unchanged():
    msg = vision.image_message("Which classes?", [(PNG, "image/png")])
    assert [p["type"] for p in msg["content"]] == ["text", "image_url"] and msg["role"] == "user"
    assert vision.count_images([msg]) == 1
    calls = []
    r = asyncio.run(_client([_ok("ok")], calls).chat("vlm", [msg], max_tokens=32))
    assert r.error is None and calls[0]["messages"] == [msg]


def test_worst_case_counts_images_not_their_base64_bytes():
    big = vision.image_message("q", [(b"y" * 3_000_000, "image/jpeg")])
    worst = vision.worst_case_usd([big], max_tokens=1000, price_in_per_m=1.0, price_out_per_m=10.0)
    tokens_in = len(json.dumps("q").encode()) + 200 + vision.IMAGE_TOKENS_MAX
    assert worst == pytest.approx((tokens_in * 1.0 + 1000 * 10.0) / 1e6)
    plain = vision.worst_case_usd([{"role": "user", "content": "q"}], 1000, 1.0, 10.0)
    assert plain < worst


def test_spend_cap_refuses_an_attempt_that_would_exceed_it():
    cap = vision.SpendCap(1.0, spent_usd=0.7)
    first, second = cap.request(0.2), cap.request(0.2)
    assert asyncio.run(first.before_attempt(0)) is True
    assert asyncio.run(second.before_attempt(0)) is False and cap.refused == 1   # 0.7 + 0.2 + 0.2 > 1.0
    assert first.finish(cost_usd=0.05, failed_attempts=0) == pytest.approx(0.05)
    assert cap.spent_usd == pytest.approx(0.75) and cap.reserved_usd == pytest.approx(0.0)
    assert asyncio.run(second.before_attempt(0)) is True                         # 0.75 + 0.2 fits now


def test_spend_cap_charges_retries_and_failures_at_their_worst_case():
    cap = vision.SpendCap(10.0)
    req = cap.request(1.0)
    calls = []
    responses = [httpx.Response(503, text="busy"), _ok("x")]
    r = asyncio.run(_client(responses, calls).chat("vlm", [{"role": "user", "content": "q"}], max_tokens=8,
                                                     before_attempt=req.before_attempt))
    assert r.error is None and len(calls) == 2 and r.failed_attempts == 1
    assert req.finish(cost_usd=0.1, failed_attempts=r.failed_attempts) == pytest.approx(1.1)
    lost = cap.request(2.0)
    asyncio.run(lost.before_attempt(0))
    assert lost.charge_all() == pytest.approx(2.0) and cap.spent_usd == pytest.approx(3.1)


def test_route_client_stops_with_budget_error_when_the_cap_is_reached():
    cap = vision.SpendCap(0.5, spent_usd=0.5)
    calls = []
    r = asyncio.run(_client([_ok()], calls).chat("vlm", [{"role": "user", "content": "q"}], max_tokens=8,
                                                  before_attempt=cap.request(0.01).before_attempt))
    assert r.error == "budget" and calls == []


def test_check_caps_follow_the_sop():
    # SOP "Checks": paid reference runs up to $5 per benchmark version; an image task's check up to $20 (owner,
    # 2026-10-30), a ceiling a caller may lower (or set to 0) but never raise
    assert vision.CHECK_CAP_USD == 5.0 and vision.IMAGE_CHECK_CAP_USD == 20.0
    assert vision.image_check_cap_usd() == 20.0
    assert vision.image_check_cap_usd(20.0) == 20.0 and vision.image_check_cap_usd(3.5) == 3.5
    assert vision.image_check_cap_usd(0) == 0.0
    for asked in (20.01, 1e9, float("inf"), float("nan"), -0.01):
        with pytest.raises(ValueError, match="image-task check cap"):
            vision.image_check_cap_usd(asked)
    with pytest.raises(ValueError):
        vision.SpendCap(float("nan"))


def test_ledger_counts_settlements_and_open_reservations(tmp_path):
    ledger = vision.SpendLedger(tmp_path / "d" / "spend.jsonl", "img@0")
    cap = vision.SpendCap(20.0, ledger=ledger, run="k1")
    done, killed = cap.request(2.0, instance="a"), cap.request(2.0, instance="b")
    assert asyncio.run(done.before_attempt(0)) and asyncio.run(killed.before_attempt(0))
    assert ledger.total() == pytest.approx(4.0)                      # both reserved before they are sent
    done.finish(cost_usd=0.5, failed_attempts=0)
    assert ledger.by_run() == {"k1": pytest.approx(2.5)}             # the killed one stays at its worst case
    other = vision.SpendLedger(ledger.path, "other@0")
    other.append(op="carry", run="k9", request="c", usd=7.0)
    with open(ledger.path, "a", encoding="utf-8") as f:
        f.write('{"task": "img@0", "op": "settle", "usd": 1')       # a write cut short: its attempt never went
    assert ledger.total() == pytest.approx(2.5) and other.total() == pytest.approx(7.0)
    ledger.append(op="carry", run="k0", request="carry-x", usd=1.25)
    assert ledger.by_run() == {"k1": pytest.approx(2.5), "k0": pytest.approx(1.25)}


def test_a_request_dearer_than_its_worst_case_stops_all_sending():
    cap = vision.SpendCap(20.0)
    req = cap.request(1.0)
    assert asyncio.run(req.before_attempt(0))
    req.finish(cost_usd=1.5, failed_attempts=0)                     # the bound the cap relies on did not hold
    assert cap.overrun and asyncio.run(cap.request(0.01).before_attempt(0)) is False


def test_the_paid_run_lock_admits_one_holder(tmp_path):
    ledger = vision.SpendLedger(tmp_path / "spend.jsonl", "img@0")
    with ledger.lock(), pytest.raises(vision.PaidRunInProgress, match="in progress"), \
            vision.SpendLedger(tmp_path / "spend.jsonl", "img@0").lock():
        pass
    with ledger.lock():                                              # released on exit
        pass


def test_an_answer_of_unknown_cost_is_charged_its_worst_case(tmp_path):
    # review of PR #104: a paid answer with no usage and no route cost was charged $0, so it could be repeated
    # without limit; an unknown cost (None, NaN, negative) is charged at the request's worst case
    ledger = vision.SpendLedger(tmp_path / "spend.jsonl", "img@0")
    cap = vision.SpendCap(20.0, ledger=ledger, run="k")
    for cost in (None, float("nan"), -1.0):
        req = cap.request(2.0)
        assert asyncio.run(req.before_attempt(0))
        assert req.finish(cost_usd=cost, failed_attempts=0) == pytest.approx(2.0)
    retried = cap.request(2.0)
    assert asyncio.run(retried.before_attempt(0)) and asyncio.run(retried.before_attempt(1))
    assert retried.finish(cost_usd=None, failed_attempts=1) == pytest.approx(4.0)
    assert cap.spent_usd == pytest.approx(10.0) and ledger.total() == pytest.approx(10.0) and not cap.overrun
    settles = [json.loads(x) for x in ledger.path.read_text().splitlines() if json.loads(x)["op"] == "settle"]
    assert len(settles) == 4 and all(s["unknown_cost"] for s in settles)
    errored = cap.request(2.0)             # a request that returned no answer (a refusal) and no failed attempt
    assert asyncio.run(errored.before_attempt(0)) and errored.finish(cost_usd=0.0, failed_attempts=0) == 0.0


def test_an_overrun_is_written_to_the_ledger_with_the_harness_version(tmp_path):
    # review of PR #104: the stop after a request dearer than its worst case lived only in memory
    ledger = vision.SpendLedger(tmp_path / "spend.jsonl", "img@0")
    cap = vision.SpendCap(20.0, ledger=ledger, run="k", harness="h1")
    req = cap.request(1.0, model="m", instance="i")
    assert asyncio.run(req.before_attempt(0))
    assert req.finish(cost_usd=1.5, failed_attempts=0) == pytest.approx(1.5) and cap.overrun
    [line] = ledger.overruns()
    assert (line["usd"], line["worst_usd"], line["harness"], line["model"], line["run"]) == (1.5, 1.0, "h1", "m", "k")
    assert ledger.total() == pytest.approx(1.5)                     # a record of the overrun, not a second charge
    assert vision.SpendLedger(ledger.path, "other@0").overruns() == []
    fine = vision.SpendCap(20.0, ledger=vision.SpendLedger(tmp_path / "b.jsonl", "img@0"), harness="h1")
    within = fine.request(1.0)
    assert asyncio.run(within.before_attempt(0)) and within.finish(cost_usd=1.0, failed_attempts=0) == 1.0
    assert fine.ledger.overruns() == [] and not fine.overrun
