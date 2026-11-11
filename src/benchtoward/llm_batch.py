"""Durable, offline-configured LLM route batches for independent frontier requests.

API contract: the lead's supplied batch notes (2026-11-05).
No capability field in the models API is documented there, so the operator's
registry/llm-batch.yaml is the source of batch support. No speculative API fields.
"""
import asyncio
import fcntl
import hashlib
import json
import logging
import math
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from enum import Enum
from pathlib import Path

import httpx
import yaml

from benchtoward.config import registry_dir
from benchtoward.llm import ChatResult, LLMClient, _BadResponse, _parse

log = logging.getLogger(__name__)
ENDPOINT = "/v1/chat/completions"
FINAL = {"completed", "failed", "expired", "cancelled"}
RUNNING = {"validating", "in_progress", "finalizing"}
# The batch contract guarantees usage accounting without the sync usage.include opt-in.
BASE_FIELDS = {"model", "messages", "max_tokens", "temperature"}
UNSUPPORTED = {"stream", "speed", "deferred_tools", "plugins", "audio", "modalities", "prediction"}


class BatchError(RuntimeError):
    """A paid job must stop; never include provider error bodies or credentials."""


class BatchMode(str, Enum):
    auto = "auto"
    sync = "sync"
    batch = "batch"


class RetryMode(str, Enum):
    sync = "sync"
    batch = "batch"


def load_registry() -> dict:
    path = registry_dir() / "llm-batch.yaml"
    return (yaml.safe_load(path.read_text()) or {}) if path.exists() else {"models": {}}


def _save(path: Path, state: dict) -> None:
    """Atomic and durable before polling, including when the process is killed."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _cost(value):
    return float(value) if isinstance(value, (float, int)) and not isinstance(value, bool) \
        and math.isfinite(value) and value >= 0 else None


def _batch_result(body):
    """Validate text content before the synchronous parser can coerce other types."""
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise _BadResponse from None
    if not isinstance(content, str):
        if not isinstance(content, list) or not all(
            isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
            for part in content
        ):
            raise _BadResponse
    return _parse(httpx.Response(200, json=body))


def batch_spend(root: Path) -> float:
    return sum(state["charged_usd"] for _, state in batch_states(root))


def batch_states(root: Path):
    for path in sorted(Path(root).rglob(".llm-batches/*.json")) if Path(root).exists() else []:
        yield path, json.loads(path.read_text())


def charge_states(root: Path):
    for path in sorted(Path(root).rglob(".llm-charges/*.json")) if Path(root).exists() else []:
        yield path, json.loads(path.read_text())


@asynccontextmanager
async def _locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(0.01)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


async def _durable(coro):
    """Finish bookkeeping before propagating cancellation, including Runner's SIGINT cancellation."""
    task = asyncio.create_task(coro)
    cancelled = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancelled = exc
    result = task.result()
    if cancelled is not None:
        raise cancelled
    return result


class BatchBackend:
    """Coalesce one wave of uncached calls by (model, endpoint), before concurrency slots.

    The backend owns paid calls and durable billing; callbacks only cache answers.
    Request locks cover cache rechecks and resolution, batch locks cover polling,
    and the budget lock covers only reservation/submission and charge writes.
    Existing batch IDs always take precedence over mode and threshold.
    A failed batch can retry once as a batch;
    subsequent failures use the synchronous path with its normal attempt budget.
    """

    def __init__(self, clients, prices, cache_dir, budget, *, mode="auto", retry="sync", registry=None,
                 now=time.time):
        if mode not in {"auto", "sync", "batch"}:
            raise ValueError("mode must be auto, sync or batch")
        if retry not in {"sync", "batch"}:
            raise ValueError("batch retry must be sync or batch")
        self.clients, self.prices, self.budget = clients, prices, budget
        self.root = Path(cache_dir) / ".llm-batches"
        self.mode, self.retry = mode, retry
        self.registry = load_registry() if registry is None else registry
        self.now = now
        self.pending = []
        self.task = None
        self.refused_models = set()
        self.inflight = {}
        self.waiters = {}
        self.dispatches = {}
        if budget.ledger_root is None:
            budget.ledger_root = Path(cache_dir).resolve()

    async def request(self, model, body, identity, *, resolve=None, cached=None, parse=None,
                      sync_body=None, semaphore=None):
        # identity is the caller's cache address and logical prompt hash; the wire
        # body is also hashed so changing request parameters cannot replay another call.
        packed = json.dumps([model, ENDPOINT, identity, body], sort_keys=True, ensure_ascii=False)
        rid = hashlib.sha256(packed.encode()).hexdigest()
        task = self.inflight.get(rid)
        if task is None:
            request = {"custom_id": rid, "body": body, "identity": identity}
            task = asyncio.create_task(self._request(model, request, resolve, cached, parse, sync_body, semaphore))
            self.inflight[rid] = task
            self.waiters[task] = 0
        self.waiters[task] += 1
        try:
            return await asyncio.shield(task)
        finally:
            self.waiters[task] -= 1
            if not self.waiters[task]:
                self.inflight.pop(rid, None)
                self.waiters.pop(task, None)
                if not task.done():
                    task.cancel()
                    # Keep caller-owned cache handles open until paid-call cleanup finishes.
                    try:
                        await _durable(self._join(task))
                    except asyncio.CancelledError:
                        pass

    @staticmethod
    async def _join(task):
        await asyncio.gather(task, return_exceptions=True)

    async def _request(self, model, request, resolve, cached, parse, sync_body, semaphore):
        rid = request["custom_id"]
        address = hashlib.sha256(f"{self.root.resolve()}:{rid}".encode()).hexdigest()
        async with _locked(self.budget.ledger_root / ".llm-requests" / f"{address}.lock"):
            hit = cached() if cached else None
            if hit is not None:
                return hit
            future = asyncio.get_running_loop().create_future()
            self.pending.append((request, model, future, parse))
            if self.task is None:
                self.task = asyncio.create_task(self._drain())
            try:
                result = await future
            except BaseException:
                work = self.dispatches.get(future)
                if work is not None:
                    task, futures = work
                    if all(f.done() for f in futures):
                        task.cancel()
                        # Cancel sleeps/polls promptly, but let a known terminal
                        # charge finish before the last caller exits its context.
                        await _durable(self._join(task))
                raise
            if result is None:
                if semaphore is None:
                    result = await self._sync(model, request, sync_body or request["body"], resolve, parse)
                else:
                    async with semaphore:
                        result = await self._sync(model, request, sync_body or request["body"], resolve, parse)
            return await resolve(result) if resolve else result

    @asynccontextmanager
    async def _budget_lock(self):
        async with _locked(self.budget.ledger_root / ".llm-budget.lock"):
            self.budget.spent_usd = self.budget.spent_in(self.budget.ledger_root)
            yield

    async def _sync(self, model, request, body, resolve, parse):
        from benchtoward.tasks.text2sql_llm_fix import worst_case_usd

        # Replay a paid success whose caller cache write was interrupted.
        saved_charges = sorted(charge_states(self.root.parent), key=lambda entry: entry[0].stat().st_mtime_ns,
                               reverse=True)
        for _, saved in saved_charges:
            if saved.get("request_id") == request["custom_id"] and saved.get("result"):
                result = ChatResult(**saved["result"])
                if not result.error:
                    # The caller parser decides whether an old unparsed response is reusable.
                    try:
                        accepted = parse is None or parse(result.content) is not None
                    except (ValueError, TypeError):
                        accepted = False
                    if accepted:
                        return result
        worst = worst_case_usd(body["messages"], body["max_tokens"], self.prices.get(model, (0, 0)))
        folder = self.root.parent / ".llm-charges"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{uuid.uuid4().hex}.json"
        state = {"model": model, "request_id": request["custom_id"], "charged_usd": 0.0,
                 "attempts": 0, "cost_unknown": True, "attempt_costs": [], "known_costs": {}}

        async def reserve():
            async with self._budget_lock():
                if not self.budget.try_reserve(worst):
                    return False
                state["attempts"] += 1
                state["charged_usd"] += worst
                state["attempt_costs"].append(worst)
                try:
                    _save(path, state)
                except BaseException:
                    self.budget.settle(worst, 0)
                    raise
                self.budget.settle(worst, worst)
                return True

        async def before_attempt(n):
            return True if n == 0 else await _durable(reserve())

        async def record_usage(n, cost):
            if _cost(cost) is None:
                return
            async with self._budget_lock():
                # A zero report on an attempt whose outcome is not settled yet (a 503 or an empty 200 can report 0)
                # keeps that attempt's reservation; only the final settlement may release it (review of #151).
                if cost > 0:
                    state["attempt_costs"][n] = cost
                state["known_costs"][str(n)] = cost
                self._charge(path, state, sum(state["attempt_costs"]))

        async def on_usage(n, cost):
            await _durable(record_usage(n, cost))

        if not await _durable(reserve()):
            return ChatResult("", None, 0, 0, "budget", attempts=0)
        extra = {k: v for k, v in body.items() if k not in BASE_FIELDS}
        try:
            result = await self.clients[model].chat(model, body["messages"], max_tokens=body["max_tokens"],
                                                    temperature=body.get("temperature", 0.0),
                                                    before_attempt=before_attempt,
                                                    **({"on_usage": on_usage} if isinstance(self.clients[model], LLMClient)
                                                       else {}),
                                                    **({"extra": extra} if extra else {}))
        except BaseException as exc:
            # The reservations are already fsynced, even if KeyboardInterrupt stops the loop.
            result = ChatResult("", None, 0, 0, f"exception_{type(exc).__name__}", cost_usd=0.0,
                                attempts=state["attempts"], failed_attempts=state["attempts"],
                                charge_id=path.stem, failed_attempts_usd=state["charged_usd"])
            await _durable(self._settle_sync(path, state, result, worst, unknown=True))
            if resolve:
                await resolve(result)
            raise
        result.charge_id = path.stem
        await _durable(self._settle_sync(path, state, result, worst))
        return result

    async def _settle_sync(self, path, state, result, worst, *, unknown=False):
        price = self.prices.get(state["model"], (0, 0))
        route_cost = _cost(result.cost_usd)
        estimate = (result.tokens_in * price[0] + result.tokens_out * price[1]) / 1e6
        # Without a route-reported cost the final answer keeps its reservation (worst case), never zero
        # (review of #151: an answer without usage refunded a paid call).
        if result.error is not None:
            cost = 0.0   # no answer: the final attempt is settled with the failed attempts below
        else:
            reported_tokens = bool(result.tokens_in or result.tokens_out)
            cost = route_cost if route_cost is not None else (estimate if reported_tokens else worst)
        # Failed attempts: the client counts those that may have been billed (result.failed_attempts; a 429 is not
        # one). Reported costs settle them, but a zero report (an unbilled 429) never stands in for a billable
        # attempt, so each billable attempt without a positive report keeps its worst-case reservation (review of #151).
        known_failed = [cost for n, cost in state["known_costs"].items()
                        if int(n) < state["attempts"] - (result.error is None)]
        billed_known = [cost for cost in known_failed if cost > 0]
        missing = max(0, result.failed_attempts - len(billed_known))
        result.failed_attempts_usd = state["charged_usd"] if unknown else sum(known_failed) + worst * missing
        async with self._budget_lock():
            state.update(cost_usd=cost, cost_source="route" if route_cost is not None else ("estimate" if (result.tokens_in or result.tokens_out) else "reservation"),
                         failed_attempts_usd=result.failed_attempts_usd,
                         cost_unknown=bool(unknown or missing or (result.error is None and route_cost is None
                                                                     and not (result.tokens_in or result.tokens_out))),
                         result=asdict(result))
            self._charge(path, state, cost + result.failed_attempts_usd)

    def _eligible(self, model, body):
        client = self.clients[model]
        if not getattr(client, "batch_enabled", False):
            return False, "route has no verified batch support"
        models = self.registry.get("models") or {}
        if model in self.refused_models:
            return False, "batch support refused by the API in this run"
        if model not in models:
            return False, "model absent from registry/llm-batch.yaml"
        if ":online" in model:
            return False, "web search is unsupported"
        allowed = set((models[model] or {}).get("allow_unverified_params", []))
        if set(body) & UNSUPPORTED or set(body) - BASE_FIELDS - allowed:
            return False, "request parameters are unsupported or unverified"
        for message in body["messages"]:
            content = message.get("content")
            if not isinstance(content, str):
                return False, "non-text input is unverified"
            if set(message) - {"role", "content"} or message.get("role") not in {"system", "user", "assistant"}:
                return False, "tool messages or message parameters are unverified"
        return True, ""

    async def _drain(self):
        # asyncio.gather creates all independent callers before this next turn.
        await asyncio.sleep(0)
        pending, self.pending = [p for p in self.pending if not p[2].done()], []
        # Later waves may dispatch while this one's batches are polling.
        self.task = None
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            results = await self._dispatch(pending)
            for request, _, future, _ in pending:
                if not future.done():
                    future.set_result(results.get(request["custom_id"]))
        except BaseException as exc:
            for _, _, future, _ in pending:
                if not future.done():
                    future.set_exception(exc)

    def _load(self, path):
        return json.loads(path.read_text())

    async def _dispatch(self, pending):
        results, fresh = {}, []
        interested = {r["custom_id"]: (future, parse) for r, _, future, parse in pending}
        # The latest state wins after a configured follow-up. Completed payloads
        # stay here so cache writes interrupted by a restart can simply replay.
        known = {}
        for path in sorted(self.root.glob("*.json"), key=lambda p: p.stat().st_mtime_ns):
            state = self._load(path)
            for request in state["requests"]:
                known[request["custom_id"]] = (path, state)
        resume = {}
        for request, model, _, _ in pending:
            rid = request["custom_id"]
            if rid in known:
                path, state = known[rid]
                resume[str(path)] = (path, state)
            else:
                fresh.append((request, model))
        groups, fallbacks = {}, set()
        for request, model in fresh:
            eligible, reason = self._eligible(model, request["body"])
            if self.mode != "sync" and not eligible and (model, reason) not in fallbacks:
                log.warning("LLM route model=%s synchronous fallback: %s", model, reason)
                fallbacks.add((model, reason))
            if self.mode != "sync" and eligible and (self.mode != "auto" or len(pending) >= 100):
                groups.setdefault(model, []).append(request)
            else:
                future = interested[request["custom_id"]][0]
                if not future.done():
                    future.set_result(None)
        submissions = [(self._resume(path, interested),
                        [r["custom_id"] for r in state["requests"] if r["custom_id"] in interested])
                       for path, state in resume.values()]
        for model, requests in groups.items():
            requests = [r for r in requests if not interested[r["custom_id"]][0].done()]
            if not requests:
                continue
            submissions.append((self._submit(model, requests, interested), [r["custom_id"] for r in requests]))
        # The models may each take hours: submit independent groups together.
        # Each coroutine reserves its estimate and fsyncs its intent before POST.
        async def deliver(submission, ids):
            try:
                answers = await submission
                results.update(answers)
                for rid in ids:
                    future = interested[rid][0]
                    if not future.done():
                        future.set_result(answers.get(rid))
            except BaseException as exc:
                for rid in ids:
                    future = interested[rid][0]
                    if not future.done():
                        future.set_exception(exc)
            finally:
                for rid in ids:
                    self.dispatches.pop(interested[rid][0], None)

        tasks = []
        for submission, ids in submissions:
            task = asyncio.create_task(deliver(submission, ids))
            tasks.append(task)
            futures = [interested[rid][0] for rid in ids]
            for future in futures:
                self.dispatches[future] = (task, futures)
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return results

    async def _resume(self, path, interested):
        # Every process re-reads after acquiring the batch's poll lock. A waiter
        # sees the owner's terminal payload without issuing another GET.
        async with _locked(path.with_suffix(".lock")):
            state = self._load(path)
            await self._wait(path, state)
            return await self._answers(path, state, interested)

    def _estimate(self, model, requests):
        # Use the sync price unless the registry supplies a verified batch price:
        # the batch discount varies by model and cannot be assumed for the cap.
        entry = (self.registry.get("models") or {}).get(model) or {}
        price = entry.get("price_per_m")
        if price is None:
            sync = self.prices.get(model)
            if sync is None or len(sync) != 2 or any(_cost(p) is None for p in sync):
                raise BatchError("Missing or invalid model price; refusing paid submission")
            price = sync
        if len(price) != 2 or any(_cost(p) is None for p in price):
            raise BatchError("Invalid registry batch price; refusing paid submission")
        total = 0.0
        for r in requests:
            body = r["body"]
            tokens = len(json.dumps(body["messages"], ensure_ascii=False).encode()) + 200
            total += (tokens * price[0] + body["max_tokens"] * price[1]) / 1e6
        if not math.isfinite(total) or total < 0:
            raise BatchError("Missing or invalid batch price; refusing paid submission")
        return total

    def _charge(self, path, state, cost):
        previous = state["charged_usd"]
        state["charged_usd"] = cost
        _save(path, state)
        self.budget.spent_usd += cost - previous
        if self.budget.spent_usd > self.budget.cap_usd:
            self.budget.stopped = True

    def _update(self, path, state, payload):
        status = payload.get("status")
        if status not in RUNNING | FINAL:
            raise BatchError("Unrecognized batch status; saved batch retained for reconciliation")
        changed = state.get("status") != status
        state.update({k: payload[k] for k in ("id", "status", "expires_at", "usage", "results", "request_counts")
                      if k in payload})
        # Only this terminal payload can finalize the charge. Earlier usage may
        # be provisional (including zero); do not retain it as terminal evidence.
        usage = payload.get("usage")
        cost = _cost(usage.get("cost")) if isinstance(usage, dict) else None
        if status in FINAL:
            state["usage"] = usage if isinstance(usage, dict) else {}
            state["cost_unknown"] = cost is None
        if status in FINAL and cost is not None:
            changed |= cost != state["charged_usd"]
            self._charge(path, state, cost)
        else:
            _save(path, state)
        if changed:
            log.warning("LLM route batch id=%s model=%s count=%d status=%s cost=$%.6f",
                        state.get("id"), state["model"], len(state["requests"]), status, state["charged_usd"])

    async def _submit(self, model, requests, interested, depth=0, parent=None):
        estimate = self._estimate(model, requests)
        path = self.root / f"{uuid.uuid4().hex}.json"
        state = {"id": None, "model": model, "endpoint": ENDPOINT, "status": "submitting",
                 "requests": requests, "charged_usd": estimate, "cost_unknown": True, "retry_depth": depth}
        if parent:
            state["parent"] = parent.name
        client = self.clients[model]
        payload = {"endpoint": ENDPOINT, "model": model, "completion_window": "24h",
                   "requests": [{"custom_id": r["custom_id"], "body": r["body"]} for r in requests]}
        async def submit_once():
            async with self._budget_lock():
                if not path.exists():
                    if not self.budget.try_reserve(estimate):
                        return None
                    try:
                        _save(path, state)
                    except BaseException:
                        self.budget.settle(estimate, 0)
                        raise
                    self.budget.settle(estimate, estimate)
                return await self._post_batch(client, path, state, payload)

        async with _locked(path.with_suffix(".lock")):
            delay = 60
            while True:
                status = await _durable(submit_once())
                if status is None:
                    return {r["custom_id"]: ChatResult("", None, 0, 0, "budget", attempts=0) for r in requests}
                if status == 429:
                    # Backoff never holds the budget lock; the durable intent remains reserved.
                    await client.sleep(delay)
                    delay = min(delay * 2, 3600)
                    continue
                if status in {400, 404}:
                    return {}
                break
            await self._wait(path, state)
            return await self._answers(path, state, interested)

    async def _post_batch(self, client, path, state, payload):
        """Called under the budget lock; persist the POST outcome before releasing it."""
        try:
            response = await self._http(client, "POST", "/batches", payload)
        except httpx.TransportError:
            raise BatchError("Batch submission outcome uncertain; reconcile saved intent with GET /batches; "
                             "set its id before resuming. No automatic resubmission") from None
        status = response.status_code
        if status == 429:
            return status
        if status in {400, 402, 404, 413}:
            state["status"] = "refused"
            self._charge(path, state, 0)
            if status == 402:
                raise BatchError("LLM route HTTP 402: insufficient balance; job stopped")
            if status in {400, 404}:
                self.refused_models.add(state["model"])
                log.warning("LLM route model=%s synchronous fallback: batch HTTP %d", state["model"], status)
                return status
            raise BatchError("LLM route batch HTTP 413: reduce request count or body size")
        if status != 202:
            raise BatchError(f"Batch submission outcome uncertain (HTTP {status}); reconcile saved intent")
        try:
            data = response.json()
            if not isinstance(data.get("id"), str) or not data["id"]:
                raise ValueError
            state["id"] = data["id"]
            _save(path, state)  # persist ID and mapping even if status is malformed
            self._update(path, state, data)
        except (ValueError, AttributeError):
            raise BatchError("Batch submission outcome uncertain; reconcile saved intent") from None
        return status

    async def _record_payload(self, path, state, payload):
        async with self._budget_lock():
            self._update(path, state, payload)

    async def _http(self, client, method, path, payload=None):
        async with httpx.AsyncClient(transport=client.transport, timeout=client.timeout) as http:
            return await http.request(method, client.base_url + path, json=payload,
                                      headers={"Authorization": f"Bearer {client.api_key}"})

    async def _wait(self, path, state):
        if state["status"] == "refused":
            return
        if not state.get("id"):
            raise BatchError("Batch submission outcome uncertain; reconcile saved intent and set its id; "
                             "automatic resubmission refused")
        client, delay = self.clients[state["model"]], 60
        while state["status"] not in FINAL:
            expiry = state.get("expires_at")
            remaining = max(0, float(expiry) - self.now()) if expiry is not None else None
            if remaining is None or remaining > 0:
                await client.sleep(min(delay, remaining) if remaining is not None else delay)
            response = await self._http(client, "GET", f"/batches/{state['id']}")
            if response.status_code == 402:
                raise BatchError("LLM route HTTP 402: insufficient balance; job stopped")
            if response.status_code == 429:
                if expiry is not None and self.now() >= float(expiry):
                    raise BatchError("Batch expires_at reached while rate limited; resume saved id later")
                delay = min(delay * 2, 3600)
                continue
            if response.status_code == 410:
                raise BatchError("Batch results no longer retained (HTTP 410); reconcile spend before retrying")
            if response.status_code != 200:
                raise BatchError(f"Batch poll HTTP {response.status_code}; resume saved id later")
            await _durable(self._record_payload(path, state, response.json()))
            delay = 60
            expiry = state.get("expires_at")
            if state["status"] not in FINAL and expiry is not None and self.now() >= float(expiry):
                raise BatchError("Batch expires_at reached but provider still running; resume saved id later; "
                                 "no resubmission")

    async def _answers(self, path, state, interested):
        if state["status"] == "refused":
            return {}
        by_id = {}
        duplicates = set()
        for row in state.get("results") or []:
            rid = row.get("custom_id")
            if rid in by_id:
                duplicates.add(rid)
            by_id[rid] = row
        answers, failed = {}, []
        for request in state["requests"]:
            rid = request["custom_id"]
            if rid in state.get("consumed", []):
                failed.append(request)
                continue
            row = by_id.get(rid) or {}
            response = row.get("response") or {}
            if response.get("status_code") == 402:
                raise BatchError("LLM route batch item HTTP 402: insufficient balance; job stopped")
            try:
                if rid in duplicates or row.get("error") or response.get("status_code") != 200:
                    raise _BadResponse
                result = _batch_result(response.get("body"))
            except (_BadResponse, ValueError, TypeError):
                failed.append(request)
                continue
            future, parse = interested.get(rid, (None, None))
            if parse is not None:
                try:
                    accepted = parse(result.content) is not None
                except (ValueError, TypeError):
                    accepted = False
                if not accepted:
                    # Persist rejection before any retry or caller cache write.
                    # A restart must never replay this unusable paid response.
                    state.setdefault("consumed", []).append(rid)
                    _save(path, state)
                    failed.append(request)
                    continue
            if _cost(result.cost_usd) is None:
                result.cost_usd = state["charged_usd"] / len(state["requests"])
            result.batch_id = state["id"]
            answers[rid] = result
        failed = [r for r in failed if r["custom_id"] in interested and
                  not interested[r["custom_id"]][0].done()]
        if failed and self.mode != "sync" and self.retry == "batch" and state["retry_depth"] == 0:
            child = next(((p, self._load(p)) for p in self.root.glob("*.json")
                          if self._load(p).get("parent") == path.name), None)
            if child:
                answers.update(await self._resume(child[0], interested))
            else:
                answers.update(await self._submit(state["model"], failed, interested, depth=1, parent=path))
        return answers
