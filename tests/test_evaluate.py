import asyncio
import json

import pandas as pd
import pytest

from conftest import REGISTRY_DIR
from medbench.db import ItemResult, Model, Run, Task
from medbench.evaluate import HARNESS_VERSION, run_llm_task
from medbench.llm import ChatResult
from medbench.registry import load_registry
from medbench.tasks.clinicalqa import QAItem


class FakeClient:
    """Answers by question text; counts calls."""

    def __init__(self, answers: dict[str, ChatResult]):
        self.answers = answers
        self.calls = 0

    async def chat(self, model_id, messages, *, max_tokens, temperature=0.0):
        self.calls += 1
        question = messages[1]["content"].split("\n")[0]
        return self.answers[question]


def ok(text):
    return ChatResult(content=text, reasoning=None, tokens_in=100, tokens_out=10, error=None)


ITEMS = [QAItem(instance_id=f"cqa-{i}", question=f"Q{i}", exam="", options={k: k.lower() for k in "ABCDE"},
                answer="A", category="Cardiology" if i < 3 else "Nephrology", chief_complaint="x")
         for i in range(5)]


@pytest.fixture
def ctx(tmp_session, tmp_path):
    load_registry(tmp_session, REGISTRY_DIR)
    model = tmp_session.get(Model, "google/medgemma-27b")
    model.price_in_usd_per_m, model.price_out_usd_per_m = 0.12, 0.35  # prices are deployment data; fixed here
    task = tmp_session.query(Task).filter_by(slug="clinicalqa").one()
    return tmp_session, model, task, tmp_path


def run(ctx, client, **kw):
    session, model, task, data_dir = ctx
    return asyncio.run(run_llm_task(session, model, task, ITEMS, client, data_dir=data_dir, **kw))


def answers(last="B"):
    return {"Q0": ok('{"answer": "A"}'), "Q1": ok("정답: A"), "Q2": ok("A"), "Q3": ok('{"answer": "A"}'),
            "Q4": ok(f'{{"answer": "{last}"}}')}


def test_metrics_and_items_stored(ctx):
    r = run(ctx, FakeClient(answers()))
    metrics = json.loads(r.metrics_json)
    assert r.status == "done" and r.tier == "baseline" and r.harness_version == HARNESS_VERSION
    assert metrics["accuracy"]["value"] == 0.8 and metrics["parse_rate"] == 1.0
    assert metrics["by_category"]["Cardiology"]["value"] == 1.0
    assert metrics["by_category"]["Nephrology"] == {"value": 0.5, "n": 2}
    assert ctx[0].query(ItemResult).filter_by(run_id=r.id).count() == 5
    assert (r.n_items, r.tokens_in, r.tokens_out) == (5, 500, 50)
    assert r.cost_usd == pytest.approx((500 * 0.12 + 50 * 0.35) / 1e6)


def test_second_run_is_deduplicated(ctx):
    first = run(ctx, FakeClient(answers()))
    client = FakeClient(answers())
    second = run(ctx, client)
    assert second.id == first.id and client.calls == 0
    assert ctx[0].query(Run).count() == 1


def test_resume_only_calls_missing_items(ctx):
    session = ctx[0]
    first = run(ctx, FakeClient(answers()))
    first.status = "running"
    session.query(ItemResult).filter(ItemResult.run_id == first.id,
                                     ItemResult.instance_id.in_(["cqa-2", "cqa-3", "cqa-4"])).delete()
    session.commit()
    client = FakeClient(answers())
    resumed = run(ctx, client)
    assert client.calls == 3 and resumed.id == first.id and resumed.status == "done"
    assert json.loads(resumed.metrics_json)["accuracy"]["value"] == 0.8


def test_empty_content_counts_as_unparsed(ctx):
    a = answers()
    a["Q4"] = ChatResult(content="", reasoning="long thinking...", tokens_in=100, tokens_out=4096, error=None)
    r = run(ctx, FakeClient(a))
    item = ctx[0].get(ItemResult, (r.id, "cqa-4"))
    assert item.prediction is None and item.correct is False
    metrics = json.loads(r.metrics_json)
    assert metrics["parse_rate"] == 0.8 and metrics["accuracy"]["n_unparsed"] == 1


def test_api_errors_leave_run_failed_and_resumable(ctx):
    a = answers()
    a["Q1"] = ChatResult(content="", reasoning=None, tokens_in=0, tokens_out=0, error="http_503", transient=True)
    r = run(ctx, FakeClient(a))
    assert r.status == "failed" and r.metrics_json is None
    client = FakeClient(answers())
    again = run(ctx, client)
    assert client.calls == 1 and again.status == "done"


def test_predictions_parquet_written(ctx):
    r = run(ctx, FakeClient(answers()))
    df = pd.read_parquet(ctx[3] / "artifacts" / r.id / "predictions.parquet")
    assert {"instance_id", "prediction", "label", "correct"} <= set(df.columns) and len(df) == 5


def test_limit_changes_run_key(ctx):
    full = run(ctx, FakeClient(answers()))
    client = FakeClient(answers())
    partial = run(ctx, client, limit=2)
    assert partial.id != full.id and client.calls == 2 and partial.n_items == 2


def test_constant_answer_baseline_recorded_once(ctx):
    from dataclasses import replace

    from medbench.evaluate import constant_answer_baseline, record_baseline_run

    session, _, task, data_dir = ctx
    items = [replace(it, answer="B") if i < 2 else it for i, it in enumerate(ITEMS)]
    rows, metrics = constant_answer_baseline(items, "B")
    assert metrics["accuracy"]["value"] == 0.4 and metrics["parse_rate"] == 1.0
    assert rows[0] == {"instance_id": "cqa-0", "prediction": "B", "label": "B", "correct": True}
    model = session.get(Model, "medbench/majority-b")
    first = record_baseline_run(session, model, task, rows, metrics, {"letter": "B"}, data_dir=data_dir)
    second = record_baseline_run(session, model, task, rows, metrics, {"letter": "B"}, data_dir=data_dir)
    assert first.id == second.id and first.status == "done" and first.n_items == 5


def test_terminal_error_counts_as_unparsed_and_run_finishes(ctx):
    a = answers()
    a["Q1"] = ChatResult(content="", reasoning=None, tokens_in=0, tokens_out=0, error="context_length_exceeded",
                         transient=False)
    r = run(ctx, FakeClient(a))
    assert r.status == "done"
    item = ctx[0].get(ItemResult, (r.id, "cqa-1"))
    assert item.prediction is None and item.correct is False and item.error == "context_length_exceeded"
    metrics = json.loads(r.metrics_json)
    assert metrics["error_rate"] == 0.2 and metrics["parse_rate"] == 0.8 and metrics["accuracy"]["value"] == 0.6


class RaisingClient(FakeClient):
    async def chat(self, model_id, messages, *, max_tokens, temperature=0.0):
        if messages[1]["content"].startswith("Q2"):
            self.calls += 1
            raise RuntimeError("boom")
        return await super().chat(model_id, messages, max_tokens=max_tokens, temperature=temperature)


def test_client_exception_is_recorded_not_raised(ctx):
    r = run(ctx, RaisingClient(answers()))
    assert r.status == "failed"
    assert ctx[0].get(ItemResult, (r.id, "cqa-2")).error == "exception_RuntimeError"
    again = run(ctx, FakeClient(answers()))
    assert again.status == "done"


def test_run_records_task_spec_hash(ctx):
    r = run(ctx, FakeClient(answers()))
    assert r.task_spec_hash == ctx[2].spec_hash


def test_rekey_restores_deduplication(ctx):
    from medbench.evaluate import rekey_runs

    session = ctx[0]
    first = run(ctx, FakeClient(answers()))
    first.run_key, first.task_spec_hash = "legacy-key", None
    session.commit()
    assert rekey_runs(session) == 1
    client = FakeClient(answers())
    assert run(ctx, client).id == first.id and client.calls == 0
    assert session.get(Run, first.id).task_spec_hash == ctx[2].spec_hash


def test_import_runs_merges_finished_runs_without_touching_other_tables(ctx, tmp_path):
    from medbench.db import PageView, ReuseEvent, init_db, make_engine, session_factory
    from medbench.evaluate import import_runs

    # source database: one finished run, one unfinished run
    src_engine = make_engine(f"sqlite:///{tmp_path}/src.db")
    init_db(src_engine)
    with session_factory(src_engine)() as src:
        load_registry(src, REGISTRY_DIR)
        model = src.get(Model, "google/medgemma-27b")
        task = src.query(Task).filter_by(slug="clinicalqa").one()
        done = asyncio.run(run_llm_task(src, model, task, ITEMS, FakeClient(answers()), data_dir=tmp_path))
        failing = answers()
        failing["Q1"] = ChatResult(content="", reasoning=None, tokens_in=0, tokens_out=0, error="http_503",
                                   transient=True)
        other = src.get(Model, "alibaba/qwen3.8-27b")
        asyncio.run(run_llm_task(src, other, task, ITEMS, FakeClient(failing), data_dir=tmp_path))

    session = ctx[0]
    session.add(PageView(day="2026-10-29", visitor="v", path="/", kind="human"))
    session.commit()
    assert import_runs(session, f"sqlite:///{tmp_path}/src.db") == 1
    assert import_runs(session, f"sqlite:///{tmp_path}/src.db") == 0  # idempotent
    imported = session.query(Run).filter_by(run_key=done.run_key).one()
    assert imported.status == "done" and session.query(ItemResult).filter_by(run_id=imported.id).count() == 5
    assert session.query(Run).count() == 1 and session.query(PageView).count() == 1


def test_raw_output_and_parser_version_are_stored(ctx):
    from medbench.tasks.clinicalqa import PARSER_VERSION

    r = run(ctx, FakeClient(answers()))
    item = ctx[0].get(ItemResult, (r.id, "cqa-0"))
    assert item.raw == '{"answer": "A"}'
    assert json.loads(r.metrics_json)["parser"] == PARSER_VERSION
