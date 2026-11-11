import asyncio
import json

import pandas as pd
import pytest

from conftest import REGISTRY_DIR
from benchtoward.db import ItemResult, Model, Run, Task
from benchtoward.evaluate import HARNESS_VERSION, run_llm_task
from benchtoward.runkey import run_key
from benchtoward.llm import ChatResult
from benchtoward.registry import load_registry
from benchtoward.tasks.clinicalqa import QAItem


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
    assert r.status == "failed" and json.loads(r.metrics_json)["error_rate"] == 0.2
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

    from benchtoward.evaluate import constant_answer_baseline, record_baseline_run

    session, _, task, data_dir = ctx
    items = [replace(it, answer="B") if i < 2 else it for i, it in enumerate(ITEMS)]
    rows, metrics = constant_answer_baseline(items, "B")
    assert metrics["accuracy"]["value"] == 0.4 and metrics["parse_rate"] == 1.0
    assert rows[0] == {"instance_id": "cqa-0", "prediction": "B", "label": "B", "correct": True}
    model = session.get(Model, "benchtoward/majority-b")
    first = record_baseline_run(session, model, task, rows, metrics, {"letter": "B"}, data_dir=data_dir)
    second = record_baseline_run(session, model, task, rows, metrics, {"letter": "B"}, data_dir=data_dir)
    assert first.id == second.id and first.status == "done" and first.n_items == 5


def test_terminal_error_fails_with_metrics_and_is_retried(ctx):
    a = answers()
    a["Q1"] = ChatResult(content="", reasoning=None, tokens_in=0, tokens_out=0, error="context_length_exceeded",
                         transient=False)
    r = run(ctx, FakeClient(a))
    assert r.status == "failed"
    reason = json.loads(r.internal_json)["failed_reason"]
    assert "1/5" in reason and "20.00%" in reason and "context_length_exceeded" in reason
    item = ctx[0].get(ItemResult, (r.id, "cqa-1"))
    assert item.prediction is None and item.correct is False and item.error == "context_length_exceeded"
    metrics = json.loads(r.metrics_json)
    assert metrics["error_rate"] == 0.2 and metrics["parse_rate"] == 0.8 and metrics["accuracy"]["value"] == 0.6
    client = FakeClient(answers())
    again = run(ctx, client)
    assert again.id == r.id and again.status == "done" and client.calls == 1
    assert "failed_reason" not in json.loads(again.internal_json)


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
    from benchtoward.evaluate import rekey_runs

    session = ctx[0]
    first = run(ctx, FakeClient(answers()))
    first.run_key = "legacy-key"
    session.commit()
    assert rekey_runs(session) == 1
    client = FakeClient(answers())
    assert run(ctx, client).id == first.id and client.calls == 0
    assert session.get(Run, first.id).task_spec_hash == ctx[2].spec_hash


def test_rekey_never_promotes_stale_runs(ctx):
    """A run on an older spec keeps its hash (it stays unlisted) and gets a key of its own; the old rekey moved every
    run to the current hash, which collided with the current run (IntegrityError) or listed the stale one."""
    from benchtoward.evaluate import rekey_runs
    from benchtoward.runkey import run_key

    session, model, task, _ = ctx
    current = run(ctx, FakeClient(answers()))
    stale = Run(id="stale0000001", run_key="legacy", model_slug=current.model_slug, task_id=task.id,
                harness_version=current.harness_version, config_json=current.config_json, tier=current.tier,
                status="done", task_spec_hash="0" * 64)
    session.add(stale)
    session.commit()
    assert rekey_runs(session) == 1
    session.expire_all()
    stale = session.get(Run, "stale0000001")
    assert stale.task_spec_hash == "0" * 64 != task.spec_hash  # listed runs need the task's current hash
    assert stale.run_key == run_key(model.digest, "0" * 64, stale.harness_version, json.loads(stale.config_json))
    assert session.get(Run, current.id).run_key != stale.run_key


def test_import_runs_merges_finished_runs_without_touching_other_tables(ctx, tmp_path):
    from benchtoward.db import PageView, ReuseEvent, init_db, make_engine, session_factory
    from benchtoward.evaluate import import_runs

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
    assert import_runs(session, f"sqlite:///{tmp_path}/src.db")["runs"] == 1
    assert import_runs(session, f"sqlite:///{tmp_path}/src.db")["runs"] == 0  # idempotent
    imported = session.query(Run).filter_by(run_key=done.run_key).one()
    assert imported.status == "done" and session.query(ItemResult).filter_by(run_id=imported.id).count() == 5
    assert session.query(Run).count() == 1 and session.query(PageView).count() == 1


def test_raw_output_and_parser_version_are_stored(ctx):
    from benchtoward.tasks.clinicalqa import PARSER_VERSION

    r = run(ctx, FakeClient(answers()))
    item = ctx[0].get(ItemResult, (r.id, "cqa-0"))
    assert item.raw == '{"answer": "A"}'
    assert json.loads(r.metrics_json)["parser"] == PARSER_VERSION



def test_new_runs_record_the_parser_in_their_config_and_read_option_lines(ctx):
    from benchtoward.tasks.clinicalqa import PARSER_VERSION
    r = run(ctx, FakeClient({**answers(), "Q4": ok("A. a")}))
    assert json.loads(r.config_json)["parser"] == PARSER_VERSION
    assert json.loads(r.metrics_json)["accuracy"]["value"] == 1.0


def _old_p02_run(ctx):  # also used by tests/test_web.py
    """A finished run made before p0.3: its config has no parser and its option-line replies were unparsed."""
    session = ctx[0]
    r = run(ctx, FakeClient({**answers(), "Q3": ok("A. a"), "Q4": ok("B. b")}))
    r.config_json = json.dumps({k: v for k, v in json.loads(r.config_json).items() if k != "parser"}, sort_keys=True)
    r.run_key = "old-p02-key"
    r.internal_json = json.dumps({"checks": [{"name": "parse_rate", "ok": False, "detail": "65% answers parsed"}]})
    metrics = json.loads(r.metrics_json)
    metrics["parser"] = "p0.2"
    r.metrics_json = json.dumps(metrics)
    for iid in ("cqa-3", "cqa-4"):
        item = session.get(ItemResult, (r.id, iid))
        item.prediction, item.correct = None, False
    session.commit()
    return r


def test_rescore_makes_a_new_run_from_stored_replies_without_model_calls(ctx):
    from benchtoward.evaluate import rescore_clinicalqa
    session, model, task, data_dir = ctx
    old = _old_p02_run(ctx)
    made = rescore_clinicalqa(session, task, ITEMS, data_dir=data_dir)
    assert [m.model_slug for m in made] == [model.slug]
    new = made[0]
    assert new.id != old.id and new.run_key != old.run_key and new.status == old.status
    assert json.loads(new.config_json)["parser"] == "p0.3"
    metrics = json.loads(new.metrics_json)
    assert metrics["parser"] == "p0.3" and metrics["accuracy"]["value"] == 0.8 and metrics["parse_rate"] == 1.0
    assert (new.tokens_in, new.tokens_out, new.cost_usd, new.eval_date) == (old.tokens_in, old.tokens_out,
                                                                            old.cost_usd, old.eval_date)
    assert json.loads(new.internal_json)["rescored_from"] == old.id
    assert "checks" not in json.loads(new.internal_json)     # the old run's checks describe the old numbers
    assert session.get(ItemResult, (new.id, "cqa-3")).prediction == "A"
    assert session.get(ItemResult, (new.id, "cqa-3")).raw == "A. a"
    assert json.loads(session.get(Run, old.id).metrics_json)["parser"] == "p0.2"     # the old run is untouched
    assert (data_dir / "artifacts" / new.id / "predictions.parquet").exists()
    assert new.created_at > old.created_at        # the leaderboard lists each model's latest run (tests/test_web.py)
    assert rescore_clinicalqa(session, task, ITEMS, data_dir=data_dir) == []          # idempotent


def test_rescore_skips_runs_already_on_the_current_parser_and_unfinished_runs(ctx):
    from benchtoward.evaluate import rescore_clinicalqa
    session, model, task, data_dir = ctx
    current = run(ctx, FakeClient(answers()))
    assert rescore_clinicalqa(session, task, ITEMS, data_dir=data_dir) == []
    current.status = "failed"
    current.config_json = "{}"
    session.commit()
    assert rescore_clinicalqa(session, task, ITEMS, data_dir=data_dir) == []



def test_rescore_dry_run_writes_nothing(ctx):
    from benchtoward.evaluate import rescore_clinicalqa
    session, model, task, data_dir = ctx
    old = _old_p02_run(ctx)
    preview = rescore_clinicalqa(session, task, ITEMS, data_dir=data_dir, dry_run=True)
    assert len(preview) == 1 and json.loads(preview[0].metrics_json)["accuracy"]["value"] == 0.8
    assert json.loads(preview[0].internal_json)["rescored_from"] == old.id
    assert session.query(Run).count() == 1 and session.query(ItemResult).count() == 5


def test_rescore_keeps_answers_already_read_and_never_drops_items_without_stored_replies(ctx):
    """Work-host preview 2026-11-08: most runs keep only the parsed letter, not the reply (raw is NULL); the first
    re-score turned their accuracy into 0. p0.3 runs p0.2 first, so a letter already read stays as it is."""
    from benchtoward.evaluate import rescore_clinicalqa
    session, model, task, data_dir = ctx
    old = _old_p02_run(ctx)
    for item in session.query(ItemResult).filter_by(run_id=old.id):
        if item.instance_id != "cqa-3":
            item.raw = None                        # only cqa-3 ("A. a") still has its reply
    session.commit()
    new = rescore_clinicalqa(session, task, ITEMS, data_dir=data_dir)[0]
    preds = {r.instance_id: r.prediction for r in session.query(ItemResult).filter_by(run_id=new.id)}
    assert preds == {"cqa-0": "A", "cqa-1": "A", "cqa-2": "A", "cqa-3": "A", "cqa-4": None}
    assert json.loads(new.metrics_json)["accuracy"]["value"] == 0.8


def test_rescore_leaves_runs_whose_answers_do_not_change(ctx):
    """No new run (and no new result URL) when the new parser reads nothing more."""
    from benchtoward.evaluate import rescore_clinicalqa
    session, model, task, data_dir = ctx
    old = _old_p02_run(ctx)
    for iid in ("cqa-3", "cqa-4"):
        session.get(ItemResult, (old.id, iid)).raw = "free text without an answer"
    session.commit()
    assert rescore_clinicalqa(session, task, ITEMS, data_dir=data_dir) == []
    assert session.query(Run).count() == 1


def test_an_unchanged_p02_run_still_deduplicates_a_new_evaluation(ctx):
    """A run the re-score left alone (nothing to read again) still counts, so the model is not called a second time."""
    session, model, task, data_dir = ctx
    old = _old_p02_run(ctx)
    old.run_key = run_key(model.digest, task.spec_hash, HARNESS_VERSION,
                          {k: v for k, v in json.loads(old.config_json).items()})
    session.commit()
    client = FakeClient(answers())
    assert run(ctx, client).id == old.id and client.calls == 0
