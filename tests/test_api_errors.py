"""Owner 2026-11-05: final API failures invalidate a model run at an inclusive 1%."""
import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from benchtoward.db import ItemResult, Model, Run, Task


@pytest.mark.parametrize("total, errors, failed", [(100, 1, True), (101, 1, False), (100, 0, False), (510, 510, True)])
def test_inclusive_threshold_and_preserved_diagnostics(total, errors, failed):
    from benchtoward.evaluate import API_ERROR_FAIL_RATE, finish_model_run

    assert API_ERROR_FAIL_RATE == 0.01
    run = SimpleNamespace(status="running", internal_json='{"cache": "keep"}', metrics_json='{"accuracy": 0}')
    items = [ItemResult(error="http_400" if i < errors else None, transient=False) for i in range(total)]
    reason = finish_model_run(run, items)
    assert run.status == ("failed" if failed else "done")
    assert run.metrics_json == '{"accuracy": 0}' and json.loads(run.internal_json)["cache"] == "keep"
    if failed:
        assert reason == json.loads(run.internal_json)["failed_reason"]
        assert f"{errors}/{total}" in reason and "http_400" in reason
    else:
        assert reason is None and "failed_reason" not in json.loads(run.internal_json)


def test_only_final_call_errors_count():
    from benchtoward.evaluate import finish_model_run

    run = SimpleNamespace(status="running", internal_json=None)
    # Parse failures and length cutoffs have no call error; neither do successful retries.
    items = [ItemResult(error=None, prediction=None, tokens_out=16384),
             ItemResult(error=None, raw="not JSON"), ItemResult(error=None, raw="answer after retry")]
    assert finish_model_run(run, items) is None and run.status == "done"


def test_audit_dry_run_and_apply_use_stored_items(tmp_session, monkeypatch):
    from contextlib import contextmanager
    from conftest import REGISTRY_DIR
    from benchtoward import cli
    from benchtoward.registry import load_registry

    load_registry(tmp_session, REGISTRY_DIR)
    task = tmp_session.query(Task).filter_by(slug="clinicalqa").one()
    for name, count, total in [("bad", 1, 100), ("below", 1, 101), ("failed", 2, 2), ("baseline", 2, 2)]:
        model = "benchtoward/majority-b" if name == "baseline" else "google/medgemma-27b"
        r = Run(id=name, run_key=name, task_id=task.id, model_slug=model, status="failed" if name == "failed" else "done",
                harness_version="test", config_json="{}", tier="baseline", n_items=total,
                metrics_json='{"accuracy": 0.9}', internal_json='{"keep": true}')
        tmp_session.add(r)
        tmp_session.add_all([ItemResult(run_id=name, instance_id=str(i), error="http_400" if i < count else None,
                                       transient=False) for i in range(total)])
    tmp_session.commit()

    @contextmanager
    def session():
        yield tmp_session

    monkeypatch.setattr(cli, "open_session", session)
    out = CliRunner().invoke(cli.app, ["audit-api-errors"])
    assert out.exit_code == 0, out.output
    assert "bad" in out.output and "clinicalqa@0" in out.output and "1/100" in out.output
    assert "below" not in out.output and "baseline" not in out.output
    assert tmp_session.get(Run, "bad").status == "done"
    out = CliRunner().invoke(cli.app, ["audit-api-errors", "--apply"])
    assert out.exit_code == 0, out.output
    bad = tmp_session.get(Run, "bad")
    assert bad.status == "failed" and bad.metrics_json == '{"accuracy": 0.9}'
    assert json.loads(bad.internal_json)["keep"] is True
    assert "http_400" in json.loads(bad.internal_json)["failed_reason"]
    assert tmp_session.get(Run, "baseline").status == "done"
    assert CliRunner().invoke(cli.app, ["audit-api-errors", "--apply"]).exit_code == 0
