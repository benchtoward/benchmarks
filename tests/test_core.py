import pytest
from sqlalchemy.exc import IntegrityError

from conftest import REGISTRY_DIR
from medbench.db import Dataset, Model, Run, Task
from medbench.registry import load_registry
from medbench.runkey import run_key, spec_hash


def test_spec_hash_ignores_key_order():
    assert spec_hash({"a": 1, "b": [1, 2]}) == spec_hash({"b": [1, 2], "a": 1})


def test_run_key_changes_with_each_input():
    base = run_key("m", "h", "p0.1", {"t": 0})
    assert base == run_key("m", "h", "p0.1", {"t": 0})
    assert base != run_key("m2", "h", "p0.1", {"t": 0})
    assert base != run_key("m", "h2", "p0.1", {"t": 0})
    assert base != run_key("m", "h", "p0.2", {"t": 0})
    assert base != run_key("m", "h", "p0.1", {"t": 1})


def test_load_registry_is_idempotent(tmp_session):
    load_registry(tmp_session, REGISTRY_DIR)
    load_registry(tmp_session, REGISTRY_DIR)
    assert tmp_session.query(Task).count() == len(list((REGISTRY_DIR / "tasks").glob("*.yaml"))) >= 6
    assert tmp_session.query(Dataset).count() == 7
    model = tmp_session.get(Model, "google/medgemma-27b")
    assert (model.type, model.developer) == ("llm", "Google")  # serving ids are deployment data
    assert model.digest == "google/medgemma-27b"
    assert tmp_session.get(Model, "medbench/majority-b").digest == "medbench/majority-b:v0"
    task = tmp_session.query(Task).filter_by(slug="vitaldb-ioh", version=0).one()
    assert task.preview is True and task.primary_metric == "auroc" and len(task.spec_hash) == 64


def test_run_key_is_unique(tmp_session):
    load_registry(tmp_session, REGISTRY_DIR)
    task = tmp_session.query(Task).filter_by(slug="clinicalqa").one()
    for run_id in ("aaaaaaaaaaaa", "bbbbbbbbbbbb"):
        tmp_session.add(Run(id=run_id, run_key="same", model_slug="google/medgemma-27b", task_id=task.id,
                            harness_version="p0.1", config_json="{}", tier="baseline", status="done"))
    with pytest.raises(IntegrityError):
        tmp_session.commit()


def test_resolve_task(tmp_session):
    from medbench.registry import resolve_task

    load_registry(tmp_session, REGISTRY_DIR)
    assert resolve_task(tmp_session, "clinicalqa@0").slug == "clinicalqa"
    assert resolve_task(tmp_session, "vitaldb-ioh@0").version == 0
    shown = [t.version for t in tmp_session.query(Task).filter_by(slug="vitaldb-ioh") if t.visibility != "hidden"]
    assert resolve_task(tmp_session, "vitaldb-ioh").version == max(shown)  # the latest version that is not hidden
    with pytest.raises(LookupError):
        resolve_task(tmp_session, "nope")
    with pytest.raises(LookupError):
        resolve_task(tmp_session, "clinicalqa@9")


def test_every_spec_file_is_named_by_its_slug_and_version(tmp_path):
    import shutil

    from medbench.registry import task_spec_files

    files = task_spec_files(REGISTRY_DIR)  # raises on a misnamed file
    assert len(files) == len(list((REGISTRY_DIR / "tasks").glob("*.yaml")))
    (tmp_path / "tasks").mkdir()
    shutil.copy(REGISTRY_DIR / "tasks" / "clinicalqa.yaml", tmp_path / "tasks" / "clinicalqa-copy.yaml")
    with pytest.raises(ValueError, match="belongs in clinicalqa.yaml"):
        task_spec_files(tmp_path)


def test_model_max_tokens(tmp_session):
    from medbench.evaluate import DEFAULT_MAX_TOKENS, max_tokens_for

    load_registry(tmp_session, REGISTRY_DIR)
    assert DEFAULT_MAX_TOKENS == 16384
    assert max_tokens_for(tmp_session.get(Model, "google/medgemma-27b")) == 8192
    assert max_tokens_for(tmp_session.get(Model, "alibaba/qwen3.8-27b")) == 16384


def test_spec_hash_ignores_presentational_fields(tmp_session, tmp_path):
    import shutil

    load_registry(tmp_session, REGISTRY_DIR)
    before = tmp_session.query(Task).filter_by(slug="clinicalqa").one().spec_hash
    copy = tmp_path / "registry"
    shutil.copytree(REGISTRY_DIR, copy)
    spec = copy / "tasks" / "clinicalqa.yaml"
    spec.write_text(spec.read_text(encoding="utf-8").replace("title: ", "title: Renamed ") + "\n", encoding="utf-8")
    load_registry(tmp_session, copy)
    assert tmp_session.query(Task).filter_by(slug="clinicalqa").one().spec_hash == before


def test_init_db_adds_missing_columns():
    from sqlalchemy import inspect, text

    from medbench.db import init_db, make_engine

    engine = make_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE item_results (run_id VARCHAR NOT NULL, instance_id VARCHAR NOT NULL, "
                          "prediction TEXT, PRIMARY KEY (run_id, instance_id))"))
    init_db(engine)
    columns = {c["name"] for c in inspect(engine).get_columns("item_results")}
    assert {"transient", "error", "score", "tokens_in"} <= columns
    init_db(engine)  # idempotent


def test_resolve_task_rejects_odd_versions(tmp_session):
    from medbench.registry import resolve_task

    load_registry(tmp_session, REGISTRY_DIR)
    for ref in ("clinicalqa@\u00b2", "clinicalqa@99999999999999999999999", "clinicalqa@-1", "clinicalqa@ 1"):
        with pytest.raises(LookupError):
            resolve_task(tmp_session, ref)


def test_eval_all_continues_after_a_model_fails(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    import medbench.cli as cli

    monkeypatch.setenv("MEDBENCH_DB_URL", f"sqlite:///{tmp_path}/m.db")
    monkeypatch.setenv("MEDBENCH_REGISTRY_DIR", str(REGISTRY_DIR))
    seen = []

    def fake_eval_one(session, model, task_ref, force, limit):
        seen.append(model.slug)
        if len(seen) == 1:
            raise RuntimeError("boom")

    monkeypatch.setattr(cli, "_eval_one", fake_eval_one)
    assert CliRunner().invoke(cli.app, ["init"]).exit_code == 0
    result = CliRunner().invoke(cli.app, ["eval-all", "--task", "clinicalqa@0"])
    assert result.exit_code == 0 and len(seen) > 1 and "boom" in result.output


def test_vitaldb_datasets_link_to_khdp(tmp_session):
    load_registry(tmp_session, REGISTRY_DIR)
    for slug in ("vitaldb", "vitaldb-arrhythmia"):
        assert tmp_session.get(Dataset, slug).source_url.startswith("https://khdp.net/database/data-search-detail/")


def test_admin_hide_survives_a_registry_reload(tmp_session):
    """The container reloads the registry on every start; a task hidden on /admin must stay hidden."""
    from pathlib import Path

    from medbench.db import Task
    from medbench.registry import load_registry
    registry = Path(__file__).resolve().parents[1] / "registry"
    load_registry(tmp_session, registry)
    tmp_session.query(Task).filter_by(slug="clinicalqa", version=0).one().visibility = "hidden"  # what /admin's hide does
    tmp_session.commit()
    load_registry(tmp_session, registry)
    assert tmp_session.query(Task).filter_by(slug="clinicalqa", version=0).one().visibility == "hidden"


def test_spec_can_hide_a_task_without_changing_its_results(tmp_session, tmp_path):
    import shutil

    from medbench.registry import PRESENTATIONAL_FIELDS
    registry = tmp_path / "registry"
    shutil.copytree(REGISTRY_DIR, registry)
    load_registry(tmp_session, registry)
    shown = tmp_session.query(Task).filter_by(slug="clinicalqa", version=0).one().spec_hash
    spec = registry / "tasks" / "clinicalqa.yaml"
    spec.write_text(spec.read_text(encoding="utf-8") + "visibility: hidden\n", encoding="utf-8")
    load_registry(tmp_session, registry)
    task = tmp_session.query(Task).filter_by(slug="clinicalqa", version=0).one()
    assert "visibility" in PRESENTATIONAL_FIELDS
    assert task.visibility == "hidden" and task.spec_hash == shown
