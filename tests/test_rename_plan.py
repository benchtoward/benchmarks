"""Rename plans (benchtoward.rename) on a synthetic registry: a task spec that names the old project, two baseline
models in the old namespace, an API model, and the plan that renames them (design: rename to BenchToward, Phase B1)."""
import json
import re
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from benchtoward.db import (DownloadEvent, ItemResult, Model, RenameProof, ReuseEvent, Run, Task, init_db, make_engine,
                            session_factory)
from benchtoward.registry import PRESENTATIONAL_FIELDS, load_registry
from benchtoward.rename import APPROVED_PLANS, RenameRefused, _listed_ids, dry_run, read_plan, rename_lag
from benchtoward.runkey import canonical_json, run_key, spec_hash

PLAN_ID = "2026-11-benchtoward"
# The plan's ordered text map, with rule 7 matching at the start of a string as well (design review R1).
RULES = [
    ["medbench-kr/benchmarks", "benchtoward/benchmarks"],
    [r"https://medbench\.kr", "https://benchtoward.com"],
    ["src/medbench/", "src/benchtoward/"],
    ["medbench/(?=[a-z0-9])", "benchtoward/"],
    [r"medbench-admin\b", "benchtoward-admin"],
    [r"\bmedbench_made\b", "benchmark_made"],
    ['(?<![^`"])medbench(?= [a-z])', "benchtoward"],
    [r"(?<=uv run )medbench\b", "benchtoward"],
    [r"medbench\.kr", "BenchToward"],
    ["medbench", "BenchToward"],
]
OLD_SPEC = """\
slug: demo-bin
version: 0
dataset: demo
title: Demo binary
kind: binary
primary_metric: auroc
visibility: open
description: "The medbench.kr definition: run `medbench baseline demo-bin`; code in src/medbench/tasks/demo.py."
protocol:
  medbench_made: [threshold, split]
  repository: https://github.com/medbench-kr/benchmarks
metrics:
  threshold: 0.5
baselines:
  medbench/prior: always the prevalence
  medbench/lr: logistic regression, see https://medbench.kr/tasks/demo-bin
command: "medbench baseline demo-bin --report FILE"
alignment:
  left_out: {746: noisy, 1165: flat}
"""
PLAIN_SPEC = """\
slug: demo-plain
version: 0
dataset: demo
title: Demo plain
kind: binary
primary_metric: auroc
metrics:
  threshold: 0.5
"""
DATASETS = """\
- {slug: demo, title: Demo, modality: tabular, license: CC-BY-4.0, license_url: "https://example.test/l",
   source_url: "https://example.test/s", citation: Demo, version: 1}
"""
HV = "p0.1"


def rewrite(text: str) -> str:
    for pattern, replacement in RULES:
        text = re.sub(pattern, replacement, text)
    return text


def spec_text_hash(text: str) -> str:
    return spec_hash({k: v for k, v in yaml.safe_load(text).items() if k not in PRESENTATIONAL_FIELDS})


def models_yaml(new: bool, prior_name: str = "Prior") -> str:
    ns, provider, developer = ("benchtoward", "benchtoward", "BenchToward") if new else ("medbench", "medbench",
                                                                                         "medbench.kr")
    return (f'- {{slug: {ns}/prior, name: "{prior_name}", type: baseline, provider: {provider}, developer: {developer}, '
            f"reference: true}}\n"
            f'- {{slug: {ns}/lr, name: "LR", type: baseline, provider: {provider}, developer: {developer}}}\n'
            f'- {{slug: lab/api, name: "API model", type: llm, provider: lab, provider_model_id: api, developer: Lab}}\n')


def plan_yaml(*, plan_id=PLAN_ID, src=None, dst="auto", new_spec=None, models=None) -> str:
    src = src or spec_text_hash(OLD_SPEC)
    dst = spec_text_hash(new_spec or rewrite(OLD_SPEC)) if dst == "auto" else dst
    models = models if models is not None else {"medbench/prior": "benchtoward/prior", "medbench/lr": "benchtoward/lr"}
    doc = {"id": plan_id, "reason": "test plan",
           "text_rules": RULES, "models": models,
           "model_fields": {"provider": {"from": "medbench", "to": "benchtoward"},
                            "developer": {"from": "medbench.kr", "to": "BenchToward"}}}
    text = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)
    return text + "tasks:\n  demo-bin@0:\n    - {from: " + src + ", to: " + ("null" if dst is None else dst) + "}\n"


def make_registry(root: Path, *, new: bool, spec: str | None = None, models: str | None = None,
                  plan: str | None = "default") -> Path:
    (root / "tasks").mkdir(parents=True)
    (root / "datasets.yaml").write_text(DATASETS)
    (root / "models.yaml").write_text(models or models_yaml(new))
    (root / "tasks" / "demo-bin.yaml").write_text(spec or (rewrite(OLD_SPEC) if new else OLD_SPEC))
    (root / "tasks" / "demo-plain.yaml").write_text(PLAIN_SPEC)
    if new and plan:
        (root / "renames").mkdir()
        text = plan_yaml() if plan == "default" else plan
        plan_id = yaml.safe_load(text)["id"]
        (root / "renames" / f"{plan_id}.yaml").write_text(text)
    return root


class Env:
    """A file database loaded from the old registry, with runs; `new` is the renamed registry with the plan."""

    def __init__(self, tmp_path: Path, name: str = "bench.db", **new_kwargs):
        self.tmp = tmp_path
        self.old = make_registry(tmp_path / f"{name}-old", new=False)
        self.new = make_registry(tmp_path / f"{name}-new", new=True, **new_kwargs)
        self.path = tmp_path / name
        self.url = f"sqlite:///{self.path}"
        engine = make_engine(self.url)
        init_db(engine)
        self.session = session_factory(engine)()
        load_registry(self.session, self.old)

    def task(self, slug: str = "demo-bin") -> Task:
        self.session.expire_all()
        return self.session.query(Task).filter_by(slug=slug, version=0).one()

    def add_run(self, run_id: str, model: str, *, slug: str = "demo-bin", config: dict | None = None,
                spec: str | None = None, status: str = "done", digest: str | None = None) -> Run:
        task = self.task(slug)
        config = {"seed": 0} if config is None else config
        spec = spec or task.spec_hash
        digest = digest or self.session.get(Model, model).digest
        run = Run(id=run_id, run_key=run_key(digest, spec, HV, config), model_slug=model, task_id=task.id,
                  harness_version=HV, config_json=json.dumps(config), tier="baseline", status=status,
                  task_spec_hash=spec, metrics_json=json.dumps({"auroc": {"value": 0.7, "ci_low": 0.6,
                                                                          "ci_high": 0.8}}))
        self.session.add(run)
        self.session.commit()
        return run

    def run(self, run_id: str) -> Run:
        self.session.expire_all()
        return self.session.get(Run, run_id)

    def snapshot(self) -> dict:
        self.session.expire_all()
        return {"runs": sorted((r.id, r.run_key, r.task_spec_hash, r.model_slug, r.internal_json)
                               for r in self.session.query(Run)),
                "models": sorted(m.slug for m in self.session.query(Model)),
                "tasks": sorted((t.slug, t.spec_hash) for t in self.session.query(Task)),
                "proofs": sorted((p.plan_id, p.task_ref, p.from_hash) for p in self.session.query(RenameProof))}

    def seed(self) -> None:
        """Current runs on both baselines and the API model, a stale run, and a run on the unchanged task."""
        self.add_run("cur-prior001", "medbench/prior")
        self.add_run("cur-lr000001", "medbench/lr")
        self.add_run("cur-api00001", "lab/api")
        self.add_run("stale-prior1", "medbench/prior", config={"seed": 1}, spec="1" * 64)
        self.add_run("plain-prior1", "medbench/prior", slug="demo-plain")


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    e.seed()
    yield e
    e.session.close()


def cli(*args, registry: Path):
    from benchtoward.cli import app

    return CliRunner().invoke(app, ["rename-rekey", *args], env={"BENCHTOWARD_REGISTRY_DIR": str(registry)})


# --- proofs and moves -------------------------------------------------------------------------------------------


def test_plan_moves_current_runs_and_rekeys(env):
    load_registry(env.session, env.new)
    task = env.task()
    to = spec_text_hash(rewrite(OLD_SPEC))
    assert task.spec_hash == to
    for run_id in ("cur-prior001", "cur-lr000001", "cur-api00001"):
        run = env.run(run_id)
        assert run.task_spec_hash == to
        model = env.session.get(Model, run.model_slug)
        assert run.run_key == run_key(model.digest, run.task_spec_hash, HV, json.loads(run.config_json))
    assert env.run("cur-prior001").model_slug == "benchtoward/prior"
    assert env.run("cur-api00001").model_slug == "lab/api"
    proof = env.session.query(RenameProof).filter_by(task_ref="demo-bin@0").one()
    assert (proof.plan_id, proof.from_hash, proof.to_hash) == (PLAN_ID, spec_text_hash(OLD_SPEC), to)


def test_plan_refuses_a_non_rename_difference(tmp_path):
    changed = rewrite(OLD_SPEC).replace("threshold: 0.5", "threshold: 0.6")
    e = Env(tmp_path, spec=changed, plan=plan_yaml(new_spec=changed))
    e.seed()
    before = e.snapshot()
    with pytest.raises(RenameRefused, match=r"demo-bin@0 differs .* first at metrics\.threshold"):
        load_registry(e.session, e.new)
    assert e.snapshot() == before


def test_plan_refuses_wrong_from_or_to(tmp_path):
    wrong_from = Env(tmp_path, name="from.db", plan=plan_yaml(src="a" * 64))
    wrong_from.seed()
    report = dry_run(wrong_from.url, wrong_from.new)
    line = next(t for t in report.tasks if t.ref == "demo-bin@0")
    assert line.status == "skipped" and line.open_task and report.open_skips
    result = cli("--check", "--db", wrong_from.url, registry=wrong_from.new)
    assert result.exit_code == 1 and "skipped" in result.output

    wrong_to = Env(tmp_path, name="to.db", plan=plan_yaml(dst="b" * 64))
    wrong_to.seed()
    before = wrong_to.snapshot()
    with pytest.raises(RenameRefused, match="the registry spec hashes to"):
        load_registry(wrong_to.session, wrong_to.new)
    assert wrong_to.snapshot() == before


def test_stale_runs_keep_their_hash(env):
    env.add_run("stale-api001", "lab/api", config={"seed": 2}, spec="2" * 64)
    api_key = env.run("stale-api001").run_key
    load_registry(env.session, env.new)
    stale = env.run("stale-prior1")
    assert stale.task_spec_hash == "1" * 64 and stale.model_slug == "benchtoward/prior"
    assert stale.run_key == run_key("benchtoward/prior:v0", "1" * 64, HV, {"seed": 1})  # changed by the digest only
    assert env.run("stale-api001").run_key == api_key and env.run("stale-api001").task_spec_hash == "2" * 64
    assert "stale-prior1" not in _listed_ids(env.session)


def test_identical_config_stale_runs_do_not_collide(tmp_path):
    e = Env(tmp_path)
    e.add_run("prior-cur001", "medbench/prior")
    e.add_run("prior-old001", "medbench/prior", spec="1" * 64)
    e.add_run("prior-old002", "medbench/prior", spec="2" * 64)
    load_registry(e.session, e.new)
    keys = {e.run(i).run_key for i in ("prior-cur001", "prior-old001", "prior-old002")}
    assert len(keys) == 3


def test_model_rename_moves_runs_and_drops_old_row(env):
    load_registry(env.session, env.new)
    s = env.session
    assert s.query(Model).filter(Model.slug.like("medbench/%")).count() == 0
    assert s.get(Model, "benchtoward/prior").digest == "benchtoward/prior:v0"
    assert s.get(Model, "benchtoward/prior").provider == "benchtoward"
    assert s.get(Model, "benchtoward/prior").developer == "BenchToward"
    slugs = {m.slug for m in s.query(Model)}
    assert all(r.model_slug in slugs for r in s.query(Run))
    assert {r.model_slug for r in s.query(Run)} == {"benchtoward/prior", "benchtoward/lr", "lab/api"}


def test_model_rename_refuses_other_field_changes(tmp_path):
    e = Env(tmp_path, models=models_yaml(True, prior_name="Prior v2"))
    e.seed()
    before = e.snapshot()
    with pytest.raises(RenameRefused, match=r"medbench/prior -> benchtoward/prior: name differs"):
        load_registry(e.session, e.new)
    assert e.snapshot() == before


def test_apply_is_idempotent(env):
    load_registry(env.session, env.new)
    first = env.snapshot()
    report = load_registry(env.session, env.new)
    assert env.snapshot() == first
    assert report.runs_touched == 0 and report.keys_changed == 0


def test_listed_set_is_invariant(env):
    before = _listed_ids(env.session)
    load_registry(env.session, env.new)
    after = _listed_ids(env.session)
    assert before == after == {"cur-prior001", "cur-lr000001", "cur-api00001", "plain-prior1"}


def test_run_ids_and_reuse_counts_survive(env):
    s = env.session
    for run_id in ("cur-prior001", "cur-api00001"):
        s.add(ItemResult(run_id=run_id, instance_id="i1", score=0.5))
        s.add(ReuseEvent(run_id=run_id, kind="lookup_hit"))
        s.add(DownloadEvent(day="2026-10-31", visitor="v", kind="predictions", target=run_id))
    s.commit()

    def counts():
        s.expire_all()
        return {run_id: (s.query(ItemResult).filter_by(run_id=run_id).count(),
                         s.query(ReuseEvent).filter_by(run_id=run_id).count(),
                         s.query(DownloadEvent).filter_by(target=run_id).count())
                for (run_id,) in s.query(Run.id)}

    before = counts()
    load_registry(s, env.new)
    assert counts() == before


def test_late_old_hash_run_moves_after_proof(env):
    load_registry(env.session, env.new)
    src, to = spec_text_hash(OLD_SPEC), spec_text_hash(rewrite(OLD_SPEC))
    env.add_run("late-api0001", "lab/api", config={"seed": 9}, spec=src)  # a job that started before the pull
    report = load_registry(env.session, env.new)
    assert env.run("late-api0001").task_spec_hash == to
    assert next(t for t in report.tasks if t.ref == "demo-bin@0").status == "late"

    env.session.query(RenameProof).filter_by(task_ref="demo-bin@0").delete()
    env.session.commit()
    env.add_run("late-api0002", "lab/api", config={"seed": 10}, spec=src)
    report = load_registry(env.session, env.new)
    assert env.run("late-api0002").task_spec_hash == src
    line = next(t for t in report.tasks if t.ref == "demo-bin@0")
    assert line.left_on_from == 1 and "without a proof record" in line.text()


def test_previous_keys_recorded(env):
    old = env.run("cur-prior001")
    old_key, old_hash = old.run_key, old.task_spec_hash
    env.session.query(Run).filter_by(id="cur-prior001").update({"internal_json": json.dumps({"publish": "ok"})})
    env.session.commit()
    load_registry(env.session, env.new)
    internal = json.loads(env.run("cur-prior001").internal_json)
    assert internal["publish"] == "ok"
    (entry,) = internal["renames"]
    assert (entry["plan"], entry["run_key"], entry["task_spec_hash"], entry["model_slug"]) == (
        PLAN_ID, old_key, old_hash, "medbench/prior")


def test_proof_handles_integer_keys(env):
    stored = env.task()
    reread = {k: v for k, v in json.loads(stored.spec_json).items() if k not in PRESENTATIONAL_FIELDS}
    assert spec_hash(reread) != stored.spec_hash  # integer keys sort as text once read back: never re-hash spec_json
    load_registry(env.session, env.new)
    assert env.task().spec_hash == spec_text_hash(rewrite(OLD_SPEC))


def test_rule_seven_matches_a_command_at_string_start():
    """Review R1: the parsed value of `command: "medbench baseline ..."` starts with the word, so the rules applied to
    the parsed spec and to the file text must agree there."""
    plan = read_plan_from_text(plan_yaml())
    assert plan.rewrite("medbench baseline demo-bin") == "benchtoward baseline demo-bin"
    assert plan.rewrite("see `medbench baseline x`") == "see `benchtoward baseline x`"
    assert plan.rewrite("the medbench agent") == "the BenchToward agent"
    assert plan.apply(yaml.safe_load(OLD_SPEC)) == yaml.safe_load(rewrite(OLD_SPEC))


def read_plan_from_text(text: str, tmp: Path | None = None):
    import tempfile

    folder = Path(tmp or tempfile.mkdtemp())
    path = folder / f"{yaml.safe_load(text)['id']}.yaml"
    path.write_text(text)
    return read_plan(path)


def test_fresh_baseline_run_hits_run_once(env, tmp_path):
    from benchtoward.evaluate import record_baseline_run

    load_registry(env.session, env.new)
    model, task = env.session.get(Model, "benchtoward/prior"), env.task()
    run = record_baseline_run(env.session, model, task, [], {}, {"seed": 0}, data_dir=tmp_path)
    assert run.id == "cur-prior001"


# --- guards -----------------------------------------------------------------------------------------------------


def test_unapproved_plan_is_refused(tmp_path):
    e = Env(tmp_path, plan=plan_yaml(plan_id="2027-01-other"))
    with pytest.raises(RenameRefused, match="2027-01-other is not approved"):
        load_registry(e.session, e.new)
    assert set(APPROVED_PLANS) == {PLAN_ID}


def test_backup_is_taken_before_the_first_move(env):
    load_registry(env.session, env.new)
    backups = list((env.tmp / "backups").glob(f"pre-{PLAN_ID}-*.db"))
    assert len(backups) == 1
    engine = make_engine(f"sqlite:///{backups[0]}")
    with session_factory(engine)() as old:
        assert old.get(Model, "medbench/prior") is not None
        assert old.get(Run, "cur-prior001").task_spec_hash == spec_text_hash(OLD_SPEC)
    load_registry(env.session, env.new)  # nothing left to move: no second backup
    assert len(list((env.tmp / "backups").glob("*.db"))) == 1


def test_forward_guard_refuses_a_database_migrated_by_an_unknown_plan(env):
    load_registry(env.session, env.new)
    after = env.snapshot()
    with pytest.raises(RenameRefused, match=f"migrated by rename plan {PLAN_ID}"):
        load_registry(env.session, env.old)  # code from before the plan
    assert env.snapshot() == after


def test_check_leaves_the_database_byte_identical(env):
    env.session.close()
    before = env.path.read_bytes()
    result = cli("--check", "--db", env.url, registry=env.new)
    assert result.exit_code == 0, result.output
    assert "demo-bin@0" in result.output and "proved, 3 run(s) moved" in result.output
    assert "model medbench/prior -> benchtoward/prior: 3 run(s) moved, old row dropped" in result.output
    assert "listed runs 4 before, 4 after" in result.output
    result = cli("--db", env.url, registry=env.new)
    assert result.exit_code == 0
    assert env.path.read_bytes() == before
    assert not (env.tmp / "backups").exists()


def test_check_fails_while_runs_stay_on_a_from_hash(env):
    """Review R2: a lagging database (its stored spec is neither from nor to) skips the pair; the runs on `from`
    would drop off the leaderboard, so --check fails."""
    task = env.task()
    task.spec_hash = "9" * 64
    env.session.commit()
    env.session.close()
    result = cli("--check", "--db", env.url, registry=env.new)
    assert result.exit_code == 1
    assert "skipped" in result.output and "run(s) on demo-bin@0 at" in result.output


def test_lag_flags_old_namespace_models_the_plan_does_not_list(env):
    """Review R6: a model a pending branch added in the old namespace is lag even though the plan cannot list it."""
    env.session.add(Model(slug="medbench/extra", name="Extra", type="baseline", provider="medbench",
                          digest="medbench/extra:v0", developer="medbench.kr"))
    env.session.commit()
    env.add_run("extra-run001", "medbench/extra", config={"seed": 5})
    load_registry(env.session, env.new)
    lag = rename_lag(env.session, read_plans(env.new))
    assert any("model row medbench/extra (not in the plan)" in line for line in lag)
    assert any("1 run(s) on medbench/extra (not in the plan)" in line for line in lag)
    env.session.close()
    assert cli("--check", "--db", env.url, registry=env.new).exit_code == 1


def read_plans(registry: Path):
    from benchtoward.rename import load_plans

    return load_plans(registry)


def test_scan_reports_unhandled_slug_columns(env):
    """A column whose whole value is an old slug or `from` hash is an identifier the plan does not move: --check
    fails (a table added later that keys on a slug)."""
    from benchtoward.db import OwnerAction

    env.session.add(OwnerAction(kind="publish_run", target="medbench/prior"))
    env.session.commit()
    env.session.close()
    report = dry_run(env.url, env.new)
    assert report.scan_hits == ["owner_actions.target: 1 row(s) name medbench/prior"]
    result = cli("--check", "--db", env.url, registry=env.new)
    assert result.exit_code == 1 and "owner_actions.target" in result.output


def test_scan_notes_old_names_in_free_text_without_failing(env):
    """Review of B1: prose and JSON history (agent messages, build steps, page paths, paper facts) keep the old slugs
    after the rename; a hit there is reported, never fatal, or every later deploy would stop at the preflight."""
    from benchtoward.db import PageView, Paper, PaperAttempt

    env.session.add(Paper(doi="10.1/x", source="test", title="X"))
    env.session.add(PaperAttempt(id="att000000001", paper_doi="10.1/x", facts_json='{"model": "medbench/prior"}'))
    env.session.add(PageView(day="2026-10-31", visitor="v", path="/models/medbench/prior", kind="human"))
    env.session.commit()
    env.session.close()
    report = dry_run(env.url, env.new)
    assert report.scan_hits == []
    assert report.scan_notes == ["page_views.path: 1 row(s) mention medbench/prior",
                                 "paper_attempts.facts_json: 1 row(s) mention medbench/prior"]
    result = cli("--check", "--db", env.url, registry=env.new)
    assert result.exit_code == 0, result.output
    assert "scan notes (history, not fatal): page_views.path" in result.output


def test_scan_skips_the_rename_history(env):
    load_registry(env.session, env.new)
    env.session.close()
    report = dry_run(env.url, env.new)
    assert report.scan_hits == [] and report.lag == [] and report.runs_touched == 0


# --- the command ------------------------------------------------------------------------------------------------


def test_apply_and_status(env):
    env.session.close()
    result = cli("--status", "--db", env.url, registry=env.new)
    assert result.exit_code == 1 and f"plan {PLAN_ID}: not applied" in result.output
    result = cli("--apply", "--db", env.url, registry=env.new)
    assert result.exit_code == 0, result.output
    assert "backup " in result.output and "lag after apply: none" in result.output
    result = cli("--status", "--db", env.url, registry=env.new)
    assert result.exit_code == 0 and f"plan {PLAN_ID}: applied, 1 task proof(s) recorded" in result.output
    assert "lag: none" in result.output


def test_no_plans(tmp_path):
    registry = make_registry(tmp_path / "reg", new=False)
    db = f"sqlite:///{tmp_path / 'empty.db'}"
    engine = make_engine(db)
    init_db(engine)
    with session_factory(engine)() as s:
        assert load_registry(s, registry).lines() == ["no rename plans"]
    for flag in ("--status", "--check"):
        result = cli(flag, "--db", db, registry=registry)
        assert result.exit_code == 0 and "no rename plans" in result.output


def test_write_to_hashes(tmp_path):
    e = Env(tmp_path, plan=plan_yaml(dst=None))
    e.seed()
    with pytest.raises(RenameRefused, match="has no `to` hash"):
        load_registry(e.session, e.new)
    plan_file = e.new / "renames" / f"{PLAN_ID}.yaml"
    result = cli("--write-to-hashes", registry=e.new)
    assert result.exit_code == 0, result.output
    to = spec_text_hash(rewrite(OLD_SPEC))
    assert f"demo-bin@0 {spec_text_hash(OLD_SPEC)[:12]} -> {to[:12]}" in result.output
    assert read_plan(plan_file).tasks["demo-bin@0"] == ((spec_text_hash(OLD_SPEC), to),)
    assert plan_file.read_text().startswith(f"id: {PLAN_ID}\n")
    load_registry(e.session, e.new)
    assert e.task().spec_hash == to


def test_canonical_json_of_stored_spec_is_the_registry_text():
    """The proof reads tasks.spec_json, which load_registry writes as canonical_json of the parsed YAML."""
    spec = yaml.safe_load(OLD_SPEC)
    assert json.loads(canonical_json(spec))["alignment"]["left_out"] == {"746": "noisy", "1165": "flat"}


def test_check_passes_without_a_database_and_refuses_one_an_unknown_plan_migrated(env, tmp_path):
    from benchtoward.cli import app

    missing = f"sqlite:///{tmp_path / 'missing.db'}"
    result = CliRunner().invoke(app, ["rename-rekey", "--check"],  # the deploy preflight on a first deploy
                                env={"BENCHTOWARD_REGISTRY_DIR": str(env.new), "BENCHTOWARD_DB_URL": missing})
    assert result.exit_code == 0 and "no database at" in result.output and not (tmp_path / "missing.db").exists()
    load_registry(env.session, env.new)
    env.session.close()
    result = cli("--check", "--db", env.url, registry=env.old)  # code without the plan (the deploy preflight)
    assert result.exit_code == 1 and f"migrated by rename plan {PLAN_ID}" in result.output


def test_an_explicit_db_must_exist_and_be_a_plain_path(env, tmp_path, monkeypatch):
    """Review of B1: a mistyped scratch path in `--check --db` passed silently, `--apply --db` created a fresh
    database, and a `sqlite:///file:...` URI made folders named `file:`."""
    monkeypatch.chdir(tmp_path)
    for args in (["--check"], ["--apply"], ["--status"], []):
        result = cli(*args, "--db", f"sqlite:///{tmp_path / 'typo.db'}", registry=env.new)
        assert result.exit_code == 1 and "no database at" in result.output, (args, result.output)
    assert not (tmp_path / "typo.db").exists()
    uri = f"sqlite:///file:{env.path}?mode=ro&uri=true"
    result = cli("--check", "--db", uri, registry=env.new)
    assert result.exit_code == 1 and "plain sqlite:///<path>" in result.output
    assert not (tmp_path / "file:").exists()


def test_make_engine_makes_no_folders_for_a_uri(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_engine(f"sqlite:///file:{tmp_path / 'x' / 'a.db'}?mode=ro&uri=true").dispose()
    assert not (tmp_path / "file:").exists()


def test_a_refused_plan_file_prints_refused_not_a_traceback(env):
    """Review of B1: load_plans sat outside the command's try, so a bad plan file printed a traceback."""
    bad = env.new / "renames" / "2027-01-other.yaml"
    bad.write_text(plan_yaml(plan_id="2027-01-other"))
    env.session.close()
    for args in (["--check", "--db", env.url], ["--status", "--db", env.url], ["--write-to-hashes"]):
        result = cli(*args, registry=env.new)
        assert result.exit_code == 1 and "refused: rename plan 2027-01-other is not approved" in result.output, args
        assert isinstance(result.exception, SystemExit), args


# --- review of B1: later loads, late runs, backups, old code ---------------------------------------------------


def test_late_baseline_run_on_a_retired_slug_moves(env):
    """Review of B1: after the migration the old model row is gone; a run a job recorded later on the old slug was
    read as unlisted before and listed after, so every later load refused instead of moving it."""
    load_registry(env.session, env.new)
    src, to = spec_text_hash(OLD_SPEC), spec_text_hash(rewrite(OLD_SPEC))
    plain = env.task("demo-plain").spec_hash
    env.add_run("late-prior01", "medbench/prior", config={"seed": 7}, digest="medbench/prior:v0")  # current hash
    env.add_run("late-prior02", "medbench/lr", config={"seed": 8}, spec=src, digest="medbench/lr:v0")
    env.add_run("late-plain01", "medbench/prior", slug="demo-plain", config={"seed": 9}, digest="medbench/prior:v0")
    report = load_registry(env.session, env.new)
    moved = {i: env.run(i) for i in ("late-prior01", "late-prior02", "late-plain01")}
    assert {r.model_slug for r in moved.values()} == {"benchtoward/prior", "benchtoward/lr"}
    assert moved["late-prior02"].task_spec_hash == to and moved["late-plain01"].task_spec_hash == plain
    for run in moved.values():
        digest = env.session.get(Model, run.model_slug).digest
        assert run.run_key == run_key(digest, run.task_spec_hash, HV, json.loads(run.config_json))
    assert set(moved) <= _listed_ids(env.session)
    assert report.runs_touched == 3
    assert rename_lag(env.session, read_plans(env.new)) == []


def test_refused_load_leaves_no_backup(tmp_path):
    """Review of B1: a backup was written on every refused load while moves were pending (a crash-looping container
    would fill the disk). It is still taken before the first write, and removed when the load fails."""
    changed = rewrite(OLD_SPEC).replace("threshold: 0.5", "threshold: 0.6")
    e = Env(tmp_path, spec=changed, plan=plan_yaml(new_spec=changed))
    e.seed()
    for _ in range(3):
        with pytest.raises(RenameRefused):
            load_registry(e.session, e.new)
    assert not list((tmp_path / "backups").glob("*.db"))


def test_migrated_database_refuses_old_model_rows_from_any_code(env):
    """Review of B1 (R5 for code older than B1): a database the plan migrated aborts any insert of a retired model,
    so `init` from a worktree cut before B1 fails loudly instead of putting the old rows back and emptying boards."""
    from sqlalchemy.exc import IntegrityError

    load_registry(env.session, env.new)
    after = env.snapshot()
    for slug, provider in (("medbench/prior", "medbench"), ("medbench/ioh-wave-cnn", "lab"),
                           ("lab/other", "medbench")):
        env.session.add(Model(slug=slug, name="Old", type="baseline", provider=provider, digest=f"{slug}:v0"))
        with pytest.raises(IntegrityError, match=f"rename plan {PLAN_ID} retired"):
            env.session.commit()
        env.session.rollback()
    assert env.snapshot() == after
    env.session.add(Model(slug="lab/new", name="New", type="llm", provider="lab", digest="lab/new"))
    env.session.commit()  # anything else still inserts
    load_registry(env.session, env.new)  # and the plan's own loads still work


def test_dry_run_does_not_install_the_guard(env):
    env.session.close()
    dry_run(env.url, env.new)
    import sqlite3

    with sqlite3.connect(env.path) as con:
        assert con.execute("SELECT count(*) FROM sqlite_master WHERE type = 'trigger'").fetchone() == (0,)


def test_open_is_judged_by_the_visibility_after_the_load(env):
    """Review of B1: --check read the stored visibility, so a task stored hidden whose spec now says open did not
    count as open (its runs would drop off a now-public board)."""
    task = env.task()
    task.spec_hash, task.visibility = "9" * 64, "hidden"
    env.session.commit()
    env.session.close()
    line = next(t for t in dry_run(env.url, env.new).tasks if t.ref == "demo-bin@0")
    assert line.status == "skipped" and line.open_task
    assert cli("--check", "--db", env.url, registry=env.new).exit_code == 1


def test_check_stays_green_after_a_later_edit_and_publish(tmp_path):
    """Review of B1: after the plan is applied, a covered hidden task that gets a spec fix and is then published is
    a skipped pair of an open task; that must not fail every later deploy's preflight."""
    hidden = OLD_SPEC.replace("visibility: open", "visibility: hidden")
    e = Env(tmp_path, spec=rewrite(hidden))
    e.seed()
    load_registry(e.session, e.new)
    spec_file = e.new / "tasks" / "demo-bin.yaml"
    spec_file.write_text(rewrite(hidden).replace("threshold: 0.5", "threshold: 0.6"))  # the fix
    load_registry(e.session, e.new)
    spec_file.write_text(rewrite(OLD_SPEC).replace("threshold: 0.5", "threshold: 0.6"))  # published
    load_registry(e.session, e.new)
    e.session.close()
    result = cli("--check", "--db", e.url, registry=e.new)
    assert result.exit_code == 0, result.output
    assert "skipped" in result.output and "lag after apply: none" in result.output


def test_unmigrated_hidden_task_edited_later_is_skipped_not_refused(tmp_path):
    """Review of B1: a scratch database still at `from` refused every load once its covered (hidden) spec had a
    later edit. For a hidden build that edit makes the runs on `from` stale anyway, so the pair is skipped and
    reported; an open task still refuses (see test_plan_refuses_wrong_from_or_to)."""
    hidden = OLD_SPEC.replace("visibility: open", "visibility: hidden")
    edited = rewrite(hidden).replace("threshold: 0.5", "threshold: 0.6")
    e = Env(tmp_path, spec=edited, plan=plan_yaml(new_spec=rewrite(OLD_SPEC)))
    e.seed()
    task = e.task()
    task.visibility = "hidden"
    e.session.commit()
    report = load_registry(e.session, e.new)
    line = next(t for t in report.tasks if t.ref == "demo-bin@0")
    assert line.status == "skipped" and not line.open_task and line.left_on_from == 3
    assert "the registry spec hashes to" in line.text()
    assert e.task().spec_hash == spec_text_hash(edited)
    assert e.run("cur-api00001").task_spec_hash == spec_text_hash(OLD_SPEC)  # stale, as after any spec edit
    assert e.run("cur-prior001").model_slug == "benchtoward/prior"  # the model rename still applies
    assert rename_lag(e.session, read_plans(e.new)) == []
    e.session.close()
    assert cli("--check", "--db", e.url, registry=e.new).exit_code == 0


def test_later_alias_or_removed_spec_does_not_block_loads(env):
    """Review of B1: once nothing is left to move, the plan no longer constrains models.yaml or the spec files."""
    load_registry(env.session, env.new)
    models = (env.new / "models.yaml").read_text().replace(
        "slug: benchtoward/prior,", "slug: benchtoward/prior-v2, aliases: [benchtoward/prior],")
    (env.new / "models.yaml").write_text(models)
    load_registry(env.session, env.new)
    assert env.run("cur-prior001").model_slug == "benchtoward/prior-v2"
    (env.new / "tasks" / "demo-bin.yaml").unlink()
    report = load_registry(env.session, env.new)
    assert next(t for t in report.tasks if t.ref == "demo-bin@0").status in ("done", "absent")
    env.session.close()
    assert cli("--check", "--db", env.url, registry=env.new).exit_code == 0


def test_a_load_that_fails_after_the_plan_leaves_no_guard(env, monkeypatch):
    """The guard trigger and the plan marker belong to the load's transaction: a failure later in load_registry
    (here in the alias step) rolls both back, with the moves and the backup."""
    import sqlite3

    import benchtoward.registry as registry

    def boom(*args, **kwargs):
        raise RuntimeError("late failure")

    monkeypatch.setattr(registry, "_apply_aliases", boom)
    before = env.snapshot()
    with pytest.raises(RuntimeError, match="late failure"):
        load_registry(env.session, env.new)
    assert env.snapshot() == before
    with sqlite3.connect(env.path) as con:
        assert con.execute("SELECT count(*) FROM sqlite_master WHERE type = 'trigger'").fetchone() == (0,)
    assert not list((env.tmp / "backups").glob("*.db"))
