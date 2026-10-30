"""The Synthetic-SNUH Text-to-SQL items as a KHDP dataset (owner, 2026-10-30): the frozen items of both tiers, the
prompts every evaluated model gets, metadata, README, licence and checksums, built from the files the task spec pins
and refused when anything differs from those pins."""
import hashlib
import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from medbench.tasks import text2sql_release as rel
from medbench.tasks import text2sql_task as t2s
from medbench.tasks.text2sql import HARD_PROMPT, P0, load_questions, render_messages, schema_summary
from test_text2sql_task import DRAFTS, EASY, HARD, SPEC_FILE, Fixture, _write_draft, needs_drafts, needs_spec


@pytest.fixture
def fx(tmp_path):
    return Fixture(tmp_path)


def _build(fx, out: Path, **kw) -> dict:
    return rel.build_release(fx.spec, fx.registry, fx.con, out, **kw)


def test_the_release_holds_the_kept_items_with_the_hash_the_spec_pins(fx, tmp_path):
    out = tmp_path / "release"
    meta = _build(fx, out, code_commit="0123abc")
    for tier, prompt, rows in (("hard", HARD_PROMPT, HARD), ("easy", P0, EASY)):
        file_meta, items = load_questions(out / "items" / f"{tier}.yaml")  # the medbench loader reads it as is
        assert t2s.items_sha256(items) == fx.spec["tiers"][tier]["items"]["sha256"]
        assert meta["tiers"][tier]["items_sha256"] == fx.spec["tiers"][tier]["items"]["sha256"]
        assert [q.id for q in items] == [r["id"] for r in rows] and file_meta["prompt"]["id"] == prompt.id
        assert file_meta["tier"] == tier and file_meta["dataset"] == "synthetic-snuh-text2sql@1.0.0"
    rows = [json.loads(line) for line in (out / "items" / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["tier"] for r in rows] == ["hard", "hard", "easy", "easy"]
    assert set(rows[0]) == set(rel.RELEASE_FIELDS)
    unanswerable = next(r for r in rows if r["id"] == "h-002")
    assert unanswerable["answerable"] is False and unanswerable["sql"] is None
    assert unanswerable["category"] == "unanswerable"
    hard = meta["tiers"]["hard"]
    assert (hard["questions"], hard["answerable"], hard["unanswerable"], hard["instances"], hard["abstention"]) == \
        (2, 1, 1, 4, True)
    assert meta["instances"] == 8 and meta["code"]["commit"] == "0123abc" and meta["doi"] is None
    assert meta["source"]["doi"] == "10.23421/2026.03183" and meta["licence"] == "CC BY-NC 4.0"


def test_the_prompts_are_the_ones_every_evaluated_model_gets(fx, tmp_path):
    out = tmp_path / "release"
    _build(fx, out)
    q = t2s.load_items(fx.registry / "drafts" / "hard.yaml")[1][0]
    for tier, spec in (("hard", HARD_PROMPT), ("easy", P0)):
        folder = out / "prompts" / tier
        schema = (folder / "schema.txt").read_text(encoding="utf-8")
        assert schema == schema_summary(fx.con, null_columns=spec.null_columns)
        assert hashlib.sha256(schema.encode()).hexdigest() == fx.spec["tiers"][tier]["prompt"]["schema_sha256"]
        assert (folder / "system.txt").read_text(encoding="utf-8") == spec.system
        assert (folder / "database_note.txt").read_text(encoding="utf-8") == spec.conventions
        template = (folder / "user_template.txt").read_text(encoding="utf-8")
        for lang in ("ko", "en"):
            user = (template.replace("{schema}", schema).replace("{database_note}", spec.conventions)
                    .replace("{question}", q.text(lang)))
            assert [{"role": "system", "content": spec.system}, {"role": "user", "content": user}] == \
                render_messages(q, lang, schema, spec)


def test_the_release_carries_readme_licence_and_checksums(fx, tmp_path):
    out = tmp_path / "release"
    _build(fx, out, code_commit="0123abc")
    readme = (out / "README.md").read_text(encoding="utf-8")
    for needed in ("CC BY-NC 4.0", "10.23421/2026.03183", "Seoul National University Hospital",
                   "https://github.com/medbench-kr/benchmarks", "0123abc", "RS(0)", "ABSTAIN",
                   "10.18653/v1/2024.clinicalnlp-1.62", "GPT-6 Astra", "Gemini 4 Argon", "Claude Opus 5.5",
                   "## Limitations", "## Citation"):
        assert needed in readme, needed
    licence = (out / "LICENSE.txt").read_text(encoding="utf-8")
    assert "https://creativecommons.org/licenses/by-nc/4.0/" in licence
    assert "Copyright (c) 2026 Seoul National University Hospital" in licence and "SNOMED CT" in licence
    sums = dict(reversed(line.split("  ", 1)) for line in (out / "SHA256SUMS").read_text().splitlines())
    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    assert sorted(sums) == files
    assert all(hashlib.sha256((out / f).read_bytes()).hexdigest() == h for f, h in sums.items())


def test_review_records_and_decisions_stay_out(fx, tmp_path):
    """The release holds what fixes an item and its scoring (and why a hard item is hard), not the review notes or
    the decisions blocks of the question files."""
    hard = [{**HARD[0], "review_note": "REVIEW NOTE TEXT", "why_hard": "two concepts in clinical words",
             "proposed_by": "x/model", "generation_batch": "clinical_terms/g1/b1"}, HARD[1]]
    _write_draft(fx.registry / "drafts" / "hard.yaml", HARD_PROMPT.id, hard)
    out = tmp_path / "release"
    _build(fx, out)
    text = "\n".join(p.read_text(encoding="utf-8") for p in out.rglob("*") if p.is_file())
    assert "REVIEW NOTE TEXT" not in text and "generation_batch" not in text and "review_note" not in text
    assert "two concepts in clinical words" in (out / "items" / "hard.yaml").read_text(encoding="utf-8")


def test_rejected_items_are_left_out(fx, tmp_path):
    _write_draft(fx.registry / "drafts" / "easy.yaml", P0.id, [*EASY, {**EASY[0], "id": "e-009", "review": "rejected"}])
    out = tmp_path / "release"
    _build(fx, out)
    _, items = load_questions(out / "items" / "easy.yaml")
    assert [q.id for q in items] == ["e-001", "e-002"]


def test_anything_that_differs_from_the_pins_is_refused_and_nothing_is_written(fx, tmp_path):
    out = tmp_path / "release"
    edited = [{**HARD[0], "question_en": "How many patients have type 2 diabetes mellitus?"}, HARD[1]]
    _write_draft(fx.registry / "drafts" / "hard.yaml", HARD_PROMPT.id, edited)
    with pytest.raises(ValueError, match="items of tier hard"):
        _build(fx, out)
    _write_draft(fx.registry / "drafts" / "hard.yaml", HARD_PROMPT.id, HARD)
    fx.spec["tiers"]["easy"]["prompt"]["schema_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="schema of tier easy"):
        _build(fx, out)
    assert not out.exists()


def test_without_the_public_commit_the_release_says_it_is_pending(fx, tmp_path):
    out = tmp_path / "release"
    meta = _build(fx, out)
    assert meta["code"]["commit"] is None
    assert rel.PENDING in (out / "README.md").read_text(encoding="utf-8")


def test_the_command_builds_into_the_data_folder_and_keeps_an_earlier_build(fx, tmp_path, monkeypatch):
    reg = fx.write_registry()
    monkeypatch.setenv("MEDBENCH_REGISTRY_DIR", str(reg))
    monkeypatch.setenv("MEDBENCH_DATA_DIR", str(fx.data_dir))
    from medbench.cli import app
    runner = CliRunner()
    result = runner.invoke(app, ["text2sql", "khdp-package"])
    assert result.exit_code == 0, result.stdout
    out = fx.data_dir / "khdp" / "synthetic-snuh-text2sql-1.0.0"
    assert (out / "items" / "hard.yaml").exists() and "--code-commit" in result.stdout  # the commit is pending
    result = runner.invoke(app, ["text2sql", "khdp-package", "--code-commit", "0123abc"])
    assert result.exit_code == 1 and "--force" in result.stdout
    result = runner.invoke(app, ["text2sql", "khdp-package", "--code-commit", "0123abc", "--force"])
    assert result.exit_code == 0, result.stdout
    assert json.loads((out / "metadata.json").read_text())["code"]["commit"] == "0123abc"


@needs_spec
@needs_drafts
def test_the_frozen_files_give_the_items_the_spec_pins(tmp_path):
    """Without the database: the real question files, written in the release format and read back, give the item
    hashes the registered spec pins (101 hard items, 18 unanswerable; 154 easy items)."""
    spec = yaml.safe_load(SPEC_FILE.read_text(encoding="utf-8"))
    tiers = t2s.load_tiers(spec, DRAFTS.parent)
    for tier in tiers:
        path = tmp_path / f"{tier.name}.yaml"
        rel.write_items(path, tier)
        _, items = load_questions(path)
        assert t2s.items_sha256(items) == spec["tiers"][tier.name]["items"]["sha256"]
        assert len(items) == spec["tiers"][tier.name]["items"]["questions"]
