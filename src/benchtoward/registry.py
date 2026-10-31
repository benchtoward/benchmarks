import re
from pathlib import Path

import yaml
from sqlalchemy.orm import Session

import json

from benchtoward.db import Dataset, Model, Run, Task
from benchtoward.runkey import canonical_json, run_key, spec_hash

DATASET_FIELDS = ("title", "modality", "license", "license_url", "source_url", "citation", "khdp_code", "version")
# Wording, badges, listing and where a task's items are released (with the DOI once assigned): changing them must
# not invalidate results.
PRESENTATIONAL_FIELDS = {"title", "preview", "visibility", "items_release"}
MODEL_FIELDS = ("name", "type", "provider", "provider_model_id", "price_in_usd_per_m", "price_out_usd_per_m",
               "price_usd_per_audio_min", "notes", "max_tokens", "reference", "developer")


def _upsert(session: Session, cls, key, values: dict):
    row = session.get(cls, key)
    if row is None:
        row = cls(**values)
        session.add(row)
    else:
        for field, value in values.items():
            setattr(row, field, value)
    return row


def spec_file_name(slug: str, version: int) -> str:
    """Version 0 of a task lives in tasks/<slug>.yaml, a later version in a sibling file tasks/<slug>@<version>.yaml."""
    return f"{slug}.yaml" if int(version) == 0 else f"{slug}@{int(version)}.yaml"


def task_spec_files(registry_dir: Path) -> dict[tuple[str, int], Path]:
    """(slug, version) -> its spec file; refuses a file not named by spec_file_name (links depend on the name)."""
    files = {}
    for path in sorted((Path(registry_dir) / "tasks").glob("*.yaml")):
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        key = (spec["slug"], int(spec["version"]))
        if path.name != spec_file_name(*key):
            raise ValueError(f"{key[0]}@{key[1]} belongs in {spec_file_name(*key)}, not {path.name}")
        files[key] = path
    return files


def load_registry(session: Session, registry_dir: Path) -> None:
    registry_dir = Path(registry_dir)

    for d in yaml.safe_load((registry_dir / "datasets.yaml").read_text(encoding="utf-8")):
        values = {"slug": d["slug"], **{f: d.get(f) for f in DATASET_FIELDS}}
        values["version"] = str(values["version"])
        values["snuh_confirmed"] = True  # registry datasets are SNUH patient data by the maintainers' choice
        _upsert(session, Dataset, d["slug"], values)

    models = yaml.safe_load((registry_dir / "models.yaml").read_text(encoding="utf-8"))
    for m in models:
        values = {"slug": m["slug"], **{f: m.get(f) for f in MODEL_FIELDS}}
        values["digest"] = m.get("digest") or (f"{m['slug']}:v0" if m["provider"] == "medbench" else m["slug"])
        values["enabled"] = m.get("enabled", True)
        _upsert(session, Model, m["slug"], values)

    session.flush()
    for path in sorted((registry_dir / "tasks").glob("*.yaml")):
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        values = {
            "slug": spec["slug"],
            "version": int(spec["version"]),
            "dataset_slug": spec["dataset"],
            "title": spec["title"],
            "kind": spec["kind"],
            "primary_metric": spec["primary_metric"],
            "spec_json": canonical_json(spec),
            "spec_hash": spec_hash({k: v for k, v in spec.items() if k not in PRESENTATIONAL_FIELDS}),
            "preview": bool(spec.get("preview", False)),
            "made_by": spec.get("made_by"),
            "source_paper": spec.get("source_paper"),
        }
        row = session.query(Task).filter_by(slug=values["slug"], version=values["version"]).one_or_none()
        if "visibility" in spec:  # otherwise keep the stored value, so a task hidden on /admin stays hidden
            values["visibility"] = spec["visibility"]
        if row is None:
            session.add(Task(**values))
        else:
            for field, value in values.items():
                setattr(row, field, value)

    session.flush()
    _apply_aliases(session, models)
    session.commit()


def _apply_aliases(session: Session, models: list[dict]) -> None:
    """Move runs stored under an earlier slug to the current one, recompute their run keys, drop the old model."""
    for m in models:
        for alias in m.get("aliases") or []:
            old = session.get(Model, alias)
            if old is None:
                continue
            new = session.get(Model, m["slug"])
            for run in session.query(Run).filter_by(model_slug=alias):
                task = session.get(Task, run.task_id)
                run.model_slug = new.slug
                run.run_key = run_key(new.digest, task.spec_hash, run.harness_version, json.loads(run.config_json or "{}"))
            session.flush()
            session.delete(old)
            session.flush()


def resolve_task(session: Session, ref: str, include_hidden: bool = False) -> Task:
    """Find a task by `slug@version`, or by `slug` (latest version). Raises LookupError (also for hidden tasks)."""
    slug, _, version = ref.partition("@")
    query = session.query(Task).filter_by(slug=slug)
    if not include_hidden:
        query = query.filter(Task.visibility != "hidden")
    if version:
        if not re.fullmatch(r"[0-9]{1,6}", version):
            raise LookupError(f"bad task version in {ref!r}")
        query = query.filter_by(version=int(version))
    task = query.order_by(Task.version.desc()).first()
    if task is None:
        raise LookupError(f"unknown task {ref!r}")
    return task
