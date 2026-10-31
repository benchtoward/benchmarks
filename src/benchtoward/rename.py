"""Rename plans: owner-approved renames of task-spec text and model slugs that keep every recorded result.

A plan file `registry/renames/<id>.yaml` holds an ordered list of text rules (regular expression, replacement), a map
of old to new model slugs, the model fields allowed to differ (with their old and new values), and per task version
the spec hash pairs `{from, to}`. `load_registry` applies it after the models upsert and before the tasks upsert, in
the same transaction:

- Proof. For a pair whose `from` equals the task's stored spec hash, the registry spec must hash to `to`, and the
  stored spec with the rules applied to every key and string value must equal the registry spec (presentational
  fields aside). Otherwise `RenameRefused` names the first differing path and nothing is written.
- Moves. Runs on `from` move to `to`; runs on an old model slug move to the new one and the old model row is dropped;
  only those runs get a new run key, from their own spec hash. The previous key, hash and slug go to the run's
  `internal_json` under "renames". Run ids never change. Stale runs (other hashes) keep their hash.
- Checks before commit: unique run keys, no run on a missing model, no old model row left, the same listed runs.

Only plan ids in APPROVED_PLANS are applied: a rename-only spec edit that keeps the task version is a one-time owner
exception, not a standing path. A database that a plan migrated refuses code that lacks that plan (check_forward), and
code older than this module too: a trigger aborts any insert of a retired model (install_guard). Once nothing is left
to move in a database, the plan no longer constrains models.yaml or the spec files.
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from benchtoward.db import Model, RenameProof, Run, Task
from benchtoward.runkey import canonical_json, run_key, spec_hash

# Plan id -> who approved it. Any other plan file is refused (owner decision 2026-10-31, design review R10).
APPROVED_PLANS = {
    "2026-11-benchtoward": "owner, 2026-10-31: the project is renamed to BenchToward; rename-only spec edits, "
                           "versions kept, run ids unchanged",
}
PLAN_MARKER = "*"
HASH_RE = re.compile(r"[0-9a-f]{64}")


class RenameRefused(RuntimeError):
    """A rename plan cannot be applied as written; the database is left as it was."""


class NoDatabase(FileNotFoundError):
    """The database file does not exist yet (a first deploy): there is nothing to rename."""


class UnsupportedURL(ValueError):
    """A database URL the rename commands cannot read as a file (a `sqlite:///file:...` URI)."""


@dataclass(frozen=True)
class Plan:
    id: str
    path: Path
    rules: tuple[tuple[re.Pattern, str], ...]
    models: dict[str, str]
    model_fields: dict[str, tuple]  # field -> (old value, new value)
    tasks: dict[str, tuple[tuple[str, str | None], ...]]  # slug@version -> ((from, to), ...)

    def rewrite(self, value: str) -> str:
        for pattern, replacement in self.rules:
            value = pattern.sub(replacement, value)
        return value

    def apply(self, obj):
        """The rules applied to every dict key and string value of a parsed spec."""
        if isinstance(obj, dict):
            return {(self.rewrite(k) if isinstance(k, str) else k): self.apply(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.apply(v) for v in obj]
        if isinstance(obj, str):
            return self.rewrite(obj)
        return obj

    @property
    def old_namespaces(self) -> tuple[str, ...]:
        """Slug prefixes ("medbench/") that the plan retires, if it also retires their provider (review R6)."""
        if "provider" not in self.model_fields:
            return ()
        return tuple(sorted({old.split("/", 1)[0] + "/" for old in self.models if "/" in old}))

    @property
    def old_provider(self) -> str | None:
        return self.model_fields.get("provider", (None, None))[0]


def _task_ref(ref: str) -> tuple[str, int]:
    slug, sep, version = str(ref).partition("@")
    if not sep or not re.fullmatch(r"[0-9]{1,6}", version) or not slug:
        raise ValueError(f"task {ref!r} must be written slug@version")
    return slug, int(version)


def read_plan(path: Path) -> Plan:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    plan_id = str(data.get("id") or "")
    if path.name != f"{plan_id}.yaml":
        raise RenameRefused(f"rename plan {path.name}: id {plan_id!r} must match the file name")
    if plan_id not in APPROVED_PLANS:
        raise RenameRefused(f"rename plan {plan_id} is not approved: a spec edit needs a new task version unless the "
                            f"owner approves a rename-only plan (APPROVED_PLANS in benchtoward/rename.py)")
    rules = tuple((re.compile(pattern), replacement) for pattern, replacement in data.get("text_rules") or [])
    model_fields = {name: (v["from"], v["to"]) for name, v in (data.get("model_fields") or {}).items()}
    tasks = {}
    for ref, pairs in (data.get("tasks") or {}).items():
        _task_ref(ref)
        parsed = []
        for pair in pairs:
            src, dst = str(pair["from"]), pair.get("to")
            if not HASH_RE.fullmatch(src) or (dst is not None and not HASH_RE.fullmatch(str(dst))):
                raise RenameRefused(f"rename plan {plan_id}: {ref} needs full 64-hex hashes, got {src!r} -> {dst!r}")
            parsed.append((src, None if dst is None else str(dst)))
        tasks[str(ref)] = tuple(parsed)
    return Plan(id=plan_id, path=path, rules=rules, models={str(k): str(v) for k, v in (data.get("models") or {}).items()},
                model_fields=model_fields, tasks=tasks)


def load_plans(registry_dir: Path) -> list[Plan]:
    folder = Path(registry_dir) / "renames"
    return [read_plan(path) for path in sorted(folder.glob("*.yaml"))] if folder.is_dir() else []


# ---------------------------------------------------------------------------------------------------------------------
# helpers


def _strip(spec: dict) -> dict:
    from benchtoward.registry import PRESENTATIONAL_FIELDS

    return {k: v for k, v in spec.items() if k not in PRESENTATIONAL_FIELDS}


def _normalize(obj):
    """Integer keys become strings and lists stay ordered: a spec read back from tasks.spec_json compares equal to
    the YAML it came from. The stored hash is never recomputed from spec_json (integer keys sort differently)."""
    return json.loads(canonical_json(obj))


def first_difference(a, b, path: str = "") -> str | None:
    if type(a) is not type(b):
        return path or "<root>"
    if isinstance(a, dict):
        for key in sorted(set(a) | set(b), key=str):
            if key not in a or key not in b:
                return f"{path}.{key}".lstrip(".")
            found = first_difference(a[key], b[key], f"{path}.{key}")
            if found:
                return found.lstrip(".")
        return None
    if isinstance(a, list):
        for i, (x, y) in enumerate(zip(a, b)):
            found = first_difference(x, y, f"{path}[{i}]")
            if found:
                return found.lstrip(".")
        return f"{path}[{min(len(a), len(b))}]".lstrip(".") if len(a) != len(b) else None
    return None if a == b else (path.lstrip(".") or "<root>")


def _has_proofs_table(session: Session) -> bool:
    return inspect(session.connection()).has_table("rename_proofs")


def applied_plan_ids(session: Session) -> set[str]:
    if not _has_proofs_table(session):
        return set()
    return {row[0] for row in session.execute(text("SELECT DISTINCT plan_id FROM rename_proofs"))}


def _listing_state(session: Session, renamed: dict[str, str] | None = None
                   ) -> tuple[dict[int, tuple[str, str]], list[tuple[str, int, str | None, bool]]]:
    """What the leaderboard listing reads: tasks (id -> spec hash, visibility) and finished full runs (id, task id,
    spec hash, whether the model row exists). With `renamed` (old slug -> new), a run on an old slug counts as having
    its model when the new row exists: read as if the plan's slug moves were applied, as a late run on a retired
    slug is meant to appear once moved."""
    tasks = {t.id: (t.spec_hash, t.visibility) for t in session.query(Task)}
    models = {slug for (slug,) in session.query(Model.slug)}
    renamed = renamed or {}
    runs = []
    for run in session.query(Run).filter(Run.status == "done"):
        config = json.loads(run.config_json or "{}")
        if "limit" not in config and not config.get("partial"):
            has_model = run.model_slug in models or renamed.get(run.model_slug) in models
            runs.append((run.id, run.task_id, run.task_spec_hash, has_model))
    return tasks, runs


def _listed(state, task_hashes: dict[int, str] | None = None,
            run_hashes: dict[tuple[int, str], str] | None = None) -> set[str]:
    """Run ids a leaderboard lists (web.data.is_listed, with the model row present as _listed_runs requires). Task
    hashes can be overridden by task id, and run hashes mapped by (task id, hash), to read a state as if the plan's
    pairs were already applied."""
    tasks, runs = state
    task_hashes, run_hashes = task_hashes or {}, run_hashes or {}
    listed = set()
    for run_id, task_id, run_hash, has_model in runs:
        if task_id not in tasks or not has_model or tasks[task_id][1] == "hidden":
            continue
        if run_hashes.get((task_id, run_hash), run_hash) == task_hashes.get(task_id, tasks[task_id][0]):
            listed.add(run_id)
    return listed


def _listed_ids(session: Session) -> set[str]:
    return _listed(_listing_state(session))


def _task_row(session: Session, ref: str) -> Task | None:
    slug, version = _task_ref(ref)
    return session.query(Task).filter_by(slug=slug, version=version).one_or_none()


def _proved(session: Session, plan_id: str, ref: str, src: str, dst: str) -> bool:
    if not _has_proofs_table(session):
        return False
    return session.query(RenameProof).filter_by(plan_id=plan_id, task_ref=ref, from_hash=src,
                                                to_hash=dst).one_or_none() is not None


# ---------------------------------------------------------------------------------------------------------------------
# report


@dataclass
class TaskLine:
    plan: str
    ref: str
    from_hash: str
    to_hash: str | None
    status: str  # proved | done | late | skipped | absent
    moved: int = 0
    stale: int = 0
    stored: str | None = None
    left_on_from: int = 0
    open_task: bool = False  # open after this load: the spec's visibility if it has one, else the stored one
    plan_applied: bool = False  # the plan had been applied in this database before this load
    reason: str | None = None

    @property
    def fatal_skip(self) -> bool:
        """A skipped pair of an open task in a database the plan has not migrated yet: its runs on `from` would drop
        off a public board (review R2). Once the plan is applied, a skip is a later spec edit, not a lag."""
        return self.status == "skipped" and self.open_task and not self.plan_applied

    def text(self) -> str:
        head = f"task {self.ref} {self.from_hash[:12]} -> {(self.to_hash or '?')[:12]}"
        if self.status == "skipped":
            why = self.reason or f"stored hash {(self.stored or '')[:12]} is neither"
            return (f"{head}: skipped, {why}; {self.left_on_from} run(s) on from left"
                    f"{' (open task)' if self.open_task else ''}{' (plan applied earlier)' if self.plan_applied else ''}")
        if self.status == "absent":
            return f"{head}: not in this database"
        label = {"proved": "proved", "done": "already applied", "late": "already applied, late runs moved"}[self.status]
        extra = f", {self.left_on_from} run(s) on from without a proof record left" if self.left_on_from else ""
        return f"{head}: {label}, {self.moved} run(s) moved, {self.stale} stale run(s) kept{extra}"


@dataclass
class RenameReport:
    plans: list[str] = field(default_factory=list)
    tasks: list[TaskLine] = field(default_factory=list)
    models: list[tuple[str, str, str, int, bool]] = field(default_factory=list)  # plan, old, new, runs, row dropped
    runs_touched: int = 0
    keys_changed: int = 0
    runs_total: int = 0
    unique_keys: int = 0
    listed_before: int = 0
    listed_after: int = 0
    backup: Path | None = None
    scan_hits: list[str] = field(default_factory=list)
    scan_notes: list[str] = field(default_factory=list)
    lag: list[str] = field(default_factory=list)

    @property
    def open_skips(self) -> list[TaskLine]:
        return [t for t in self.tasks if t.fatal_skip]

    def lines(self) -> list[str]:
        if not self.plans:
            return ["no rename plans"]
        out = [f"plan {p} ({APPROVED_PLANS[p]})" for p in self.plans]
        out += [t.text() for t in self.tasks]
        out += [f"model {old} -> {new}: {n} run(s) moved{', old row dropped' if dropped else ''}"
                for _, old, new, n, dropped in self.models]
        out.append(f"runs touched {self.runs_touched}, run keys changed {self.keys_changed}, "
                   f"unique keys {self.unique_keys} of {self.runs_total} runs")
        out.append(f"listed runs {self.listed_before} before, {self.listed_after} after")
        if self.backup:
            out.append(f"backup {self.backup}")
        out.append("scan hits: " + ("none" if not self.scan_hits else "; ".join(self.scan_hits)))
        if self.scan_notes:
            out.append("scan notes (history, not fatal): " + "; ".join(self.scan_notes))
        out.append("lag after apply: " + ("none" if not self.lag else "; ".join(self.lag)))
        return out


# ---------------------------------------------------------------------------------------------------------------------
# guards


def check_forward(session: Session, plans: list[Plan]) -> None:
    """Refuse a database that a plan migrated when this checkout lacks that plan (review R5): older code would put
    the old model rows and spec hashes back and unlist every moved run."""
    unknown = applied_plan_ids(session) - {p.id for p in plans}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise RenameRefused(f"this database was migrated by rename plan {names}, which this checkout does not have "
                            f"(registry/renames/{sorted(unknown)[0]}.yaml): rebase or pull onto it before using it")


def rename_lag(session: Session, plans: list[Plan], *, explicit_only: bool = False) -> list[str]:
    """What this database still holds from the plans: model rows or runs on an old slug, and runs on a `from` hash of
    a covered task. Without explicit_only, also any model in a retired namespace or with the retired provider, listed
    in the plan or not (review R6)."""
    out = []
    for plan in plans:
        old = set(plan.models)
        flagged = {slug: "" for slug in old}
        if not explicit_only:
            for slug, provider in session.query(Model.slug, Model.provider):
                retired = bool(plan.old_namespaces) and slug.startswith(plan.old_namespaces)
                retired = retired or (plan.old_provider is not None and provider == plan.old_provider)
                if slug not in old and retired:
                    flagged[slug] = " (not in the plan)"
            if plan.old_namespaces:
                for (slug,) in session.query(Run.model_slug).distinct():
                    if slug not in flagged and slug.startswith(plan.old_namespaces):
                        flagged[slug] = " (not in the plan)"
        rows = {slug for (slug,) in session.query(Model.slug).filter(Model.slug.in_(list(flagged)))}
        for slug in sorted(rows):
            out.append(f"{plan.id}: model row {slug}{flagged[slug]}")
        for slug, n in sorted(_count_by(session, Run.model_slug, list(flagged)).items()):
            out.append(f"{plan.id}: {n} run(s) on {slug}{flagged[slug]}")
        for ref, pairs in plan.tasks.items():
            task = _task_row(session, ref)
            if task is None:
                continue
            for src, dst in pairs:
                if task.spec_hash not in (src, dst):
                    continue  # the spec moved on since: runs on `from` are stale, as after any spec edit
                n = session.query(Run).filter(Run.task_id == task.id, Run.task_spec_hash == src).count()
                if n:
                    out.append(f"{plan.id}: {n} run(s) on {ref} at {src[:12]}")
    return out


def _count_by(session: Session, column, values: list[str]) -> dict[str, int]:
    from sqlalchemy import func

    if not values:
        return {}
    return dict(session.query(column, func.count()).filter(column.in_(values)).group_by(column).all())


def _pending_moves(session: Session, plans: list[Plan]) -> bool:
    """Whether applying the plans would move or drop anything here (so a backup is taken first, review R10)."""
    for plan in plans:
        old = list(plan.models)
        if session.query(Model).filter(Model.slug.in_(old)).count() or _count_by(session, Run.model_slug, old):
            return True
        for ref, pairs in plan.tasks.items():
            task = _task_row(session, ref)
            if task is None:
                continue
            for src, dst in pairs:
                moves = task.spec_hash == src or (dst is not None and task.spec_hash == dst
                                                  and _proved(session, plan.id, ref, src, dst))
                if moves and session.query(Run).filter(Run.task_id == task.id, Run.task_spec_hash == src).count():
                    return True
    return False


def sqlite_file(session_or_url) -> Path | None:
    """The database file behind a session or URL, or None for an in-memory or non-SQLite database."""
    from sqlalchemy.engine import make_url

    url = session_or_url.get_bind().url if isinstance(session_or_url, Session) else make_url(str(session_or_url))
    if url.get_backend_name() != "sqlite" or not url.database or url.database == ":memory:":
        return None
    if url.database.startswith("file:") or "uri" in url.query:
        raise UnsupportedURL(f"{url.render_as_string(hide_password=True)}: give a plain sqlite:///<path> URL "
                             f"(the rename commands open the file read-only themselves)")
    return Path(url.database)


def backup_file(src: Path, dest: Path) -> Path:
    """SQLite online backup of src (opened read-only) to dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(f"file:{Path(src).resolve()}?mode=ro", uri=True)
    try:
        target = sqlite3.connect(dest)
        try:
            source.backup(target)
        finally:
            target.close()
    finally:
        source.close()
    return dest


def backup_before_moves(session: Session, plans: list[Plan]) -> Path | None:
    """Back up the database file next to it, in backups/pre-<plan id>-<UTC time>.db, before the first move."""
    src = sqlite_file(session)
    if src is None or not plans or not _pending_moves(session, plans):
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return backup_file(src, src.parent / "backups" / f"pre-{'+'.join(p.id for p in plans)}-{stamp}.db")


def _sql_text(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def guard_trigger_sql(plan: Plan) -> str | None:
    """A trigger that aborts any insert of a model the plan retired (its old slugs, its old namespace, its old
    provider). Installed with the plan marker, so code older than the forward guard (check_forward), which would put
    the old model rows back and empty the boards, fails on its first models upsert instead (review of B1, R5)."""
    conditions = []
    if plan.models:
        conditions.append("NEW.slug IN (" + ", ".join(_sql_text(s) for s in sorted(plan.models)) + ")")
    conditions += [f"substr(NEW.slug, 1, {len(ns)}) = {_sql_text(ns)}" for ns in plan.old_namespaces]
    if plan.old_provider is not None:
        conditions.append(f"NEW.provider = {_sql_text(plan.old_provider)}")
    if not conditions:
        return None
    name = "rename_guard_" + re.sub(r"[^0-9A-Za-z]", "_", plan.id)
    message = (f"rename plan {plan.id} retired this model slug or provider: this code predates the plan, rebase or "
               f"pull onto it (registry/renames/{plan.id}.yaml)")
    return (f'CREATE TRIGGER IF NOT EXISTS "{name}" BEFORE INSERT ON models WHEN {" OR ".join(conditions)} '
            f"BEGIN SELECT RAISE(ABORT, {_sql_text(message)}); END")


def install_guard(session: Session, plans: list[Plan]) -> None:
    if session.get_bind().dialect.name != "sqlite":
        return
    for plan in plans:
        sql = guard_trigger_sql(plan)
        if sql:
            session.execute(text(sql))


# ---------------------------------------------------------------------------------------------------------------------
# apply


def apply_renames(session: Session, registry_dir: Path, plans: list[Plan], models_yaml: list[dict]) -> RenameReport:
    """Prove and apply the plans inside the caller's transaction (load_registry, after the models upsert and before
    the tasks upsert). Raises RenameRefused; the caller rolls back."""
    from benchtoward.registry import MODEL_FIELDS, task_spec_files

    report = RenameReport(plans=[p.id for p in plans])
    if not plans:
        return report
    files = task_spec_files(registry_dir)
    by_slug = {m["slug"]: m for m in models_yaml}
    applied_before = applied_plan_ids(session)
    renamed = {old: new for p in plans for old, new in p.models.items()}
    with session.no_autoflush:
        state_before = _listing_state(session, renamed)
        overrides: dict[int, str] = {}  # task id -> the hash the tasks upsert will store
        mapped: dict[tuple[int, str], str] = {}  # (task id, from) -> to, for every pair whose runs move
        touched: dict[str, dict] = {}

        def touch(run: Run, plan: Plan) -> None:
            touched.setdefault(run.id, {"run": run, "plan": plan.id, "run_key": run.run_key,
                                        "task_spec_hash": run.task_spec_hash, "model_slug": run.model_slug})

        for plan in plans:
            # models: the new entry must equal the old row except for the plan's fields. Checked only while
            # something is left to move, so a later alias or removal of the new slug does not block every load.
            for old, new in plan.models.items():
                if old in by_slug:
                    raise RenameRefused(f"rename plan {plan.id}: models.yaml still lists {old} (renamed to {new})")
                old_row = session.get(Model, old)
                if old_row is None and session.query(Run.id).filter_by(model_slug=old).first() is None:
                    continue
                if new not in by_slug:
                    raise RenameRefused(f"rename plan {plan.id}: models.yaml has no {new} (renamed from {old})")
                new_row = session.get(Model, new)
                if old_row is not None:
                    for name in MODEL_FIELDS:
                        before, after = getattr(old_row, name), getattr(new_row, name)
                        if name in plan.model_fields:
                            if (before, after) != plan.model_fields[name] and before != after:
                                raise RenameRefused(f"rename plan {plan.id}: model {old} -> {new}: {name} is "
                                                    f"{before!r} -> {after!r}, the plan allows "
                                                    f"{plan.model_fields[name][0]!r} -> {plan.model_fields[name][1]!r}")
                        elif before != after:
                            raise RenameRefused(f"rename plan {plan.id}: model {old} -> {new}: {name} differs "
                                                f"({before!r} -> {after!r})")
            # tasks: prove each pair whose `from` is the stored hash
            for ref, pairs in plan.tasks.items():
                key = _task_ref(ref)
                spec = yaml.safe_load(files[key].read_text(encoding="utf-8")) if key in files else None
                task = _task_row(session, ref)
                for src, dst in pairs:
                    if dst is None:
                        raise RenameRefused(f"rename plan {plan.id}: {ref} has no `to` hash "
                                            f"(benchtoward rename-rekey --write-to-hashes)")
                    line = TaskLine(plan.id, ref, src, dst, "absent", plan_applied=plan.id in applied_before)
                    report.tasks.append(line)
                    if task is None:
                        continue
                    visibility = (spec or {}).get("visibility", task.visibility)
                    line.stored, line.open_task = task.spec_hash, visibility != "hidden"
                    on_from = session.query(Run).filter(Run.task_id == task.id, Run.task_spec_hash == src).all()
                    if task.spec_hash == src:
                        if spec is None:
                            raise RenameRefused(f"rename plan {plan.id}: {ref} has no spec file in "
                                                f"{registry_dir}/tasks")
                        new_hash = spec_hash(_strip(spec))
                        if new_hash != dst:
                            if line.open_task:
                                raise RenameRefused(f"rename plan {plan.id}: {ref}: the registry spec hashes to "
                                                    f"{new_hash[:12]}, the plan says {dst[:12]}")
                            # a hidden build edited its spec after the plan: its runs on `from` go stale with that
                            # edit anyway, so nothing moves and nothing is proved
                            line.status, line.left_on_from = "skipped", len(on_from)
                            line.reason = f"the registry spec hashes to {new_hash[:12]}, not the plan's to (edited since)"
                            continue
                        old_spec = _normalize(_strip(plan.apply(json.loads(task.spec_json))))
                        diff = first_difference(old_spec, _normalize(_strip(spec)))
                        if diff is not None:
                            raise RenameRefused(f"rename plan {plan.id}: {ref} differs from its stored spec by more "
                                                f"than the rename, first at {diff}")
                        if not _proved(session, plan.id, ref, src, dst):
                            session.add(RenameProof(plan_id=plan.id, task_ref=ref, from_hash=src, to_hash=dst))
                        line.status = "proved"
                        overrides[task.id] = dst
                    elif task.spec_hash == dst:
                        if on_from and not _proved(session, plan.id, ref, src, dst):
                            line.status, line.left_on_from = "done", len(on_from)
                            on_from = []
                        else:
                            line.status = "late" if on_from else "done"
                    else:
                        line.status, line.left_on_from = "skipped", len(on_from)
                        continue
                    if on_from:
                        mapped[(task.id, src)] = dst
                    for run in on_from:
                        touch(run, plan)
                        run.task_spec_hash = dst
                    line.moved = len(on_from)
                    line.stale = session.query(Run).filter(Run.task_id == task.id,
                                                           Run.task_spec_hash.notin_([src, dst])).count()
            # model slugs: move runs whether or not the old row still exists, then drop the old row
            for old, new in plan.models.items():
                runs = session.query(Run).filter_by(model_slug=old).all()
                for run in runs:
                    touch(run, plan)
                    run.model_slug = new
                old_row = session.get(Model, old)
                if old_row is not None:
                    session.delete(old_row)
                if runs or old_row is not None:
                    report.models.append((plan.id, old, new, len(runs), old_row is not None))

        # run keys of the touched runs only, from each run's own spec hash
        digests = {slug: digest for slug, digest in session.query(Model.slug, Model.digest)}
        for old in renamed:
            digests.pop(old, None)
        keys = {run_id: key for run_id, key in session.query(Run.id, Run.run_key)}
        for run_id, info in touched.items():
            run = info["run"]
            if run.model_slug not in digests:
                raise RenameRefused(f"run {run.id} would point at a missing model {run.model_slug}")
            keys[run_id] = run_key(digests[run.model_slug], run.task_spec_hash, run.harness_version,
                                   json.loads(run.config_json or "{}"))
        seen: dict[str, str] = {}
        for run_id, key in sorted(keys.items()):
            if key in seen:
                raise RenameRefused(f"runs {seen[key]} and {run_id} would share run key {key[:12]}")
            seen[key] = run_id
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for run_id, info in touched.items():
            run = info["run"]
            internal = json.loads(run.internal_json) if run.internal_json else {}
            if not isinstance(internal, dict):
                raise RenameRefused(f"run {run.id}: internal_json is not an object")
            internal.setdefault("renames", []).append(
                {"plan": info["plan"], "run_key": info["run_key"], "task_spec_hash": info["task_spec_hash"],
                 "model_slug": info["model_slug"], "at": stamp})
            run.internal_json = json.dumps(internal, ensure_ascii=False)
            if keys[run_id] != run.run_key:
                run.run_key = keys[run_id]
                report.keys_changed += 1
        report.runs_touched = len(touched)
        report.runs_total, report.unique_keys = len(keys), len(seen)

    session.flush()
    left = session.query(Model.slug).filter(Model.slug.in_(list(renamed))).all()
    if left:
        raise RenameRefused(f"old model rows remain: {', '.join(s for (s,) in left)}")
    # the same runs are listed as before, read with the moved pairs and slugs applied (a late run is meant to appear)
    listed_before = _listed(state_before, overrides, mapped)
    listed_after = _listed(_listing_state(session), overrides)
    if listed_after != listed_before:
        added, dropped = sorted(listed_after - listed_before), sorted(listed_before - listed_after)
        raise RenameRefused(f"the listed runs would change: added {added}, dropped {dropped}")
    report.listed_before, report.listed_after = len(listed_before), len(listed_after)
    for plan in plans:
        if session.query(RenameProof).filter_by(plan_id=plan.id, task_ref=PLAN_MARKER).one_or_none() is None:
            session.add(RenameProof(plan_id=plan.id, task_ref=PLAN_MARKER, from_hash="", to_hash=""))
    session.flush()
    install_guard(session, plans)  # after the marker's insert, so it is part of this transaction
    return report


# ---------------------------------------------------------------------------------------------------------------------
# scan


HANDLED_COLUMNS = {("models", "slug"), ("runs", "model_slug"), ("runs", "task_spec_hash"), ("tasks", "spec_hash")}


def scan_old_values(session: Session, plans: list[Plan]) -> tuple[list[str], list[str]]:
    """Every text column of every table, searched for the plans' old slugs and `from` hashes. Columns the plan moves
    are covered by rename_lag; the history it writes (runs.internal_json "renames", rename_proofs) is skipped.

    Returns (hits, notes). A hit is a cell whose whole value is an old slug or hash: an identifier the plan does not
    move (a table added later that keys on a slug), which fails --check. A note is a mention inside longer text or
    JSON (agent messages, build steps, page paths, paper facts): history that keeps the old names, reported only, or
    every later deploy would stop at the preflight."""
    needles = sorted({o for p in plans for o in p.models} | {s for p in plans for pairs in p.tasks.values()
                                                             for s, _ in pairs})
    if not needles:
        return [], []
    conn = session.connection()
    params = {f"n{i}": n for i, n in enumerate(needles)}
    listed = ", ".join(f":n{i}" for i in range(len(needles)))

    def short(found: list[str]) -> str:
        names = ", ".join(n[:12] if HASH_RE.fullmatch(n) else n for n in found[:5])
        return names + (" ..." if len(found) > 5 else "")

    hits, notes = [], []
    for (table,) in conn.execute(text("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE "
                                      "'sqlite_%' ORDER BY name")):
        if table == "rename_proofs":
            continue
        for column in conn.execute(text(f'PRAGMA table_info("{table}")')).mappings():
            name, kind = column["name"], (column["type"] or "").upper()
            if (table, name) in HANDLED_COLUMNS or not any(t in kind for t in ("CHAR", "TEXT", "CLOB")):
                continue
            col = f'"{name}"'
            exact = conn.execute(text(f'SELECT {col}, COUNT(*) FROM "{table}" WHERE {col} IN ({listed}) '
                                      f"GROUP BY {col}"), params).all()
            if exact:
                hits.append(f"{table}.{name}: {sum(n for _, n in exact)} row(s) name {short(sorted(v for v, _ in exact))}")
            mentions = " OR ".join(f"instr({col}, :n{i}) > 0" for i in range(len(needles)))
            rows = conn.execute(text(f'SELECT {col} FROM "{table}" WHERE ({mentions}) AND {col} NOT IN ({listed})'),
                                params).scalars().all()
            found: set[str] = set()
            count = 0
            for value in rows:
                if (table, name) == ("runs", "internal_json"):
                    try:
                        doc = json.loads(value)
                    except ValueError:
                        doc = value
                    if isinstance(doc, dict):
                        doc = {k: v for k, v in doc.items() if k != "renames"}
                    value = json.dumps(doc, ensure_ascii=False)
                named = {n for n in needles if n in value}
                found |= named
                count += bool(named)
            if count:
                notes.append(f"{table}.{name}: {count} row(s) mention {short(sorted(found))}")
    return hits, notes


# ---------------------------------------------------------------------------------------------------------------------
# authoring helper


def write_to_hashes(plan_path: Path, registry_dir: Path) -> list[str]:
    """Fill each pair's `to` with the hash of the registry spec as it stands (B2 authoring helper). Pairs are written
    in flow style, `- {from: <hash>, to: <hash or null>}`; the rest of the file is kept as written."""
    plan_path = Path(plan_path)
    text_ = plan_path.read_text(encoding="utf-8")
    plan = read_plan(plan_path)
    from benchtoward.registry import task_spec_files

    files = task_spec_files(registry_dir)
    out = []
    for ref, pairs in plan.tasks.items():
        key = _task_ref(ref)
        if key not in files:
            raise RenameRefused(f"{ref} has no spec file in {registry_dir}/tasks")
        new_hash = spec_hash(_strip(yaml.safe_load(files[key].read_text(encoding="utf-8"))))
        for src, dst in pairs:
            pattern = re.compile(r"(\{\s*from:\s*['\"]?" + src + r"['\"]?\s*,\s*to:\s*)([^}]*?)(\s*\})")
            text_, n = pattern.subn(lambda m: m.group(1) + new_hash + m.group(3), text_)
            if n != 1:
                raise RenameRefused(f"{ref}: expected one flow-style pair with from {src[:12]}, found {n}")
            out.append(f"{ref} {src[:12]} -> {new_hash[:12]}" + ("" if dst in (None, new_hash) else
                                                                 f" (was {str(dst)[:12]})"))
    plan_path.write_text(text_, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------------------------------------------------
# command helpers (benchtoward rename-rekey)


def dry_run(db_url: str, registry_dir: Path) -> RenameReport:
    """load_registry with the plans on an online-backup copy of the database, rolled back; the database itself is only
    read (review R9: a check never migrates the live database while the old container serves)."""
    import tempfile

    from benchtoward.db import init_db, make_engine, session_factory
    from benchtoward.registry import load_registry

    src = sqlite_file(db_url)
    if src is not None and not src.is_file():
        raise NoDatabase(f"no database at {src}: nothing to check")
    plans = load_plans(registry_dir)
    if not plans:  # nothing to apply; still refuse a database that a plan this code lacks migrated
        with _readonly_session(db_url) as session:
            check_forward(session, plans)
        return RenameReport()
    with tempfile.TemporaryDirectory(prefix="rename-dry-run-") as tmp:
        url = db_url
        if src is not None:
            url = f"sqlite:///{backup_file(src, Path(tmp) / 'copy.db')}"
        engine = make_engine(url)
        try:
            init_db(engine)
            with session_factory(engine)() as session:
                try:
                    report = load_registry(session, registry_dir, commit=False)
                    report.lag = rename_lag(session, plans)
                    report.scan_hits, report.scan_notes = scan_old_values(session, plans)
                finally:
                    session.rollback()
        finally:
            engine.dispose()
    return report


def _readonly_session(db_url: str) -> Session:
    from sqlalchemy import create_engine

    from benchtoward.db import make_engine

    src = sqlite_file(db_url)
    if src is None:
        return Session(make_engine(db_url))
    if not src.is_file():
        raise NoDatabase(f"no database at {src}")
    uri = f"file:{src.resolve()}?mode=ro"
    return Session(create_engine("sqlite://", creator=lambda: sqlite3.connect(uri, uri=True)))


def status_lines(db_url: str, plans: list[Plan]) -> tuple[list[str], bool]:
    """Applied plans, proof records and lag of a database, read-only. Second value: whether anything is left to do."""
    with _readonly_session(db_url) as session:
        applied = applied_plan_ids(session)
        lines, failed = [], False
        for plan_id in sorted(applied - {p.id for p in plans}):
            lines.append(f"plan {plan_id}: applied here, but this checkout does not have it")
            failed = True
        if not plans:
            return (lines or ["no rename plans"]), failed
        for plan in plans:
            proofs = 0
            if _has_proofs_table(session):
                proofs = session.query(RenameProof).filter(RenameProof.plan_id == plan.id,
                                                           RenameProof.task_ref != PLAN_MARKER).count()
            lines.append(f"plan {plan.id}: {'applied' if plan.id in applied else 'not applied'}, "
                         f"{proofs} task proof(s) recorded")
        lag = rename_lag(session, plans)
        lines.append("lag: " + ("none" if not lag else "; ".join(lag)))
        return lines, failed or bool(lag)
