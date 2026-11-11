from datetime import date, datetime, timezone

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Dataset(Base):
    __tablename__ = "datasets"
    slug: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String)
    modality: Mapped[str] = mapped_column(String)
    license: Mapped[str] = mapped_column(String)
    license_url: Mapped[str] = mapped_column(String)
    source_url: Mapped[str] = mapped_column(String)
    citation: Mapped[str] = mapped_column(Text)
    khdp_code: Mapped[str | None] = mapped_column(String, nullable=True)
    version: Mapped[str] = mapped_column(String)
    snuh_confirmed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)  # a maintainer confirmed it is SNUH patient data


class Task(Base):
    __tablename__ = "tasks"
    __table_args__ = (UniqueConstraint("slug", "version"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    slug: Mapped[str] = mapped_column(String, index=True)
    version: Mapped[int] = mapped_column(Integer)
    dataset_slug: Mapped[str] = mapped_column(ForeignKey("datasets.slug"))
    title: Mapped[str] = mapped_column(String)
    kind: Mapped[str] = mapped_column(String)
    primary_metric: Mapped[str] = mapped_column(String)
    spec_json: Mapped[str] = mapped_column(Text)
    spec_hash: Mapped[str] = mapped_column(String)
    preview: Mapped[bool] = mapped_column(Boolean, default=False)
    visibility: Mapped[str] = mapped_column(String, default="open")
    made_by: Mapped[str | None] = mapped_column(String, nullable=True)  # "agent" for tasks the paper agent made
    source_paper: Mapped[str | None] = mapped_column(String, nullable=True)  # DOI the task was defined from


class Model(Base):
    __tablename__ = "models"
    slug: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    type: Mapped[str] = mapped_column(String)  # llm | asr | research | baseline
    provider: Mapped[str] = mapped_column(String)  # serving route (registry/providers.yaml); baselines: benchtoward
    provider_model_id: Mapped[str | None] = mapped_column(String, nullable=True)
    digest: Mapped[str] = mapped_column(String)
    price_in_usd_per_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_out_usd_per_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_usd_per_audio_min: Mapped[float | None] = mapped_column(Float, nullable=True)  # speech models
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    max_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reference: Mapped[bool | None] = mapped_column(Boolean, nullable=True)  # shown as a reference line, not ranked
    developer: Mapped[str | None] = mapped_column(String, nullable=True)  # shown publicly; provider is internal
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class Run(Base):
    __tablename__ = "runs"
    id: Mapped[str] = mapped_column(String(12), primary_key=True)
    run_key: Mapped[str] = mapped_column(String, unique=True)
    model_slug: Mapped[str] = mapped_column(ForeignKey("models.slug"))
    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id"))
    harness_version: Mapped[str] = mapped_column(String)
    config_json: Mapped[str] = mapped_column(Text)
    tier: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String)  # running | done | failed
    eval_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    n_items: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    gpu_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    task_spec_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    internal_json: Mapped[str | None] = mapped_column(Text, nullable=True)  # never public (e.g. publish checks)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ItemResult(Base):
    __tablename__ = "item_results"
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    instance_id: Mapped[str] = mapped_column(String, primary_key=True)
    prediction: Mapped[str | None] = mapped_column(Text, nullable=True)
    label: Mapped[str | None] = mapped_column(String, nullable=True)
    correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    transient: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    raw: Mapped[str | None] = mapped_column(Text, nullable=True)  # model output, kept so answers can be re-parsed
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)


class ReuseEvent(Base):
    __tablename__ = "reuse_events"
    __table_args__ = (Index("uq_reuse_daily", "day", "visitor", "kind", "run_id", unique=True),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"))
    kind: Mapped[str] = mapped_column(String)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    day: Mapped[str | None] = mapped_column(String, nullable=True)
    visitor: Mapped[str | None] = mapped_column(String, nullable=True)


class DownloadEvent(Base):
    """Daily deduplicated download reuses. New downloads use user_id and a user marker in visitor;
    legacy rows retain their daily visitor hash. Deletion removes user_id and randomises each visitor marker.
    A separate table supports both run files and task-level leaderboard exports."""
    __tablename__ = "download_events"
    __table_args__ = (UniqueConstraint("day", "visitor", "kind", "target"),
                     Index("uq_download_user_daily", "day", "user_id", "kind", "target", unique=True))
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    day: Mapped[str] = mapped_column(String, index=True)  # YYYY-MM-DD in Asia/Seoul
    visitor: Mapped[str] = mapped_column(String)
    kind: Mapped[str] = mapped_column(String)  # predictions | result_json | leaderboard_csv
    target: Mapped[str] = mapped_column(String)  # run id, or slug@version for a leaderboard CSV
    agent_family: Mapped[str | None] = mapped_column(String, nullable=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)


class User(Base):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("provider", "provider_subject"), {"sqlite_autoincrement": True})
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String)
    provider_subject: Mapped[str] = mapped_column(String)
    email: Mapped[str] = mapped_column(String)
    name: Mapped[str] = mapped_column(String)
    affiliation: Mapped[str | None] = mapped_column(String, nullable=True)
    organisation_type: Mapped[str | None] = mapped_column(String, nullable=True)
    consented_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_login_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_active_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, default=utcnow)


class UserConsent(Base):
    """Versioned decisions: privacy_notice acknowledges contract processing; it is not consent.
    Append decisions so withdrawal and subsequent grants retain their own timestamps."""
    __tablename__ = "user_consents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    item: Mapped[str] = mapped_column(String)
    policy_version: Mapped[str] = mapped_column(String)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    accepted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    withdrawn_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AuthSession(Base):
    __tablename__ = "auth_sessions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # hash of cookie; never the bearer value
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    csrf_token: Mapped[str] = mapped_column(String)
    payload: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class DownloadQuota(Base):
    __tablename__ = "download_quotas"
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    day: Mapped[str] = mapped_column(String, primary_key=True)
    count: Mapped[int] = mapped_column(Integer, default=0)


class DeletedIdentityQuota(Base):
    """Today's download count of a deleted account, keyed by an HMAC of provider and subject (no email, no name),
    so deleting and registering again does not reset the daily limit. Rows of earlier days are purged."""
    __tablename__ = "deleted_identity_quotas"
    identity_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    day: Mapped[str] = mapped_column(String, primary_key=True)
    count: Mapped[int] = mapped_column(Integer, default=0)


class Paper(Base):
    """A paper the agent may validate. Holds metadata and the agent's own extraction, never the paper's text."""
    __tablename__ = "papers"
    doi: Mapped[str] = mapped_column(String, primary_key=True)
    source: Mapped[str] = mapped_column(String)  # where the paper agent found it (journal archive, arxiv, ...)
    title: Mapped[str] = mapped_column(Text)
    authors_json: Mapped[str] = mapped_column(Text, default="[]")
    venue: Mapped[str | None] = mapped_column(String, nullable=True)
    published: Mapped[date | None] = mapped_column(Date, nullable=True)
    paper_license: Mapped[str | None] = mapped_column(String, nullable=True)
    code_urls_json: Mapped[str] = mapped_column(Text, default="[]")
    wiki_path: Mapped[str | None] = mapped_column(String, nullable=True)
    fulltext_path: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="queued")  # queued|working|validated|stopped|held|skipped
    stage: Mapped[int] = mapped_column(Integer, default=0)
    fit_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    public_json: Mapped[str | None] = mapped_column(Text, nullable=True)  # summary, glance, exposure (our words)
    internal_json: Mapped[str | None] = mapped_column(Text, nullable=True)  # task definition, reported metrics
    hidden: Mapped[bool] = mapped_column(Boolean, default=False)  # set by a maintainer on the admin page
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class PaperAttempt(Base):
    __tablename__ = "paper_attempts"
    id: Mapped[str] = mapped_column(String(12), primary_key=True)
    paper_doi: Mapped[str] = mapped_column(ForeignKey("papers.doi"), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    stage_reached: Mapped[int] = mapped_column(Integer, default=1)
    outcome: Mapped[str | None] = mapped_column(String, nullable=True)  # published | held | stopped
    stop_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    facts_json: Mapped[str] = mapped_column(Text, default="{}")
    gpu_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    wall_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    pod_id: Mapped[str | None] = mapped_column(String, nullable=True)
    pod_started_at: Mapped[float | None] = mapped_column(Float, nullable=True)  # epoch seconds when the current compute pod started
    run_id: Mapped[str | None] = mapped_column(String, nullable=True)
    task_slug: Mapped[str | None] = mapped_column(String, nullable=True)
    runbook_version: Mapped[str | None] = mapped_column(String, nullable=True)
    claude_model: Mapped[str | None] = mapped_column(String, nullable=True)


class AgentEvent(Base):
    __tablename__ = "agent_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    paper_doi: Mapped[str] = mapped_column(String, index=True)
    stage: Mapped[int] = mapped_column(Integer)
    message: Mapped[str] = mapped_column(Text)
    public: Mapped[bool] = mapped_column(Boolean, default=True)
    extra_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class BenchBuild(Base):
    """A benchmark being built or in review (bench agent), shown on /live and /benchmarks."""
    __tablename__ = "bench_builds"
    key: Mapped[str] = mapped_column(String, primary_key=True)  # task@version, e.g. vitaldb-ioh@1
    title: Mapped[str] = mapped_column(String)
    dataset: Mapped[str] = mapped_column(String, default="")
    adopted: Mapped[str | None] = mapped_column(String, nullable=True)  # the published protocol it adopts
    adopted_url: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="building")  # building | review | done | stopped
    stage: Mapped[int] = mapped_column(Integer, default=1)  # 1 Protocol .. 6 Published
    step: Mapped[str] = mapped_column(Text, default="")
    done: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    public: Mapped[bool] = mapped_column(Boolean, default=True)
    # the dataset this build publishes on KHDP, if any: draft | under review | published (+ DOI and page)
    khdp_code: Mapped[str | None] = mapped_column(String, nullable=True)
    khdp_version: Mapped[str | None] = mapped_column(String, nullable=True)
    khdp_status: Mapped[str | None] = mapped_column(String, nullable=True)
    khdp_doi: Mapped[str | None] = mapped_column(String, nullable=True)
    khdp_url: Mapped[str | None] = mapped_column(String, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class BuildEvent(Base):
    """One line of a benchmark build's log on /live: a new stage, status or step, or a 10% progress milestone."""
    __tablename__ = "build_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String, index=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    stage: Mapped[int] = mapped_column(Integer, default=0)
    message: Mapped[str] = mapped_column(Text)


class OwnerAction(Base):
    """Something a maintainer did on the admin page that the paper agent's own database must apply before the next sync."""
    __tablename__ = "owner_actions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    kind: Mapped[str] = mapped_column(String)  # requeue | publish_run | hide_task | hide_paper | dataset_*
    target: Mapped[str] = mapped_column(String)  # DOI, run id, or slug@version
    done: Mapped[bool] = mapped_column(Boolean, default=False)


class DatasetProposal(Base):
    """An open KHDP dataset the agent wants to use; a maintainer confirms on the admin page that it is SNUH patient data."""
    __tablename__ = "dataset_proposals"
    id: Mapped[str] = mapped_column(String(12), primary_key=True)
    khdp_code: Mapped[str] = mapped_column(String, index=True)
    version: Mapped[str] = mapped_column(String)
    title: Mapped[str] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    license: Mapped[str | None] = mapped_column(String, nullable=True)
    khdp_type: Mapped[str | None] = mapped_column(String, nullable=True)  # uploader affiliation, not the patients' hospital
    page_url: Mapped[str] = mapped_column(String)
    modality: Mapped[str] = mapped_column(String)
    paper_doi: Mapped[str] = mapped_column(String)
    why: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String, default="pending")  # pending | confirmed | rejected
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class DataNeed(Base):
    """A paper that needed a credentialed KHDP dataset; repeated needs suggest building a synthetic version."""
    __tablename__ = "data_needs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    paper_doi: Mapped[str] = mapped_column(String)
    khdp_code: Mapped[str] = mapped_column(String, index=True)
    version: Mapped[str] = mapped_column(String)
    title: Mapped[str] = mapped_column(Text)
    page_url: Mapped[str] = mapped_column(String)
    need: Mapped[str] = mapped_column(Text)
    why_not_open: Mapped[str] = mapped_column(Text)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class PageView(Base):
    """One request to a public page or API route. No raw IP is stored: `visitor` is a salted daily hash."""
    __tablename__ = "page_views"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    day: Mapped[str] = mapped_column(String, index=True)  # YYYY-MM-DD in Asia/Seoul
    visitor: Mapped[str] = mapped_column(String)
    path: Mapped[str] = mapped_column(String)
    source: Mapped[str | None] = mapped_column(String, nullable=True)  # ?ref= or ?utm_source=
    referrer_host: Mapped[str | None] = mapped_column(String, nullable=True)
    kind: Mapped[str] = mapped_column(String)  # human | bot | agent | internal | beacon
    agent_family: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[int | None] = mapped_column(Integer, nullable=True)


class RenameProof(Base):
    """What a committed rename plan (registry/renames/, benchtoward.rename) did to this database. One row per task pair
    whose stored spec differed from the registry spec only by the plan's text rules, so a run that arrives on
    `from_hash` later is moved without a second proof (the old spec text is gone by then). One row with task_ref "*"
    marks that the plan was applied here at all, so code without that plan refuses the database."""
    __tablename__ = "rename_proofs"
    __table_args__ = (UniqueConstraint("plan_id", "task_ref", "from_hash"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plan_id: Mapped[str] = mapped_column(String, index=True)
    task_ref: Mapped[str] = mapped_column(String)  # slug@version, or "*" for the plan marker
    from_hash: Mapped[str] = mapped_column(String)
    to_hash: Mapped[str] = mapped_column(String)
    proved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


def make_engine(db_url: str) -> Engine:
    if db_url in ("sqlite://", "sqlite:///:memory:"):
        return _secure_delete(create_engine(db_url, connect_args={"check_same_thread": False}, poolclass=StaticPool))
    if db_url.startswith("sqlite:///"):
        from pathlib import Path

        path = db_url.removeprefix("sqlite:///")
        if not path.startswith("file:"):  # a URI (sqlite:///file:...?uri=true) is no folder path
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        return _secure_delete(create_engine(db_url, connect_args={"check_same_thread": False, "timeout": 60}))
    return create_engine(db_url)


def _secure_delete(engine: Engine) -> Engine:
    """Deleted rows are overwritten with zeros in the database file, so destroyed personal information cannot be
    recovered from it (privacy policy, Article 6). Set on every new connection: the pragma is per connection."""
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA secure_delete = ON")
        cursor.close()

    return engine


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)
    _add_missing_columns(engine)
    # create_all cannot add a constraint to an existing table. NULLs keep legacy rows outside this dedup window.
    for table in (ReuseEvent.__table__, DownloadEvent.__table__):
        for index in table.indexes:
            index.create(engine, checkfirst=True)


def _add_missing_columns(engine: Engine) -> None:
    """Minimal forward migration: add columns that exist in the models but not in the database (as nullable)."""
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            existing = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name not in existing:
                    kind = column.type.compile(dialect=engine.dialect)
                    conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN "{column.name}" {kind}'))


def session_factory(engine: Engine) -> sessionmaker:
    return sessionmaker(bind=engine, expire_on_commit=False)
