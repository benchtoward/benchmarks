"""Commands for ptbxl-superdiagnostic@0 (PTB-XL 1.0.3, PhysioNet, CC BY 4.0).

    benchtoward fetch ptb-xl                          # 1.7 GB ZIP; every file checked against SHA256SUMS.txt
    benchtoward build ptbxl-superdiagnostic           # instances, 500 Hz signals; refuses counts that differ
    benchtoward baseline ptbxl-superdiagnostic --report out.json
"""
import json
from pathlib import Path

import typer

from benchtoward.config import get_settings


def fetch_cmd() -> None:
    """Download PTB-XL 1.0.3 from PhysioNet, extract it and verify every file."""
    from benchtoward.tasks import ptbxl

    typer.echo(f"ok {ptbxl.fetch(get_settings().data_dir)}")


def build_cmd() -> None:
    """Label each record with its diagnostic superclasses, split by fold, write instances and signals."""
    from benchtoward.tasks.ptbxl_superdiagnostic import build_task

    try:
        paths = build_task(get_settings().data_dir)
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(1)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    typer.echo(f"records {summary['records']}; patients {summary['patients']}; excluded {summary['excluded']}")


def baseline_cmd(n_boot: int = 1000, workers: int = 4,
                 report: Path = typer.Option(None, help="also write per-class metrics here"),
                 record: bool = typer.Option(True, help="record the runs in the database")) -> None:
    """The prior and per-class gradient boosting on beat features (train folds 1-8; scored on fold 10)."""
    import hashlib

    import numpy as np

    from benchtoward.baselines import ptbxl_ecg as B
    from benchtoward.cli import _summary, open_session
    from benchtoward.db import Model
    from benchtoward.evaluate import record_baseline_run
    from benchtoward.registry import resolve_task
    from benchtoward.tasks import ptbxl_superdiagnostic as T

    data_dir = get_settings().data_dir
    inst = T.load_instances(data_dir)
    y = T.label_matrix(inst)
    masks = {p: (inst["split"] == p).to_numpy() for p in T.PARTS}
    test = inst[masks["test"]]
    groups = test["patient_id"].to_numpy()
    cache = T.task_dir(data_dir) / "features-lydus-beat-v1.npz"
    key = hashlib.sha256(",".join(map(str, inst["row"])).encode()).hexdigest()
    X = None
    if cache.exists():
        stored = np.load(cache)
        X = stored["X"] if str(stored["key"]) == key else None
    if X is None:
        X = B.features_for(T.task_dir(data_dir) / "signals-500hz.npy", inst["row"].to_numpy(), workers=workers)
        np.savez(cache, key=key, X=X)
    X = B.with_demographics(X, inst)
    out, runs = {}, []
    prior = T.prior_scores(y[masks["train"]], int(masks["test"].sum()))
    m_prior = T.metrics(y[masks["test"]], prior, groups, n_boot=n_boot)
    model = B.fit(X[masks["train"]], y[masks["train"]])
    scores = model.predict_scores(X[masks["test"]])
    m_hgb = T.metrics(y[masks["test"]], scores, groups, n_boot=n_boot)
    out = {"prior": m_prior, "hgb": m_hgb, "test_records": int(masks["test"].sum())}
    for name, m in (("prior", m_prior), ("hgb", m_hgb)):
        a = m["macro_auroc"]
        typer.echo(f"{name}: macro AUROC {a['value']:.4f} ({a['ci_low']:.4f}-{a['ci_high']:.4f})")
    if record:
        with open_session() as session:
            task = resolve_task(session, T.REF, include_hidden=True)
            runs.append(record_baseline_run(session, session.get(Model, "benchtoward/prior"), task,
                                            B.run_rows(test, y[masks["test"]], prior), m_prior,
                                            {"baseline": "prior", "scores": "train prevalence"}, data_dir=data_dir))
            runs.append(record_baseline_run(session, session.get(Model, "benchtoward/ptbxl-hgb-beat-features"), task,
                                            B.run_rows(test, y[masks["test"]], scores), m_hgb,
                                            {"baseline": "hgb-beat-features", "feature_version": B.FEATURE_VERSION,
                                             "hgb": B.HGB, "train_folds": "1-8", "age_sex": True}, data_dir=data_dir))
            for run in runs:
                typer.echo(_summary(run))
    if report:
        report.write_text(json.dumps(out, indent=2, default=float) + "\n")
        typer.echo(f"wrote {report}")


def register(app: typer.Typer, fetch_app: typer.Typer, build_app: typer.Typer, baseline_app: typer.Typer) -> None:
    fetch_app.command("ptb-xl")(fetch_cmd)
    build_app.command("ptbxl-superdiagnostic")(build_cmd)
    baseline_app.command("ptbxl-superdiagnostic")(baseline_cmd)
    try:  # the ResNet reference trains on a private GPU pod; that command stays in the private repository
        from benchtoward.tasks.ptbxl_pods_cli import resnet_cmd
    except ModuleNotFoundError:
        resnet_cmd = None
    if resnet_cmd is not None:
        baseline_app.command("ptbxl-superdiagnostic-resnet")(resnet_cmd)
