"""Commands for the LYDUS tasks (lydus-ecg12-dx@0).

    benchtoward fetch lydus                          # KHDP_TOKEN; 4.0 GB, sha256-checked
    benchtoward build lydus-ecg12-dx                 # checks registry/splits/lydus-snuh-ecg-50k-1.1.1-split.csv
    benchtoward baseline lydus-ecg12-dx --report out.json
    benchtoward lydus record-predictions --model benchtoward/lydus-resnet1d --file test_predictions.npz
"""
import json
from pathlib import Path

import typer

from benchtoward.tasks.lydus_reporting import metrics as classification_metrics

from benchtoward.config import get_settings

lydus_app = typer.Typer(help="LYDUS-SNUH-ECG-50K tasks", no_args_is_help=True)


def fetch_lydus() -> None:
    """Download LYDUS-SNUH-ECG-50K 1.1.1 (metadata CSV and 500 Hz NPZ) from KHDP and verify sha256."""
    from benchtoward.khdp import KhdpClient
    from benchtoward.tasks import lydus

    settings = get_settings()
    if not settings.khdp_token:
        typer.echo("KHDP_TOKEN is not set")
        raise typer.Exit(1)
    for path in lydus.fetch(KhdpClient(settings.khdp_token), settings.data_dir):
        typer.echo(f"ok {path}")


def build_cmd(write_split: bool = typer.Option(False, help="write the split file instead of checking it (a new "
                                               "split version only)"),
              workers: int = 8) -> None:
    """Encode dataset-provided labels, link likely repeat ECGs, split by group, write the instances."""
    from benchtoward.tasks.lydus_ecg12_dx import build_task, count_mismatches

    try:
        paths = build_task(get_settings().data_dir, write_split=write_split, workers=workers)
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(1)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    typer.echo(f"records {summary['records']}; groups {summary['groups']}; excluded {summary['excluded']}; "
               f"split sha256 {summary['split_sha256']}")
    mismatches = count_mismatches(summary)
    for line in mismatches:
        typer.echo(f"differs from the spec: {line}")
    if mismatches and not write_split:
        raise typer.Exit(1)


def _features(data_dir: Path, inst, workers: int):
    import hashlib

    import numpy as np


    from benchtoward.baselines.lydus_ecg import FEATURE_VERSION, features_for
    from benchtoward.tasks.lydus_ecg12_dx import task_dir

    key = hashlib.sha256(",".join(map(str, inst["row"])).encode()).hexdigest()
    cache = task_dir(data_dir) / f"features-{FEATURE_VERSION}.npz"
    if cache.exists():
        stored = np.load(cache)
        if str(stored["key"]) == key:
            return stored["X"]
    from benchtoward.tasks import lydus

    X = features_for(lydus.dataset_dir(data_dir) / lydus.SIGNAL_FILE, inst["row"].to_numpy(), workers=workers)
    np.savez(cache, key=key, X=X)
    return X


def baseline_cmd(n_boot: int = 1000, workers: int = 8,
                 lead_sets: str = typer.Option("12,6,4,3,2", help="gradient boosting lead sets (12 is recorded)"),
                 leakage: bool = typer.Option(True, help="also fit the report-only leakage checks"),
                 report: Path = typer.Option(None, help="also write reduced lead sets and leakage checks here"),
                 record: bool = typer.Option(True, help="record the runs in the database")) -> None:
    """Prior and beat-feature gradient boosting (12 leads as runs; 6, 4, 3, 2 leads and leakage checks reported)."""
    import numpy as np

    from benchtoward.baselines.lydus_ecg import run_lydus_baselines
    from benchtoward.cli import _summary, open_session
    from benchtoward.registry import resolve_task
    from benchtoward.tasks import lydus
    from benchtoward.tasks.lydus_ecg12_dx import REF, load_instances, load_spec

    data_dir = get_settings().data_dir
    inst = load_instances(data_dir)
    meta = lydus.read_metadata(data_dir).set_index("new_id")
    inst = inst.join(meta[list(lydus.MEASUREMENTS)], on="new_id")
    X = _features(data_dir, inst, workers)
    spec = load_spec()["split"]
    rng = np.random.default_rng(spec["seed"])
    parts = np.array(["train", "validation", "test"], dtype=object)
    record_level = parts[np.searchsorted(np.cumsum(spec["fractions"]), rng.random(len(meta)),
                                         side="right").clip(0, 2)][inst["row"].to_numpy()]
    with open_session() as session:
        task = resolve_task(session, REF, include_hidden=True)
        runs, out = run_lydus_baselines(session, task, inst, X, data_dir=data_dir, n_boot=n_boot, record=record,
                                        lead_sets=tuple(lead_sets.split(",")), leakage=leakage,
                                        record_level_split=record_level if leakage else None)
        for run in runs:
            typer.echo(_summary(run))
    if report:
        report.write_text(json.dumps(out, indent=2, default=float) + "\n")
        typer.echo(f"wrote {report}")


@lydus_app.command("record-predictions")
def record_predictions(model: str = typer.Option(..., help="registry model slug, e.g. benchtoward/lydus-resnet1d"),
                       file: Path = typer.Option(..., help="npz with instance_id and scores (n, 25); validation thresholds in --config"),
                       config: str = typer.Option("{}", help="run config as JSON"),
                       n_boot: int = 1000) -> None:
    """Score and record a run from test-split predictions made elsewhere (e.g. the ResNet trained on a GPU pod)."""
    import numpy as np

    from benchtoward.tasks import lydus_labels as ll
    from benchtoward.baselines.lydus_ecg import run_rows
    from benchtoward.cli import _summary, open_session
    from benchtoward.db import Model
    from benchtoward.evaluate import record_baseline_run
    from benchtoward.registry import resolve_task
    from benchtoward.tasks.lydus_ecg12_dx import REF, label_matrices, load_instances

    data_dir = get_settings().data_dir
    test = load_instances(data_dir).query("split == 'test'").set_index("instance_id")
    data = np.load(file, allow_pickle=False)
    ids = [str(i) for i in data["instance_id"]]
    if sorted(ids) != sorted(test.index):
        typer.echo(f"{file}: predictions must cover exactly the {len(test)} test instances")
        raise typer.Exit(1)
    test = test.loc[ids].reset_index()
    y, known = label_matrices(test)
    scores = np.asarray(data["scores"], float)
    if scores.shape != y.shape or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise typer.BadParameter("scores must be finite [0,1] values with shape (test records, 25)")
    thresholds = json.loads(config).get("thresholds", {})
    outputs = scores >= np.array([thresholds.get(t, 0.5) for t in ll.TARGETS])
    metrics = classification_metrics(y, known, scores, test["group"].to_numpy(object), thresholds=thresholds, n_boot=n_boot) | {
        "lead_set": json.loads(config).get("lead_set", "12")}
    with open_session() as session:
        run = record_baseline_run(session, session.get(Model, model), resolve_task(session, REF, include_hidden=True),
                                  run_rows(test, y, known, outputs, scores), metrics, json.loads(config), data_dir=data_dir)
        typer.echo(_summary(run))


def _resnet_dir(data_dir: Path) -> Path:
    from benchtoward.tasks.lydus_ecg12_dx import engine_dir

    return engine_dir(data_dir) / "resnet-v0"


def _hgb_dir(data_dir: Path) -> Path:
    from benchtoward.tasks.lydus_ecg12_dx import engine_dir

    return engine_dir(data_dir) / "hgb-v0"


def record_level_split(data_dir: Path, rows):
    """The record-level random split of the leakage check (same fractions and seed as the group split)."""
    import numpy as np

    from benchtoward.tasks import lydus
    from benchtoward.tasks.lydus_ecg12_dx import load_spec

    spec = load_spec()["split"]
    n = len(lydus.read_metadata(data_dir))
    rng = np.random.default_rng(spec["seed"])
    parts = np.array(["train", "validation", "test"], dtype=object)
    return parts[np.searchsorted(np.cumsum(spec["fractions"]), rng.random(n), side="right").clip(0, 2)][
        np.asarray(rows, dtype=int)]


def hgb_instances(data_dir: Path):
    """Instances with what the gradient-boosting fits need (and nothing else): row, split, record_level_split,
    targets, unknown_targets, new_id, age, sex and the electrocardiograph's measurements (leakage check only)."""
    from benchtoward.tasks import lydus
    from benchtoward.tasks.lydus_ecg12_dx import load_instances

    inst = load_instances(data_dir)
    meta = lydus.read_metadata(data_dir).set_index("new_id")
    inst = inst.join(meta[list(lydus.MEASUREMENTS)], on="new_id")
    inst["record_level_split"] = record_level_split(data_dir, inst["row"])
    return inst[["row", "split", "record_level_split", "targets", "unknown_targets", "new_id", "age", "sex", *lydus.MEASUREMENTS]]


@lydus_app.command("hgb-record")
def hgb_record(n_boot: int = 1000, report_boot: int = 200,
               report: Path = typer.Option(None, help="write every configuration's metrics here"),
               record: bool = typer.Option(True, help="record the prior and the 12-lead run in the database")) -> None:
    """Score the gradient-boosting pod outputs (thresholds were chosen on validation there); record prior and 12."""
    import numpy as np

    from benchtoward.tasks import lydus_labels as ll
    from benchtoward.baselines.lydus_ecg import (CONFIGS, FEATURE_VERSION, HGB, MIN_POSITIVES, THRESHOLD_RULE, _short,
                                              prior, run_rows, labels_digest)
    from benchtoward.cli import _summary, open_session
    from benchtoward.db import Model
    from benchtoward.evaluate import record_baseline_run
    from benchtoward.registry import resolve_task
    from benchtoward.tasks.lydus_ecg12_dx import REF, label_matrices, load_instances

    data_dir = get_settings().data_dir
    inst = load_instances(data_dir)
    y, known = label_matrices(inst)
    groups = inst["group"].to_numpy(object)
    folder = _hgb_dir(data_dir) / "pod0"
    out, runs = {}, []
    test = (inst["split"] == "test").to_numpy()
    outputs, scores = prior(y[(inst["split"] == "train").to_numpy()],
                           known[(inst["split"] == "train").to_numpy()], int(test.sum()))
    prior_thresholds = {t: 0.0 if outputs[0, k] else 1.0 for k, t in enumerate(ll.TARGETS)}
    metrics = classification_metrics(y[test], known[test], scores, groups[test], thresholds=prior_thresholds,
                         n_boot=n_boot) | {"lead_set": "12"}
    out["prior"] = _short(metrics)
    with open_session() as session:
        task = resolve_task(session, REF, include_hidden=True)
        if record:
            runs.append(record_baseline_run(session, session.get(Model, "benchtoward/prior"), task,
                                            run_rows(inst[test], y[test], known[test], outputs, scores), metrics,
                                            {"baseline": "prior", "outputs": "every target observed positive in train",
                                             "scores": "train prevalence"}, data_dir=data_dir))
        for name in CONFIGS:
            data = np.load(folder / f"preds-{name}.npz")
            if str(data.get("labels_sha256", "")) != labels_digest(inst, y, known):
                raise ValueError(f"preds-{name}.npz: other labels/order/masks; train again")
            idx = data["test_idx"]
            if not name.startswith("record_level") and not np.array_equal(idx, np.flatnonzero(test)):
                raise ValueError(f"preds-{name}.npz: not the test split")
            boot = n_boot if name == "hgb-12" else report_boot
            metrics = classification_metrics(y[idx], known[idx], data["scores"], groups[idx],
                                 thresholds=dict(zip(ll.TARGETS, map(float, data["thresholds"]))), n_boot=boot)
            out[name] = _short(metrics) | {"test_records": int(len(idx))}
            typer.echo(f"{name}: {out[name]}")
            if name == "hgb-12":
                out["hgb-12_per_class"] = metrics["per_target"]
                out["hgb-12_fitted_classes"] = [ll.TARGETS[k] for k in np.flatnonzero(data["fitted"])]
                if record:
                    config = {"baseline": "hgb-beat-features", "feature_version": FEATURE_VERSION, "lead_set": "12",
                              "min_positives": MIN_POSITIVES, "hgb": HGB, "thresholds": THRESHOLD_RULE}
                    runs.append(record_baseline_run(session, session.get(Model, "benchtoward/lydus-hgb-beat-features"),
                                                    task, run_rows(inst[test], y[test], known[test], data["outputs"],
                                                                   data["scores"]),
                                                    metrics | {"lead_set": "12"}, config, data_dir=data_dir))
        for run in runs:
            typer.echo(_summary(run))
    if report:
        report.write_text(json.dumps(out, indent=2, default=float) + "\n")


@lydus_app.command("resnet-record")
def resnet_record(lead_sets: str = "12,6,4,3,2", n_boot: int = 1000, report_boot: int = 200,
                  report: Path = typer.Option(None, help="write every lead set's metrics here"),
                  record: bool = typer.Option(True, help="record the 12-lead run in the database")) -> None:
    """Score the ResNet's pod outputs (thresholds from validation); record the 12-lead run."""
    from benchtoward.baselines.lydus_ecg import THRESHOLD_RULE, _short, run_rows
    from benchtoward.baselines.lydus_resnet import COMMIT, TRAINING, evaluate_lead_set
    from benchtoward.cli import _summary, open_session
    from benchtoward.db import Model
    from benchtoward.evaluate import record_baseline_run
    from benchtoward.registry import resolve_task
    from benchtoward.tasks.lydus_ecg12_dx import REF, label_matrices, load_instances

    data_dir = get_settings().data_dir
    inst = load_instances(data_dir)
    root = _resnet_dir(data_dir)
    folders = {f.name.split("-", 1)[1].removesuffix(".npz"): f.parent for f in root.glob("pod*/scores-*.npz")}
    out = {}
    for lead_set in lead_sets.split(","):
        if lead_set not in folders:
            typer.echo(f"no pod output for lead set {lead_set}")
            continue
        history = json.loads((folders[lead_set] / f"history-{lead_set}.json").read_text())
        boot = n_boot if lead_set == "12" else report_boot
        metrics, outputs, scores, t_idx = evaluate_lead_set(folders[lead_set], lead_set, inst, boot)
        out[lead_set] = _short(metrics) | {"epochs": history["epochs"], "best_val_loss": history["best_val_loss"],
                                           "device": history["device"], "torch": history["torch"]}
        typer.echo(f"lead set {lead_set}: {out[lead_set]}")
        if lead_set == "12":
            out["per_class_12"] = metrics["per_target"]
            if record:
                test = inst.iloc[t_idx].reset_index(drop=True)
                config = {"baseline": "resnet1d", "architecture": f"Ribeiro et al. 2020 ({COMMIT[:7]}), PyTorch port",
                          "lead_set": "12", "training": TRAINING, "epochs": history["epochs"],
                          "thresholds": THRESHOLD_RULE, "device": history["device"]}
                with open_session() as session:
                    run = record_baseline_run(session, session.get(Model, "benchtoward/lydus-resnet1d"),
                                              resolve_task(session, REF, include_hidden=True),
                                              run_rows(test, *label_matrices(test), outputs, scores), metrics, config,
                                              data_dir=data_dir)
                    typer.echo(_summary(run))
    if report:
        report.write_text(json.dumps(out, indent=2, default=float) + "\n")


def register(app: typer.Typer, fetch_app: typer.Typer, build_app: typer.Typer, baseline_app: typer.Typer) -> None:
    fetch_app.command("lydus")(fetch_lydus)
    build_app.command("lydus-ecg12-dx")(build_cmd)
    baseline_app.command("lydus-ecg12-dx")(baseline_cmd)
    app.add_typer(lydus_app, name="lydus")



try:  # the GPU pod commands are part of the deployment, not of the published code
    from benchtoward.tasks.lydus_pods import register_commands as _register_pods  # noqa: E402
except ModuleNotFoundError:
    _register_pods = None
if _register_pods is not None:
    _register_pods(lydus_app)
