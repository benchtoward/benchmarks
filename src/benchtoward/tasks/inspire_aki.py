"""Postoperative AKI (KDIGO stage 2-3) after noncardiac surgery from pre-operative data, version 0.

The task follows the paper (owner decision 2026-10-30): cohort, labels, inputs, the GS-AKI clinical score and the
patient split are those of the authors' code of Do J et al., JAMIA Open 2026 (StealthX051/Inspire-AKI-prediction, tag
v1.0.0-jamia-open, commit 4dc799c, MIT), which they ran on INSPIRE 1.3, with the deviations listed in
registry/tasks/inspire-aki.yaml. Ported parts:

- cohort/filters.py and cohort/preop.py (build_preop_features): operation filters, derived inputs (op_len and
  booking_case_length among them), the pre-operative laboratory and ward values (merge_asof within 90 days before the
  operation start);
- reporting/procedure_audit.py: its per-operation keep or exclude decision, as a table of the decisions the authors'
  code made on INSPIRE 1.3 (registry/inspire-aki/procedure-audit-inspire-1.3.csv), keyed by what the rules read;
- cohort/labels.py (derive_aki_labels): the KDIGO stage rules with the authors' CRRT rule (any CRRT record of the
  patient, at any time); the earlier BenchToward rule (CRRT only in the 7 days after the operation) for the report;
- clinical_baselines/gs_aki.py: the adapted GS-AKI count;
- evaluation/split_manager.py: the authors' patient-grouped 5-fold split (published files): outer fold 0 is the test
  split, all five folds serve the report.

Data: KHDP `inspire` 1.3 (CC BY 4.0): operations, labs, ward_vitals, diagnosis, and vitals (only which operations
have an intraoperative record, which the authors' cohort requires).
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from benchtoward.config import registry_dir

TASK_KEY = "inspire-aki@0"
CODE, VERSION = "inspire", "1.3"
FILES = ("operations.csv", "labs.csv", "ward_vitals.csv", "diagnosis.csv", "vitals.csv")
STABLE = "stable"
DAY = 24 * 60
OPS_COLUMNS = ["op_id", "subject_id", "age", "sex", "height", "weight", "asa", "emop", "department", "antype",
               "icd10_pcs", "opdate", "orin_time", "orout_time", "opstart_time", "opend_time", "cpbon_time",
               "cpboff_time"]
ANTYPE_CODES = {"General": 0.0, "MAC": 1.0, "Neuraxial": 1.0}  # filters.py; anything else is missing


@lru_cache(maxsize=1)
def load_spec() -> dict:
    return yaml.safe_load((registry_dir() / "tasks" / "inspire-aki.yaml").read_text(encoding="utf-8"))


def dataset_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "datasets" / CODE / VERSION


def task_dir(data_dir: Path) -> Path:
    return dataset_dir(data_dir) / "aki-v0"


def published_split_path() -> Path:
    return registry_dir() / "splits" / "inspire-aki-1.3-patients.csv"


def outer_folds_path() -> Path:
    return registry_dir() / "inspire-aki" / "outer-folds-inspire-1.3.csv"


def audit_table_path() -> Path:
    return registry_dir() / "inspire-aki" / "procedure-audit-inspire-1.3.csv"


def proxy_map_path() -> Path:
    return registry_dir() / "inspire-aki" / "intraperitoneal_proxy_map_5char.csv"


def fetch_inspire_aki(data_dir: Path, client) -> Path:
    """Download the five INSPIRE 1.3 tables the task reads (resumable; a file is complete when its size matches the
    KHDP listing, as the release has no checksums)."""
    root = dataset_dir(data_dir)
    listing = {f.key.removeprefix(f"{VERSION}/"): f for f in client.files(CODE, VERSION)}
    for name in FILES:
        f = listing.get(name)
        if f is None:
            raise FileNotFoundError(f"{name} is not in the KHDP {CODE} {VERSION} release")
        dest = root / name
        if dest.exists() and dest.stat().st_size == f.size:
            continue
        client.download(f, dest)
        if dest.stat().st_size != f.size:
            raise ValueError(f"{name}: downloaded {dest.stat().st_size} bytes, KHDP lists {f.size}")
    return root


# ---- cohort ----

def _count(steps: list, name: str, df: pd.DataFrame) -> None:
    steps.append({"step": name, "operations": len(df), "patients": int(df["subject_id"].nunique())})


def read_audit_table(path: Path | None = None) -> pd.DataFrame:
    table = pd.read_csv(path or audit_table_path(), dtype={"pcs_prefix": str})
    table["is_cts"] = table["is_cts"].astype(str).eq("True")
    table["cpb_present"] = table["cpb_present"].astype(str).eq("True")
    return table


def audit_actions(ops: pd.DataFrame, table: pd.DataFrame) -> pd.Series:
    """The authors' procedure-audit decision (retain or exclude) for each operation, looked up by the three things
    their rules read: the 5-character ICD-10-PCS code, a CTS department and recorded CPB times. A combination the
    table does not hold means a different data release, so it is an error."""
    keys = pd.DataFrame({
        "pcs_prefix": ops["icd10_pcs"].astype("string").str.strip().str.upper(),
        "is_cts": ops["department"].astype("string").str.strip().str.upper().eq("CTS").fillna(False).astype(bool),
        "cpb_present": (ops["cpbon_time"].notna() | ops["cpboff_time"].notna()).astype(bool),
    }, index=ops.index)
    lookup = table.set_index(["pcs_prefix", "is_cts", "cpb_present"])["action"]
    index = pd.MultiIndex.from_frame(keys.astype({"pcs_prefix": object}))
    actions = pd.Series(lookup.reindex(index).to_numpy(), index=ops.index)
    if actions.isna().any():
        missing = keys[actions.isna()].drop_duplicates().head(5).to_dict("records")
        raise ValueError(f"procedure audit table has no decision for {int(actions.isna().sum())} operations, e.g. "
                         f"{missing}; it was made for INSPIRE {VERSION}")
    return actions


def select_operations(ops: pd.DataFrame, diagnosis: pd.DataFrame, audit: pd.DataFrame,
                      spec: dict | None = None) -> tuple[pd.DataFrame, list]:
    """Operations kept by build_preop_features (filters.py, then the antype, department, ICD-10-PCS prefix and
    procedure-audit exclusions), with the derived inputs it computes there. Returns the rows and the step counts."""
    spec = spec or load_spec()
    c = spec["cohort"]["rules"]
    steps: list = []
    df = ops[OPS_COLUMNS].copy()
    _count(steps, "raw_operations", df)
    df = df[df["asa"] < c["max_asa_exclusive"]]
    _count(steps, "asa_lt_6", df)
    df = df[df["age"] >= c["min_age"]]
    _count(steps, "adult", df)
    df = df.dropna(subset=["opend_time", "opstart_time"])
    _count(steps, "has_opstart_and_opend_time", df)
    df = df.assign(op_len=df["opend_time"] - df["opstart_time"])
    df = df[df["op_len"] > 0]
    _count(steps, "positive_op_len", df)
    df = df[df["weight"].notna() & df["height"].notna() & (df["weight"] != 0) & (df["height"] != 0)]
    _count(steps, "nonzero_height_and_weight", df)
    df = df[~df["antype"].isin(c["exclude_antype"]) & ~df["department"].isin(c["exclude_department"])]
    _count(steps, "antype_and_department", df)
    df = df[~df["icd10_pcs"].astype(str).str.startswith(tuple(c["exclude_icd10_pcs_prefixes"]))]
    _count(steps, "icd10_pcs_prefix_exclusions", df)
    df = df[audit_actions(df, audit) == "retain"]
    _count(steps, "procedure_audit", df)

    df = df.copy()
    df["sex"] = (df["sex"] == "M").astype(float)
    df["BSA"] = np.sqrt(df["height"] * df["weight"] / 3600.0)
    df["BMI"] = df["weight"] / (df["height"] / 100.0) ** 2
    df["booking_case_length"] = df["orout_time"] - df["orin_time"]
    df["antype"] = df["antype"].map(ANTYPE_CODES).astype(float)
    df["num_card_events"] = card_event_counts(df, diagnosis, c["cardiovascular_prefix"])
    return df.reset_index(drop=True), steps


def card_event_counts(ops: pd.DataFrame, diagnosis: pd.DataFrame, prefix: str = "I") -> pd.Series:
    """Number of diagnosis rows whose ICD-10 code starts with `prefix`, charted before the operation start."""
    diag = diagnosis[diagnosis["icd10_cm"].astype(str).str.startswith(prefix)][["subject_id", "chart_time"]]
    merged = ops[["op_id", "subject_id", "opstart_time"]].merge(diag, on="subject_id", how="inner")
    counts = merged[merged["chart_time"] < merged["opstart_time"]].groupby("op_id").size()
    return ops["op_id"].map(counts).fillna(0).astype(int).set_axis(ops.index)


def same_day_card_events(ops: pd.DataFrame, diagnosis: pd.DataFrame, prefix: str = "I") -> pd.Series:
    """Of the rows card_event_counts counts, those stamped on the day of the operation (opdate <= chart_time <
    opstart_time). Diagnosis times are whole days, so these may have been made after the start; reported, not used."""
    diag = diagnosis[diagnosis["icd10_cm"].astype(str).str.startswith(prefix)][["subject_id", "chart_time"]]
    merged = ops[["op_id", "subject_id", "opdate", "opstart_time"]].merge(diag, on="subject_id", how="inner")
    same_day = (merged["chart_time"] >= merged["opdate"]) & (merged["chart_time"] < merged["opstart_time"])
    counts = merged[same_day].groupby("op_id").size()
    return ops["op_id"].map(counts).fillna(0).astype(int).set_axis(ops.index)


def latest_before(ops: pd.DataFrame, events: pd.DataFrame, items: list[str], prefix: str,
                  window_min: int) -> pd.DataFrame:
    """For each item, the latest value of the patient charted at or before the operation start and at most
    `window_min` minutes before it (preop.py: merge_asof, direction backward, tolerance). Equal chart times keep the
    later row in file order."""
    out = pd.DataFrame({"op_id": ops["op_id"].to_numpy()})
    left = ops[["op_id", "subject_id", "opstart_time"]].sort_values("opstart_time", kind=STABLE)
    left = left.astype({"opstart_time": float})
    for item in items:
        right = events.loc[events["item_name"] == item, ["subject_id", "chart_time", "value"]]
        right = right.astype({"chart_time": float}).sort_values("chart_time", kind=STABLE)
        merged = pd.merge_asof(left, right, left_on="opstart_time", right_on="chart_time", by="subject_id",
                               tolerance=float(window_min), direction="backward")
        out[f"{prefix}{item}"] = out["op_id"].map(merged.set_index("op_id")["value"])
    return out


def preop_max(ops: pd.DataFrame, events: pd.DataFrame, window_min: int) -> pd.Series:
    """Highest value charted at or before the operation start and at most `window_min` minutes before it (the window
    latest_before reads)."""
    merged = ops[["op_id", "subject_id", "opstart_time"]].merge(events[["subject_id", "chart_time", "value"]],
                                                                on="subject_id", how="inner")
    start = merged["opstart_time"]
    inside = (merged["chart_time"] <= start) & (merged["chart_time"] >= start - window_min)
    return ops["op_id"].map(merged[inside].groupby("op_id")["value"].max()).set_axis(ops.index)


def kidney_failure_before(df: pd.DataFrame, spec: dict | None = None) -> pd.Series:
    """Kidney failure that predates the operation (spec subgroups.preexisting_kidney_failure): a creatinine at or above
    the cohort's 4.5 mg/dl limit anywhere in the 90 days up to the start (preop_creatinine_max_90d; the cohort rule
    reads only the latest value), or CRRT in the 7 days up to the start (crrt_preop_7d). Reported, not excluded (the
    paper does not exclude it)."""
    spec = spec or load_spec()
    limit = spec["cohort"]["rules"]["max_preop_creatinine_exclusive"]
    return (df["preop_creatinine_max_90d"] >= limit).fillna(False).astype(bool) | df["crrt_preop_7d"].astype(bool)


def postop_max(ops: pd.DataFrame, events: pd.DataFrame, days: int) -> pd.Series:
    """Highest value charted after the operation end and at most `days` days after it (labels.py)."""
    merged = ops[["op_id", "subject_id", "opend_time"]].merge(events[["subject_id", "chart_time", "value"]],
                                                              on="subject_id", how="inner")
    inside = (merged["chart_time"] > merged["opend_time"]) & (merged["chart_time"] <= merged["opend_time"] + days * DAY)
    return ops["op_id"].map(merged[inside].groupby("op_id")["value"].max()).set_axis(ops.index)


def crrt_flags(ops: pd.DataFrame, crrt: pd.DataFrame, days: int) -> pd.DataFrame:
    """crrt_ever: the patient has a CRRT record at any time (labels.py's rule); crrt_postop: a record after the
    operation end and at most `days` days after it; crrt_preop_7d: a record in the 7 days up to the operation start."""
    merged = ops[["op_id", "subject_id", "opstart_time", "opend_time"]].merge(
        crrt[["subject_id", "chart_time"]], on="subject_id", how="inner")
    post = merged[(merged["chart_time"] > merged["opend_time"])
                  & (merged["chart_time"] <= merged["opend_time"] + days * DAY)]["op_id"]
    pre = merged[(merged["chart_time"] <= merged["opstart_time"])
                 & (merged["chart_time"] > merged["opstart_time"] - 7 * DAY)]["op_id"]
    return pd.DataFrame({"crrt_ever": ops["subject_id"].isin(crrt["subject_id"]).to_numpy(),
                         "crrt_postop": ops["op_id"].isin(post).to_numpy(),
                         "crrt_preop_7d": ops["op_id"].isin(pre).to_numpy()}, index=ops.index)


def aki_labels(df: pd.DataFrame, spec: dict | None = None) -> pd.DataFrame:
    """KDIGO stages as labels.py computes them, from preop_creatinine, postop_creatinine_2_days and _7_days.

    The paper's rule (`label`, cohort `in_cohort`): stage 3 also when the patient has any CRRT record (crrt_ever);
    kept when either post-operative creatinine or crrt_ever exists. The earlier BenchToward rule (`label_crrt_7d`,
    cohort `in_crrt_7d_cohort`, report only): CRRT counts only in the 7 days after the operation end (crrt_postop)."""
    spec = spec or load_spec()
    r = spec["label"]["rules"]
    ratio = df["postop_creatinine_7_days"] / df["preop_creatinine"]
    rise_48h = df["postop_creatinine_2_days"] - df["preop_creatinine"]
    stage2 = (ratio >= 2) & (ratio < 3)
    stage3_cr = (ratio >= 3) | (df["postop_creatinine_7_days"] > r["stage3_creatinine_above"])
    has_cr = df["postop_creatinine_2_days"].notna() | df["postop_creatinine_7_days"].notna()
    out = pd.DataFrame(index=df.index)
    out["crt_7_day_ratio"] = ratio
    out["aki_1"] = ((ratio > 1.5) & (ratio < 2)) | (rise_48h > 0.3)
    out["aki_2"] = stage2
    out["aki_3"] = stage3_cr | df["crrt_ever"]
    out["aki_3_crrt_7d"] = stage3_cr | df["crrt_postop"]
    out["label"] = (stage2 | out["aki_3"]).astype(int)
    out["label_crrt_7d"] = (stage2 | out["aki_3_crrt_7d"]).astype(int)
    out["in_cohort"] = has_cr | df["crrt_ever"]
    out["in_crrt_7d_cohort"] = has_cr | df["crrt_postop"]
    return out


def gs_aki(df: pd.DataFrame, diagnosis: pd.DataFrame, proxy_map: pd.DataFrame, spec: dict | None = None) -> pd.DataFrame:
    """The adapted GS-AKI count (clinical_baselines/gs_aki.py): nine factors, each 0 or 1, from age, sex, emop, the
    intraperitoneal proxy of the ICD-10-PCS code, diagnoses charted before the operation start (heart failure and
    ascites only in the 30 days before it) and preop_creatinine."""
    spec = spec or load_spec()
    g = spec["baselines_detail"]["gs_aki"]
    pcs = df["icd10_pcs"].astype("string").str.strip().str.upper()
    proxy = proxy_map.assign(icd10_pcs_5char=proxy_map["icd10_pcs_5char"].astype(str).str.strip().str.upper())
    intraperitoneal = pcs.map(proxy.set_index("icd10_pcs_5char")["intraperitoneal_proxy"].astype(int))
    if intraperitoneal.isna().any():
        raise ValueError(f"no intraperitoneal proxy for {sorted(set(pcs[intraperitoneal.isna()].dropna()))[:10]}")

    diag = diagnosis.assign(prefix=diagnosis["icd10_cm"].astype(str).str.strip().str.upper().str[:3])
    wanted = set(g["diabetes_prefixes"]) | set(g["hypertension_prefixes"]) | set(g["chf_prefixes"]) | set(g["ascites_prefixes"])
    diag = diag[diag["prefix"].isin(wanted) & diag["subject_id"].isin(df["subject_id"])]
    merged = df[["op_id", "subject_id", "opstart_time"]].merge(diag[["subject_id", "chart_time", "prefix"]],
                                                              on="subject_id", how="inner")
    merged = merged[merged["chart_time"] < merged["opstart_time"]]
    recent = merged["chart_time"] >= merged["opstart_time"] - g["recent_window_days"] * DAY

    def flag(mask) -> np.ndarray:
        return df["op_id"].isin(merged.loc[mask, "op_id"]).astype(int).to_numpy()

    creatinine = df["preop_creatinine"]
    factors = pd.DataFrame({
        "gs_aki_age_ge_56": (df["age"] >= 56).astype(int).to_numpy(),
        "gs_aki_male": (df["sex"] == 1).astype(int).to_numpy(),
        "gs_aki_emergency": (df["emop"] == 1).astype(int).to_numpy(),
        "gs_aki_intraperitoneal": intraperitoneal.astype(int).to_numpy(),
        "gs_aki_diabetes": flag(merged["prefix"].isin(g["diabetes_prefixes"])),
        "gs_aki_chf_30d": flag(recent & merged["prefix"].isin(g["chf_prefixes"])),
        "gs_aki_ascites_30d": flag(recent & merged["prefix"].isin(g["ascites_prefixes"])),
        "gs_aki_hypertension": flag(merged["prefix"].isin(g["hypertension_prefixes"])),
        "gs_aki_renal_insufficiency": (creatinine >= 1.2).astype(int).to_numpy(),
    }, index=df.index)
    factors["gs_aki_count"] = factors.sum(axis=1)
    return factors


# ---- split ----

def make_split(df: pd.DataFrame, label: str = "label", seed: int = 42, n_splits: int = 5) -> pd.DataFrame:
    """Outer fold 0 of the authors' split (split_manager._build_grouped_kfold_splitter): StratifiedGroupKFold(5,
    shuffle=True, random_state=42) over operations (op_id order), grouped by patient, stratified by the paper's label.
    Its result depends on the scikit-learn release (1.5.2 and 1.9.1 differ), so the published files were made with the
    authors' 1.5.2 (scripts/inspire_aki_reference.py) and builds read them; this function documents the procedure."""
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, test_idx = next(iter(splitter.split(df[["op_id"]], df[label].astype(int), groups=df["subject_id"])))
    test = set(df["subject_id"].iloc[test_idx])
    subjects = sorted(df["subject_id"].unique())
    return pd.DataFrame({"subject_id": subjects, "split": ["test" if s in test else "train" for s in subjects]})


def read_split(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or published_split_path())


def assign_split(df: pd.DataFrame, split: pd.DataFrame) -> pd.Series:
    mapped = df["subject_id"].map(split.set_index("subject_id")["split"])
    if mapped.isna().any():
        raise ValueError(f"{int(mapped.isna().sum())} operations of patients missing from the split file")
    return mapped


def read_outer_folds(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or outer_folds_path())


def assign_fold(df: pd.DataFrame, folds: pd.DataFrame) -> pd.Series:
    """The authors' outer fold (0-4) of each operation's patient; fold 0 is the test split."""
    mapped = df["subject_id"].map(folds.set_index("subject_id")["outer_fold"])
    if mapped.isna().any():
        raise ValueError(f"{int(mapped.isna().sum())} operations of patients missing from the outer folds file")
    return mapped.astype(int)


# ---- build ----

def input_columns(spec: dict | None = None) -> list[str]:
    spec = spec or load_spec()
    return list(spec["inputs"]["columns"])


def _read(root: Path, name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(root / name, engine="pyarrow", **kw)


def build_frames(root: Path, spec: dict | None = None) -> tuple[pd.DataFrame, list]:
    """Every operation of the paper's cohort (57,055 on INSPIRE 1.3) with inputs, labels, GS-AKI, split and fold."""
    spec = spec or load_spec()
    window = spec["inputs"]["window_days"] * DAY
    ops = _read(root, "operations.csv")
    diagnosis = _read(root, "diagnosis.csv", usecols=["subject_id", "chart_time", "icd10_cm"])
    df, steps = select_operations(ops, diagnosis, read_audit_table(), spec)

    labs = _read(root, "labs.csv", usecols=["subject_id", "chart_time", "item_name", "value"])
    ward = _read(root, "ward_vitals.csv", usecols=["subject_id", "chart_time", "item_name", "value"])
    ward = ward[ward["item_name"].isin([*spec["inputs"]["ward_items"], "crrt"])]
    vitals_ops = set(_read(root, "vitals.csv", usecols=["op_id"])["op_id"].unique())

    df = df.merge(latest_before(df, labs, spec["inputs"]["lab_items"], "preop_", window), on="op_id")
    df = df.merge(latest_before(df, ward, spec["inputs"]["ward_items"], "ward_", window), on="op_id")
    df = df[df["op_id"].isin(vitals_ops)]
    _count(steps, "has_intraoperative_record", df)
    df = df[df["preop_creatinine"].notna()]
    _count(steps, "has_preop_creatinine", df)
    df = df[df["preop_creatinine"] < spec["cohort"]["rules"]["max_preop_creatinine_exclusive"]].copy()
    _count(steps, "preop_creatinine_lt_4.5", df)

    creatinine = labs[labs["item_name"] == "creatinine"]
    for days in (2, 7):
        df[f"postop_creatinine_{days}_days"] = postop_max(df, creatinine, days)
    df = pd.concat([df, crrt_flags(df, ward[ward["item_name"] == "crrt"], 7)], axis=1)
    df = pd.concat([df, aki_labels(df, spec)], axis=1)
    df["preop_creatinine_max_90d"] = preop_max(df, creatinine, window)
    df["preexisting_kidney_failure"] = kidney_failure_before(df, spec)
    df["num_card_events_same_day"] = same_day_card_events(df, diagnosis,
                                                          spec["cohort"]["rules"]["cardiovascular_prefix"])
    df = df[df["in_cohort"]].copy()
    _count(steps, "cohort_postop_creatinine_or_crrt_ever", df)
    _count(steps, "report_crrt_7d_cohort_postop_creatinine_or_crrt_7d", df[df["in_crrt_7d_cohort"]])
    df = pd.concat([df, gs_aki(df, diagnosis, pd.read_csv(proxy_map_path(), dtype={"icd10_pcs_5char": str}), spec)],
                   axis=1)
    df["split"] = assign_split(df, read_split())
    df["outer_fold"] = assign_fold(df, read_outer_folds())
    if not ((df["outer_fold"] == 0) == (df["split"] == "test")).all():
        raise ValueError("outer fold 0 and the test split disagree")
    df.insert(0, "instance_id", "inspire-aki-" + df["op_id"].astype(str))
    return df.sort_values("op_id", kind=STABLE).reset_index(drop=True), steps


INSTANCE_COLUMNS = ["instance_id", "op_id", "subject_id", "split", "outer_fold", "label", "label_crrt_7d"]
REPORT_COLUMNS = ["gs_aki_count", "crrt_ever", "crrt_postop", "crrt_preop_7d", "aki_1", "aki_2", "aki_3",
                  "aki_3_crrt_7d", "preop_creatinine", "postop_creatinine_2_days", "postop_creatinine_7_days",
                  "in_crrt_7d_cohort", "preop_creatinine_max_90d", "preexisting_kidney_failure",
                  "num_card_events_same_day"]


def build_task(data_dir: Path) -> Path:
    """instances.parquet: the paper's cohort with the paper's label (and the earlier CRRT-in-7-days label and cohort
    flag, for the report); summary.json: counts of every cohort step and split."""
    import json

    spec = load_spec()
    df, steps = build_frames(dataset_dir(data_dir), spec)
    cols = [*INSTANCE_COLUMNS, *input_columns(spec), *[c for c in REPORT_COLUMNS if c not in input_columns(spec)]]
    out = task_dir(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    stale = out / "authors-cohort.parquet"  # written by earlier builds; the instances now hold that cohort
    if stale.exists():
        stale.unlink()
    df[cols].to_parquet(out / "instances.parquet", index=False)
    summary = {"steps": steps, "splits": {}, "outer_folds": {}}
    crrt = df[df["in_crrt_7d_cohort"]]
    for name, frame, label in (("paper", df, "label"), ("crrt_7d", crrt, "label_crrt_7d")):
        for split, part in frame.groupby("split"):
            summary["splits"][f"{name}/{split}"] = {"operations": len(part),
                                                    "patients": int(part["subject_id"].nunique()),
                                                    "positives": int(part[label].sum()),
                                                    "prevalence": round(float(part[label].mean()), 5)}
    for fold, part in df.groupby("outer_fold"):
        summary["outer_folds"][str(fold)] = {"operations": len(part), "patients": int(part["subject_id"].nunique()),
                                             "positives": int(part["label"].sum())}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return out / "instances.parquet"


def load_instances(data_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(task_dir(data_dir) / "instances.parquet")
