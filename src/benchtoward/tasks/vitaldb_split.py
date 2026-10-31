"""Dataset-level VitalDB split: one assignment per subject over ALL VitalDB subjects, so a patient is on the same
side of every task built on VitalDB (agent-made tasks and later versions of hand-built ones).

The hypotension task v0 split only its own cohort's subjects. `vitaldb-ioh@1` keeps its authors' fixed test case list
instead (maintainers, 2026-10-29: comparability with the publication), and its spec counts the overlap.
"""
import pandas as pd

from benchtoward.tasks.vitaldb_ioh import split_subjects

SEED = 20261029
FRACTIONS = (0.7, 0.1, 0.2)


def vitaldb_subject_split(cases: pd.DataFrame) -> dict[int, str]:
    return split_subjects(cases["subjectid"].dropna().astype(int).unique(), seed=SEED, fractions=FRACTIONS)


def caseid_split(cases: pd.DataFrame) -> dict[int, str]:
    by_subject = vitaldb_subject_split(cases)
    return {int(c): by_subject[int(s)] for c, s in zip(cases["caseid"], cases["subjectid"]) if pd.notna(s)}


def load_cases() -> pd.DataFrame:
    """All VitalDB cases (caseid, subjectid, ...) from the VitalDB API."""
    from benchtoward.tasks.vitaldb_ioh import _read_cases

    return _read_cases()
