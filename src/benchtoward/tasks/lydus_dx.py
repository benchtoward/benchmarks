"""LYDUS diagnosis text (`dx`, the electrocardiograph's interpretation statements) to SNOMED CT codes.

The shared label map of the LYDUS tasks. `dx` holds one statement per line, mostly in the GE 12SL wording
("Sinus rhythm", "with 1st degree A-V block", "Possible", "Left atrial enlargement"), a few hundred records in the
Schiller wording with "because ..." explanation lines, and a few free-text lines. The map
(registry/maps/lydus-dx-snomed-v0.csv) is BenchToward-made: one row per distinct line (whitespace collapsed) with
- role: finding (a statement; `codes` may be empty when the Challenge 2021 tables have no code for it), modifier
  (continuation words such as "with", ", age undetermined", "with variable A-V block"; may carry a code),
  hedge ("Possible", "Cannot rule out": the next finding's codes become hedged, not labels), alternative ("or"
  between two findings: both become hedged), explanation (Schiller
  "because ..." lines and their wrapped continuations), summary ("Abnormal ECG"), quality (signal notes; the record
  is kept) or no_analysis (the electrocardiograph made no interpretation: the record is left out);
- codes: SNOMED CT codes of the PhysioNet/CinC Challenge 2021 tables (dx_mapping_scored.csv and
  dx_mapping_unscored.csv at e2a75fc), definite;
- hedged_codes: codes the statement names with "possible", "probable", "consider", "suspected" or "R/O"; never
  labels, reported beside them.
Every line in the data must be in the map: `parse_all` refuses an unknown line.
"""
import csv
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from benchtoward import metrics_challenge2021 as cm

MAP_FILE = "lydus-dx-snomed-v0.csv"
ROLES = ("finding", "modifier", "hedge", "alternative", "explanation", "summary", "quality", "no_analysis")


def normalize(line: str) -> str:
    return " ".join(line.split())


def lines(text: str) -> list[str]:
    return [n for n in (normalize(line) for line in str(text or "").split("\n")) if n]


@dataclass(frozen=True)
class Entry:
    role: str
    codes: tuple[str, ...]
    hedged: tuple[str, ...]


def load_map(path: Path) -> dict[str, Entry]:
    out = {}
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["role"] not in ROLES:
                raise ValueError(f"unknown role {row['role']!r} for {row['statement']!r}")
            out[row["statement"]] = Entry(row["role"], tuple(filter(None, row["codes"].split(";"))),
                                          tuple(filter(None, row["hedged_codes"].split(";"))))
    return out


@lru_cache(maxsize=1)
def vocabulary() -> dict[str, str]:
    """SNOMED CT code -> name in the Challenge 2021 tables (scored and unscored)."""
    out = {}
    for name in ("dx_mapping_scored.csv", "dx_mapping_unscored.csv"):
        with (cm.VENDORED / name).open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                out[row["SNOMEDCTCode"]] = row["Dx"]
    return out


@dataclass
class Parsed:
    codes: set[str] = field(default_factory=set)
    hedged: set[str] = field(default_factory=set)
    no_analysis: bool = False
    unknown: list[str] = field(default_factory=list)


def parse_dx(text: str, table: dict[str, Entry]) -> Parsed:
    out = Parsed()
    definite: list[tuple[str, ...]] = []    # each line's definite codes, in order
    last_finding = None                      # index in `definite` of the latest definite finding
    pending_hedge = False
    for line in lines(text):
        entry = table.get(line)
        if entry is None:
            out.unknown.append(line)
            continue
        if entry.role == "no_analysis":
            out.no_analysis = True
        elif entry.role == "hedge":
            pending_hedge = True
            continue
        elif entry.role == "alternative":    # "A / or / B": both findings become hedged (owner, 2026-11-03)
            if last_finding is not None:
                out.hedged.update(definite[last_finding])
                definite[last_finding] = ()
            pending_hedge = True
            continue
        elif entry.role == "summary":
            pending_hedge = False
            last_finding = None              # a summary ends the statement group: "or" never reaches back across it
        if entry.role == "finding" and pending_hedge:
            out.hedged.update(entry.codes)
            pending_hedge = False
            last_finding = None
        else:
            definite.append(entry.codes)
            if entry.role == "finding":
                pending_hedge = False
                last_finding = len(definite) - 1
        out.hedged.update(entry.hedged)
    out.codes = {code for codes in definite for code in codes}
    out.hedged -= out.codes
    return out


def challenge_labels(codes) -> np.ndarray:
    """Boolean vector over the 26 scored Challenge classes (weights.csv order)."""
    y = np.zeros(len(cm.classes()), dtype=bool)
    for code in codes:
        k = cm.class_index(code)
        if k is not None:
            y[k] = True
    return y


@dataclass
class ParsedAll:
    codes: list[list[str]]
    hedged: list[list[str]]
    labels: np.ndarray          # (records, 26) definite scored classes
    hedged_labels: np.ndarray   # (records, 26) hedged only
    no_analysis: np.ndarray


def parse_all(dx: pd.Series, table: dict[str, Entry]) -> ParsedAll:
    parsed = [parse_dx(t, table) for t in dx]
    unknown = sorted({u for p in parsed for u in p.unknown})
    if unknown:
        raise ValueError(f"{len(unknown)} dx lines are not in the map, e.g. {unknown[:3]}")
    return ParsedAll(codes=[sorted(p.codes) for p in parsed], hedged=[sorted(p.hedged) for p in parsed],
                     labels=np.stack([challenge_labels(p.codes) for p in parsed]),
                     hedged_labels=np.stack([challenge_labels(p.hedged) & ~challenge_labels(p.codes)
                                             for p in parsed]),
                     no_analysis=np.array([p.no_analysis for p in parsed]))


def statement_counts(dx: pd.Series) -> dict[str, int]:
    """Records in which each normalised line appears (the map's n_records)."""
    counts = Counter()
    for text in dx:
        counts.update(set(lines(text)))
    return dict(counts)
