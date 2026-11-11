import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import httpx
import pandas as pd

CLINICALQA_URL = ("https://huggingface.co/datasets/snuh/ClinicalQA/resolve/"
                  "29bdc9043f3e693b7671f868e2038ec7ac288acd/data/train-00000-of-00001.parquet")
PROMPT_ID = "clinicalqa-p0"
PARSER_VERSION = "p0.3"  # p0.2: LaTeX \boxed{}, "answer is X", bold labels, Korean particles (정답은 X);
# p0.3 (owner 2026-11-03, rule reviewed with agy 2026-11-08): a last line "X. <option X text>" when p0.2 finds nothing
SYSTEM_PROMPT = ('You are answering a Korean medical licensing examination question. Think as needed, '
                 'then give your final answer on the last line as JSON: {"answer": "<letter>"}')
LETTERS = "ABCDE"

_JSON_ANSWER = re.compile(r'"answer"\s*:\s*"([A-E])"')
_BOXED = re.compile(r'\\boxed\{\s*(?:\\text\{)?\s*\(?([A-E])\)?\s*\}?\s*\}')
# "Answer: C", "**Answer:** C", "the answer is (D)", "정답은 E입니다"; a following Latin letter (as in "CBC") blocks it.
_LABELED_ANSWER = re.compile(r'(?:정답|answer|Answer|ANSWER)\s*(?:is|은|는)?\s*[:：]?\s*\**\s*\(?([A-E])(?![A-Za-z])')
_BARE_LETTER = re.compile(r'^\(?([A-E])[.)]?$')
# p0.3: "C. text", "C) text", "(C) text", "C번 text" (after NFKC, so full-width letters and stops count).
_OPTION_LINE = re.compile(r'^\(?([A-E])(?:\)|\.|번)\s*(.+)$')


@dataclass
class QAItem:
    instance_id: str
    question: str
    exam: str
    options: dict[str, str]
    answer: str
    category: str
    chief_complaint: str


def fetch_clinicalqa(data_dir: Path) -> Path:
    path = Path(data_dir) / "datasets" / "clinicalqa" / "1.0.0" / "train.parquet"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        response = httpx.get(CLINICALQA_URL, follow_redirects=True, timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path


def load_items(path: Path) -> list[QAItem]:
    items = []
    for row in pd.read_parquet(path).to_dict("records"):
        raw = row["options"] or {}
        options = {k: str(raw.get(f"option_{k}") or "").strip() for k in LETTERS}
        items.append(QAItem(
            instance_id=f"cqa-{int(row['question_id'])}",
            question=str(row["question"]),
            exam=str(row.get("exam") or "").strip(),
            options={k: v for k, v in options.items() if v},
            answer=str(row["answer"]).strip(),
            category=str(row.get("category") or ""),
            chief_complaint=str(row.get("chief_complaint") or ""),
        ))
    return items


def render_messages(item: QAItem) -> list[dict]:
    parts = [item.question]
    if item.exam:
        parts.append(f"검사 소견:\n{item.exam}")
    parts.append("\n".join(f"{k}. {v}" for k, v in item.options.items()))
    parts.append(f"Answer with one letter from: {', '.join(item.options)}.")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n\n".join(parts)}]


def _clean(line: str) -> str:
    return unicodedata.normalize("NFKC", line).strip().strip('*"\'“”').strip()


def _same_text(a: str, b: str) -> bool:
    def norm(s: str) -> str:
        return " ".join(_clean(s).split()).rstrip(".").strip().casefold()
    return norm(a) == norm(b)


def _option_line(text: str, options: dict[str, str]) -> str | None:
    """p0.3: the last non-empty line is a letter with exactly that option's text, and the line above is not another
    option line (a cut-off or restated option list)."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    match = _OPTION_LINE.match(_clean(lines[-1]))
    if not match or match.group(1) not in options or not _same_text(match.group(2), options[match.group(1)]):
        return None
    if len(lines) > 1 and _OPTION_LINE.match(_clean(lines[-2])):
        return None
    return match.group(1)


def parse_answer(text: str, allowed: set[str], options: dict[str, str] | None = None) -> str | None:
    """p0.2 patterns first; with the item's option texts, p0.3's option-line rule when they find nothing."""
    text = text or ""
    letter = None
    for pattern in (_JSON_ANSWER, _BOXED, _LABELED_ANSWER):
        matches = pattern.findall(text)
        if matches:
            letter = matches[-1]
            break
    if letter is None:
        bare = _BARE_LETTER.match(text.strip())
        letter = bare.group(1) if bare else None
    if letter is None and options:
        letter = _option_line(text, options)
    return letter if letter in allowed else None
