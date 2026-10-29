import re
from dataclasses import dataclass
from pathlib import Path

import httpx
import pandas as pd

CLINICALQA_URL = ("https://huggingface.co/datasets/snuh/ClinicalQA/resolve/"
                  "29bdc9043f3e693b7671f868e2038ec7ac288acd/data/train-00000-of-00001.parquet")
PROMPT_ID = "clinicalqa-p0"
PARSER_VERSION = "p0.2"  # p0.2: LaTeX \boxed{}, "answer is X", bold labels, Korean particles (정답은 X)
SYSTEM_PROMPT = ('You are answering a Korean medical licensing examination question. Think as needed, '
                 'then give your final answer on the last line as JSON: {"answer": "<letter>"}')
LETTERS = "ABCDE"

_JSON_ANSWER = re.compile(r'"answer"\s*:\s*"([A-E])"')
_BOXED = re.compile(r'\\boxed\{\s*(?:\\text\{)?\s*\(?([A-E])\)?\s*\}?\s*\}')
# "Answer: C", "**Answer:** C", "the answer is (D)", "정답은 E입니다"; a following Latin letter (as in "CBC") blocks it.
_LABELED_ANSWER = re.compile(r'(?:정답|answer|Answer|ANSWER)\s*(?:is|은|는)?\s*[:：]?\s*\**\s*\(?([A-E])(?![A-Za-z])')
_BARE_LETTER = re.compile(r'^\(?([A-E])[.)]?$')


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


def parse_answer(text: str, allowed: set[str]) -> str | None:
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
    return letter if letter in allowed else None
