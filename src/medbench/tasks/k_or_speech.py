"""Korean operating-room speech recognition on Synthetic K-OR Speech Audio (KHDP `Synthetic-K-OR-Speech-Audio` 1.0.0).

8,000 WAV clips = 2,000 utterances x 4 synthetic voices, 24 kHz mono, with reference text, intent, code-switching style
and speaker role. License: Apache-2.0 with a use restriction (no TTS, voice-cloning or speaker-identification training;
redistribution keeps the restriction). Audio is read straight from the pinned release archive.

Protocol source: the dataset's usage notes ask for CER and WER against the v1.0 text, stratified by intent,
code-switching style, voice and speaker role. They give no normalisation, so `normalise` below is medbench's
(version NORMALISER_VERSION, documented in registry/tasks/k-or-speech-asr.yaml).
"""
import csv
import hashlib
import io
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

from medbench.khdp import KhdpClient
from medbench.scoring import edit_distance, error_rate_ci

CODE, VERSION = "Synthetic-K-OR-Speech-Audio", "1.0.0"
SLUG = "k-or-speech"
ARCHIVE = "synthetic_kor_speech_audio_v1.zip"
ARCHIVE_SHA256 = "fb60266a06cf710535d3e4468b8a1748adda6b58de231e05a5e407b5c2de772d"  # KHDP publishes no checksum
ROOT_IN_ZIP = "synthetic_kor_speech_audio_v1"
VOICES = ("chul", "sohee", "ethan", "maia")
NORMALISER_VERSION = "kor-norm-0"


@dataclass(frozen=True)
class Clip:
    instance_id: str  # <utterance id>-<voice>, e.g. MMO-M00002-chul
    utterance_id: str
    voice: str
    reference: str
    intent: str
    cs_style: str
    speaker: str
    member: str  # path inside the release archive

    @property
    def filename(self) -> str:
        return PurePosixPath(self.member).name


def dataset_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "datasets" / SLUG / VERSION


def archive_path(data_dir: Path) -> Path:
    return dataset_dir(data_dir) / ARCHIVE


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_audio(data_dir: Path, client: KhdpClient, *, sha256: str = ARCHIVE_SHA256) -> Path:
    """Download the release archive into data/datasets/k-or-speech/1.0.0/ (resumable) and verify its SHA-256."""
    path = archive_path(data_dir)
    if path.exists() and _sha256(path) == sha256:
        return path
    files = {f.key.removeprefix(f"{VERSION}/"): f for f in client.files(CODE, VERSION)}
    if ARCHIVE not in files:
        raise ValueError(f"{ARCHIVE} missing from the KHDP release (found {sorted(files)})")
    client.download(files[ARCHIVE], path)
    if _sha256(path) != sha256:
        path.unlink()
        raise ValueError(f"checksum mismatch for {ARCHIVE}: the release changed or the download is corrupt")
    return path


def load_clips(data_dir: Path) -> list[Clip]:
    """All clips from the archive's metadata.csv, ordered by utterance then voice. Every utterance must have 4 voices."""
    with zipfile.ZipFile(archive_path(data_dir)) as z:
        names = set(z.namelist())
        text = z.read(f"{ROOT_IN_ZIP}/metadata.csv").decode("utf-8-sig")
    clips = []
    for row in csv.DictReader(io.StringIO(text)):
        member = f"{ROOT_IN_ZIP}/{row['filename']}"
        if member not in names:
            raise ValueError(f"metadata.csv lists {row['filename']}, which is not in the archive")
        clips.append(Clip(instance_id=f"{row['id']}-{row['voice_label']}", utterance_id=row["id"],
                          voice=row["voice_label"], reference=row["text"], intent=row["intent"],
                          cs_style=row["cs_style"], speaker=row["speaker"], member=member))
    voices: dict[str, set[str]] = defaultdict(set)
    for clip in clips:
        voices[clip.utterance_id].add(clip.voice)
    incomplete = [u for u, v in voices.items() if v != set(VOICES)]
    if incomplete or len(clips) != len(voices) * len(VOICES):
        raise ValueError(f"{len(incomplete)} utterance(s) do not have exactly the voices {VOICES}, e.g. {incomplete[:3]}")
    order = {v: i for i, v in enumerate(VOICES)}
    return sorted(clips, key=lambda c: (c.utterance_id, order[c.voice]))


class AudioReader:
    """`with AudioReader(data_dir) as read: read(clip) -> WAV bytes`, straight from the archive (CRC-checked)."""

    def __init__(self, data_dir: Path):
        self.path = archive_path(data_dir)
        self._zip: zipfile.ZipFile | None = None

    def __enter__(self) -> "AudioReader":
        self._zip = zipfile.ZipFile(self.path)
        return self

    def __exit__(self, *exc) -> None:
        if self._zip is not None:
            self._zip.close()
            self._zip = None

    def __call__(self, clip: Clip) -> bytes:
        return self._zip.read(clip.member)


# ---- normalisation (kor-norm-0) ----

_SINO = "영일이삼사오육칠팔구"
_NATIVE_UNITS = ["", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉"]
_NATIVE_TENS = ["", "열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔"]
# Counters read with native Korean numbers (두 개, 세 시간). Everything else (번, 미리, 도, 분, %, 유닛 ...) is
# Sino-Korean, which is how the release text uses numbers.
NATIVE_COUNTERS = ("번째", "시간", "개", "명", "장", "살", "마리", "가지", "군데")
_NUMBER = re.compile(r"(?<![a-z0-9])(\d+)(?:\.(\d+))?")
_NATIVE_AHEAD = re.compile(r"\s?(?:" + "|".join(NATIVE_COUNTERS) + ")")
_BRACKETED = (re.compile(r"[<\[][^>\]]*[>\]]"), re.compile(r"\(([^)]+?)\)"))
# A spoken decimal point between Hangul numerals (칠 점 오, 영쩜오): optional, because two of the four voices say it
# and two mostly do not.
_SPOKEN_POINT = re.compile(r"(?<=[영일이삼사오육칠팔구십백천만])\s?[점쩜]\s?(?=[영일이삼사오육칠팔구])")


def _sino_below_10000(n: int) -> str:
    out = ""
    for value, name in ((1000, "천"), (100, "백"), (10, "십"), (1, "")):
        d = n // value % 10
        if d:
            out += ("" if d == 1 and name else _SINO[d]) + name
    return out


def sino_korean(digits: str) -> str:
    """'300' -> 삼백, '10000' -> 만, '12345' -> 만이천삼백사십오; leading zeros and very long runs digit by digit."""
    if len(digits) > 1 and digits.startswith("0") or len(digits) > 16:
        return "".join(_SINO[int(d)] for d in digits)
    n = int(digits)
    if n == 0:
        return "영"
    out = ""
    for value, name in ((10 ** 12, "조"), (10 ** 8, "억"), (10 ** 4, "만"), (1, "")):
        chunk = n // value % 10000
        if chunk:
            out += ("" if chunk == 1 and name == "만" else _sino_below_10000(chunk)) + name
    return out


def native_korean(n: int) -> str:
    """Attributive native number, 1-99: 2 -> 두, 12 -> 열두, 20 -> 스무, 21 -> 스물한."""
    tens, units = divmod(n, 10)
    return "스무" if (tens, units) == (2, 0) else _NATIVE_TENS[tens] + _NATIVE_UNITS[units]


def _read_number(match: re.Match) -> str:
    whole, fraction = match.group(1), match.group(2)
    if fraction is not None:
        return sino_korean(whole) + "".join(_SINO[int(d)] for d in fraction)  # 7.5 -> 칠오 (점 is optional)
    rest = match.string[match.end():]
    if not whole.startswith("0") and 1 <= int(whole) <= 99 and _NATIVE_AHEAD.match(rest):
        return native_korean(int(whole))
    return sino_korean(whole)


def normalise(text: str | None, *, numbers: bool = True) -> str:
    """Space-separated normalised words. Applied identically to references and transcripts.

    1. Unicode NFKC (NFC plus compatibility folding: full-width letters and digits, ㎎, …), then lowercase.
    2. Remove bracketed tags: [..], <..>, (..) (non-speech markers some ASR models emit).
    3. Numbers (numbers=True): drop commas inside numbers; read Arabic numerals in Hangul, Sino-Korean (5 번 -> 오 번),
       native before native counters (12개 -> 열두개); decimals digit by digit with the point optional (7.5, 칠 점 오
       and 칠오 all become 칠오); digits attached to Latin letters (SpO2, EtCO2, R2P3) stay as written; % -> 퍼센트.
    4. Every punctuation, symbol and mark character becomes a space; whitespace collapses.
    English words stay as written (lowercased): a code-switched term written in Hangul (비피) against Latin (BP) is an error.
    """
    s = unicodedata.normalize("NFKC", text or "").lower()
    for pattern in _BRACKETED:
        s = pattern.sub(" ", s)
    if numbers:
        s = re.sub(r"(?<=\d),(?=\d)", "", s)
        s = _NUMBER.sub(_read_number, s)
        s = s.replace("%", " 퍼센트 ")
        s = _SPOKEN_POINT.sub("", s)
    s = "".join(" " if unicodedata.category(c)[0] in "MSP" else c for c in s)
    return " ".join(s.split())


@dataclass(frozen=True)
class ItemScore:
    char_errors: int
    ref_chars: int
    word_errors: int
    ref_words: int

    @property
    def cer(self) -> float:
        return self.char_errors / self.ref_chars if self.ref_chars else float(self.char_errors > 0)


def score_item(reference: str, hypothesis: str | None, *, numbers: bool = True) -> ItemScore:
    """Character errors with all spaces removed (Korean spacing varies), word errors over whitespace tokens."""
    ref, hyp = normalise(reference, numbers=numbers), normalise(hypothesis, numbers=numbers)
    ref_chars, hyp_chars = ref.replace(" ", ""), hyp.replace(" ", "")
    ref_words, hyp_words = ref.split(), hyp.split()
    return ItemScore(char_errors=edit_distance(ref_chars, hyp_chars), ref_chars=len(ref_chars),
                     word_errors=edit_distance(ref_words, hyp_words), ref_words=len(ref_words))


def _rates(scores: list[ItemScore], groups: list[str], n_boot: int) -> tuple[dict, dict, dict]:
    cer = error_rate_ci([s.char_errors for s in scores], [s.ref_chars for s in scores], groups, n_boot=n_boot)
    wer = error_rate_ci([s.word_errors for s in scores], [s.ref_words for s in scores], groups, n_boot=n_boot)
    ser = error_rate_ci([int(s.char_errors > 0) for s in scores], [1] * len(scores), groups, n_boot=n_boot)
    return cer, wer, ser


def asr_metrics(clips: list[Clip], hypotheses: list[str | None], errors: list[str | None],
                n_boot: int = 1000) -> dict:
    """CER (primary), WER and sentence error rate with 95% intervals from a bootstrap over utterances (the 4 voices of
    an utterance are resampled together), plus the same by voice, code-switching style, intent, speaker role and
    whether the reference contains digits. A clip without a transcript counts as an empty transcript."""
    scores = [score_item(c.reference, h if e is None else None) for c, h, e in zip(clips, hypotheses, errors)]
    groups = [c.utterance_id for c in clips]
    cer, wer, ser = _rates(scores, groups, n_boot)
    n = len(clips)
    metrics = {
        "cer": cer | {"n_missing": sum(e is not None for e in errors)},
        "wer": wer,
        "ser": ser,
        "parse_rate": sum(e is None for e in errors) / n if n else 0.0,
        "error_rate": sum(e is not None for e in errors) / n if n else 0.0,
        "empty_rate": sum(e is None and not normalise(h) for h, e in zip(hypotheses, errors)) / n if n else 0.0,
        "errors": dict(Counter(e for e in errors if e is not None)),
        "normaliser": NORMALISER_VERSION,
    }
    facets = {"voice": lambda c: c.voice, "cs_style": lambda c: c.cs_style, "intent": lambda c: c.intent,
              "speaker": lambda c: c.speaker,
              "digits": lambda c: "with digits" if re.search(r"[0-9]", c.reference) else "without digits"}
    for name, key in facets.items():
        members: dict[str, list[int]] = defaultdict(list)
        for i, clip in enumerate(clips):
            members[key(clip)].append(i)
        breakdown = {}
        for value, idx in sorted(members.items()):
            sub_cer, sub_wer, _ = _rates([scores[i] for i in idx], [groups[i] for i in idx], n_boot)
            breakdown[value] = {"value": sub_cer["value"], "ci_low": sub_cer["ci_low"], "ci_high": sub_cer["ci_high"],
                                "wer": sub_wer["value"], "n": len(idx)}
        metrics[f"by_{name}"] = breakdown
    return metrics


def item_rows(clips: list[Clip], hypotheses: list[str | None], errors: list[str | None]) -> list[dict]:
    """Per-clip rows for predictions.parquet."""
    rows = []
    for clip, hyp, err in zip(clips, hypotheses, errors):
        s = score_item(clip.reference, hyp if err is None else None)
        rows.append({"instance_id": clip.instance_id, "utterance_id": clip.utterance_id, "voice": clip.voice,
                     "intent": clip.intent, "cs_style": clip.cs_style, "speaker": clip.speaker,
                     "reference": clip.reference, "prediction": hyp if err is None else None,
                     "reference_normalised": normalise(clip.reference),
                     "prediction_normalised": normalise(hyp) if err is None else None,
                     "cer": s.cer, "char_errors": s.char_errors, "ref_chars": s.ref_chars,
                     "word_errors": s.word_errors, "ref_words": s.ref_words, "error": err})
    return rows


def pooled_cer(char_errors: np.ndarray, ref_chars: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """CER per model (rows of `char_errors`) on the items `idx`; used for paired rank statistics."""
    return char_errors[:, idx].sum(axis=1) / max(ref_chars[idx].sum(), 1)
