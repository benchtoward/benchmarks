import hashlib
import io
import zipfile

import httpx
import numpy as np
import pytest

from medbench.khdp import KhdpClient
from medbench.scoring import edit_distance, error_rate_ci
from medbench.tasks.k_or_speech import (ROOT_IN_ZIP, VOICES, AudioReader, Clip, asr_metrics, fetch_audio, load_clips,
                                        normalise, score_item)

# ---- normalisation ----


def test_punctuation_case_and_whitespace():
    assert normalise("BP 급격히   떨어집니다.") == "bp 급격히 떨어집니다"
    assert normalise("네, 맞습니다? 아니요!") == "네 맞습니다 아니요"
    assert normalise("L-tube 석션…") == "l tube 석션"


def test_nfkc_folds_full_width_forms():
    assert normalise("ＢＰ　１００") == "bp 백"


def test_bracketed_tags_are_removed():
    assert normalise("[음악] 석션 주세요 (웃음) <noise>") == "석션 주세요"


def test_numbers_read_as_sino_korean():
    assert normalise("펨버는 5 번, 티비아도 5 번으로") == "펨버는 오 번 티비아도 오 번으로"
    assert normalise("투니켓 압력 300으로") == "투니켓 압력 삼백으로"
    assert normalise("헤파린 3만 유닛") == "헤파린 삼만 유닛"
    assert normalise("10000 12345 1,000 0 도") == "만 만이천삼백사십오 천 영 도"
    assert normalise("10 100 111") == "십 백 백십일"


def test_decimals_percent_and_sutures():
    # Two of the four voices say 점 for the decimal point and two mostly do not, so 점 between numerals is optional.
    assert normalise("튜브 7.5 들어갔고") == "튜브 칠오 들어갔고"
    assert normalise("튜브 칠 점 오, 영쩜오 미리") == "튜브 칠오 영오 미리"
    assert score_item("튜브 7.0 넣었고", "튜브 칠점영 넣었고").char_errors == 0
    assert score_item("튜브 7.0 넣었고", "튜브 칠 영 넣었고").char_errors == 0
    assert normalise("문제점 확인") == "문제점 확인"
    assert normalise("세푸록심 1.5g") == "세푸록심 일오g"
    assert normalise("산소포화도 99%") == "산소포화도 구십구 퍼센트"
    assert normalise("모노크릴 4-0 장전") == "모노크릴 사 영 장전"


def test_native_numbers_before_native_counters():
    assert normalise("사용 12개, 미사용 8개") == "사용 열두개 미사용 여덟개"
    assert normalise("랩 패드 2 장, 거즈 20 개") == "랩 패드 두 장 거즈 스무 개"
    assert normalise("수술 시간 3 시간 20 분") == "수술 시간 세 시간 이십 분"
    assert normalise("21명") == "스물한명"


def test_digits_attached_to_latin_letters_stay_as_written():
    assert normalise("SpO2 100%, EtCO2 35, R2P3") == "spo2 백 퍼센트 etco2 삼십오 r2p3"


def test_leading_zero_read_digit_by_digit():
    assert normalise("05") == "영오"


def test_numbers_rule_can_be_switched_off():
    assert normalise("5 번 99%", numbers=False) == "5 번 99"


# ---- error counts ----


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3 and edit_distance("abc", "") == 3 and edit_distance("", "") == 0
    assert edit_distance(["a", "b"], ["a", "c", "b"]) == 1


def test_cer_ignores_spaces_and_wer_counts_words():
    s = score_item("넣어 주세요.", "넣어주세요")
    assert (s.char_errors, s.ref_chars) == (0, 5)
    assert (s.word_errors, s.ref_words) == (2, 2)


def test_hangul_numerals_match_digit_reference():
    assert score_item("펨버는 5 번, 티비아도 5 번으로 잡겠습니다.", "펨버는 오번 티비아도 오번으로 잡겠습니다").char_errors == 0
    assert score_item("사용 12개", "사용 열두 개").char_errors == 0


def test_empty_or_missing_hypothesis_is_all_deletions():
    s = score_item("석션 주세요", "")
    assert s.char_errors == s.ref_chars == 5 and s.word_errors == s.ref_words == 2
    assert score_item("석션 주세요", None).char_errors == 5


# ---- clustered bootstrap ----


def test_error_rate_is_pooled_over_items():
    r = error_rate_ci([1, 0, 3], [10, 10, 10], ["a", "b", "c"])
    assert r["value"] == pytest.approx(4 / 30) and r["n"] == 3 and r["n_groups"] == 3
    assert r["ci_low"] <= r["value"] <= r["ci_high"]


def test_one_cluster_gives_a_point_interval():
    r = error_rate_ci([1, 0, 2, 0], [5, 5, 5, 5], ["u1"] * 4)
    assert r["ci_low"] == r["ci_high"] == r["value"] == pytest.approx(3 / 20)


def test_cluster_bootstrap_resamples_whole_utterances():
    groups = np.repeat(np.arange(20), 4)              # 20 utterances x 4 voices
    errors = np.repeat(np.arange(20) % 2, 4)          # an utterance is wrong in all 4 voices or in none
    lengths = np.ones(80)
    clustered = error_rate_ci(errors, lengths, groups)
    naive = error_rate_ci(errors, lengths, np.arange(80))
    assert clustered["value"] == naive["value"] == 0.5
    width = lambda r: r["ci_high"] - r["ci_low"]  # noqa: E731
    assert width(clustered) > 1.5 * width(naive)


# ---- release archive, clips, metrics ----

HEADER = ("id,voice_label,voice_id,model,intent,instruction,text,speaker,register,cs_style,source,phase,"
          "case_patient_age,case_patient_sex,case_asa,case_department,case_surgery,case_antype,case_emop,filename,bytes")
UTTERANCES = [("GEN-G00002", "vital_callout", "BP 90 대로 떨어집니다.", "anesthesia_resident", "english"),
              ("MMO-M00001", "instrument_request", "석션 주세요.", "surgeon", "phonetic_kr")]


def make_release(path, utterances=UTTERANCES, voices=VOICES) -> bytes:
    rows = [HEADER]
    members = {}
    for utt, intent, text, speaker, style in utterances:
        for voice in voices:
            name = f"qwen3m_{voice}/{utt}__qwen3m_{voice}.wav"
            members[name] = f"RIFF-{utt}-{voice}".encode()
            rows.append(f'{utt},{voice},v,m,{intent},,"{text}",{speaker},polite,{style},synthetic_main,intraop,'
                        f',,,,,,,{name},{len(members[name])}')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(f"{ROOT_IN_ZIP}/metadata.csv", "\n".join(rows) + "\n")
        z.writestr(f"{ROOT_IN_ZIP}/LICENSE", "Apache-2.0 with use restriction")
        for name, data in members.items():
            z.writestr(f"{ROOT_IN_ZIP}/{name}", data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())
    return buf.getvalue()


def _archive(data_dir):
    return data_dir / "datasets" / "k-or-speech" / "1.0.0" / "synthetic_kor_speech_audio_v1.zip"


def test_load_clips_and_read_audio(tmp_path):
    make_release(_archive(tmp_path))
    clips = load_clips(tmp_path)
    assert len(clips) == 8 and clips[0].instance_id == "GEN-G00002-chul"
    assert [c.voice for c in clips[:4]] == list(VOICES)
    first = clips[0]
    assert (first.utterance_id, first.reference, first.intent, first.cs_style, first.speaker) == (
        "GEN-G00002", "BP 90 대로 떨어집니다.", "vital_callout", "english", "anesthesia_resident")
    assert first.filename == "GEN-G00002__qwen3m_chul.wav"
    with AudioReader(tmp_path) as read:
        assert read(clips[5]) == b"RIFF-MMO-M00001-sohee"


def test_load_clips_rejects_an_utterance_without_all_voices(tmp_path):
    make_release(_archive(tmp_path), voices=VOICES[:3])
    with pytest.raises(ValueError, match="voices"):
        load_clips(tmp_path)


def _clip(utt, voice, ref, style="none", intent="status_update"):
    return Clip(instance_id=f"{utt}-{voice}", utterance_id=utt, voice=voice, reference=ref, intent=intent,
                cs_style=style, speaker="surgeon", member=f"{ROOT_IN_ZIP}/x.wav")


def test_asr_metrics_overall_and_breakdowns():
    clips = [_clip("U1", "chul", "석션 주세요"), _clip("U1", "maia", "석션 주세요"),
             _clip("U2", "chul", "BP 5 미리", style="english"), _clip("U2", "maia", "BP 5 미리", style="english")]
    hyps = ["석션 주세요", "석션 줘요", "비피 오 미리", None]
    m = asr_metrics(clips, hyps, [None, None, None, "http_503"])
    # chars: 석션주세요 (5) x2, bp오미리 (5) x2; errors 0, 2 (주세요 -> 줘요), 2 (bp -> 비피), 5 (missing)
    assert m["cer"]["value"] == pytest.approx(9 / 20) and m["cer"]["n"] == 4 and m["cer"]["n_groups"] == 2
    assert m["cer"]["n_missing"] == 1
    assert m["wer"]["value"] == pytest.approx((0 + 1 + 1 + 3) / 10)
    assert m["ser"]["value"] == pytest.approx(3 / 4)
    assert m["parse_rate"] == 0.75 and m["error_rate"] == 0.25 and m["errors"] == {"http_503": 1}
    assert m["by_voice"]["chul"]["value"] == pytest.approx(2 / 10) and m["by_voice"]["chul"]["n"] == 2
    assert m["by_voice"]["maia"]["value"] == pytest.approx(7 / 10)
    assert set(m["by_cs_style"]) == {"none", "english"} and "wer" in m["by_cs_style"]["none"]
    assert m["by_digits"]["with digits"]["n"] == 2 and m["by_digits"]["without digits"]["n"] == 2
    assert {"by_intent", "by_speaker", "normaliser"} <= set(m)


# ---- fetch from KHDP (mock) ----


class StorageMock:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.requests: list[httpx.Request] = []
        self.client = httpx.Client(transport=httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.host == "api.khdp.net":
            assert request.url.path == "/datasets/Synthetic-K-OR-Speech-Audio/1.0.0/files"
            return httpx.Response(200, json={"items": [{"key": "1.0.0/synthetic_kor_speech_audio_v1.zip",
                                                         "size": len(self.payload),
                                                         "url": "https://storage.test/774/a.zip?X-Amz-Signature=x"}]})
        assert "authorization" not in request.headers  # the KHDP token never goes to storage
        start = int(request.headers.get("range", "bytes=0-").removeprefix("bytes=").rstrip("-"))
        if start:
            return httpx.Response(206, content=self.payload[start:])
        return httpx.Response(200, content=self.payload)

    def storage_gets(self):
        return [r for r in self.requests if r.url.host == "storage.test"]


def test_fetch_downloads_verifies_and_skips_when_present(tmp_path):
    payload = make_release(tmp_path / "src.zip")
    mock = StorageMock(payload)
    sha = hashlib.sha256(payload).hexdigest()
    path = fetch_audio(tmp_path, KhdpClient("khdp_pat_test", http=mock.client), sha256=sha)
    assert path == _archive(tmp_path) and path.read_bytes() == payload
    fetch_audio(tmp_path, KhdpClient("khdp_pat_test", http=mock.client), sha256=sha)
    assert len(mock.storage_gets()) == 1
    assert len(load_clips(tmp_path)) == 8


def test_fetch_resumes_a_partial_download(tmp_path):
    payload = make_release(tmp_path / "src.zip")
    part = _archive(tmp_path).with_suffix(".zip.part")
    part.parent.mkdir(parents=True)
    part.write_bytes(payload[:100])
    mock = StorageMock(payload)
    fetch_audio(tmp_path, KhdpClient("khdp_pat_test", http=mock.client), sha256=hashlib.sha256(payload).hexdigest())
    [get] = mock.storage_gets()
    assert get.headers["range"] == "bytes=100-"
    assert _archive(tmp_path).read_bytes() == payload and not part.exists()


def test_fetch_rejects_a_checksum_mismatch(tmp_path):
    payload = make_release(tmp_path / "src.zip")
    mock = StorageMock(payload)
    with pytest.raises(ValueError, match="checksum"):
        fetch_audio(tmp_path, KhdpClient("khdp_pat_test", http=mock.client), sha256="0" * 64)
    assert not _archive(tmp_path).exists()
