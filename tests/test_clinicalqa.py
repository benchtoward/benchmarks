import pandas as pd

from benchtoward.tasks.clinicalqa import QAItem, load_items, parse_answer, render_messages

ABCDE = set("ABCDE")


def test_parse_json_last_line():
    assert parse_answer('생각해 보면...\n{"answer": "A"}\n{"answer": "C"}', ABCDE) == "C"


def test_parse_korean_label():
    assert parse_answer("정답: B", ABCDE) == "B"


def test_parse_bare_letter():
    assert parse_answer(" D. ", ABCDE) == "D"


def test_no_false_letter_from_cbc():
    assert parse_answer("CBC 결과상 빈혈이 있고 CT에서 이상 소견", ABCDE) is None


def test_letter_not_allowed():
    assert parse_answer('{"answer": "E"}', set("ABCD")) is None


def _item(exam="", options=None):
    options = options or {"A": "가", "B": "나", "C": "다", "D": "라", "E": "마"}
    return QAItem(instance_id="cqa-1", question="질문입니다", exam=exam, options=options, answer="A",
                  category="Cardiology", chief_complaint="흉통")


def test_render_skips_empty_exam_and_options():
    msgs = render_messages(_item(exam="", options={"A": "가", "B": "나", "C": "다", "D": "라"}))
    assert msgs[0]["role"] == "system" and '{"answer": "<letter>"}' in msgs[0]["content"]
    user = msgs[1]["content"]
    assert "검사 소견" not in user and "E." not in user and "D. 라" in user
    assert user.rstrip().endswith("Answer with one letter from: A, B, C, D.")


def test_render_includes_exam_when_present():
    user = render_messages(_item(exam="Hb 7.0"))[1]["content"]
    assert "검사 소견:\nHb 7.0" in user


def test_load_items(tmp_path):
    path = tmp_path / "train.parquet"
    pd.DataFrame([
        {"question_id": 7, "chief_complaint": "흉통", "purpose": "", "question": "Q", "exam": "",
         "options": {"option_A": "a", "option_B": "b", "option_C": "c", "option_D": "d", "option_E": ""},
         "answer": "B", "explanation": "", "source": "", "category": "Cardiology"},
    ]).to_parquet(path)
    [item] = load_items(path)
    assert item.instance_id == "cqa-7" and item.answer == "B"
    assert list(item.options) == ["A", "B", "C", "D"]


def test_parse_latex_boxed():
    assert parse_answer("...\n\nFinal Answer: The final answer is $\\boxed{B}$", ABCDE) == "B"
    assert parse_answer("so \\boxed{\\text{D}}", ABCDE) == "D"


def test_parse_answer_is_and_bold_and_korean_particle():
    assert parse_answer("Therefore the answer is (D).", ABCDE) == "D"
    assert parse_answer("**Answer:** C", ABCDE) == "C"
    assert parse_answer("따라서 정답은 E입니다.", ABCDE) == "E"


def test_json_still_wins_over_other_mentions():
    assert parse_answer('The answer is A at first glance... {"answer": "B"}', ABCDE) == "B"
