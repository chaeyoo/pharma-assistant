"""검증기 메타 평가(eval/verify_eval.py)의 회귀 가드 — '핀에 알람을 단다'.

verify_eval 의 탐지율 축들은 '근거에 없는 값·방향·역할'로 합성한 변조라
1.0 이 정상이고, 깨지는 순간이 곧 검증기 회귀다 — 그래서 문서들은 이 수치를
"회귀를 고정하는 핀"이라 불러 왔다. 그런데 CI 는 eval 스크립트를 실행만 할 뿐
결과가 나빠져도 exit 0 이었다: **핀이라 부르지만 아무것도 강제하지 않는
상태**(계기판은 있는데 알람이 없는 형태의 논리적 비약)였다. PV 평가에는
회귀 가드(test_pv_eval.py)가 있으면서 검증기 평가에는 없던 비대칭이기도 하다.
이 테스트가 그 핀을 CI 실패로 강제한다.

표본 수 하한도 함께 고정한다 — 변조 하네스는 '치환 실패는 표본에서 제외'
하므로, 치환 정규식이 조용히 깨지면 표본이 줄어든 채 탐지율 1.0 이 유지된다.
'측정이 조용히 수축하는' 실패는 통과처럼 보인다(분자만 보는 지표의 사각지대).
"""
from __future__ import annotations

import pytest

from eval.verify_eval import _e2e_pass_rate, evaluate

# 표본 수 하한 — 현재 실측(n_swap=20, n_offset=23, n_direction=16, n_native=4,
# n_date=8, n_role=5, n_partial_role=5, n_range=4, n_clean=32, n_paraphrase=2)
# 에서 보수적 여유를 둔 값. 코퍼스·평가셋이 커지면 늘어나는 방향만 정상이다.
_MIN_N = {
    "n_clean": 30,
    "n_swap": 15,
    "n_offset": 18,
    "n_direction": 12,
    "n_native": 3,
    "n_paraphrase": 1,
    "n_date": 6,
    "n_role": 4,
    "n_partial_role": 4,  # v8 — 부분 날짜 표기 역할 스왑
    "n_range": 3,         # v8 — 하이픈 범위 하한 위조
    "n_partial": 6,
}


@pytest.fixture(scope="module")
def res() -> dict:
    return evaluate()


def test_sample_sizes_do_not_silently_shrink(res):
    for key, floor in _MIN_N.items():
        assert res[key] >= floor, f"{key}={res[key]} < {floor} — 변조/수집 하네스가 조용히 수축했다"


def test_detection_axes_are_pinned_at_full(res):
    """합성 변조는 정의상 '근거에 없는 값'이므로 탐지율 1.0 이 정상 — 미만이면 회귀."""
    assert res["swap_detected"] == res["n_swap"], "교차문서 치환 탐지 회귀"
    assert res["off_detected"] == res["n_offset"], "오프셋 변조 탐지 회귀"
    assert res["dir_detected"] == res["n_direction"], "방향 뒤집기 탐지 회귀"
    assert res["nat_detected"] == res["n_native"], "고유어 치환 탐지 회귀"
    assert res["date_detected"] == res["n_date"], "날짜 시프트 탐지 회귀"
    assert res["role_detected"] == res["n_role"], "날짜 역할 스왑 탐지 회귀"
    assert res["partial_role_detected"] == res["n_partial_role"], "부분 날짜 역할 스왑 탐지 회귀 (v8)"
    assert res["range_detected"] == res["n_range"], "하이픈 범위 하한 위조 탐지 회귀 (v8)"
    assert res["partial_detected"] == res["n_partial"], "부분 날짜 시프트 탐지 회귀"


def test_false_positive_axes_are_pinned_at_zero(res):
    """오탐이 늘면 alert fatigue 로 검증 계층 전체가 죽는다 — 통과율 1.0 을 지킨다."""
    assert res["clean_pass"] == res["n_clean"], "근거 발췌 답변에 오탐 — 추출 비대칭 회귀"
    assert res["para_pass"] == res["n_paraphrase"], "동치 고유어 표기(보름=15일)에 오탐"
    assert res["date_clean"] == res["n_date"], "도구 계산 마감일 인용에 오탐"
    assert res["partial_clean"] == res["n_partial"], "연도 없는 'M월 D일' 재서술에 오탐"


def test_version_axes(res):
    assert res["SupersededFlagged"], "폐지본 인용이 감지되지 않음"
    assert res["HistoryModeAllowed"], "이력 조회 모드에서 폐지본 인용이 오탐됨"


@pytest.mark.asyncio
async def test_e2e_pass_rate_is_full():
    """구성이 개입하지 않는 유일한 실측 — 오프라인 에이전트 실응답 전수 통과."""
    e2e = await _e2e_pass_rate()
    assert e2e["n"] >= 35
    assert e2e["ok"] == e2e["n"], e2e["failures"]
