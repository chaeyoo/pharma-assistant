"""RA 업무 도구 순수 로직(src/ra/tasks.py) 단위 테스트 — MCP 계층 없이.

창 필터·D-day 계산·연체 포함·에러 계약을 합성 데이터와 고정 기준일로
결정론적으로 검증한다(실데이터 마감일은 시한부라 오늘 날짜에 따라 결과가
변한다 — 계약 자체는 데이터·날짜와 무관해야 한다). MCP 도구 경유의 계약은
tests/test_mcp.py 가 그대로 커버한다(래퍼화 전후 무수정 통과가 이동의 조건).
"""
from __future__ import annotations

import datetime as _dt

from src.ra.tasks import deadlines_within, load_ra_tasks, submission_checklist

_TODAY = _dt.date(2026, 7, 15)

_DATA = {
    "deadlines": [
        {"id": "T-1", "item": "창 안 마감", "type": "품목허가",
         "due_date": "2026-07-25", "owner": "RA1팀", "status": "진행중"},
        {"id": "T-2", "item": "창 밖 마감", "type": "품목허가",
         "due_date": "2026-09-01", "owner": "RA1팀", "status": "대기"},
        {"id": "T-3", "item": "연체 마감", "type": "안전관리",
         "due_date": "2026-07-10", "owner": "PV팀", "status": "긴급"},
        {"id": "T-4", "item": "오늘 마감", "type": "안전관리",
         "due_date": "2026-07-15", "owner": "PV팀", "status": "진행중"},
    ],
    "checklists": {"품목허가": ["GMP 적합판정 유효기간 확인"]},
}


def test_window_filter_upper_bound_only():
    """창 필터는 상한(within_days)만 건다 — 창 밖 미래는 제외, 과거(연체)는 포함.

    연체를 제외하면 '지연 위험 항목' 질문에 가장 급한 항목이 조용히 빠진다."""
    out = deadlines_within(within_days=30, data=_DATA, today=_TODAY)
    ids = [d["id"] for d in out["deadlines"]]
    assert "T-2" not in ids, "창 밖(48일 뒤) 마감이 30일 창에 들어왔다"
    assert set(ids) == {"T-1", "T-3", "T-4"}
    assert out["count"] == len(out["deadlines"]) == 3


def test_d_day_arithmetic_and_overdue_sign():
    out = deadlines_within(within_days=30, data=_DATA, today=_TODAY)
    by_id = {d["id"]: d for d in out["deadlines"]}
    assert by_id["T-1"]["d_day"] == 10   # 2026-07-25 - 2026-07-15
    assert by_id["T-4"]["d_day"] == 0    # 오늘 마감
    assert by_id["T-3"]["d_day"] == -5   # 연체는 음수 D-day 로 표시된다
    assert out["today"] == "2026-07-15"


def test_deadlines_sorted_by_due_date():
    out = deadlines_within(within_days=365, data=_DATA, today=_TODAY)
    dues = [d["due_date"] for d in out["deadlines"]]
    assert dues == sorted(dues)


def test_type_filter_and_unknown_type_error_contract():
    """유형 필터는 정확 일치 — 미존재 유형은 빈 목록이 아니라 error+available
    (조용한 빈 결과 금지: 오타 필터에 '마감 없음'은 자신 있는 오답이다)."""
    ok = deadlines_within(within_days=365, task_type="안전관리", data=_DATA, today=_TODAY)
    assert {d["id"] for d in ok["deadlines"]} == {"T-3", "T-4"}
    bad = deadlines_within(within_days=365, task_type="존재하지않는유형", data=_DATA, today=_TODAY)
    assert "error" in bad and bad["available"] == ["안전관리", "품목허가"]
    assert "deadlines" not in bad, "에러 응답에 빈 목록이 섞이면 계약이 흐려진다"


def test_checklist_contract_known_and_unknown():
    ok = submission_checklist("품목허가", data=_DATA)
    assert ok == {"category": "품목허가", "items": ["GMP 적합판정 유효기간 확인"]}
    bad = submission_checklist("존재하지않는유형", data=_DATA)
    assert "error" in bad and bad["available"] == ["품목허가"]
    # 빈 카테고리는 '(미지정)' 표기로 에코 없이 안내(원 동작 보존)
    empty = submission_checklist("", data=_DATA)
    assert "error" in empty and "(미지정)" in empty["error"]


def test_default_data_loads_real_file():
    """data 미주입 시 실데이터(data/ra_tasks.json)를 로드한다 — 스키마 존재만
    확인한다(마감일 값은 시한부라 preflight 가 부패를 별도 감시)."""
    data = load_ra_tasks()
    assert data["deadlines"] and data["checklists"]
    out = deadlines_within(within_days=3650, data=data, today=_TODAY)
    assert out["count"] == len(data["deadlines"])
