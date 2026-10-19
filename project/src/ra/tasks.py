"""RA 도메인 ② 업무 도구 순수 로직 — 마감일 조회·제출 체크리스트.

src/pv/ 가 PV 케이스 처리(트리아지·인과성·코딩·보고서)를 MCP 계층 밖의
순수 로직으로 들고 있는 것과 대칭인 RA 갈래다. MCP 계층
(src/mcp_server/server.py 의 get_ra_deadlines·get_submission_checklist)은
자유 텍스트 인자 마스킹과 도구 노출만 담당하고, 데이터 로드·기간 창 필터·
D-day 계산·에러 계약 구성은 전부 여기서 한다 — MCP 없이 단위 테스트가
가능하고(tests/test_ra.py), 응답 스키마는 도구 계약과 1:1 이라 이동 전후
도구의 외부 동작은 불변이다.

에러 계약: 존재하지 않는 유형/카테고리는 빈 목록이 아니라
{"error", "available": [...]} 로 답한다 — 오타 난 필터에 "임박한 마감이
없습니다"라고 답하는 것은 '자신 있는 오답'이다(조용한 빈 결과 금지).
available 목록은 LLM 에이전트의 자가 정정 재료가 된다.
"""
from __future__ import annotations

import datetime as _dt
import json

from .. import config


def load_ra_tasks() -> dict:
    """RA·PV 업무 데이터(data/ra_tasks.json)를 로드한다.

    파일에는 RA팀 마감뿐 아니라 PV팀 마감(PSUR·신속보고)도 포함되어 있다
    — 도구 이름(get_ra_deadlines)은 코드 호환성을 위해 유지된 식별자다.
    """
    return json.loads(config.RA_TASKS_FILE.read_text(encoding="utf-8"))


def deadlines_within(
    within_days: int = 30,
    task_type: str = "",
    data: dict | None = None,
    today: _dt.date | None = None,
) -> dict:
    """오늘 기준 within_days 이내(지난 마감 포함)의 항목을 마감일 순으로 반환한다.

    Args:
        within_days: 오늘부터 며칠 이내의 마감을 조회할지.
        task_type: 특정 유형만 필터. 빈 문자열이면 전체.
        data: 업무 데이터(생략 시 파일 로드) — 테스트 주입 지점.
        today: 기준일(생략 시 오늘) — D-day 계산을 결정론적으로 테스트하는 지점.

    Returns:
        {"today", "count", "deadlines": [{..., "d_day"}...]} 또는
        미지원 유형이면 {"error", "available": [...]}.
        d_day 가 음수면 이미 지난(연체) 마감이다 — 창 필터는 상한(within_days)만
        두고 과거를 제외하지 않는다: 지난 마감을 숨기면 "지연 위험 항목" 질문에
        연체 항목이 조용히 빠진다(연체야말로 가장 급한 마감이다).
    """
    if data is None:
        data = load_ra_tasks()
    if today is None:
        today = _dt.date.today()
    # 존재하지 않는 유형은 빈 목록이 아니라 에러로 답한다 — 오타 난 필터에
    # "임박한 마감이 없습니다"라고 답하는 것은 '자신 있는 오답'이다(조용한 빈
    # 결과 금지). available 을 함께 주므로 LLM 에이전트가 스스로 정정 재시도한다
    # (submission_checklist 와 동일한 에러 계약).
    known_types = sorted({d["type"] for d in data["deadlines"]})
    if task_type and task_type not in known_types:
        return {"error": f"'{task_type}' 유형의 업무가 없음", "available": known_types}
    out = []
    for d in data["deadlines"]:
        due = _dt.date.fromisoformat(d["due_date"])
        d_day = (due - today).days
        if d_day > within_days:
            continue
        if task_type and d["type"] != task_type:
            continue
        out.append({**d, "d_day": d_day})
    out.sort(key=lambda x: x["due_date"])
    return {"today": today.isoformat(), "count": len(out), "deadlines": out}


def submission_checklist(category: str, data: dict | None = None) -> dict:
    """규제 제출 유형별 준비 체크리스트를 반환한다.

    Returns:
        {"category", "items": [...]} 또는 미지원 카테고리면
        {"error", "available": [...]} (deadlines_within 과 동일한 에러 계약).
    """
    if data is None:
        data = load_ra_tasks()
    checklists = data["checklists"]
    if category not in checklists:
        return {"error": f"'{category or '(미지정)'}' 체크리스트 없음", "available": list(checklists)}
    return {"category": category, "items": checklists[category]}
