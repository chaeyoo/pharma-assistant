"""불변식(property) 테스트 — 예시가 아니라 '성질'을 입력 공간 전체에서 고정한다.

예시 기반 테스트(test_pv.py 등)는 '아는 실패'를 고정하지만, 컴플라이언스
도구의 위험은 **아무도 예상 못 한 입력**에서 나온다. 그래서 고정 시드로
케이스 서술을 다양하게 합성(fuzz)해, 어떤 조합에서도 깨지면 안 되는
성질(불변식)을 검증한다:

  1. 결정론  — 같은 입력이면 항상 같은 출력(감사 가능성의 전제).
  2. 닫힌 출력 집합 — 보고기한은 규정이 정의한 값(0·15·None) 밖으로
     절대 나가지 않는다. LLM이라면 보장할 수 없는 성질.
  3. 날짜 일관성 — deadline_date == awareness_date + deadline_days.
  4. 마스킹 멱등성 — redact(redact(x)) == redact(x). 마스킹이 자기 출력을
     다시 마스킹(이중 마스킹)하면 존재 신호([이름]님)가 파괴되어
     최소보고요건 판정이 무너지므로, 멱등성은 미관이 아니라 요건이다.
  5. PII 비유출 — 어떤 케이스 조합이든 MCP 도구의 전체 응답 JSON 어디에도
     원문 PII(주민번호·전화·이름)가 남지 않는다.
  6. 코딩 계층 배타성 — 확정/후보/미코딩 3계층은 서로 겹치지 않는다.
     겹치는 순간 '검수된 집계'와 '검수 대기 큐'의 신뢰 등급 구분이 무너진다.
  7. 검증기 자기일관성 — 어떤 텍스트도 자기 자신을 신뢰 소스로 주면 항상
     통과한다: verify_answer(x, [x]).ok. 답변 쪽과 근거 쪽의 클레임 추출이
     비대칭이 되는 회귀(정규화·단위 목록을 한쪽만 고친 경우)를 잡는다 —
     CleanPassRate(verify_eval)의 성질 버전.

시드를 고정하는 이유: CI에서 같은 케이스가 재현되어야 실패를 디버깅할 수
있다. 무작위성은 '입력 다양성'을 위한 것이지 '매번 다른 테스트'가 목적이
아니다.
"""
from __future__ import annotations

import datetime as _dt
import json
import random

import pytest

# 서버 내부 단위 테스트 — 도구 함수를 직접 호출한다(HTTP 클라이언트 경로는 test_mcp/test_agent 가 커버)
from src.mcp_server.server import assess_adverse_event, draft_ae_report
from src.pv.redactor import redact
from src.pv.report import build_report
from src.pv.triage import assess_case

# ---------------------------------------------------------------------------
# 고정 시드 케이스 합성기 — 증상/경과/약물/PII/잡음 조각을 조합
# ---------------------------------------------------------------------------
_SYMPTOMS = [
    "두드러기가 온몸에 났다", "숨쉬기 힘들어했다", "아나필락시스 쇼크가 왔다",
    "저혈당 증상을 보였다", "청력 저하를 호소했다", "저릿저릿한 감각이 있다고 했다",
    "몸이 좋지 않다고 했다", "가벼운 두통이 있었다", "혈압이 떨어졌다", "간수치가 올랐다",
]
_OUTCOMES = ["입원했다", "사망했다", "회복했다", "중환자실로 옮겨졌다", "경과 관찰 중이다", ""]
_DRUGS = ["세파졸린주사를 투여받고", "타이레놀정을 복용하고", "환자가 약을 먹고", ""]
_PII = [
    "김철수님(750101-1234567)", "이영희님", "연락처 010-9876-5432,",
    "보호자 이메일 guardian@example.com,", "환자번호: A-2291,", "",
]
_REPORTERS = ["담당 의사가 보고했다.", "약사가 접수했다.", "", "보호자가 전화로 알렸다."]

_RAW_PII_VALUES = ["750101-1234567", "010-9876-5432", "guardian@example.com", "김철수", "이영희", "A-2291"]


def _fuzz_cases(n: int = 60) -> list[str]:
    rng = random.Random(20260710)  # 고정 시드 — CI 재현성
    cases = []
    for _ in range(n):
        parts = [
            rng.choice(_PII), rng.choice(_DRUGS), rng.choice(_SYMPTOMS) + ".",
            rng.choice(_OUTCOMES), rng.choice(_REPORTERS),
        ]
        cases.append(" ".join(p for p in parts if p))
    return cases


_CASES = _fuzz_cases()


# ---------------------------------------------------------------------------
# 1. 결정론 — 같은 입력, 같은 출력
# ---------------------------------------------------------------------------
def test_build_report_is_deterministic():
    for case in _CASES:
        a = build_report(case, awareness_date="2026-07-01")
        b = build_report(case, awareness_date="2026-07-01")
        assert a.draft_markdown == b.draft_markdown
        assert a.reportable == b.reportable and a.missing == b.missing
        assert [t.pt for t in a.coded_terms] == [t.pt for t in b.coded_terms]


# ---------------------------------------------------------------------------
# 2·3. 트리아지 닫힌 출력 집합 + 날짜 일관성
# ---------------------------------------------------------------------------
def test_triage_outputs_stay_in_closed_set():
    for case in _CASES:
        t = assess_case(case, awareness_date="2026-07-01")
        # 기한은 규정(REG-005)이 정의한 값 밖으로 절대 나가지 않는다
        assert t.deadline_days in (0, 15, None), f"규정 밖 기한: {t.deadline_days} ← {case}"
        # 중대성 판정과 충족 기준 목록은 논리적으로 결합돼 있다
        assert t.is_serious == bool(t.criteria_met)
        # 비중대 ↔ 기한 없음(PSUR), 중대 ↔ 기한 날짜 존재
        assert (t.deadline_date is None) == (not t.is_serious)


def test_triage_deadline_date_arithmetic():
    aware = _dt.date(2026, 7, 1)
    for case in _CASES:
        t = assess_case(case, awareness_date=aware.isoformat())
        if t.deadline_days is not None:
            expected = (aware + _dt.timedelta(days=t.deadline_days)).isoformat()
            assert t.deadline_date == expected


# ---------------------------------------------------------------------------
# 4. 마스킹 멱등성
# ---------------------------------------------------------------------------
def test_redaction_is_idempotent():
    for case in _CASES:
        once = redact(case)
        twice = redact(once.text)
        assert twice.text == once.text, f"이중 마스킹: {once.text!r} → {twice.text!r}"
        assert not twice.counts, f"재마스킹 발생: {twice.counts} ← {once.text!r}"


# ---------------------------------------------------------------------------
# 5. PII 비유출 — MCP 도구 응답 전체(JSON)를 뒤져도 원문 PII가 없다
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tool", [assess_adverse_event, lambda c: draft_ae_report(c)])
def test_no_pii_leaks_through_tool_output(tool):
    for case in _CASES:
        if not any(v in case for v in _RAW_PII_VALUES):
            continue
        dumped = json.dumps(tool(case), ensure_ascii=False, default=str)
        for v in ["750101-1234567", "010-9876-5432", "guardian@example.com", "김철수님", "이영희님"]:
            assert v not in dumped, f"PII 유출: {v} ← {case}"


# ---------------------------------------------------------------------------
# 6. 코딩 계층 배타성 — 확정/후보/미코딩은 겹치지 않는다
# ---------------------------------------------------------------------------
def test_coding_layers_are_mutually_exclusive():
    for case in _CASES:
        r = build_report(case)
        coded_pts = [t.pt for t in r.coded_terms]
        assert len(coded_pts) == len(set(coded_pts)), f"확정 PT 중복: {coded_pts}"
        cand_pts = {t.pt for t in r.candidate_terms}
        assert not (set(coded_pts) & cand_pts), f"확정∩후보: {set(coded_pts) & cand_pts}"
        coded_verbatims = {t.verbatim for t in r.coded_terms} | {t.verbatim for t in r.candidate_terms}
        for u in r.uncoded_expressions:
            assert u not in coded_verbatims, f"미코딩이 코딩 계층과 중복: {u}"


# ---------------------------------------------------------------------------
# 7. 검증기 자기일관성 — verify_answer(x, [x]) 는 어떤 x 에서도 통과한다
# ---------------------------------------------------------------------------
_CLAIM_FRAGMENTS = [
    "인지일로부터 15일 이내 보고한다.", "심사는 **120 근무일** 이내에 처리된다.",
    "보완 회신은 90일 이내, 재보완은 6개월 미만.", "유효기간은 03년이며 2026-07-25 까지다.",
    "접수 후 15~30일 소요된다.", "보름 안에 제출하고 한 달 뒤 재심사.",
    "수수료는 1,000회 기준 5% 할인.", "2주일 이상 지속되면 재평가한다.",
    "지체 없이 보고한다(수치 없음).", "안정성시험은 6개월 이상 수행한다.",
]


def test_verifier_self_consistency():
    rng = random.Random(20260710)
    for _ in range(60):
        text = " ".join(rng.sample(_CLAIM_FRAGMENTS, k=rng.randint(1, len(_CLAIM_FRAGMENTS))))
        from src.verify.verifier import verify_answer

        v = verify_answer(text, [text])
        assert v.ok, f"자기 자신을 근거로 줘도 실패: {v.summary()} ← {text!r}"
