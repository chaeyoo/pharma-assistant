"""답변 사후 검증(src/verify) 테스트 — 수치 대조 · 버전 점검 · 에이전트 통합."""
from __future__ import annotations

import pytest

from src.agent.agent import RaAgent
from src.verify.verifier import (
    BLOCKING_AXES,
    WARNING_AXES,
    ClaimCheck,
    VerificationResult,
    extract_claims,
    verify_answer,
    warning_text,
)


# ---------------------------------------------------------------------------
# 클레임 추출
# ---------------------------------------------------------------------------
def test_extract_normalizes_spacing_and_zeros():
    nums, dates = extract_claims("보고는 15 일 이내, 유효기간 03년, 마감 2026-07-25")
    assert ("15", "일") in nums and ("3", "년") in nums
    assert "2026-07-25" in dates


def test_extract_ignores_bare_numbers():
    nums, _ = extract_claims("근거 3건 중 REG-005 문서 1순위")  # 단위 없는 숫자·코드
    assert ("5", "일") not in nums and not any(u == "일" for _, u in nums)


def test_extract_working_days_as_distinct_unit():
    """'근무일'은 '일'과 다른 단위 — 코퍼스 핵심 처리기한의 검증 사각지대였다."""
    nums, _ = extract_claims("신약: 신청일로부터 **120 근무일** 이내")
    assert ("120", "근무일") in nums
    assert ("120", "일") not in nums  # '일'로 오분류되면 환산 오류를 못 잡는다


def test_extract_range_lower_bound():
    """'15~30일'의 하한(15일)도 클레임이다 — 주 정규식은 상한만 잡는다."""
    nums, _ = extract_claims("접수 후 15~30일 소요")
    assert ("15", "일") in nums and ("30", "일") in nums


def test_extract_comma_grouped_number():
    """'1,000회'가 '000회'로 오추출되면 값 대조가 무의미해진다."""
    nums, _ = extract_claims("수수료는 1,000회 기준")
    assert ("1000", "회") in nums and not any(n == "0" for n, _ in nums)


def test_extract_native_numerals_symmetric():
    """고유어 수사(보름=15일)는 답변·근거 양쪽에서 같은 canonical 로 추출된다."""
    nums, _ = extract_claims("보름 이내 보고, 한 달 뒤 재심사, 일주일 관찰")
    assert ("15", "일") in nums and ("1", "개월") in nums and ("7", "일") in nums


def test_extract_week_suffix_normalized():
    nums, _ = extract_claims("2주일 이상 지속")
    assert ("2", "주") in nums


# ---------------------------------------------------------------------------
# 수치 대조
# ---------------------------------------------------------------------------
def test_supported_claim_passes():
    v = verify_answer("중대 이상사례는 15일 이내 보고합니다.", ["규정: 인지일로부터 15일 이내 신속보고"])
    assert v.ok and v.summary()["checked"] == 1


def test_unsupported_claim_is_flagged():
    v = verify_answer("중대 이상사례는 30일 이내 보고합니다.", ["규정: 인지일로부터 15일 이내 신속보고"])
    assert not v.ok and "30일" in v.unsupported
    assert "30일" in warning_text(v)


def test_unit_paraphrase_is_flagged():
    """'15일'을 '약 2주'로 환산하면 근거에 없는 값 — 마감일 환산 오차는 리스크다."""
    v = verify_answer("약 2주 이내에 보고하면 됩니다.", ["인지일로부터 15일 이내 신속보고"])
    assert not v.ok and "2주" in v.unsupported


def test_date_claim_checked_against_tool_output():
    trusted = ['{"awareness_date": "2026-07-10", "deadline_date": "2026-07-25"}']
    ok = verify_answer("보고 기한은 2026-07-25 입니다.", trusted)
    bad = verify_answer("보고 기한은 2026-07-30 입니다.", trusted)
    assert ok.ok and not bad.ok


def test_no_claims_is_trivially_ok():
    v = verify_answer("근거를 찾지 못했습니다.", [])
    assert v.ok and v.summary()["checked"] == 0


def test_working_day_paraphrase_is_flagged():
    """'120 근무일'을 '120일'로 옮기면 실제 달력 기한이 달라진다 — 단위 환산 오류."""
    v = verify_answer("심사는 120일 이내입니다.", ["신청일로부터 **120 근무일** 이내"])
    assert not v.ok and "120일" in v.unsupported


def test_native_numeral_equivalence_passes():
    """보름=15일은 환산이 아니라 표기 변형 — 오탐을 내면 안 된다(alert fatigue)."""
    v = verify_answer("보름 이내에 보고하면 됩니다.", ["인지일로부터 15일 이내 신속보고"])
    assert v.ok


def test_wrong_native_numeral_is_flagged():
    """값이 다른 고유어(열흘=10일)는 v1에서 아예 추출되지 않아 조용히 통과했다."""
    v = verify_answer("열흘 이내에 보고하면 됩니다.", ["인지일로부터 15일 이내 신속보고"])
    assert not v.ok and "열흘" in v.unsupported


def test_supported_claim_carries_evidence_snippet():
    """지원된 클레임에는 근거 위치 스니펫이 붙는다 — 사람의 대조를 빠르게."""
    v = verify_answer("15일 이내 보고합니다.", ["규정 제5조: 인지일로부터 15일 이내 신속보고한다."])
    check = next(c for c in v.checks if c.claim == "15일")
    assert check.supported and "15일 이내 신속보고" in check.evidence


# ---------------------------------------------------------------------------
# 방향 한정어 — 수치가 맞아도 방향이 뒤집히면 컴플라이언스 오류
# ---------------------------------------------------------------------------
def test_direction_flip_is_flagged():
    v = verify_answer("90일 이후에 제출하면 됩니다.", ["보완 회신은 통상 **90일** 이내에 제출한다."])
    assert not v.ok and "90일 이후" in v.direction_conflicts
    assert "방향" in warning_text(v)


def test_direction_same_class_passes():
    """이내↔이하는 같은 방향(상한) — 동의 표현을 충돌로 오탐하지 않는다."""
    v = verify_answer("90일 이하로 제출합니다.", ["보완 회신은 통상 90일 이내에 제출한다."])
    assert v.ok


def test_direction_without_source_qualifier_not_flagged():
    """근거에 한정어가 없으면 판단 근거가 없다 — 보수적으로 플래그하지 않는다."""
    v = verify_answer("90일 이후에 제출합니다.", ["보완 회신 기간은 90일로 한다."])
    assert v.ok


def test_direction_matching_lower_bound_passes():
    v = verify_answer("6개월 이상 수행합니다.", ["안정성시험은 6개월 이상 수행한다."])
    assert v.ok


def test_date_direction_flip_is_flagged():
    """근거의 "…까지"(상한) 기한 날짜를 "… 이후"로 뒤집는 왜곡 — 날짜는 근거에
    실존해 존재 대조를 통과하므로, 방향 축이 날짜에도 있어야 잡힌다(수치에만
    방향 축이 있던 비대칭의 해소)."""
    v = verify_answer(
        "보완자료는 2026-07-25 이후에 제출하면 됩니다.",
        ["보완자료는 2026-07-25까지 제출한다."],
    )
    assert not v.ok and "2026-07-25 이후" in v.direction_conflicts
    assert "방향" in warning_text(v)


def test_date_direction_korean_notation_symmetric():
    """근거가 한국어 날짜 표기여도 정규화 후 대칭으로 대조된다."""
    v = verify_answer(
        "2026-07-25 이후 제출 가능합니다.",
        ["보완자료는 2026년 7월 25일까지 제출한다."],
    )
    assert not v.ok and "2026-07-25 이후" in v.direction_conflicts


def test_date_direction_same_class_passes():
    """까지↔이내는 같은 방향(상한) — 동의 표현을 충돌로 오탐하지 않는다."""
    v = verify_answer("2026-07-25까지 제출합니다.", ["제출 기한은 2026-07-25 이내로 한다."])
    assert v.ok


def test_date_direction_without_source_qualifier_not_flagged():
    """근거에 그 날짜의 한정어가 없으면(도구 라벨 JSON 등) 판단하지 않는다 — 보수성.
    "마감일 이후에는 지연보고로 처리된다" 같은 정당한 '이후' 용례를 오탐하지 않기
    위한 규칙이기도 하다(수치 방향 대조와 동일)."""
    v = verify_answer(
        "2026-07-25 이후에는 지연보고로 처리됩니다.",
        ['{"deadline_date": "2026-07-25"}'],
    )
    assert v.ok and not v.direction_conflicts


# ---------------------------------------------------------------------------
# 질문 에코 — 사용자 전제는 신뢰 소스가 아니라 '전제 확인' 라벨
# ---------------------------------------------------------------------------
def test_question_origin_claim_labeled_separately():
    v = verify_answer(
        "30일이 아니라 15일 이내입니다.",
        ["인지일로부터 15일 이내 신속보고"],
        question="보고 기한이 30일 맞나요?",
    )
    assert not v.ok and "30일" in v.question_origin
    w = warning_text(v)
    assert "전제 확인" in w and "환각" not in w


def test_question_does_not_promote_claim_to_supported():
    """질문에 있던 수치라도 supported 로 승격되지는 않는다 — 라벨만 다르다."""
    v = verify_answer("답은 30일입니다.", ["15일 이내"], question="30일인가요?")
    assert not v.ok and "30일" in v.unsupported


# ---------------------------------------------------------------------------
# 케이스 에코 — 사용자 사실은 지지 근거로 인정하되 from_case 라벨로 구분한다
# ---------------------------------------------------------------------------
def test_case_only_support_is_labeled_not_warned():
    """케이스의 "30일간 복용"이 답변의 "보고 기한 30일"을 지지하는 조용한 통과 —
    차단(재서술 오탐)도 침묵(승격 은폐)도 아닌 라벨로 가시화한다."""
    v = verify_answer(
        "케이스상 복용 기간은 30일이며, 보고 기한 규정은 15일 이내입니다.",
        ["규정: 인지일로부터 15일 이내 신속보고"],
        user_fact_texts=["환자가 A정을 30일간 복용 후 두드러기 발생"],
    )
    assert v.ok                                   # 경고는 아니다
    assert v.case_origin == ["30일"]              # 그러나 등급이 다름을 라벨로
    assert "15일" not in v.case_origin            # 규정 지지 클레임은 라벨 없음
    check = next(c for c in v.checks if c.claim == "30일")
    assert check.supported and check.from_case


def test_regulation_supported_claim_not_labeled_from_case():
    """같은 값이 규정 근거에도 있으면 케이스 라벨을 붙이지 않는다(strict 우선)."""
    v = verify_answer(
        "보고 기한은 15일 이내입니다.",
        ["인지일로부터 15일 이내 신속보고"],
        user_fact_texts=["환자가 15일 전부터 복용"],
    )
    assert v.ok and v.case_origin == []


def test_claim_nowhere_is_still_unsupported_with_facts_present():
    """케이스 계층이 있어도 어디에도 없는 값은 종전대로 미확인 경고다."""
    v = verify_answer(
        "보고 기한은 45일 이내입니다.",
        ["인지일로부터 15일 이내"],
        user_fact_texts=["환자가 30일간 복용"],
    )
    assert not v.ok and "45일" in v.unsupported


# ---------------------------------------------------------------------------
# 웹 계층(3계층) — 인터넷 검색 결과는 지지 근거로 인정하되 from_web 라벨로 분리
# ---------------------------------------------------------------------------
def test_web_only_support_is_labeled_not_warned():
    """명시 요청된 웹 검색의 재서술은 경고하지 않되(오탐 방지) 사내 규정
    근거로 승격되지도 않는다 — web_origin 라벨이 그 분리의 검증 계층 표현."""
    v = verify_answer(
        "인터넷 검색 결과, 개정 고시는 2026-05-01 시행 예정입니다.",
        ["규정: 인지일로부터 15일 이내 신속보고"],
        web_texts=["개정 고시 2026-05-01 시행 예정 (뉴스)"],
    )
    assert v.ok                                    # 경고는 아니다
    assert v.web_origin == ["2026-05-01"]          # 그러나 웹 유래임을 라벨로
    check = next(c for c in v.checks if c.claim == "2026-05-01")
    assert check.supported and check.from_web and not check.from_case


def test_strict_supported_claim_not_labeled_from_web():
    """같은 값이 규정 근거에도 있으면 웹 라벨을 붙이지 않는다(strict 우선)."""
    v = verify_answer(
        "보고 기한은 15일 이내입니다.",
        ["인지일로부터 15일 이내 신속보고"],
        web_texts=["웹 문서에도 15일 이내라고 적혀 있다"],
    )
    assert v.ok and v.web_origin == []


def test_case_layer_precedes_web_layer():
    """케이스 서술과 웹 양쪽에만 있는 값은 from_case — 계층 우선순위(규정·도구 >
    케이스 > 웹)에서 상위 계층이 지지하면 하위 라벨은 붙지 않는다."""
    v = verify_answer(
        "케이스상 복용 기간은 30일입니다.",
        ["규정: 15일 이내"],
        user_fact_texts=["환자가 30일간 복용"],
        web_texts=["웹 기사에도 30일 복용 사례"],
    )
    assert v.ok and v.case_origin == ["30일"] and v.web_origin == []


def test_claim_nowhere_is_still_unsupported_with_web_present():
    """웹 계층이 있어도 어디에도 없는 값은 종전대로 미확인 경고다 — '웹 검색
    답변이니 게이트 통과 불요'가 아니라 게이트는 계속 돌고 계층만 분리된다."""
    v = verify_answer(
        "보고 기한은 45일 이내입니다.",
        ["인지일로부터 15일 이내"],
        web_texts=["무관한 웹 검색 결과 본문"],
    )
    assert not v.ok and "45일" in v.unsupported


def test_web_origin_is_label_axis_not_blocking():
    """web_origin 은 case_origin 과 같은 등급 라벨 축 — ok 를 깨지 않고
    summary 에만 노출된다(라벨 축 정본 등록의 회귀 가드)."""
    from src.verify.verifier import LABEL_AXES

    assert "web_origin" in LABEL_AXES and "web_origin" not in BLOCKING_AXES
    v = verify_answer("개정 고시는 2026-05-01 시행", [], web_texts=["2026-05-01 시행"])
    assert v.ok and v.summary()["web_origin"] == ["2026-05-01"]


def test_gate_stats_counts_web_labeled():
    """web_labeled 계기판 — 웹 유래 지지 응답이 늘면 답변이 사내 규정 대신
    인터넷 결과에 기대기 시작했다는 신호(라벨은 죽어도 소리가 없다)."""
    from src.observability import GateStats

    gs = GateStats()
    gs.record({"ok": True, "checked": 1, "web_origin": ["2026-05-01"]})
    gs.record({"ok": True, "checked": 1, "web_origin": []})
    snap = gs.snapshot()
    assert snap["web_labeled"] == 1


# ---------------------------------------------------------------------------
# 날짜 역할 대조 — 두 날짜가 모두 근거에 있어도 역할(기한↔인지일)이 뒤바뀌면 잡는다
# ---------------------------------------------------------------------------
_TOOL_OUT = '{"awareness_date": "2026-07-10", "deadline_date": "2026-07-25", "deadline_days": 15}'


def test_role_swap_detected_even_when_both_dates_exist():
    """존재 대조를 '정의상' 통과하는 변조 — 역할 라벨 대조 축의 존재 이유."""
    v = verify_answer("보고 기한: 2026-07-10 (인지일 2026-07-25 기준)", [_TOOL_OUT])
    assert not v.ok
    assert "기한 2026-07-10" in v.role_conflicts and "인지일 2026-07-25" in v.role_conflicts
    assert not v.unsupported  # 두 날짜 모두 신뢰 소스에 실존 — 존재 축은 통과
    assert "날짜 역할" in warning_text(v)


def test_correct_roles_pass():
    v = verify_answer(
        "보고 기한: 2026-07-25 (인지일 2026-07-10 기준, 15일 이내)", [_TOOL_OUT, "15일 이내"]
    )
    assert v.ok and not v.role_conflicts


def test_role_check_needs_tool_labels():
    """검색 근거만 있으면(역할 라벨 없음) 판단 근거가 없다 — 플래그하지 않는다."""
    v = verify_answer("보고 기한은 2025-04-01 이후 적용", ["시행일 2025-04-01 명시"])
    assert v.ok and not v.role_conflicts


def test_role_keyword_must_be_adjacent():
    """'기한 규정은 <날짜> 시행'처럼 키워드가 날짜에 직접 붙지 않으면 역할 주장이 아니다."""
    v = verify_answer("기한 규정은 2026-07-10 시행 문서를 참조하세요", [_TOOL_OUT])
    assert not v.role_conflicts


def test_unsupported_date_near_keyword_is_existence_issue_not_role():
    """근거에 아예 없는 날짜는 존재 대조 축이 잡는다 — 역할 축과 중복 경고하지 않는다."""
    v = verify_answer("보고 기한: 2026-08-01", [_TOOL_OUT])
    assert "2026-08-01" in v.unsupported and not v.role_conflicts


# ---------------------------------------------------------------------------
# 검증 게이트 운영 계기판 (경고율 집계 — alert fatigue 조기 신호)
# ---------------------------------------------------------------------------
def test_gate_stats_records_and_snapshots():
    from src.observability import GateStats

    gs = GateStats()
    gs.record({"ok": True, "unsupported": [], "checked": 2})
    gs.record({"ok": False, "unsupported": ["30일"], "role_conflicts": ["기한 2026-07-10"], "checked": 3})
    snap = gs.snapshot()
    assert snap["responses"] == 2 and snap["warned"] == 1 and snap["warn_rate"] == 0.5
    assert snap["by_axis"] == {"unsupported": 1, "role_conflicts": 1}


def test_gate_stats_checked_rate_is_immune_to_traffic_mix():
    """warn_rate 는 회피·무클레임 응답이 분모에 섞여 트래픽 믹스에 따라 착시를
    만든다 — warn_rate_checked 는 '검증할 것이 있던 응답'만 분모로 잡는다."""
    from src.observability import GateStats

    gs = GateStats()
    gs.record({"ok": True, "checked": 2})                       # 클레임 있음·통과
    gs.record({"ok": False, "unsupported": ["30일"], "checked": 1})  # 클레임 있음·경고
    for _ in range(8):
        gs.record({"ok": True, "checked": 0})                   # 회피/무클레임 홍수
    snap = gs.snapshot()
    assert snap["warn_rate"] == 0.1          # 믹스에 희석된 값
    assert snap["checked_responses"] == 2
    assert snap["warn_rate_checked"] == 0.5  # 믹스와 무관한 실질 경고율


@pytest.mark.asyncio
async def test_agent_chat_feeds_gate_stats():
    """모든 chat 응답이 계기판에 집계된다 — /health 노출의 데이터 소스."""
    from src.observability import gate_stats

    before = gate_stats.snapshot()["responses"]
    r = await RaAgent().chat("신약 품목허가 심사 며칠 걸려?")
    assert r.verification
    assert gate_stats.snapshot()["responses"] == before + 1


# ---------------------------------------------------------------------------
# 인용 버전 점검 (폐지본 감지)
# ---------------------------------------------------------------------------
def test_superseded_citation_is_flagged():
    cites = [{"doc_id": "REG-013", "status": "superseded"}]
    v = verify_answer("이상사례는 30일 이내 보고", ["30일"], cites)
    assert not v.ok and v.superseded_cited == ["REG-013"]
    assert "REG-013" in warning_text(v)


def test_history_mode_allows_superseded():
    cites = [{"doc_id": "REG-013", "status": "superseded"}]
    v = verify_answer("구판 기준은 30일이었다", ["30일"], cites, allow_superseded=True)
    assert v.ok


def test_superseded_allowance_is_per_document():
    """경고를 끄는 스위치의 면적은 문서 단위 — 이력 검색이 반환한 문서(REG-013)의
    폐지본 인용만 면제되고, 같은 응답의 다른 폐지본 인용(현행 검색에 상류
    결함으로 섞여 든 문서)은 계속 경고된다. 전역 bool 이면 이력 턴 동안 버전
    축이 통째로 꺼진다."""
    cites = [
        {"doc_id": "REG-013", "status": "superseded"},
        {"doc_id": "REG-099", "status": "superseded"},
    ]
    v = verify_answer("구판 기준은 30일이었다", ["30일"], cites,
                      allowed_superseded_ids={"REG-013"})
    assert not v.ok and v.superseded_cited == ["REG-099"]
    # 허용 집합이 모든 인용을 덮으면 통과
    v2 = verify_answer("구판 기준은 30일이었다", ["30일"], cites,
                       allowed_superseded_ids={"REG-013", "REG-099"})
    assert v2.ok


def test_active_citation_not_flagged():
    v = verify_answer("15일 이내 보고", ["15일"], [{"doc_id": "REG-005", "status": "active"}])
    assert v.ok


# ---------------------------------------------------------------------------
# 에이전트 통합 — 모든 응답에 verification 이 부착되고, 정상 경로는 통과한다
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_agent_attaches_verification_and_passes():
    agent = RaAgent()
    for q in [
        "신약 품목허가 심사 며칠 걸려?",                       # 검색 경로
        "환자가 복용 후 아나필락시스로 입원했어요. 언제까지 보고?",  # 트리아지 경로
        "이번 주 마감 임박한 규제 업무는?",                     # 마감일 경로
    ]:
        r = await agent.chat(q)
        assert r.verification, f"verification 누락: {q}"
        assert r.verification["ok"], f"정상 경로 오탐: {q} → {r.verification}"
        assert "⚠ 자동 검증 경고" not in r.answer


@pytest.mark.asyncio
async def test_agent_abstention_has_trivial_verification():
    agent = RaAgent()
    r = await agent.chat("오늘 점심 메뉴 추천해줘")
    assert not r.grounded
    assert r.verification["ok"] and r.verification["checked"] == 0


def test_korean_date_notation_matches_iso_evidence():
    """한국어 날짜 표기("2026년 7월 25일")는 ISO 근거(2026-07-25)와 표기 정규화로
    대조된다 — 정규화가 없으면 (1) 올바른 답변이 미확인 날짜 오탐을 받고,
    (2) 날짜의 '25일' 성분이 기간 클레임으로 오추출된다."""
    v = verify_answer(
        "보고 기한은 2026년 7월 25일입니다.",
        ['"deadline_date": "2026-07-25", "awareness_date": "2026-07-10"'],
    )
    assert v.ok, v.summary()
    # 역방향(근거가 한국어 표기, 답변이 ISO)도 대칭으로 통과
    v2 = verify_answer("보고 기한은 2026-07-25입니다.", ["마감일은 2026년 7월 25일이다"])
    assert v2.ok, v2.summary()


def test_korean_date_in_question_gets_premise_label():
    """질문에 한국어 표기로 들어온 날짜를 답변이 ISO 로 재서술한 경우 —
    '환각'이 아니라 '전제 확인 필요'(from_question)로 라벨링되어야 한다."""
    v = verify_answer(
        "2024-06-01 기준으로는 심사 기간 규정이 없습니다.",
        ["심사 기간은 120 근무일"],
        question="2024년 6월 1일 기준 심사 기간은?",
    )
    assert "2024-06-01" in v.question_origin


def test_korean_date_component_not_extracted_as_duration():
    """"7월 15일"의 '15일'이 기간 클레임으로 추출되면, 근거의 '15일 이내'(기간)와
    우연히 일치해 통과하는 오염 경로가 생긴다 — 날짜 문맥은 날짜로만 읽는다."""
    from src.verify.verifier import extract_claims

    nums, dates = extract_claims("처리 시한은 2026년 7월 15일까지입니다")
    assert ("15", "일") not in nums
    assert "2026-07-15" in dates


# ---------------------------------------------------------------------------
# v5 — 표기 사각지대: 고유어 방향 · 부분 날짜 · 연도 표기 · ISO+'일' 접미 오염
# ---------------------------------------------------------------------------
def test_native_numeral_direction_flip_is_flagged():
    """"보름 이후"는 존재 축(보름=15일, 근거 실존)을 통과한다 — 방향 수집이
    숫자 표기에만 있으면 같은 왜곡이 표기에 따라 한쪽만 잡히는 축 간 비대칭."""
    v = verify_answer("보고는 보름 이후에 하면 됩니다.", ["인지일로부터 15일 이내 신속보고"])
    assert not v.ok and "보름 이후" in v.direction_conflicts


def test_native_numeral_direction_same_class_passes():
    """"보름 이내"(=15일 이내)는 근거와 같은 방향 — 오탐하지 않는다."""
    v = verify_answer("보름 이내에 보고합니다.", ["인지일로부터 15일 이내 신속보고"])
    assert v.ok


def test_source_native_qualifier_checks_digit_answer():
    """근거가 "보름 이내"라 쓰고 답변이 "15일 이후"라 써도 대칭으로 잡힌다."""
    v = verify_answer("15일 이후에 보고하면 됩니다.", ["접수 후 보름 이내 신속보고한다"])
    assert not v.ok and "15일 이후" in v.direction_conflicts


def test_case_echo_cannot_defuse_direction_conflict():
    """케이스 서술의 "15일 이후 증상"이 규정 "15일 이내"의 방향 뒤집기 경고를
    조용히 무력화하던 구멍 — 방향 판정 기준은 strict 계층이고, 케이스에 같은
    방향이 있으면 경고를 끄는 대신 from_case 라벨로 모호성을 가시화한다."""
    v = verify_answer(
        "보고는 15일 이후에 하면 됩니다.",
        ["인지일로부터 15일 이내 신속보고"],
        user_fact_texts=["환자가 복용 15일 이후 증상 발생"],
    )
    assert not v.ok and "15일 이후" in v.direction_conflicts
    check = next(c for c in v.checks if c.kind == "direction")
    assert check.from_case  # 경고 문구가 '재서술 가능성'을 함께 안내
    assert "재서술" in warning_text(v)


def test_case_echo_cannot_defuse_date_direction_conflict():
    """날짜 방향 축도 동일 — 케이스 서술이 기한 날짜의 방향 경고를 못 끈다."""
    v = verify_answer(
        "2026-07-25 이후 제출하면 됩니다.",
        ["보완자료는 2026-07-25까지 제출한다"],
        user_fact_texts=["진료기록은 2026-07-25 이후 확보 예정"],
    )
    assert not v.ok and "2026-07-25 이후" in v.direction_conflicts


def test_case_only_qualifier_does_not_create_conflict():
    """strict 에 그 값의 한정어가 없으면 케이스 서술의 한정어만으로 충돌을
    만들지 않는다 — 규정 근거 없는 판정 금지(보수성). 정당한 케이스 재서술
    ("30일 이후 증상 발생")은 방향 오탐 없이 case_origin 라벨만 받는다."""
    v = verify_answer(
        "환자는 복용 30일 이후 증상이 발생했습니다.",
        ["인지일로부터 15일 이내 신속보고"],
        user_fact_texts=["환자가 복용 30일 이후 증상 발생"],
    )
    assert v.ok and not v.direction_conflicts and v.case_origin == ["30일"]


def test_partial_date_restatement_passes():
    """마감일의 연도 없는 재서술("7월 25일")은 표기 변형 — '25일' 성분이 기간
    클레임으로 오추출되어 옳은 답변에 오탐이 붙던 경로의 봉합."""
    v = verify_answer("보고 기한은 7월 25일입니다.", ['"deadline_date": "2026-07-25"'])
    assert v.ok, v.summary()
    check = next(c for c in v.checks if c.claim == "7월 25일")
    assert check.kind == "date" and check.supported


def test_wrong_partial_date_is_flagged_as_date():
    """틀린 부분 날짜는 '30일'(기간)이 아니라 '7월 30일'(날짜)로 잡힌다 —
    부분 날짜가 아예 추출되지 않아 기간 오추출에 우연히 기대던 상태의 해소."""
    v = verify_answer("보고 기한은 7월 30일입니다.", ['"deadline_date": "2026-07-25"'])
    assert not v.ok and "7월 30일" in v.unsupported
    assert "30일" not in v.unsupported  # 기간 클레임으로 오추출되지 않는다


def test_partial_date_contamination_blocked():
    """근거에 우연히 기간 '25일'이 있어도, 틀린 날짜 "9월 25일"이 그 값에
    지지되어 통과하는 오염이 없어야 한다 — 부분 날짜는 날짜 축으로만 대조."""
    v = verify_answer(
        "마감은 9월 25일입니다.",
        ['"deadline_date": "2026-07-25"', "처리 기간은 25일 이내"],
    )
    assert not v.ok and "9월 25일" in v.unsupported


def test_partial_date_in_question_gets_premise_label():
    v = verify_answer(
        "7월 30일 마감이라는 전제는 확인되지 않습니다.",
        ['"deadline_date": "2026-07-25"'],
        question="마감이 7월 30일 맞나요?",
    )
    assert "7월 30일" in v.question_origin


def test_bare_year_supported_by_full_date():
    """"2025년 4월 개정" 속 '2025년'은 근거 날짜(2025-04-01)의 연도 성분 재서술 —
    미확인 오탐(alert fatigue)을 내지 않는다. 표기 변형이지 환산이 아니다."""
    v = verify_answer("이 규정은 2025년 4월 개정판 기준입니다.", ['"effective_date": "2025-04-01"'])
    assert v.ok, v.summary()


def test_bare_year_without_any_date_still_flagged():
    """근거 어디에도 없는 연도는 종전대로 미확인 경고다."""
    v = verify_answer("2019년 개정 기준입니다.", ['"effective_date": "2025-04-01"'])
    assert not v.ok and "2019년" in v.unsupported


def test_duration_years_not_matched_to_dates():
    """'3년'(기간)은 연도 폴백의 대상이 아니다 — 달력 연도 형태(19xx·20xx)만."""
    v = verify_answer("유효기간은 3년입니다.", ['"effective_date": "2025-04-01"'])
    assert not v.ok and "3년" in v.unsupported


def test_iso_date_with_il_suffix_not_extracted_as_duration():
    """근거의 "2026-07-25일이다" 표기에서 '25일'(기간)이 오추출되면, 답변의
    지어낸 '25일 기한'이 그 오염된 값에 지지되어 통과한다 — 경계로 차단."""
    nums, dates = extract_claims("시행일은 2026-07-25일이다")
    assert ("25", "일") not in nums and "2026-07-25" in dates
    v = verify_answer("처리 기한은 25일 이내입니다.", ["시행일은 2026-07-25일이다"])
    assert not v.ok and "25일" in v.unsupported


def test_legal_range_conjunction_covered_by_unit_extraction():
    """법령체 범위 표기 "15일 내지 30일"은 단위가 양쪽에 붙어 기존 추출로
    커버된다 — '내지' 전용 처리가 필요 없음을 회귀로 고정(루프 검토 결과)."""
    nums, _ = extract_claims("보완 기간은 15일 내지 30일로 한다")
    assert ("15", "일") in nums and ("30", "일") in nums


def test_gate_stats_counts_case_label():
    """case_origin 라벨은 경고가 아니라서 죽어도 소리가 없다 — 계기판이 라벨률
    추이를 세어야 '답변이 규정 대신 사용자 서술에 기대기 시작'하는 이동이 보인다."""
    from src.observability import GateStats

    gs = GateStats()
    gs.record({"ok": True, "checked": 2, "case_origin": ["30일"]})
    gs.record({"ok": True, "checked": 1})
    snap = gs.snapshot()
    assert snap["case_labeled"] == 1 and snap["warned"] == 0


# ---------------------------------------------------------------------------
# v8 — 부분 날짜의 방향·역할 축, 하이픈 범위, 방향 어휘 '이전'·'부터'
# ---------------------------------------------------------------------------
def test_partial_date_role_swap_detected():
    """부분 날짜 표기("7월 25일")로 역할을 뒤바꿔도 역할 축이 발화한다. (v8)

    존재 축은 접미 대조로 부분 표기를 지지하는데 역할 축이 ISO 만 수집하면,
    두 날짜가 모두 근거에 실존하는 역할 스왑이 그 표기로만 조용히 통과한다."""
    tool = '{"awareness_date":"2026-07-10","deadline_date":"2026-07-25"}'
    r = verify_answer("보고 기한은 7월 10일입니다 (인지일 7월 25일)", [tool])
    assert not r.ok and len(r.role_conflicts) == 2


def test_partial_date_role_correct_no_flag():
    tool = '{"awareness_date":"2026-07-10","deadline_date":"2026-07-25"}'
    r = verify_answer("보고 기한은 7월 25일입니다 (인지일 7월 10일)", [tool])
    assert r.ok and not r.role_conflicts


def test_partial_date_direction_flip_detected():
    """부분 날짜 표기의 방향 뒤집기("7월 25일 이후" vs 근거 "…까지")를 잡는다. (v8)"""
    r = verify_answer("7월 25일 이후에 제출하면 됩니다", ["제출 기한은 2026-07-25까지"])
    assert not r.ok and r.direction_conflicts == ["7월 25일 이후"]


def test_partial_date_direction_same_no_flag():
    r = verify_answer("7월 25일까지 제출하세요", ["제출 기한은 2026-07-25까지"])
    assert r.ok and not r.direction_conflicts


def test_hyphen_range_both_bounds_collected():
    """하이픈 범위("10-15일")는 상·하한이 모두 수집·대조된다. (v8)

    '-' 를 구분자에서 뺀 대가로 상한('15일' — 앞이 '-')까지 룩비하인드에
    걸려 표현 전체가 검증 사각지대였다(위조 하한 포함 조용한 통과)."""
    r = verify_answer("처리기간은 10-15일입니다", ["처리기간은 15일이다"])
    assert not r.ok and r.unsupported == ["10일"]
    ok = verify_answer("처리기간은 10-15일입니다", ["처리기간은 10-15일이다"])
    assert ok.ok


def test_hyphen_range_does_not_pollute_dates():
    """날짜+'일' 접미("2026-07-25일")가 하이픈 범위로 오추출되지 않는다. (v8)"""
    r = verify_answer("기한은 2026-07-25일입니다", ['{"deadline_date":"2026-07-25"}'])
    assert r.ok and not r.unsupported


def test_direction_word_ijeon_detected():
    """'이전'(상한 어휘) 뒤집기 — "15일 이전" vs 근거 "15일 이후". (v8)"""
    r = verify_answer("15일 이전에 보고해야 합니다", ["인지일로부터 15일 이후 보고한다"])
    assert not r.ok and r.direction_conflicts == ["15일 이전"]


def test_direction_word_buteo_detected():
    """'부터'(하한 기산점) 뒤집기 — "2026-07-25부터" vs 근거 "…까지". (v8)"""
    r = verify_answer("2026-07-25부터 제출 가능합니다", ["제출 기한은 2026-07-25까지"])
    assert not r.ok and len(r.direction_conflicts) == 1


# ---------------------------------------------------------------------------
# v9 — 단위 표기 변형('주'+어미 직결, '달', '퍼센트')의 수집
# ---------------------------------------------------------------------------
def test_week_unit_collected_with_hangul_suffix():
    """'주' lookahead 의 전면 한글 배제가 "2주간"·"2주입니다"·"2주로" 표기의
    수집을 통째로 차단했다(v9) — 그 표기로만 오는 "15일 → 약 2주간" 환산
    위조는 조용히 통과하고(미탐), 근거가 "2주간"이면 옳은 답변 "2주"에
    미확인 오탐이 붙는 양방향 결함. \b 가 한글 직결에서 경계가 아니던 v6
    교훈이 검증기 자신의 단위 표기에는 미전파된 형태였다."""
    r = verify_answer("보관 기간은 약 2주간입니다", ["보관 기간은 15일이다"])
    assert not r.ok and "2주" in r.unsupported
    # 역방향(오탐 감시): 근거 쪽 표기가 "2주간"이어도 답변 "2주"는 지지된다
    r = verify_answer("안정성 시험은 2주 이내에 완료합니다",
                      ["안정성 시험은 2주간 이내에 완료한다"])
    assert r.ok, r.summary()


def test_week_compound_words_not_collected_as_unit():
    """'주년·주기·주차' 등 비단위 합성어는 여전히 수집하지 않는다 — lookahead
    를 좁힌 대가로 관용 합성어가 기간 클레임이 되면 오탐 경로가 열린다."""
    nums, _ = extract_claims("창립 10주년 행사와 사업 2주기 평가, 3주차 교육")
    assert not any(u == "주" for _, u in nums)


def test_native_month_unit_dal_normalized():
    """'6달'은 '6개월'의 표기 변형(환산 아님) — 종전에는 아예 수집되지 않아
    "60일 → 6달" 위조가 조용히 통과했다. '달러'는 단위가 아니다."""
    r = verify_answer("갱신 신청은 6달 이내에 해야 합니다", ["갱신 신청은 60일 이내에 한다"])
    assert not r.ok and "6개월" in r.unsupported
    r = verify_answer("갱신 주기는 6달입니다", ["갱신 주기는 6개월이다"])
    assert r.ok, r.summary()
    nums, _ = extract_claims("수수료는 100달러입니다")
    assert not any(u == "개월" for _, u in nums)


def test_percent_spelled_out_normalized():
    """'90 퍼센트'는 '90%'의 표기 변형 — 철자 표기만 미수집이면 그 표기로 오는
    수치 위조가 축을 우회한다."""
    r = verify_answer("적합률은 90 퍼센트입니다", ["적합률은 90%이다"])
    assert r.ok, r.summary()
    r = verify_answer("적합률은 95 퍼센트입니다", ["적합률은 90%이다"])
    assert not r.ok and "95%" in r.unsupported


# ---------------------------------------------------------------------------
# v9 — 방향·역할 축의 전제(from_question) 라벨: 정정 답변의 오탐 종류 조정
# ---------------------------------------------------------------------------
def test_direction_conflict_from_question_labeled():
    """질문의 틀린 방향 전제를 정정하는 답변("이후가 아니라 이내")은 전제를
    재서술할 수밖에 없다 — 경고는 유지하되(정정인지 왜곡인지 기계는 모른다)
    from_question 라벨과 '전제 확인' 문구로 종류가 조정되어야 한다. 완화
    라벨이 존재 축에만 배선되어 옳은 정정에 '컴플라이언스 오류' 단정이 붙던
    축 × 라벨 매트릭스의 빈 칸(v9)."""
    r = verify_answer(
        "아니요, 15일 이후가 아니라 인지일로부터 15일 이내에 보고해야 합니다",
        ["인지일로부터 15일 이내 신속보고"],
        question="신속보고는 15일 이후에 하면 되나요?",
    )
    check = next(c for c in r.checks if c.kind == "direction")
    assert not r.ok and check.from_question       # 경고 유지 + 라벨
    # F6(v10): 방향 클레임은 question_origin(수치·날짜 전제 축)으로 새지 않는다 —
    # 방향의 from_question 완화는 방향 블록이 직접 처리하므로, 이 축에 섞이면
    # '⚠ 전제 확인 필요: 수치 …' 블록으로 오분류·이중 경고된다.
    assert "15일 이후" not in r.question_origin
    wt = warning_text(r)
    assert "전제" in wt and "방향" in wt            # 방향 블록이 전제 문구로 조정
    assert "전제 확인 필요: 수치" not in wt          # 수치 전제 블록으로 오분류 안 됨
    # 질문에 그 방향이 없으면 종전대로 무조건 경고 문구
    r2 = verify_answer("15일 이후에 보고하면 됩니다", ["인지일로부터 15일 이내 신속보고"])
    check2 = next(c for c in r2.checks if c.kind == "direction")
    assert not check2.from_question and "컴플라이언스 오류" in warning_text(r2)


def test_role_conflict_from_question_labeled():
    """역할 축의 전제 라벨 — 질문이 "기한이 <인지일> 맞나요?"라고 물었을 때
    그 역할-날짜 조합을 재서술하며 정정하는 답변에는 from_question 이 붙는다."""
    r = verify_answer(
        "기한은 2026-07-10 이 아니라 2026-07-25 입니다",
        ['{"awareness_date": "2026-07-10", "deadline_date": "2026-07-25"}'],
        question="보고 기한이 2026-07-10 맞나요?",
    )
    check = next(c for c in r.checks if c.kind == "role")
    assert check.from_question and "전제" in warning_text(r)


def test_direction_from_case_included_in_case_origin():
    """방향 충돌의 from_case 라벨이 case_origin(summary·계기판 case_labeled·감사
    로그의 재료)에 포함된다 — 종전에는 supported=True 를 요구해 방향축 라벨이
    어디에도 집계되지 않았다(라벨은 죽어도 소리가 없다는 원칙의 빈틈, v9)."""
    r = verify_answer(
        "15일 이후에 보고하면 됩니다",
        ["인지일로부터 15일 이내 신속보고"],
        user_fact_texts=["환자가 복용 15일 이후 증상 발생"],
    )
    assert "15일 이후" in r.case_origin
    assert "15일 이후" in r.summary()["case_origin"]


def test_question_origin_is_numeric_date_only_not_direction_role():
    """question_origin(전제 에코 축)은 numeric·date 로 한정된다 — 방향·역할의
    from_question 완화는 각 축 블록이 직접 처리하므로, 이 축에 섞이면 '⚠ 전제
    확인 필요: 수치 …' 로 오분류·이중 경고되고 계기판 by_axis 도 이중 집계된다. (v10)"""
    # 방향 클레임(from_question) — 방향 축에서 완화되고 question_origin 엔 안 뜬다
    rd = verify_answer(
        "아니요, 15일 이후가 아니라 15일 이내입니다",
        ["인지일로부터 15일 이내 신속보고"],
        question="신속보고는 15일 이후에 하면 되나요?",
    )
    assert rd.question_origin == []
    wt = warning_text(rd)
    assert "방향" in wt and "전제" in wt          # 방향 블록이 전제 문구로 조정
    assert "전제 확인 필요: 수치" not in wt        # 수치 전제 블록으로 오분류 안 됨
    # 수치 클레임(from_question)은 여전히 question_origin 에 잡힌다(회귀 방지)
    rn = verify_answer(
        "30일이 아니라 15일 이내입니다",
        ["인지일로부터 15일 이내"],
        question="보고 기한이 30일 맞나요?",
    )
    assert "30일" in rn.question_origin


def test_ok_derived_from_blocking_axes_subset_of_warning_axes():
    """ok 는 BLOCKING_AXES 정본에서 파생하고, 그 집합은 WARNING_AXES 의
    부분집합이다 — 새 차단 축을 ok 에 배선하면서 계기판·자가 테스트(WARNING_AXES)
    등록을 빠뜨리면 warn_rate 만 오르고 by_axis 귀속 없는 축이 배포되는 결합을
    구조로 막는다(7-7 패턴3 정본 앵커가 봉합하지 않은 마지막 결합, v10)."""
    assert set(BLOCKING_AXES) <= set(WARNING_AXES)
    # 각 차단 축이 단독으로 ok 를 False 로 만든다(파생의 정합) — superseded_cited
    # 까지 4축 전부(v11: 종전 메타테스트는 3축만 돌아 4번째 차단 축의 파생이
    # 강제되지 않았다).
    axis_cases = {
        "unsupported": VerificationResult(checks=[ClaimCheck(claim="x", kind="numeric", supported=False)]),
        "direction_conflicts": VerificationResult(checks=[ClaimCheck(claim="x", kind="direction", supported=False)]),
        "role_conflicts": VerificationResult(checks=[ClaimCheck(claim="x", kind="role", supported=False)]),
        "superseded_cited": VerificationResult(superseded_cited=["REG-013"]),
    }
    assert set(axis_cases) == set(BLOCKING_AXES)          # 4축 전수 — 누락 시 실패
    for axis, v in axis_cases.items():
        assert not v.ok and getattr(v, axis)
    # ok 가 정확히 `not any(BLOCKING)` 임을 고정 — 빈 result 는 다른 프로퍼티 값과
    # 무관하게 ok=True, 각 차단 축 단독은 ok=False(배타성). 손코딩으로 BLOCKING
    # 밖 조건을 ok 에 끼워넣는 우회를 잡는다.
    assert VerificationResult().ok is True
    for v in axis_cases.values():
        assert v.ok == (not any(getattr(v, a) for a in BLOCKING_AXES))
    # 비차단 경고 축(question_origin)만으로는 ok 를 깨지 않는다 — 전제 라벨은
    # 경고 '문구'만 바꾸는 축이라 WARNING_AXES 이되 BLOCKING_AXES 는 아니다
    assert "question_origin" in WARNING_AXES
    assert "question_origin" not in BLOCKING_AXES
