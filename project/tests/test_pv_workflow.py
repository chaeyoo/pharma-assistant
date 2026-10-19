"""PV 워크플로 심화 테스트 — 인과성(WHO-UMC) · 용어 코딩 · ICSR 보고서 초안.

트리아지 이후의 PV 업무(인과성 평가 → 코딩 → 보고서 작성)가
규칙 기반으로 재현 가능하게 동작하는지 보증한다.
"""
from __future__ import annotations

from src.agent.agent import RaAgent, _route_intent
# 서버 내부 단위 테스트 — 도구 함수를 직접 호출한다(HTTP 클라이언트 경로는 test_mcp/test_agent 가 커버)
from src.mcp_server.server import assess_adverse_event, draft_ae_report
from src.pv.causality import (
    CERTAIN,
    POSSIBLE,
    PROBABLE,
    UNASSESSABLE,
    UNLIKELY,
    assess_causality,
)
from src.pv.coding import code_terms, suggest_candidates
from src.pv.report import build_report
from src.pv.triage import assess_case


# ---------------------------------------------------------------------------
# 인과성 평가 (WHO-UMC 제안)
# ---------------------------------------------------------------------------
def test_causality_certain_needs_all_signals():
    """Certain 은 대체 원인의 '명시 배제'까지 확인될 때만 제안된다. (v8 강화)

    WHO-UMC Certain 은 "다른 약물·질환으로 설명되지 않음"의 확인을 요구한다 —
    대체원인 '미언급'을 배제로 치면, 정보가 없는 요소가 등급을 올리는
    docstring 원칙 위반이 대체원인 축에만 남는다."""
    c = assess_causality("복용 후 두드러기 발생, 중단 후 호전, 재투여 후 재발. 병용약물은 없었음")
    assert c.suggested == CERTAIN


def test_causality_certain_withheld_without_alternative_exclusion():
    """대체 원인이 미언급이면 3신호가 다 있어도 Certain 을 보류하고 Probable. (v8)"""
    c = assess_causality("복용 후 두드러기 발생, 중단 후 호전, 재투여 후 재발")
    assert c.suggested == PROBABLE
    assert any("대체 원인" in q for q in c.missing_info)  # 되물을 질문으로 안내


def test_causality_negative_rechallenge_is_not_positive():
    """재투여 후 '재발하지 않음'은 인과성의 반증이다 — 양성 오탐 금지. (v8)

    이전에는 "다시 복용하니" 마커가 경과를 안 봐서, 미재발 케이스가
    rechallenge 양성 → Certain 으로 제안되고 rationale 문장까지 사실과
    어긋났다(오판 방향이 최고 등급 쪽인 가장 위험한 형태)."""
    c = assess_causality("복용 후 두통, 중단 후 호전, 다시 복용하니 아무 증상이 없었다")
    assert not c.signals["재투여 후 재발(rechallenge)"]
    assert c.suggested == POSSIBLE
    assert "재발하지 않아" in c.rationale


def test_causality_negative_dechallenge_is_not_positive():
    """'중단하니 오히려 악화'는 positive dechallenge 가 아니다. (v8)"""
    c = assess_causality("복용 후 두통이 생겼고 중단하니 오히려 악화되었다")
    assert not c.signals["중단 후 호전(dechallenge)"]
    assert c.suggested == POSSIBLE


def test_causality_adjacent_clause_negation_does_not_flip_outcome():
    """다음 절의 무관한 부정("회복, 이상없음"의 '없음')이 경과 어휘의 부정으로
    삼켜지지 않는다 — 양성 dechallenge/rechallenge 를 반증으로 뒤집는 오탐 금지. (v10)

    v9 의 부정 어미 일반 규칙(_NEGATION_SPAN=6)이 쉼표로 이어진 짧은 후속 절의
    '다른 증상에 대한 부정'까지 창에 넣어, "중단하니 회복, 이상없음"이
    not_improved 로 뒤집히고(v8 대비 회귀) rationale 이 케이스("회복")와 정면
    모순됐다. 절 경계(쉼표·마침표)에서 창을 끊어 봉합."""
    c = assess_causality("아스피린 복용 후 두드러기. 중단하니 회복, 이상없음.")
    assert c.signals["중단 후 호전(dechallenge)"]        # 회복 = 양성 dechallenge
    c2 = assess_causality("복용 후 두통. 재투여하니 재발, 발열 없음.")
    assert c2.signals["재투여 후 재발(rechallenge)"]      # 재발 = 양성 rechallenge
    # 같은 절 부정("호전되지 않았다")은 여전히 부정으로 잡힌다(과교정 방지)
    c3 = assess_causality("복용 후 두통, 중단해도 호전되지 않았다")
    assert not c3.signals["중단 후 호전(dechallenge)"]


def test_causality_negated_alternative_is_not_present():
    """"병용약물은 없었다" 부정문이 대체원인 '있음'으로 오탐되지 않는다. (v8)

    이전에는 '병용' 부분 매칭으로 대체원인 존재가 되어 등급이 아래로 밀리고
    rationale("대체 원인 가능성이 있다")도 사실과 반대였다."""
    c = assess_causality("투여 후 발진, 중단 후 호전. 병용약물은 없었다")
    assert not c.signals["대체 원인 가능성(병용약·기저질환)"]
    assert c.suggested == PROBABLE
    assert not any("대체 원인" in q for q in c.missing_info)  # 정보가 있으므로 되묻지 않는다


def test_causality_probable_without_rechallenge():
    c = assess_causality("복용 후 두드러기 발생, 중단 후 호전됨")
    assert c.suggested == PROBABLE
    # 재투여 정보가 없으므로 되물을 질문에 포함된다
    assert any("재투여" in q for q in c.missing_info)


def test_causality_alternative_cause_caps_at_possible():
    """대체 원인(병용약)이 있으면 dechallenge 가 있어도 Possible 이상 못 올라간다."""
    c = assess_causality("복용 후 발진, 중단 후 호전. 다만 항생제를 병용 중이었음")
    assert c.suggested == POSSIBLE
    assert c.signals["대체 원인 가능성(병용약·기저질환)"]


def test_causality_unlikely_and_unassessable():
    assert assess_causality("기저질환 악화로 입원").suggested == UNLIKELY
    assert assess_causality("두통이 있었다").suggested == UNASSESSABLE


def test_causality_is_suggestion_not_verdict():
    """정보가 없는 요소는 충족으로 치지 않는다(등급은 보수적으로만)."""
    c = assess_causality("복용 후 구토")
    assert c.suggested == POSSIBLE  # 시간관계만으로 Probable 로 올려주지 않는다
    assert len(c.missing_info) == 3  # dechallenge·rechallenge·대체원인 전부 확인 필요


# ---------------------------------------------------------------------------
# 표준 용어 코딩 (MedDRA 방식)
# ---------------------------------------------------------------------------
def test_coding_maps_colloquial_to_pt():
    coded = code_terms("복용 후 숨쉬기 힘들고 온몸에 두드러기가 났다")
    pts = {t.pt for t in coded}
    assert {"호흡곤란", "두드러기"} <= pts
    dysp = next(t for t in coded if t.pt == "호흡곤란")
    assert dysp.pt_en == "Dyspnoea" and "호흡기" in dysp.soc


def test_coding_dedupes_same_pt():
    """같은 PT 를 가리키는 표현이 여러 번 나와도 1건으로 집계한다."""
    coded = code_terms("어지러움과 현기증, 어지럼 증상")
    assert len([t for t in coded if t.pt == "어지러움"]) == 1


def test_coding_preserves_narrative_order():
    coded = code_terms("두통이 먼저 왔고 이후 구토를 했다")
    assert [t.pt for t in coded] == ["두통", "구토"]


def test_coding_no_match_returns_empty():
    assert code_terms("품목허가 심사 기간은?") == []


# ---------------------------------------------------------------------------
# ICSR 보고서 초안 (최소보고요건 + 조립)
# ---------------------------------------------------------------------------
def test_report_complete_case_is_reportable():
    r = build_report(
        "45세 남성 환자가 A정 복용 후 아나필락시스로 입원",
        reporter="의사", awareness_date="2026-07-01",
    )
    assert r.reportable and not r.missing
    assert r.triage.is_serious and r.triage.deadline_days == 0
    assert "최소보고요건 충족" in r.draft_markdown


def test_report_detects_missing_minimum_criteria():
    """의심약·보고자가 없으면 reportable=False + 보완 항목 안내(ICH E2D 4요소)."""
    r = build_report("복용 후 두드러기가 생겨 입원")
    assert not r.reportable
    assert any("환자" in m for m in r.missing)
    assert any("보고자" in m for m in r.missing)
    assert any("의심 의약품" in m for m in r.missing)
    assert any("최소보고요건 보완" in f for f in r.followups)


def test_report_draft_contains_full_pv_workflow():
    r = build_report(
        "환자가 B캡슐 복용 후 두드러기 발생으로 입원, 중단 후 호전",
        reporter="약사", awareness_date="2026-07-01",
    )
    d = r.draft_markdown
    assert "중대성" in d and "2026-07-16" in d          # 트리아지(15일 기한)
    assert PROBABLE in d                                  # 인과성 제안
    assert "Urticaria" in d                               # 용어 코딩
    assert "PV 담당자" in d                               # 사람 확정 caveat 강제


# ---------------------------------------------------------------------------
# MCP 도구 계약
# ---------------------------------------------------------------------------
def test_draft_tool_contract_and_basis():
    out = draft_ae_report(
        "환자가 복용 후 아나필락시스로 입원", reporter="의사", suspected_drug="A정",
        awareness_date="2026-07-01",
    )
    for key in ["reportable", "missing", "followups", "draft_markdown",
                "causality", "coded_terms", "pii_masked", "basis"]:
        assert key in out
    assert "REG-005" in [r["doc_id"] for r in out["basis"]["results"]]


def test_draft_tool_masks_pii_before_drafting():
    out = draft_ae_report("환자 김철수님(010-1234-5678)이 C정 복용 후 사망", reporter="의사")
    assert "김철수" not in out["draft_markdown"] and "1234" not in out["draft_markdown"]
    assert out["pii_masked"]
    # 마스킹 후에도 '환자 존재 신호'([이름]님)는 남아 최소요건 판정이 가능하다
    assert not any("환자" in m for m in out["missing"])


def test_draft_tool_masks_pii_in_all_freetext_args():
    """case_description 만 마스킹하면 reporter·patient_info 인자가 PII 우회로가
    된다 — stdio 단독 사용 시 인자는 입구 마스킹을 거치지 않고 직접 들어온다."""
    out = draft_ae_report(
        "환자가 C정 복용 후 사망",
        reporter="담당 약사 홍길동님 (010-9999-8888)",
        patient_info="45세 남성, 차트번호: A-1234",
        awareness_date="2026-07-01",
    )
    dump = str(out)  # 초안·필드·팔로업 어디에도 원 값이 남으면 안 된다
    assert "홍길동" not in dump and "010-9999-8888" not in dump and "A-1234" not in dump
    # 마스킹 리포트는 전 필드 합산으로 하나만 보고된다
    kinds = {m["type"] for m in out["pii_masked"]}
    assert {"이름(호칭)", "전화번호", "환자/차트번호"} <= kinds
    # 마스킹 후에도 존재 신호가 남아 최소보고요건(환자·보고자) 판정은 유지된다
    assert out["reportable"] is True


def test_assess_tool_now_includes_causality_and_coding():
    out = assess_adverse_event("복용 후 두드러기로 입원, 중단 후 호전", awareness_date="2026-07-01")
    assert out["causality"]["suggested"] == PROBABLE
    assert any(t["pt"] == "두드러기" for t in out["coded_terms"])


# ---------------------------------------------------------------------------
# 에이전트 라우팅 + 오프라인 E2E
# ---------------------------------------------------------------------------
def test_route_report_request_to_draft_tool():
    assert _route_intent("환자가 복용 후 아나필락시스로 입원했습니다. KAERS 보고서 초안 작성해줘") == "ae_report"


def test_route_case_without_report_request_stays_triage():
    assert _route_intent("환자가 복용 후 아나필락시스로 입원했습니다. 언제까지 보고해야 하나요?") == "ae_triage"


async def test_agent_report_end_to_end():
    agent = RaAgent()
    r = await agent.chat(
        "환자 박영희님(010-1111-2222)이 D정 복용 후 두드러기로 입원했습니다. 보고서 초안 만들어줘"
    )
    assert [t.name for t in r.tool_calls] == ["draft_ae_report"]
    assert "개별사례보고(ICSR) 초안" in r.answer and "Urticaria" in r.answer
    assert r.grounded and any(c["doc_id"] == "REG-005" for c in r.citations)
    # 입구 마스킹: 초안·트레이스 어디에도 원 PII가 없다
    assert "박영희" not in r.answer and "1111" not in str(r.trace)


# ---------------------------------------------------------------------------
# 코딩 2·3계층 — 후보(LLT 참조) 제시와 미코딩 감지, 보고요건 연쇄 차단
# ---------------------------------------------------------------------------
def test_candidate_tier_suggests_but_never_confirms():
    """1계층 미수록 표현('청력 상실')은 후보로만 제시된다 — 확정 목록에
    섞이면 시그널 집계가 오염되므로 타입/필드 수준에서 분리를 보증한다."""
    from src.pv.coding import suggest_candidates

    case = "환자가 항결핵제O정을 복용 후 청력 상실이 발생했습니다."
    coded = code_terms(case)
    cands = suggest_candidates(case, coded)
    assert all(t.pt != "난청" for t in coded), "후보가 확정에 섞이면 안 된다"
    assert [c.pt for c in cands] == ["난청"]
    assert cands[0].needs_confirmation


def test_candidate_tier_skips_already_confirmed_pt():
    """확정 사전이 이미 잡은 PT는 후보로 중복 제시하지 않는다(이중 집계 방지)."""
    from src.pv.coding import suggest_candidates

    case = "복용 후 오심과 함께 속이 울렁거린다고 합니다."  # 오심은 1계층 수록
    coded = code_terms(case)
    assert any(t.pt == "오심" for t in coded)
    assert suggest_candidates(case, coded) == []


def test_uncoded_flag_detects_specific_symptom_only():
    """3계층은 '구체적 증상 서술'만 감지한다 — 막연한 서술("몸이 좋지 않다")은
    ICH E2D의 specificity 요구상 ④요건 미충족이 올바른 판정이라 잡지 않는다."""
    from src.pv.coding import flag_uncoded_expressions

    hit = flag_uncoded_expressions("복용 후 손발이 저릿저릿하다고 호소", [], [])
    assert hit and "저릿" in hit[0]
    assert flag_uncoded_expressions("지인이 약을 먹고 몸이 좋지 않다고 합니다", [], []) == []


def test_report_candidate_satisfies_event_criterion():
    """④요건의 본질은 '구체적 이상사례 서술의 존재'다 — 1계층 코딩이 실패해도
    후보(2계층)가 있으면 reportable 이고, 확정은 follow-up 으로 요청된다."""
    r = build_report(
        "환자가 위장약M정을 복용 후 속이 울렁거리고 힘들다고 호소했습니다. 약사가 보고했습니다."
    )
    assert r.coded_terms == [] and [c.pt for c in r.candidate_terms] == ["오심"]
    assert r.reportable, r.missing
    assert any("후보 승인/기각" in f for f in r.followups)
    assert "후보(승인/기각 필요)" in r.draft_markdown


def test_report_uncoded_signal_satisfies_event_criterion():
    """어느 사전에도 없는 심층 롱테일도 '증상 서술 감지'로 ④요건은 충족 —
    코딩 실패가 '보고 불가' 오판으로 연쇄되지 않는다(PT 부여는 사람 몫)."""
    r = build_report(
        "환자가 진통제R정을 복용 후 손발이 저릿저릿하다고 호소했습니다. 의사가 보고했습니다."
    )
    assert r.coded_terms == [] and r.candidate_terms == []
    assert r.uncoded_expressions and r.reportable
    assert any("PT 부여 필요" in f for f in r.followups)


def test_report_vague_case_still_fails_event_criterion():
    """보수성 유지 가드: 막연한 서술만 있는 케이스는 여전히 ④요건 미충족 —
    후보/감지 계층이 '아무거나 통과'로 변질되지 않았음을 고정한다."""
    r = build_report("지인이 약을 먹고 몸이 좋지 않다고 합니다.")
    assert not r.reportable
    assert any("④" in m for m in r.missing)


def test_tool_exposes_candidate_and_uncoded_layers():
    """MCP 도구 계약: 확정/후보/미코딩 3계층이 응답 스키마로 구분 노출된다."""
    out = draft_ae_report(
        "환자가 당뇨약C정을 복용 후 심한 저혈당으로 입원했습니다. 약사가 보고했습니다."
    )
    assert out["coded_terms"] == []
    assert [c["pt"] for c in out["candidate_terms"]] == ["저혈당"]
    assert all(c["needs_confirmation"] for c in out["candidate_terms"])


# ---------------------------------------------------------------------------
# v8 — 트리아지·보고요건·코딩의 오탐/오매핑 봉합
# ---------------------------------------------------------------------------
def test_triage_disorder_word_is_not_disability():
    """"위장 장애"·"수면장애"의 '장애'(disorder)는 중대성 기준(disability)이 아니다. (v8)

    단독 키워드 매칭은 한국어 의무기록에서 빈도 높은 동음이의를 전부
    15일 신속보고로 밀어 올렸다 — disability 의미가 확정되는 결합형만 잡는다."""
    from src.pv.triage import assess_case
    assert not assess_case("복용 후 경미한 위장 장애가 있었다").is_serious
    assert not assess_case("복용 후 수면장애를 호소했다").is_serious


def test_triage_real_disability_still_detected():
    """진짜 disability 서술(영구 장애·장애 판정·청력 상실)은 여전히 중대다. (v8)"""
    from src.pv.triage import assess_case
    assert assess_case("복용 후 청력 상실이 발생해 영구 장애 판정을 받았다").is_serious
    assert assess_case("투여 후 실명하였다").is_serious


def test_report_hospital_treatment_is_not_reporter():
    """"병원에서 치료받았다"는 보고자 정보가 아니다 — 조용한 통과 금지. (v8)

    보고자 없는 케이스가 reportable=True 로 통과하는 것은 이 모듈에서
    유일하게 실패 방향이 '조용한 쪽'인 결함이었다."""
    r = build_report("45세 여성이 진통제E정을 복용 후 두통으로 병원에서 치료받았다")
    assert not r.reportable
    assert any("보고자" in m for m in r.missing)


def test_report_hospital_reporting_still_counts():
    """보고 행위가 결합된 "병원에서 보고"는 여전히 보고자로 인정된다. (v8)"""
    r = build_report("45세 남성 환자가 항생제A정을 복용 후 두드러기 발생. 병원에서 보고된 케이스")
    assert not any("보고자" in m for m in r.missing)


def test_report_reporter_not_fired_by_embedded_compound(monkeypatch=None):
    """'주의사항'·'제약사'의 '의사'/'약사'는 보고자가 아니다 — 조용한 통과 금지. (v10)

    v9 는 '의사(?!소통|…)' 룩어헤드로 접두 합성어만 막았으나, '의사'가 어절
    중간에 박힌 최빈 합성어(주(注)+의사(意事)+항=주의사항, 유의사항)와 '제약사'
    (제+약사)는 그대로 보고자로 발화해, 보고자 없는 케이스가 reportable=True 로
    통과했다(v9 봉합의 반대편 미탐). 어절 경계(앞이 한글이면 비매칭)로 봉합."""
    r = build_report("45세 남성 환자가 아스피린정을 복용 후 두드러기 발생. 복용 주의사항을 안내받음.")
    assert not r.reportable
    assert any("보고자" in m for m in r.missing)
    # 진짜 보고자(어절 첫 음절의 '의사'/'약사')는 여전히 인정된다(과교정 방지)
    assert not any(
        "보고자" in m
        for m in build_report(
            "45세 남성 환자가 아스피린정을 복용 후 두드러기. 담당 의사가 신고함."
        ).missing
    )


def test_coding_jaundice_is_independent_pt():
    """황달은 간손상이 아니라 독립 PT(Jaundice)로 코딩된다. (v8)

    MedDRA 에서 황달(Jaundice)은 간담도 장애의 독립 PT — 용혈성·폐쇄성 황달은
    간손상이 아니므로, '검수 사전=자동 확정' 1계층의 오매핑은 집계를 오염시킨다."""
    coded = code_terms("복용 후 황달이 나타났다")
    assert [(t.pt, t.pt_en) for t in coded] == [("황달", "Jaundice")]
    coded2 = code_terms("복용 후 간독성 소견")
    assert [(t.pt, t.pt_en) for t in coded2] == [("간손상", "Liver injury")]


# ---------------------------------------------------------------------------
# v9 — 인과성 극성 판정(부정 어미 일반 규칙)·rationale 모순·보고요건 오탐 봉합
# ---------------------------------------------------------------------------
def test_causality_negated_improvement_is_not_positive_dechallenge():
    """"중단했지만 호전되지 않았다"는 positive dechallenge 가 아니다. (v9)

    "호전되지 않"이 부정 목록에 없어 IMPROVE 의 "호전"이 먼저 매칭됐다 —
    부정 활용은 열린 집합이라 열거 확장 대신 '긍정 어휘 매칭 직후 N자 이내
    부정 어미(않·없·아니)' 일반 규칙으로 판정하고, 부정은 판단 불가가 아니라
    반증(not_improved)으로 매핑한다."""
    for case in [
        "복용 후 발진이 생겼고 중단했지만 발진이 호전되지 않았다",
        "복용 후 발진, 중단했으나 회복되지 않았다",
        "복용 후 발진, 중단 후에도 사라지지 않았다",
        "복용 후 발진, 중단해도 좋아지지 않았다",
    ]:
        c = assess_causality(case)
        assert not c.signals["중단 후 호전(dechallenge)"], case
        assert c.suggested == POSSIBLE, case


def test_causality_compact_no_recurrence_blocks_certain():
    """"재투여 후 재발 없음"(조사 생략형)이 rechallenge 양성으로 뒤집히지 않는다. (v9)

    "재발 없"이 부정 목록에 없어 RECUR "재발"이 매칭 — 대체원인 명시 배제까지
    결합되면 반증 케이스가 최고 등급(Certain)을 지지하는 치명 오류였다.
    반증 신호는 Possible 상한으로 눌러야 한다."""
    c = assess_causality("복용 후 발진. 중단하니 호전. 재투여 후 재발 없음. 병용약물은 없었다")
    assert not c.signals["재투여 후 재발(rechallenge)"]
    assert c.suggested == POSSIBLE
    assert "재발하지 않아" in c.rationale


def test_causality_negated_dechallenge_marker_opens_no_window():
    """"약을 중단하지 않았는데도 호전"에서 dechallenge 창이 열리면 안 된다. (v9)

    맥락 마커("중단") 자체가 부정된 서술은 그 경과가 일어난 적이 없다 —
    부정된 마커로 창이 열리면 뒤따르는 "호전"이 존재하지 않는 중단 경과의
    양성 신호로 둔갑한다."""
    for case in [
        "복용 후 두통이 생겼고, 약을 중단하지 않았는데도 증상이 호전되었다",
        "복용 후 두통. 중단 없이 유지했는데 증상이 호전되었다",
    ]:
        c = assess_causality(case)
        assert not c.signals["중단 후 호전(dechallenge)"], case
        assert c.suggested == POSSIBLE, case
        # 중단 경과는 '미확인'이므로 되물을 질문에 남는다
        assert any("dechallenge" in q for q in c.missing_info), case


def test_causality_dechallenge_without_temporal_has_consistent_rationale():
    """중단 경과는 양성인데 시간 마커가 없을 때 rationale 이 신호와 모순되지 않는다. (v9)

    이전에는 Unassessable 로 떨어지며 "판단 요소가 감지되지 않는다"는 사유가
    붙었다 — 감지된 신호(중단 후 호전)와 정면 모순. WHO-UMC 근사의 보수
    원칙대로 Possible 에 머물되, 시간관계 확인을 질문으로 넘긴다."""
    c = assess_causality("발진이 있어 약을 중단하니 호전되었다")
    assert c.signals["중단 후 호전(dechallenge)"]
    assert c.suggested == POSSIBLE
    assert "감지되지 않는다" not in c.rationale
    assert "시간관계" in c.rationale and "확인" in c.rationale
    assert any("시간적 선후관계" in q for q in c.missing_info)


def test_report_intent_compound_is_not_reporter():
    """"의사소통"·"의사결정"·"약사법"의 '의사/약사'는 보고자가 아니다. (v9)

    substring 매칭은 intent(의사소통)·법령명(약사법)을 ②요건 충족으로 밀어,
    보고자 없는 케이스가 보완 요청 없이 조용히 통과했다 — 이 모듈의 실패
    방향은 '시끄러운 보완 요청'이어야 한다(v8 병원 마커와 같은 원칙)."""
    r = build_report("45세 남성 환자가 A정을 복용 후 발진 발생. 향후 치료는 가족과의 의사소통을 통해 결정")
    assert any("보고자" in m for m in r.missing)
    r2 = build_report("45세 여성 환자가 B정을 복용 후 두통. 약사법에 따른 절차로 의사결정이 필요")
    assert any("보고자" in m for m in r2.missing)
    # 진짜 직역("의사가 보고")은 여전히 인정된다(오탐 차단이 미탐을 만들면 안 된다)
    r3 = build_report("45세 남성 환자가 A정을 복용 후 발진 발생. 담당 의사가 보고했습니다")
    assert not any("보고자" in m for m in r3.missing)


def test_report_generation_notation_is_not_patient_age():
    """"3세대 세팔로스포린"의 "3세"는 환자 나이가 아니다. (v9)

    나이 정규식(\\d{1,3}세)이 세대(generation) 표기에 부분 매칭해, 환자 정보
    없는 케이스의 ①요건이 조용히 충족됐다 — '세' 뒤 '대'를 후방 경계로 배제."""
    r = build_report("3세대 세팔로스포린 항생제 투여 후 발진 발생. 의사가 보고했습니다")
    assert any("환자" in m for m in r.missing)
    # 진짜 나이 표기는 여전히 ①신호다
    r2 = build_report("3세 남아가 시럽 복용 후 발진 발생. 의사가 보고했습니다")
    assert not any("환자" in m for m in r2.missing)


def test_report_quantity_token_is_not_suspected_drug():
    """"아스피린 1정을 복용"의 "1정"(수량)만으로 ③요건이 충족되면 안 된다. (v9)

    수량 토큰은 제품명이 아니다 — 제품명·성분 없는 케이스는 미충족(시끄러운
    보완 요청)이 올바른 판정이라, 숫자로만 시작하는 토큰은 배제한다."""
    r = build_report("환자가 아스피린 1정을 복용 후 두통 발생. 의사가 보고했습니다")
    assert any("의심 의약품" in m for m in r.missing)
    # 제품명 패턴("타이레놀정을 복용")은 여전히 감지된다
    r2 = build_report("환자가 타이레놀정을 복용 후 두통 발생. 의사가 보고했습니다")
    assert not any("의심 의약품" in m for m in r2.missing)


def test_report_reporter_name_token_is_not_patient_signal():
    """보고자 직역 바로 뒤의 [이름]은 환자 요건 ①의 신호가 아니다. (v9)

    마스킹 토큰 [이름]은 역할을 구분하지 않아, "담당 약사 [이름]님"의 보고자
    성명이 환자 존재 신호로 전용됐다 — 직역+공백 0~1자 뒤에 직결된 [이름]만
    좁게 제외해 독립 [이름](PV-009)의 ①신호는 유지한다(보수적 구현)."""
    r = build_report("담당 약사 [이름]님이 발진 사례를 보고했습니다")
    assert any("환자" in m for m in r.missing)          # ① 미충족(보완 요청)
    assert not any("보고자" in m for m in r.missing)    # ② 는 직역으로 충족
    # 직역에 붙지 않은 독립 [이름]은 여전히 환자 신호다(PV-009 라벨과 동일 방향)
    r2 = build_report("[이름]님이 B정 복용 후 발진이 생겼다고 간호사가 보고했습니다")
    assert not any("환자" in m for m in r2.missing)


def test_coding_visceral_cramp_is_not_seizure():
    """"위경련"·"근육경련"의 '경련'은 Seizure(신경계 발작)로 확정되지 않는다. (v9)

    1계층은 자동 확정 계층이라 부분 매칭 오매핑이 곧 시그널 집계의 구조적
    오염이다(황달≠간손상과 같은 원칙) — 전방 합성어(위·근육)를 배제한다."""
    assert code_terms("복용 후 위경련이 있었다") == []
    assert code_terms("복용 후 근육경련 증상을 보였다") == []
    # 진짜 신경계 경련·발작은 여전히 확정된다(배제가 미탐을 만들면 안 된다)
    coded = code_terms("복용 후 전신 경련과 발작이 발생했다")
    assert [(t.pt, t.pt_en) for t in coded] == [("경련", "Seizure")]


# ---------------------------------------------------------------------------
# v11 회귀 — v10 봉합의 반대편/열거 재확산 봉합
# ---------------------------------------------------------------------------
def test_triage_death_markers_recall_first():
    """사망은 미탐이 치명적 — 최빈 사망 표현(돌연사·급사·별세·운명·숨을 거두다)도
    중대로 잡는다(v11). 좁은 닫힌 목록이 원외 사망을 정기보고로 흘리던 갭."""
    for txt in ["환자가 숨을 거두었다", "돌연사하였다", "급사함", "별세하였다", "운명하셨다"]:
        r = assess_case(txt, "")
        assert r.is_serious and "사망" in r.criteria_met, txt
    # 기존 표현 회귀 없음
    assert assess_case("사망하였다", "").is_serious


def test_report_child_marker_not_fired_by_remain_verb():
    """소아 명사 '남아'(男兒)가 동사 '남아 있다'(잔존)에 오발화하지 않는다(v11).

    "증상이 남아 있음"은 환자 식별정보가 아니다 — 이것이 ①요건을 조용히
    충족시켜 환자정보 없는 케이스가 reportable=True 로 통과하던 갭."""
    r = build_report("타이레놀정 복용 후 두드러기가 생겼고 통증이 남아 있음. 담당 의사가 보고함.")
    assert not r.reportable
    assert any(m.startswith("①") for m in r.missing)
    # 명사 용법은 여전히 환자 신호로 인정(과교정 방지)
    r2 = build_report("남아가 B정 복용 후 발진 발생. 어머니가 보고함.")
    assert not any(m.startswith("①") for m in r2.missing)


def test_coding_paroxysm_attack_is_candidate_not_confirmed():
    """다의어 '발작'은 1계층 자동확정에서 빠져, 질환명+발작(공황·협심증·천식)이
    경련(Seizure)으로 오확정되지 않는다 — 후보(2계층)로 사람이 확정(v11)."""
    for txt in ["공황발작이 있었다", "협심증 발작", "천식 발작"]:
        coded = code_terms(txt)
        assert "경련" not in {t.pt for t in coded}, txt
        cands = suggest_candidates(txt, coded)
        assert "경련" in {c.pt for c in cands}, txt   # 후보로는 제시(검수 큐 도달)
    # 정당한 경련은 여전히 확정
    assert "경련" in {t.pt for t in code_terms("전신 경련이 있었다")}
    assert "경련" not in {t.pt for t in code_terms("위경련이 있었다")}  # v9 봉합 유지


def test_report_hanuisa_hanyaksa_are_reporters():
    """한의사·한약사는 KAERS 유효 보고자 — F1(v10) 룩비하인드가 한글 직결이라
    놓치던 정당 보고자를 마커로 인정(v11, 경계 봉합의 반대편 미탐 봉합)."""
    for txt in ["한의사가 처방한 약 복용 후 환자에게 두드러기 발생했다고 보고",
                "한약사가 조제한 약 복용 후 45세 남성 환자가 어지럼증 신고"]:
        assert not any("보고자" in m for m in build_report(txt).missing), txt


def test_triage_unexpected_nominal_form_not_flipped():
    """'알려진 부작용이 아님'(명사형)이 expected 로 뒤집히지 않는다(v11) —
    v9 가 아니/아닌만 열거하고 종결 명사형 '아님'을 빠뜨린 갭."""
    assert assess_case("이 반응은 알려진 부작용이 아님", "").expectedness == "unexpected"
    assert assess_case("알려진 부작용이 아니다", "").expectedness == "unexpected"


def test_causality_boundary_is_clause_terminators_only():
    """절 경계는 절 종결자(쉼표·마침표…)만 — 어절 내부 표기(가운뎃점·말줄임표·
    줄바꿈)를 경계로 보면 v9 마커가드를 되뚫고 정당 부정을 놓쳐 인과성을
    과대평가한다(v11, 경계 봉합의 반대편 미탐)."""
    # v10 fix(쉼표 절 경계) 유지 — 양방향 회귀 방지
    assert assess_causality("중단하니 회복, 이상없음.").signals["중단 후 호전(dechallenge)"]
    # 어절 내부 줄바꿈/말줄임표/가운뎃점에서 부정을 놓치지 않는다
    assert not assess_causality("중단 후 호전\n되지 않음.").signals["중단 후 호전(dechallenge)"]
    assert not assess_causality("호전·회복되지 않음").signals["중단 후 호전(dechallenge)"]
    # v9 '부정된 마커 건너뛰기' 가드가 줄바꿈에도 성립(마커가 부정되면 창 안 열림)
    assert not assess_causality("중단하지\n않았는데도 호전됨.").signals["중단 후 호전(dechallenge)"]


# ---------------------------------------------------------------------------
# v12 회귀 — v11 봉합의 반대편(과탐/미탐) 재봉합
# ---------------------------------------------------------------------------
def test_triage_death_short_markers_no_substring_overfire():
    """짧은 사망 마커가 무관 행정·조세 어휘에 substring 오발화하지 않는다(v12).

    v11 recall-first 가 '급사'·'별세'·'운명하'를 bare 로 넣어 '공급사'·'지급사유'·
    '특별세'·'운명하에'가 serious=True·D+0(지체없이)로 오라우팅됐다 — 형태소
    확정형으로 좁혀 과탐을 막되, 진짜 사망은 잡는다(recall 유지)."""
    for txt in ["백신 공급사가 변경된 로트", "보험 지급사유 확인 중 경증 오심",
                "특별세액공제 관련 문의", "개별세대로 독거 중"]:
        assert not assess_case(txt, "").is_serious, txt
    # 진짜 사망(형태소 확정형·완곡어 보강)은 여전히 중대
    for txt in ["환자가 급사하였다", "향년 82세로 별세하셨다", "소천하셨다",
                "타계하셨습니다", "서거함", "숨을 거두었다"]:
        r = assess_case(txt, "")
        assert r.is_serious and "사망" in r.criteria_met, txt


def test_report_hanuisa_institution_not_reporter():
    """기관명 '대한한의사협회'·'한약사회'는 보고자가 아니다 — 조용한 통과 금지(v12).

    v11 이 한의사·한약사를 substring 마커로 둬 기관명에 ② 오발화했다. 의사/약사에
    이미 쓴 어절 경계 룩비하인드를 신규 마커에 전파해 대칭화."""
    for txt in ["대한한의사협회가 안전성 서한 발행. 50세 남성이 아스피린정 복용 후 두드러기.",
                "한약사회 회보에 게재된 사례. 50세 남성이 아스피린정 복용 후 발진."]:
        assert any("보고자" in m for m in build_report(txt).missing), txt
    # 진짜 보고자(한의사가/한약사가)는 여전히 인정
    assert not any("보고자" in m for m in build_report(
        "한의사가 처방한 약 복용 후 환자에게 두드러기 발생했다고 보고").missing)


def test_report_child_marker_positive_noun_boundary():
    """소아 '남아'는 명사 문맥(조사·환자 직후)에서만 ①신호 — 동사 남다 배제(v12).

    v11 의 부정 열거(있|계|서|났|나서)가 '약이 남아도'·종결 '남아' 를 놓쳐 조용한
    통과가 재발했다. 명사 용법 긍정 경계로 뒤집어 동사 활용 전체를 배제."""
    assert any(m.startswith("①") for m in build_report(
        "지난 약이 남아도 타이레놀정을 복용 후 두드러기. 의사가 보고.").missing)
    # 명사 용법(남아가/3세 남아)은 유지
    assert not any(m.startswith("①") for m in build_report(
        "3세 남아가 B정 복용 후 발진. 어머니가 보고함.").missing)


def test_causality_newline_boundary_is_continuation_aware():
    """줄바꿈은 절 종결자이되, 직후가 용언 연속이면 같은 절로 본다(v12).

    v11 이 \\n 을 경계에서 빼 '호전\\n되지 않음'(용언 랩)은 살렸으나 '회복\\n이상
    없음'(절 구분)에서 '없'이 회복을 부정해 rationale 이 사실과 반대로 단언됐다.
    연속 인지 경계로 양방향을 모두 봉합."""
    # 절 구분 줄바꿈: 다음 절 부정이 회복을 오염시키지 않는다(양성 dechallenge 유지)
    assert assess_causality("중단하니 회복\n이상없음").signals["중단 후 호전(dechallenge)"]
    assert assess_causality("중단하니 회복, 이상없음").signals["중단 후 호전(dechallenge)"]  # v10 유지
    # 용언 내부 줄바꿈: 부정은 여전히 잡는다(과대평가 방지 유지)
    assert not assess_causality("중단 후 호전\n되지 않음").signals["중단 후 호전(dechallenge)"]
    assert not assess_causality("중단하지\n않았는데도 호전됨").signals["중단 후 호전(dechallenge)"]
