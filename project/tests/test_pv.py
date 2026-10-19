"""PV(약물감시) 모듈 테스트 — AE 트리아지 · PII 비식별화 · 질의 확장."""
from __future__ import annotations

import pytest

from src.agent.agent import RaAgent, _route_intent
# 서버 내부 단위 테스트 — 도구 함수를 직접 호출한다(HTTP 클라이언트 경로는 test_mcp/test_agent 가 커버)
from src.mcp_server.server import assess_adverse_event
from src.pv.redactor import redact
from src.pv.triage import assess_case
from src.rag.synonyms import expand_query


# ---------------------------------------------------------------------------
# AE 트리아지 (규칙 기반 중대성 판정 + 기한 계산)
# ---------------------------------------------------------------------------
def test_triage_death_is_immediate():
    t = assess_case("환자가 투여 3일 후 사망", awareness_date="2026-07-08")
    assert t.is_serious and "사망" in t.criteria_met
    assert t.deadline_days == 0
    assert t.deadline_date == "2026-07-08"  # 지체 없이 = 인지일 당일


def test_triage_hospitalization_is_15_days():
    t = assess_case(
        "복용 후 두드러기가 생겨 입원했고, 허가사항에 없는 예상치 못한 반응",
        awareness_date="2026-07-01",
    )
    assert t.is_serious and t.expectedness == "unexpected"
    assert t.deadline_days == 15
    assert t.deadline_date == "2026-07-16"  # 인지일 + 15일


def test_triage_non_serious_goes_to_psur():
    t = assess_case("가벼운 두통이 있었으나 회복")
    assert not t.is_serious
    assert t.deadline_date is None
    assert "PSUR" in t.route


def test_triage_unknown_expectedness_is_conservative():
    """예상 여부를 모르면 '예상치 못한 사례'로 보수 적용(안전한 실패)."""
    t = assess_case("복용 후 입원", awareness_date="2026-07-01")
    assert t.expectedness == "unknown"
    assert t.deadline_days == 15  # 완화하지 않고 15일 트래킹
    assert any("보수" in c for c in t.caveats)


def test_triage_always_flags_human_confirmation():
    for case in ["환자가 사망", "가벼운 두통"]:
        t = assess_case(case)
        assert any("담당자가 확정" in c for c in t.caveats)


def test_triage_invalid_awareness_date_is_noisy():
    """잘못된 인지일 형식은 '조용히' 오늘로 폴백하지 않는다 — 기한이 틀린
    기준일로 계산됐다는 신호(caveat)가 반드시 남아야 한다(시끄러운 실패)."""
    t = assess_case("복용 후 입원", awareness_date="2026/07/01")
    assert t.deadline_date is not None  # 계산 자체는 폴백으로 계속된다
    assert any("YYYY-MM-DD" in c and "재계산" in c for c in t.caveats)
    # 비중대 케이스에도 같은 신호가 남는다(인지일은 판정과 무관하게 에코되므로)
    t2 = assess_case("가벼운 두통", awareness_date="7월 1일")
    assert any("YYYY-MM-DD" in c for c in t2.caveats)


def test_triage_valid_awareness_date_has_no_date_caveat():
    t = assess_case("복용 후 입원", awareness_date="2026-07-01")
    assert not any("YYYY-MM-DD" in c for c in t.caveats)


def test_triage_future_awareness_date_is_noisy():
    """형식은 유효해도 '미래'인 인지일은 오타 가능성이 높다 — 형식 오류에만
    caveat 를 달고 값 오류는 조용히 통과시키던 비대칭의 봉합. 계산은 입력값대로
    수행한다(자동 정정은 또 다른 조용한 폴백)."""
    import datetime as dt

    future = (dt.date.today() + dt.timedelta(days=400)).isoformat()
    t = assess_case("복용 후 입원", awareness_date=future)
    assert t.deadline_date is not None  # 계산은 입력값 기준으로 계속된다
    assert any("미래" in c and "확인" in c for c in t.caveats)
    # 과거·오늘 인지일에는 미래 caveat 가 붙지 않는다(오탐 방지)
    t2 = assess_case("복용 후 입원", awareness_date="2026-07-01")
    assert not any("미래" in c for c in t2.caveats)


# ---------------------------------------------------------------------------
# PII 비식별화
# ---------------------------------------------------------------------------
def test_redact_masks_common_pii():
    r = redact("환자 김철수님(750101-1234567, 010-1234-5678, kim@test.com) 입원")
    assert "750101" not in r.text and "010-1234" not in r.text and "kim@test.com" not in r.text
    assert "김철수" not in r.text
    assert {"주민등록번호", "전화번호", "이메일", "이름(호칭)"} <= set(r.counts)


def test_redact_masks_unhyphenated_rrn_and_dotted_phone():
    """구분자 변형(무하이픈 주민번호·점 구분 전화)도 마스킹한다 — 구분자 어휘가
    좁으면 그만큼이 PII 우회로다(마스킹은 과탐이 미탐보다 싼 안전장치)."""
    r = redact("환자 9001011234567, 연락처 010.1234.5678 로 회신")
    assert "9001011234567" not in r.text and "010.1234.5678" not in r.text
    assert r.counts.get("주민등록번호") == 1 and r.counts.get("전화번호") == 1


def test_redact_masks_pii_adjacent_to_hangul():
    """한글이 직결된 표기("…는 010-…로", "주민번호900101-…입니다")도 마스킹한다.

    정규식 \\b 는 한글도 \\w 로 취급하므로 조사·명사가 숫자에 직결된(한국어에서
    가장 흔한) 표기에서는 경계가 성립하지 않아 매칭 전체가 빠진다 — 검증기의
    날짜 추출(verifier._DATE_RE)에서 이미 배운 교훈이 마스킹 계층에는 적용되지
    않았던 비대칭(대칭성 감사에서 발견). 검증기의 비검출은 경고 누락이지만
    마스킹의 비검출은 곧 유출이다."""
    r = redact("연락처는 010-1234-5678로 부탁드립니다. 주민번호900101-1234567입니다.")
    assert "010-1234-5678" not in r.text and "900101-1234567" not in r.text
    assert r.counts.get("전화번호") == 1 and r.counts.get("주민등록번호") == 1


def test_redact_email_does_not_swallow_hangul():
    """이메일 마스킹이 뒤따르는 한글("…com입니다")까지 삼키지 않는다.

    [\\w.+-] 는 한글도 매칭해 마스킹 범위가 원문보다 넓어진다 — 값은 가려지니
    유출은 아니지만, 마스킹 리포트가 원문 범위를 왜곡하면 감사가 어려워진다
    (과잉 매칭도 경계 결함으로 센다 — 오탐을 결함으로 세는 검증기와 같은 철학)."""
    r = redact("메일은 hong@example.com입니다")
    assert "hong@example.com" not in r.text
    assert r.text == "메일은 [이메일]입니다"
    assert r.counts.get("이메일") == 1


def test_redact_report_never_contains_original_values():
    r = redact("연락처 010-9876-5432")
    assert "9876" not in str(r.summary())  # 요약에는 유형·건수만


def test_redact_does_not_mask_role_nouns_with_honorific():
    """호칭(님/씨)이 붙은 일반 호칭·역할어("아저씨"·"보호자님")는 이름이 아니다. (v10)

    v9 는 이름 정규식의 조사 룩어헤드를 제거(유출 방향 봉합)하고 오탐은
    스톱리스트로 방어하는데, '보호자'·'아저'가 목록에 없어 [이름]으로
    과마스킹됐다 — 특히 '보호자'는 보고자 역할 신호를 지워 최소보고요건 ②
    미탐으로 연쇄된다(교수님 과마스킹과 같은 클래스의 미해결 인스턴스)."""
    assert redact("아저씨가 이상반응을 신고했다").text == "아저씨가 이상반응을 신고했다"
    assert redact("보호자님께서 대신 신고하셨다").text == "보호자님께서 대신 신고하셨다"
    # 진짜 이름은 여전히 마스킹된다(유출 방향 회귀 방지)
    assert "[이름]" in redact("홍길동님이 부작용을 호소했다").text
    # '보호자'가 살아 있어 보고자 역할 신호가 downstream 에 남는다
    from src.pv.report import build_report
    r = build_report("보호자님께서 45세 남아 환자의 두드러기를 신고. 타이레놀정 복용 후 발생.")
    assert not any("보고자" in m for m in r.missing)


def test_redact_skips_common_titles():
    """'선생님/담당자님' 같은 일반 호칭은 이름으로 오탐하지 않는다."""
    r = redact("선생님, 담당자님께 전달해 주세요")
    assert r.text == "선생님, 담당자님께 전달해 주세요"
    assert not r.redacted


def test_redact_clean_text_untouched():
    r = redact("품목허가 심사 기간은 얼마나 걸리나요?")
    assert not r.redacted and r.text == "품목허가 심사 기간은 얼마나 걸리나요?"


# ---------------------------------------------------------------------------
# MCP 도구 (assess_adverse_event)
# ---------------------------------------------------------------------------
def test_assess_tool_returns_grounded_basis():
    out = assess_adverse_event("환자가 복용 후 아나필락시스로 입원", awareness_date="2026-07-01")
    assert out["is_serious"] and out["deadline_days"] == 0
    # 판정 근거 규정 문단(REG-005)이 출처와 함께 부착된다
    ids = [r["doc_id"] for r in out["basis"]["results"]]
    assert "REG-005" in ids


def test_assess_tool_masks_pii_in_result():
    out = assess_adverse_event("환자 박영희님(010-1111-2222)이 복용 후 사망")
    assert "박영희" not in out["case"] and "1111" not in out["case"]
    assert out["pii_masked"]


# ---------------------------------------------------------------------------
# 질의 확장 (도메인 동의어)
# ---------------------------------------------------------------------------
def test_expand_query_bridges_vocabulary_mismatch():
    q = expand_query("부작용 보고 기한은?")
    assert "이상사례" in q
    assert q.startswith("부작용 보고 기한은?")  # 원 질의는 항상 보존


def test_expand_query_no_hit_returns_original():
    q = "품목허가 심사 기간"
    assert expand_query(q) == q


def test_expansion_improves_colloquial_retrieval(pipeline):
    """'설명서'(구어) → '첨부문서'(문서 용어) 확장이 정답 문서를 회수한다."""
    q = "제품 설명서에 경고 문구는 어디에 표시하나요?"
    with_exp = pipeline.retriever.retrieve(q, top_k=8, rerank_n=1, expand=True)
    assert with_exp[0].chunk.doc_id == "REG-004"


# ---------------------------------------------------------------------------
# 에이전트 통합 (라우팅 + 입구 마스킹)
# ---------------------------------------------------------------------------
def test_route_case_description_to_triage():
    assert _route_intent("환자가 복용 후 아나필락시스로 입원했습니다. 언제까지 보고해야 하나요?") == "ae_triage"


def test_route_regulation_question_stays_search():
    # 케이스 서술이 아닌 '규정 질문'은 문서 검색으로
    assert _route_intent("중대한 이상사례는 며칠 안에 보고해야 하나요?") == "search"


@pytest.mark.asyncio
async def test_agent_triage_end_to_end():
    agent = RaAgent()
    r = await agent.chat("환자 김철수님(010-1234-5678)이 복용 후 아나필락시스로 입원했습니다. 언제까지 보고해야 하나요?")
    assert [t.name for t in r.tool_calls] == ["assess_adverse_event"]
    assert r.grounded and any(c["doc_id"] == "REG-005" for c in r.citations)
    # 입구에서 마스킹: 답변·트레이스 어디에도 원 PII가 없다
    assert "김철수" not in r.answer and "1234" not in r.answer
    assert "김철수" not in str(r.trace)
    assert {x["type"] for x in r.redactions} == {"전화번호", "이름(호칭)"}


# ---------------------------------------------------------------------------
# v8 — PII 마스킹 표기 변형 봉합 (호칭+조사·외국인등록번호·전화 변형·환자 번호)
# ---------------------------------------------------------------------------
def test_redact_masks_name_followed_by_josa():
    """호칭 뒤에 조사가 직결된 표기("홍길동님이", "김철수씨가")도 마스킹한다. (v8)

    (?=[^가-힣]|$) 룩어헤드는 님/씨 뒤에 조사(한글)가 붙는 — 한국어에서 가장
    흔한 — 표기를 전부 놓쳤다. 전화·주민번호가 숫자 룩어라운드로 봉합한 바로
    그 '한글 직결' 사각지대가 이름 정규식에는 남아 있던 비대칭."""
    r = redact("홍길동님이 어지러움을 호소했고, 김철수씨가 신고했습니다")
    assert "홍길동" not in r.text and "김철수" not in r.text
    assert r.counts.get("이름(호칭)") == 2


def test_redact_josa_boundary_keeps_stoplist():
    """조사 허용으로 매칭 면적이 넓어져도 일반 호칭어 오탐은 스톱리스트가 막는다. (v8)"""
    r = redact("담당자님이 확인했고 어머님이 보호자로 동행했습니다")
    assert not r.redacted


def test_redact_masks_foreign_resident_number():
    """외국인등록번호(뒤 7자리 첫 숫자 5~8)도 마스킹한다. (v8)

    성별코드를 [1-4]로 좁히면 내국인만 잡히고, 완전한 식별번호인
    외국인등록번호가 통째로 외부 API 경계를 통과한다."""
    r = redact("외국인등록번호는 900101-5234567입니다")
    assert "5234567" not in r.text
    assert r.counts.get("주민등록번호") == 1


def test_redact_phone_variant_notations():
    """en-dash(–)·국제표기(+82, 0 생략)·괄호 지역번호 표기도 마스킹한다. (v8)

    주민번호 패턴은 –를 받는데 전화 패턴만 못 받던 같은 파일 안의 구분자
    비대칭 — 구분자 어휘가 좁으면 그만큼이 우회로다."""
    r = redact("연락처는 010–1234–5678, +82 10-9999-8888, (02)345-6789")
    assert r.counts.get("전화번호") == 3
    for leaked in ("1234", "9999", "345-6789"):
        assert leaked not in r.text


def test_redact_patient_number_with_space():
    """띄어 쓴 "환자 번호"도 잡는다 — 붙여 쓴 표기만 받으면 공백이 우회로다. (v8)"""
    r = redact("환자 번호: A-1023 케이스입니다")
    assert "A-1023" not in r.text
    assert r.counts.get("환자/차트번호") == 1


# ---------------------------------------------------------------------------
# v8 — 라우팅 어휘를 코딩 사전과 공유
# ---------------------------------------------------------------------------
def test_route_common_symptom_case_to_triage():
    """코딩 사전이 아는 일반 증상(두드러기 등)도 케이스 맥락과 결합하면 PV 도구로 간다. (v8)

    이벤트 마커가 중대 어휘뿐이면 가이드 3-2의 예시("B캡슐 투여 후 두드러기…
    인과성은?")가 검색으로 빠져 회피 응답이 된다 — 코딩 사전(coding.symptom_keywords)과
    어휘를 공유해 사전 갱신이 라우팅에 자동 반영된다."""
    assert _route_intent("환자가 B캡슐 투여 후 두드러기가 났고, 중단하니 호전됐습니다. 인과성은?") == "ae_triage"


def test_route_symptom_regulation_question_stays_search():
    """증상 어휘가 있어도 케이스 맥락(_AE_CASE_MARKERS)이 없는 규정 질문은 검색으로 남는다. (v8)"""
    assert _route_intent("두드러기 이상사례 보고 기한은 며칠인가요?") == "search"


# ---------------------------------------------------------------------------
# v9 — 트리아지 부정문·caveat 에코, PII 마스킹 잔여 사각지대 봉합
# ---------------------------------------------------------------------------
def test_triage_negated_expectedness_is_unexpected():
    """"허가사항에 기재되어 있지 않은" 부정문은 unexpected 다. (v9)

    expected 마커("허가사항에 기재")가 부정문의 부분 문자열이라, 부정형이
    unexpected 마커에 없으면 '기재되지 않았다'는 서술이 정반대(expected)로
    뒤집힌다 — expectedness 오판은 정기보고 전환 안내(caveat)의 방향을 바꾼다."""
    t = assess_case("복용 후 입원했고, 허가사항에 기재되어 있지 않은 반응이다")
    assert t.expectedness == "unexpected"
    t2 = assess_case("허가사항에 기재되지 않은 발진으로 입원")
    assert t2.expectedness == "unexpected"
    # "아닌"은 완성형 음절이라 "아니"의 부분 문자열이 아니다 — 별도 마커 검증
    t3 = assess_case("알려진 부작용이 아닌 반응으로 입원")
    assert t3.expectedness == "unexpected"
    # 긍정문은 여전히 expected (PV-010 라벨과 동일 방향 — 오탐 방지 가드)
    t4 = assess_case("허가사항에 기재된 알려진 부작용인 설사가 발생해 입원")
    assert t4.expectedness == "expected"


def test_triage_invalid_awareness_date_caveat_masks_pii():
    """형식 오류 인지일 caveat 가 원문을 비마스킹 에코하지 않는다. (v9)

    awareness_date 는 임의의 자유 텍스트가 들어올 수 있는 입력인데, 형식 오류
    caveat 의 f-string 에코가 이름·전화번호를 그대로 노출했다 — 도구 계층이
    as_of·필터 에코를 redact 로 감싸는 방어와 같은 대칭이 caveat 에도 필요하다."""
    t = assess_case("복용 후 입원", awareness_date="문의는 홍길동님 010-1234-5678")
    joined = " ".join(t.caveats)
    assert "홍길동" not in joined and "010-1234-5678" not in joined
    # 시끄러운 실패 신호 자체는 유지된다(마스킹이 caveat 를 지우면 안 된다)
    assert any("YYYY-MM-DD" in c and "재계산" in c for c in t.caveats)


def test_redact_name_followed_by_any_josa():
    """호칭 뒤 조사가 허용 집합 밖이어도("~님으로부터"·"~님께서") 마스킹한다. (v9)

    조사는 열린 활용 집합이라 첫 글자 열거([이가은는…])로는 집합 밖 조사마다
    유출이 재발한다 — 호칭 뒤는 임의 한글을 허용하고 오탐은 스톱리스트로
    방어하는 설계 반전(마스킹은 과탐이 미탐보다 싸다)."""
    for probe in [
        "홍길동님으로부터 문의가 왔습니다", "김철수님하고 통화했습니다",
        "박영희님처럼 호소했습니다", "이민준님부터 순서대로",
        "정수아님보다 증상이 심했고", "최지훈님마저 발열이 있었습니다",
        "오유진님조차 몰랐습니다", "장서준님밖에 없습니다", "한도윤님께서 보고했습니다",
    ]:
        r = redact(probe)
        assert r.counts.get("이름(호칭)") == 1, f"이름 미마스킹(유출): {probe}"
        assert "[이름]" in r.text


def test_redact_skips_frequent_job_titles():
    """빈출 직함·호칭(교수·박사·원장 등)은 이름으로 오마스킹하지 않는다. (v9)

    "교수님이" → [이름] 오탐은 감사 리포트를 왜곡하고 보고자 직함(자격)
    정보까지 지운다 — 조사 전면 허용으로 넓어진 매칭 면적만큼 스톱리스트도
    함께 넓힌다(오탐도 경계 결함으로 센다)."""
    r = redact("교수님이 문의했고 박사님과 원장님, 실장님, 이사님, 대표님이 검토했습니다")
    assert not r.redacted
    assert r.text == "교수님이 문의했고 박사님과 원장님, 실장님, 이사님, 대표님이 검토했습니다"


def test_redact_masks_dotted_rrn():
    """점(.) 구분 주민번호("900101.1234567")도 마스킹한다. (v9)

    전화 패턴은 점 구분 표기를 받는데 주민번호만 [-–] 로 좁혀져 있던 같은
    파일 안의 구분자 비대칭 — 비대칭은 곧 우회로다(전화·주민번호 구분자
    대칭화의 완결)."""
    r = redact("주민번호는 900101.1234567 입니다")
    assert "900101.1234567" not in r.text
    assert r.counts.get("주민등록번호") == 1


def test_redact_phone_with_spaced_separators():
    """전화번호 구분자 양옆 공백("010 - 1234 - 5678")도 마스킹한다(v11) —
    주민번호는 공백 패딩을 받는데 전화만 못 받던 같은 파일 안 구분자 비대칭
    (비대칭은 곧 우회로 = 유출)."""
    for txt in ["010 - 1234 - 5678", "010- 1234- 5678", "010 . 1234 . 5678"]:
        r = redact(txt)
        assert "[전화번호]" in r.text and "5678" not in r.text, txt
    # 기존 표기 회귀 없음
    assert redact("010-1234-5678").text == "[전화번호]"


def test_redact_fullwidth_digits_via_nfkc():
    """전각 숫자 주민번호·전화("０１０－…")도 마스킹한다(v11) — `\\d`는 전각을
    받지만 성별코드 [1-8]·앞자리 01[016789] 같은 ASCII 앵커는 못 받아 통째로
    유출되던 표기 축을, 진입부 NFKC 정규화로 클래스 차원에서 봉합."""
    r1 = redact("주민번호 ９００１０１－１２３４５６７")
    assert "[주민번호]" in r1.text and "1234567" not in r1.text
    r2 = redact("연락처 ０１０－１２３４－５６７８")
    assert "[전화번호]" in r2.text
