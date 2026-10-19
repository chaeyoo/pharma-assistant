"""에이전트 테스트: 인텐트 라우팅 · abstention · 멀티턴 · 도구 에러 견고성 · 트레이스.

오프라인 모드(ANTHROPIC_API_KEY 없음)를 기준으로 검증한다.
"""
from __future__ import annotations

import pytest

from src.agent.agent import RaAgent
from src.observability import Span, Trace


@pytest.mark.asyncio
async def test_intent_routing():
    agent = RaAgent()
    r1 = await agent.chat("품목허가 심사 기간은?")
    assert r1.tool_calls[0].name == "search_regulations"
    r2 = await agent.chat("이번 주 마감 임박한 업무 알려줘")
    assert r2.tool_calls[0].name == "get_ra_deadlines"
    r3 = await agent.chat("변경허가 체크리스트 알려줘")
    assert r3.tool_calls[0].name == "get_submission_checklist"
    assert all(r.answer for r in (r1, r2, r3))


@pytest.mark.asyncio
async def test_in_scope_is_grounded_and_cited():
    agent = RaAgent()
    r = await agent.chat("중대한 이상사례는 며칠 이내에 보고하나요?")
    assert r.grounded is True
    assert r.citations, "범위내 답변에는 출처가 있어야 함"


@pytest.mark.asyncio
async def test_out_of_scope_abstains():
    agent = RaAgent()
    for q in ["비트코인 지금 사도 될까요?", "우리 팀 회식 장소 추천해줘"]:
        r = await agent.chat(q)
        assert r.grounded is False, f"범위밖 '{q}' 은 abstain 해야 함"
        assert not r.citations


@pytest.mark.asyncio
async def test_offline_multiturn_followup():
    agent = RaAgent()
    history = [
        {"role": "user", "content": "변경허가 처리기한은?"},
        {"role": "assistant", "content": "..."},
    ]
    r = await agent.chat("그럼 변경신고는?", history)
    # 후속질문이 직전 맥락과 병합되어 변경 관련 문서로 검색되어야 함
    assert "변경허가" in r.tool_calls[0].args["query"]
    assert any(c["doc_id"] == "REG-002" for c in r.citations)


@pytest.mark.asyncio
async def test_trace_is_populated():
    agent = RaAgent()
    r = await agent.chat("GMP 적합판정 유효기간은?")
    assert r.trace, "트레이스 span 이 기록되어야 함"
    assert r.latency_ms > 0
    assert any(s["kind"] == "tool" for s in r.trace)
    # 총 지연은 최상위 agent span 의 wall-clock 이다 — 스텝 span 까지 합산하면
    # 같은 시간이 두 번 세어져 계기판이 ~2배 지연을 보고한다(관측 왜곡 회귀 가드)
    agent_span = next(s for s in r.trace if s["kind"] == "agent")
    assert r.latency_ms == agent_span["ms"]


def test_latency_is_wall_clock_not_span_sum():
    """Trace.total_ms — 중첩 span 합산이 아니라 최상위 span 의 wall-clock."""
    trace = Trace()
    trace.add(Span(name="tool.search", kind="tool", duration_ms=80.0))
    trace.add(Span(name="chat", kind="agent", duration_ms=100.0))
    assert trace.total_ms == 100.0  # 180(합산)이 아니라 100(wall-clock)
    # 최상위 span 이 없는 부분 트레이스는 합산으로 폴백
    partial = Trace()
    partial.add(Span(name="tool.a", kind="tool", duration_ms=30.0))
    partial.add(Span(name="tool.b", kind="tool", duration_ms=20.0))
    assert partial.total_ms == 50.0


class _FailingClient:
    async def call_tool(self, name, args):
        raise RuntimeError("simulated MCP failure")


@pytest.mark.asyncio
async def test_tool_failure_is_absorbed_not_crashed():
    """MCP 도구가 예외를 던져도 크래시 대신 (에러메시지, is_error=True) 로 흡수."""
    agent = RaAgent()
    data, is_error = await agent._safe_tool_call(
        _FailingClient(), "search_regulations", {"query": "x"}, Trace()
    )
    assert is_error is True
    assert "실행 실패" in data


@pytest.mark.asyncio
async def test_colloquial_in_scope_not_over_abstained():
    """abstention 문턱 보정의 회귀 가드: 검색이 동의어 확장으로 정답을 찾는
    구어 질의는 '근거 없음'으로 회피하면 안 된다(과회피였던 실제 사례)."""
    agent = RaAgent()
    r = await agent.chat("부작용이 심각하게 나타났을 때 당국에 얼마나 빨리 알려야 하나요?")
    assert r.grounded is True, "확장 커버리지를 반영하면 범위내로 판정되어야 함"
    assert r.citations


@pytest.mark.asyncio
async def test_unmatched_checklist_falls_back_instead_of_guessing():
    """'GMP 체크리스트' — 지원 목록에 없는 유형. 이전에는 라우터가 '품목허가'를
    기본 추측해 엉뚱한 체크리스트를 자신 있게 반환했다(도구에는 error+available
    계약을 만들어 놓고 라우터가 항상 유효 카테고리를 추측해 그 계약을 우회).
    이제 지원 목록 안내 + 규정 검색 폴백으로 답한다."""
    agent = RaAgent()
    r = await agent.chat("GMP 체크리스트 줘")
    names = [t.name for t in r.tool_calls]
    assert names[0] == "get_submission_checklist"
    assert "search_regulations" in names, "검색 폴백이 수행되어야 함"
    assert "지원:" in r.answer, "지원 체크리스트 목록 안내가 있어야 함"
    assert "[품목허가] 준비 체크리스트" not in r.answer, "엉뚱한 체크리스트를 추측 반환하면 안 됨"


@pytest.mark.asyncio
async def test_matched_checklist_still_direct():
    """확신 있는 신호(변경/안전/허가 어휘)는 기존대로 체크리스트를 바로 답한다."""
    agent = RaAgent()
    r = await agent.chat("변경허가 체크리스트 알려줘")
    assert r.tool_calls[0].name == "get_submission_checklist"
    assert "[변경허가] 준비 체크리스트" in r.answer


# 인터넷 검색 스텁 — 실제 네트워크 없이 오프라인 웹 라우팅 경로를 태운다.
_DDG_HTML = """
<div class="result">
  <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fnews%2F1&amp;rut=abc">MFDS 이상사례 보고 고시 개정</a>
  <a class="result__snippet" href="#">개정 고시는 2026-05-01 시행 예정이다.</a>
</div>
"""


@pytest.mark.asyncio
async def test_explicit_web_request_routes_to_search_web_and_separates(monkeypatch):
    """'인터넷에서 검색해줘'라는 명시 요청 → search_web 라우팅. 결과는 사내
    근거와 전면 분리된다: 🌐 표시가 본문에, 결과는 citations 가 아닌
    web_results 로, grounded(사내 근거 배지)는 False 로, 웹 유래 수치는
    검증에서 경고 대신 web_origin 라벨로."""
    import src.websearch as websearch

    monkeypatch.setattr(websearch, "_fetch_html", lambda q: _DDG_HTML)
    agent = RaAgent()
    r = await agent.chat("이상사례 보고 고시 개정 최신 뉴스 인터넷에서 검색해줘")
    assert [t.name for t in r.tool_calls] == ["search_web"]
    assert "🌐 인터넷 검색 결과" in r.answer and "example.com" in r.answer
    assert r.citations == []                      # 사내 출처 카드에 섞이지 않는다
    assert r.grounded is False                    # '사내 도구 근거 확보'가 아니다
    assert r.web_results and r.web_results[0]["origin"] == "web"
    # 웹 스니펫의 날짜(2026-05-01)는 미확인 경고가 아니라 웹 유래 라벨
    assert r.verification["ok"], r.verification
    assert "2026-05-01" in r.verification["web_origin"], r.verification


@pytest.mark.asyncio
async def test_no_web_fallback_without_explicit_request(monkeypatch):
    """명시 요청이 없으면 — 사내 문서에서 못 찾는 질문도 웹 폴백 없이 종전
    그대로 회피(abstention)한다. '못 찾으면 못 찾는다고 답한다'의 회귀 가드."""
    import src.websearch as websearch

    def _never(q):
        raise AssertionError("명시 요청 없는 턴에 인터넷 검색이 수행됐다")

    monkeypatch.setattr(websearch, "_fetch_html", _never)
    agent = RaAgent()
    r = await agent.chat("비트코인 지금 사도 될까요?")
    assert r.grounded is False and not r.citations          # 종전과 같은 회피
    assert "search_web" not in [t.name for t in r.tool_calls]
    assert r.web_results == []


@pytest.mark.asyncio
async def test_web_topic_word_alone_is_not_a_web_request():
    """질문 '주제'에 인터넷/온라인이 들어가는 것은 검색 요청이 아니다 —
    외부 송신 경로를 여는 스위치의 오탐 방지(마커는 요청형 구문만)."""
    agent = RaAgent()
    r = await agent.chat("의약품 인터넷 판매 광고 규정은?")
    assert r.tool_calls[0].name == "search_regulations"
    assert "search_web" not in [t.name for t in r.tool_calls]


@pytest.mark.asyncio
async def test_web_search_failure_is_explicit(monkeypatch):
    """네트워크 실패·차단 — 조용한 빈 결과 대신 원인을 말하는 실패 안내."""
    import src.websearch as websearch

    def _boom(q):
        raise RuntimeError("blocked")

    monkeypatch.setattr(websearch, "_fetch_html", _boom)
    agent = RaAgent()
    r = await agent.chat("이상사례 최신 동향 웹에서 검색해줘")
    assert "인터넷 검색을 수행하지 못했습니다" in r.answer
    assert r.grounded is False and r.web_results == []


def test_strip_echo_removes_all_user_input_fields():
    """도구 출력의 사용자 입력 에코(query·as_of)는 신뢰 소스에서 제외 — query만
    빼고 as_of를 남기면 사용자가 지정한 기준일이 날짜 클레임의 '근거'로 승격된다."""
    from src.agent.agent import _strip_query_echo

    data = {"query": "보고 기한 30일?", "as_of": "2024-06-01",
            "results": [{"text": "15일 이내", "query": "중첩 에코"}]}
    out = _strip_query_echo(data)
    assert "query" not in out and "as_of" not in out
    assert out["results"][0]["text"] == "15일 이내"
    assert "query" not in out["results"][0]


def test_question_context_labels_multiturn_premise():
    """이전 턴의 전제 수치를 이어받은 답변은 '환각'이 아니라 '전제 확인 필요'로
    라벨링되어야 한다 — 단 경고 자체는 여전히 붙는다(면제가 아니라 종류 조정)."""
    from src.agent.agent import _question_context
    from src.verify.verifier import verify_answer

    history = [
        {"role": "user", "content": "보고 기한이 30일 맞지?"},
        {"role": "assistant", "content": "규정 확인이 필요합니다."},
    ]
    q = _question_context("그럼 그 기준으로 언제까지야?", history)
    assert "30일" in q  # 이력의 사용자 발화가 맥락에 포함된다
    v = verify_answer("보고 기한은 30일입니다", ["중대한 이상사례 보고 기한은 15일 이내"], question=q)
    assert not v.ok, "전제 라벨은 경고 면제가 아니다"
    assert "30일" in v.question_origin, "이전 턴 전제는 from_question 으로 라벨링"


def test_question_context_excludes_assistant_turns():
    """어시스턴트의 이전 답변 속 수치는 전제로 완화 라벨링하지 않는다 — 한 번
    새어 나간 미확인 수치가 다음 턴부터 '전제'로 면책되는 자기 강화 루프 차단."""
    from src.agent.agent import _question_context

    history = [
        {"role": "user", "content": "보고 기한 알려줘"},
        {"role": "assistant", "content": "보고 기한은 45일입니다."},
    ]
    q = _question_context("확실해?", history)
    assert "45일" not in q


@pytest.mark.asyncio
async def test_offline_history_intent_includes_superseded():
    """'이력을 요청하면 구판도 노출'이라는 사용 계약은 LLM 모드(모델이 스스로
    include_superseded 지정)에서만 참이었다 — 오프라인 라우터에는 그 경로 자체가
    없었다. 이제 이력 의도 감지 시 폐지본 포함 + 폐지본 인용 경고 억제."""
    agent = RaAgent()
    r = await agent.chat("이상사례 보고 기한, 예전 규정에서는 어땠어? 개정 이력 포함해서 알려줘")
    assert r.tool_calls[0].args.get("include_superseded") is True
    assert any(c.get("doc_id") == "REG-013" for c in r.citations), "구판이 출처에 노출되어야 함"
    assert not r.verification.get("superseded_cited"), "이력 조회의 폐지본 인용은 결함이 아니다"


@pytest.mark.asyncio
async def test_offline_default_search_stays_current_only():
    """이력 어휘가 없는 일반 질문은 여전히 현행만 검색한다(기본값 회귀 가드)."""
    agent = RaAgent()
    r = await agent.chat("중대한 이상사례 보고 기한은?")
    assert not r.tool_calls[0].args.get("include_superseded")
    assert all(c.get("doc_id") != "REG-013" for c in r.citations)


def test_question_context_bounded_to_previous_turn():
    """전제 완화 라벨의 창은 직전 사용자 턴까지 — 대화 전체를 누적하면 오래전
    무관한 맥락의 수치가 이후 모든 환각을 '전제'로 완화 라벨링한다."""
    from src.agent.agent import _question_context

    history = [
        {"role": "user", "content": "시장점유율이 90%인 제품인데 잘 팔려요"},  # 오래된 무관 맥락
        {"role": "assistant", "content": "네."},
        {"role": "user", "content": "중대한 이상사례 보고 절차가 궁금해요"},   # 직전 턴
        {"role": "assistant", "content": "규정을 검색해 안내드릴게요."},
    ]
    q = _question_context("그럼 기한은?", history)
    assert "보고 절차" in q       # 직전 턴은 포함
    assert "90%" not in q         # 오래된 턴은 미포함


def test_contract_error_is_not_trusted_source():
    """도구의 에러 계약({"error", ...})은 자가 정정용 되먹임이지 근거가 아니다 —
    에러 문구에 에코된 사용자 입력(잘못된 as_of 등)이 신뢰 소스로 승격되면
    query·as_of 키 제거로 막은 구멍이 에러 경로로 다시 열린다."""
    from src.agent.agent import _is_contract_error

    assert _is_contract_error({"error": "as_of '2024/06/01' 가 YYYY-MM-DD 형식이 아님",
                               "expected": "YYYY-MM-DD"})
    assert not _is_contract_error({"results": []})
    assert not _is_contract_error("텍스트 출력")


@pytest.mark.asyncio
async def test_history_markers_do_not_overmatch_current_questions():
    """'이력 관리 요건' 같은 현행 질문의 일상 어휘가 이력 조회로 오인되면
    폐지본이 검색에 섞이고 버전 경고(allow_superseded)까지 꺼진다 — 안전장치를
    끄는 스위치는 명시적 요청 구문에만 반응해야 한다."""
    agent = RaAgent()
    r = await agent.chat("이상사례 접수·처리 이력 관리 요건은?")
    assert not r.tool_calls[0].args.get("include_superseded")


@pytest.mark.asyncio
async def test_unmatched_checklist_out_of_corpus_stays_coherent():
    """체크리스트 미매칭 + 폴백 검색까지 회피된 경우 — 안내문이 '검색 결과'를
    미리 약속하면 회피 문구와 자기모순이 된다. 지원 목록 안내와 회피 안내가
    모순 없이 공존해야 한다."""
    agent = RaAgent()
    r = await agent.chat("ISO 9001 인증 체크리스트 줘")
    assert "지원:" in r.answer
    assert "아래는 규정 검색 결과" not in r.answer


def test_redact_history_masks_block_content():
    """이력 마스킹의 표기 변형 — content 가 문자열이 아니라 anthropic 블록
    리스트(content=[{"type":"text",...}])여도 PII 가 마스킹되어야 한다.
    문자열 표기만 마스킹하고 리스트 표기를 원문 통과시키면, 그 표기로 들어온
    개인정보가 외부 LLM API 로 그대로 나간다(v7 발견 — _user_texts 는 블록
    표기를 예상 입력으로 처리하는데 마스킹만 그 표기를 몰랐다)."""
    from src.agent.agent import _redact_history

    turns = _redact_history([
        {"role": "user", "content": "주민번호는 900101-1234567입니다"},
        {"role": "user", "content": [
            {"type": "text", "text": "보호자 연락처는 010-9876-5432로 부탁드립니다"},
            # tool_result 블록의 문자열 본문(content 키)도 텍스트 표기다 —
            # raw transcript 를 이력으로 되돌려 보내는 사용에서 이 키만 빠지면
            # 같은 우회가 키 이름 차이로 반복된다.
            {"type": "tool_result", "tool_use_id": "t1", "content": "환자 전화 010-1111-2222"},
        ]},
        {"role": "user", "content": 42},  # 비문자열 스칼라 — 텍스트 아님, 그대로
    ])
    assert "900101-1234567" not in turns[0]["content"]
    assert "010-9876-5432" not in turns[1]["content"][0]["text"]
    assert "[전화번호]" in turns[1]["content"][0]["text"]
    assert "010-1111-2222" not in turns[1]["content"][1]["content"]
    assert turns[1]["content"][1]["tool_use_id"] == "t1"
    assert turns[2]["content"] == 42


# ---------------------------------------------------------------------------
# v8 — 오프라인 라우터의 인지일 승격 (조용한 폴백 금지)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_offline_awareness_date_extracted_from_message():
    """메시지에 명시된 인지일이 도구 인자로 승격돼 기한 계산에 반영된다. (v8)

    라우터가 case_description 만 전달하면 awareness_date 는 항상 빈 값이라
    도구가 경고 없이 '오늘'로 폴백했다 — 사용자가 명시한 인지일이 조용히
    무시되는, '조용한 폴백 금지' 원칙의 라우터 계층 위반."""
    agent = RaAgent()
    r = await agent.chat("환자가 A정 복용 후 중환자실에 입원했습니다. 인지일은 2026-07-01. 언제까지 보고해야 하나요?")
    assert [t.name for t in r.tool_calls] == ["assess_adverse_event"]
    assert "인지일 2026-07-01 기준" in r.answer


@pytest.mark.asyncio
async def test_offline_malformed_awareness_date_gets_loud_caveat():
    """형식이 틀린 인지일은 라우터가 걸러 조용히 버리지 않고 그대로 전달해,
    도구(triage)의 형식 오류 caveat 계약이 발화하게 한다. (v8)

    라우터가 ISO 정규화·필터링을 하면 도구의 에러 계약을 층 아래에서
    우회하는 형태가 된다(_guess_category 와 같은 규율)."""
    agent = RaAgent()
    r = await agent.chat("환자가 A정 복용 후 입원했습니다. 인지일은 2026-7-1. 보고 기한은?")
    assert "형식이 아니" in r.answer  # 시끄러운 caveat
