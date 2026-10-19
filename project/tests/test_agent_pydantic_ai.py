"""PydanticAI 백엔드(AGENT_BACKEND=pydantic_ai)의 계약 테스트 — 키 없이 실행.

test_agent_llm.py 가 고정한 LLM 모드의 계약(신뢰 소스 수집·이력 허용 집합·
grounded 판정·검증 게이트 통과)을 PydanticAI 백엔드도 동일하게 지키는지
같은 시나리오로 검증한다. 모델만 FunctionModel/TestModel 스텁으로 바꾸고
MCPToolset·MCP 서버·RAG·검증 게이트는 전부 실물을 태운다 — 두 백엔드가
'같은 계약'이라는 주장을 양쪽 모두 실행으로 고정하는 것.
"""
from __future__ import annotations

import pytest

from pydantic_ai import models
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from src import config
from src.agent import pydantic_agent
from src.agent.agent import RaAgent

models.ALLOW_MODEL_REQUESTS = False  # 실 API 호출 원천 차단 — 키 없는 테스트 보증


def _use_backend(monkeypatch, model) -> None:
    """PydanticAI 백엔드를 켜고 모델 봉합선(_build_model)을 스텁으로 교체."""
    monkeypatch.setattr(config, "LLM_AVAILABLE", True)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(config, "AGENT_BACKEND", "pydantic_ai")
    monkeypatch.setattr(pydantic_agent, "_build_model", lambda: model)


def _scripted(steps: list[list]) -> FunctionModel:
    """준비된 응답 파트를 순서대로 돌려주는 모델 — direct 테스트의 _stub_anthropic 대응물."""
    state = {"i": 0}

    def fn(messages, info: AgentInfo) -> ModelResponse:
        parts = steps[state["i"]]
        state["i"] += 1
        return ModelResponse(parts=parts)

    return FunctionModel(fn)


async def test_pa_history_search_allows_only_returned_superseded_docs(monkeypatch):
    """as_of(이력) 검색이 '실제로 반환한' 폐지본 인용은 경고에서 면제된다 —
    process_tool_call 훅의 history_doc_ids 배선이 direct 루프와 동일한지."""
    _use_backend(monkeypatch, _scripted([
        [ToolCallPart(tool_name="search_regulations",
                      args={"query": "중대한 이상사례 보고 기한", "as_of": "2025-01-01"})],
        [TextPart(content="그 시점 당시 현행이던 구판 기준으로 보고 기한은 30일 이내였습니다.")],
    ]))
    r = await RaAgent().chat("2025년 1월 시점의 이상사례 보고 기한은?")
    assert r.mode == "llm"
    assert any(c.get("doc_id") == "REG-013" for c in r.citations), r.citations
    assert r.verification["superseded_cited"] == []
    assert r.verification["ok"], r.verification


async def test_pa_case_echo_support_is_labeled(monkeypatch):
    """케이스 서술 수치의 재서술은 from_case(case_origin) 라벨 — user_facts
    분리 계층이 훅 경로에서도 유지되는지."""
    _use_backend(monkeypatch, _scripted([
        [ToolCallPart(tool_name="assess_adverse_event",
                      args={"case_description": "환자가 A정을 30일간 복용 후 두드러기로 입원했습니다"})],
        [TextPart(content="복용 기간 30일의 입원 케이스로 중대에 해당하며, 보고 기한은 15일 이내입니다.")],
    ]))
    r = await RaAgent().chat("이 케이스 언제까지 보고해야 하나요?")
    assert r.mode == "llm"
    assert r.verification["ok"], r.verification
    assert r.verification["case_origin"] == ["30일"]
    assert "15일" not in r.verification["case_origin"]


async def test_pa_failed_as_of_does_not_open_allowance(monkeypatch):
    """형식이 틀린 as_of(에러 계약 응답)는 허용 집합·신뢰 소스를 채우지 못한다."""
    _use_backend(monkeypatch, _scripted([
        [ToolCallPart(tool_name="search_regulations",
                      args={"query": "이상사례 보고 기한", "as_of": "2025/01/01"})],  # 형식 오류
        [TextPart(content="보고 기한 규정을 확인하지 못했습니다.")],
    ]))
    r = await RaAgent().chat("2025년 1월 시점의 이상사례 보고 기한은?")
    assert r.mode == "llm"
    assert r.citations == []
    assert r.verification["ok"]
    assert all(not c["supported"] for c in r.verification["checks"]) or not r.verification["checks"]
    assert r.grounded is False


async def test_pa_no_tool_answer_is_not_grounded(monkeypatch):
    """도구를 한 번도 부르지 않은 답변은 grounded=False."""
    _use_backend(monkeypatch, _scripted([
        [TextPart(content="일반적으로 신속히 보고하는 것이 좋습니다.")],
    ]))
    r = await RaAgent().chat("이상사례는 어떻게 보고하나요?")
    assert r.mode == "llm"
    assert r.citations == [] and r.tool_calls == []
    assert r.grounded is False


async def test_pa_tool_grounded_answer_is_grounded(monkeypatch):
    """성공한 도구 출력이 신뢰 소스로 확보된 답변은 grounded=True."""
    _use_backend(monkeypatch, _scripted([
        [ToolCallPart(tool_name="search_regulations",
                      args={"query": "중대한 이상사례 보고 기한"})],
        [TextPart(content="중대한 이상사례는 15일 이내에 보고합니다.")],
    ]))
    r = await RaAgent().chat("중대한 이상사례 보고 기한은?")
    assert r.mode == "llm"
    assert r.grounded is True and r.citations


async def test_pa_empty_search_result_is_not_grounded(monkeypatch):
    """'성공했으나 결과 0건'인 검색은 grounded=True 로 만들지 않는다(v10 대칭)."""
    _use_backend(monkeypatch, _scripted([
        [ToolCallPart(tool_name="search_regulations",
                      args={"query": "중대한 이상사례 보고 기한", "as_of": "1900-01-01"})],
        [TextPart(content="해당 시점에는 확인되는 규정이 없습니다.")],
    ]))
    r = await RaAgent().chat("1900년 시점의 이상사례 보고 기한은?")
    assert r.mode == "llm"
    assert r.citations == []
    assert r.grounded is False


async def test_pa_api_failure_is_explicit_not_500(monkeypatch):
    """모델 호출 실패는 예외 전파 대신 명시적 안내 — 타입명만 싣고 메시지는 비에코."""
    monkeypatch.setattr(config, "LLM_AVAILABLE", True)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "bad-key")
    monkeypatch.setattr(config, "AGENT_BACKEND", "pydantic_ai")

    def _boom():
        raise RuntimeError("secret-request-detail")

    monkeypatch.setattr(pydantic_agent, "_build_model", _boom)
    r = await RaAgent().chat("중대한 이상사례 보고 기한은?")
    assert r.mode == "llm"
    assert "LLM API 호출에 실패" in r.answer and "RuntimeError" in r.answer
    assert "secret-request-detail" not in r.answer
    assert r.grounded is False
    assert r.verification


async def test_pa_request_limit_maps_to_explicit_answer(monkeypatch):
    """도구 호출만 반복되면 UsageLimits 상한이 걸리고 — direct 6-step 상한과 같은
    확정 실패 문구로 마감된다. 상한까지 쌓인 증거(tool_calls)는 보존된다."""
    def always_tool(messages, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[
            ToolCallPart(tool_name="search_regulations", args={"query": "중대한 이상사례 보고 기한"})
        ])

    _use_backend(monkeypatch, FunctionModel(always_tool))
    r = await RaAgent().chat("중대한 이상사례 보고 기한은?")
    assert r.mode == "llm"
    assert "응답을 확정하지 못했습니다" in r.answer
    assert r.grounded is False
    assert r.tool_calls  # 상한 초과여도 축적된 북키핑은 리셋되지 않는다


async def test_pa_multiturn_history_carries_context(monkeypatch):
    """dict 이력이 ModelMessage 로 변환되어 전달되고, 직전 사용자 발화의 수치는
    검증에서 '전제(question_origin)'로 라벨링된다 — direct 경로와 같은 완화 창."""
    seen = {}

    def fn(messages, info: AgentInfo) -> ModelResponse:
        seen["n_messages"] = len(messages)
        return ModelResponse(parts=[TextPart(content="네, 전제하신 보고 기한 30일 기준으로는 그렇습니다.")])

    _use_backend(monkeypatch, FunctionModel(fn))
    r = await RaAgent().chat(
        "그럼 그 기한 안에 어떻게 하나요?",
        history=[
            {"role": "user", "content": "보고 기한이 30일 맞지?"},
            {"role": "assistant", "content": [{"type": "text", "text": "확인해 보겠습니다."}]},
        ],
    )
    assert r.mode == "llm"
    assert seen["n_messages"] >= 2  # 이력이 실제로 모델에 전달됐다
    # 30일은 도구 근거가 없으므로 경고가 붙되, 직전 턴 전제로 라벨링된다
    assert "30일" in r.verification["question_origin"], r.verification


async def test_pa_testmodel_smoke(monkeypatch):
    """TestModel(스키마 기반 자동 도구 호출) 스모크 — call_tool 반환 형태가
    dict 로 정규화되어 북키핑이 성립하는지의 카나리."""
    _use_backend(monkeypatch, TestModel(call_tools=["get_ra_deadlines"]))
    r = await RaAgent().chat("이번 주 마감 뭐 있어?")
    assert r.mode == "llm"
    assert [t.name for t in r.tool_calls] == ["get_ra_deadlines"]
    assert r.verification  # 게이트 통과
    assert r.grounded is True  # 마감 데이터는 내용 있는 결정론 도구 출력


_DDG_HTML = """
<div class="result">
  <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fnews%2F1&amp;rut=abc">MFDS 이상사례 보고 고시 개정</a>
  <a class="result__snippet" href="#">개정 고시는 2026-05-01 시행 예정이다.</a>
</div>
"""


async def test_pa_web_search_refused_without_explicit_request(monkeypatch):
    """명시 요청 없는 턴의 search_web 호출은 훅에서 에러 계약으로 거부된다 —
    MCPToolset 은 서버 도구를 그대로 노출하므로(direct 의 도구 목록 필터와
    다른 봉합선) 실행 관문이 같은 계약('미요청 턴에 웹 검색 없음')을 지킨다."""
    import src.websearch as websearch

    def _never(q):
        raise AssertionError("명시 요청 없는 턴에 인터넷 검색이 수행됐다")

    monkeypatch.setattr(websearch, "_fetch_html", _never)
    _use_backend(monkeypatch, _scripted([
        [ToolCallPart(tool_name="search_web", args={"query": "이상사례 최신 동향"})],
        [TextPart(content="사내 규정문서 기준으로는 근거를 찾지 못했습니다.")],
    ]))
    r = await RaAgent().chat("이상사례 최신 동향 알려줘")   # 웹 요청 구문 없음
    assert r.mode == "llm"
    assert r.web_results == []
    assert r.grounded is False                # 거부(에러 계약)는 근거가 아니다
    assert r.tool_calls and r.tool_calls[0].name == "search_web"  # 시도는 기록된다


async def test_pa_web_results_are_separated_when_requested(monkeypatch):
    """명시 요청 턴 — 웹 출력은 web_texts/web_results 로만 흐르고 사내 신뢰
    소스·grounded 를 오염시키지 않는다(direct 백엔드와 같은 분리 계약)."""
    import src.websearch as websearch

    monkeypatch.setattr(websearch, "_fetch_html", lambda q: _DDG_HTML)
    _use_backend(monkeypatch, _scripted([
        [ToolCallPart(tool_name="search_web", args={"query": "이상사례 보고 고시 개정 뉴스"})],
        [TextPart(content="🌐 인터넷 검색 결과에 따르면 개정 고시는 2026-05-01 시행 예정입니다.")],
    ]))
    r = await RaAgent().chat("이상사례 보고 고시 개정 소식 웹에서 검색해줘")
    assert r.mode == "llm"
    assert r.web_results and r.web_results[0]["origin"] == "web"
    assert r.citations == [] and r.grounded is False
    assert r.verification["ok"], r.verification
    assert "2026-05-01" in r.verification["web_origin"]


async def test_pa_default_backend_is_direct():
    """AGENT_BACKEND 기본값은 'direct' — 명시적으로 켜지 않으면 PydanticAI 백엔드로
    라우팅되지 않는다(기존 경로·평가 수치 보존 가드)."""
    assert config.AGENT_BACKEND == "direct"
