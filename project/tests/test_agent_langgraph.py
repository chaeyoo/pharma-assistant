"""LangGraph 백엔드(AGENT_BACKEND=langgraph)의 계약 테스트 — 키 없이 실행.

test_agent_llm.py(direct)·test_agent_pydantic_ai.py 가 고정한 LLM 모드의 계약
(신뢰 소스 수집·이력 허용 집합·grounded 판정·검증 게이트 통과)을 LangGraph
백엔드도 동일하게 지키는지 같은 시나리오로 검증한다. 모델만 스크립트 스텁으로
바꾸고 StateGraph·ToolNode·MCP 서버·RAG·검증 게이트는 전부 실물을 태운다 —
세 백엔드가 '같은 계약'이라는 주장을 전부 실행으로 고정하는 것.
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src import config
from src.agent import langgraph_agent
from src.agent.agent import RaAgent


class _ScriptedChat(BaseChatModel):
    """준비된 AIMessage 를 순서대로 돌려주는 모델 — FunctionModel 대응물.

    스크립트를 소진하면 마지막 스텝을 반복한다(recursion_limit 테스트용).
    """

    steps: list
    i: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):  # noqa: D102 - 스텁: 바인딩 무시
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        msg = self.steps[min(self.i, len(self.steps) - 1)]
        self.i += 1
        # 매번 새 사본(id 제거)을 반환한다 — 같은 객체를 반복 반환하면 LangGraph
        # add_messages 리듀서가 같은 message id 로 보고 append 대신 '교체'해
        # 그래프가 조기 종료된다(실모델은 매 응답이 새 메시지라 없는 문제).
        return ChatResult(generations=[ChatGeneration(message=msg.model_copy(update={"id": None}))])


def _tool_call(name: str, args: dict, call_id: str = "c1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def _use_backend(monkeypatch, model: BaseChatModel) -> None:
    """LangGraph 백엔드를 켜고 모델 봉합선(_build_model)을 스텁으로 교체."""
    monkeypatch.setattr(config, "LLM_AVAILABLE", True)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(config, "AGENT_BACKEND", "langgraph")
    monkeypatch.setattr(langgraph_agent, "_build_model", lambda: model)


async def test_lg_history_search_allows_only_returned_superseded_docs(monkeypatch):
    """as_of(이력) 검색이 '실제로 반환한' 폐지본 인용은 경고에서 면제된다 —
    도구 래퍼의 history_doc_ids 배선이 direct/PydanticAI 와 동일한지."""
    _use_backend(monkeypatch, _ScriptedChat(steps=[
        _tool_call("search_regulations",
                   {"query": "중대한 이상사례 보고 기한", "as_of": "2025-01-01"}),
        AIMessage(content="그 시점 당시 현행이던 구판 기준으로 보고 기한은 30일 이내였습니다."),
    ]))
    r = await RaAgent().chat("2025년 1월 시점의 이상사례 보고 기한은?")
    assert r.mode == "llm"
    assert any(c.get("doc_id") == "REG-013" for c in r.citations), r.citations
    assert r.verification["superseded_cited"] == []
    assert r.verification["ok"], r.verification


async def test_lg_no_tool_answer_is_not_grounded(monkeypatch):
    """도구를 한 번도 부르지 않은 답변은 grounded=False."""
    _use_backend(monkeypatch, _ScriptedChat(steps=[
        AIMessage(content="일반적으로 신속히 보고하는 것이 좋습니다."),
    ]))
    r = await RaAgent().chat("이상사례는 어떻게 보고하나요?")
    assert r.mode == "llm"
    assert r.citations == [] and r.tool_calls == []
    assert r.grounded is False


async def test_lg_tool_grounded_answer_is_grounded(monkeypatch):
    """성공한 도구 출력이 신뢰 소스로 확보된 답변은 grounded=True."""
    _use_backend(monkeypatch, _ScriptedChat(steps=[
        _tool_call("search_regulations", {"query": "중대한 이상사례 보고 기한"}),
        AIMessage(content="중대한 이상사례는 15일 이내에 보고합니다."),
    ]))
    r = await RaAgent().chat("중대한 이상사례 보고 기한은?")
    assert r.mode == "llm"
    assert r.grounded is True and r.citations


async def test_lg_empty_search_result_is_not_grounded(monkeypatch):
    """'성공했으나 결과 0건'인 검색은 grounded=True 로 만들지 않는다(v10 대칭)."""
    _use_backend(monkeypatch, _ScriptedChat(steps=[
        _tool_call("search_regulations",
                   {"query": "중대한 이상사례 보고 기한", "as_of": "1900-01-01"}),
        AIMessage(content="해당 시점에는 확인되는 규정이 없습니다."),
    ]))
    r = await RaAgent().chat("1900년 시점의 이상사례 보고 기한은?")
    assert r.mode == "llm"
    assert r.citations == []
    assert r.grounded is False


async def test_lg_api_failure_is_explicit_not_500(monkeypatch):
    """모델 호출 실패는 예외 전파 대신 명시적 안내 — 타입명만 싣고 메시지는 비에코."""
    monkeypatch.setattr(config, "LLM_AVAILABLE", True)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "bad-key")
    monkeypatch.setattr(config, "AGENT_BACKEND", "langgraph")

    def _boom():
        raise RuntimeError("secret-request-detail")

    monkeypatch.setattr(langgraph_agent, "_build_model", _boom)
    r = await RaAgent().chat("중대한 이상사례 보고 기한은?")
    assert r.mode == "llm"
    assert "LLM API 호출에 실패" in r.answer and "RuntimeError" in r.answer
    assert "secret-request-detail" not in r.answer
    assert r.grounded is False
    assert r.verification


async def test_lg_recursion_limit_maps_to_explicit_answer(monkeypatch):
    """도구 호출만 반복되면 recursion_limit 이 걸리고 — direct 6-step 상한과 같은
    확정 실패 문구로 마감된다. 상한까지 쌓인 증거(tool_calls)는 보존된다."""
    _use_backend(monkeypatch, _ScriptedChat(steps=[
        _tool_call("search_regulations", {"query": "중대한 이상사례 보고 기한"}),
    ]))  # 스크립트 소진 후에도 같은 도구 호출 반복 → 상한 도달
    r = await RaAgent().chat("중대한 이상사례 보고 기한은?")
    assert r.mode == "llm"
    assert "응답을 확정하지 못했습니다" in r.answer
    assert r.grounded is False
    assert r.tool_calls  # 상한 초과여도 축적된 북키핑은 리셋되지 않는다


async def test_lg_multiturn_history_carries_context(monkeypatch):
    """dict 이력이 LangChain 메시지로 변환되어 전달되고, 직전 사용자 발화의
    수치는 검증에서 '전제(question_origin)'로 라벨링된다."""
    seen = {}

    class _Peek(_ScriptedChat):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
            seen["n_messages"] = len(messages)
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    _use_backend(monkeypatch, _Peek(steps=[
        AIMessage(content="네, 전제하신 보고 기한 30일 기준으로는 그렇습니다."),
    ]))
    r = await RaAgent().chat(
        "그럼 그 기한 안에 어떻게 하나요?",
        history=[
            {"role": "user", "content": "보고 기한이 30일 맞지?"},
            {"role": "assistant", "content": [{"type": "text", "text": "확인해 보겠습니다."}]},
        ],
    )
    assert r.mode == "llm"
    assert seen["n_messages"] >= 3  # system + 이력 + 현재 질문이 실제로 모델에 전달됐다
    assert "30일" in r.verification["question_origin"], r.verification


_DDG_HTML = """
<div class="result">
  <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fnews%2F1&amp;rut=abc">MFDS 이상사례 보고 고시 개정</a>
  <a class="result__snippet" href="#">개정 고시는 2026-05-01 시행 예정이다.</a>
</div>
"""


async def test_lg_web_search_refused_without_explicit_request(monkeypatch):
    """명시 요청 없는 턴의 search_web 호출은 래퍼에서 에러 계약으로 거부된다 —
    도구 목록을 서버 그대로 감싸므로(direct 의 목록 필터와 다른 봉합선) 실행
    관문이 같은 계약('미요청 턴에 웹 검색 없음')을 지킨다."""
    import src.websearch as websearch

    def _never(q):
        raise AssertionError("명시 요청 없는 턴에 인터넷 검색이 수행됐다")

    monkeypatch.setattr(websearch, "_fetch_html", _never)
    _use_backend(monkeypatch, _ScriptedChat(steps=[
        _tool_call("search_web", {"query": "이상사례 최신 동향"}),
        AIMessage(content="사내 규정문서 기준으로는 근거를 찾지 못했습니다."),
    ]))
    r = await RaAgent().chat("이상사례 최신 동향 알려줘")   # 웹 요청 구문 없음
    assert r.mode == "llm"
    assert r.web_results == []
    assert r.grounded is False                # 거부(에러 계약)는 근거가 아니다
    assert r.tool_calls and r.tool_calls[0].name == "search_web"  # 시도는 기록된다


async def test_lg_web_results_are_separated_when_requested(monkeypatch):
    """명시 요청 턴 — 웹 출력은 web_texts/web_results 로만 흐르고 사내 신뢰
    소스·grounded 를 오염시키지 않는다(direct/PydanticAI 와 같은 분리 계약)."""
    import src.websearch as websearch

    monkeypatch.setattr(websearch, "_fetch_html", lambda q: _DDG_HTML)
    _use_backend(monkeypatch, _ScriptedChat(steps=[
        _tool_call("search_web", {"query": "이상사례 보고 고시 개정 뉴스"}),
        AIMessage(content="🌐 인터넷 검색 결과에 따르면 개정 고시는 2026-05-01 시행 예정입니다."),
    ]))
    r = await RaAgent().chat("이상사례 보고 고시 개정 소식 웹에서 검색해줘")
    assert r.mode == "llm"
    assert r.web_results and r.web_results[0]["origin"] == "web"
    assert r.citations == [] and r.grounded is False
    assert r.verification["ok"], r.verification
    assert "2026-05-01" in r.verification["web_origin"]
