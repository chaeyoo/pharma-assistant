"""RA·PV 어시스턴트의 LangGraph 백엔드 (AGENT_BACKEND=langgraph 선택 시).

기본 백엔드(agent.py `_chat_llm`)가 anthropic SDK 로 직접 구현한 tool-use
루프를, 같은 계약 그대로 LangGraph 상태 그래프 위에서 실행하는 세 번째
병렬 구현이다(direct ↔ PydanticAI ↔ LangGraph). 모든 백엔드가 항상 같은
것을 보장해야 한다:

  - 도구는 MCP 를 통해 호출한다 — direct 백엔드와 같은 헬퍼(_mcp_client)로
    MCP_SERVER_URL(HTTP)에 붙고, 서버의 도구 이름·설명·스키마를 그대로
    LangChain `StructuredTool` 로 감싼다(스키마 이중 유지 없음).
  - 도구 호출 1건마다의 북키핑(신뢰 소스·출처·grounded·이력 허용 집합)은
    도구 래퍼에서 direct 루프 본문과 같은 헬퍼로 수행한다.
  - 모든 답변은 direct 백엔드와 동일 인자로 `_finalize()` 검증 게이트를 지난다.

의도된 차이(프레임워크의 계산 방식): direct 루프는 '왕복 6회'를 한 카운터로
세지만, 그래프에서는 recursion_limit(agent 노드+tools 노드 = 왕복당 슈퍼스텝
2)이 상한을 맡는다 — 상한이 존재한다는 계약은 같고, 초과 시 사용자 문구도
같다. 도구 실패의 자가 정정 되먹임은 ToolNode 의 에러→ToolMessage 변환이
대신한다(direct 루프의 is_error=True 되먹임과 동치).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .. import config
from ..observability import Trace, flow, timed
from .agent import (
    SYSTEM_PROMPT,
    AgentResult,
    ToolCall,
    _collect_citations,
    _finalize,
    _has_evidence,
    _is_contract_error,
    _mcp_client,
    _question_context,
    _split_user_facts,
    _stringify,
    _summarize,
    _wants_web_search,
)

MAX_LLM_REQUESTS = 6  # direct 루프의 6-step 상한과 같은 역할(모델 요청 횟수 상한)
# ReAct 왕복 1회 = agent 노드(모델 요청) + tools 노드 = 슈퍼스텝 2 →
# 모델 요청 6회를 넘기려면 recursion_limit 을 넘겨야 하는 값으로 환산한다.
_RECURSION_LIMIT = MAX_LLM_REQUESTS * 2


@dataclass
class LgDeps:
    """per-run 가변 상태 — direct 루프의 '증거 바구니' 지역변수들과 1:1 대응.

    PydanticAI 백엔드의 PaDeps 와 같은 바구니다. LangGraph 에서는 루프가
    그래프 안에 있으므로 도구 래퍼 클로저로 같은 바구니를 흘려보낸다.
    """

    trace: Trace
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_search_results: list[dict] = field(default_factory=list)
    trusted_texts: list[str] = field(default_factory=list)
    user_facts: list[str] = field(default_factory=list)
    web_texts: list[str] = field(default_factory=list)     # 웹 검색 출력(검증 3계층)
    web_results: list[dict] = field(default_factory=list)  # 웹 결과 카드(분리 노출)
    history_doc_ids: set[str] = field(default_factory=set)
    grounded_evidence: bool = False
    # 이 턴에 사용자가 인터넷 검색을 명시 요청했는가 — direct 백엔드는 도구
    # 목록에서 search_web 을 빼는 방식으로 강제하지만, 여기서는 MCP 서버의
    # 도구 목록을 그대로 감싸므로 실행 관문(래퍼)에서 막는다(PydanticAI 훅과
    # 같은 봉합선 — '미요청 턴에 웹 검색이 수행되지 않는다'는 계약은 같다).
    web_requested: bool = False


def _normalize(data: Any) -> Any:
    """call_tool 반환(.data)을 direct 백엔드와 동형인 dict/list 로 정규화."""
    if isinstance(data, (list, tuple)) and len(data) == 1 and isinstance(data[0], dict):
        return data[0]
    return data


def _bookkeep(deps: LgDeps, name: str, tool_args: dict[str, Any], data: Any) -> None:
    """도구 성공 출력 1건의 북키핑 — direct 루프(agent.py _chat_llm)와 동일."""
    if not _is_contract_error(data):
        # 성공 출력만 신뢰 소스로. 입력 에코(query·as_of)는 버리고 케이스
        # 서술(case)은 별도 바구니(user_facts)로 — 근거 승격 구멍 차단.
        # 웹 검색 출력은 사내 신뢰 소스가 아니라 웹 계층(web_texts)으로만.
        stripped, facts = _split_user_facts(data)
        if name == "search_web":
            deps.web_texts.append(_stringify(stripped))
            if isinstance(stripped, dict):
                deps.web_results.extend(stripped.get("results", []))
        else:
            deps.trusted_texts.append(_stringify(stripped))
            deps.user_facts.extend(facts)
            if _has_evidence(stripped):  # 결과 0건짜리 '성공'은 근거로 안 친다
                deps.grounded_evidence = True
    if isinstance(data, dict):
        if name == "search_regulations" and "results" in data:
            deps.raw_search_results.append(data)
            if tool_args.get("include_superseded") or tool_args.get("as_of"):
                # 이력 검색이 '실제로 반환한' 문서만 폐지본 경고 면제(문서 단위)
                deps.history_doc_ids |= {
                    r["doc_id"] for r in data["results"] if r.get("doc_id")
                }
        elif name in ("assess_adverse_event", "draft_ae_report") and data.get("basis"):
            deps.raw_search_results.append(data["basis"])  # 판정의 근거 규정도 출처로
    deps.tool_calls.append(ToolCall(name, dict(tool_args), _summarize(data)))


def _make_tools(client, mcp_tools, deps: LgDeps) -> list:
    """MCP 서버의 도구 목록을 북키핑 래퍼가 달린 LangChain 도구로 변환.

    스키마·설명은 MCP 서버가 준 것을 그대로 쓴다 — direct 백엔드의
    _to_anthropic_tools, PydanticAI 의 MCPToolset 과 같은 '단일 출처' 원칙.
    """
    from langchain_core.tools import StructuredTool

    def _runner(name: str):
        async def _run(**tool_args):
            if name == "search_web" and not deps.web_requested:
                # 명시 요청 없는 턴의 인터넷 검색은 실행하지 않는다 — 에러
                # 계약으로 되먹여 모델이 자가 정정하게 한다(신뢰 소스 미축적).
                refusal = {
                    "error": "인터넷 검색은 사용자가 명시적으로 요청한 턴에만 사용할 수 있음",
                    "expected": "사내 도구로 조회하고, 근거가 없으면 없다고 답할 것",
                }
                deps.tool_calls.append(ToolCall(name, dict(tool_args), _summarize(refusal)))
                return json.dumps(refusal, ensure_ascii=False)
            flow(
                "langgraph._run()",
                "LangGraph ToolNode 의 도구 실행 요청 — 실행은 HTTP MCP 서버가 대신한다",
                tool=name, args=dict(tool_args),
                next="call_tool() — 성공 출력은 신뢰 소스로 축적, 실패는 ToolMessage 로 모델에 되먹임",
            )
            try:
                with timed(deps.trace, f"tool.{name}", "tool", {"args": dict(tool_args)}):
                    data = _normalize((await client.call_tool(name, tool_args)).data)
            except Exception as e:
                # 실패도 UI 표시용으로 기록하고 그대로 재-raise 한다 — ToolNode 가
                # 에러 문구를 ToolMessage 로 모델에 되먹여 자가 정정시킨다
                # (direct 루프의 is_error=True 되먹임과 동치).
                deps.tool_calls.append(
                    ToolCall(name, dict(tool_args), f"오류: 도구 '{name}' 실행 실패: {type(e).__name__}: {e}")
                )
                raise
            _bookkeep(deps, name, tool_args, data)
            # 모델에게는 JSON 문자열로 — direct 백엔드가 tool_result 블록에 싣는
            # 직렬화 표현과 같은 정보량(한국어 보존: ensure_ascii=False).
            return json.dumps(data, ensure_ascii=False, default=str)

        return _run

    return [
        StructuredTool(
            name=t.name,
            description=t.description or "",
            args_schema=t.inputSchema or {"type": "object", "properties": {}},
            coroutine=_runner(t.name),
        )
        for t in mcp_tools
    ]


def _build_model():
    """실행 시점에만 모델을 만든다 — 키 없이도 모듈 import 가 가능해야 하고,
    테스트는 이 봉합선을 스크립트 모델 스텁으로 바꿔 끼운다."""
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=config.LLM_MODEL,
        api_key=config.ANTHROPIC_API_KEY,
        # direct 백엔드와 같은 생성 상한 — 두 백엔드가 같은 길이 예산을 갖는다.
        max_tokens=config.LLM_MAX_TOKENS,
    )


def _build_graph(model, tools):
    """ReAct 상태 그래프 — agent(모델) ⇄ tools(ToolNode) 순환.

    prebuilt create_react_agent 대신 명시적 StateGraph 로 조립한다 — 노드·엣지
    가 보이는 형태가 이 백엔드의 존재 이유(그래프 기반 워크플로 실증)에 맞다.
    """
    from langgraph.graph import START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition

    bound = model.bind_tools(tools)

    async def agent_node(state: MessagesState):
        return {"messages": [await bound.ainvoke(state["messages"])]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent_node)
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "agent")
    # 마지막 모델 응답에 tool_calls 가 있으면 tools 로, 없으면 END 로
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    return g.compile()


def _to_lc_messages(history: list[dict]) -> list:
    """anthropic 스타일 role/content dict 이력 → LangChain 메시지.

    텍스트만 이월한다(direct 백엔드도 이력의 텍스트 표기만 실질 사용) —
    도구 블록 등 비텍스트 표기는 이전 턴의 내부 산출물이라 재전송하지 않는다.
    """
    from langchain_core.messages import AIMessage, HumanMessage

    out = []
    for turn in history:
        content = turn.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ).strip()
        else:
            continue
        if not text:
            continue
        out.append(HumanMessage(content=text) if turn.get("role") == "user" else AIMessage(content=text))
    return out


def _msg_text(msg) -> str:
    """모델 최종 메시지의 텍스트 표현 — content 가 블록 리스트일 수 있다."""
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ).strip()
    return str(content).strip()


async def chat_llm_langgraph(message: str, history: list[dict], trace: Trace) -> AgentResult:
    """LangGraph 백엔드의 LLM 모드 진입점 — 반환 계약은 `_chat_llm` 과 동일.

    PII 마스킹은 호출부(RaAgent.chat)에서 이미 끝났고, 여기서는 그래프 실행과
    북키핑·검증 게이트만 담당한다. mode 는 "llm" 그대로 — API/UI 는 백엔드를
    구분하지 않는다(배너·trace 로만 노출).
    """
    deps = LgDeps(trace=trace, web_requested=_wants_web_search(message))
    flow(
        "chat_llm_langgraph()",
        "LangGraph 백엔드 시작 — StateGraph(agent⇄ToolNode)+recursion_limit 로 tool-use 루프를 그래프에 위임",
        model=config.LLM_MODEL, recursion_limit=_RECURSION_LIMIT,
        next="graph.ainvoke() — 도구 호출마다 StructuredTool 래퍼가 신뢰 소스를 축적한다",
    )
    grounded = False
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langgraph.errors import GraphRecursionError

        model = _build_model()
        # 호출마다 클라이언트를 새로 연다 — direct 백엔드와 동형(세션 공유 없음).
        async with _mcp_client() as client:
            tools = _make_tools(client, await client.list_tools(), deps)
            graph = _build_graph(model, tools)
            msgs = [
                SystemMessage(content=SYSTEM_PROMPT),
                *_to_lc_messages(history),
                HumanMessage(content=message),
            ]
            try:
                with timed(trace, "llm.run", "llm", {"model": config.LLM_MODEL, "backend": "langgraph"}):
                    state = await graph.ainvoke(
                        {"messages": msgs}, config={"recursion_limit": _RECURSION_LIMIT}
                    )
                answer = _msg_text(state["messages"][-1])
                grounded = deps.grounded_evidence
                flow(
                    "chat_llm_langgraph()",
                    "LangGraph 실행 완료 — 답변 확정, 사후 검증 게이트로",
                    answer_len=len(answer), tool_calls=len(deps.tool_calls), grounded=grounded,
                    next="_finalize() — direct 백엔드와 동일 인자로 답변 속 수치·인용을 신뢰 소스와 대조",
                )
            except GraphRecursionError:
                # 슈퍼스텝 상한 초과 — direct 루프가 6바퀴를 다 쓴 경우와 같은
                # 문구로 마감. deps 에 쌓인 증거(출처·신뢰 소스)는 그대로 반환.
                answer = "(도구 호출이 반복되어 응답을 확정하지 못했습니다. 질문을 좁혀 다시 시도해 주세요.)"
                flow(
                    "chat_llm_langgraph()",
                    "recursion_limit 초과(GraphRecursionError) — 확정 실패 안내로 마감(direct 6-step 상한과 같은 계약)",
                    tool_calls=len(deps.tool_calls),
                    next="_finalize() — 실패 안내문도 검증 게이트는 통과시켜 반환한다",
                )
    except Exception as e:  # noqa: BLE001 - 외부 API 실패는 명시적 안내로 흡수
        # direct 백엔드와 같은 규율: 예외 '타입명'만 싣고 메시지 원문은 싣지 않는다
        # (외부 라이브러리 에러 문구에는 요청 내용이 에코될 수 있다).
        answer = (
            f"(LLM API 호출에 실패했습니다: {type(e).__name__}. "
            "ANTHROPIC_API_KEY 가 유효한지, 네트워크와 LLM_MODEL 설정을 "
            "확인해 주세요. MCP 서버 연결 실패일 수도 있습니다 — MCP_SERVER_URL 과 "
            "MCP 서버 프로세스 기동 상태도 확인해 주세요. 키를 비우면 오프라인 모드로 동작합니다.)"
        )
        flow(
            "chat_llm_langgraph()",
            "LLM API 호출 실패 — 크래시(HTTP 500) 대신 확인 안내문으로 응답(원인 타입만 노출)",
            error=type(e).__name__,
            next="_finalize() — 실패 안내문도 검증 게이트는 통과시켜 반환한다",
        )
    return _finalize(
        AgentResult(
            answer=answer,
            mode="llm",
            tool_calls=deps.tool_calls,
            citations=_collect_citations(deps.tool_calls, deps.raw_search_results),
            grounded=grounded,
            web_results=deps.web_results,
        ),
        deps.trusted_texts,
        question=_question_context(message, history),
        allowed_superseded_ids=deps.history_doc_ids,
        user_facts=deps.user_facts,
        web_texts=deps.web_texts,
    )
