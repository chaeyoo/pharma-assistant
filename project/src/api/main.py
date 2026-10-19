"""FastAPI 백엔드 — RA·PV 어시스턴트 웹 서비스.

엔드포인트:
  GET  /            → 웹 챗 UI (single page)
  GET  /health      → 실행 모드/MCP 서버·인덱스 상태 + 검증 게이트 경고율 계기판
  POST /chat        → 사용자 메시지 → 에이전트 응답(+도구호출·출처)
  GET  /api/deadlines → 대시보드용 마감일 (부가)
  GET  /dashboard   → 운영 계기판 — /health 를 폴링해 경고율·축별 분해를 그리는 화면
  GET  /dictionary  → 용어 사전 플래시카드 (description/dictionary.html, self-contained)
  GET  /blank       → 예제문100 빈칸 퀴즈 (description/blank.html, self-contained)

설계 관점: 에이전트(백엔드)를 API로 서빙하고 프론트(챗 UI)와 연결하는
전형적인 풀스택 구조. 실제 배포 시 인증·로깅·관측성을 이 계층에 추가한다.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

import httpx

from .. import config
from ..agent.agent import RaAgent
from ..observability import flow, flow_reset, gate_stats
from ..ra.tasks import load_ra_tasks

agent = RaAgent()


async def _mcp_status() -> dict:
    """분리된 MCP 서버의 /health 를 조회한다(3초 타임아웃).

    RAG 인덱스는 이제 MCP 서버 프로세스가 소유한다 — API 는 상태를 조회만 한다.
    도달 불가는 예외 대신 상태 dict 로 흡수한다(fail-closed 게이트는
    preflight --role api 가 담당, 여기서는 관측만).
    """
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(config.MCP_SERVER_HEALTH_URL)
            r.raise_for_status()
            return r.json()
    except Exception as e:  # noqa: BLE001 - 도달 불가는 상태로 표현
        return {"status": "unreachable", "error": type(e).__name__}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 완전 분리: RAG 인덱스는 MCP 서버 프로세스가 부팅 시 구축한다.
    # API 는 연결 상태만 확인해 로그로 남긴다(비치명 — 기동 순서 게이트는 preflight).
    info = await _mcp_status()
    app.state.mcp_info = info
    print(config.mode_banner())
    if info.get("status") == "ok":
        rag = info.get("rag", {})
        print(f"[MCP] {config.MCP_SERVER_URL} 연결 확인 · 문서 {rag.get('docs')}건 · 청크 {rag.get('chunks')}개")
    else:
        print(f"[MCP] {config.MCP_SERVER_URL} 도달 불가({info.get('error')}) — degraded 상태로 기동")
    yield


app = FastAPI(title="RAPV-Assistant", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str = Field(..., description="사용자 질문")
    history: list[dict] = Field(default_factory=list, description="이전 대화(LLM 모드)")


class ChatResponse(BaseModel):
    answer: str
    mode: str
    tool_calls: list[dict]
    citations: list[dict]
    grounded: bool = True
    trace: list[dict] = Field(default_factory=list)
    latency_ms: float = 0.0
    redactions: list[dict] = Field(default_factory=list)  # PII 마스킹 내역(유형·건수만)
    verification: dict = Field(default_factory=dict)  # 답변 사후 검증(수치 대조·버전 점검)
    # 인터넷 검색 결과(origin="web") — 사내 출처(citations)와 별도 필드로 분리
    # (명시 요청 턴에만 채워진다 — 웹 결과가 규제문서 출처 카드에 섞이지 않는다)
    web_results: list[dict] = Field(default_factory=list)


@app.get("/health")
async def health() -> JSONResponse:
    mcp_info = await _mcp_status()
    return JSONResponse(
        {
            # MCP 서버 도달 불가면 degraded — 라우팅(200)은 유지하되 상태를 드러낸다.
            "status": "ok" if mcp_info.get("status") == "ok" else "degraded",
            "mode": "llm" if config.LLM_AVAILABLE else "offline",
            "banner": config.mode_banner(),
            "rag": mcp_info.get("rag", {}),
            "mcp": {"url": config.MCP_SERVER_URL, "status": mcp_info.get("status")},
            "rag_params": {
                "chunk_size": config.CHUNK_SIZE,
                "chunk_overlap": config.CHUNK_OVERLAP,
                "retrieve_top_k": config.RETRIEVE_TOP_K,
                "rerank_top_n": config.RERANK_TOP_N,
                "hybrid_alpha": config.HYBRID_ALPHA,
            },
            # 검증 게이트 운영 계기판 — 경고율이 오르면 (a) 답변 품질 회귀 또는
            # (b) 검증기 오탐 증가(alert fatigue 위험)의 조기 신호다.
            "verification_gate": gate_stats.snapshot(),
        }
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    flow_reset()
    flow(
        "chat()",
        "HTTP 진입점 — UI(web/index.html)의 POST /chat 수신. 원문은 아직 마스킹 전이라 길이만 기록",
        message_len=len(req.message), history_turns=len(req.history),
        next="agent.chat() 호출 — 모든 요청은 에이전트 단일 진입점(RaAgent)으로 위임",
    )
    result = await agent.chat(req.message, req.history)
    flow(
        "chat()",
        "응답 직렬화 — AgentResult 를 ChatResponse(JSON)로 바꿔 UI 에 반환(여기서 요청 끝)",
        mode=result.mode, grounded=result.grounded,
        citations=len(result.citations), latency_ms=result.latency_ms,
    )
    return ChatResponse(
        answer=result.answer,
        mode=result.mode,
        tool_calls=[
            {"name": t.name, "args": t.args, "summary": t.result_summary}
            for t in result.tool_calls
        ],
        citations=result.citations,
        grounded=result.grounded,
        trace=result.trace,
        latency_ms=result.latency_ms,
        redactions=result.redactions,
        verification=result.verification,
        web_results=result.web_results,
    )


@app.get("/api/deadlines")
async def deadlines() -> JSONResponse:
    return JSONResponse(load_ra_tasks()["deadlines"])


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html = (config.WEB_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    # /health 의 JSON 을 사람이 읽는 화면 — 폴링만 하는 정적 단일 파일(외부 의존 없음).
    # 지표의 정본은 여전히 /health 이고 이 페이지는 표현 계층일 뿐이다:
    # 새 축이 생겨도 서버가 by_axis 에 실어 보내면 라벨 없이 키 그대로 그려진다
    # (계기판에서 축이 조용히 빠지는 실패를 만들지 않는다 — observability.py 와 같은 규율).
    html = (config.WEB_DIR / "dashboard.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/dictionary", response_class=HTMLResponse)
async def dictionary() -> HTMLResponse:
    # dictionary.md 원문이 mdsrc 블록에 주입된 정적 단일 파일(외부 의존 없음).
    # 갱신은 description/build_dictionary_html.py 실행 후 재배포.
    html = config.DICTIONARY_HTML.read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/blank", response_class=HTMLResponse)
async def blank_quiz() -> HTMLResponse:
    # 예제문100(dictionary.md 말미 섹션)을 빈칸 퀴즈로 푸는 정적 단일 파일.
    # 문장을 고치면 blank.html 의 DATA 배열도 함께 갱신해야 한다.
    html = config.BLANK_HTML.read_text(encoding="utf-8")
    return HTMLResponse(html)
