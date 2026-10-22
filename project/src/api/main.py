"""FastAPI 백엔드 — RA·PV 어시스턴트 웹 서비스.

엔드포인트:
  GET  /            → 웹 챗 UI (single page)
  GET  /health      → 실행 모드/MCP 서버·인덱스 상태 + 검증 게이트 경고율 계기판
  POST /chat        → 사용자 메시지 → 에이전트 응답(+도구호출·출처)
  POST /chat/stream → 같은 처리를 SSE 로 — 진행 단계(검색 서버 연결·도구·답변 생성·검증) 실시간 + 마지막 result
  GET  /warmup      → 검색 서버(MCP) 예열 — 챗 페이지 로드 시 호출해 콜드스타트를 타이핑 시간 뒤로 숨긴다
  GET  /api/deadlines → 대시보드용 마감일 (부가)
  GET  /dashboard   → 운영 계기판 — /health 를 폴링해 경고율·축별 분해를 그리는 화면
  GET  /model-compare/report → 모델 비교 실험 보고서(eval/model_compare/REPORT.md) 원문
  GET  /dictionary  → 용어 사전 플래시카드 (description/dictionary.html, self-contained)
  GET  /blank       → 예제문100 빈칸 퀴즈 (description/blank.html, self-contained)
  GET  /login · /auth/google · /auth/callback · /logout · /api/me → Google 로그인(src/api/auth.py)

인증: GOOGLE_CLIENT_ID 가 설정되면 챗 UI·/chat·/api/deadlines·/dashboard 는 로그인이
필요하고, /chat 은 사용자별 하루 질문 상한(DAILY_CHAT_LIMIT)을 적용한다. /health 는
플랫폼 헬스체크용으로 공개 유지(사용자 데이터 없음).

설계 관점: 에이전트(백엔드)를 API로 서빙하고 프론트(챗 UI)와 연결하는
전형적인 풀스택 구조. 실제 배포 시 인증·로깅·관측성을 이 계층에 추가한다.
"""
from __future__ import annotations

import asyncio
import json
import secrets
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel, Field

import httpx

from .. import config
from ..agent.agent import RaAgent
from ..observability import flow, flow_reset, gate_stats, progress_scope
from ..ra.tasks import load_ra_tasks
from . import auth

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
# 서명된 세션 쿠키 — 로그인이 꺼져 있으면 쓰이지 않으므로 프로세스별 임시 키로 둔다
# (로그인을 켜면 preflight 가 SESSION_SECRET 을 강제한다). same_site=lax 는 Google 에서
# 콜백으로 돌아올 때 OAuth state 쿠키가 실려 오게 하는 최소 설정이다.
app.add_middleware(
    SessionMiddleware,
    secret_key=config.SESSION_SECRET or secrets.token_urlsafe(48),
    session_cookie="pa_session",
    same_site="lax",
    https_only=config.SESSION_COOKIE_SECURE and bool(config.GOOGLE_CLIENT_ID),
    max_age=60 * 60 * 24 * 7,
)
app.include_router(auth.router)


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
    # 오늘 남은 질문 수(로그인 꺼짐·무제한이면 None)
    remaining_today: int | None = None


MODEL_COMPARE_DIR = config.BASE_DIR / "eval" / "model_compare"


def _model_compare_summary() -> dict:
    """최근 비교 실험 집계 — 파일이 없거나 깨졌으면 available=False 로 드러낸다(조용한 빈 표 금지)."""
    path = MODEL_COMPARE_DIR / "summary.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"available": False, "report_url": "/model-compare/report"}
    return {
        "available": True,
        "generated_at": data.get("generated_at"),
        "judge_model": data.get("judge_model"),
        "rows": data.get("rows", []),
        "report_url": "/model-compare/report",
    }


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
            # 작업별 현재 모델(config/models.json + 환경변수) — 계기판 '모델 선택' 섹션.
            "models": {
                "answer": config.LLM_MODEL,
                "judge": config.JUDGE_MODEL,
                "summary": config.SUMMARY_MODEL,
            },
            # 최근 모델 비교 실험 집계(eval/model_compare/summary.json) — 없으면 available=False.
            "model_compare": _model_compare_summary(),
        }
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, user: dict = Depends(auth.require_user)) -> ChatResponse:
    remaining = auth.consume_chat_quota(user)
    flow_reset()
    flow(
        "chat()",
        "HTTP 진입점 — UI(web/index.html)의 POST /chat 수신. 원문은 아직 마스킹 전이라 길이만 기록",
        message_len=len(req.message), history_turns=len(req.history),
        user=user.get("email"),
        next="agent.chat() 호출 — 모든 요청은 에이전트 단일 진입점(RaAgent)으로 위임",
    )
    result = await agent.chat(req.message, req.history)
    flow(
        "chat()",
        "응답 직렬화 — AgentResult 를 ChatResponse(JSON)로 바꿔 UI 에 반환(여기서 요청 끝)",
        mode=result.mode, grounded=result.grounded,
        citations=len(result.citations), latency_ms=result.latency_ms,
    )
    return _to_chat_response(result, remaining)


def _to_chat_response(result, remaining: int | None) -> ChatResponse:
    """AgentResult → ChatResponse. /chat(동기)과 /chat/stream(마지막 이벤트)이 같은 모양을 쓴다."""
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
        remaining_today=remaining,
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, user: dict = Depends(auth.require_user)) -> StreamingResponse:
    """/chat 과 같은 처리를 하되 진행 단계를 SSE 로 흘려보낸다.

    이벤트(`data: {json}`)는 observability.timed() 가 내보내는 start/end(이름·kind·ms)와
    _finalize 의 verify 단계이고, 마지막 이벤트 `{"type":"result", ...}` 가 /chat 의 JSON 과
    같은 본문이다. 오류는 `{"type":"error"}` 로 끝난다. 인증·쿼터는 /chat 과 동일하며
    스트림이 열리기 전에 판정되므로 401/429 는 HTTP 상태 그대로 나간다.

    왜 별도 엔드포인트인가: 검색 서버(MCP)가 콜드스타트 중이면 첫 질문이 연결에서 수십 초
    걸린다 — 동기 /chat 은 그동안 아무 신호도 못 주지만, 여기서는 'mcp.connect' 시작
    이벤트가 즉시 나가 UI 가 '검색 서버 깨우는 중'을 보여 줄 수 있다.
    """
    remaining = auth.consume_chat_quota(user)
    flow_reset()
    flow(
        "chat_stream()",
        "HTTP 진입점(SSE) — UI 의 POST /chat/stream 수신. 진행 이벤트를 흘리며 agent.chat() 을 수행",
        message_len=len(req.message), history_turns=len(req.history), user=user.get("email"),
        next="agent.chat() 을 task 로 띄우고 timed() 이벤트 큐를 소비해 data: 라인으로 전송",
    )
    queue: asyncio.Queue = asyncio.Queue()
    _DONE = object()

    async def _run():
        try:
            result = await agent.chat(req.message, req.history)
            payload = _to_chat_response(result, remaining).model_dump()
            queue.put_nowait({"type": "result", **payload})
        except Exception as e:  # noqa: BLE001 - 스트림은 HTTP 500 을 못 쓰므로 이벤트로 알린다
            queue.put_nowait({"type": "error", "message": f"서버 오류 ({type(e).__name__})"})
        finally:
            queue.put_nowait(_DONE)

    with progress_scope(queue):
        task = asyncio.create_task(_run())  # ContextVar 는 task 생성 시점에 복사된다

    async def _events():
        try:
            while True:
                ev = await queue.get()
                if ev is _DONE:
                    break
                yield "data: " + json.dumps(ev, ensure_ascii=False, default=str) + "\n\n"
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        _events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/warmup")
async def warmup() -> JSONResponse:
    """검색 서버(MCP)를 미리 깨운다 — 챗 페이지가 열리는 순간 호출한다.

    Cloud Run 의 MCP 인스턴스는 요청이 없으면 0대로 내려가고, 다음 요청이 콜드스타트
    (모델 로드·색인, 수십 초)를 유발한다. 사용자가 질문을 읽고 타이핑하는 동안 그 비용을
    미리 치르게 하려고, /health 의 3초 제한 대신 긴 제한으로 MCP /health 가 200 을 줄 때까지
    기다렸다가 소요 시간을 돌려준다. 이미 떠 있으면 즉시 ok. 과금은 어차피 첫 질문이
    유발했을 기동을 앞당기는 것이라 추가 비용이 없다.
    """
    t0 = time.perf_counter()
    info = await _mcp_status()
    if info.get("status") != "ok":
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.get(config.MCP_SERVER_HEALTH_URL)
                r.raise_for_status()
                info = r.json()
        except Exception as e:  # noqa: BLE001 - 도달 불가는 상태로 표현
            info = {"status": "unreachable", "error": type(e).__name__}
    return JSONResponse({
        "status": info.get("status", "unreachable"),
        "ms": round((time.perf_counter() - t0) * 1000),
        "rag": info.get("rag"),
    })


@app.get("/api/deadlines")
async def deadlines(user: dict = Depends(auth.require_user)) -> JSONResponse:
    return JSONResponse(load_ra_tasks()["deadlines"])


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if not auth.current_user(request):
        return RedirectResponse("/login")
    html = (config.WEB_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    if not auth.current_user(request):
        return RedirectResponse("/login")
    # /health 의 JSON 을 사람이 읽는 화면 — 폴링만 하는 정적 단일 파일(외부 의존 없음).
    # 지표의 정본은 여전히 /health 이고 이 페이지는 표현 계층일 뿐이다:
    # 새 축이 생겨도 서버가 by_axis 에 실어 보내면 라벨 없이 키 그대로 그려진다
    # (계기판에서 축이 조용히 빠지는 실패를 만들지 않는다 — observability.py 와 같은 규율).
    html = (config.WEB_DIR / "dashboard.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/model-compare/report")
async def model_compare_report(request: Request):
    """모델 비교 실험 보고서(eval/model_compare/REPORT.md) 원문 — 계기판 링크 대상.
    /dashboard 와 같은 로그인 게이트. 실험 미실행이면 404(빈 보고서를 지어내지 않는다)."""
    if not auth.current_user(request):
        return RedirectResponse("/login")
    path = MODEL_COMPARE_DIR / "REPORT.md"
    if not path.exists():
        return JSONResponse({"error": "모델 비교 실험 보고서가 아직 없다(python -m eval.model_compare.run)"}, status_code=404)
    from fastapi.responses import PlainTextResponse

    return PlainTextResponse(path.read_text(encoding="utf-8"), media_type="text/markdown; charset=utf-8")


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
