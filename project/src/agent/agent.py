"""RA·PV 어시스턴트 에이전트 (Agentic Workflow + Function Calling).

에이전트가 MCP 서버의 도구를 자율적으로 호출해 사용자의 규제업무 질문에 답한다.

두 가지 실행 경로 (동일 인터페이스):
  - LLM 모드 (ANTHROPIC_API_KEY 있음): 실제 Claude가 tool-use 루프로
    "관찰→생각→도구호출→관찰" 을 반복하며 스스로 도구를 선택·조합한다. (진짜 Agentic)
  - 오프라인 모드 (키 없음): 규칙 기반 라우터가 적절한 MCP 도구를 호출하고,
    검색 근거에 기반한 추출형 답변을 조립한다. (환각 없는 grounded 답변)

두 경로 모두 '도구는 MCP를 통해' 호출한다 — 모델과 도구가 MCP 규격으로 분리되어 있다.
"""
from __future__ import annotations

import re
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from fastmcp import Client

from .. import config
from ..observability import Trace, flow, record_verification, timed
from ..pv.coding import symptom_keywords
from ..pv.redactor import redact
from ..rag.synonyms import expand_query
from ..rag.textutil import tokenize
from ..verify.verifier import verify_answer, warning_text

# 근거 부족(abstention) 판단 임계값 — 제약 규제 도메인의 환각 안전장치.
# 두 신호를 함께 본다: (1) 최상위 근거의 검색 관련도 점수, (2) 질의-근거 토큰 커버리지.
# 둘 다 문턱 아래일 때만("확실히 근거 없음") 답을 지어내지 않고 회피한다.
# AND 조건을 쓰는 이유: 콜로퀄한 범위내 질문은 점수는 낮아도 커버리지가 남아있어
# 과도한 회피(over-abstention)를 피할 수 있다.
#
# 문턱값은 범위내(qa_dataset)/범위밖(abstention_dataset) 분포를 실측해 마진
# 중앙으로 보정한 값이다(리랭커 공식이 바뀌면 점수 스케일이 바뀌므로 재보정 필수):
#   범위밖 최대  score=0.167, cov=0.250  → 이보다 위에 문턱
#   범위내 최소  score=0.156(cov 0.314) / cov=0.227(score 0.201)
#   → SCORE_FLOOR ∈ (0.167, 0.201), COVERAGE_FLOOR ∈ (0.250, 0.314)
# 리랭커 v3(섹션 타입 prior) 반영 후 재실측(32문항): 페널티는 대조/서두 섹션에만
# 작용해 경계 분포가 사실상 불변(범위내 최소 0.156/0.227, 범위밖 최대 0.167/0.250)
# — 문턱 유지. 재보정 절차 자체는 리랭커·검색 후보집합을 바꿀 때마다 반복한다.
# v10 재실측(오늘 2026-07-15): v9 가 기본 검색을 as_of=오늘 동치로 바꿔 후보집합이
# 벽시계 날짜의 함수가 됐으나, 현행 코퍼스의 전 규정 시행일이 ≤2025-06-15 라
# 오늘 이후로는 후보집합이 status-only 시절과 동일 — 경계 분포·문턱 불변(재확인).
# 단 코퍼스에 미래 시행일 규정·새 supersession 이 추가되면 이 고정 상수는 특정
# 날짜 슬라이스에 보정된 값과 어긋나므로, 그때 재보정한다(preflight 가 미래
# 시행일 '시행 중 구판 부재'를 기동 전에 잡아 이 결합을 표면화한다).
SCORE_FLOOR = 0.19       # 최상위 근거 관련도 하한
COVERAGE_FLOOR = 0.28    # (확장)질의 토큰 커버리지 하한

_MCP_UNAVAILABLE_MSG = (
    "(MCP 서버에 연결하지 못했습니다: {}. MCP_SERVER_URL 설정과 MCP 서버 프로세스"
    "(docker compose 의 mcp 서비스) 기동 상태를 확인해 주세요.)"
)


def _mcp_client() -> Client:
    """MCP 서버 HTTP 클라이언트를 만든다.

    완전 분리 구조: 프로덕션 에이전트는 항상 MCP_SERVER_URL(HTTP)로 붙는다.
    MCP_AUTH_TOKEN 이 설정돼 있으면 Bearer 헤더로 자동 부착한다(공개 서버 배포용).
    config 속성은 호출 시점에 읽는다 — 테스트 하니스가 세션 중 URL 을 주입한다.
    """
    return Client(config.MCP_SERVER_URL, auth=config.MCP_AUTH_TOKEN or None)


SYSTEM_PROMPT = (
    "당신은 제약회사 RA(인허가·규제업무)·PV(약물감시) 담당자를 돕는 어시스턴트다. "
    "반드시 제공된 MCP 도구로 사내 규제문서와 업무 데이터를 조회한 뒤, "
    "그 근거에 기반해서만 답한다. 근거가 없으면 모른다고 답하고 추측하지 않는다. "
    "규제 정보(기한·절차)는 정확성이 생명이므로 항상 출처(문서명·섹션)를 함께 제시한다. "
    "인터넷 검색(search_web)은 사용자가 명시적으로 요청한 경우에만 사용한다 — "
    "사내 문서에서 근거를 못 찾았다는 이유로 대신 쓰지 않는다(그 경우 근거 없음을 그대로 답한다). "
    "웹 검색 결과를 쓸 때는 답변을 두 구역으로 완전히 분리한다: 사내 규제문서 근거 구역을 "
    "먼저 쓰고, 웹 내용은 '🌐 인터넷 검색 결과 (사내 규제문서 아님)' 제목의 별도 구역에만 "
    "쓰며 각 항목에 출처 URL 을 붙인다. 두 구역의 근거를 한 문장·한 문단에 섞지 않는다. "
    "답변은 한국어로, 간결한 실무체로 작성한다. "
    # 웹 UI(web/index.html)는 XSS 방지를 위해 답변을 이스케이프만 하고 마크다운을
    # 렌더링하지 않는다(white-space: pre-wrap) — 기호가 화면에 그대로 노출되므로
    # 구조는 마크다운이 아니라 줄바꿈·번호로 만들게 한다.
    "마크다운 제목(#)·강조(*)·표·코드블록은 쓰지 않고, 구조가 필요하면 "
    "줄바꿈과 번호(1. 2. 3.) 또는 하이픈(-) 목록으로 표현한다."
)

# 인터넷 검색의 '명시 요청' 감지 — 닫힌 어휘·요청형 구문만 좁게 잡는다.
# "인터넷 판매 의약품 규정"처럼 질문 '주제'에 인터넷/온라인이 들어가는 경우와
# 구분해야 하므로, (인터넷|웹|온라인)+검색·찾기 '요청' 구문 또는 검색엔진
# 고유명만 매칭한다. 오탐은 외부 송신 경로를 여는 스위치의 오탐이라 이력
# 마커(_HISTORY_MARKERS)와 같은 철학으로 확신 없는 어휘는 넣지 않는다.
_WEB_REQUEST_RE = re.compile(
    r"(?:인터넷|웹|온라인)\s*(?:에서|으로|을|를)?\s*(?:검색|찾아|서치|조회)"
    r"|구글링|구글|네이버\s*검색|검색\s*엔진|웹서핑"
)


def _wants_web_search(message: str) -> bool:
    """이 턴의 사용자 메시지가 인터넷 검색을 '명시적으로' 요청했는가.

    True 일 때만 search_web 도구가 에이전트에 노출·허용된다(명시와 분리의
    '명시' 쪽 절반 — 프롬프트 지시가 아니라 도구 가용성 자체로 강제한다).
    """
    return bool(_WEB_REQUEST_RE.search(message))


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    result_summary: str


@dataclass
class AgentResult:
    answer: str
    mode: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    # 도구 근거(검색 결과·결정론 도구 출력)로 뒷받침된 답변인지. 오프라인 모드는
    # abstention 이면 False. LLM 모드는 문턱 판정이 없으므로(회피 판정은 오프라인
    # 검색 경로 전용) '성공한 도구 출력이 신뢰 소스로 확보되었는가'로 계산한다 —
    # 모델이 도구를 한 번도 부르지 않고 답하면 False(근거 보증이 아니라는 신호).
    grounded: bool = True
    trace: list[dict] = field(default_factory=list)   # 스텝별 지연·성패(관측성)
    latency_ms: float = 0.0        # 총 처리 지연
    redactions: list[dict] = field(default_factory=list)  # PII 마스킹 내역(유형·건수만)
    verification: dict = field(default_factory=dict)  # 사후 검증 결과(수치 대조·버전 점검)
    # 인터넷 검색 결과(origin="web") — 사내 출처(citations)와 '별도 필드'로
    # 분리해 내보낸다. citations 에 섞으면 UI 출처 카드에서 웹 결과가 규제문서
    # 근거처럼 읽힌다(명시와 분리 원칙의 API 계층 적용).
    web_results: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 공통: MCP 도구 스키마 → Anthropic 도구 포맷
# ---------------------------------------------------------------------------
def _to_anthropic_tools(mcp_tools) -> list[dict]:
    tools = []
    for t in mcp_tools:
        tools.append(
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.inputSchema or {"type": "object", "properties": {}},
            }
        )
    return tools


def _collect_citations(tool_calls: list[ToolCall], raw_results: list[dict]) -> list[dict]:
    """search_regulations 결과에서 출처 목록을 모은다."""
    cites: list[dict] = []
    seen = set()
    for res in raw_results:
        for r in res.get("results", []):
            key = (r.get("source"), r.get("section"))
            if key in seen:
                continue
            seen.add(key)
            cites.append(
                {
                    "doc_id": r.get("doc_id"),
                    "title": r.get("title"),
                    "source": r.get("source"),
                    "section": r.get("section"),
                    "version": r.get("version"),
                    "effective_date": r.get("effective_date"),
                    "status": r.get("status"),  # 사후 검증(폐지본 인용 감지)에 사용
                    "score": r.get("score"),
                }
            )
    return cites


_ECHO_KEYS = ("query", "as_of")  # 도구 출력에 에코되는 '사용자 입력' 필드


def _strip_query_echo(data: Any) -> Any:
    """도구 출력에서 사용자 입력 에코 필드(query·as_of)를 재귀 제거한 사본을 만든다.

    신뢰 소스는 '근거 문단 ∪ 도구 출력'인데, search 계열 도구는 받은 입력을
    출력에 에코한다. 이를 그대로 신뢰하면 **사용자 입력 속 수치·날짜가 신뢰
    소스로 승격**되어, 사용자가 틀린 값을 전제로 물었을 때 모델이 맞장구쳐도
    검증을 통과하는 구멍이 생긴다. query 만 제거하고 as_of(기준일 에코)를
    남기는 것은 같은 구멍을 반쪽만 막은 비대칭이었다 — 사용자가 지정한
    기준일이 답변 날짜 클레임의 '근거'가 되어 버린다. 케이스 서술(case)은
    남긴다 — 그것은 검증할 규정 클레임이 아니라 사용자가 제공한 '사실'이다.
    """
    if isinstance(data, dict):
        return {k: _strip_query_echo(v) for k, v in data.items() if k not in _ECHO_KEYS}
    if isinstance(data, list):
        return [_strip_query_echo(v) for v in data]
    return data


_USER_FACT_KEYS = ("case",)  # 도구 출력 중 '사용자 제공 사실' 에코 필드


def _split_user_facts(data: Any) -> tuple[Any, list[str]]:
    """도구 출력에서 입력 에코를 제거하고 사용자 사실 필드(case)를 분리한다.

    반환: (에코 제거된 사본, 사용자 사실 텍스트 목록). query·as_of 는 종전대로
    버리고(전제의 승격 차단), case 는 버리는 대신 **별도 계층**으로 분리한다 —
    검증기가 지지 근거로는 인정하되 from_case 라벨을 붙일 수 있도록.
    '케이스는 사실이므로 신뢰 소스'라는 종전 규칙에는 "사실이라는 성질이 그
    값을 규정 클레임의 근거로 승격시키지는 않는다"는 반례가 있었다(케이스의
    "30일간 복용"이 답변의 "보고 기한 30일"을 지지하는 조용한 통과).
    한계의 명시: draft_ae_report 의 draft_markdown 처럼 케이스 서술이 다른
    필드 '안에' 재조립된 경우는 이 분리가 잡지 못한다(필드 단위 분리의 경계).
    """
    facts: list[str] = []

    def _strip(d: Any) -> Any:
        if isinstance(d, dict):
            out = {}
            for k, v in d.items():
                if k in _ECHO_KEYS:
                    continue
                if k in _USER_FACT_KEYS and isinstance(v, str):
                    facts.append(v)
                    continue
                out[k] = _strip(v)
            return out
        if isinstance(d, list):
            return [_strip(v) for v in d]
        return d

    return _strip(data), facts


def _is_contract_error(data: Any) -> bool:
    """도구의 명시적 에러 계약({"error", ...}) 응답인지 판정한다.

    에러 계약은 에이전트의 자가 정정용 되먹임이지 **근거가 아니다** — 에러
    문구에는 사용자 입력이 에코된다(예: "as_of '2024/06/01' 가 … 형식이 아님").
    이를 신뢰 소스에 넣으면 잘못된 입력 값이 검증의 '근거'로 승격된다.
    query·as_of 키만 걷어내는 필드 단위 접근으로는 못 막는 경로라(값이 error
    문자열 안에 들어 있다), 에러 계약 응답은 통째로 신뢰 소스에서 제외한다.
    """
    return isinstance(data, dict) and "error" in data


def _has_evidence(data: Any) -> bool:
    """도구 출력이 '내용 있는 실질 근거'를 담았는가(v10).

    grounded 계산용 — 빈 검색 성공 결과({"results": []})처럼 성공했으나 근거가
    없는 출력을 배제한다. 검색은 results 가 비지 않았을 때만, 그 밖의 결정론
    도구(트리아지·초안·마감일·체크리스트)는 출력 dict/list 에 값이 있을 때
    근거로 센다 — 오프라인 검색 경로가 빈 결과에 grounded=False 로 강등하는
    것과 대칭. (에러·에러 계약 응답은 상류에서 이미 걸러져 여기 오지 않는다.)
    """
    if isinstance(data, dict):
        if "results" in data:
            return bool(data["results"])
        return any(v for v in data.values())
    if isinstance(data, list):
        return bool(data)
    return bool(data)


def _finalize(
    result: AgentResult,
    trusted_texts: list[str],
    allow_superseded: bool = False,
    question: str = "",
    allowed_superseded_ids: set[str] | None = None,
    user_facts: list[str] | None = None,
    web_texts: list[str] | None = None,
) -> AgentResult:
    """모든 응답이 통과하는 사후 검증 게이트 — 답변 속 수치·날짜·방향 한정어를
    신뢰 소스(검색 근거 + 결정론적 도구 출력, 질문 에코 제외)와 대조하고
    폐지본 인용을 점검한다.

    실패해도 답변을 차단하지 않는다 — 경고를 본문에 부착하고 verification
    필드로 노출한다(시끄러운 실패, 최종 확정은 사람). 오프라인 모드는 추출형이라
    통과가 정상이며, 통과 자체가 '포매터가 근거 밖 수치를 만들지 않는다'는
    회귀 가드가 된다. LLM 모드는 생성 답변이므로 이 게이트가 실질 방어선이다.
    question 을 넘기는 이유: 미확인 수치가 질문에 있던 값이면 '환각'이 아니라
    '전제 확인 필요'로 경고 문구를 조정한다(정정 답변의 오탐 완화).
    """
    if result.web_results and "🌐" not in result.answer:
        # 분리 표시의 결정론적 백스톱 — 웹 결과가 사용된 턴인데 답변 본문에
        # 🌐 구분 표시가 없으면(프롬프트는 어겨질 수 있는 규칙이고, 길이 절단
        # 으로 웹 구역이 잘려나갈 수도 있다) 표시를 덧붙인다. 본문 어느 대목이
        # 웹 유래인지는 검증의 web_origin 라벨과 하단 웹 출처 블록이 보완한다.
        result.answer += (
            "\n\n🌐 이 답변에는 인터넷 검색 결과가 사용되었습니다 — 사내 규제문서 "
            "근거가 아니므로, 하단 '인터넷 검색 출처' 목록의 원문을 반드시 확인하세요."
        )
    v = verify_answer(
        result.answer,
        trusted_texts,
        result.citations,
        allow_superseded,
        question=question,
        allowed_superseded_ids=allowed_superseded_ids,
        user_fact_texts=user_facts,
        # 웹 검색 출력은 3번째 계층 — 게이트를 '건너뛰는' 것이 아니라 계층
        # 라벨(from_web)로 분리한다: 웹 재서술 수치에 미확인 오탐이 붙지 않되
        # (검증은 계속 돌고) 사내 규정 근거로 승격되지도 않는다.
        web_texts=web_texts,
    )
    flow(
        "_finalize()",
        "사후 검증 게이트 — 답변 속 수치·날짜·방향·인용 버전을 신뢰 소스(도구 출력)와 대조. "
        "실패해도 차단하지 않고 경고를 부착(최종 확정은 사람)",
        ok=v.ok, summary=v.summary(),
        next=(
            "record_verification() 집계 후 API 계층으로 반환 — 검증 통과라 답변 원문 그대로"
            if v.ok
            else "warning_text() 를 답변 끝에 부착 후 반환 — 검증 경고가 있으므로(시끄러운 실패)"
        ),
    )
    body = result.answer  # judge 는 경고문이 붙기 전의 답변 본문만 판정한다
    if not v.ok:
        result.answer = f"{result.answer}\n\n{warning_text(v)}"
    result.verification = v.summary()
    if config.JUDGE_ENABLED:
        # LLM judge 2차 계층(opt-in) — 결정론 축 정본(canon) '밖'에 부착한다:
        # verification["judge"] 서브딕셔너리이지 새 축이 아니다. 축으로 넣으면
        # preflight 자가 테스트가 키 없는 부팅에서 그 축을 태울 수 없어 오프라인
        # 기동이 깨진다(설계 근거는 src/verify/llm_judge.py docstring). 1차 결과와
        # 무관하게 항상 돈다 — judge 의 관할(수치 없는 의미 왜곡)은 1차가 못 보는
        # 직교 축이라, 1차 통과가 judge 생략의 근거가 되지 못한다. 비차단:
        # flags 가 있어도 verification["ok"] 는 불변이고 경고문만 부착한다.
        from ..verify.llm_judge import judge_answer, judge_warning_text

        j = judge_answer(body, trusted_texts, question=question)
        result.verification["judge"] = j
        if j["flags"]:
            result.answer = f"{result.answer}\n\n{judge_warning_text(j)}"
        flow(
            "_finalize()",
            "LLM judge 2차 검증(opt-in) — 결정론 게이트 범위 밖의 의미 왜곡(부정문·주어 스왑) 판정",
            status=j["status"], flags=len(j["flags"]),
        )
    # 운영 계기판 집계 + 감사 로그 — 경고율(alert fatigue 조기 신호)을 /health 로 상시 노출
    record_verification(result.verification)
    return result


class RaAgent:
    """RA·PV 어시스턴트. chat() 하나로 두 모드를 투명하게 처리."""

    async def chat(self, message: str, history: list[dict] | None = None) -> AgentResult:
        history = history or []
        trace = Trace()
        # PII 마스킹은 '에이전트 입구'에서 — 이후의 모든 경로(외부 LLM API·검색·
        # 로그·트레이스)에 원문 개인정보가 흘러들지 않는다. 원 값은 보존하지 않는다.
        red = redact(message)
        message = red.text
        history = _redact_history(history)  # 클라이언트가 보낸 이전 턴에도 PII가 있을 수 있다
        flow(
            "RaAgent.chat()",
            "에이전트 입구 PII 마스킹 — 이후 모든 경로(LLM API·검색·로그)에는 마스킹본만 흐른다",
            masked=red.summary(), message=message,
            next="모드 분기 — 마스킹은 무조건 거치는 공통 전처리, 다음 줄에서 경로가 갈린다",
        )
        flow(
            "RaAgent.chat()",
            "모드 분기 — ANTHROPIC_API_KEY 유무로 결정(llm: Claude tool-use 루프 / offline: 규칙 라우터)",
            mode="llm" if config.LLM_AVAILABLE else "offline",
            next=(
                (
                    "chat_llm_pydantic() 호출 — API 키가 있고 AGENT_BACKEND=pydantic_ai 이므로 "
                    "PydanticAI 백엔드가 tool-use 루프를 수행한다"
                    if config.AGENT_BACKEND == "pydantic_ai"
                    else "chat_llm_langgraph() 호출 — API 키가 있고 AGENT_BACKEND=langgraph 이므로 "
                    "LangGraph 상태 그래프가 tool-use 루프를 수행한다"
                    if config.AGENT_BACKEND == "langgraph"
                    else "_chat_llm() 호출 — API 키가 있으므로 Claude 가 스스로 도구를 선택한다"
                )
                if config.LLM_AVAILABLE
                else "_chat_offline() 호출 — API 키가 없으므로 키워드 규칙 라우터가 도구를 선택한다"
            ),
        )
        with timed(trace, "chat", "agent", {"mode": "llm" if config.LLM_AVAILABLE else "offline"}):
            if config.LLM_AVAILABLE:
                if config.AGENT_BACKEND == "pydantic_ai":
                    # 병렬 백엔드: 같은 계약(_finalize 게이트·mode="llm")을
                    # PydanticAI 프레임워크 루프로 수행. 지연 import — 미설치
                    # 환경에서 기본(direct) 경로가 영향받지 않는다.
                    from .pydantic_agent import chat_llm_pydantic

                    result = await chat_llm_pydantic(message, history, trace)
                elif config.AGENT_BACKEND == "langgraph":
                    # 세 번째 병렬 백엔드: 같은 계약을 LangGraph 상태 그래프
                    # (agent⇄ToolNode)로 수행. 지연 import 는 위와 같은 이유.
                    from .langgraph_agent import chat_llm_langgraph

                    result = await chat_llm_langgraph(message, history, trace)
                else:
                    result = await self._chat_llm(message, history, trace)
            else:
                result = await self._chat_offline(message, history, trace)
        result.trace = trace.to_list()
        result.latency_ms = trace.total_ms
        result.redactions = red.summary()
        return result

    # ---- LLM 모드: 실제 tool-use 루프 ----
    async def _chat_llm(self, message: str, history: list[dict], trace: Trace) -> AgentResult:
        import anthropic

        # ── 이 함수의 뼈대 ────────────────────────────────────────────
        # "Claude 에게 물어본다 → Claude 가 '도구를 불러 달라'고 하면 대신
        # 실행해 결과를 돌려준다 → Claude 가 '답변을 확정하겠다'고 할 때까지
        # 반복한다." 나머지 코드는 이 반복을 안전하게 만드는 장치다.
        #
        # 아래 변수들은 루프가 도는 동안 '증거'를 쌓아 두는 바구니다 —
        # 생성 답변은 매번 다르므로, 마지막 _finalize() 검증에서 답변 속
        # 수치·날짜가 진짜 도구 출력에 있던 값인지 대조하려면
        # '이번 턴에 도구가 실제로 반환한 것'을 따로 모아 둬야 한다.
        client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
        tool_calls: list[ToolCall] = []       # UI 표시용: 어떤 도구를 어떤 인자로 불렀나
        raw_search_results: list[dict] = []   # 출처(citations) 카드의 재료
        trusted_texts: list[str] = []   # 검증의 '신뢰 소스' — 성공한 도구 출력(입력 에코 제외)
        # 이력(as_of·include_superseded) 검색이 '실제로 반환한' 문서 집합.
        # 이 문서들의 폐지본 인용만 검증 경고에서 면제한다 — 전역 on/off 로
        # 켜면 같은 턴의 현행 검색에 섞여 든 폐지본 인용까지 경고가 꺼지므로,
        # 안전장치를 끄는 범위는 '근거가 성립하는 문서 단위'로 좁힌다.
        history_doc_ids: set[str] = set()
        # '내용 있는 도구 결과를 받은 적이 있는가' — 최종 grounded 배지의 근거.
        # bool(trusted_texts) 로 판정하면 안 된다: 검색이 성공했지만 결과 0건
        # ({"results": []})인 출력도 trusted_texts 에는 쌓이므로, 출처가 0건인데
        # grounded=True 인 모순이 생긴다(오프라인 모드가 빈 검색을
        # grounded=False 로 강등하는 것과 대칭 — v10).
        grounded_evidence = False
        user_facts: list[str] = []      # 도구 출력 속 '사용자가 서술한 사실'(case 에코) — 별도 계층
        web_texts: list[str] = []       # 인터넷 검색 출력 — 검증의 3번째 계층(from_web 라벨용)
        web_results: list[dict] = []    # 웹 결과 카드(사내 citations 와 분리 노출)
        # 명시와 분리의 '명시': 이 턴에 인터넷 검색을 명시 요청하지 않았으면
        # search_web 은 도구 목록에서 아예 빠진다 — 프롬프트 지시(어겨질 수
        # 있는 규칙)가 아니라 도구 가용성(구조)으로 강제한다.
        web_requested = _wants_web_search(message)

        # MCP 서버에 연결한다 — 서버는 별도 프로세스(docker compose 의 mcp 서비스 /
        # Render 의 rapv-mcp)로 떠 있고, MCP_SERVER_URL 로 streamable HTTP 연결한다.
        # 오가는 메시지(list_tools/call_tool)는 MCP 규격 그대로다 — 인메모리
        # 시절과 transport 만 다르다(완전 분리).
        async with AsyncExitStack() as stack:
            try:
                mcp_client = await stack.enter_async_context(_mcp_client())
                # 도구 7종의 이름·설명(docstring)·입력 스키마를 Anthropic tools
                # 포맷으로 변환. 이 '설명'이 Claude 가 어떤 도구를 쓸지 판단하는
                # 유일한 근거다.
                tools = _to_anthropic_tools(await mcp_client.list_tools())
                if not web_requested:
                    # 인터넷 검색은 명시 요청 턴에만 노출 — 미요청 턴에는 모델이
                    # 선택할 수조차 없다(사내 문서 abstention 이 웹 폴백으로
                    # 조용히 대체되는 경로의 구조적 차단).
                    tools = [t for t in tools if t["name"] != "search_web"]
            except Exception as e:  # noqa: BLE001 - 연결 실패는 명시적 안내로 흡수
                # 연결·도구목록 확보 실패(서버 미기동·URL 오설정)는 크래시 대신
                # '무엇을 확인하라'는 안내문으로 응답한다(LLM API 실패와 같은 규율).
                flow(
                    "_chat_llm()",
                    "MCP 서버 연결 실패 — 크래시(HTTP 500) 대신 확인 안내문으로 응답(원인 타입만 노출)",
                    error=type(e).__name__,
                    next="_finalize() — 실패 안내문도 검증 게이트는 통과시켜 반환한다",
                )
                return _finalize(
                    AgentResult(
                        answer=_MCP_UNAVAILABLE_MSG.format(type(e).__name__),
                        mode="llm",
                        grounded=False,  # 실패 안내문 — 근거 보증이 아니다
                    ),
                    [],
                    question=_question_context(message, history),
                )
            flow(
                "_chat_llm()",
                "MCP HTTP 연결·도구 목록 확보 — 이 이름·설명(docstring)이 Claude 의 도구 선택 근거",
                tools=[t["name"] for t in tools],
                next="step 0 LLM 호출 — 이 도구 목록을 tools= 로 실어 Claude 에게 보낸다",
            )
            # Claude 와의 왕복 대화록 — 루프를 돌며 도구 결과가 계속 덧붙는다.
            messages: list[dict] = [*history, {"role": "user", "content": message}]

            # 에이전트 루프: 한 바퀴 = Claude API 호출 1번. Claude 의 응답은
            # 둘 중 하나다 — "도구를 불러 달라"(stop_reason == "tool_use") 또는
            # "답변을 확정하겠다"(그 외). 6은 도구 왕복 허용 상한(무한루프 방지).
            for step in range(6):
                flow(
                    "_chat_llm()",
                    f"LLM 호출(step {step}) — 대화록+도구 목록을 Claude 에 보내 '도구 요청 or 답변 확정'을 묻는다",
                    model=config.LLM_MODEL, messages=len(messages),
                    next="응답의 stop_reason 으로 분기 — 'tool_use' 면 도구 실행, 그 외면 답변 확정",
                )
                try:
                    with timed(trace, f"llm.step{step}", "llm", {"model": config.LLM_MODEL}):
                        resp = await client.messages.create(
                            model=config.LLM_MODEL,
                            max_tokens=config.LLM_MAX_TOKENS,
                            system=SYSTEM_PROMPT,
                            tools=tools,
                            messages=messages,
                        )
                except Exception as e:  # noqa: BLE001 - 외부 API 실패는 명시적 안내로 흡수
                    # API 호출 실패(잘못된 키·네트워크 오류·틀린 모델명)는 크래시
                    # 대신 '무엇을 확인하라'는 안내문을 답변으로 반환한다 —
                    # 예외를 그대로 전파하면 사용자에게는 원인 불명의 HTTP 500 이
                    # 된다. 예외 '타입명'만 싣고 메시지 원문은 싣지 않는다:
                    # 외부 라이브러리 에러 문구에는 요청 내용이 에코될 수 있다
                    # (에러 계약 응답을 신뢰 소스에서 빼는 것과 같은 규율).
                    flow(
                        "_chat_llm()",
                        "LLM API 호출 실패 — 크래시(HTTP 500) 대신 확인 안내문으로 응답(원인 타입만 노출)",
                        error=type(e).__name__,
                        next="_finalize() — 실패 안내문도 검증 게이트는 통과시켜 반환한다",
                    )
                    return _finalize(
                        AgentResult(
                            answer=(
                                f"(LLM API 호출에 실패했습니다: {type(e).__name__}. "
                                "ANTHROPIC_API_KEY 가 유효한지, 네트워크와 LLM_MODEL 설정을 "
                                "확인해 주세요. 키를 비우면 오프라인 모드로 동작합니다.)"
                            ),
                            mode="llm",
                            tool_calls=tool_calls,
                            citations=_collect_citations(tool_calls, raw_search_results),
                            grounded=False,  # 실패 안내문 — 근거 보증이 아니다
                            web_results=web_results,
                        ),
                        trusted_texts,
                        question=_question_context(message, history),
                        allowed_superseded_ids=history_doc_ids,
                        user_facts=user_facts,
                        web_texts=web_texts,
                    )
                # Claude 가 도구 요청 없이 답변을 확정한 경우 — 루프의 정상 출구.
                # 텍스트 블록을 이어 붙여 최종 답변으로 확정하고, 사후 검증
                # 게이트(_finalize)를 거쳐 반환하면 이 함수 전체가 끝난다.
                if resp.stop_reason != "tool_use":
                    text = "".join(
                        b.text for b in resp.content if b.type == "text"
                    ).strip()
                    if resp.stop_reason == "max_tokens":
                        # 잘린 답변을 조용히 내보내지 않는다 — 문장이 중간에 끊긴
                        # 답변은 뒷부분(예: 🌐 웹 구역)이 통째로 사라져도 사용자
                        # 에게는 완결된 답처럼 읽힌다(실배포에서 "…필요합"으로
                        # 끝난 실측 — 조용한 절단은 조용한 폴백과 같은 결함).
                        text += (
                            "\n\n⚠ 응답이 길이 제한(max_tokens)에 걸려 여기서 잘렸습니다. "
                            "질문을 나누어 다시 시도해 주세요."
                        )
                    flow(
                        "_chat_llm()",
                        "Claude 가 답변 확정(stop_reason≠tool_use) — 루프의 정상 출구, 사후 검증 게이트로",
                        stop_reason=resp.stop_reason, answer_len=len(text),
                        grounded=grounded_evidence,
                        next="_finalize() — stop_reason 이 tool_use 가 아니므로 루프를 끝내고 이 답변을 검증한다",
                    )
                    # 생성 답변은 매번 다르므로 평가셋이 아니라 '이 답변'을 검증한다
                    return _finalize(
                        AgentResult(
                            answer=text,
                            mode="llm",
                            tool_calls=tool_calls,
                            citations=_collect_citations(tool_calls, raw_search_results),
                            # grounded = '내용 있는 도구 출력을 받은 적 있는가'.
                            # Claude 가 도구를 한 번도 안 부르고 답하면 False —
                            # 근거 보증이 없다는 표시다(과거에는 dataclass 기본값
                            # True 가 그대로 노출되어 출처 0건 답변이 근거 배지를
                            # 달고 나갔다 — v7 발견, v10 에서 '빈 검색 성공'까지 보강).
                            grounded=grounded_evidence,
                            web_results=web_results,
                        ),
                        trusted_texts,
                        question=_question_context(message, history),
                        allowed_superseded_ids=history_doc_ids,
                        user_facts=user_facts,
                        web_texts=web_texts,
                    )

                # ── Claude 가 "이 도구를 이 인자로 불러 달라"고 요청한 경우 ──
                # Claude 는 도구를 직접 실행하지 못한다: 요청(tool_use 블록)만
                # 보내고, 실행은 여기서 대신한 뒤 결과를 대화에 붙여 돌려준다.
                messages.append({"role": "assistant", "content": resp.content})  # 도구 요청도 대화록에 남긴다
                tool_results_content = []
                for block in resp.content:
                    if block.type != "tool_use":
                        continue
                    flow(
                        "_chat_llm()",
                        "Claude 의 도구 실행 요청 — Claude 는 요청만 보내고, 실행은 MCP 를 통해 여기서 대신한다",
                        tool=block.name, args=dict(block.input),
                        next=f"src/mcp_server/server.py 의 {block.name}() 실행 — stop_reason=tool_use 였고 "
                             "Claude 가 도구 설명(docstring)을 근거로 이 도구를 골랐기 때문",
                    )
                    data, is_error = await self._safe_tool_call(
                        mcp_client, block.name, dict(block.input), trace
                    )
                    flow(
                        "_chat_llm()",
                        "도구 실행 완료 — 성공 출력은 검증용 신뢰 소스(trusted_texts)에 축적",
                        tool=block.name, error=is_error,
                        result=("오류: " + str(data)) if is_error else _summarize(data),
                        next="결과를 tool_result 블록으로 포장 — 실패였다면 에러 그대로 되먹여 Claude 가 자가 정정",
                    )
                    if not is_error and not _is_contract_error(data):
                        # 성공한 도구 출력만 검증의 신뢰 소스로 쌓는다. 단:
                        # - 입력 에코(query·as_of)는 버린다 — 사용자가 틀린 값을
                        #   전제로 물으면 그 값이 도구 출력에 메아리치는데, 이를
                        #   신뢰 소스에 넣으면 모델이 맞장구친 오답도 검증을
                        #   통과해 버린다.
                        # - {"error": ...} 계약 응답은 통째로 제외 — 에러 문구
                        #   '안'에도 사용자 입력이 에코된다.
                        # - 케이스 서술 에코(case)는 별도 바구니(user_facts)로 —
                        #   지지 근거로는 인정하되 from_case 라벨이 붙는다.
                        # - 인터넷 검색(search_web) 출력은 사내 신뢰 소스가
                        #   아니다 — 검증의 3번째 계층(web_texts)으로만 쌓고,
                        #   grounded(사내 도구 근거 확보) 판정에도 넣지 않는다.
                        stripped, facts = _split_user_facts(data)
                        if block.name == "search_web":
                            web_texts.append(_stringify(stripped))
                            if isinstance(stripped, dict):
                                web_results.extend(stripped.get("results", []))
                        else:
                            trusted_texts.append(_stringify(stripped))
                            user_facts.extend(facts)
                            if _has_evidence(stripped):  # 결과 0건짜리 '성공'은 근거로 안 친다
                                grounded_evidence = True
                    # 출처(citations) 수집 — UI 출처 카드와 폐지본 인용 검증의 재료.
                    if not is_error and isinstance(data, dict):
                        if block.name == "search_regulations" and "results" in data:
                            raw_search_results.append(data)
                            if block.input.get("include_superseded") or block.input.get("as_of"):
                                # 이력 조회 검색이 '실제로 반환한' 문서만 폐지본
                                # 인용 경고에서 면제한다(문서 단위 허용). 실패한
                                # as_of 호출까지 집계하면, 시점 검색이 한 번도
                                # 성공하지 않았는데 경고만 꺼진다.
                                history_doc_ids |= {
                                    r["doc_id"] for r in data["results"] if r.get("doc_id")
                                }
                        elif block.name in ("assess_adverse_event", "draft_ae_report") and data.get("basis"):
                            raw_search_results.append(data["basis"])  # 트리아지/초안의 근거 규정도 출처로
                    tool_calls.append(
                        ToolCall(
                            name=block.name,
                            args=dict(block.input),
                            result_summary=("오류: " + str(data)) if is_error else _summarize(data),
                        )
                    )
                    # 도구 결과를 tool_result 블록으로 만들어 Claude 에게 돌려줄 준비.
                    tool_results_content.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": _stringify(data),
                            # 실패도 그대로 되먹인다 — Claude 가 에러를 보고
                            # 스스로 인자를 고치거나 다른 도구로 재시도하게 한다.
                            "is_error": is_error,
                        }
                    )
                # 결과를 대화록에 붙이고 루프 선두로 — 다음 messages.create() 에서
                # Claude 는 이 결과를 '관찰'한 상태로 추가 도구 호출 또는 답변
                # 확정을 판단한다.
                messages.append({"role": "user", "content": tool_results_content})
                flow(
                    "_chat_llm()",
                    "도구 결과를 tool_result 블록으로 대화록에 되먹임 — 다음 step 에서 Claude 가 '관찰'한다",
                    tool_results=len(tool_results_content), trusted_texts=len(trusted_texts),
                    next=f"step {step + 1} LLM 재호출 — 커진 대화록을 보고 Claude 가 '추가 도구 or 답변 확정'을 다시 판단",
                )

            # 6바퀴를 다 썼는데도 도구 호출만 반복된 경우 — 답변 확정 실패 안내로 마감.
            return _finalize(
                AgentResult(
                    answer="(도구 호출이 반복되어 응답을 확정하지 못했습니다. 질문을 좁혀 다시 시도해 주세요.)",
                    mode="llm",
                    tool_calls=tool_calls,
                    citations=_collect_citations(tool_calls, raw_search_results),
                    grounded=False,  # 확정 실패 안내문 — 근거 보증이 아니다
                    web_results=web_results,
                ),
                trusted_texts,
                question=_question_context(message, history),
                allowed_superseded_ids=history_doc_ids,
                user_facts=user_facts,
                web_texts=web_texts,
            )

    async def _safe_tool_call(self, mcp_client, name: str, args: dict, trace: Trace):
        """MCP 도구를 지연 측정과 함께 호출하고, 실패해도 크래시하지 않는다.

        Returns: (data, is_error). 실패 시 data는 에러 메시지 문자열.
        """
        try:
            with timed(trace, f"tool.{name}", "tool", {"args": args}):
                call = await mcp_client.call_tool(name, args)
            return call.data, False
        except Exception as e:  # noqa: BLE001 - 도구 실패는 응답으로 흡수(견고성)
            return f"도구 '{name}' 실행 실패: {type(e).__name__}: {e}", True

    # ---- 오프라인 모드: 규칙 라우팅 + grounded 추출 답변 ----
    async def _chat_offline(self, message: str, history: list[dict], trace: Trace) -> AgentResult:
        tool_calls: list[ToolCall] = []
        # 멀티턴: 짧은 후속질문("그럼 그건 며칠?")은 직전 사용자 발화와 병합해 맥락 복원
        resolved = _resolve_followup(message, history)
        async with AsyncExitStack() as stack:
            try:
                # LLM 모드와 동일한 HTTP 연결 — 오프라인 모드도 도구는 MCP 서버를 거친다.
                mcp_client = await stack.enter_async_context(_mcp_client())
            except Exception as e:  # noqa: BLE001 - 연결 실패는 명시적 안내로 흡수
                return _finalize(
                    AgentResult(
                        answer=_MCP_UNAVAILABLE_MSG.format(type(e).__name__),
                        mode="offline",
                        grounded=False,  # 실패 안내문 — 근거 보증이 아니다
                    ),
                    [],
                    question=resolved,
                )
            intent = _route_intent(resolved)
            flow(
                "_chat_offline()",
                "오프라인 규칙 라우팅 — LLM 대신 키워드 규칙으로 의도를 분류해 MCP 도구를 선택",
                intent=intent, resolved=resolved,
                next=_INTENT_NEXT[intent],
            )

            if intent == "web":
                # 명시적 인터넷 검색 요청 — 이 경로 밖에서는 search_web 이 절대
                # 호출되지 않는다(사내 문서 abstention 의 웹 폴백 금지). 결과는
                # 사내 근거와 분리된 표시(🌐)·필드(web_results)·검증 계층
                # (web_texts)으로 나간다.
                args = {"query": resolved, "top_n": 3}
                with timed(trace, "tool.search_web", "tool", {"args": args}):
                    data = (await mcp_client.call_tool("search_web", args)).data
                tool_calls.append(ToolCall("search_web", args, _summarize(data)))
                if _is_contract_error(data):
                    # 비활성화·네트워크 실패 — 조용한 빈 결과 대신 원인을 말하는
                    # 실패(에러 계약의 문구는 정적 텍스트 + 예외 타입명뿐이다).
                    return _finalize(
                        AgentResult(
                            answer=(
                                f"🌐 인터넷 검색을 수행하지 못했습니다 — {data['error']}."
                                + (f" ({data['expected']})" if data.get("expected") else "")
                            ),
                            mode="offline",
                            tool_calls=tool_calls,
                            grounded=False,
                        ),
                        [],
                        question=resolved,
                    )
                return _finalize(
                    AgentResult(
                        answer=_format_web_answer(data),
                        mode="offline",
                        tool_calls=tool_calls,
                        citations=[],           # 웹 결과는 사내 출처 카드에 넣지 않는다
                        grounded=False,         # '사내 도구 근거 확보' 배지의 대상이 아니다
                        web_results=data.get("results", []),
                    ),
                    [],
                    question=resolved,
                    web_texts=[_stringify(_strip_query_echo(data))],
                )

            if intent == "ae_report":
                # 케이스 서술 + '보고서 작성' 요청 → ICSR 초안 도구(트리아지+인과성+코딩+최소요건)
                args = {"case_description": resolved}
                if aware := _extract_awareness_date(resolved):
                    args["awareness_date"] = aware
                with timed(trace, "tool.draft_ae_report", "tool", {"args": args}):
                    data = (await mcp_client.call_tool("draft_ae_report", args)).data
                tool_calls.append(ToolCall("draft_ae_report", args, _summarize(data)))
                stripped, facts = _split_user_facts(data)
                return _finalize(
                    AgentResult(
                        answer=_format_report(data),
                        mode="offline",
                        tool_calls=tool_calls,
                        citations=_collect_citations(tool_calls, [data.get("basis", {})]),
                        grounded=True,
                    ),
                    [_stringify(stripped)],
                    question=resolved,
                    user_facts=facts,
                )

            if intent == "ae_triage":
                # 구체적 케이스 서술 → PV 트리아지 도구(중대성 판정+기한 계산)
                args = {"case_description": resolved}
                if aware := _extract_awareness_date(resolved):
                    args["awareness_date"] = aware
                with timed(trace, "tool.assess_adverse_event", "tool", {"args": args}):
                    data = (await mcp_client.call_tool("assess_adverse_event", args)).data
                tool_calls.append(ToolCall("assess_adverse_event", args, _summarize(data)))
                stripped, facts = _split_user_facts(data)
                return _finalize(
                    AgentResult(
                        answer=_format_triage(data),
                        mode="offline",
                        tool_calls=tool_calls,
                        citations=_collect_citations(tool_calls, [data.get("basis", {})]),
                        grounded=True,
                    ),
                    [_stringify(stripped)],
                    question=resolved,
                    user_facts=facts,
                )

            if intent == "deadlines":
                args = {"within_days": 30}
                with timed(trace, "tool.get_ra_deadlines", "tool", {"args": args}):
                    data = (await mcp_client.call_tool("get_ra_deadlines", args)).data
                tool_calls.append(ToolCall("get_ra_deadlines", args, _summarize(data)))
                answer = _format_deadlines(data)
                return _finalize(
                    AgentResult(answer=answer, mode="offline", tool_calls=tool_calls),
                    [_stringify(data)],
                    question=resolved,
                )

            checklist_note = ""
            if intent == "checklist":
                category = _guess_category(resolved)
                args = {"category": category}
                with timed(trace, "tool.get_submission_checklist", "tool", {"args": args}):
                    data = (await mcp_client.call_tool("get_submission_checklist", args)).data
                tool_calls.append(ToolCall("get_submission_checklist", args, _summarize(data)))
                if "error" not in data:
                    answer = _format_checklist(data)
                    return _finalize(
                        AgentResult(answer=answer, mode="offline", tool_calls=tool_calls),
                        [_stringify(data)],
                        question=resolved,
                    )
                # 카테고리 매칭 실패 — 엉뚱한 체크리스트를 자신 있게 내는 대신,
                # 지원 목록을 안내하고 규정 검색으로 폴백한다. LLM 모드가
                # error+available 을 되먹여 받아 자가 정정하는 것과 같은 계약을
                # 오프라인 라우터도 따르는 것(계약을 라우터가 우회하지 않는다).
                # 문구는 검색 결과를 약속하지 않는다 — 폴백 검색이 회피(abstention)로
                # 끝나면 "아래는 검색 결과입니다 / 근거를 찾지 못했습니다"가 한 답변에
                # 공존하는 자기모순이 된다(결과를 본 뒤에만 말할 수 있는 것을 미리
                # 말하지 않는다).
                available = ", ".join(data.get("available", []))
                checklist_note = (
                    f"※ 요청과 일치하는 체크리스트 유형을 찾지 못했습니다 (지원: {available}).\n\n"
                )

            # 기본: 규제문서 검색 후 근거 기반 답변.
            # 이력 조회 의도(예전/구판/개정 이력)를 감지하면 폐지본을 포함해 검색한다
            # — "이력을 요청하면 구판도 노출"이라는 사용 계약이 LLM 모드(모델이
            # include_superseded 를 스스로 지정)에서만 참이고 오프라인 라우터에는
            # 그 경로 자체가 없던 불일치의 해소.
            history_intent = any(k in resolved for k in _HISTORY_MARKERS)
            args = {"query": resolved, "top_n": 3}
            if history_intent:
                args["include_superseded"] = True
            with timed(trace, "tool.search_regulations", "tool", {"args": args}):
                data = (await mcp_client.call_tool("search_regulations", args)).data
            tool_calls.append(ToolCall("search_regulations", args, _summarize(data)))

            # 근거 충분성(grounding) 판정 → 부족하면 환각 대신 abstention
            results = data.get("results", [])
            top_score = results[0].get("score", 0.0) if results else 0.0
            coverage = _grounding_coverage(resolved, results)
            if top_score < SCORE_FLOOR and coverage < COVERAGE_FLOOR:
                return _finalize(
                    AgentResult(
                        answer=checklist_note
                        + (
                            "제공된 사내 규제문서에서 이 질문에 답할 근거를 찾지 못했습니다. "
                            "질문을 규제업무(허가·변경·GMP·라벨링·약물감시·임상) 범위로 좁혀 다시 시도해 주세요."
                        ),
                        mode="offline",
                        tool_calls=tool_calls,
                        citations=[],
                        grounded=False,
                    ),
                    [],  # 회피 답변은 수치 클레임이 없어 자명하게 통과 — 그래야 정상
                    question=resolved,
                )
            answer = checklist_note + _format_search_answer(data)
            return _finalize(
                AgentResult(
                    answer=answer,
                    mode="offline",
                    tool_calls=tool_calls,
                    citations=_collect_citations(tool_calls, [data]),
                    grounded=True,
                ),
                [_stringify(_strip_query_echo(data))],
                allow_superseded=history_intent,  # 이력 조회에서 폐지본 인용은 결함이 아니라 목적
                question=resolved,
            )


# ---------------------------------------------------------------------------
# 보조 함수 (오프라인 포매팅 · 라우팅)
# ---------------------------------------------------------------------------
_FOLLOWUP_MARKERS = ("그건", "그거", "그럼", "그게", "이건", "위", "방금", "아까", "그 경우", "그 때")

# 이력(폐지본 포함) 조회 의도 — **명시적 이력 요청 구문만** 좁게 잡는다.
# 단독 "이력"은 현행 질문의 일상 어휘("접수·처리 이력 관리 요건")와, 단독
# "폐지"는 현행 상태 질문("이 규정 폐지됐나요?")과 겹친다 — 이력 의도 오탐은
# 폐지본을 검색에 섞을 뿐 아니라 allow_superseded 로 **버전 경고까지 끈다**.
# 안전장치를 끄는 스위치의 오탐은 일반 라우팅 오탐보다 비싸므로, 확신 없는
# 어휘는 넣지 않는다(코딩 사전과 같은 철학).
_HISTORY_MARKERS = ("예전", "구판", "개정 이력", "개정 전", "이전 버전", "과거 규정", "당시 규정")


def _redact_history(history: list[dict]) -> list[dict]:
    """대화 이력의 콘텐츠에서 PII를 마스킹한다(외부 API로 나가기 전 방어).

    문자열 콘텐츠뿐 아니라 **anthropic 블록 리스트 표기**(content=[{"type":
    "text", "text": …}])도 마스킹한다 — 블록 표기는 이 파일의 _user_texts 가
    명시적으로 처리하는 '예상된 입력 형태'인데, 마스킹만 문자열 표기에서
    성립하는 것은 표기의 우회였다(정규식 \\b 경계가 한글 직결 표기에서만
    깨지던 v6 발견 C 와 동형 — 이력 리스트형 콘텐츠 속 개인정보가 마스킹
    없이 외부 LLM API 로 나가는 경로, v7 발견). 블록의 텍스트 표기는 두 키다
    — text(텍스트 블록)와 **content(tool_result 블록의 문자열 본문)**: raw
    transcript 를 이력으로 되돌려 보내는 사용에서 tool_result 표기만 빠지면
    같은 우회가 한 키 이름 차이로 반복된다(v7 리뷰 잔여분). 그 외 타입
    (비문자열 스칼라 등)은 텍스트가 아니므로 그대로 둔다.
    """

    def _redact_block(b: Any) -> Any:
        if not isinstance(b, dict):
            return b
        out_b = dict(b)
        if isinstance(out_b.get("text"), str):
            out_b["text"] = redact(out_b["text"]).text
        if isinstance(out_b.get("content"), str):  # tool_result 블록의 문자열 본문
            out_b["content"] = redact(out_b["content"]).text
        elif isinstance(out_b.get("content"), list):  # 중첩 블록 리스트
            out_b["content"] = [_redact_block(x) for x in out_b["content"]]
        return out_b

    out: list[dict] = []
    for turn in history:
        content = turn.get("content")
        if isinstance(content, str):
            out.append({**turn, "content": redact(content).text})
        elif isinstance(content, list):
            out.append({**turn, "content": [_redact_block(b) for b in content]})
        else:
            out.append(turn)
    return out


def _user_texts(history: list[dict]) -> list[str]:
    """대화 이력에서 사용자 발화 텍스트만 순서대로 추출한다."""
    out: list[str] = []
    for turn in history:
        if turn.get("role") != "user":
            continue
        content = turn.get("content", "")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):  # anthropic 블록 형식 대비
            parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
            if parts:
                out.append(" ".join(parts))
    return out


def _last_user_text(history: list[dict]) -> str:
    """대화 이력에서 마지막 사용자 발화 텍스트를 추출(멀티턴 맥락 복원용)."""
    texts = _user_texts(history)
    return texts[-1] if texts else ""


def _question_context(message: str, history: list[dict]) -> str:
    """검증의 from_question 라벨링용 질문 맥락 — 직전 사용자 발화까지만 포함한다.

    LLM 모드 답변은 이전 턴의 전제("보고 기한이 30일 맞지?" → 다음 턴 "그럼
    그 30일은…")를 이어받아 재서술할 수 있는데, 현재 턴 질문만 대조하면 그 값이
    '환각' 경고로 라벨링된다 — 실제로 맞는 경고는 '전제 확인 필요'다(둘 다
    경고는 붙는다 — 라벨은 경고의 종류를 조정할 뿐, 면제하지 않는다).

    범위를 두 방향으로 제한한다:
    - **사용자 발화만**: 어시스턴트의 이전 답변까지 넣으면 한 번 새어 나간
      미확인 수치가 다음 턴부터 '전제'로 완화 라벨링되는 자기 강화 루프가 생긴다.
    - **직전 턴까지만**(전체 이력 아님): 대화 전체를 누적하면 10턴 전 무관한
      맥락의 수치("점유율 90%")가 이후 모든 환각을 '전제'로 완화 라벨링한다 —
      완화 라벨의 면적은 실제 전제 이월이 일어나는 창(직전 턴, 오프라인
      후속질문 병합과 같은 창)으로 최소화한다.
    신뢰 소스(trusted_texts)에는 어느 쪽도 넣지 않는 것은 동일하다.
    """
    prev = _last_user_text(history)
    return f"{prev}\n{message}" if prev else message


def _resolve_followup(message: str, history: list[dict]) -> str:
    """짧거나 지시대명사로 시작하는 후속질문을 직전 발화와 병합한다.

    예) 이전: "변경허가 처리기한은?" / 현재: "그럼 변경신고는?"
        → 검색 질의를 "변경허가 처리기한은? 그럼 변경신고는?" 로 확장해 맥락 유지.
    오프라인 규칙 라우팅에서도 대명사 후속질문이 동작하게 하는 경량 처리.
    """
    if not history:
        return message
    m = message.strip()
    is_short = len(m) <= 12
    has_marker = any(mark in m for mark in _FOLLOWUP_MARKERS)
    if is_short or has_marker:
        prev = _last_user_text(history)
        if prev:
            return f"{prev} {message}".strip()
    return message


def _grounding_coverage(query: str, results: list[dict]) -> float:
    """질의 토큰 중 검색된 근거 텍스트에 실제로 등장한 비율(0~1).

    낮으면 검색이 답할 '재료'를 못 찾은 것 → abstention 신호.
    (LLM 모드에서 생성된 답의 faithfulness 를 사후 점검하는 데도 같은 지표를 쓴다.)

    커버리지는 '확장 질의' 기준으로 계산한다. 검색 자체가 동의어 확장
    ("부작용"→"이상사례")으로 정답을 찾는데, 회피 판정만 원 질의 토큰으로
    보면 정답을 찾아 놓고도 "근거 없음"이라 답하는 자기모순(과회피)이 생긴다
    — 범위밖 질문은 도메인 용어가 없어 확장되지 않으므로 이 보정은
    범위밖 회피 정확도를 해치지 않는다(faithfulness eval 로 검증).
    """
    q_tokens = set(tokenize(expand_query(query)))
    if not q_tokens:
        return 0.0
    ctx = "\n".join(r.get("text", "") for r in results)
    ctx_tokens = set(tokenize(ctx))
    return len(q_tokens & ctx_tokens) / len(q_tokens)


_AE_CASE_MARKERS = ("환자", "복용 후", "투여 후", "복용했", "투여했", "접종 후")
# 사건 어휘 = 중대·일반 총칭 + PV 코딩 사전의 증상 표면형 전체(v8).
# 이전에는 중대 어휘만 있어 "두드러기·발진" 같은 일반 증상 케이스가
# 검색으로 빠져 회피됐다 — 코딩 사전과 어휘를 공유해 사전 갱신이 라우팅에
# 자동 반영된다. 케이스 맥락(_AE_CASE_MARKERS)과의 AND 조건은 유지하므로
# "두드러기 보고 기한은?" 같은 규정 질문은 여전히 검색으로 간다(과잉 매칭 방지).
_AE_EVENT_MARKERS = (
    "사망", "입원", "생명", "쇼크", "아나필락시스", "중환자실", "기형", "장애",
    "부작용", "이상사례", "이상반응",
) + symptom_keywords()


_AE_REPORT_MARKERS = ("보고서", "초안", "kaers", "icsr", "보고서 작성", "보고서 만들")


# 인지일 표기 — 값 부분은 일부러 느슨하게 잡는다(아래 docstring 참고).
_AWARENESS_RE = re.compile(r"인지일\s*[:은는]?\s*(\d{4}[-./]\d{1,2}[-./]\d{1,2})")


def _extract_awareness_date(message: str) -> str:
    """서술 속 인지일을 도구 인자로 승격한다 — 없으면 "" (v8).

    오프라인 라우터가 case_description 만 전달하면 awareness_date 는 항상
    빈 값이라, 사용자가 "인지일은 2026-07-01" 이라고 명시해도 도구가 경고
    없이 '오늘'로 폴백했다 — 15일 기한 케이스라면 기한이 그만큼 밀린 값이
    caveat 도 없이 나가는, '조용한 폴백 금지' 원칙의 라우터 계층 위반.

    값 표기는 느슨하게 잡아 그대로 전달한다 — "2026-7-1"·"2026/07/01" 같은
    형식 오류의 판정과 caveat 부착은 도구(triage)의 계약이므로, 라우터가
    여기서 정규화하거나 걸러 그 계약을 우회하지 않는다(_guess_category 와
    같은 규율: 안전장치를 만든 층 아래에서 무력화하지 않는다).
    """
    m = _AWARENESS_RE.search(message)
    return m.group(1) if m else ""


# 흐름 로그의 next= 안내문 — 라우팅 결과(intent)가 왜 그 도구 호출로 이어지는지.
# 판정 규칙 자체는 _route_intent 에 있다(여기는 로그 문구만 — 규칙과 문구를
# 이중 유지하지 않도록 '어떤 어휘 계열이 신호였는지' 수준으로만 적는다).
_INTENT_NEXT = {
    "web": "search_web 호출 — 사용자가 인터넷/웹 검색을 명시적으로 요청하는 구문이 감지되어서(이 경로에서만 호출)",
    "ae_report": "draft_ae_report 호출 — 케이스 서술(환자/복용)+사건 어휘에 더해 '보고서/초안' 요청 어휘가 있어서",
    "ae_triage": "assess_adverse_event 호출 — 케이스 서술(환자/복용)+사건 어휘(입원·쇼크 등)가 함께 감지되어서",
    "deadlines": "get_ra_deadlines 호출 — '마감/일정/디데이' 등 업무 일정 어휘가 감지되어서",
    "checklist": "get_submission_checklist 호출 — '체크리스트/구비서류' 등 준비물 어휘가 감지되어서",
    "search": "search_regulations 호출 — 특정 도구 신호가 없어 기본 경로(규정 검색)로",
}


def _route_intent(message: str) -> str:
    m = message.lower()
    # 인터넷 검색: 명시적 요청 구문이 있을 때만 — 다른 어떤 인텐트보다 먼저
    # 본다("이상사례 최신 동향 인터넷에서 검색해줘"는 검색 요청이지 케이스가
    # 아니다). 이 분기 밖에서는 search_web 이 호출되는 경로 자체가 없다.
    if _wants_web_search(m):
        return "web"
    # AE 트리아지: '구체적 케이스 서술'일 때만 (환자/복용 맥락 + 사건 어휘).
    # "중대한 이상사례는 며칠 안에 보고?" 같은 '규정 질문'은 search 로 남긴다.
    if any(k in m for k in _AE_CASE_MARKERS) and any(k in m for k in _AE_EVENT_MARKERS):
        # 같은 케이스라도 '보고서 작성' 요청이면 초안 도구로(트리아지는 판정만).
        if any(k in m for k in _AE_REPORT_MARKERS):
            return "ae_report"
        return "ae_triage"
    if any(k in m for k in ["마감", "기한", "일정", "언제까지", "d-day", "디데이", "며칠", "남은"]):
        # '보고 기한' 같은 규정 질문과 구분: 업무 일정 신호가 강할 때만
        if any(k in m for k in ["마감", "일정", "남은", "이번 주", "오늘", "디데이", "d-day"]):
            return "deadlines"
    if any(k in m for k in ["체크리스트", "준비물", "구비서류", "무엇이 필요", "뭐가 필요", "준비할"]):
        return "checklist"
    return "search"


def _guess_category(message: str) -> str:
    """체크리스트 카테고리 추측 — 확신 있는 신호가 있을 때만 답한다.

    이전에는 아무 신호가 없어도 '품목허가'를 기본 추측으로 반환했다.
    그러면 "GMP 체크리스트 줘" 같은 미지원 요청에 엉뚱한 품목허가 체크리스트가
    자신 있게 나간다 — 도구에는 error+available 계약(조용한 빈 결과 금지)을
    만들어 놓고, 라우터가 항상 유효한 카테고리를 추측해 그 계약을 우회하고
    있었다(안전장치를 만든 층 아래에서 무력화하는 형태의 사각지대).
    확신 없으면 "" 를 반환하고, 호출부가 지원 목록 안내 + 검색 폴백으로 처리한다.
    """
    m = message
    if "변경" in m:
        return "변경허가"
    if any(k in m for k in ["안전", "이상사례", "부작용", "보고"]):
        return "안전성보고"
    if any(k in m for k in ["품목허가", "허가", "신약", "신규"]):
        return "품목허가"
    return ""


def _format_deadlines(data: dict) -> str:
    if not data.get("deadlines"):
        return f"오늘({data.get('today')}) 기준 임박한 규제 업무 마감이 없습니다."
    lines = [f"오늘({data['today']}) 기준 다가오는 규제 업무 마감입니다:", ""]
    for d in data["deadlines"]:
        dday = d["d_day"]
        tag = "🔴 지남/오늘" if dday <= 0 else ("🟠 임박" if dday <= 7 else "🟢")
        lines.append(f"- {tag} D{dday:+d} · {d['due_date']} · {d['item']} ({d['type']}, {d['owner']}, {d['status']})")
    return "\n".join(lines)


def _format_checklist(data: dict) -> str:
    # 에러 계약은 호출부(_chat_offline)가 먼저 걸러 검색 폴백으로 처리한다 —
    # 여기 도달하는 data 는 항상 정상 체크리스트다.
    lines = [f"[{data['category']}] 준비 체크리스트:", ""]
    lines += [f"{i}. {item}" for i, item in enumerate(data["items"], 1)]
    return "\n".join(lines)


def _format_triage(data: dict) -> str:
    lines = ["📋 이상사례(AE) 트리아지 결과", ""]
    if data.get("is_serious"):
        lines.append(f"- 중대성 판정: **중대(Serious)** — 충족 기준: {', '.join(data.get('criteria_met', []))}")
    else:
        lines.append("- 중대성 판정: 비중대 (중대성 기준 미감지)")
    lines.append(f"- 보고 경로: {data.get('route', '')}")
    if data.get("deadline_date"):
        lines.append(
            f"- 보고 기한: {data['deadline_date']} (인지일 {data.get('awareness_date')} 기준"
            + (", 지체 없이)" if data.get("deadline_days") == 0 else f", {data.get('deadline_days')}일 이내)")
        )
    lines.append(f"- 판정 사유: {data.get('rationale', '')}")
    if data.get("coded_terms"):
        coded = " · ".join(f"{t['verbatim']}→{t['pt']}({t['pt_en']})" for t in data["coded_terms"])
        lines.append(f"- 표준 용어 코딩(MedDRA 방식): {coded}")
    cz = data.get("causality") or {}
    if cz.get("suggested"):
        lines.append(f"- 인과성(WHO-UMC) 제안: {cz['suggested']} — {cz.get('rationale', '')}")
        if cz.get("missing_info"):
            lines.append(f"  · 보고자 확인 필요: {'; '.join(cz['missing_info'][:2])} 등")
    if data.get("pii_masked"):
        masked = ", ".join(f"{m['type']} {m['count']}건" for m in data["pii_masked"])
        lines.append(f"- 🔒 개인정보 마스킹: {masked}")
    for c in data.get("caveats", []):
        lines.append(f"- ⚠ {c}")
    return "\n".join(lines)


def _format_report(data: dict) -> str:
    """draft_ae_report 결과 → 초안 본문 + 보완 안내."""
    lines = [data.get("draft_markdown", "").strip()]
    followups = data.get("followups", [])
    if followups:
        lines += ["", "📌 보고 확정 전 확인/보완할 항목:"]
        lines += [f"- {f}" for f in followups]
    if data.get("pii_masked"):
        masked = ", ".join(f"{m['type']} {m['count']}건" for m in data["pii_masked"])
        lines += ["", f"🔒 개인정보 마스킹: {masked} (초안에는 비식별 서술만 포함)"]
    return "\n".join(lines)


def _format_search_answer(data: dict) -> str:
    results = data.get("results", [])
    if not results:
        return "관련 규정을 찾지 못했습니다. 질문을 바꿔 다시 시도해 주세요."
    # grounded 추출: 최상위 근거를 중심으로 요약 제시(오프라인이라 생성 대신 발췌).
    # 질문을 본문에 재인쇄하지 않는다 — 답변 본문은 검증 대상 클레임 공간이라,
    # 사용자 전제(질문 속 수치)를 그대로 옮기면 그것도 클레임이 된다.
    lines = [
        "※ 오프라인 모드: LLM 없이 검색 근거를 발췌해 제시합니다 "
        "(ANTHROPIC_API_KEY 설정 시 근거를 종합한 자연어 답변으로 자동 전환).",
        "",
        "가장 관련 높은 규정 근거:",
    ]
    for i, r in enumerate(results, 1):
        body = r["text"].split("]\n", 1)[-1].strip()
        snippet = body[:260] + ("…" if len(body) > 260 else "")
        lines.append(f"\n[근거 {i}] {r['title']} > {r['section']} (출처: {r['source']})\n{snippet}")
    return "\n".join(lines)


def _format_web_answer(data: dict) -> str:
    """search_web 결과 → 분리 표시(🌐)가 붙은 답변 본문.

    사내 검색 답변(_format_search_answer)과 형태를 일부러 다르게 한다 —
    첫 줄의 분리 고지와 항목별 URL 로, 어느 대목을 발췌해 읽어도 이것이
    사내 규제문서 근거가 아님이 드러나야 한다(명시와 분리의 '분리').
    """
    header = (
        "🌐 인터넷 검색 결과 — 아래 내용은 사내 규제문서가 아닌 공개 웹에서 검색된 것입니다. "
        "규제 판단의 근거로 사용하기 전 원문과 사내 규정을 반드시 확인하세요."
    )
    results = data.get("results", [])
    if not results:
        return header + "\n\n검색 결과를 찾지 못했습니다. 검색어를 바꿔 다시 시도해 주세요."
    lines = [header]
    for i, r in enumerate(results, 1):
        snippet = (r.get("snippet") or "").strip()
        if len(snippet) > 260:
            snippet = snippet[:260] + "…"
        lines.append(
            f"\n[웹 {i}] {r.get('title', '').strip()}\n{r.get('url', '')}"
            + (f"\n{snippet}" if snippet else "")
        )
    return "\n".join(lines)


def _summarize(data: Any) -> str:
    if isinstance(data, dict):
        if data.get("origin") == "web":
            # 웹 검색은 요약에도 분리 표시 — UI 도구 트레이스에서 사내 검색
            # ("N건 검색")과 구분되어 읽힌다.
            return f"🌐 웹 {len(data.get('results', []))}건 검색"
        if "reportable" in data:
            n_missing = len(data.get("missing", []))
            return "초안 완성(요건 충족)" if data["reportable"] else f"초안 생성(보완 {n_missing}건 필요)"
        if "is_serious" in data:
            return "중대 → " + data.get("route", "") if data["is_serious"] else "비중대 → PSUR"
        if "results" in data:
            return f"{len(data['results'])}건 검색"
        if "deadlines" in data:
            return f"{data.get('count', 0)}건 마감"
        if "items" in data:
            return f"체크리스트 {len(data['items'])}항목"
        if "documents" in data:
            return f"문서 {data.get('count', 0)}건"
    return str(data)[:80]


def _stringify(data: Any) -> str:
    import json

    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(data)
