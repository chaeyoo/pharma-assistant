"""RA(규제업무)·PV(약물감시) 어시스턴트를 위한 MCP 서버.

MCP 구조로 사내 시스템을 통합하는 패턴 그대로, 제약 RA·PV 담당자의
업무 시스템(규제문서·마감일·체크리스트·이상사례 처리)을 MCP 도구로 노출한다.

노출하는 MCP primitive:
  - Tools:     search_regulations / get_ra_deadlines / get_submission_checklist
               / assess_adverse_event (PV 트리아지+인과성+코딩)
               / draft_ae_report (ICSR 초안+최소보고요건 검증)
               / list_regulation_documents
               / search_web (인터넷 검색 — 사용자 명시 요청 전용, 결과는 origin="web" 분리)
  - Resources: regulation://{doc_id}  (규제문서 원문 조회)
  - Prompts:   pv_case_intake  (케이스 처리 SOP — 클라이언트 무관 동일 절차)

이 서버는 두 가지로 사용된다.
  (1) stdio:  python -m src.mcp_server.server   (Claude Desktop/Cursor 연결)
  (2) HTTP:   python -m src.mcp_server.server --transport http --port 8001
              — 에이전트·API 가 fastmcp.Client(MCP_SERVER_URL) 로 원격 연결(운영 기본값,
              완전 분리 구조: docker-compose.yml 의 mcp 서비스 / Render 의 rapv-mcp)

도구의 '이름·설명(docstring)·타입힌트'가 곧 LLM에게 주는 사용설명서다.
도구 설계의 품질은 여기서 드러난다 — 에이전트가 언제 어떤 도구를 쓸지 이 설명으로 판단한다.
"""
from __future__ import annotations

import datetime as _dt

from fastmcp import FastMCP

from .. import config
from ..observability import flow
from ..rag.pipeline import RagPipeline

def _build_auth():
    """MCP_AUTH_TOKEN 이 설정된 경우에만 Bearer 토큰 검증기를 만든다.

    공개 URL 배포(Render web)용 — 내부 네트워크 전용(private service·compose
    내부·로컬 stdio)에서는 토큰 없이 무인증으로 동작한다. /health custom route
    는 인증 밖에 남는다(Render 헬스체크가 무토큰으로 호출) — 실측 확인.
    """
    if not config.MCP_AUTH_TOKEN:
        return None
    from fastmcp.server.auth import StaticTokenVerifier

    return StaticTokenVerifier(tokens={config.MCP_AUTH_TOKEN: {"client_id": "rapv-client"}})


mcp = FastMCP("RAPV-Assistant", auth=_build_auth())

# 규제문서 RAG 인덱스는 서버 로드시 1회 구축 (무거운 초기화를 캐시)
_pipeline: RagPipeline | None = None


def _get_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline().build()
    return _pipeline


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@mcp.tool
def search_regulations(
    query: str, top_n: int = 3, as_of: str = "", include_superseded: bool = False
) -> dict:
    """제약 규제문서(품목허가·변경허가·GMP·라벨링·약물감시·임상)를 의미 기반으로 검색한다.

    허가 절차, 심사 기간, 보고 기한, 제출자료, GMP 요건 등 '규정이 어떻게 되어 있는지'를
    묻는 질문에 사용한다. 하이브리드 검색 + 리랭킹으로 가장 관련 높은 근거 문단을 반환하며,
    각 결과에는 출처(문서명·섹션·버전·시행일)가 포함되어 답변의 근거 추적이 가능하다.

    버전 인지: 폐지(superseded)된 구판은 기본 제외한다(현행 규정만 답하도록).
    과거 시점의 규정을 조회하려면 as_of(YYYY-MM-DD)를, 폐지본 이력까지 보려면
    include_superseded=True 를 사용한다. as_of 시점 조회는 그 시점에 시행 중이던
    버전을 반환한다 — 이후 개정으로 폐지된 버전이라도 당시 현행이었다면 포함된다.

    Args:
        query: 검색할 질문/키워드 (예: "중대한 이상사례 보고 기한"). 개인정보가
            섞여 있으면 검색 전에 마스킹된다(아래 참고).
        top_n: 반환할 근거 문단 수 (기본 3, 유효 범위 1~5 — 범위 밖 값은 경계로
            보정된다. 보정을 명시하는 이유: 문서화되지 않은 클램프는 '10건을
            요청했는데 5건이 온' 호출자에게 조용한 축소로 보인다).
        as_of: 기준일(YYYY-MM-DD). 지정 시 그 시점에 시행 중이던 규정만 검색.
        include_superseded: True면 폐지된 구판도 포함(이력 조회용).

    Returns:
        {"results": [{"text","title","source","section","version","effective_date","score"}...]} 형태.
        as_of 가 YYYY-MM-DD 형식이 아니거나 query 가 비어 있으면
        {"error", "expected"} — 형식 오류를 조용히 무시하고 현행 기준으로
        답하면 사용자는 '그 시점 규정'을 받았다고 믿게 된다(자신 있는 오답).
        명시적 에러라야 에이전트가 정정 재시도한다.
    """
    from ..pv.redactor import redact

    if not query.strip():
        # 빈 질의는 무신호라 리트리버가 빈 결과를 반환한다(v9 계약) — 그것을
        # 그대로 흘리면 "results": [] 가 '관련 규정 없음'이라는 자신 있는
        # 오답으로 소비된다(조용한 빈 결과 금지). as_of 형식 오류와 동일한
        # 명시적 에러 계약으로 답해 에이전트가 질의를 채워 재시도하게 한다.
        # 에러 문구에 예시 질의를 넣지 않는 것도 as_of 의 예시 날짜 금지와
        # 같은 규율(예시가 실제 검색 클레임처럼 소비되는 것 차단)이다.
        return {"error": "query 가 비어 있음", "expected": "비어 있지 않은 검색어"}

    # 심층방어: query 도 자유 텍스트 인자다. 에이전트 경유 시에는 이미 마스킹된
    # 메시지에서 파생되지만, stdio 단독 사용(Claude Desktop 등)에서는 외부에서
    # 직접 들어온다 — 특히 EMBEDDER_KIND=voyage 구성이면 질의 원문이 외부 임베딩
    # API 로 그대로 송신된다. PV 도구 인자만 마스킹하던 것은 "도구 계층은 모든
    # 자유 텍스트 인자를 마스킹한다" 주장의 범위 과확장이었다(v7 발견).
    query = redact(query).text
    if as_of:
        try:
            _dt.date.fromisoformat(as_of)
        except ValueError:
            # 에러 문구에 예시 '날짜'를 넣지 않는다 — 에러 응답이 어딘가에서
            # 텍스트로 소비될 때 예시 날짜가 실제 날짜 클레임처럼 읽히는 것을
            # 원천 차단(형식은 포맷 문자열로 충분히 전달된다).
            # 형식이 틀린 as_of 는 임의 문자열일 수 있다 — 에러 에코도 마스킹
            # (필터 인자 에코와 같은 근거: 에러 문구에 원문이 실려 나간다).
            return {
                "error": f"as_of '{redact(as_of).text}' 가 YYYY-MM-DD 형식이 아님",
                "expected": "YYYY-MM-DD",
            }
    flow(
        "search_regulations()",
        "규제문서 RAG 검색 — 동의어 확장 → 하이브리드(벡터+BM25) 1차 회수 → 리랭킹 상위 N",
        query=query, top_n=top_n, as_of=as_of or "(현행)",
        include_superseded=include_superseded,
        next="RagPipeline.retrieve() 호출 — 부팅 시 만든 인덱스에서 실제 검색을 수행",
    )
    ctx = _get_pipeline().retrieve(
        query,
        rerank_n=max(1, min(top_n, 5)),
        as_of=as_of,
        include_superseded=include_superseded,
    )
    flow(
        "search_regulations()",
        "검색 완료 — 근거 문단+출처 메타(문서·섹션·버전·시행일)를 반환(답변의 근거 추적용)",
        results=len(ctx.chunks),
        top=f"{ctx.chunks[0].chunk.doc_id} {ctx.chunks[0].chunk.section} ({round(ctx.chunks[0].score, 4)})" if ctx.chunks else "(없음)",
        next="호출자에게 dict 반환 — 에이전트라면 신뢰 소스·출처로 축적, 도구 내부 호출(basis)이라면 판정 근거로 부착",
    )
    return {
        "query": query,
        "as_of": as_of or None,
        "results": [
            {
                "text": s.chunk.text.strip(),
                "doc_id": s.chunk.doc_id,
                "title": s.chunk.title,
                "source": s.chunk.source,
                "section": s.chunk.section,
                "version": s.chunk.version,
                "effective_date": s.chunk.effective_date,
                "status": s.chunk.status,
                "score": round(s.score, 4),
            }
            for s in ctx.chunks
        ],
    }


@mcp.tool
def get_ra_deadlines(within_days: int = 30, task_type: str = "") -> dict:
    """RA·PV 담당자의 규제 업무 마감일/기한을 조회한다.

    "이번 주 마감", "곧 처리해야 할 규제 업무", "지연 위험 항목" 등
    일정·기한 관련 질문에 사용한다. 오늘 기준 within_days 이내 항목을 마감일 순으로 반환한다.

    Args:
        within_days: 오늘부터 며칠 이내의 마감을 조회할지 (기본 30).
        task_type: 특정 유형만 필터 (예: "안전관리", "품목허가", "변경관리", "라벨링", "임상"). 빈 문자열이면 전체.

    Returns:
        {"today", "deadlines": [{"item","type","due_date","d_day","owner","status"}...]}.
        task_type 이 데이터에 없는 유형이면 {"error", "available": [...]} — 매칭 0건과
        구분되는 명시적 에러다.
    """
    from ..pv.redactor import redact
    from ..ra.tasks import deadlines_within

    # 필터 인자도 자유 텍스트다 — 미매칭 시 에러 문구에 그대로 에코되므로,
    # 개인정보가 들어오면 에코 경로로 노출된다(마스킹해도 매칭 의미는 불변:
    # 유효한 유형명에는 PII 패턴이 없다).
    task_type = redact(task_type).text if task_type else task_type
    flow(
        "get_ra_deadlines()",
        "RA 마감일 조회 — data/ra_tasks.json 에서 오늘 기준 임박 항목을 마감일 순으로(순수 함수 위임)",
        within_days=within_days, task_type=task_type or "(전체)",
        next="src/ra/tasks.py deadlines_within() 호출 — 검색(RAG)이 아닌 결정론 조회라 파이프라인을 안 거친다",
    )
    # 조회 로직은 src/ra/tasks.py 의 순수 함수에 위임한다 — PV 도구가 src/pv/
    # 에 위임하는 것과 대칭인 RA 갈래. 이 계층은 인자 마스킹과 MCP 노출만 담당.
    return deadlines_within(within_days=within_days, task_type=task_type)


@mcp.tool
def get_submission_checklist(category: str) -> dict:
    """규제 제출 유형별 준비 체크리스트를 반환한다.

    "품목허가 준비할 때 뭐가 필요하냐", "변경허가 체크리스트" 같은 질문에 사용한다.

    Args:
        category: 체크리스트 유형. 지원값: "품목허가", "변경허가", "안전성보고".

    Returns:
        {"category", "items": [...]} 또는 미지원 시 {"error", "available": [...]}.
    """
    from ..pv.redactor import redact
    from ..ra.tasks import submission_checklist

    category = redact(category).text if category else category  # 에러 에코 경로 방어(위와 동일)
    flow(
        "get_submission_checklist()",
        "제출 체크리스트 조회 — 카테고리별 준비 항목(미지원 카테고리는 error+available 계약)",
        category=category,
        next="src/ra/tasks.py submission_checklist() 호출 — 미지원 카테고리면 에러 계약이 반환돼 호출자가 폴백/재시도",
    )
    return submission_checklist(category)  # 조회 로직은 src/ra/ 위임(위와 동일한 대칭)


@mcp.tool
def assess_adverse_event(case_description: str, awareness_date: str = "") -> dict:
    """이상사례(AE) 케이스를 트리아지한다: 중대성(Serious) 판정 + 보고 경로/기한 계산.

    "환자가 복용 후 입원했는데 언제까지 보고해야 하나" 같은 '구체적 케이스'가
    주어졌을 때 사용한다. (규정 자체가 궁금한 질문은 search_regulations 를 사용.)

    규칙 기반(결정론적)으로 판정한다 — 보고기한 계산은 컴플라이언스라 LLM 추론이
    아닌 감사 가능한 규칙으로 수행하고, 판정 근거 규정(REG-005) 문단을 함께 반환한다.
    입력 속 개인정보(주민번호·연락처·이름 등)는 반환 전에 마스킹된다.
    부가로 인과성(WHO-UMC) 제안과 이상사례 표준 용어 코딩(MedDRA 방식)도 포함한다.

    Args:
        case_description: 이상사례 케이스 서술 (예: "환자가 복용 3일 후 아나필락시스로 입원").
        awareness_date: 회사 인지일(YYYY-MM-DD). 생략 시 오늘 기준으로 기한을 계산.

    Returns:
        {"case", "is_serious", "criteria_met", "expectedness", "route",
         "awareness_date", "deadline_date", "rationale", "caveats",
         "causality": {"suggested","rationale","signals","missing_info"},
         "coded_terms": [{"verbatim","pt","pt_en","soc"}...] (확정 코딩),
         "candidate_terms": [...] (LLT 참조 매칭 후보 — 사람 승인/기각 필요),
         "uncoded_expressions": [...] (감지만 된 미코딩 증상 표현),
         "basis": {"results": [...]}}  — basis 는 판정 근거 규정 문단(출처 포함).
    """
    from ..pv.causality import assess_causality
    from ..pv.coding import code_terms, flag_uncoded_expressions, suggest_candidates
    from ..pv.redactor import redact
    from ..pv.triage import assess_case

    masked = redact(case_description)  # 심층방어: 도구 단독 사용(stdio) 시에도 PII 비노출
    flow(
        "assess_adverse_event()",
        "PV 트리아지 — 규칙 기반(결정론)으로 중대성 판정·보고기한 계산 + 인과성 제안·용어 코딩",
        awareness_date=awareness_date or "(미지정→오늘)", case=masked.text,
        next="src/pv/ 규칙 모듈 순차 실행(triage→causality→coding) 후 근거 규정 회수를 위해 search_regulations 중첩 호출",
    )
    t = assess_case(masked.text, awareness_date)
    c = assess_causality(masked.text)
    coded = code_terms(masked.text)
    candidates = suggest_candidates(masked.text, coded)
    uncoded = flag_uncoded_expressions(masked.text, coded, candidates)
    # 판정 근거 규정 문단을 RAG로 회수해 부착(추적성) — 보고기한의 출처는 REG-005
    basis = search_regulations("중대한 이상사례 보고 기한 신속보고", top_n=2)
    flow(
        "assess_adverse_event()",
        "트리아지 완료 — 판정 결과에 근거 규정(basis)을 붙여 반환(모든 판정은 사람이 최종 확정)",
        is_serious=t.is_serious, deadline_date=t.deadline_date, route=t.route,
        next="호출자(에이전트)로 dict 반환 — basis 는 출처로, deadline_date 등 계산값은 검증 신뢰 소스로 쓰인다",
    )
    return {
        "case": masked.text,
        "pii_masked": masked.summary(),
        "is_serious": t.is_serious,
        "criteria_met": t.criteria_met,
        "expectedness": t.expectedness,
        "route": t.route,
        "awareness_date": t.awareness_date,
        "deadline_days": t.deadline_days,
        "deadline_date": t.deadline_date,
        "rationale": t.rationale,
        "caveats": t.caveats,
        "causality": {
            "suggested": c.suggested,
            "rationale": c.rationale,
            "signals": c.signals,
            "missing_info": c.missing_info,
        },
        "coded_terms": [ct.as_dict() for ct in coded],
        "candidate_terms": [ct.as_dict() for ct in candidates],
        "uncoded_expressions": uncoded,
        "basis": basis,
    }


@mcp.tool
def draft_ae_report(
    case_description: str,
    suspected_drug: str = "",
    reporter: str = "",
    patient_info: str = "",
    awareness_date: str = "",
) -> dict:
    """이상사례 개별사례보고(ICSR) 초안을 작성한다 — KAERS 제출용.

    "이 케이스 보고서 초안 만들어줘", "KAERS 보고서 작성해줘" 처럼
    '보고서 작성'을 요청받았을 때 사용한다. (중대성/기한만 궁금하면
    assess_adverse_event, 규정 자체가 궁금하면 search_regulations 를 사용.)

    수행 내용(전부 규칙 기반·결정론적):
      1) 최소보고요건(ICH E2D 4요소: 환자·보고자·의심약·이상사례) 충족 검증
         — 미충족이면 reportable=False 와 보완 항목을 반환한다.
      2) 중대성 판정 + 보고기한 계산 (assess_adverse_event 와 동일 규칙)
      3) 인과성(WHO-UMC) 등급 제안 + 부족 정보 follow-up 질문 생성
      4) 이상사례 표준 용어 코딩(MedDRA 방식 PT/SOC) — 확정/후보/미코딩 3계층
         (사전 미수록 표현은 LLT 참조 후보 또는 '미코딩 감지'로 표시하고
         사람 확정을 follow-up 으로 요청)
      5) 위를 종합한 마크다운 초안(draft_markdown) 조립
    개인정보는 초안 생성 전에 마스킹된다 — case_description 뿐 아니라 reporter·
    patient_info 등 모든 자유 텍스트 인자가 대상이다(초안에는 비식별 서술만 남음).

    Args:
        case_description: 이상사례 케이스 서술(자유 텍스트).
        suspected_drug: 의심 의약품명(알면 지정 — 최소요건 ③).
        reporter: 보고자 정보(예: "의사", "약사 홍OO" — 최소요건 ②).
        patient_info: 환자 비식별 정보(예: "45세 남성" — 최소요건 ①).
        awareness_date: 회사 인지일(YYYY-MM-DD). 생략 시 오늘.

    Returns:
        {"reportable", "missing", "followups", "draft_markdown",
         "is_serious", "deadline_date", "causality", "coded_terms",
         "candidate_terms", "uncoded_expressions",
         "pii_masked", "basis": {"results": [...]}}.
    """
    from ..pv.redactor import merged_summary, redact
    from ..pv.report import build_report

    # 자유 텍스트 인자는 '전부' 마스킹한다 — case_description 만 막으면 나머지
    # 인자(reporter="약사 홍길동님 010-…")가 PII 우회로가 된다. 에이전트 경유
    # 시에는 인자가 이미 마스킹된 메시지에서 파생되지만, stdio 단독 사용
    # (Claude Desktop 등)에서는 인자가 외부에서 직접 들어온다(심층방어).
    masked = redact(case_description)
    masked_drug = redact(suspected_drug)
    masked_reporter = redact(reporter)
    masked_patient = redact(patient_info)
    r = build_report(
        masked.text,
        suspected_drug=masked_drug.text,
        reporter=masked_reporter.text,
        patient_info=masked_patient.text,
        awareness_date=awareness_date,
    )
    basis = search_regulations("중대한 이상사례 보고 기한 신속보고 인과성 평가", top_n=2)
    flow(
        "draft_ae_report()",
        "ICSR 초안 작성 완료 — 최소보고요건(ICH E2D 4요소) 검증→트리아지→인과성→코딩→초안 조립(전부 규칙 기반)",
        reportable=r.reportable, missing=r.missing, deadline_date=r.triage.deadline_date,
        next="호출자(에이전트)로 dict 반환 — missing 이 있으면 답변에 보완 안내(followups)가 붙는다",
    )
    return {
        "reportable": r.reportable,
        "missing": r.missing,
        "followups": r.followups,
        "draft_markdown": r.draft_markdown,
        "is_serious": r.triage.is_serious,
        "deadline_date": r.triage.deadline_date,
        "causality": {
            "suggested": r.causality.suggested,
            "rationale": r.causality.rationale,
            "missing_info": r.causality.missing_info,
        },
        "coded_terms": [ct.as_dict() for ct in r.coded_terms],
        "candidate_terms": [ct.as_dict() for ct in r.candidate_terms],
        "uncoded_expressions": r.uncoded_expressions,
        "pii_masked": merged_summary(masked, masked_drug, masked_reporter, masked_patient),
        "basis": basis,
    }


# ---------------------------------------------------------------------------
# Prompts — MCP 의 세 번째 primitive (Tools·Resources·Prompts 3종 완성).
# 서버가 '도구를 올바른 순서로 쓰는 워크플로'까지 배포한다: 어느 클라이언트
# (Claude Desktop·Cursor·사내 에이전트)가 붙어도 같은 SOP 로 케이스를 처리한다.
# ---------------------------------------------------------------------------
@mcp.prompt
def pv_case_intake(case_description: str) -> str:
    """PV 이상사례 케이스 접수 SOP 프롬프트 — 접수부터 보고서 초안까지의 표준 절차.

    케이스 서술 속 개인정보는 프롬프트에 삽입되기 전에 마스킹된다 — SOP 를
    클라이언트 무관하게 배포한다는 프롬프트가 정작 PII 경계 절차만 빼고
    배포되면, 어느 클라이언트가 붙어도 '같은 절차로' 원문 개인정보를 LLM 에
    싣게 된다(도구 인자 마스킹과 같은 심층방어를 Prompt primitive 에도 적용).
    """
    from ..pv.redactor import redact

    case_description = redact(case_description).text
    return (
        "다음 이상사례 케이스를 PV 접수 SOP에 따라 처리하라.\n\n"
        f"케이스: {case_description}\n\n"
        "절차:\n"
        "1. assess_adverse_event 로 중대성·보고기한·인과성 제안·용어 코딩을 확인한다.\n"
        "2. 보고 기한의 근거 규정이 필요하면 search_regulations 로 원문을 확인한다.\n"
        "3. draft_ae_report 로 ICSR 초안을 만들고, 최소보고요건 누락(missing)과\n"
        "   follow-up 질문(followups)을 사용자에게 명확히 안내한다.\n"
        "4. 모든 판정에 '최종 확정은 PV 담당자' 임을 밝히고, 출처(문서·섹션)를 제시한다."
    )


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------
def _regulation_text(doc_id: str) -> str:
    """doc_id 로 규제문서 원문 전체를 찾는다 — 템플릿·정적 리소스가 공유하는 본체."""
    from ..pv.redactor import redact
    from ..rag.loader import load_documents

    for doc in load_documents(config.REG_DIR):
        if doc.doc_id.lower() == doc_id.lower():
            # 본문이 이미 같은 H1 제목으로 시작하면 중복 부착하지 않는다
            body = doc.text.lstrip()
            if body.startswith(f"# {doc.title}"):
                return body
            return f"# {doc.title}\n\n{doc.text}"
    # 미매칭 doc_id 는 임의 문자열일 수 있다 — 에러 에코도 마스킹(도구 인자와 동일 근거)
    return f"문서를 찾을 수 없음: {redact(doc_id).text}"


@mcp.resource("regulation://{doc_id}")
def get_regulation_document(doc_id: str) -> str:
    """규제문서 원문 전체를 doc_id(예: REG-003)로 조회한다."""
    return _regulation_text(doc_id)


def _register_static_regulation_resources() -> None:
    """코퍼스 13종을 구체 URI 의 정적 리소스로도 등록한다.

    템플릿(regulation://{doc_id})은 resources/templates/list 에만 잡혀서,
    resources/list(구체 URI 열거)만 보여주는 클라이언트 첨부 메뉴(Claude Desktop)
    에는 나타나지 않는다 — Prompt 는 인자 선언 스펙이 있어 입력 폼이 그려지지만
    리소스 템플릿에는 그런 UI 가 없다(실측). 같은 문서의 두 접근 경로를 함께 둔다:
    열거·선택은 정적 리소스, 파라미터 접근은 템플릿.
    """
    from ..rag.loader import load_documents

    def _make_reader(doc_id: str):
        def reader() -> str:
            return _regulation_text(doc_id)

        reader.__name__ = f"regulation_{doc_id.replace('-', '_').lower()}"
        return reader

    for d in load_documents(config.REG_DIR):
        mcp.resource(
            f"regulation://{d.doc_id}",
            name=d.doc_id,
            description=f"{d.title} — 원문 전체",
            mime_type="text/markdown",
        )(_make_reader(d.doc_id))


_register_static_regulation_resources()


@mcp.tool
def search_web(query: str, top_n: int = 3) -> dict:
    """인터넷(공개 웹)을 검색한다 — **사용자가 명시적으로 인터넷 검색을 요청한 경우에만** 사용한다.

    사내 규제문서·업무 데이터에 없는 정보(최신 뉴스·해외 규제 동향 등)를
    사용자가 "인터넷에서/웹에서 검색해 달라"고 **직접 요청**했을 때만 쓴다.
    사내 문서에서 근거를 찾지 못했다는 이유로 이 도구를 대신 쓰지 않는다 —
    그 경우는 "근거를 찾지 못했다"고 답하는 것이 옳다(회피는 회피로 남긴다).

    결과는 사내 규제문서 근거가 **아니다** — 반환 dict 와 각 결과 항목에
    origin="web" 이 박혀 있고 notice(분리 고지)가 동봉된다. 이 결과를 답변에
    사용할 때는 반드시 인터넷 검색 결과임을 표시하고 출처 URL 을 함께 제시해,
    사내 규제문서 근거와 섞이지 않게 구분한다.

    Args:
        query: 검색할 질문/키워드. 개인정보가 섞여 있으면 검색 전에 마스킹된다
            (질의가 외부 검색엔진으로 송신되므로 다른 도구보다도 필수적이다).
        top_n: 반환할 결과 수 (기본 3, 유효 범위 1~5 — 범위 밖 값은 경계로 보정).

    Returns:
        {"query", "origin": "web", "notice", "results": [{"title","url","snippet","origin"}...]}.
        질의가 비었거나, 인터넷 검색이 비활성화(WEB_SEARCH=0)이거나, 네트워크
        실패면 {"error", "expected"} — 조용한 빈 결과는 '관련 정보 없음'이라는
        자신 있는 오답으로 소비되므로 명시적 에러 계약으로 답한다.
    """
    from ..pv.redactor import redact
    from ..websearch import search as _web_search

    # 질의는 외부 검색엔진으로 '실제 송신'된다 — 도구 계층 마스킹이 다른 자유
    # 텍스트 인자보다도 앞서는 필수 경계다(voyage 임베더 송신과 같은 계열).
    query = redact(query).text
    flow(
        "search_web()",
        "인터넷 검색 — 사용자 명시 요청 전용 도구. 결과는 origin=web 로 사내 근거와 분리",
        query=query, top_n=top_n,
        next="src/websearch.py search() 호출 — 성공 결과는 웹 계층(web_texts)으로만 축적된다(사내 신뢰 소스 아님)",
    )
    out = _web_search(query, top_n=top_n)
    flow(
        "search_web()",
        "인터넷 검색 완료 — 에러 계약 또는 origin=web 결과 반환(분리 고지 동봉)",
        results=len(out.get("results", [])), error=out.get("error", ""),
        next="호출자(에이전트)로 dict 반환 — 답변에는 '인터넷 검색 결과' 표시와 URL 이 함께 나가야 한다",
    )
    return out


@mcp.tool
def list_regulation_documents() -> dict:
    """검색 가능한 규제문서 목록(제목·카테고리·doc_id)을 반환한다."""
    from ..rag.loader import load_documents

    docs = load_documents(config.REG_DIR)
    return {
        "count": len(docs),
        "documents": [
            {
                "doc_id": d.doc_id,
                "title": d.title,
                "category": d.metadata.get("category", ""),
                "version": d.metadata.get("version", ""),
            }
            for d in docs
        ],
    }


# ---------------------------------------------------------------------------
# 헬스체크 (HTTP transport 전용 custom route — stdio 모드에는 노출되지 않는다)
# ---------------------------------------------------------------------------
@mcp.custom_route("/health", methods=["GET"])
async def health(request):  # noqa: ANN001 - starlette Request
    """컨테이너 오케스트레이션(compose healthcheck·preflight) 용 준비 상태 응답.

    _get_pipeline() 이 완료된 뒤에만 200 — RAG 인덱스 구축 완료가 곧 준비 완료다.
    """
    from starlette.responses import JSONResponse

    p = _get_pipeline()
    return JSONResponse({"status": "ok", "rag": {"docs": p.n_docs, "chunks": p.n_chunks}})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAPV-Assistant MCP 서버")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    _get_pipeline()  # 인덱스 미리 구축 (헬스체크가 준비 완료 후에만 200 을 주도록)
    if args.transport == "http":
        # allowed_hosts=["*"]: fastmcp 의 Host 헤더 가드는 기본이 loopback 전용이라
        # compose/Render 내부 호스트명(예: mcp:8001)의 요청이 421 로 거절된다.
        # 이 서버는 내부 네트워크 전용(호스트 미공개)이므로 전체 허용이 안전하다.
        # stateless_http=True: 프로토콜 세션 장부를 만들지 않는다. 기본(stateful)은
        # 세션 테이블이 프로세스 메모리에 있어 재배포가 곧 전 세션 무효화다 —
        # 세션을 들고 있던 외부 클라이언트(Claude Desktop 의 mcp-remote)가
        # 'Session not found' 404 로 죽고 클라이언트 재시작 전까지 복구되지 않는
        # 것을 실측했다. 이 서버는 서버발 기능(sampling·elicitation·알림·구독)을
        # 쓰지 않는 순수 요청-응답이라 세션의 이점이 없고, 무세션이 곧 재배포
        # 무중단이다. (앱 수준 상태 — RAG 인덱스·계기판 — 는 층이 다르며 그대로다.)
        mcp.run(
            transport="http", host=args.host, port=args.port,
            allowed_hosts=["*"], stateless_http=True,
        )
    else:
        mcp.run()  # stdio — Claude Desktop/Cursor 연결(종전과 동일)
