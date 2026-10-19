"""프로젝트 전역 설정.

환경변수로 동작 모드를 제어한다.
- ANTHROPIC_API_KEY 가 있으면 실제 Claude(Enterprise LLM API)로 에이전트가 동작한다.
- 없으면 오프라인 모드로 폴백한다(검색 근거 기반 추출형 답변). 데모는 키 없이도 항상 실행된다.
"""
from __future__ import annotations

import os
from pathlib import Path

# 경로
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REG_DIR = DATA_DIR / "regulations"
RA_TASKS_FILE = DATA_DIR / "ra_tasks.json"
WEB_DIR = BASE_DIR / "web"
DESCRIPTION_DIR = BASE_DIR / "description"
DICTIONARY_HTML = DESCRIPTION_DIR / "dictionary.html"
BLANK_HTML = DESCRIPTION_DIR / "blank.html"

# MCP 서버(HTTP) 연결 URL — 에이전트·API 는 항상 이 URL 로 도구를 호출한다(완전 분리).
# Render 는 fromService 로 호스트명만 주입한다(MCP_SERVER_HOST) — host 속성은 서비스
# 생성 즉시 확정되는 정적 값이라 이것을 쓴다. hostport/port 속성은 대상 서비스의
# '열린 포트 감지'에 의존해 첫 sync 시점에 빈 값이 들어올 수 있다(실배포에서 확인).
# 포트는 우리 startCommand 가 8001 로 고정하므로 감지에 기댈 이유가 없다.
_host = os.environ.get("MCP_SERVER_HOST", "").strip()
_port = os.environ.get("MCP_SERVER_PORT", "8001").strip()
MCP_SERVER_URL = (
    f"http://{_host}:{_port}/mcp" if _host
    else os.environ.get("MCP_SERVER_URL", "http://127.0.0.1:8001/mcp").strip()
)
# 헬스체크 URL — /mcp 엔드포인트와 같은 서버의 custom route
MCP_SERVER_HEALTH_URL = MCP_SERVER_URL.rsplit("/mcp", 1)[0] + "/health"

# MCP 서버 Bearer 토큰 (선택) — 설정하면 서버는 이 토큰만 허용하고, 클라이언트
# (에이전트·preflight)는 요청에 자동 부착한다. MCP 서버를 공개 URL 로 열 때 필수.
# 내부 네트워크 전용(private service·compose 내부·로컬)에서는 비워 둔다.
MCP_AUTH_TOKEN = os.environ.get("MCP_AUTH_TOKEN", "").strip()
# 공개 배포 가드 — 켜면 preflight 가 '토큰 없는 공개 서버' 상태로 기동을 거부한다.
# 공개(web) 서비스에는 이 플래그를 함께 배포해, 나중에 토큰이 실수로 지워져도
# 무인증 공개 상태로 조용히 돌아가지 않게 한다(fail-closed).
MCP_REQUIRE_AUTH = os.environ.get("MCP_REQUIRE_AUTH", "").strip() in ("1", "true", "on")

# LLM (Enterprise LLM API) 설정 — 있으면 사용, 없으면 오프라인 폴백
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-opus-4-8")
LLM_AVAILABLE = bool(ANTHROPIC_API_KEY)
# 응답 생성 토큰 상한 — 1024 는 '사내 근거 구역 + 🌐 웹 구역' 2구역 답변이
# 중간에 잘리는 값이었다(실배포에서 웹 구역이 통째로 잘려 분리 표시가 사라진
# 실측). 잘림은 조용히 내보내지 않고 본문에 표시한다(agent 참고).
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "4096"))
# LLM 모드의 에이전트 루프 구현 선택 — 'direct'(기본: anthropic SDK 직접 구현 루프)
# | 'pydantic_ai'(PydanticAI 프레임워크 백엔드) | 'langgraph'(LangGraph 상태 그래프
# 백엔드). 세 백엔드는 같은 계약(_finalize 게이트·MCP 도구·스텝 상한)을 지킨다.
# 오프라인 모드는 백엔드 무관.
AGENT_BACKEND = os.environ.get("AGENT_BACKEND", "direct").strip().lower()

# LLM judge 2차 검증 (opt-in) — 결정론 게이트(src/verify)가 스스로 경계로 선언한
# '수치 없는 의미 왜곡'(부정문 뒤집기·주어 바꿔치기)을 보는 확률적 2차 계층.
# 기본 off 인 이유: 전 응답마다 도는 유료 API 콜 + 확률적 판정이므로, 켜는 것은
# 비용·비결정론을 받아들이는 명시적 결정이어야 한다. 켰는데 API 키가 없으면
# preflight 가 기동을 거부한다(fail-closed — MCP_REQUIRE_AUTH 와 같은 계열).
JUDGE_ENABLED = os.environ.get("LLM_JUDGE", "").strip() in ("1", "true", "on")
# 판정 모델은 생성 모델(LLM_MODEL)과 분리한다 — (1) 전 응답 실행이라 저비용
# 모델이 기본값이어야 하고, (2) 만든 모델과 비판하는 모델을 나누면 self-critique
# 순환(같은 편향으로 같은 오류를 통과시키는 것)이 부분 완화된다.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-haiku-4-5")

# 인터넷 검색(search_web) — 사용자가 '명시적으로' 요청했을 때만 에이전트에 노출되는
# 보조 도구다(명시와 분리 원칙 — 사내 문서에서 못 찾은 질문을 조용히 웹으로 대신
# 답하지 않는다). 사내망 차단 환경·외부 송신 금지 정책에서는 0 으로 끈다 —
# 끄면 도구가 명시적 에러 계약으로 답한다(조용한 빈 결과 금지).
WEB_SEARCH_ENABLED = os.environ.get("WEB_SEARCH", "1") not in ("0", "false", "off")
WEB_SEARCH_TIMEOUT = float(os.environ.get("WEB_SEARCH_TIMEOUT", "6.0"))  # 초

# RAG 하이퍼파라미터 (RAG '최적화'의 손잡이들)
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))       # 청크 크기(문자)
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "80"))  # 겹침(경계 손실 방지)
RETRIEVE_TOP_K = int(os.environ.get("RETRIEVE_TOP_K", "8"))  # 1차 회수 개수
RERANK_TOP_N = int(os.environ.get("RERANK_TOP_N", "3"))     # 리랭킹 후 최종 개수
# 벡터(임베딩) vs 키워드(BM25) 가중. 0.4 근거(dense 전환 후 재스윕, eval/sweep.py):
# TF-IDF 시절엔 0.3~0.7 전 구간 둔감해 중립값 0.5 였으나, e5 밀집임베딩은 0.5 에서
# 의미 인접 이웃 문서 1건이 역전(Hit@1 0.969)하고 0.3~0.4 에서 1.000 — 벡터 신호가
# 강해진 만큼 가중을 살짝 낮춘 0.4 채택. tfidf 폴백은 이 값에도 둔감(재실측 확인).
HYBRID_ALPHA = float(os.environ.get("HYBRID_ALPHA", "0.4"))
RERANK_WEIGHT = float(os.environ.get("RERANK_WEIGHT", "0.9"))  # 리랭커 신호 vs 1차점수 prior 결합
RERANK_IDF_POWER = float(os.environ.get("RERANK_IDF_POWER", "0.5"))  # 리랭커 토큰 가중 idf^p (0=균등)
# 섹션 타입 prior (리랭커 v3): 질의 의도로 게이트되는 구조 신호.
#  - contrast: "X와의 차이/구분(주의)" 대조 섹션 페널티(질의가 비교를 묻지 않을 때만)
#  - preamble: "목적/개요/총칙" 서두 섹션 감쇠(질의가 정의/취지를 묻지 않을 때만)
# 크기 근거는 eval/sweep.py 의 스윕 참고(contrast 는 0.25 이상에서 플랫,
# preamble 은 0.04~0.07 유효 밴드의 중앙).
RERANK_CONTRAST_PENALTY = float(os.environ.get("RERANK_CONTRAST_PENALTY", "0.3"))
RERANK_PREAMBLE_PENALTY = float(os.environ.get("RERANK_PREAMBLE_PENALTY", "0.055"))
# auto(기본) = 밀집임베딩(sentence-transformers) 시도 → 미설치/모델 미캐시면 tfidf 폴백.
# dense 는 명시 요구라 폴백 없이 실패한다(명시 설정은 시끄럽게 실패 — voyage 와 동일 원칙).
EMBEDDER_KIND = os.environ.get("EMBEDDER_KIND", "auto")      # auto | dense | tfidf | hashing | voyage
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
# dense 무신호 바닥값 — 밀집 코사인은 무관 질의에도 0 이 안 나오므로, 이 값 이하를
# '벡터 신호 없음'으로 판정한다(무신호 계약 일반화, retriever._hybrid 참고).
# 0.835 근거(e5-small, 후보군 내 max-cosine 실측): 개발셋 실질의 32건 min=0.846,
# 무관 질의 10건(gibberish·타 도메인 한국어) max=0.825 — 갭 +0.021 의 중앙값.
# 갭이 좁지만 무신호 판정은 'BM25 도 0'일 때만 발동하는 2중 조건이라(어휘가 한
# 토큰이라도 겹치면 BM25 가 구제) 실질의 오차단 위험은 낮다. 모델 교체 시 재보정.
DENSE_SIGNAL_FLOOR = float(os.environ.get("DENSE_SIGNAL_FLOOR", "0.835"))
QUERY_EXPANSION = os.environ.get("QUERY_EXPANSION", "1") not in ("0", "false", "off")
# 도메인 동의어 질의 확장(부작용→이상사례 등). 1단계 회수에만 적용.


def mode_banner() -> str:
    """현재 실행 모드를 한 줄로 반환(로그/헬스체크용)."""
    if LLM_AVAILABLE:
        backend = f" · backend={AGENT_BACKEND}" if AGENT_BACKEND != "direct" else ""
        return f"[LLM 모드] Enterprise LLM API 사용 · model={LLM_MODEL}{backend}"
    return "[오프라인 모드] API 키 없음 → 검색 근거 기반 추출형 답변으로 폴백"
