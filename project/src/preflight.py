"""배포 전 점검(preflight) — 서버를 띄우기 전에 통과해야 하는 Day-0 게이트.

시스템을 새 환경에 배치할 때 가장 먼저 깨지는 것은 코드가 아니라
**데이터와 설정**이다: 문서 메타데이터 누락, 폐지 체인 단절, 업무 데이터의
날짜 오타, 상충하는 하이퍼파라미터. 이런 결함은 부팅은 통과하고 **운영 중에
오답으로** 나타난다 — 규제 도메인에서 가장 나쁜 실패 방향이다(예: status
필드가 빠진 폐지 구판은 버전 필터를 그대로 통과해 현행 답변에 섞인다).

그래서 기동 전에 결정론적 점검을 강제한다. 네 그룹:

  1. 설정 불변식 — 하이퍼파라미터끼리의 관계(overlap < chunk_size,
     rerank_top_n ≤ retrieve_top_k, 0 ≤ α ≤ 1 …). 각각은 유효해도 조합이
     모순이면 조용히 품질이 깨진다.
  2. 코퍼스 무결성 — frontmatter 필수 필드(doc_id·title·version·
     effective_date), doc_id 유일성, status 어휘, **폐지 체인**(superseded
     문서는 superseded_by 가 실존하는 현행 문서를 가리켜야 한다 — 버전 인지
     검색의 전제가 데이터에서 성립하는지).
  3. 업무 데이터 스키마 — ra_tasks.json 의 마감일 필수 필드·날짜 형식,
     체크리스트 비어있지 않음, 마감일 전건 과거(시한부 샘플 데이터의 부패 —
     스키마는 유효한 채로 데모 서사만 조용히 죽는 형태) 감지.
  4. 스모크(canary) — 실제 파이프라인을 태워 본다: 대표 질의가 정답 문서를
     1위로 회수하는가, 대표 케이스가 중대+15일로 트리아지되는가, 그리고
     **안전장치들의 자가 테스트** — 검증 게이트(근거 밖 수치를 심은 답변이
     실제로 걸리는가 / 정상 답변이 통과하는가)와 PII 마스킹(심은 개인정보가
     실제로 지워지는가). 안전장치가 고장난 채 배포되는 것은 안전장치가 없는
     것보다 나쁘다 — 안전장치 자신도 기동 전에 검사받는다.

실패 방향: 문제가 하나라도 있으면 **exit 1** — run.sh 와 CI 가 이 게이트를
통과해야 서버를 띄운다(운영 중 경고 부착과 달리, 배포 시점은 차단이 옳다:
아직 사용자가 없으므로 시끄럽게 멈추는 비용이 0이다).

실행:  python -m src.preflight                # 전 그룹(로컬 개발·CI)
       python -m src.preflight --role mcp     # MCP 서버 컨테이너 게이트
       python -m src.preflight --role api     # API 컨테이너 게이트(MCP 연결 대기 포함)
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from pathlib import Path

from . import config

# 코퍼스 frontmatter 계약 — 출처 표시·버전 인지 검색이 의존하는 최소 필드
_REQUIRED_META = ("doc_id", "title", "version", "effective_date")
_VALID_STATUS = ("", "active", "superseded")  # 없음(현행 암시)·현행·폐지

# 스모크 canary — 코퍼스의 핵심 문서·핵심 판정이 살아 있는지 확인하는 대표 케이스.
# canary 는 '어려운' 케이스가 아니라 '무조건 되어야 하는' 케이스다: 이것이
# 깨졌다면 데이터 교체·설정 실수로 시스템의 전제 자체가 무너진 것이다.
_CANARY_QUERY = "중대한 이상사례는 며칠 안에 보고해야 하나?"
_CANARY_DOC = "REG-005"
# 입원(중대) but 생명위협 아님 → '인지일+15일' 경로: 기한 '일수'와 '날짜 연산'을 함께 검사
_CANARY_CASE = "환자가 A정 복용 후 심한 두드러기로 입원했습니다."
# 시점(as_of) 조회 canary — 현행(REG-005, 2025-04-01 시행) 이전 시점에는
# 폐지 구판(REG-013)이 '당시 현행'으로 반환되어야 한다. 이것이 깨지는 대표
# 형태가 '폐지본 기본 제외를 as_of에도 적용 → 개정된 규정은 과거 시점에서
# 아무 버전도 안 나옴'이라, 시점 조회의 의미론 자체를 기동 전에 검사한다.
_CANARY_AS_OF = "2025-01-01"
_CANARY_AS_OF_DOC = "REG-013"

# 검증 게이트 자가 테스트 케이스 — 런타임 게이트의 **모든 경고 축**을 기동 전에
# 1건씩 태워 본다. 처음에는 수치 존재 축 1개만 심어 봤는데, 그러면 방향·역할·
# 버전·전제 라벨 축은 고장난 채 배포될 수 있다 — '자가 테스트를 한 안전장치에만
# 적용하는 비대칭이 사각지대'라는 원칙(게이트 vs PII 마스킹)은 게이트 **내부의
# 축들 사이에도** 똑같이 적용되어야 한다. planted(심은 오류) 케이스는 기대한
# 바로 그 축에 걸려야 하고(다른 축에 우연히 걸린 통과는 통과가 아니다),
# clean(정상) 케이스는 오탐 없이 통과해야 한다. 축 목록과 테이블의 커버리지가
# 어긋나면 테스트(test_preflight)가 실패한다 — 새 축을 추가하면 자가 테스트도
# 함께 추가하도록 구조로 강제한 것. 경고 축뿐 아니라 **비경고 라벨 축**
# (case_origin — 2계층 신뢰 소스의 케이스 유래 라벨)도 심어 본다: 라벨은
# 경고와 달리 죽어도 아무 소리가 나지 않으므로(응답은 계속 ok) 자가 테스트가
# 없으면 고장을 알 수 있는 신호 자체가 없다. 같은 이유로 한 축의 '표기 변형'
# (고유어 방향·부분 날짜)도 대표 1건씩 태운다 — 축이 살아 있어도 특정 표기의
# 수집이 죽으면 그 표기로만 오는 왜곡은 전부 조용히 통과한다.
_GATE_TOOL_LABELED = '{"awareness_date": "2026-07-10", "deadline_date": "2026-07-25"}'
_GATE_SELF_TESTS: tuple[dict, ...] = (
    {"name": "수치 존재·심은 오류", "answer": "보고 기한은 30일 이내입니다",
     "trusted": ["보고 기한은 15일 이내"], "expect_ok": False, "axis": "unsupported"},
    {"name": "수치 존재·정상", "answer": "보고 기한은 15일 이내입니다",
     "trusted": ["보고 기한은 15일 이내"], "expect_ok": True},
    {"name": "날짜 존재·심은 오류", "answer": "마감일은 2026-07-30 입니다",
     "trusted": [_GATE_TOOL_LABELED], "expect_ok": False, "axis": "unsupported"},
    {"name": "방향 한정어·심은 오류", "answer": "15일 이후에 보고하면 됩니다",
     "trusted": ["인지일로부터 15일 이내 신속보고"], "expect_ok": False, "axis": "direction_conflicts"},
    {"name": "고유어 방향 한정어·심은 오류", "answer": "보름 이후에 보고하면 됩니다",
     "trusted": ["인지일로부터 15일 이내 신속보고"], "expect_ok": False, "axis": "direction_conflicts"},
    {"name": "날짜 방향 한정어·심은 오류", "answer": "2026-07-25 이후에 제출하면 됩니다",
     "trusted": ["보완자료는 2026-07-25까지 제출한다"], "expect_ok": False, "axis": "direction_conflicts"},
    # 케이스 간섭 케이스는 축(direction_conflicts) 발화만이 아니라 **from_case
    # 라벨의 생존**도 함께 단언한다(v9) — 축은 살아 있어도 라벨만 죽으면 케이스
    # 재서술에 '컴플라이언스 오류' 단정 문구가 붙는 오탐 방향으로 조용히
    # 퇴행하는데, 라벨은 ok 를 바꾸지 않아 다른 어떤 신호도 없다.
    {"name": "케이스 간섭 방향 한정어·심은 오류(2계층)",
     "answer": "15일 이후에 보고하면 됩니다",
     "trusted": ["인지일로부터 15일 이내 신속보고"],
     "user_facts": ["환자가 복용 15일 이후 증상 발생"],
     "expect_ok": False, "axis": "direction_conflicts", "label": "case_origin"},
    {"name": "부분 날짜 표기·심은 오류", "answer": "보고 기한은 7월 30일입니다",
     "trusted": [_GATE_TOOL_LABELED], "expect_ok": False, "axis": "unsupported"},
    {"name": "부분 날짜 표기·정상", "answer": "보고 기한은 7월 25일입니다",
     "trusted": [_GATE_TOOL_LABELED], "expect_ok": True},
    # v8 — 부분 날짜 표기의 방향·역할 축: 존재 축은 접미 대조로 살아 있는데
    # 방향·역할 축이 ISO 만 수집하면, 그 표기로 오는 왜곡은 존재 축의 지지를
    # 받은 채(supported) 통과한다 — 표기 변형 자가 테스트를 존재 축에만 두는
    # 것은 이 파일 자신의 원칙(축 내부의 비대칭 금지) 위반이었다.
    {"name": "부분 날짜 방향 한정어·심은 오류", "answer": "7월 25일 이후에 제출하면 됩니다",
     "trusted": ["보완자료는 2026-07-25까지 제출한다"], "expect_ok": False, "axis": "direction_conflicts"},
    {"name": "부분 날짜 역할 스왑·심은 오류", "answer": "보고 기한은 7월 10일입니다 (인지일 7월 25일)",
     "trusted": [_GATE_TOOL_LABELED], "expect_ok": False, "axis": "role_conflicts"},
    {"name": "부분 날짜 방향·역할 정상", "answer": "보고 기한은 7월 25일까지입니다 (인지일 7월 10일)",
     "trusted": [_GATE_TOOL_LABELED + " 기한 2026-07-25까지"], "expect_ok": True},
    # v8 — 하이픈 범위 표기: '-' 구분자를 뺀 대가로 상한·하한이 모두 미수집
    # → 위조 하한 포함 표현 전체가 조용히 통과하던 사각지대.
    {"name": "하이픈 범위·심은 오류", "answer": "처리기간은 10-15일입니다",
     "trusted": ["처리기간은 15일이다"], "expect_ok": False, "axis": "unsupported"},
    {"name": "하이픈 범위·정상", "answer": "처리기간은 10-15일입니다",
     "trusted": ["처리기간은 10-15일이다"], "expect_ok": True},
    # v8 — 방향 어휘 '이전'(상한)·'부터'(하한 기산점)
    {"name": "방향 한정어(이전)·심은 오류", "answer": "15일 이전에 보고해야 합니다",
     "trusted": ["인지일로부터 15일 이후 보고한다"], "expect_ok": False, "axis": "direction_conflicts"},
    {"name": "날짜 방향 한정어(부터)·심은 오류", "answer": "2026-07-25부터 제출 가능합니다",
     "trusted": ["제출 기한은 2026-07-25까지"], "expect_ok": False, "axis": "direction_conflicts"},
    {"name": "연도 표기·정상", "answer": "기한 산정은 2026년 도구 계산 기준입니다",
     "trusted": [_GATE_TOOL_LABELED], "expect_ok": True},
    {"name": "케이스 유래 지지·정상 라벨(2계층)",
     "answer": "케이스상 복용 기간은 30일입니다",
     "trusted": ["보고 기한은 15일 이내"],
     "user_facts": ["환자가 30일간 복용 후 두드러기 발생"],
     "expect_ok": True, "label": "case_origin"},
    # 웹 계층(3계층) — 인터넷 검색 결과에서'만' 지지되는 클레임은 경고 대신
    # from_web(web_origin) 라벨이 붙는다: 명시 요청된 웹 검색의 재서술은
    # 정당하되(경고하면 웹 답변마다 오탐), 사내 규정 근거로 승격되지도 않는다.
    # 라벨 축이므로 죽어도 소리가 없다 — case_origin 과 같은 이유로 심어 본다.
    {"name": "웹 유래 지지·정상 라벨(3계층)",
     "answer": "인터넷 검색 결과에 따르면 개정 고시는 2026-05-01 시행 예정입니다",
     "trusted": ["보고 기한은 15일 이내"],
     "web_texts": ["개정 고시 2026-05-01 시행 예정 (뉴스)"],
     "expect_ok": True, "label": "web_origin"},
    # 웹 계층이 있어도 어느 계층에도 없는 수치는 여전히 경고다 — '웹 검색을
    # 썼으니 검증 게이트를 통째로 꺼도 되지 않느냐'는 방향의 퇴행(게이트
    # 우회)을 기동 전에 잡는 심은 오류.
    {"name": "웹 계층 존재·심은 오류", "answer": "보고 기한은 30일 이내입니다",
     "trusted": ["보고 기한은 15일 이내"],
     "web_texts": ["관련 없는 웹 검색 결과 본문"],
     "expect_ok": False, "axis": "unsupported"},
    {"name": "날짜 역할 스왑·심은 오류", "answer": "보고 기한: 2026-07-10 (인지일 2026-07-25 기준)",
     "trusted": [_GATE_TOOL_LABELED], "expect_ok": False, "axis": "role_conflicts"},
    {"name": "날짜 역할·정상", "answer": "보고 기한: 2026-07-25 (인지일 2026-07-10 기준)",
     "trusted": [_GATE_TOOL_LABELED], "expect_ok": True},
    {"name": "폐지본 인용·심은 오류", "answer": "이상사례는 30일 이내 보고한다",
     "trusted": ["30일 이내"], "citations": [{"doc_id": "REG-013", "status": "superseded"}],
     "expect_ok": False, "axis": "superseded_cited"},
    {"name": "폐지본 인용·이력 모드 정상", "answer": "구판 기준은 30일이었다",
     "trusted": ["30일"], "citations": [{"doc_id": "REG-013", "status": "superseded"}],
     "allow_superseded": True, "expect_ok": True},
    {"name": "전제 에코 라벨", "answer": "30일이 아니라 15일 이내입니다",
     "trusted": ["인지일로부터 15일 이내"], "question": "보고 기한이 30일 맞나요?",
     "expect_ok": False, "axis": "question_origin"},
    # v9 — 단위 표기 변형('주'+어미 직결): 존재 축이 살아 있어도 "2주간"·
    # "2주입니다" 표기의 수집이 죽으면(종전 전면 한글 배제 lookahead) 그
    # 표기로 오는 "15일 → 약 2주간" 환산 위조는 전부 조용히 통과하고, 근거의
    # "2주간"은 옳은 답변 "2주"에 오탐을 만든다 — 양방향을 함께 태운다.
    {"name": "단위 표기(주간)·심은 오류", "answer": "보고 기한은 약 2주간입니다",
     "trusted": ["보고 기한은 15일 이내"], "expect_ok": False, "axis": "unsupported"},
    {"name": "단위 표기(주간)·정상", "answer": "안정성 시험은 2주 이내에 완료합니다",
     "trusted": ["안정성 시험은 2주간 이내에 완료한다"], "expect_ok": True},
    # v9 — 방향 축의 전제 정정: 질문의 틀린 방향 전제를 정정하는 옳은 답변은
    # 그 전제를 재서술할 수밖에 없다 — 경고는 유지하되(정정인지 왜곡인지
    # 기계는 모른다) 방향 블록의 from_question 문구로 종류가 조정되는지 확인한다.
    # v10: 방향 클레임은 question_origin(수치·날짜 전제 축)으로 새면 '⚠ 전제
    # 확인 필요: 수치 …' 로 오분류·이중 경고되므로, 완화가 방향 블록에서
    # 나오고(전제·방향 문구) 수치 전제 블록으로는 안 새는지를 warn_has/warn_not
    # 문자열로 함께 태운다(축 × 표기 오분류의 재발 방지 핀).
    {"name": "방향 전제 정정 라벨", "answer": "아니요, 15일 이후가 아니라 15일 이내입니다",
     "trusted": ["인지일로부터 15일 이내 신속보고"],
     "question": "신속보고는 15일 이후에 하면 되나요?",
     "expect_ok": False, "axis": "direction_conflicts",
     "warn_has": ["전제", "방향"], "warn_not": ["전제 확인 필요: 수치"]},
)


def check_config() -> list[str]:
    """하이퍼파라미터 간 불변식 — 값 각각이 아니라 '조합'의 모순을 잡는다."""
    problems: list[str] = []
    # 공개 배포 인증 가드 — MCP 서버를 공개 URL 로 여는 배포(MCP_REQUIRE_AUTH=1)
    # 에서 토큰이 비면 '무인증 공개 서버'라는 조용한 보안 결함이 된다. 값 각각은
    # 유효한 조합 모순이므로 여기서 잡는다(다른 불변식과 같은 계열).
    if config.MCP_REQUIRE_AUTH and not config.MCP_AUTH_TOKEN:
        problems.append(
            "MCP_REQUIRE_AUTH=1 인데 MCP_AUTH_TOKEN 이 비어 있다 — 공개 MCP 서버를"
            " 무인증으로 열 수 없다(토큰을 설정하거나 공개 가드를 끌 것)"
        )
    if config.JUDGE_ENABLED and not config.ANTHROPIC_API_KEY:
        # LLM judge 를 켰는데 키가 없으면 모든 응답의 judge 가 unavailable 로
        # 도는 '고장난 안전장치' 배포가 된다 — 조용히 내려가는 대신 기동을
        # 거부한다(fail-closed, MCP_REQUIRE_AUTH 가드와 같은 계열).
        problems.append(
            "LLM_JUDGE=1 인데 ANTHROPIC_API_KEY 가 없다 — judge 2차 검증은 API 키가"
            " 필요하다(키를 설정하거나 LLM_JUDGE 를 끌 것)"
        )
    if not (0 < config.CHUNK_OVERLAP < config.CHUNK_SIZE):
        problems.append(
            f"CHUNK_OVERLAP({config.CHUNK_OVERLAP})은 0보다 크고 CHUNK_SIZE({config.CHUNK_SIZE})보다 작아야 한다"
        )
    if not (0.0 <= config.HYBRID_ALPHA <= 1.0):
        problems.append(f"HYBRID_ALPHA({config.HYBRID_ALPHA})는 [0,1] 범위여야 한다")
    if not (0.0 <= config.RERANK_WEIGHT <= 1.0):
        problems.append(f"RERANK_WEIGHT({config.RERANK_WEIGHT})는 [0,1] 범위여야 한다")
    if config.RERANK_TOP_N > config.RETRIEVE_TOP_K:
        problems.append(
            f"RERANK_TOP_N({config.RERANK_TOP_N})이 RETRIEVE_TOP_K({config.RETRIEVE_TOP_K})보다 크다"
            " — 리랭킹이 1차 회수보다 많이 반환할 수 없다"
        )
    if config.RERANK_TOP_N < 1 or config.RETRIEVE_TOP_K < 1:
        problems.append("RERANK_TOP_N/RETRIEVE_TOP_K 는 1 이상이어야 한다")
    if config.EMBEDDER_KIND not in ("auto", "dense", "tfidf", "hashing", "voyage"):
        problems.append(
            f"EMBEDDER_KIND('{config.EMBEDDER_KIND}')는 auto|dense|tfidf|hashing|voyage 중 하나여야 한다"
        )
    if config.EMBEDDER_KIND == "voyage" and not os.environ.get("VOYAGE_API_KEY", ""):
        problems.append("EMBEDDER_KIND=voyage 인데 VOYAGE_API_KEY 가 없다")
    if config.EMBEDDER_KIND == "dense":
        # dense 는 명시 요구(폴백 없음) — 기동 후 수백 MB 다운로드로 블록되거나
        # 죽기 전에, 로컬 캐시에서 로드 가능함을 여기서 확인한다(voyage 키 검사와
        # 같은 계열: 명시 설정의 전제조건 점검). lru_cache 라 스모크가 재사용한다.
        try:
            from .rag.embedder import _load_st_model

            _load_st_model(config.EMBEDDING_MODEL, True)
        except Exception as e:
            problems.append(
                f"EMBEDDER_KIND=dense 인데 밀집임베딩 모델 로드 불가({type(e).__name__}) — "
                "pip install -r requirements-ml.txt 후 "
                "python scripts/download_embedding_model.py 로 모델을 받아둘 것"
            )
    if not (0.0 <= config.DENSE_SIGNAL_FLOOR < 1.0):
        problems.append(f"DENSE_SIGNAL_FLOOR({config.DENSE_SIGNAL_FLOOR})는 [0,1) 범위여야 한다")
    if config.AGENT_BACKEND not in ("direct", "pydantic_ai", "langgraph"):
        problems.append(
            f"AGENT_BACKEND('{config.AGENT_BACKEND}')는 direct|pydantic_ai|langgraph 중 하나여야 한다"
        )
    if config.AGENT_BACKEND == "langgraph":
        # 명시 설정의 전제조건 점검(voyage 키·dense 모델 검사와 같은 계열) —
        # LLM 모드 첫 요청에서 ImportError 로 죽는 대신 기동 전에 잡는다.
        try:
            import langgraph  # noqa: F401
        except ImportError:
            problems.append(
                "AGENT_BACKEND=langgraph 인데 langgraph 미설치 — pip install -r requirements.txt 재실행"
            )
    return problems


def check_corpus(reg_dir: Path | None = None) -> list[str]:
    """규제문서 frontmatter 계약 + 폐지 체인 무결성."""
    from .rag.loader import load_documents

    reg_dir = reg_dir or config.REG_DIR
    problems: list[str] = []
    docs = load_documents(reg_dir)
    if not docs:
        return [f"규제문서가 없다: {reg_dir}"]

    seen_ids: dict[str, str] = {}
    all_ids = {d.metadata.get("doc_id") for d in docs}
    # 미래 시행일 검사(v9)의 재료 — '아직 시행 전인 현행(active) 문서'는 오늘
    # 기준 검색에서 제외되는 것이 옳다(기본 검색 = as_of 오늘). 그 문서가
    # 시행 중인 구판(그 문서를 superseded_by 로 가리키는, 시행일이 지난 폐지본)
    # 없이 단독 존재하면, 시행일까지 그 주제는 검색에서 **아예 나오지 않는다**
    # — 스키마는 전부 유효한 채로 주제 하나가 조용히 사라지는 형태라 형식
    # 검사로는 영원히 통과한다(마감일 전건 과거 감지와 같은 계열: 형식과 값
    # 타당성은 다른 층이다).
    today = _dt.date.today()
    in_force_successor_of: set[str] = set()  # '시행 중 구판'이 가리키는 후속본 doc_id
    for d in docs:
        eff_s = str(d.metadata.get("effective_date", ""))
        by = str(d.metadata.get("superseded_by") or "")
        if str(d.metadata.get("status", "")) == "superseded" and by and eff_s:
            try:
                if _dt.date.fromisoformat(eff_s) <= today:
                    in_force_successor_of.add(by)
            except ValueError:
                pass  # 형식 오류는 별도 검사가 보고한다
    for d in docs:
        meta = d.metadata
        missing = [k for k in _REQUIRED_META if not meta.get(k)]
        if missing:
            problems.append(f"{d.source}: frontmatter 필수 필드 누락 {missing}")
        doc_id = meta.get("doc_id")
        if doc_id:
            if doc_id in seen_ids:
                problems.append(f"{d.source}: doc_id '{doc_id}' 중복 ({seen_ids[doc_id]}와 충돌)")
            seen_ids[doc_id] = d.source
        status = str(meta.get("status", ""))
        if status not in _VALID_STATUS:
            problems.append(f"{d.source}: status '{status}' 는 허용 어휘 {_VALID_STATUS} 밖")
        eff = str(meta.get("effective_date", ""))
        if eff:
            try:
                eff_date = _dt.date.fromisoformat(eff)
            except ValueError:
                problems.append(f"{d.source}: effective_date '{eff}' 가 YYYY-MM-DD 형식이 아님")
            else:
                # 미래 시행일 active 문서는 '시행 중인 구판'이 함께 있어야 한다 —
                # 없으면 시행일까지 이 주제가 오늘 기준 검색에서 통째로 사라진다
                # (개정 예고본만 넣고 현행본을 빼는 데이터 교체 실수의 형태).
                if (
                    eff_date > today
                    and status != "superseded"
                    and str(doc_id) not in in_force_successor_of
                ):
                    problems.append(
                        f"{d.source}: 시행일({eff})이 미래인 현행 문서인데 시행 중인 구판이 없다"
                        " — 시행 전까지 이 주제는 검색에서 나오지 않는다(구판을 함께 두거나 시행일 확인)"
                    )
        if not d.text.strip():
            problems.append(f"{d.source}: 본문이 비어 있음")
        # 폐지 체인: 폐지본은 후속 문서를 가리켜야 하고, 체인을 따라가면
        # 실존하는 현행(active) 종점에 도달해야 한다. 단절된 체인은 '이력
        # 조회'와 '현행 대체본 안내'를 조용히 망가뜨린다.
        #
        # v8 — 체인은 '다단 순회'로 검사한다. superseded_by 규약은 최종
        # 현행본이 아니라 **직전 후속본**이다: 리트리버의 as_of 구간 판정
        # [시행일, 후속본 시행일) 이 이 규약을 전제한다(최종본 직결로 쓰면
        # 중간 구간에서 두 버전이 동시에 '당시 현행'으로 반환된다). 그런데
        # 종전 검사는 '후속본이 폐지본이면 결함'이라 정당한 2단 체인
        # (구판→중간판(폐지)→현행)을 거부했다 — 리트리버와 preflight 가
        # 서로 반대 규약을 요구해, 다단 이력을 넣는 순간 어느 쪽으로 써도
        # 한쪽이 깨지는 잠복 상충. 각 링크의 시행일 단조성과 순환도 함께 본다.
        if status == "superseded":
            if not meta.get("superseded_by"):
                problems.append(f"{d.source}: superseded 인데 superseded_by 가 없다")
            else:
                by_id = {str(x.metadata.get("doc_id")): x for x in docs}
                visited = [str(meta.get("doc_id"))]
                cur, cur_eff = d, eff
                while True:
                    nxt_id = str(cur.metadata.get("superseded_by") or "")
                    if not nxt_id:
                        problems.append(
                            f"{d.source}: 체인의 '{visited[-1]}' 가 superseded 인데 superseded_by 가 없다"
                        )
                        break
                    if nxt_id in visited:
                        problems.append(
                            f"{d.source}: 폐지 체인에 순환이 있다 ({' → '.join(visited)} → {nxt_id})"
                        )
                        break
                    if nxt_id not in all_ids:
                        problems.append(f"{d.source}: superseded_by '{nxt_id}' 문서가 코퍼스에 없다")
                        break
                    visited.append(nxt_id)
                    nxt = by_id[nxt_id]
                    nxt_eff = str(nxt.metadata.get("effective_date", ""))
                    try:
                        if (
                            nxt_eff and cur_eff
                            and _dt.date.fromisoformat(nxt_eff) <= _dt.date.fromisoformat(cur_eff)
                        ):
                            problems.append(
                                f"{d.source}: 후속본 '{nxt_id}' 시행일({nxt_eff})이 구판 시행일({cur_eff})보다"
                                " 늦지 않다 — 시점(as_of) 조회의 구간 판정이 깨진다"
                            )
                    except ValueError:
                        pass  # 날짜 형식 오류는 위의 형식 검사가 별도로 보고한다
                    if str(nxt.metadata.get("status", "")) != "superseded":
                        break  # active 종점 도달 — 정상 체인
                    cur, cur_eff = nxt, nxt_eff
    return problems


def check_tasks(tasks_file: Path | None = None) -> list[str]:
    """RA·PV 업무 데이터(ra_tasks.json) 스키마 — 도구가 읽다가 죽거나 오답을 내기 전에."""
    tasks_file = tasks_file or config.RA_TASKS_FILE
    problems: list[str] = []
    try:
        data = json.loads(tasks_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"{tasks_file.name}: 읽기/파싱 실패 — {e}"]

    parsed_dues: list[_dt.date] = []
    for i, d in enumerate(data.get("deadlines", [])):
        missing = [k for k in ("item", "due_date", "type", "owner", "status") if not d.get(k)]
        if missing:
            problems.append(f"deadlines[{i}]: 필수 필드 누락 {missing}")
        due = str(d.get("due_date", ""))
        if due:
            try:
                parsed_dues.append(_dt.date.fromisoformat(due))
            except ValueError:
                problems.append(f"deadlines[{i}]: due_date '{due}' 가 YYYY-MM-DD 형식이 아님")
    # 시한부 데이터의 조용한 부패 감지 — 샘플 마감일이 전부 과거가 되면 "이번 주
    # 마감" 데모 질의가 전항목 연체 목록으로 답해 서사가 조용히 죽는다(스키마는
    # 유효하므로 형식 검사만으로는 영원히 통과한다 — 형식/값 타당성은 다른 층).
    # 과거 마감 '일부'는 의도된 연출(지남/긴급 표시)이므로 전건 과거일 때만 결함.
    if parsed_dues and all(due < _dt.date.today() for due in parsed_dues):
        problems.append(
            f"deadlines: 마감일 {len(parsed_dues)}건이 모두 과거 — 시한부 샘플 데이터가 부패했다."
            f" {tasks_file.name} 의 due_date 를 미래로 갱신할 것(데모 서사 유지)"
        )
    checklists = data.get("checklists", {})
    if not checklists:
        problems.append("checklists 가 비어 있다")
    for cat, items in checklists.items():
        if not items:
            problems.append(f"checklists['{cat}'] 가 비어 있다")
    return problems


def smoke_checks() -> list[str]:
    """실제 파이프라인 canary + 검증 게이트 자가 테스트."""
    from .mcp_server.server import assess_adverse_event, search_regulations
    from .verify.verifier import verify_answer, warning_text

    problems: list[str] = []

    # (a) RAG canary: 핵심 규정 질의가 정답 문서를 1위로 회수하는가
    res = search_regulations(_CANARY_QUERY, top_n=1).get("results", [])
    if not res:
        problems.append(f"RAG canary 실패: '{_CANARY_QUERY}' 검색 결과 0건")
    elif res[0].get("doc_id") != _CANARY_DOC:
        problems.append(
            f"RAG canary 실패: '{_CANARY_QUERY}' 1위가 {res[0].get('doc_id')} (기대: {_CANARY_DOC})"
            " — 코퍼스/인덱스/하이퍼파라미터 변경이 핵심 검색을 깨뜨렸다"
        )

    # (a') 시점 조회 canary: 과거 시점(as_of)에는 당시 시행 중이던 구판이 나오는가
    hist = search_regulations(_CANARY_QUERY, top_n=1, as_of=_CANARY_AS_OF).get("results", [])
    if not hist:
        problems.append(
            f"시점 조회 canary 실패: as_of={_CANARY_AS_OF} 검색 결과 0건"
            " — 당시 시행 중이던 버전이 필터에서 전부 걸러졌다"
        )
    elif hist[0].get("doc_id") != _CANARY_AS_OF_DOC:
        problems.append(
            f"시점 조회 canary 실패: as_of={_CANARY_AS_OF} 1위가 {hist[0].get('doc_id')}"
            f" (기대: {_CANARY_AS_OF_DOC} — 그 시점의 현행 버전)"
        )

    # (b) PV canary: 대표 케이스(입원=중대)가 인지일+15일로 트리아지되는가
    tri = assess_adverse_event(_CANARY_CASE, awareness_date="2026-01-01")
    if not tri.get("is_serious"):
        problems.append("PV canary 실패: 입원 케이스가 비중대로 판정됨")
    elif tri.get("deadline_days") != 15:
        problems.append(f"PV canary 실패: 보고기한 {tri.get('deadline_days')}일 (기대: 15일)")
    elif tri.get("deadline_date") != "2026-01-16":
        problems.append(
            f"PV canary 실패: 마감일 {tri.get('deadline_date')} (기대: 2026-01-16 = 인지일+15일)"
        )

    # (c) 검증 게이트 자가 테스트: 게이트가 고장난 채 배포되면 안전장치가
    #     없는 것보다 나쁘다(있다고 믿게 만들므로). 런타임 게이트의 모든 경고
    #     축(존재·방향·역할·버전·전제 라벨)에 대해 '심은 오류는 그 축에 걸리고,
    #     정상 케이스는 통과한다'를 테이블로 태워 본다(_GATE_SELF_TESTS).
    for t in _GATE_SELF_TESTS:
        v = verify_answer(
            t["answer"],
            t["trusted"],
            t.get("citations"),
            t.get("allow_superseded", False),
            question=t.get("question", ""),
            user_fact_texts=t.get("user_facts"),
            web_texts=t.get("web_texts"),
        )
        s = v.summary()
        if t["expect_ok"]:
            if not v.ok:
                problems.append(f"검증 게이트 자가 테스트 실패[{t['name']}]: 정상 케이스에 오탐 — {s}")
            elif t.get("label") and not s.get(t["label"]):
                # 경고는 아니지만 붙어야 하는 등급 라벨(case_origin) — 라벨이
                # 조용히 죽으면 사용자 서술 유래 지지가 규정 근거처럼 읽힌다.
                problems.append(
                    f"검증 게이트 자가 테스트 실패[{t['name']}]: 기대 라벨({t['label']})이 붙지 않았다 — {s}"
                )
        elif v.ok:
            problems.append(f"검증 게이트 자가 테스트 실패[{t['name']}]: 심은 오류가 통과됨")
        elif not s.get(t["axis"]):
            problems.append(
                f"검증 게이트 자가 테스트 실패[{t['name']}]: 기대 축({t['axis']})이 아닌"
                f" 다른 축에 걸렸다 — 해당 축의 탐지가 죽었을 수 있음: {s}"
            )
        elif t.get("label") and not s.get(t["label"]):
            # 경고 축이 발화해도 함께 붙어야 하는 완화·등급 라벨(from_case 의
            # case_origin, from_question 의 question_origin)이 죽으면 경고
            # '종류'가 조용히 퇴행한다 — 라벨은 ok 를 안 바꿔 소리가 없다.
            problems.append(
                f"검증 게이트 자가 테스트 실패[{t['name']}]: 경고 축은 발화했으나"
                f" 기대 라벨({t['label']})이 붙지 않았다 — {s}"
            )
        # 경고 '문구' 자체의 오분류·이중 경고 감시(v10) — 방향·역할의
        # from_question 완화가 수치 전제 블록으로 새면 warn_not 이, 방향 블록의
        # 완화가 죽으면 warn_has 가 잡는다. 축(summary)만으로는 안 보이는 결함.
        if t.get("warn_has") or t.get("warn_not"):
            wt = warning_text(v)
            for sub in t.get("warn_has", []):
                if sub not in wt:
                    problems.append(
                        f"검증 게이트 자가 테스트 실패[{t['name']}]: 경고 문구에"
                        f" '{sub}' 가 없다(완화·축 문구 퇴행) — {wt!r}"
                    )
            for sub in t.get("warn_not", []):
                if sub in wt:
                    problems.append(
                        f"검증 게이트 자가 테스트 실패[{t['name']}]: 경고 문구에"
                        f" '{sub}' 가 있다(오분류·이중 경고) — {wt!r}"
                    )

    # (d) PII 마스킹 자가 테스트: 안전장치 자가 검사를 검증 게이트에만 하고
    #     입구 경계(마스킹)에는 안 하는 비대칭이 사각지대였다 — 마스킹이 고장난
    #     채 배포되면 첫 실사용 케이스의 개인정보가 외부 API·로그로 나간다.
    #     심는 표기에는 '한글 직결 변형'("…는 010-…로", "주민번호는 …입니다")을
    #     포함한다 — 검증 게이트의 표기 변형 자가 테스트와 같은 원리로, 축이
    #     살아 있어도 특정 표기(정규식 \b 는 한글 직결에서 경계가 성립하지
    #     않는다)의 수집이 죽으면 그 표기로 들어온 개인정보만 조용히 샌다.
    from .pv.redactor import redact

    #     v8 추가: 호칭 뒤 '조사' 직결("박영희님이" — 괄호·공백 표기만 심으면
    #     한국어에서 가장 흔한 조사 표기의 수집이 죽어도 못 잡는다),
    #     외국인등록번호(성별코드 5~8), en-dash(–) 전화, 띄어 쓴 "환자 번호".
    #     v9 추가: 첫 글자가 종전 허용 집합 밖인 조사("…님으로부터" — '으로'는
    #     한국어 최빈 조사 계열인데 룩어헤드 집합이 첫 글자 단위라 통째로
    #     샜다), 점(.) 구분 주민번호("900101.1234567" — 전화 패턴만 점을
    #     허용하던 같은 파일 안 구분자 비대칭).
    #     v11 추가: 구분자 양옆 공백 전화("010 - 7777 - 8888" — 주민번호는 공백
    #     패딩을 받는데 전화만 단일 구분자라 통째로 샜다), 전각 숫자 주민번호·전화
    #     ("８８０８０８－…"·"０１０－…" — `\d`는 전각을 받지만 성별코드 [1-8]·앞자리
    #     ASCII 앵커가 못 받아 유출, NFKC 진입 정규화로 봉합). 전각은 NFKC 후
    #     반각이 되므로 마스킹 실패 시 반각 표기로 남는다 → 반각으로 심어 검사.
    probe = redact(
        "환자 김철수님(연락처 010-1234-5678, 주민번호 900101-1234567) 케이스. "
        "보호자 전화는 010-9876-5432로, 주민번호는 850505-2345678입니다. "
        "박영희님이 동행했고 외국인등록번호는 900101-5234567, "
        "연락처 010–2222–3333, 환자 번호: A-1023. "
        "이민준님으로부터 문의가 왔고 주민번호는 920202.1234567. "
        "긴급 연락처 010 - 7777 - 8888 로 회신 바람. "
        "전각 표기: 주민번호 ８８０８０８－１２３４５６７, 전화 ０１０－４４４４－５５５５."
    )
    for planted in (
        "김철수", "010-1234-5678", "900101-1234567",
        "010-9876-5432", "850505-2345678",  # 한글 직결 표기 변형
        "박영희", "900101-5234567", "010–2222–3333", "A-1023",  # v8 표기 변형
        "이민준", "920202.1234567",  # v9 표기 변형(조사 집합 밖·점 구분자)
        "7777 - 8888", "880808-1234567", "4444-5555",  # v11(공백 패딩·전각 NFKC)
    ):
        if planted in probe.text:
            problems.append(f"PII 마스킹 자가 테스트 실패: 심은 개인정보 '{planted}' 가 마스킹되지 않았다")

    # (d') 이력 마스킹의 '표기 변형' — 대화 이력의 content 는 문자열만이 아니라
    #      anthropic 블록 리스트(content=[{"type":"text",...}])로도 들어온다.
    #      문자열 표기만 마스킹되고 블록 표기가 통과하면, 그 표기로 들어온
    #      개인정보만 조용히 외부 LLM API 로 샌다(v7 발견 — 한글 직결 \b 경계와
    #      동형의 표기 우회). 입구 경계의 표기 변형도 기동 전에 태워 본다.
    from .agent.agent import _redact_history

    turns = _redact_history(
        [{"role": "user", "content": [{"type": "text", "text": "보호자 연락처는 010-5555-6666으로"}]}]
    )
    block_text = turns[0]["content"][0].get("text", "")
    if "010-5555-6666" in block_text:
        problems.append(
            "PII 마스킹 자가 테스트 실패: 이력 블록 표기(content=[{type:text}]) 속 개인정보가 마스킹되지 않았다"
        )
    return problems


def check_mcp_reachable(
    timeout_s: float = 60.0,
    probe=None,
    poll_interval_s: float = 2.0,
) -> list[str]:
    """API 롤 전용 — 분리된 MCP 서버 기동을 기다린 뒤 도구 7종 노출까지 실연결로 검증.

    완전 분리 구조에서 API 컨테이너의 가장 흔한 Day-0 실패는 코드가 아니라
    'MCP 서버가 아직 안 떴다/URL 이 틀렸다'이다 — wait-loop 로 기동 순서를
    흡수하고, list_tools 실왕복으로 도구 계약까지 확인한다.

    도구 '누락'도 deadline 안에서 재시도한다(v12) — 무중단 롤링 배포에서는
    새 MCP 인스턴스가 준비될 때까지 **구버전 인스턴스가 정상 응답**하므로,
    새 도구를 추가한 커밋의 API 가 먼저 점검을 돌면 '연결 성공 + 기대 도구
    없음'이 과도기의 정상 상태다. '아직 안 떴다'는 기다리면서 '아직
    구버전이다'는 즉시 실패하는 것은 wait-loop 의 비대칭이었다(search_web
    추가 배포에서 실측 — rapv-mcp 재배포 완료 전에 rapv-assistant 의
    preflight 가 구버전 도구 목록을 보고 기동을 거부했다).

    Args:
        timeout_s: 연결·도구 계약 충족을 기다리는 상한(초).
        probe: 도구 이름 집합을 반환하는 호출자(테스트 봉합선) — 기본은
            fastmcp Client 로 MCP_SERVER_URL 에 실연결하는 프로브.
        poll_interval_s: 재시도 간격(초) — 테스트가 줄여 쓴다.
    """
    import asyncio
    import time as _time

    expected = {
        "search_regulations", "get_ra_deadlines", "get_submission_checklist",
        "assess_adverse_event", "draft_ae_report", "list_regulation_documents",
        "search_web",
    }

    if probe is None:
        from fastmcp import Client

        async def _probe() -> set[str]:
            async with Client(config.MCP_SERVER_URL, auth=config.MCP_AUTH_TOKEN or None) as c:
                return {t.name for t in await c.list_tools()}

        probe = lambda: asyncio.run(_probe())  # noqa: E731 - 기본 프로브 봉합선

    deadline = _time.monotonic() + timeout_s
    last = ""
    missing: set[str] | None = None
    while True:
        try:
            missing = expected - probe()
            if not missing:
                return []
            last = ""  # 연결은 성공 — 버전 스큐 대기로 전환
        except Exception as e:  # noqa: BLE001 - 기동 대기 중 연결 실패는 재시도
            missing = None
            last = type(e).__name__
        if _time.monotonic() >= deadline:
            break
        _time.sleep(poll_interval_s)
    if missing:
        return [
            f"MCP 서버 도구 누락: {sorted(missing)} — 서버는 응답하지만 기대 도구가 없다."
            " MCP 서비스가 아직 구버전 코드로 떠 있을 가능성(롤링 배포 버전 스큐)이"
            " 크다: MCP 서비스(rapv-mcp / compose mcp)의 새 버전 배포 완료를 확인한 뒤"
            " API 를 재기동/재배포할 것"
        ]
    # 실패 시 '주소가 어디서 왔는지'까지 진단한다 — 실배포의 최빈 원인은 연결
    # 자체가 아니라 주소 주입 실패(플랫폼 env 미주입 → 로컬 기본값 폴백)였다.
    env_src = (
        "MCP_SERVER_HOST 환경변수" if os.environ.get("MCP_SERVER_HOST", "").strip()
        else "MCP_SERVER_URL 환경변수" if os.environ.get("MCP_SERVER_URL", "").strip()
        else "환경변수 미주입 → 로컬 기본값 폴백(플랫폼의 MCP 주소 주입이 누락된 신호)"
    )
    return [
        f"MCP 서버({config.MCP_SERVER_URL}) 연결 실패 — 마지막 오류: {last} · 주소 출처: {env_src}"
    ]


def check_judge() -> list[str]:
    """LLM judge 자가 테스트 — **JUDGE_ENABLED 일 때만** 실행되는 조건부 그룹.

    게이트 자가 테스트(_GATE_SELF_TESTS)에 합류시키지 않는 이유: 그 테이블은
    키 없는 부팅에서도 전 축이 돌아야 하는 결정론 계약이고, judge 는 실 API
    가 필요한 확률 계층이다 — 축 정본 밖에 둔 설계(llm_judge.py)와 대칭으로,
    자가 테스트도 별도 그룹으로 이원화한다. 검사의 이원화: 배선·파싱 계약
    (비활성 기본·malformed 흡수·비차단)은 tests/test_judge.py 가 스텁으로
    핀 고정하고, 여기서는 '살아있는 판정 능력'만 canary 2건(planted 부정문
    뒤집기 1 + clean 1)으로 확인한다 — 안전장치가 고장난 채 배포되는 것은
    안전장치가 없는 것보다 나쁘다(게이트·PII 자가 테스트와 같은 원칙).
    enabled 부팅마다 실 API 2콜이 드는 것은 그 원칙의 수용된 비용이다.
    """
    from .verify.llm_judge import judge_answer

    problems: list[str] = []
    trusted = ["중대한 이상사례는 보고 대상이다. 인지일로부터 15일 이내 신속보고한다."]
    planted = judge_answer("중대한 이상사례는 보고 대상이 아닙니다.", trusted)
    if planted["status"] == "unavailable":
        problems.append(f"judge 호출 불가({planted['reason']}) — 키·모델명을 확인할 것")
    elif planted["status"] != "flagged":
        problems.append("judge 가 심은 부정문 뒤집기를 탐지하지 못했다 — 판정 능력 canary 실패")
    clean = judge_answer("중대한 이상사례는 보고 대상이며 15일 이내 신속보고합니다.", trusted)
    if clean["status"] == "flagged":
        problems.append(f"judge 가 정상 답변에 오탐을 냈다({clean['flags']}) — 경고 피로 위험")
    return problems


def run_preflight(role: str = "all") -> dict[str, list[str]]:
    """롤별 그룹을 실행해 {그룹명: 문제 목록} 을 반환한다(빈 목록 = 통과).

    완전 분리 배포에서 점검 책임이 프로세스별로 갈린다:
      - all (기본): 종전과 동일 — 단일 환경(로컬 개발·CI)에서 전 그룹 자체 실행
      - mcp: MCP 서버 컨테이너 — 코퍼스·스모크(파이프라인은 이 프로세스 소유)
      - api: API 컨테이너 — 자기 데이터(ra_tasks.json) + MCP 서버 실연결 대기
    """
    checks: dict[str, list[str]] = {"설정 불변식": check_config()}
    if role in ("all", "mcp"):
        checks["코퍼스 무결성"] = check_corpus()
    if role in ("all", "api"):
        checks["업무 데이터 스키마"] = check_tasks()
    if role in ("all", "mcp"):
        checks["스모크(canary)"] = smoke_checks()
    if config.JUDGE_ENABLED and role in ("all", "api"):
        # judge 는 에이전트(_finalize) 계층에서 돌므로 api 롤의 책임이다.
        # 비활성(기본)이면 그룹 자체가 없다 — 키 없는 오프라인 부팅 무영향.
        checks["LLM judge 자가 테스트"] = check_judge()
    if role == "api":
        checks["MCP 연결"] = check_mcp_reachable()
    return checks


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="배포 전 점검(preflight)")
    parser.add_argument(
        "--role", choices=["all", "api", "mcp"], default="all",
        help="점검 롤 — all(기본: 전 그룹) | mcp(코퍼스·스모크) | api(업무 데이터·MCP 연결)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"배포 전 점검 (preflight, role={args.role}) — 데이터·설정·스모크·게이트 자가 테스트")
    print("=" * 60)
    # auto 는 환경에 따라 dense/tfidf 로 갈리므로 '실제로 무엇이 잡혔는지'를 명시한다
    # (폴백이 조용히 지나가면 dense 라고 믿은 배포가 tfidf 로 돌 수 있다).
    try:
        from .rag.embedder import get_embedder

        resolved = type(get_embedder(config.EMBEDDER_KIND)).__name__
    except Exception as e:
        resolved = f"로드 실패({type(e).__name__})"
    print(f"임베더: EMBEDDER_KIND={config.EMBEDDER_KIND} → {resolved}")
    report = run_preflight(role=args.role)
    n_problems = 0
    for group, problems in report.items():
        mark = "✓" if not problems else "✗"
        print(f"[{mark}] {group}")
        for p in problems:
            print(f"    - {p}")
        n_problems += len(problems)
    print("-" * 60)
    if n_problems:
        print(f"결과: 실패 — 문제 {n_problems}건. 서버를 띄우기 전에 해결해야 한다.")
        return 1
    print("결과: 통과 — 배포 가능.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
