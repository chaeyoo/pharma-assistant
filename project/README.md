# 🧬 RAPV-Assistant — 제약 규제업무(RA·PV)를 위한 RAG + MCP Agentic 어시스턴트

> 제약회사 **RA(인허가/규제업무)·PV(약물감시)** 담당자를 위한 규제문서 검색·업무 자동화
> AI 어시스턴트의 **작동하는 최소 데모(MVP)**.
> **Agentic AI + MCP 로 사내 시스템을 통합하는 구조 원리**로 설계 —
> RAG 최적화 · Agentic Workflow · Function Calling · MCP/FastMCP · FastAPI 를 한 프로젝트로 증명한다.

## 1. 문제 정의 (왜 RA·PV인가)

RA·PV 담당자는 규제문서의 바다에서 일한다 — "이 변경은 허가야 신고야?", "중대 이상사례는 며칠 안에 보고?"의
답은 고시·SOP 어딘가에 있지만 찾는 데 시간이 걸리고 틀리면 규제 리스크다. 제출 기한을 놓치면 컴플라이언스
사고이고, 허가 후에는 **이상사례 접수→중대성 판정→보고기한 계산→당국 보고**(PV)가 기다린다.
→ **① 규제문서를 근거와 함께 즉시 검색(RAG) + ② 마감·체크리스트·이상사례 처리 도구를 에이전트가 자율 호출(Agent+MCP)**.

## 2. 무엇을 하는가

| 사용자 질문 예 | 동작 | 기술 |
|---|---|---|
| "신약 품목허가 심사 며칠 걸려?" | 규제문서 검색 → 근거+출처 답변 | **RAG** (하이브리드+리랭킹+질의확장) |
| "복용 후 아나필락시스로 입원했어요. 언제까지 보고?" | 중대성 판정+기한 계산+인과성(WHO-UMC) 제안+코딩 | **MCP Tool** (assess_adverse_event) |
| "이 케이스 KAERS 보고서 초안 만들어줘" | 최소보고요건(ICH E2D) 검증+초안+보완 질문 | **MCP Tool** (draft_ae_report) |
| "이번 주 마감 임박한 규제 업무는?" | 마감일 D-day 정리 | **MCP Tool** (get_ra_deadlines) |
| (복합) "GMP 변경인데 뭘 준비하고 언제까지?" | 검색+체크리스트+마감일 조합 | **Agentic Workflow** |
| "최신 개정 동향 **인터넷에서 검색해줘**" (명시 요청) | 🌐 웹 검색 — 결과를 사내 근거와 분리 표시 | **MCP Tool** (search_web) |

인터넷 검색은 **명시와 분리** 원칙을 따른다 — 사용자가 명시적으로 요청한 턴에만 수행되고(사내
문서에서 못 찾은 질문은 웹 폴백 없이 그대로 '근거 없음'으로 답한다), 결과는 🌐 표시·별도
필드(web_results)·검증 라벨(web_origin)로 사내 규제문서 근거와 전 계층에서 구분된다.
모든 답변에 **출처(문서·섹션·버전)** 부착, 환자 **PII는 입구에서 마스킹**(외부 API·로그 비유출),
나가는 답변은 **사후 검증 게이트**(수치·날짜·방향·역할 ↔ 근거 대조)를 전수 통과, 기동 전에는
**preflight**가 데이터·설정·안전장치를 검사해 실패 시 기동을 차단한다.

## 3. 아키텍처

```mermaid
flowchart TB
    U["👤 RA·PV 담당자<br/>(웹 챗 UI)"] -->|질문| API["⚙️ FastAPI 백엔드<br/>/chat"]
    API --> AG["🤖 RA·PV 에이전트<br/>(Agentic Loop + Function Calling)"]

    AG -->|"도구 선택·호출<br/>(MCP 프로토콜)"| MCP["🔌 FastMCP 서버<br/>RAPV-Assistant"]

    subgraph MCP_TOOLS["MCP Tools / Resources / Prompts"]
        T1["search_regulations<br/>(규제문서 검색)"]
        T2["get_ra_deadlines<br/>(마감일)"]
        T3["get_submission_checklist<br/>(체크리스트)"]
        T4["assess_adverse_event<br/>(PV 트리아지: 중대성+기한<br/>+인과성+코딩)"]
        T5["draft_ae_report<br/>(ICSR 초안: 최소요건 검증<br/>+KAERS 초안)"]
        T6["list_regulation_documents<br/>(문서 목록)"]
        T7["search_web 🌐<br/>(인터넷 검색 —<br/>명시 요청 전용·결과 분리)"]
        R1["regulation://{doc_id}<br/>(문서 원문 Resource)"]
        P1["pv_case_intake<br/>(케이스 처리 SOP Prompt)"]
    end
    MCP --> MCP_TOOLS

    T1 --> RAG["📚 RA 도메인 ①<br/>규제문서 검색 (RAG)"]
    T4 -.->|"판정 근거 부착"| RAG
    T5 -.->|"판정 근거 부착"| RAG
    T5 --> PV["💊 PV 도메인 케이스 처리<br/>트리아지→인과성(WHO-UMC)<br/>→코딩(MedDRA 방식)→최소요건 검증"]
    T4 --> PV
    subgraph RAG_PIPE["RAG (검색 최적화)"]
        C1["청킹<br/>(구조+overlap)"] --> C2["임베딩<br/>(밀집 e5 기본 ·<br/>TF-IDF 폴백, pluggable)"]
        C0["질의확장<br/>(도메인 동의어)"] --> C3["하이브리드 검색<br/>벡터 + BM25"]
        C2 --> VS["벡터 스토어<br/>(인메모리 기본 ·<br/>pgvector 영속+증분 색인)"]
        VS --> C3
        C3 --> C4["리랭킹 v3<br/>(Cross-Encoder 근사<br/>4신호+섹션 타입 prior<br/>·원질의+확장 보조)"]
    end
    RAG --> RAG_PIPE
    RAG_PIPE --> DOCS["📄 규제문서<br/>(품목허가·GMP·라벨링·PV·임상)"]

    T2 --> DB["🗂 RA·PV 업무 데이터<br/>(src/ra/ · 마감일/체크리스트)"]
    T3 --> DB

    AG -->|"키 있으면: 근거 종합 답변 생성"| LLM["🧠 Enterprise LLM API<br/>(Claude · 선택)"]
    AG -.->|"키 없으면: 규칙 라우팅"| FB["오프라인 폴백<br/>(grounded 추출 답변)"]

    AG --> VER["🔍 답변 사후 검증<br/>(수치·날짜·방향·날짜 역할<br/>↔ 근거 대조 + 폐지본 인용 점검)"]
    VER -->|"답변 + 출처 + 도구트레이스<br/>+ 검증 결과"| API --> U
```

**핵심 설계 포인트:** 모델(에이전트)과 도구(RA·PV 업무 시스템)가 **MCP 규격으로 분리** —
한 번 만든 도구를 Claude Desktop·Cursor·사내 에이전트 어디서든 재사용한다.

## 4. 기술 스택

| 핵심 역량 | 이 프로젝트에서 | 위치 |
|---|---|---|
| **RAG 최적화** | 구조 청킹, 하이브리드(밀집임베딩 e5+BM25 — TF-IDF 자동 폴백), 4신호 리랭킹+섹션 prior, 질의확장, 스윕/ablation 재현 | `src/rag/`, `eval/` |
| **벡터 DB (pgvector)** | 스토어 교체형 설계 — 인메모리(기본) ↔ Postgres+pgvector 영속 스토어(`VECTOR_STORE=pgvector`): 부팅 시 텍스트 해시 기반 **증분 색인**(바뀐 청크만 재임베딩), 버전 필터(시행일·폐지본 구간)를 SQL pre-filter 로, HNSW 코사인 검색. SQL 필터 ↔ 파이썬 필터 동치·인메모리와 순위 동일성을 실 DB 테스트(CI 서비스 컨테이너)로 고정 | `src/rag/vectorstore.py`, `src/rag/versionfilter.py`, `tests/test_vectorstore.py` |
| **제약/바이오 도메인** | RA·PV 도메인 도구 — RA: 검색·마감·체크리스트 / PV: 트리아지→인과성→코딩(3계층)→ICSR 초안 + 라벨 22케이스 평가 | `src/ra/`, `src/pv/`, `eval/pv_eval.py` |
| **개인정보 보호** | PII 비식별화 2겹(에이전트 입구+도구 계층) — 한글 직결·조사 표기까지 룩어라운드 경계 | `src/pv/redactor.py` |
| **MCP / FastMCP** | Tools 7 + Resource + Prompt(3대 primitive), 독립 HTTP 서버(운영)/stdio(Claude Desktop) | `src/mcp_server/server.py` |
| **Agentic / Function Calling** | tool-use 루프, 도구 에러 자가복구, 오프라인 규칙 라우터 폴백 | `src/agent/agent.py` |
| **PydanticAI** | 동일 계약(검증 게이트·MCP 도구·grounded 판정)의 병렬 백엔드 — `AGENT_BACKEND=pydantic_ai` 로 직접 구현 루프와 교체·비교 가능, MCPToolset 이 같은 MCP_SERVER_URL(HTTP)에 직결 | `src/agent/pydantic_agent.py` |
| **LangChain / LangGraph** | 동일 계약의 세 번째 병렬 백엔드 — `AGENT_BACKEND=langgraph`, 명시적 StateGraph(agent⇄ToolNode 순환 + recursion_limit)로 tool-use 루프 구현, MCP 도구를 LangChain StructuredTool 로 무변환 래핑 | `src/agent/langgraph_agent.py` |
| **FastAPI · 프론트엔드** | `/chat`·`/health` + 단일 페이지 챗 UI(출처·트레이스·검증 배지) | `src/api/`, `web/` |
| **신뢰성(환각 억제)** | groundedness·abstention(AND 문턱)·출처/버전 추적·`as_of` 시점 조회 | `src/agent/`, `src/rag/retriever.py` |
| **답변 사후 검증** | 전 응답 통과 런타임 게이트 + 게이트 자체의 메타모픽 평가(pytest 가드로 CI 강제) + opt-in LLM judge 2차 계층(수치 없는 의미 왜곡 — 결정론 경계 밖) | `src/verify/`, `eval/verify_eval.py`, `eval/judge_eval.py` |
| **에이전트 평가 확장** | 계획 능력(도구 선택·인자, 결정론 라우터 1.0 핀)·입력 견고성(오타 섭동·구어·프롬프트 인젝션)·LLM 기반 평가셋 후보 생성(사람 리뷰 게이트 강제) | `eval/plan_eval.py`, `eval/robustness_eval.py`, `eval/generate_eval_items.py` |
| **배포 전 점검(Day-0)** | 설정·코퍼스·업무데이터·스모크+안전장치 자가 테스트 — 실패 시 기동 차단 | `src/preflight.py` |
| **사용자 인증·호출 제한** | Google 로그인(OIDC — 인증서버 없이 Google 을 IdP 로 사용, 사내 IdP 로 교체 가능)·허용 목록·사용자별 하루 질문 상한·서명 세션 쿠키, 공개 배포 가드(`REQUIRE_LOGIN`)를 preflight 가 강제 | `src/api/auth.py` |
| **운영 계기판** | 검증 경고율(`warn_rate_checked`)·감사 로그를 `/health` 노출 | `src/observability.py` |
| **모델 선택 실험** | 같은 정답셋·같은 검색 결과·같은 프롬프트로 모델 N개의 생성·도구 선택 단계를 비교(정확도·근거 충실도·출처·지연·비용·실패율, 결정론 지표와 심판 지표 분리) → 작업별 모델을 `config/models.json` 으로 설정화, 계기판에 현재 모델·비교 표 노출 | `eval/model_compare/`, `config/models.json` |
| **실데이터 인제스트** | PDF→코퍼스 변환 경로(헤딩 휴리스틱+frontmatter, 상용 파서 교체 자리) | `scripts/ingest_pdf.py` |
| **테스트/CI** | pytest 425케이스(불변식/fuzz 포함) + Actions(preflight+평가 6종 회귀) | `tests/`, 루트 `.github/workflows/ci.yml` |

## 5. 실행 방법

```bash
cd project
./run.sh          # venv + 의존성 + 배포 전 점검(preflight) + 서버 → http://127.0.0.1:8000
export ANTHROPIC_API_KEY=sk-ant-...   # (선택) LLM 모드 — 없어도 오프라인 grounded 답변으로 항상 동작

# (선택) 밀집임베딩 활성화 — 미설치 환경은 TF-IDF 로 자동 폴백해 항상 동작한다
.venv/bin/pip install -r requirements-ml.txt
.venv/bin/python scripts/download_embedding_model.py   # multilingual-e5-small 1회 다운로드

# (선택) pgvector 영속 벡터 스토어 — 재부팅 때 바뀐 청크만 재임베딩, 유사도 검색은 DB(HNSW)
#   로컬 DB 까지 한 번에: (레포 루트) docker compose -f docker-compose.yml -f docker-compose.pgvector.yml up --build
export VECTOR_STORE=pgvector PGVECTOR_URL=postgresql://rapv:rapv@127.0.0.1:5432/rapv
```

**벡터 스토어 선택 기준.** 기본 인메모리는 부팅마다 코퍼스 전체를 임베딩한다(72청크·e5-small 로 수 초).
`pgvector` 는 밀집 임베더 전제(TF-IDF 는 코퍼스 전체로 IDF 를 학습해 문서 하나가 바뀌면 모든 벡터가
바뀌므로 영속화 대상이 아니다 — preflight 가 조합 오류를 기동 전에 거부). 검색 결과는 `PGVECTOR_POOL=0`
(전수, 기본)에서 인메모리와 **순위·점수 동일**(테스트 고정)이고, 양수로 두면 HNSW 상위 n 후보만 받아
코퍼스 크기와 무관한 질의 비용을 얻는 대신 풀 밖 후보의 벡터 점수를 풀 최솟값으로 채우는 근사가 들어간다
(BM25·리랭킹은 한국어 토크나이즈 일관성 때문에 파이썬에 남긴다).

로그인 켜기(선택 · 공개 배포 시 필수): `GOOGLE_CLIENT_ID`·`GOOGLE_CLIENT_SECRET`·`SESSION_SECRET` 를 설정하면
챗 UI·`/chat` 에 Google 로그인과 사용자별 하루 질문 상한(`DAILY_CHAT_LIMIT`, 기본 30)이 적용된다.
설정 방법은 `.env.example` 의 '사용자 로그인' 절. 비워 두면 로그인 없이 로컬에서 그대로 동작한다.

품질 확인: `.venv/bin/python -m src.preflight`(배포 전 점검) · `-m pytest`(425케이스) ·
`-m eval.evaluate / sweep / faithfulness / pv_eval / verify_eval / plan_eval / robustness_eval`
(평가 7종, 95% CI 병기). 키 필요 수동 3종: `-m eval.judge_eval`(LLM judge 실측) ·
`-m eval.generate_eval_items`(평가셋 후보 생성 — 리뷰 게이트) ·
`-m eval.model_compare.run`(모델 성능 비교 → 작업별 모델 선택, 보고서 `eval/model_compare/REPORT.md`).
CI는 매 푸시마다 preflight+테스트+평가 6종을 실행한다.

## 6. 핵심 수치 (전부 스크립트 1회 실행으로 재현)

> ⚠️ **읽기 규율:** 검색 32문항은 오류 분석 루프에 쓴 **개발셋**이라 그 위의 1.000은 일반화가
> 아니라 개발셋 포화다. 일반화의 근사는 **홀드아웃**(튜닝 미사용 6문항, `eval/holdout_dataset.json`)
> 수치로 따로 본다. 최초 측정은 0.833이었는데, 유일 실패의 실체는 검색기 결함이 아니라 **복수
> 정답 문항의 라벨 모호성**이었다(REG-011 §3 도 동등 정답) — v9 에서 `accept_doc_ids` 로 정답
> 집합만 교정했고 **검색기 튜닝은 하지 않았다**(홀드아웃 규율 유지). 교정 후 1.000 도 n=6·CI
> 하한 0.610 이라 **일반화 보장이 아니다** — '이 표본에서 실패 관측 0'으로만 읽는다.

| 영역 | 지표 | 값 |
|---|---|---|
| 검색(개발셋 32) | Hit@1 (벡터만 → 전체 파이프라인) | 0.938 → **1.000** [0.893, 1.000] |
| 검색(개발셋 32) | HardNegHit@1 / ContextRecall (전체 파이프라인) | **1.000** / **0.969** — 밀집임베딩(e5) 전환으로 벡터 단독 HardNeg 0.786→**1.000**, ContextRecall 0.781→**0.969** |
| **검색(홀드아웃 6)** | **Hit@1 — 일반화 근사** | **1.000** [0.610, 1.000] (v9 라벨 교정 — 교정 전 0.833, 유일 실패는 복수 정답 라벨 모호성) |
| 신뢰성(32+8) | Groundedness / AbstentionAcc / OverAbstain | **1.000** / **1.000** (n=8) / **0.000** |
| PV(라벨 22) | 중대성·기한·인과성·보고요건 정확도 | 모두 **1.000** (규칙 회귀는 CI가 잡음) |
| PV(라벨 22) | 코딩 P / R (확정) · R (확정∪후보) | **1.000** / 0.792 · **0.958** — 갭 = 사전 롱테일(확장 지점 크기) |
| 검증 게이트 | 변조 탐지 9축 / 오탐 감시 4축 | 모두 **1.000** (n=4~40, 표본 하한은 pytest 가드가 CI 강제) |
| 운영 실측 | E2EPassRate (오프라인 실응답 전수) | **1.000** [0.912, 1.000] (n=40) |
| 계획(18문항) | ToolRecall / ToolPrecision / ParamAccuracy | 모두 **1.000** (결정론 라우터 — pytest 핀, 깨지면 라우팅 회귀) |
| 견고성(섭동 128) | TypoHit@1 — 실측 (Clean 1.000 대비) | **0.969** [0.922, 0.988] (delta +0.031 — 열화를 숨기지 않음, 핀은 하한 0.90) |
| 견고성(구어 6) | ColloquialOk — 실측 | **0.500** — 갭 = 동의어 사전·회피 문턱의 개선 여지(라벨 하향으로 숨기지 않음) |
| 견고성(적대 9) | 주입 무순응 / 도구 판정 불변 | 모두 **1.000** (무관용 핀 — 추출형 구조 + 게이트 상시 부착의 회귀 가드) |
| **모델 비교(생성 5층 89 + 도구 18)** | Sonnet 5.5 vs Opus 5.5 vs Haiku 4.5 — 핵심어 일치(단순) / 근거 제거 회피 / 도구 선택(관대) | Sonnet **32/32** / **18/18** / 16/18 · Opus 32/32 / 18/18 / 17/18 · Haiku 31/32 / 16/18 / 13/18 — 답변 생성은 Sonnet(Opus 대비 비용 45%·p50 61%), 판정·요약은 Haiku. 근거 `eval/model_compare/README.md` |

- 수치의 상세(모드 4종 비교표·오류 분석 루프 0.867→1.000·검증기 축별 표)는 각 평가 스크립트
  출력에 있다. 소표본 1.000에는 항상 Wilson 95% CI를 병기한다 — '완벽'이 아니라
  '이 표본에서 실패 관측 0'이며, 구간이 겹치면 개선이 아니라 구분 불가로 읽는다.
- **답변 사후 검증 게이트**(6줄 요약): 평가는 표본을 지키지만 LLM 답변은 매번 새로 생성되므로,
  **모든 응답**이 수치·단위(근무일≠일)·날짜(한국어·부분 표기 정규화)·방향 한정어(이내↔이후)·
  날짜 역할(인지일↔마감일)을 '검색 근거 ∪ 결정론적 도구 출력 − 입력 에코'와 대조하는 게이트를
  통과한다. 실패는 차단이 아니라 경고 부착(사람의 최종 확정을 빠르게), 케이스 서술 유래 지지는
  `from_case` 라벨로 구분한다. 게이트 자체도 메타모픽 변조(9축)로 측정하고 preflight가 기동 전에
  전 축 자가 테스트를 강제한다 — 구현은 `src/verify/`.
  결정론이 못 보는 **수치 없는 의미 왜곡**(부정문 뒤집기·주어 바꿔치기)은 opt-in **LLM judge
  2차 계층**(`LLM_JUDGE=1`, 기본 off·비차단·결정론 축 정본 밖)이 담당하고, 그 판정 능력은
  `eval/judge_eval.py`(키 필요 실측)와 preflight canary 로, 배선 계약은 pytest 스텁으로 잰다.
- **PV 워크플로**: 트리아지(중대성=기준 대조는 결정론, 서술→기준 매핑은 과탐 보수+사람 확정) →
  인과성(WHO-UMC 6범주의 5범주 근사, '제안'+되물을 질문) → 코딩(확정/후보/미코딩 3계층, 자동 확정
  무관용) → ICSR 초안(ICH E2D 4요소 검증). 기한 계산은 역일(calendar day) 기준이며 LLM이 아니라
  규칙 도구가 한다 — 구현은 `src/pv/`.

## 7. 구조

핵심 코드는 `src/`(rag · ra · pv · agent · mcp_server · api · verify), 데이터는 `data/`,
평가 스크립트는 `eval/`, 테스트는 `tests/`, 챗 UI는 `web/`,
호스팅 정적 페이지는 `description/`(`dictionary.html` · `blank.html`)에 있다.
CI 워크플로는 저장소 루트 `.github/workflows/ci.yml`
(GitHub Actions는 루트만 인식, `working-directory: project`로 실행).

> ℹ️ 규제문서·수치(처리기한 등)는 **데모용 가상 샘플**로(각 문서 frontmatter에 disclaimer 명시),
> 실제 법령과 다를 수 있다. 이 프로젝트의 목적은 규제 자문이 아니라 **아키텍처·엔지니어링 역량 증명**이다.
