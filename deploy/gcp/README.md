# Google Cloud Run 배포

RAPV-Assistant 를 Cloud Run 서비스 2개로 배포한다. 같은 컨테이너 이미지를 쓰고 실행 명령만 다르다.

```
사용자 ──HTTPS──▶ pharma-api (FastAPI · 챗 UI · Google 로그인 · 일일 질문 상한)
                      │  Bearer 토큰(Secret Manager)
                      ▼
                  pharma-mcp (FastMCP 도구 서버 · RAG 인덱스)
                      │  (4차 선택) PGVECTOR_URL · sslmode=require
                      ▼
                  Supabase Postgres (pgvector — 청크·임베딩 영속, 증분 색인)
```

| 리소스 | 용도 |
|---|---|
| Artifact Registry `pharma` | 컨테이너 이미지 (최신 3개만 보관 — 무료 0.5GB 한도. 4차 ML 이미지는 한도를 넘는다, 아래 비용 참고) |
| Cloud Build | `project/Dockerfile` 빌드 (`cloudbuild.yaml` — `WITH_ML` 빌드 인자 전달) |
| Secret Manager | `pa-mcp-auth-token` · `pa-session-secret` · `pa-google-client-secret` · `pa-anthropic-api-key` · `pa-pgvector-url`(4차) |
| Cloud Run `pharma-mcp` / `pharma-api` | us-central1, 인스턴스 0~2대, 1 vCPU · 1GiB (4차: `pharma-mcp` 만 2GiB + CPU 부스트) |
| Supabase (4차 선택) | 무료 티어 Postgres + `vector` 확장 — GCP 밖의 외부 서비스, 비용 0 |

요청이 없으면 인스턴스가 0대로 내려가 비용이 들지 않는다. 테스터 수십 명 규모는 Cloud Run 무료 등급 안에서 돈다.

## 실행 (Cloud Shell)

Google Cloud 콘솔 오른쪽 위 `>_` 아이콘으로 Cloud Shell 을 열고:

```bash
git clone https://github.com/chaeyoo/pharma-assistant.git && cd pharma-assistant
bash deploy/gcp/deploy.sh <PROJECT_ID>
```

스크립트는 여러 번 실행해도 안전하다. 기능은 단계별로 켠다.

### 1차 — 오프라인 데모

```bash
bash deploy/gcp/deploy.sh <PROJECT_ID>
```

API 키 없이 RAG·PV·검증 전 기능이 도는 공개 데모. 끝에 출력되는 `…/auth/callback` 주소를 2차에서 쓴다.

### 2차 — Google 로그인

1. 콘솔 → **API 및 서비스 → OAuth 동의 화면**: 사용자 유형 *외부*, 앱 이름·지원 이메일 입력. 프로덕션 게시에는 **브랜딩** 페이지의 **애플리케이션 홈페이지**(1차 끝에 출력된 서비스 URL)와 **개인정보처리방침 링크**(`<서비스 URL>/privacy` — 앱이 로그인 없이 서빙)가 필수다. 두 URL 의 도메인은 **승인된 도메인**에 있어야 하는데 리디렉션 URI 의 run.app 주소가 자동 등록되므로 외부 호스팅 없이 같은 서비스 주소를 쓰면 된다. 로고는 올리지 않는다(올리면 브랜드 인증 심사 대상). 테스터 몇 명에게만 열 때는 게시 상태를 *테스트*로 두고 **테스트 사용자**에 Gmail 을 추가한다. **포트폴리오처럼 불특정 검토자가 로그인해야 하면 "앱 게시"로 *프로덕션* 전환** — 이 앱은 `openid email profile` 만 요청하므로(민감 범위 없음) Google 의 앱 검증 없이 바로 게시되고 '확인되지 않은 앱' 경고도 뜨지 않는다. 테스트 상태로 두면 미등록 계정은 "승인된 테스터만 접근 가능" 화면에서 막힌다. 공개 후의 비용 보호는 사용자별 일일 상한(`DAILY_CHAT_LIMIT`)과 Anthropic 콘솔의 월 한도.
2. **API 및 서비스 → 사용자 인증 정보 → 사용자 인증 정보 만들기 → OAuth 클라이언트 ID**: 유형 *웹 애플리케이션*, **승인된 리디렉션 URI** 에 1차 끝에 출력된 주소를 그대로 넣는다.
3. 발급된 클라이언트 ID·보안 비밀로 다시 실행:

```bash
GOOGLE_CLIENT_ID=xxxx.apps.googleusercontent.com \
GOOGLE_CLIENT_SECRET=GOCSPX-xxxx \
ALLOWED_EMAILS=tester1@gmail.com,tester2@gmail.com \
bash deploy/gcp/deploy.sh <PROJECT_ID>
```

이후 `REQUIRE_LOGIN=1` 이 걸려, 로그인 설정이 빠진 배포는 preflight 가 기동을 거부한다.

허용 목록은 한 번 넣으면 재실행 때 유지된다. OAuth 를 프로덕션으로 게시한 뒤 테스터 목록을 풀려면
**빈 값을 명시**해서 재실행한다(안 주면 기존 값 유지). 목록이 비면 Google 계정이면 누구나 들어오므로,
사내 계정만 열 때는 `ALLOWED_DOMAINS` 를 대신 둔다. 목록에 없는 계정은 로그인 화면에
"이 계정은 아직 이용 권한이 없습니다" 로 막힌다(Google 쪽이 아니라 앱 쪽 거부).

```bash
ALLOWED_EMAILS= bash deploy/gcp/deploy.sh <PROJECT_ID>                      # 누구나
ALLOWED_EMAILS= ALLOWED_DOMAINS=example.com bash deploy/gcp/deploy.sh <PROJECT_ID>   # 사내 도메인만
```

### 3차 — LLM 모드

```bash
ANTHROPIC_API_KEY=sk-ant-xxxx bash deploy/gcp/deploy.sh <PROJECT_ID>
```

로그인이 켜져 있지 않으면 스크립트가 거부한다(공개 URL 의 API 비용 보호). 사용자별 하루 상한은 `DAILY_CHAT_LIMIT`(기본 30)으로 바꾼다. Anthropic 콘솔에서도 월 사용 한도를 걸어 두는 것을 권장한다.

### 4차 — pgvector 영속 벡터 스토어 (Supabase)

1~3차는 경량 이미지(TF-IDF 폴백·인메모리 스토어)다. 4차는 `pharma-mcp` 를 밀집임베딩(e5) 이미지로 바꾸고, 청크·임베딩을 외부 Postgres(pgvector)에 영속화한다 — 콜드스타트마다 코퍼스를 다시 임베딩하지 않고, 바뀐 청크만 증분 임베딩한다. `pharma-api` 는 바뀌지 않는다.

**왜 서버 밖의 Postgres 인가.** Cloud Run 컨테이너의 디스크·메모리는 **임시**다 — 인스턴스가 0대로 내려가면 그 안에 쓴 것은 전부 사라지고, 다음 요청은 깨끗한 컨테이너로 뜬다. 그래서 서버 컨테이너 안에 Postgres 를 같이 넣어도 콜드스타트마다 빈 DB 가 되어 72청크를 다시 임베딩해야 하고, 결과적으로 인메모리와 같은 비용에 Postgres 기동 시간과 이미지 크기만 더 얹는 셈이 된다. 1~3차가 인메모리로 가는 이유가 이것이다 — 영속성이 없다면 더 가벼운 쪽이 낫다. pgvector 의 존재 이유가 "서버가 꺼져도 벡터가 남는 것"이므로 DB 는 서버 수명과 분리된 곳에 있어야 하고, 로컬에서는 docker-compose 의 볼륨(`pgvector-data`)이, Cloud Run 에서는 외부 Postgres 가 그 역할을 한다. (Cloud Storage 버킷 마운트 같은 네트워크 파일시스템 위에 Postgres 를 올리는 것은 잠금 문제로 권장되지 않고, 파일 기반 저장으로 가면 'DB 측 유사도 검색'이라는 두 번째 이점을 잃는다.)

**DB 선택지 비교 — 왜 Supabase 인가.** 코드는 `PGVECTOR_URL` 하나만 보므로 `vector` 확장이 켜지는 Postgres 라면 어디든 붙는다(Render Postgres·Neon·Cloud SQL 모두 동작). 선택 기준은 포트폴리오 단계의 **비용 0** 이었다 — GCP 안에는 Postgres+pgvector 를 무료로 쓰는 길이 없다. 금액은 2026-10 기준 대략값이므로 결정 전에 각 요금표를 다시 확인한다.

| 선택지 | pgvector | 최소 비용(대략) | 비고 |
|---|---|---|---|
| **Supabase** (채택) | O | 무료 티어(500MB) | 7일 미사용 시 일시정지 → 대시보드 Restore. GCP 밖의 외부 서비스 |
| Neon | O | 무료 티어(0.5GB) | 5분 미사용 시 자동 휴면·자동 복구(첫 접속 지연). Supabase 보다 휴면이 잦다 |
| Cloud SQL for PostgreSQL | O (확장 활성화) | 월 1만원 안팎(공유코어 최소 인스턴스 + 디스크), 무료 티어 없음 | Cloud Run 과 같은 프로젝트·사설 연결·IAM 한 곳. "전부 GCP 안" 서사가 필요하면 이쪽 — URL 교체만으로 이동 가능 |
| AlloyDB | O (ScaNN 인덱스) | 월 수십만원(최소 사양이 큼) | 대규모 벡터용, 이 규모엔 과함 |
| Vertex AI Vector Search · Firestore 벡터 검색 | 전용 API | Vertex 는 인덱스 상시 과금, Firestore 는 무료 티어 | Postgres 가 아니라 `PgVectorStore` 를 못 쓰고 스토어를 새로 구현해야 함 |

운영 전환 시 비용을 감수하고 Cloud SQL 로 옮기는 것이 자연스러운 다음 단계이며, 그때도 바뀌는 것은 시크릿의 URL 뿐이다.

1. [supabase.com](https://supabase.com) → **New project**(Free 플랜, 리전은 Cloud Run 과 가까운 *East US*), 데이터베이스 비밀번호를 기록한다.
2. (선택) **Database → Extensions** 에서 `vector` 를 켠다(`extensions` 스키마에 설치된다). 건너뛰어도 된다 — 코드의 `CREATE EXTENSION IF NOT EXISTS vector` 가 첫 부팅 때 `public` 스키마에 켜며(2026-10-22 로컬→Supabase 실측, vector 0.8.2), 이미 있으면 아무것도 안 한다. 기본 search_path(`"$user", public, extensions`)가 두 스키마를 모두 포함하므로 어느 쪽이든 비정규 `vector` 타입과 `<=>` 연산자가 그대로 풀린다.
3. 상단 **Connect** → **Session pooler** 의 URI 를 복사한다(포트 **5432**, 사용자 `postgres.<프로젝트 ref>`, 호스트 `aws-N-<리전>.pooler.supabase.com`). 직결(`db.<ref>.supabase.co`)은 IPv6 전용이라 Cloud Run(IPv4)에서 안 붙고, Transaction pooler(6543)는 색인이 쓰는 advisory lock·세션 설정을 지원하지 않는다. 끝에 `?sslmode=require` 를 붙이고, 비밀번호의 특수문자는 퍼센트 인코딩한다.

```bash
WITH_ML=1 \
PGVECTOR_URL='postgresql://postgres.<ref>:<비밀번호>@aws-0-us-east-1.pooler.supabase.com:5432/postgres?sslmode=require' \
bash deploy/gcp/deploy.sh <PROJECT_ID>
```

URL 은 `pa-pgvector-url` 시크릿에 저장되고, 이후 재실행은 `WITH_ML` 없이도 배포된 상태(밀집임베딩·pgvector)를 유지한다. `WITH_ML=1` 만 주면 밀집임베딩 + 인메모리 스토어(DB 없음)로 돈다. 되돌리기: `gcloud secrets delete pa-pgvector-url` 뒤 `WITH_ML=0 bash deploy/gcp/deploy.sh <PROJECT_ID>` (pgvector 시크릿이 남아 있는데 `WITH_ML=0` 이면 스크립트가 거부한다 — 밀집 임베더 없이는 pgvector 를 못 쓴다는 preflight 규칙을 배포 전에 적용).

**비용.** Supabase 무료 티어는 0원이지만 **7일 미사용 시 프로젝트가 일시정지**된다 — 그러면 `pharma-mcp` 가 콜드스타트 때 preflight 의 DB 접속 검사에서 기동을 거부해 요청이 실패한다. Supabase 대시보드에서 **Restore** 한 뒤 다시 요청하면 된다(데이터는 보존된다). ML 이미지는 CPU torch + e5 모델이 들어가 Artifact Registry 무료 0.5GB 를 혼자서 넘는다(로컬 빌드 비압축 2.9GB, 레지스트리 압축 크기는 콘솔에서 확인) — 최신 3개 보관 정책 기준 월 수십 센트~1달러(GB당 $0.10). `pharma-api` 도 같은 이미지를 쓰므로 pull 만 커질 뿐 자원 설정은 그대로다. `pharma-mcp` 2GiB 는 Cloud Run 무료 등급(월 36만 GiB-초) 안이다. Cloud Build 는 ML 이미지 빌드가 8~12분 걸린다(무료 할당 내).

**콜드스타트 실측.** Cloud Run 은 요청이 없으면 0대라 요청마다 부팅이 끼어들 수 있다. 같은 ML 이미지로 로컬 Docker(Apple Silicon, Docker Desktop)에서 `RagPipeline.build()` 의 분해 로그를 각 3회 잰 값이다(중앙값). Cloud Run 행은 2026-10-22 실배포(리비전 00007, us-central1 · 1 vCPU + CPU 부스트 · 2GiB) 의 `pharma-mcp` 로그 1회 값이다.

| 시나리오 | 모델로드 | 색인(임베딩) | build() 합계 | 컨테이너 시작→준비 |
|---|---|---|---|---|
| 인메모리 (부팅마다 72청크 임베딩) | 8.8s | 2.1s | 10.9s | 24.4s |
| pgvector 첫 부팅 (빈 테이블 → 72 임베딩) | 9.2s | 2.3s | 11.5s | — |
| pgvector 재부팅 (재사용 72 · 신규 0) | 9.9s | **0.03s** | 9.9s | **20.2s** |
| Supabase(us-east-1) 재부팅 — 로컬(한국)에서 접속 | 6.1s | 3.9s* | 10.0s | — |
| **Cloud Run(us-central1) → Supabase(us-east-1) 재부팅 — 실배포** | **31.6s** | **0.64s** | **32.2s** | — (preflight 별도 ~0.7s 색인) |
| Cloud Run — 1단계(HF Hub 왕복 제거) 후 | ~20s | — | — | — (preflight 프로세스가 한 번 더 로드) |
| **Cloud Run — 2단계(인프로세스 preflight) 후, 2026-10-23** | **16.1s** | — | — | 로드 1회 — `RAG 인덱스 구축` 로그 1줄 |

\* Supabase 행은 임베딩이 아니라 한국↔미국 동부의 네트워크 왕복(해시 대조·메타 갱신 등 질의당 수백 ms)이 쌓인 값이다 — 검색도 질의당 ~620ms 로 같은 이유. Cloud Run(us-central1)에서는 같은 대륙이라 왕복이 수십 ms 로 줄며, 실측은 배포 뒤 로그로 갱신한다. 수치(Hit@1 등)는 인메모리·로컬 pgvector 와 동일했다.

Cloud Run 실측 읽는 법: 같은 대륙이라 증분 색인은 0.64s(해시 대조 + 메타 갱신의 DB 왕복)로 한국 접속의 3.9s 에서 내려왔고 "재사용 72 · 신규 0" 으로 영속화가 동작했다. 대신 **모델 로드가 31.6s** 로 로컬(Apple Silicon ~9s)의 3배 이상이다 — Cloud Run 의 1 vCPU 에서 torch 임포트 + e5 로드가 그만큼 느리고, preflight 프로세스가 한 번 더 로드하므로(preflight 의 build() 는 canary 가 먼저 로드해 0ms 로 찍힘) 컨테이너 시작→준비는 2분 안팎이다. Cloud Run 기본 기동 제한(240s) 안이지만 여유가 크지 않다. 결론은 로컬 실측과 같다 — 다음 병목은 pgvector 가 아니라 모델 로드다.

**모델 로드 개선 1단계 — 허브 왕복 제거(2026-10-22 실배포, 모델로드 31.6s → ~20s).** 로그에 preflight 시작 48초 뒤 "unauthenticated requests to the HF Hub" 경고가 찍혔다 — 모델을 이미지에 구워 넣었는데도 `EMBEDDER_KIND=dense` 가 `local_files_only=False` 라 기동마다 Hugging Face Hub 에 메타데이터를 조회한 것이다. ML 이미지에 `HF_HUB_OFFLINE=1` 을 고정하고 dense 모드도 받아둔 모델을 먼저 쓰도록 바꿨다. 로컬 실측: 허브 확인 포함 로드 6.0s → 1.3s, 임포트 포함 전체 10.7s → 4.9s. Cloud Run 수치는 재배포 뒤 위 표에 추가한다. 다음 후보는 (3) `--cpu 2`, (4) ONNX 백엔드 또는 외부 임베딩 API 로 torch 제거 — 둘 다 비용(과금·전환)이 있어 보류.

**모델 로드 개선 2단계 — 이중 로드 제거(2026-10-23 실배포, 모델로드 16.1s × 1회).** 종전 기동 명령 `preflight --role mcp && server` 는 preflight 프로세스와 서버 프로세스가 각각 모델을 로드했다(로그에 `RAG 인덱스 구축:` 이 2줄). 이제 `server.py --preflight` 가 같은 점검을 서버 프로세스 안에서 돌리고, 스모크가 만든 인덱스(`server._pipeline`)와 lru_cache 된 모델을 서버가 그대로 물려받는다 — 로드 1회, 로그 1줄. 판정·출력은 CLI 와 같다(실패하면 서버를 띄우지 않음). 비용 0.

**체감 개선 — 예열과 진행 표시(코드 반영).** 콜드스타트를 없애지 못하는 만큼, 사용자가 그 시간을 '기다리는 이유 없이' 보내지 않게 한다. 챗 페이지는 열리는 순간 `GET /warmup`(MCP /health 를 120s 제한으로 대기)으로 검색 서버를 깨우고 "검색 서버 준비 중 → 준비됨(N s)" 배지를 보여 준다 — 사용자가 예시 질문을 읽고 타이핑하는 동안 기동이 진행된다. 질문은 `POST /chat/stream`(SSE)으로 보내 "검색 서버 연결(콜드스타트면 여기서 수십 초) → 규정 검색 → 답변 생성 → 수치·날짜 검증" 을 실시간으로 그리고, 답변 뒤에는 단계별 ms 타임라인이 남는다. 과금 영향 없음(첫 질문이 유발했을 기동을 앞당길 뿐).

읽는 법: 증분 색인은 임베딩 몫(~2.2초)을 통째로 없애고, 컨테이너 시작→준비는 약 4초 준다(preflight 와 서버 프로세스가 각각 인덱스를 구축해 인메모리는 임베딩이 두 번 든다). 남는 비용은 torch 임포트 + e5 로드(프로세스당 ~9초, 두 프로세스)라 다음 단계는 pgvector 가 아니라 모델 로드 쪽이다 — preflight 의 RAG 스모크를 mcp 롤에서 생략하거나, 임베딩 질의만 외부 API(`EMBEDDER_KIND=voyage`)로 넘겨 torch 를 이미지에서 빼는 선택지가 있다. 코퍼스가 커질수록(수천 청크) 임베딩 몫이 선형으로 늘어 pgvector 의 효과가 커진다.


**API 컨테이너의 콜드스타트 — 첫 페이지까지 2분(2026-10-24 실측)의 분해와 제거.** 예열 배지는 페이지가 뜬 뒤에야 동작하므로, 페이지 자체가 늦게 뜨면 소용이 없다. 첫 방문 시 일어나던 일: ① `pharma-api` 인스턴스 기동(ML 이미지 공유) → ② api 의 preflight 가 '임베더 해석'을 출력하려고 **e5 모델을 로드**(검색을 안 하는 컨테이너에서 수십 초, 1GiB) → ③ preflight 가 MCP 준비를 **최대 60초 직렬 대기**(그제야 MCP 가 깨어나 16s 모델 로드) → ④ uvicorn 기동. 비용 0 으로 두 가지를 끊었다.
- **A. api 롤은 모델을 로드하지 않는다** — `preflight.run_and_report` 가 api 롤에서는 임베더를 해석하지 않는다(② 삭제).
- **B. MCP 대기를 짧게·soft 로** — api 서비스에 `MCP_WAIT_TIMEOUT_S=5 MCP_WAIT_SOFT=1`. 5초 안에 연결이 안 되면 경고만 남기고 기동해 페이지를 먼저 내고, UI 의 `/warmup` 이 "검색 서버 준비 중 → 준비됨"을 보여 준다(③ 삭제). 연결은 되는데 기대 도구가 없는 버전 스큐는 soft 여도 기동을 막는다(롤링 배포 보호 유지). 로컬·compose 는 기본값(60s, 차단) 그대로.
- **C. 깨어 있게 유지(선택, 무료 한도 내)** — Cloud Scheduler 가 10분마다 `GET /warmup` 을 호출하면 두 인스턴스가 유휴 종료(약 15분) 전에 갱신되어 방문자가 콜드스타트를 거의 만나지 않는다. Scheduler 무료 3개 작업, Cloud Run 은 요청 처리 시간만 과금(호출당 수 ms)이라 사실상 0원. 유휴 인스턴스의 메모리는 요청 기반 과금에서 청구되지 않는다.
  ```bash
  gcloud services enable cloudscheduler.googleapis.com --project <PROJECT_ID>
  gcloud scheduler jobs create http pharma-keepalive --project <PROJECT_ID> --location us-central1 \
    --schedule "*/10 * * * *" --uri "https://<pharma-api URL>/warmup" --http-method GET --attempt-deadline 180s
  ```
  끄려면 `gcloud scheduler jobs pause pharma-keepalive --location us-central1`.

## 확인

- 챗 UI: `https://<pharma-api URL>/`
- 상태: `…/health` · 운영 계기판: `…/dashboard`
- 로그: 콘솔 → Cloud Run → 서비스 → **로그** 탭. `pharma-mcp` 부팅 로그의 `RAG 인덱스 구축:` 줄이 콜드스타트 분해(모델로드·로드/청킹·색인 ms)와 pgvector 증분 색인 결과(`재사용 N · 신규/변경 임베딩 M`)를 보여 준다.

## 설계 메모

- **MCP 서버가 공개 URL 인 이유**: 외부 MCP 클라이언트(Claude Desktop 등)도 같은 도구 서버에 붙을 수 있게 한다. 대신 Bearer 토큰이 필수이고 `MCP_REQUIRE_AUTH=1` 이라 토큰 없이 뜨지 않는다. 사내 배포라면 ingress 를 내부 전용으로 바꾸고 Cloud Run IAM(서비스 계정 ID 토큰)으로 서비스 간 인증을 하는 것이 다음 단계다.
- **로그인**: Google 을 IdP 로 쓰는 OIDC. 사내 SSO 로 옮기려면 IdP 메타데이터 주소와 클라이언트 자격만 교체한다(`project/src/api/auth.py`).
- **호출 상한 저장소**: 현재는 인스턴스 메모리라 인스턴스가 재시작·증설되면 카운트가 갈라진다. 운영 규모가 커지면 같은 인터페이스로 Firestore 구현을 끼운다.
