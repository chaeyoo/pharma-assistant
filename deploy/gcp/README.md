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

1. 콘솔 → **API 및 서비스 → OAuth 동의 화면**: 사용자 유형 *외부*, 앱 이름·지원 이메일 입력, 게시 상태는 *테스트*로 두고 **테스트 사용자**에 테스터 Gmail 을 추가한다.
2. **API 및 서비스 → 사용자 인증 정보 → 사용자 인증 정보 만들기 → OAuth 클라이언트 ID**: 유형 *웹 애플리케이션*, **승인된 리디렉션 URI** 에 1차 끝에 출력된 주소를 그대로 넣는다.
3. 발급된 클라이언트 ID·보안 비밀로 다시 실행:

```bash
GOOGLE_CLIENT_ID=xxxx.apps.googleusercontent.com \
GOOGLE_CLIENT_SECRET=GOCSPX-xxxx \
ALLOWED_EMAILS=tester1@gmail.com,tester2@gmail.com \
bash deploy/gcp/deploy.sh <PROJECT_ID>
```

이후 `REQUIRE_LOGIN=1` 이 걸려, 로그인 설정이 빠진 배포는 preflight 가 기동을 거부한다.

### 3차 — LLM 모드

```bash
ANTHROPIC_API_KEY=sk-ant-xxxx bash deploy/gcp/deploy.sh <PROJECT_ID>
```

로그인이 켜져 있지 않으면 스크립트가 거부한다(공개 URL 의 API 비용 보호). 사용자별 하루 상한은 `DAILY_CHAT_LIMIT`(기본 30)으로 바꾼다. Anthropic 콘솔에서도 월 사용 한도를 걸어 두는 것을 권장한다.

### 4차 — pgvector 영속 벡터 스토어 (Supabase)

1~3차는 경량 이미지(TF-IDF 폴백·인메모리 스토어)다. 4차는 `pharma-mcp` 를 밀집임베딩(e5) 이미지로 바꾸고, 청크·임베딩을 외부 Postgres(pgvector)에 영속화한다 — 콜드스타트마다 코퍼스를 다시 임베딩하지 않고, 바뀐 청크만 증분 임베딩한다. `pharma-api` 는 바뀌지 않는다.

1. [supabase.com](https://supabase.com) → **New project**(Free 플랜, 리전은 Cloud Run 과 가까운 *East US*), 데이터베이스 비밀번호를 기록한다.
2. **Database → Extensions** 에서 `vector` 를 켠다(`extensions` 스키마에 설치된다. 코드의 `CREATE EXTENSION IF NOT EXISTS vector` 는 이미 있으면 아무것도 안 한다).
3. 상단 **Connect** → **Session pooler** 의 URI 를 복사한다(포트 **5432**, 사용자 `postgres.<프로젝트 ref>`, 호스트 `aws-N-<리전>.pooler.supabase.com`). 직결(`db.<ref>.supabase.co`)은 IPv6 전용이라 Cloud Run(IPv4)에서 안 붙고, Transaction pooler(6543)는 색인이 쓰는 advisory lock·세션 설정을 지원하지 않는다. 끝에 `?sslmode=require` 를 붙이고, 비밀번호의 특수문자는 퍼센트 인코딩한다.

```bash
WITH_ML=1 \
PGVECTOR_URL='postgresql://postgres.<ref>:<비밀번호>@aws-0-us-east-1.pooler.supabase.com:5432/postgres?sslmode=require' \
bash deploy/gcp/deploy.sh <PROJECT_ID>
```

URL 은 `pa-pgvector-url` 시크릿에 저장되고, 이후 재실행은 `WITH_ML` 없이도 배포된 상태(밀집임베딩·pgvector)를 유지한다. `WITH_ML=1` 만 주면 밀집임베딩 + 인메모리 스토어(DB 없음)로 돈다. 되돌리기: `gcloud secrets delete pa-pgvector-url` 뒤 `WITH_ML=0 bash deploy/gcp/deploy.sh <PROJECT_ID>` (pgvector 시크릿이 남아 있는데 `WITH_ML=0` 이면 스크립트가 거부한다 — 밀집 임베더 없이는 pgvector 를 못 쓴다는 preflight 규칙을 배포 전에 적용).

**비용.** Supabase 무료 티어는 0원이지만 **7일 미사용 시 프로젝트가 일시정지**된다 — 그러면 `pharma-mcp` 가 콜드스타트 때 preflight 의 DB 접속 검사에서 기동을 거부해 요청이 실패한다. Supabase 대시보드에서 **Restore** 한 뒤 다시 요청하면 된다(데이터는 보존된다). ML 이미지는 CPU torch + e5 모델이 들어가 Artifact Registry 무료 0.5GB 를 혼자서 넘는다(로컬 빌드 비압축 2.9GB, 레지스트리 압축 크기는 콘솔에서 확인) — 최신 3개 보관 정책 기준 월 수십 센트~1달러(GB당 $0.10). `pharma-api` 도 같은 이미지를 쓰므로 pull 만 커질 뿐 자원 설정은 그대로다. `pharma-mcp` 2GiB 는 Cloud Run 무료 등급(월 36만 GiB-초) 안이다. Cloud Build 는 ML 이미지 빌드가 8~12분 걸린다(무료 할당 내).

**콜드스타트 실측.** Cloud Run 은 요청이 없으면 0대라 요청마다 부팅이 끼어들 수 있다. 같은 ML 이미지로 로컬 Docker(Apple Silicon, Docker Desktop)에서 `RagPipeline.build()` 의 분해 로그를 각 3회 잰 값이다(중앙값). Cloud Run 수치는 배포 뒤 `pharma-mcp` 로그의 같은 줄로 읽는다.

| 시나리오 | 모델로드 | 색인(임베딩) | build() 합계 | 컨테이너 시작→준비 |
|---|---|---|---|---|
| 인메모리 (부팅마다 72청크 임베딩) | 8.8s | 2.1s | 10.9s | 24.4s |
| pgvector 첫 부팅 (빈 테이블 → 72 임베딩) | 9.2s | 2.3s | 11.5s | — |
| pgvector 재부팅 (재사용 72 · 신규 0) | 9.9s | **0.03s** | 9.9s | **20.2s** |

읽는 법: 증분 색인은 임베딩 몫(~2.2초)을 통째로 없애고, 컨테이너 시작→준비는 약 4초 준다(preflight 와 서버 프로세스가 각각 인덱스를 구축해 인메모리는 임베딩이 두 번 든다). 남는 비용은 torch 임포트 + e5 로드(프로세스당 ~9초, 두 프로세스)라 다음 단계는 pgvector 가 아니라 모델 로드 쪽이다 — preflight 의 RAG 스모크를 mcp 롤에서 생략하거나, 임베딩 질의만 외부 API(`EMBEDDER_KIND=voyage`)로 넘겨 torch 를 이미지에서 빼는 선택지가 있다. 코퍼스가 커질수록(수천 청크) 임베딩 몫이 선형으로 늘어 pgvector 의 효과가 커진다.


## 확인

- 챗 UI: `https://<pharma-api URL>/`
- 상태: `…/health` · 운영 계기판: `…/dashboard`
- 로그: 콘솔 → Cloud Run → 서비스 → **로그** 탭. `pharma-mcp` 부팅 로그의 `RAG 인덱스 구축:` 줄이 콜드스타트 분해(모델로드·로드/청킹·색인 ms)와 pgvector 증분 색인 결과(`재사용 N · 신규/변경 임베딩 M`)를 보여 준다.

## 설계 메모

- **MCP 서버가 공개 URL 인 이유**: 외부 MCP 클라이언트(Claude Desktop 등)도 같은 도구 서버에 붙을 수 있게 한다. 대신 Bearer 토큰이 필수이고 `MCP_REQUIRE_AUTH=1` 이라 토큰 없이 뜨지 않는다. 사내 배포라면 ingress 를 내부 전용으로 바꾸고 Cloud Run IAM(서비스 계정 ID 토큰)으로 서비스 간 인증을 하는 것이 다음 단계다.
- **로그인**: Google 을 IdP 로 쓰는 OIDC. 사내 SSO 로 옮기려면 IdP 메타데이터 주소와 클라이언트 자격만 교체한다(`project/src/api/auth.py`).
- **호출 상한 저장소**: 현재는 인스턴스 메모리라 인스턴스가 재시작·증설되면 카운트가 갈라진다. 운영 규모가 커지면 같은 인터페이스로 Firestore 구현을 끼운다.
