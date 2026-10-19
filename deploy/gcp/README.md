# Google Cloud Run 배포

RAPV-Assistant 를 Cloud Run 서비스 2개로 배포한다. 같은 컨테이너 이미지를 쓰고 실행 명령만 다르다.

```
사용자 ──HTTPS──▶ pharma-api (FastAPI · 챗 UI · Google 로그인 · 일일 질문 상한)
                      │  Bearer 토큰(Secret Manager)
                      ▼
                  pharma-mcp (FastMCP 도구 서버 · RAG 인덱스)
```

| 리소스 | 용도 |
|---|---|
| Artifact Registry `pharma` | 컨테이너 이미지 (최신 3개만 보관 — 무료 0.5GB 한도) |
| Cloud Build | `project/Dockerfile` 빌드 |
| Secret Manager | `pa-mcp-auth-token` · `pa-session-secret` · `pa-google-client-secret` · `pa-anthropic-api-key` |
| Cloud Run `pharma-mcp` / `pharma-api` | us-central1, 인스턴스 0~2대, 1 vCPU · 1GiB |

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

## 확인

- 챗 UI: `https://<pharma-api URL>/`
- 상태: `…/health` · 운영 계기판: `…/dashboard`
- 로그: 콘솔 → Cloud Run → 서비스 → **로그** 탭

## 설계 메모

- **MCP 서버가 공개 URL 인 이유**: 외부 MCP 클라이언트(Claude Desktop 등)도 같은 도구 서버에 붙을 수 있게 한다. 대신 Bearer 토큰이 필수이고 `MCP_REQUIRE_AUTH=1` 이라 토큰 없이 뜨지 않는다. 사내 배포라면 ingress 를 내부 전용으로 바꾸고 Cloud Run IAM(서비스 계정 ID 토큰)으로 서비스 간 인증을 하는 것이 다음 단계다.
- **로그인**: Google 을 IdP 로 쓰는 OIDC. 사내 SSO 로 옮기려면 IdP 메타데이터 주소와 클라이언트 자격만 교체한다(`project/src/api/auth.py`).
- **호출 상한 저장소**: 현재는 인스턴스 메모리라 인스턴스가 재시작·증설되면 카운트가 갈라진다. 운영 규모가 커지면 같은 인터페이스로 Firestore 구현을 끼운다.
