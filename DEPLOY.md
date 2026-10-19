# 배포 가이드 — RAPV-Assistant

이 데모(`project/`)를 **공개 URL**로 올리는 방법. 누구나 링크를 눌러
바로 써볼 수 있게 하는 것이 목적이다.

## 0. 먼저 알아둘 것

- **2-프로세스 완전 분리 구조다.** MCP 서버(RAG 인덱스·도구 실행 소유)와
  FastAPI(API·챗 UI)가 별도 프로세스로 뜨고, API 는 `MCP_SERVER_URL`(streamable
  HTTP)로만 도구를 호출한다. 배포도 서비스 2개(또는 컨테이너 2개)로 올린다.
- **오프라인 모드로 배포한다(권장).** API 키 없이도 RAG·PV·검증 전 기능이 동작한다.
  공개 URL 에 `ANTHROPIC_API_KEY` 를 넣으면 **방문자가 당신의 API 비용을 태우므로**
  넣지 않는다. LLM 모드는 라이브 시연처럼 통제된 상황에서만 임시로 켠다(§4).
- 이 앱은 **오래 사는 서버**다(MCP 서버 부팅 시 1회 RAG 인덱싱 + 인메모리 계기판).
  그래서 서버리스(Vercel)보다 **영속 프로세스/컨테이너 호스트(Render·Railway·Fly)** 가 맞다.
- 리포에 배포 파일이 들어 있다: `render.yaml`(레포 루트, 2-서비스 Blueprint),
  `docker-compose.yml`(레포 루트), `project/Dockerfile`, `project/.dockerignore`.
- 자작 규제 코퍼스라 실 PII·비밀이 없어 공개 배포는 안전하다.

로컬에서 배포와 동일한 2-프로세스 방식으로 미리 확인하려면 `./run.sh`
(MCP 서버 8001 기동 → 헬스체크 대기 → API 8000 기동) 또는 수동으로:

```bash
cd project
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# 터미널 1 — MCP 서버
.venv/bin/python -m src.preflight --role mcp && \
  .venv/bin/python -m src.mcp_server.server --transport http --host 127.0.0.1 --port 8001
# 터미널 2 — API 서버
MCP_SERVER_URL=http://127.0.0.1:8001/mcp sh -c \
  '.venv/bin/python -m src.preflight --role api && \
   .venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000'
# → http://127.0.0.1:8000 접속, /health·/api/deadlines·/chat 확인
```

---

## 1. 경로 A — Render (권장, 2-서비스 Blueprint)

유료 플랜(starter) 기준. MCP 서버(rapv-mcp)는 **공개 web 서비스 + Bearer 토큰
인증**으로 띄운다 — 외부 MCP 클라이언트(Claude Desktop 등)의 원격 연결 시연용.
`render.yaml` 을 자동 인식한다.

1. 코드를 GitHub 에 푸시(이미 되어 있으면 생략).
2. [dashboard.render.com](https://dashboard.render.com) 로그인 → **New +** → **Blueprint**.
3. 이 GitHub 리포를 선택 → Render 가 레포 루트의 `render.yaml` 을 읽어 서비스 2개를
   자동 구성한다:
   - **rapv-mcp** (web) — `preflight --role mcp` 통과 후 MCP HTTP 서버 기동(RAG
     인덱스 구축). `MCP_REQUIRE_AUTH=1` 이라 **`MCP_AUTH_TOKEN` 을 대시보드에서
     넣기 전에는 기동을 거부**한다(무인증 공개 서버 방지). `/health` 만 인증 밖
     (헬스체크용).
   - **rapv-assistant** (web) — `preflight --role api` 가 MCP 서버 실연결(도구 6종
     노출)까지 확인한 뒤 uvicorn 기동. MCP 호스트명은 `MCP_SERVER_HOST` 로
     `fromService`(property: host) 자동 주입되고, 포트는 8001 고정 —
     코드의 `src/config.py` 가 URL 로 조립한다. `hostport`/`port` 속성은
     대상 서비스의 '열린 포트 감지' 뒤에야 채워져 첫 sync 에 빈 값이 될 수
     있으므로 쓰지 않는다. 서비스 간 통신은 rapv-mcp 가 공개여도 내부
     네트워크 주소를 유지한다.
4. 두 서비스의 **Environment 에 같은 `MCP_AUTH_TOKEN` 값을 입력**한다
   (`openssl rand -hex 24` 등으로 생성, 커밋 금지).
5. **Apply** → 빌드·배포가 돈다. rapv-mcp 로그에 `배포 전 점검 … 통과` 후 서버
   기동, rapv-assistant 로그에 `MCP 연결` 통과가 보이면 정상.
6. 발급된 URL(`https://rapv-assistant-xxxx.onrender.com`)에 접속 → 챗 UI 가 뜬다.

> **pserv → web 전환 주의**: Render 서비스 타입은 생성 후 변경이 불가하다.
> 기존에 rapv-mcp 를 private service 로 만들었다면 대시보드에서 **삭제 후
> Blueprint sync** 로 web 재생성한다(삭제 → 다음 push 또는 Manual Sync).
>
> MCP 서버를 외부에 열 필요가 없으면 `render.yaml` 에서 rapv-mcp 를
> `type: pserv` 로 되돌리고 인증 env 두 개를 빼면 된다 — 내부 전용이 더
> 보수적인 기본값이다.

**외부 MCP 클라이언트 원격 연결** (Claude Desktop): 커넥터 UI 는 정적 Bearer
토큰을 지원하지 않으므로 `mcp-remote` 프록시를 stdio 로 끼운다
(`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "RAPV-Remote": {
      "command": "npx",
      "args": ["mcp-remote", "https://rapv-mcp-xxxx.onrender.com/mcp",
               "--header", "Authorization: Bearer <MCP_AUTH_TOKEN>"]
    }
  }
}
```

---

## 2. 경로 B — Docker Compose (로컬/VM, 슬립 없음)

레포 루트의 `docker-compose.yml` 이 같은 이미지(`project/Dockerfile`)로 두
컨테이너를 띄운다 — `mcp`(내부 전용 8001, healthcheck 로 인덱스 준비 완료 감지)와
`api`(8000 공개, mcp 가 healthy 된 뒤 기동).

```bash
docker compose up --build
# → http://localhost:8000 (mcp 는 호스트에 노출되지 않음)
```

VM(EC2·클라우드 인스턴스 등)에서도 동일 — 방화벽에서 8000 만 열면 된다.
LLM 모드는 `ANTHROPIC_API_KEY=sk-ant-... docker compose up` 처럼 환경변수로 주입.

**Railway / Fly.io**: `project/Dockerfile` 은 API 롤 단일 컨테이너 기준이므로,
두 서비스를 각각 만들고 mcp 쪽 command 를
`python -m src.preflight --role mcp && python -m src.mcp_server.server --transport http --host 0.0.0.0 --port 8001`
로 오버라이드한 뒤 api 쪽에 `MCP_SERVER_URL` 을 내부 주소로 넣는다.

---

## 3. 배포 확인

URL 뒤에 붙여 확인:

- `/` — 챗 UI(질문 입력 → 근거·출처·검증 배지 표시)
- `/health` — `mode: offline`, `mcp.status: ok`(MCP 서버 연결), 인덱스 문서/청크 수,
  검증 게이트 계기판 JSON. `status: degraded` 면 MCP 서버 쪽을 확인한다.
- `/api/deadlines` — 마감일 목록 JSON
- `/chat` (POST) — `{"message":"..."}` 로 답변·citations·verification 반환

예: `curl -s https://<your-url>/health`

---

## 4. (선택) LLM 모드 켜기 — 통제된 시연에만

오프라인 모드는 규칙 라우터라 도구를 하나씩 부른다. LLM 모드는 Claude 가 복합 질문
("GMP 변경인데 뭘 준비하고 언제까지?")에 여러 도구를 **스스로 연쇄 호출**해 자연어로
종합한다 — 시각적으로 가장 인상적인 부분이다.

- **공개 URL 에는 켜지 말 것**(방문자가 API 비용 소모). 시연이 필요하면 직전에
  플랫폼 대시보드에서 **rapv-assistant(web) 서비스에만** `ANTHROPIC_API_KEY`
  환경변수를 추가 → 재배포 → 시연 후 제거(MCP 서버는 키가 필요 없다).
- 또는 **로컬**에서만 켠다: `project/.env` 에 `ANTHROPIC_API_KEY=sk-ant-...` 후 `./run.sh`.
- 모델 기본값은 `claude-opus-4-8`(`LLM_MODEL` 로 변경 가능).
- 키 오류·네트워크 실패는 조용한 폴백 없이 **명시적 안내**로 처리되므로 안전하다.

---

## 5. 트러블슈팅

| 증상 | 원인·해결 |
|---|---|
| 빌드는 됐는데 기동 실패 | 로그에 preflight 실패가 보이면 데이터·설정 결함 — 로컬에서 `python -m src.preflight` 로 재현·수정 후 재배포(이것이 fail-closed 게이트의 의도된 동작) |
| api 가 `MCP 연결` 실패로 기동 중단 | 실패 메시지의 '주소 출처'를 본다 — "환경변수 미주입"이면 플랫폼의 `MCP_SERVER_HOST`(또는 `MCP_SERVER_URL`) 주입 문제, 주입은 됐는데 실패면 rapv-mcp(또는 mcp 컨테이너)가 아직 안 떴거나 죽은 것 — 해당 서비스 로그 확인. preflight 는 60초까지 기동을 기다리고, Render 는 실패 시 자동 재시도한다 |
| `/health` 가 `status: degraded` | API 는 떠 있는데 MCP 서버 도달 불가 — MCP 서비스 재시작·주소 확인 |
| MCP 호출이 421 Misdirected Request | HTTP 기동 시 `allowed_hosts` 누락 — `src/mcp_server/server.py` 의 http 분기(`allowed_hosts=["*"]`)가 살아 있는지 확인 |
| 404 / 앱이 안 뜸 | Root Directory 가 `project` 인지 확인(레포 루트가 아님) |
| 포트 에러 | 기동 명령이 `--port $PORT` 인지 확인(플랫폼이 포트를 주입) |
| LLM 모드가 안 켜짐 | `ANTHROPIC_API_KEY` 환경변수 설정 후 **재배포** 필요(빌드 시점 주입) |

---

**요약**: Render 에 `render.yaml` 블루프린트로 2-서비스(오프라인 모드) 배포 →
공개 URL 획득. 로컬·VM 은 `docker compose up --build` 한 방이다.
LLM 라이브 시연이 필요할 때만 web 서비스에 키를 임시로 켠다.
