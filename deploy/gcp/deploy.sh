#!/usr/bin/env bash
# RAPV-Assistant — Google Cloud Run 배포 스크립트 (Cloud Shell 에서 실행)
#
#   bash deploy/gcp/deploy.sh <PROJECT_ID>
#
# 여러 번 실행해도 안전하다(이미 있는 리소스는 건너뛰고, 서비스는 새 이미지로 갱신).
# 단계별로 기능이 켜진다 — 환경변수를 주고 다시 실행하면 그 기능이 추가된다:
#
#   1차  bash deploy/gcp/deploy.sh <PROJECT_ID>
#        → 오프라인 모드(API 키 없음) 공개 데모. 로그인 없음.
#   2차  GOOGLE_CLIENT_ID=... GOOGLE_CLIENT_SECRET=... ALLOWED_EMAILS=a@gmail.com,b@gmail.com \
#        bash deploy/gcp/deploy.sh <PROJECT_ID>
#        → Google 로그인 + 사용자별 하루 질문 상한 + 공개 가드(REQUIRE_LOGIN=1)
#   3차  ANTHROPIC_API_KEY=sk-ant-... bash deploy/gcp/deploy.sh <PROJECT_ID>
#        → LLM 모드. 로그인이 켜져 있지 않으면 거부한다(비용 보호).
#   4차  WITH_ML=1 PGVECTOR_URL='postgresql://...?sslmode=require' bash deploy/gcp/deploy.sh <PROJECT_ID>
#        → 밀집임베딩(e5) 이미지 + pgvector 영속 벡터 스토어(Supabase 등 외부 Postgres).
#          pharma-mcp 만 바뀐다(메모리 2GiB·EMBEDDER_KIND=dense·VECTOR_STORE=pgvector).
#          WITH_ML=1 만 주면 밀집임베딩 + 인메모리 스토어. 되돌리기: WITH_ML=0 으로 재실행.
#
# 비밀값(API 키·시크릿)은 Secret Manager 에 한 번 저장되고, 이후 실행에서는 다시 줄
# 필요가 없다. 비용: 요청이 없으면 인스턴스 0대(min-instances=0), 최대 2대로 제한.
set -euo pipefail

PROJECT_ID="${1:-${PROJECT_ID:-}}"
if [[ -z "$PROJECT_ID" ]]; then
  echo "사용법: bash deploy/gcp/deploy.sh <PROJECT_ID>" >&2
  exit 1
fi
REGION="${REGION:-us-central1}"        # 무료 등급 데이터 전송 한도가 북미 기준
AR_REPO="${AR_REPO:-pharma}"
MCP_SERVICE="${MCP_SERVICE:-pharma-mcp}"
API_SERVICE="${API_SERVICE:-pharma-api}"
DAILY_CHAT_LIMIT="${DAILY_CHAT_LIMIT:-30}"
AGENT_BACKEND="${AGENT_BACKEND:-langgraph}"
PGVECTOR_POOL="${PGVECTOR_POOL:-0}"      # 4차: 벡터 후보 풀(0=전수, 인메모리와 동일 결과)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TAG="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)"
IMAGE_BASE="$REGION-docker.pkg.dev/$PROJECT_ID/$AR_REPO/app"   # 태그는 4차 판정 뒤(ML 여부)에 붙인다

say() { printf '\n\033[1;36m▶ %s\033[0m\n' "$*"; }

gcloud config set project "$PROJECT_ID" >/dev/null
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
RUNTIME_SA="$PROJECT_NUMBER-compute@developer.gserviceaccount.com"

# ---------------------------------------------------------------- 1. API 켜기
say "필요한 Google Cloud API 활성화 (처음 한 번은 1~2분 걸림)"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  cloudbuild.googleapis.com secretmanager.googleapis.com storage.googleapis.com --quiet

# ---------------------------------------------------------------- 2. 이미지 저장소
if ! gcloud artifacts repositories describe "$AR_REPO" --location "$REGION" >/dev/null 2>&1; then
  say "Artifact Registry 저장소 생성: $AR_REPO"
  gcloud artifacts repositories create "$AR_REPO" --repository-format=docker \
    --location "$REGION" --description "RAPV-Assistant images" --quiet
fi
# 무료 저장 한도(0.5GB)를 넘지 않게 최신 3개만 남긴다
POLICY="$(mktemp)"
cat > "$POLICY" <<'JSON'
[
  {"name": "keep-latest-3", "action": {"type": "Keep"}, "mostRecentVersions": {"keepCount": 3}},
  {"name": "delete-older", "action": {"type": "Delete"}, "condition": {"tagState": "any", "olderThan": "1d"}}
]
JSON
gcloud artifacts repositories set-cleanup-policies "$AR_REPO" --location "$REGION" \
  --policy "$POLICY" --no-dry-run --quiet >/dev/null
rm -f "$POLICY"

# 빌드 소스 업로드용 버킷 — Cloud Build 가 자동으로 만드는 <PROJECT>_cloudbuild 버킷은
# 새 프로젝트에서 권한 전파 전이면 PERMISSION_DENIED 로 막히는 일이 잦아 직접 만든다.
# 7일 지난 소스 압축본은 자동 삭제(무료 저장 한도 보호).
BUILD_BUCKET="gs://$PROJECT_ID-build-src"
if ! gcloud storage buckets describe "$BUILD_BUCKET" >/dev/null 2>&1; then
  say "빌드 소스 버킷 생성: $BUILD_BUCKET"
  gcloud storage buckets create "$BUILD_BUCKET" --location "$REGION" \
    --uniform-bucket-level-access --quiet
  LIFECYCLE="$(mktemp)"
  echo '{"rule":[{"action":{"type":"Delete"},"condition":{"age":7}}]}' > "$LIFECYCLE"
  gcloud storage buckets update "$BUILD_BUCKET" --lifecycle-file "$LIFECYCLE" --quiet >/dev/null
  rm -f "$LIFECYCLE"
fi

# ---------------------------------------------------------------- 3. 권한
# 최근 프로젝트는 Cloud Build 가 기본 Compute 서비스 계정으로 돈다 — 빌드·푸시 권한 부여.
say "서비스 계정 권한 확인"
for ROLE in roles/cloudbuild.builds.builder roles/artifactregistry.writer roles/logging.logWriter; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --member "serviceAccount:$RUNTIME_SA" \
    --role "$ROLE" --condition=None --quiet >/dev/null
done
gcloud storage buckets add-iam-policy-binding "$BUILD_BUCKET" \
  --member "serviceAccount:$RUNTIME_SA" --role roles/storage.objectAdmin --quiet >/dev/null

# ---------------------------------------------------------------- 4. 비밀값
secret_exists() { gcloud secrets describe "$1" >/dev/null 2>&1; }
put_secret() {  # put_secret NAME VALUE — 없으면 만들고, 있으면 새 버전 추가
  local name="$1" value="$2"
  if secret_exists "$name"; then
    printf '%s' "$value" | gcloud secrets versions add "$name" --data-file=- --quiet >/dev/null
  else
    printf '%s' "$value" | gcloud secrets create "$name" --replication-policy=automatic \
      --data-file=- --quiet >/dev/null
  fi
  gcloud secrets add-iam-policy-binding "$name" --member "serviceAccount:$RUNTIME_SA" \
    --role roles/secretmanager.secretAccessor --quiet >/dev/null
}
rand() { python3 -c 'import secrets;print(secrets.token_urlsafe(48))'; }

say "Secret Manager 비밀값 준비"
secret_exists pa-mcp-auth-token  || put_secret pa-mcp-auth-token "$(rand)"
secret_exists pa-session-secret  || put_secret pa-session-secret "$(rand)"
# 붙여넣기 실수(예시 문구가 그대로 남음, 공백·한글 섞임)를 저장 전에 거른다 —
# 이런 값은 배포는 되지만 로그인 때 토큰 교환이 UnicodeEncodeError 로 실패한다.
check_ascii_token() {  # check_ascii_token NAME VALUE PREFIX
  if ! printf '%s' "$2" | LC_ALL=C grep -qE '^[A-Za-z0-9._~+/=-]+$'; then
    echo "$1 값에 영문·숫자 외 문자(한글·공백 등)가 있습니다. 콘솔에서 복사한 값을 그대로 넣어 주세요." >&2; exit 1
  fi
  if [[ -n "$3" && "$2" != "$3"* ]]; then
    echo "$1 값이 '$3' 로 시작하지 않습니다. 올바른 값인지 확인해 주세요." >&2; exit 1
  fi
}
[[ -n "${GOOGLE_CLIENT_ID:-}"     ]] && check_ascii_token GOOGLE_CLIENT_ID "$GOOGLE_CLIENT_ID" ""
[[ -n "${GOOGLE_CLIENT_SECRET:-}" ]] && check_ascii_token GOOGLE_CLIENT_SECRET "$GOOGLE_CLIENT_SECRET" "GOCSPX-"
[[ -n "${ANTHROPIC_API_KEY:-}"    ]] && check_ascii_token ANTHROPIC_API_KEY "$ANTHROPIC_API_KEY" "sk-ant-"
[[ -n "${GOOGLE_CLIENT_SECRET:-}" ]] && put_secret pa-google-client-secret "$GOOGLE_CLIENT_SECRET"
[[ -n "${ANTHROPIC_API_KEY:-}"   ]] && put_secret pa-anthropic-api-key "$ANTHROPIC_API_KEY"
# 4차 — pgvector 접속 URL. 접속 문자열은 ':@/?=' 를 품으므로 check_ascii_token 대신 따로 검사한다.
if [[ -n "${PGVECTOR_URL:-}" ]]; then
  if ! printf '%s' "$PGVECTOR_URL" | LC_ALL=C grep -qE '^postgres(ql)?://[!-~]+$'; then
    echo "PGVECTOR_URL 은 postgresql:// 로 시작하고 공백·한글이 없어야 합니다(비밀번호 특수문자는 퍼센트 인코딩)." >&2; exit 1
  fi
  case "$PGVECTOR_URL" in
    *sslmode=*) ;;
    *) echo "경고: PGVECTOR_URL 에 sslmode=require 가 없습니다 — 외부 Postgres(Supabase 등)는 TLS 를 켜 두세요." >&2 ;;
  esac
  put_secret pa-pgvector-url "$PGVECTOR_URL"
fi

# 로그인 설정은 서비스에 이미 있으면 유지한다(2차 이후 재실행 때 다시 안 줘도 되게)
current_env() {  # current_env SERVICE KEY
  { gcloud run services describe "$1" --region "$REGION" --format=json 2>/dev/null || true; } \
    | python3 -c "import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit()
for e in d['spec']['template']['spec']['containers'][0].get('env',[]):
    if e.get('name')=='$2' and 'value' in e: print(e['value'])"
}
GOOGLE_CLIENT_ID="${GOOGLE_CLIENT_ID:-$(current_env "$API_SERVICE" GOOGLE_CLIENT_ID)}"
ALLOWED_EMAILS="${ALLOWED_EMAILS:-$(current_env "$API_SERVICE" ALLOWED_EMAILS)}"
ALLOWED_DOMAINS="${ALLOWED_DOMAINS:-$(current_env "$API_SERVICE" ALLOWED_DOMAINS)}"
# LLM 모드 생성 모델 — 기본값은 코드가 아니라 project/config/models.json 의 answer 에서 온다
# (선정 근거: project/eval/model_compare/REPORT.md). 바꾸려면 LLM_MODEL=... 로 재실행.
LLM_MODEL="${LLM_MODEL:-$(current_env "$API_SERVICE" LLM_MODEL)}"
LLM_MODEL="${LLM_MODEL:-$(python3 -c "import json;print(json.load(open('$ROOT/project/config/models.json'))['answer'])")}"

# 4차 — 밀집임베딩 이미지 여부. 미지정이면 배포된 MCP 서비스의 상태를 따른다(재실행 때 유지).
# 되돌리려면 WITH_ML=0 을 명시한다.
if [[ -z "${WITH_ML:-}" ]]; then
  [[ "$(current_env "$MCP_SERVICE" EMBEDDER_KIND)" == "dense" ]] && WITH_ML=1 || WITH_ML=0
fi
[[ "$WITH_ML" == 1 ]] || WITH_ML=0
PGVECTOR_ON=0
if secret_exists pa-pgvector-url; then
  if [[ "$WITH_ML" != 1 ]]; then
    echo "pgvector 스토어(pa-pgvector-url)는 밀집임베딩 이미지가 필요합니다 — WITH_ML=1 로 재실행하세요." >&2
    echo "(pgvector 를 끄려면: gcloud secrets delete pa-pgvector-url)" >&2
    exit 1
  fi
  PGVECTOR_ON=1
fi
IMAGE="$IMAGE_BASE:$TAG"; [[ "$WITH_ML" == 1 ]] && IMAGE="$IMAGE-ml"   # 같은 커밋의 경량·ML 이미지 구분

LOGIN_ON=0
if [[ -n "$GOOGLE_CLIENT_ID" ]]; then
  secret_exists pa-google-client-secret || { echo "GOOGLE_CLIENT_SECRET 도 함께 주세요." >&2; exit 1; }
  LOGIN_ON=1
fi
LLM_ON=0
if secret_exists pa-anthropic-api-key; then
  if [[ "$LOGIN_ON" != 1 ]]; then
    echo "LLM 모드는 로그인을 켠 뒤에만 배포한다(공개 URL 의 API 비용 보호). 2차 단계를 먼저 하세요." >&2
    exit 1
  fi
  LLM_ON=1
fi

# ---------------------------------------------------------------- 5. 빌드
say "컨테이너 이미지 빌드 (Cloud Build, $([[ $WITH_ML == 1 ]] && echo '밀집임베딩 포함 8~12분' || echo '3~5분')): $IMAGE"
# --tag 는 build-arg 를 못 받으므로 cloudbuild.yaml 을 거친다(WITH_ML 치환).
# 방금 부여한 권한이 퍼지기 전이면 실패할 수 있어 1분 간격으로 최대 3번 시도한다.
for TRY in 1 2 3; do
  if gcloud builds submit "$ROOT/project" --config "$ROOT/deploy/gcp/cloudbuild.yaml" \
       --substitutions "_IMAGE=$IMAGE,_WITH_ML=$WITH_ML" --region "$REGION" \
       --gcs-source-staging-dir "$BUILD_BUCKET/source" \
       --default-buckets-behavior regional-user-owned-bucket --quiet; then
    break
  fi
  [[ "$TRY" == 3 ]] && { echo "빌드 실패 — 위 에러를 확인하세요." >&2; exit 1; }
  echo "권한 전파 대기 후 다시 시도합니다 ($TRY/3)…"; sleep 60
done

# ---------------------------------------------------------------- 6. MCP 서버
# 도구 서버. 외부 MCP 클라이언트도 붙을 수 있게 공개 URL 이지만 Bearer 토큰이 필수이고,
# 토큰이 비면 preflight 가 기동을 거부한다(MCP_REQUIRE_AUTH=1). preflight 는 서버 프로세스 안에서
# 돈다(--preflight) — 별도 프로세스로 돌리면 임베딩 모델을 두 번 로드해 콜드스타트가 ~20s 길어진다.
# 4차: 밀집임베딩이면 torch+e5 로딩 몫으로 메모리 2GiB, 콜드스타트 단축용 CPU 부스트.
# --set-env-vars/--set-secrets 는 전체 교체라 WITH_ML=0 으로 재실행하면 그대로 원복된다.
MCP_ENV="MCP_REQUIRE_AUTH=1"
MCP_SECRETS="MCP_AUTH_TOKEN=pa-mcp-auth-token:latest"
MCP_RES=(--memory 1Gi --cpu 1)
if [[ "$WITH_ML" == 1 ]]; then
  MCP_ENV="$MCP_ENV##EMBEDDER_KIND=dense"
  MCP_RES=(--memory 2Gi --cpu 1 --cpu-boost)
fi
if [[ "$PGVECTOR_ON" == 1 ]]; then
  MCP_ENV="$MCP_ENV##VECTOR_STORE=pgvector##PGVECTOR_POOL=$PGVECTOR_POOL"
  MCP_SECRETS="$MCP_SECRETS,PGVECTOR_URL=pa-pgvector-url:latest"
fi
say "MCP 서버 배포: $MCP_SERVICE"
gcloud run deploy "$MCP_SERVICE" --image "$IMAGE" --region "$REGION" --quiet \
  --allow-unauthenticated --port 8080 "${MCP_RES[@]}" \
  --min-instances 0 --max-instances 2 --timeout 300 \
  --set-env-vars "^##^$MCP_ENV" \
  --set-secrets "$MCP_SECRETS" \
  --command sh \
  --args "^##^-c##exec python -m src.mcp_server.server --preflight --transport http --host 0.0.0.0 --port 8080"
MCP_URL="$(gcloud run services describe "$MCP_SERVICE" --region "$REGION" --format='value(status.url)')"

# ---------------------------------------------------------------- 7. API + 챗 UI
API_URL="$(gcloud run services describe "$API_SERVICE" --region "$REGION" --format='value(status.url)' 2>/dev/null || true)"
API_URL="${API_URL:-https://$API_SERVICE-$PROJECT_NUMBER.$REGION.run.app}"

# MCP 대기는 짧게·soft — Cloud Run 은 두 서비스가 함께 0대로 내려가므로 API 가 MCP 콜드스타트를
# 직렬로 기다리면 첫 페이지까지 2분이 걸렸다(실측). 페이지를 먼저 내고 UI 의 /warmup 이 준비를 보여 준다.
ENV="MCP_SERVER_URL=$MCP_URL/mcp##AGENT_BACKEND=$AGENT_BACKEND##DAILY_CHAT_LIMIT=$DAILY_CHAT_LIMIT##MCP_WAIT_TIMEOUT_S=5##MCP_WAIT_SOFT=1"
SECRETS="MCP_AUTH_TOKEN=pa-mcp-auth-token:latest,SESSION_SECRET=pa-session-secret:latest"
if [[ "$LOGIN_ON" == 1 ]]; then
  ENV="$ENV##GOOGLE_CLIENT_ID=$GOOGLE_CLIENT_ID##PUBLIC_BASE_URL=$API_URL##REQUIRE_LOGIN=1"
  [[ -n "$ALLOWED_EMAILS"  ]] && ENV="$ENV##ALLOWED_EMAILS=$ALLOWED_EMAILS"
  [[ -n "$ALLOWED_DOMAINS" ]] && ENV="$ENV##ALLOWED_DOMAINS=$ALLOWED_DOMAINS"
  SECRETS="$SECRETS,GOOGLE_CLIENT_SECRET=pa-google-client-secret:latest"
fi
if [[ "$LLM_ON" == 1 ]]; then
  SECRETS="$SECRETS,ANTHROPIC_API_KEY=pa-anthropic-api-key:latest"
  ENV="$ENV##LLM_MODEL=$LLM_MODEL"
fi

say "API·챗 UI 배포: $API_SERVICE"
gcloud run deploy "$API_SERVICE" --image "$IMAGE" --region "$REGION" --quiet \
  --allow-unauthenticated --memory 1Gi --cpu 1 \
  --min-instances 0 --max-instances 2 --timeout 300 \
  --set-env-vars "^##^$ENV" \
  --set-secrets "$SECRETS"
API_URL="$(gcloud run services describe "$API_SERVICE" --region "$REGION" --format='value(status.url)')"

# ---------------------------------------------------------------- 결과
echo
echo "=================================================================="
echo " 배포 완료"
echo "   챗 UI     : $API_URL"
echo "   운영 계기판: $API_URL/dashboard"
echo "   MCP 서버  : $MCP_URL/mcp"
echo "   로그인    : $([[ $LOGIN_ON == 1 ]] && echo '켜짐 (Google)' || echo '꺼짐')"
echo "   모드      : $([[ $LLM_ON == 1 ]] && echo "LLM 모드 ($LLM_MODEL)" || echo '오프라인 모드')"
echo "   임베더    : $([[ $WITH_ML == 1 ]] && echo '밀집임베딩 (e5, 이미지 포함)' || echo 'TF-IDF 폴백 (경량 이미지)')"
echo "   벡터 스토어: $([[ $PGVECTOR_ON == 1 ]] && echo "pgvector (pa-pgvector-url, pool=$PGVECTOR_POOL)" || echo '인메모리 (부팅마다 재색인)')"
if [[ "$LOGIN_ON" != 1 ]]; then
  echo
  echo " 다음 단계 — Google 로그인용 OAuth 클라이언트의 승인된 리디렉션 URI:"
  echo "   $API_URL/auth/callback"
fi
echo "=================================================================="
