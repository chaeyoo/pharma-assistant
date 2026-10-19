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

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TAG="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)"
IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$AR_REPO/app:$TAG"

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
[[ -n "${GOOGLE_CLIENT_SECRET:-}" ]] && put_secret pa-google-client-secret "$GOOGLE_CLIENT_SECRET"
[[ -n "${ANTHROPIC_API_KEY:-}"   ]] && put_secret pa-anthropic-api-key "$ANTHROPIC_API_KEY"

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
say "컨테이너 이미지 빌드 (Cloud Build, 3~5분): $IMAGE"
# 방금 부여한 권한이 퍼지기 전이면 실패할 수 있어 1분 간격으로 최대 3번 시도한다.
for TRY in 1 2 3; do
  if gcloud builds submit "$ROOT/project" --tag "$IMAGE" --region "$REGION" \
       --gcs-source-staging-dir "$BUILD_BUCKET/source" \
       --default-buckets-behavior regional-user-owned-bucket --quiet; then
    break
  fi
  [[ "$TRY" == 3 ]] && { echo "빌드 실패 — 위 에러를 확인하세요." >&2; exit 1; }
  echo "권한 전파 대기 후 다시 시도합니다 ($TRY/3)…"; sleep 60
done

# ---------------------------------------------------------------- 6. MCP 서버
# 도구 서버. 외부 MCP 클라이언트도 붙을 수 있게 공개 URL 이지만 Bearer 토큰이 필수이고,
# 토큰이 비면 preflight 가 기동을 거부한다(MCP_REQUIRE_AUTH=1).
say "MCP 서버 배포: $MCP_SERVICE"
gcloud run deploy "$MCP_SERVICE" --image "$IMAGE" --region "$REGION" --quiet \
  --allow-unauthenticated --port 8080 --memory 1Gi --cpu 1 \
  --min-instances 0 --max-instances 2 --timeout 300 \
  --set-env-vars "MCP_REQUIRE_AUTH=1" \
  --set-secrets "MCP_AUTH_TOKEN=pa-mcp-auth-token:latest" \
  --command sh \
  --args "^##^-c##python -m src.preflight --role mcp && exec python -m src.mcp_server.server --transport http --host 0.0.0.0 --port 8080"
MCP_URL="$(gcloud run services describe "$MCP_SERVICE" --region "$REGION" --format='value(status.url)')"

# ---------------------------------------------------------------- 7. API + 챗 UI
API_URL="$(gcloud run services describe "$API_SERVICE" --region "$REGION" --format='value(status.url)' 2>/dev/null || true)"
API_URL="${API_URL:-https://$API_SERVICE-$PROJECT_NUMBER.$REGION.run.app}"

ENV="MCP_SERVER_URL=$MCP_URL/mcp##AGENT_BACKEND=$AGENT_BACKEND##DAILY_CHAT_LIMIT=$DAILY_CHAT_LIMIT"
SECRETS="MCP_AUTH_TOKEN=pa-mcp-auth-token:latest,SESSION_SECRET=pa-session-secret:latest"
if [[ "$LOGIN_ON" == 1 ]]; then
  ENV="$ENV##GOOGLE_CLIENT_ID=$GOOGLE_CLIENT_ID##PUBLIC_BASE_URL=$API_URL##REQUIRE_LOGIN=1"
  [[ -n "$ALLOWED_EMAILS"  ]] && ENV="$ENV##ALLOWED_EMAILS=$ALLOWED_EMAILS"
  [[ -n "$ALLOWED_DOMAINS" ]] && ENV="$ENV##ALLOWED_DOMAINS=$ALLOWED_DOMAINS"
  SECRETS="$SECRETS,GOOGLE_CLIENT_SECRET=pa-google-client-secret:latest"
fi
[[ "$LLM_ON" == 1 ]] && SECRETS="$SECRETS,ANTHROPIC_API_KEY=pa-anthropic-api-key:latest"

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
echo "   모드      : $([[ $LLM_ON == 1 ]] && echo 'LLM 모드' || echo '오프라인 모드')"
if [[ "$LOGIN_ON" != 1 ]]; then
  echo
  echo " 다음 단계 — Google 로그인용 OAuth 클라이언트의 승인된 리디렉션 URI:"
  echo "   $API_URL/auth/callback"
fi
echo "=================================================================="
