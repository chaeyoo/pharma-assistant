#!/usr/bin/env bash
# RAPV-Assistant 원커맨드 실행 스크립트
set -e

cd "$(dirname "$0")"

# 1) 가상환경 준비
if [ ! -d ".venv" ]; then
  echo "▶ 가상환경 생성 및 의존성 설치..."
  python3 -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet -r requirements.txt
fi

# 2) .env 로드 (있으면)
if [ -f ".env" ]; then
  set -a; . ./.env; set +a
fi

# 3) 배포 전 점검 — 데이터·설정·스모크·검증 게이트 자가 테스트 (실패 시 기동 중단)
echo "▶ 배포 전 점검(preflight)..."
.venv/bin/python -m src.preflight

# 4) MCP 서버 기동 (완전 분리 — 별도 프로세스, HTTP transport)
echo "▶ MCP 서버 시작 → http://127.0.0.1:8001/mcp"
.venv/bin/python -m src.mcp_server.server --transport http --host 127.0.0.1 --port 8001 &
MCP_PID=$!
trap 'kill $MCP_PID 2>/dev/null' EXIT
export MCP_SERVER_URL="http://127.0.0.1:8001/mcp"
until .venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=2)" 2>/dev/null; do
  sleep 1
done

# 5) API 서버 실행 (exec 금지 — EXIT trap 이 MCP 자식 프로세스를 정리해야 한다)
echo "▶ RAPV-Assistant 서버 시작 → http://127.0.0.1:8000"
.venv/bin/python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
