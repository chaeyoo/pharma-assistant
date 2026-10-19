# RAPV-Assistant — 배포용 컨테이너 이미지
# 컨테이너 호스트(Railway·Fly.io·Render(Docker)·Cloud Run 등) 어디에나 올라간다.
# 기본 이미지는 밀집임베딩(ml) 없이 빌드한다 — EMBEDDER_KIND=auto 가 TF-IDF 로
# 폴백하므로 이미지가 가볍고 콜드스타트가 빠르다. 밀집임베딩을 컨테이너에서 쓰려면
# 아래 requirements-ml 설치 + 모델 다운로드 두 줄을 활성화한다(이미지 수백 MB 증가).
FROM python:3.11-slim

WORKDIR /app

# 런타임 의존성만 설치(테스트 의존성 requirements-dev 는 제외)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 소스 복사
COPY . .

# 밀집임베딩 이미지로 빌드하려면 주석 해제(모델까지 이미지에 굽는다):
# RUN pip install --no-cache-dir -r requirements-ml.txt \
#  && python scripts/download_embedding_model.py

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# 기본 CMD = API 롤. 완전 분리 구조라 MCP_SERVER_URL 이 실행 중인 MCP 서버를
# 가리켜야 한다(docker-compose.yml 의 mcp 서비스가 같은 이미지에서 command 로
# MCP 롤을 기동한다). 기동 게이트: preflight --role api 는 자기 데이터 점검 후
# MCP 서버 실연결(도구 6종 노출)까지 기다린다 — 실패하면 컨테이너가 뜨지 않는다.
# 플랫폼이 주입하는 $PORT 로 0.0.0.0 바인딩(없으면 8000). 오프라인 모드가 기본,
# LLM 모드는 ANTHROPIC_API_KEY 환경변수를 줄 때만 활성화된다.
CMD ["sh", "-c", "python -m src.preflight --role api && exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
