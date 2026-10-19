"""pytest 공용 fixture. 무거운 인덱스 구축은 세션 1회로 공유."""
from __future__ import annotations

import pytest

from src import config
from src.mcp_server.http_harness import run_mcp_http_server
from src.rag.pipeline import RagPipeline


@pytest.fixture(scope="session", autouse=True)
def mcp_server_url():
    """세션 1회, 실제 streamable HTTP transport 로 MCP 서버를 기동한다.

    프로덕션 코드(에이전트·API)는 인메모리 경로 없이 항상 MCP_SERVER_URL(HTTP)
    로 붙는다 — 테스트도 같은 전송 계층을 태우되, 호스팅만 테스트 프로세스 안
    (uvicorn 데몬 스레드, ephemeral 포트)에서 한다(도커 없이 플레인 pytest 로
    실행 가능해야 하므로). URL 은 config 속성 주입으로 전달한다 — 에이전트·
    preflight 가 호출 시점에 config 를 읽는 이유다.
    """
    with run_mcp_http_server() as url:
        config.MCP_SERVER_URL = url
        config.MCP_SERVER_HEALTH_URL = url.rsplit("/mcp", 1)[0] + "/health"
        yield url


@pytest.fixture(scope="session")
def pipeline() -> RagPipeline:
    return RagPipeline().build()
