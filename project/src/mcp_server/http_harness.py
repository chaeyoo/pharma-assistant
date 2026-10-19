"""테스트·평가용 하니스 — MCP 서버를 실제 streamable HTTP transport 로 프로세스 내 기동.

프로덕션 코드(에이전트·API)는 인메모리 경로 없이 항상 MCP_SERVER_URL(HTTP)로 붙는다.
pytest·eval 스크립트를 도커 없이 플레인하게 돌리기 위해, 같은 HTTP wire 스택을
uvicorn 데몬 스레드로 프로세스 안에 띄우고 ephemeral 포트의 URL 을 돌려준다.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import uvicorn

from .server import mcp


@contextmanager
def run_mcp_http_server(host: str = "127.0.0.1", server_obj=None) -> Iterator[str]:
    """MCP HTTP 서버를 백그라운드 스레드로 기동하고 접속 URL 을 yield 한다.

    port=0 → OS 가 빈 포트를 할당(테스트 병렬 실행 충돌 방지).
    server_obj: 기본은 프로덕션 서버(mcp) — 인증 등 다른 구성의 FastMCP 인스턴스를
    검증하는 테스트만 별도 서버를 넘긴다.
    """
    # stateless_http=True — 프로덕션 기동(server.py --transport http)과 동일 구성.
    # 테스트가 프로덕션과 다른 세션 모드로 돌면 세션 계층 결함이 여기서 안 보인다.
    app = (server_obj or mcp).http_app(stateless_http=True)
    server = uvicorn.Server(
        uvicorn.Config(app, host=host, port=0, log_level="error", lifespan="on")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("MCP HTTP 서버 기동 실패(30초 초과 또는 스레드 종료)")
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://{host}:{port}/mcp"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
