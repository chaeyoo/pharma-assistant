"""FastAPI 엔트리포인트 부팅·엔드포인트 계약 테스트.

배경(v12): 종전 테스트는 전부 모듈·함수 단위였고 **ASGI 앱(src.api.main)을 실제로
부팅하는 테스트가 없었다** — 그 결과 v9 의 src/ra 추출이 `_load_ra_tasks` 를
`src/ra/tasks.py::load_ra_tasks` 로 옮겼는데 main.py 의 import 는 옛 이름 그대로라,
**API 엔트리포인트가 import 단계에서 깨진 채 4라운드(v9~v12) 내내 방치**됐다.
pytest·preflight 는 main.py 를 import 하지 않아 못 잡았고, 서버를 실제 기동해야만
드러났다. 이 테스트가 그 사각을 닫는다 — 앱을 TestClient 로 부팅(lifespan 포함)해
대표 엔드포인트가 200 을 주는지 계약으로 고정한다.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app


def test_app_boots_and_serves_endpoints():
    """앱이 부팅되고(lifespan 인덱싱 포함) 4개 엔드포인트가 응답한다."""
    with TestClient(app) as client:
        # 챗 UI
        r_root = client.get("/")
        assert r_root.status_code == 200 and "<" in r_root.text

        # 상태 계기판
        r_health = client.get("/health")
        assert r_health.status_code == 200
        body = r_health.json()
        assert "mode" in body and "verification_gate" in body

        # 마감일 API — v9 의 src/ra 추출 후 import 가 깨졌던 바로 그 경로
        r_dl = client.get("/api/deadlines")
        assert r_dl.status_code == 200
        assert isinstance(r_dl.json(), list)

        # 운영 계기판 — web/dashboard.html 정적 서빙
        r_dash = client.get("/dashboard")
        assert r_dash.status_code == 200
        assert "운영 계기판" in r_dash.text
        # 계기판이 읽는 대상은 /health 다 — 폴링 대상이 바뀌면 화면이 조용히
        # 빈 채로 남으므로(표현 계층의 무증상 고장) 계약으로 고정한다.
        assert "'/health'" in r_dash.text or '"/health"' in r_dash.text

        # 용어 사전 플래시카드 — description/dictionary.html 정적 서빙
        r_dict = client.get("/dictionary")
        assert r_dict.status_code == 200
        assert 'id="mdsrc"' in r_dict.text  # md 원문이 주입된 self-contained 페이지

        # 예제문100 빈칸 퀴즈 — description/blank.html 정적 서빙
        r_blank = client.get("/blank")
        assert r_blank.status_code == 200
        assert "예제문100 빈칸 퀴즈" in r_blank.text
        assert r_blank.text.count("{{") >= 100  # 100문장 분량의 빈칸 마킹이 실려 있다

        # 챗 — 오프라인 모드 근거 기반 답변
        r_chat = client.post("/chat", json={"message": "중대한 이상사례는 며칠 안에 보고?"})
        assert r_chat.status_code == 200
        chat = r_chat.json()
        assert chat.get("answer") and "citations" in chat and "verification" in chat
