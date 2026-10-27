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

import re
import shutil
import subprocess
from pathlib import Path

import pytest
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

        # 개인정보처리방침 — web/privacy.html 정적 서빙. Google OAuth 동의 화면이 링크하는
        # 페이지라 로그인 전에도 열려야 한다(로그인 게이트가 붙으면 Google 측 링크가 깨진다).
        r_priv = client.get("/privacy")
        assert r_priv.status_code == 200
        assert "개인정보처리방침" in r_priv.text
        assert "openid" in r_priv.text and "email" in r_priv.text  # 실제 요청 스코프와 일치

        # 챗 — 오프라인 모드 근거 기반 답변
        r_chat = client.post("/chat", json={"message": "중대한 이상사례는 며칠 안에 보고?"})
        assert r_chat.status_code == 200
        chat = r_chat.json()
        assert chat.get("answer") and "citations" in chat and "verification" in chat


def _sse_events(resp) -> list[dict]:
    import json as _json

    events = []
    for line in resp.iter_lines():
        if line.startswith("data: "):
            events.append(_json.loads(line[6:]))
    return events


def test_chat_stream_emits_steps_then_result():
    """POST /chat/stream — 진행 단계(start/end)가 흘러나온 뒤 /chat 과 같은 모양의 result 로 끝난다.

    UI 가 실시간 단계를 그리는 계약: 도구 호출(kind=tool)의 start 와 end 가 모두 있고,
    검색 서버 연결(mcp.connect)이 가장 먼저 시작되며(콜드스타트 표시의 근거), 마지막
    이벤트는 type=result 이고 /chat 응답과 같은 키(answer·citations·verification)를 가진다."""
    with TestClient(app) as client:
        with client.stream("POST", "/chat/stream", json={"message": "중대한 이상사례는 며칠 안에 보고?"}) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            events = _sse_events(r)
    assert events, "이벤트가 하나도 없다"
    assert events[-1]["type"] == "result"
    res = events[-1]
    assert res.get("answer") and "citations" in res and "verification" in res
    kinds = [(e["type"], e.get("kind"), e.get("name")) for e in events[:-1]]
    assert kinds[0] == ("start", "agent", "chat")  # 요청 전체 span — UI 는 이 kind 를 생략한다
    steps = [k for k in kinds if k[1] != "agent"]
    assert steps[0] == ("start", "pipeline", "mcp.connect")
    assert any(t == "start" and k == "tool" for t, k, _ in kinds)
    assert any(t == "end" and k == "tool" for t, k, _ in kinds)
    assert any(n == "verify" and t == "end" for t, _, n in kinds)
    # 동기 /chat 과 같은 답을 준다(경로만 다를 뿐 처리는 동일)
    sync = client.post("/chat", json={"message": "중대한 이상사례는 며칠 안에 보고?"}).json()
    assert sync["answer"] == res["answer"]


def test_warmup_reports_mcp_ready():
    """GET /warmup — 검색 서버가 떠 있으면 ok 와 소요 ms 를 준다(페이지 로드 시 예열 호출의 계약)."""
    with TestClient(app) as client:
        r = client.get("/warmup")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok" and isinstance(body["ms"], int)
        assert body["rag"]["chunks"] > 0


_WEB_DIR = Path(__file__).resolve().parents[1] / "web"
_HTML_WITH_SCRIPT = sorted(
    p for p in _WEB_DIR.glob("*.html") if "<script" in p.read_text(encoding="utf-8")
)


@pytest.mark.skipif(shutil.which("node") is None, reason="node 미설치 — JS 문법 검사 생략")
@pytest.mark.parametrize("html", _HTML_WITH_SCRIPT, ids=lambda p: p.name)
def test_inline_scripts_parse(html: Path, tmp_path: Path):
    """web/*.html 의 인라인 <script> 가 문법 오류 없이 파싱된다.

    배경: dashboard.html 의 refresh() 안에 같은 이름의 const 가 두 번 선언돼
    SyntaxError 로 스크립트 전체가 죽었고, 계기판은 초기값(`…`/`—`) 그대로 멈춘 채
    "집계가 안 된다"고 보였다. catch 블록조차 실행되지 않는 **표현 계층의 무증상 고장**
    이라 200 응답·문자열 포함 검사로는 잡히지 않는다 — node --check 로 계약을 고정한다.
    """
    text = html.read_text(encoding="utf-8")
    blocks = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", text, re.S)
    assert blocks, f"{html.name}: 인라인 <script> 블록을 찾지 못했다"
    for i, js in enumerate(blocks):
        f = tmp_path / f"{html.stem}_{i}.js"
        f.write_text(js, encoding="utf-8")
        r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
        assert r.returncode == 0, f"{html.name} script#{i}: {r.stderr.strip()}"
