"""Google 로그인 + 사용자별 일일 호출 제한 계약 테스트 (src/api/auth.py).

고정하는 것:
- 로그인 꺼짐(GOOGLE_CLIENT_ID 없음): 기존 동작 그대로 — 익명 통과, 제한 없음
- 로그인 켜짐: 보호 경로는 비로그인 시 401/리다이렉트, /health 는 공개 유지
- 콜백: 이메일 미검증·허용 목록 밖 계정은 세션을 만들지 않는다
- 호출 제한: 한도까지 통과, 초과 시 429(차감 없음), 사용자별로 분리
- preflight: 공개 가드·시크릿 누락·약한 세션 키를 기동 전에 잡는다

실제 Google 과의 통신은 하지 않는다 — OAuth 클라이언트의 토큰 교환만 가짜로 바꾼다.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src import config, preflight
from src.api import auth
from src.api.main import app


@pytest.fixture
def login_on(monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_CLIENT_ID", "test-client.apps.googleusercontent.com")
    monkeypatch.setattr(config, "GOOGLE_CLIENT_SECRET", "test-secret")
    monkeypatch.setattr(config, "SESSION_SECRET", "x" * 48)
    monkeypatch.setattr(config, "ALLOWED_EMAILS", set())
    monkeypatch.setattr(config, "ALLOWED_DOMAINS", set())
    monkeypatch.setattr(config, "DAILY_CHAT_LIMIT", 2)
    auth.usage.reset()
    yield
    auth.usage.reset()


def _fake_google(monkeypatch, userinfo: dict):
    class _Fake:
        async def authorize_access_token(self, request):
            return {"userinfo": userinfo}

    monkeypatch.setattr(auth, "_google", lambda: _Fake())


def _login(client, monkeypatch, email="tester@gmail.com", verified=True, sub="u-1"):
    _fake_google(monkeypatch, {"sub": sub, "email": email, "email_verified": verified, "name": "테스터"})
    return client.get("/auth/callback", follow_redirects=False)


# ---------------------------------------------------------------- 로그인 꺼짐
def test_login_off_keeps_anonymous_access():
    with TestClient(app) as c:
        assert c.get("/", follow_redirects=False).status_code == 200
        me = c.get("/api/me").json()
        assert me["authenticated"] and not me["auth_enabled"] and me["remaining"] is None
        r = c.post("/chat", json={"message": "중대한 이상사례는 며칠 안에 보고?"})
        assert r.status_code == 200 and r.json()["remaining_today"] is None
        # 로그인 화면은 의미가 없으므로 홈으로 보낸다
        assert c.get("/login", follow_redirects=False).headers["location"] == "/"


# ---------------------------------------------------------------- 로그인 켜짐
def test_protected_routes_require_login(login_on):
    with TestClient(app) as c:
        assert c.get("/", follow_redirects=False).headers["location"] == "/login"
        assert c.get("/dashboard", follow_redirects=False).headers["location"] == "/login"
        assert c.post("/chat", json={"message": "안녕"}).status_code == 401
        assert c.get("/api/deadlines").status_code == 401
        assert c.get("/api/me").status_code == 401
        # 헬스체크·로그인 화면은 공개
        assert c.get("/health").status_code == 200
        r = c.get("/login")
        assert r.status_code == 200 and "/auth/google" in r.text


def test_callback_creates_session_and_logout_clears_it(login_on, monkeypatch):
    with TestClient(app) as c:
        r = _login(c, monkeypatch)
        assert r.status_code in (302, 307) and r.headers["location"] == "/"
        me = c.get("/api/me").json()
        assert me["email"] == "tester@gmail.com" and me["daily_limit"] == 2 and me["remaining"] == 2
        assert c.get("/", follow_redirects=False).status_code == 200
        c.get("/logout", follow_redirects=False)
        assert c.get("/api/me").status_code == 401


def test_unverified_email_is_rejected(login_on, monkeypatch):
    with TestClient(app) as c:
        r = _login(c, monkeypatch, verified=False)
        assert r.headers["location"] == "/login?error=denied"
        assert c.get("/api/me").status_code == 401


def test_allowlist_by_email_and_domain(login_on, monkeypatch):
    monkeypatch.setattr(config, "ALLOWED_EMAILS", {"invited@gmail.com"})
    monkeypatch.setattr(config, "ALLOWED_DOMAINS", {"pharma.example"})
    assert auth.is_allowed("Invited@Gmail.com")
    assert auth.is_allowed("kim@pharma.example")
    assert not auth.is_allowed("kim@evil-pharma.example.com")
    assert not auth.is_allowed("stranger@gmail.com")
    with TestClient(app) as c:
        assert _login(c, monkeypatch, email="stranger@gmail.com").headers["location"] == "/login?error=denied"
        assert _login(c, monkeypatch, email="kim@pharma.example").headers["location"] == "/"


def test_token_exchange_failure_returns_to_login(login_on, monkeypatch):
    class _Broken:
        async def authorize_access_token(self, request):
            raise RuntimeError("state mismatch")

    monkeypatch.setattr(auth, "_google", lambda: _Broken())
    with TestClient(app) as c:
        r = c.get("/auth/callback", follow_redirects=False)
        assert r.headers["location"] == "/login?error=failed"


# ---------------------------------------------------------------- 호출 제한
def test_daily_limit_blocks_after_quota(login_on, monkeypatch):
    with TestClient(app) as c:
        _login(c, monkeypatch)
        q = {"message": "중대한 이상사례는 며칠 안에 보고?"}
        r1, r2 = c.post("/chat", json=q), c.post("/chat", json=q)
        assert r1.json()["remaining_today"] == 1 and r2.json()["remaining_today"] == 0
        r3 = c.post("/chat", json=q)
        assert r3.status_code == 429 and "2회" in r3.json()["detail"]
        assert c.get("/api/me").json()["remaining"] == 0  # 초과 요청은 차감하지 않는다


def test_limit_is_per_user(login_on):
    lim = auth.UsageLimiter()
    assert lim.check_and_consume("a", 1) and not lim.check_and_consume("a", 1)
    assert lim.check_and_consume("b", 1)  # 다른 사용자는 별도 카운트
    assert lim.check_and_consume("a", 0) and lim.remaining("a", 0) is None  # 0 = 무제한


# ---------------------------------------------------------------- preflight
def _config_problems():
    return preflight.check_config()


def test_preflight_require_login_without_client(monkeypatch):
    monkeypatch.setattr(config, "REQUIRE_LOGIN", True)
    monkeypatch.setattr(config, "GOOGLE_CLIENT_ID", "")
    assert any("REQUIRE_LOGIN" in p for p in _config_problems())


def test_preflight_weak_session_secret(login_on, monkeypatch):
    monkeypatch.setattr(config, "SESSION_SECRET", "short")
    assert any("SESSION_SECRET" in p for p in _config_problems())
    monkeypatch.setattr(config, "SESSION_SECRET", "x" * 48)
    monkeypatch.setattr(config, "GOOGLE_CLIENT_SECRET", "")
    assert any("GOOGLE_CLIENT_SECRET" in p for p in _config_problems())


def test_preflight_passes_with_full_login_config(login_on):
    assert not [p for p in _config_problems() if "GOOGLE" in p or "SESSION" in p or "REQUIRE_LOGIN" in p]


def test_login_starts_on_canonical_host(login_on, monkeypatch):
    """Cloud Run 의 두 번째 주소로 들어와도 콜백과 같은 호스트에서 로그인을 시작한다
    (다른 호스트면 OAuth state 쿠키가 안 따라와 state 불일치로 실패하던 경로)."""
    monkeypatch.setattr(config, "PUBLIC_BASE_URL", "https://pharma-api-abc-uc.a.run.app")
    with TestClient(app, base_url="https://pharma-api-123.us-central1.run.app") as c:
        r = c.get("/auth/google", follow_redirects=False)
        assert r.headers["location"] == "https://pharma-api-abc-uc.a.run.app/auth/google"
        r = c.get("/login?error=failed", follow_redirects=False)
        assert r.headers["location"] == "https://pharma-api-abc-uc.a.run.app/login?error=failed"
    with TestClient(app, base_url="https://pharma-api-abc-uc.a.run.app") as c:
        assert c.get("/login").status_code == 200   # 이미 공개 주소면 그대로
