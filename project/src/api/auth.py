"""가벼운 Google 로그인(OIDC) + 사용자별 일일 호출 제한.

왜 이 계층이 필요한가
---------------------
LLM 모드를 공개 URL 에서 켜면 방문자가 API 비용을 쓴다. 그래서 종전 배포는
LLM 모드를 꺼둔 오프라인 데모였다. 실사용자에게 LLM 기능을 제공하려면
(1) 누가 쓰는지 알아야 하고(인증), (2) 한 사람이 쓸 수 있는 양에 상한이
있어야 한다(호출 제한). 이 모듈이 그 두 가지를 맡는다.

구조 — 인증서버를 직접 만들지 않는다
------------------------------------
Google 이 IdP(인증서버) 역할을 하고, 이 앱은 OIDC 클라이언트(relying party)다.
  /login → /auth/google(Google 로 리다이렉트) → /auth/callback(ID 토큰 검증,
  이메일 확인, 허용 목록 대조) → 서명된 세션 쿠키 발급
사내 SSO 로 옮길 때는 server_metadata_url 과 클라이언트 자격만 사내 IdP
(Okta·Azure AD·Keycloak 등)로 바꾸면 같은 흐름이 그대로 동작한다.

켜고 끄기
---------
GOOGLE_CLIENT_ID 가 설정돼 있을 때만 로그인이 켜진다. 비어 있으면 로컬 개발·
테스트처럼 익명 사용자로 통과한다(기존 동작 유지). 공개 배포에는
REQUIRE_LOGIN=1 을 함께 두어, 설정이 실수로 빠져도 무인증 공개 상태로
조용히 뜨지 않게 한다(preflight 가 기동을 거부 — MCP_REQUIRE_AUTH 와 같은 계열).

호출 제한 저장소
----------------
UsageLimiter 는 인터페이스를 작게 두고(check_and_consume / remaining) 현재는
프로세스 메모리에 센다. 인스턴스가 여러 대거나 재시작되면 카운트가 갈라지거나
초기화되므로, 운영 규모가 커지면 같은 인터페이스로 Firestore·Redis 구현을
끼우는 것이 확장 지점이다.
"""
from __future__ import annotations

import datetime as _dt
import threading
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from .. import config

_KST = ZoneInfo("Asia/Seoul")
_GOOGLE_METADATA = "https://accounts.google.com/.well-known/openid-configuration"

ANONYMOUS = {"sub": "anonymous", "email": "local", "name": "로컬 사용자", "anonymous": True}


def auth_enabled() -> bool:
    """요청 시점에 config 를 읽는다 — 테스트가 config 속성을 바꿔 켤 수 있게."""
    return bool(config.GOOGLE_CLIENT_ID)


# --------------------------------------------------------------------------
# 허용 목록
# --------------------------------------------------------------------------
def is_allowed(email: str) -> bool:
    """ALLOWED_EMAILS / ALLOWED_DOMAINS 둘 다 비면 Google 계정이면 누구나 허용.

    하나라도 설정돼 있으면 그 목록에 맞는 계정만 들어온다(테스터 초대 운영).
    """
    email = (email or "").strip().lower()
    emails, domains = config.ALLOWED_EMAILS, config.ALLOWED_DOMAINS
    if not emails and not domains:
        return bool(email)
    if email in emails:
        return True
    return any(email.endswith("@" + d) for d in domains)


# --------------------------------------------------------------------------
# 사용자별 일일 호출 제한
# --------------------------------------------------------------------------
@dataclass
class UsageLimiter:
    """(사용자, KST 날짜) 단위 카운터. limit <= 0 이면 무제한."""

    _counts: dict[tuple[str, str], int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @staticmethod
    def _today() -> str:
        return _dt.datetime.now(_KST).date().isoformat()

    def remaining(self, user_key: str, limit: int) -> int | None:
        if limit <= 0:
            return None
        with self._lock:
            return max(0, limit - self._counts.get((user_key, self._today()), 0))

    def check_and_consume(self, user_key: str, limit: int) -> bool:
        """한도 안이면 1 차감하고 True, 초과면 차감 없이 False."""
        if limit <= 0:
            return True
        key = (user_key, self._today())
        with self._lock:
            used = self._counts.get(key, 0)
            if used >= limit:
                return False
            self._counts[key] = used + 1
            # 지난 날짜 키 정리 — 메모리가 날짜 수만큼 자라지 않게
            today = key[1]
            for k in [k for k in self._counts if k[1] != today]:
                del self._counts[k]
            return True

    def reset(self) -> None:
        with self._lock:
            self._counts.clear()


usage = UsageLimiter()


# --------------------------------------------------------------------------
# 현재 사용자
# --------------------------------------------------------------------------
def current_user(request: Request) -> dict | None:
    if not auth_enabled():
        return dict(ANONYMOUS)
    return request.session.get("user")


def require_user(request: Request) -> dict:
    """FastAPI 의존성 — 로그인 안 했으면 401(UI 는 /login 으로 보낸다)."""
    user = current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
    return user


def consume_chat_quota(user: dict) -> int | None:
    """질문 1회 차감. 초과면 429. 남은 횟수(무제한이면 None)를 돌려준다.

    익명(로그인 꺼짐)일 때는 제한하지 않는다 — 로컬 개발·테스트 경로.
    """
    if user.get("anonymous"):
        return None
    limit = config.DAILY_CHAT_LIMIT
    if not usage.check_and_consume(user["sub"], limit):
        raise HTTPException(
            status_code=429,
            detail=f"오늘 사용 가능한 질문 {limit}회를 모두 썼습니다. 내일(KST 0시) 다시 이용해 주세요.",
        )
    return usage.remaining(user["sub"], limit)


# --------------------------------------------------------------------------
# OAuth 클라이언트 (첫 사용 시 등록 — 설정이 바뀌는 테스트에서도 안전하게)
# --------------------------------------------------------------------------
_oauth = None


def _google():
    global _oauth
    if _oauth is None:
        from authlib.integrations.starlette_client import OAuth

        oauth = OAuth()
        oauth.register(
            name="google",
            client_id=config.GOOGLE_CLIENT_ID,
            client_secret=config.GOOGLE_CLIENT_SECRET,
            server_metadata_url=_GOOGLE_METADATA,
            client_kwargs={"scope": "openid email profile"},
        )
        _oauth = oauth
    return _oauth.google


def _redirect_uri(request: Request) -> str:
    # Cloud Run 같은 프록시 뒤에서는 request.url 의 scheme 이 http 로 보일 수 있다.
    # Google 콘솔에 등록한 주소와 글자 하나까지 같아야 하므로 공개 주소를 명시할 수 있게 한다.
    if config.PUBLIC_BASE_URL:
        return config.PUBLIC_BASE_URL.rstrip("/") + "/auth/callback"
    return str(request.url_for("auth_callback"))


router = APIRouter()


@router.get("/login", response_class=HTMLResponse, include_in_schema=False)
async def login_page(request: Request):
    if not auth_enabled() or request.session.get("user"):
        return RedirectResponse("/")
    html = (config.WEB_DIR / "login.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@router.get("/auth/google", include_in_schema=False)
async def auth_google(request: Request):
    if not auth_enabled():
        return RedirectResponse("/")
    try:
        return await _google().authorize_redirect(request, _redirect_uri(request))
    except Exception:  # noqa: BLE001 - Google 메타데이터 조회 실패(네트워크 등)는 500 대신 안내
        return RedirectResponse("/login?error=failed")


@router.get("/auth/callback", name="auth_callback", include_in_schema=False)
async def auth_callback(request: Request):
    if not auth_enabled():
        return RedirectResponse("/")
    try:
        token = await _google().authorize_access_token(request)
    except Exception:  # noqa: BLE001 - state 불일치·취소 등은 로그인 화면으로 되돌린다
        return RedirectResponse("/login?error=failed")
    info = token.get("userinfo") or {}
    email = (info.get("email") or "").lower()
    if not info.get("email_verified") or not is_allowed(email):
        return RedirectResponse("/login?error=denied")
    request.session["user"] = {
        "sub": info.get("sub") or email,
        "email": email,
        "name": info.get("name") or email,
        "picture": info.get("picture", ""),
    }
    return RedirectResponse("/")


@router.get("/logout", include_in_schema=False)
async def logout(request: Request):
    request.session.pop("user", None)
    return RedirectResponse("/login" if auth_enabled() else "/")


@router.get("/api/me")
async def me(request: Request) -> JSONResponse:
    user = current_user(request)
    if not user:
        return JSONResponse({"authenticated": False, "auth_enabled": True}, status_code=401)
    limit = config.DAILY_CHAT_LIMIT
    return JSONResponse({
        "authenticated": True,
        "auth_enabled": auth_enabled(),
        "email": user.get("email"),
        "name": user.get("name"),
        "picture": user.get("picture", ""),
        "daily_limit": None if user.get("anonymous") or limit <= 0 else limit,
        "remaining": None if user.get("anonymous") else usage.remaining(user["sub"], limit),
    })
