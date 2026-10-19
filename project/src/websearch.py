"""인터넷(공개 웹) 검색 — MCP 도구 search_web 의 백엔드.

설계 원칙은 **명시와 분리**다:
  - 명시: 이 모듈을 부르는 유일한 경로(search_web 도구)는 사용자가 인터넷
    검색을 '명시적으로' 요청한 턴에만 에이전트에 노출·허용된다. 사내 문서에서
    근거를 못 찾은 질문을 조용히 웹 검색으로 대신 답하는 폴백 경로는 없다 —
    abstention(회피)은 종전 그대로 회피로 남는다.
  - 분리: 반환 계약이 결과 전체에 origin="web" 를 박아, 답변·출처 카드·검증
    계층 어디서든 사내 규제문서 근거와 다른 신뢰 계층임을 구분할 수 있게 한다.
    (검증기는 web_texts 3계층으로 받아 from_web 라벨을 붙인다 — 사내 근거로
    승격되지 않는다.)

백엔드는 DuckDuckGo HTML 엔드포인트를 파싱하는 최소 구현이다(외부 의존성 없이
httpx + 정규식). 실배포에서는 사내 승인된 검색 API(예: 검색 프록시·Bing API)로
_fetch_html 봉합선만 교체한다 — 테스트·preflight 도 같은 봉합선을 스텁으로 쓴다.

실패 방향: 네트워크 불가·차단·파싱 실패는 예외 전파가 아니라 명시적 에러
계약({"error", ...})으로 답한다 — 에이전트의 다른 도구들과 같은 규율이다
(조용한 빈 결과는 '관련 정보 없음'이라는 자신 있는 오답으로 소비된다).
"""
from __future__ import annotations

import html as _html
import re
import urllib.parse

from . import config

# 사용자·에이전트에게 항상 함께 전달되는 분리 고지 — 결과 dict 에 동봉된다.
WEB_NOTICE = (
    "인터넷 검색 결과 — 사내 규제문서 근거가 아닙니다. "
    "규제 판단에 사용하기 전 원문과 사내 규정을 반드시 확인하세요."
)

_SEARCH_URL = "https://html.duckduckgo.com/html/"

# DuckDuckGo HTML 결과 파싱 — 제목 앵커(result__a)와 스니펫(result__snippet).
_RESULT_A_RE = re.compile(
    r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
    re.S,
)
_SNIPPET_RE = re.compile(
    r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
    re.S,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _fetch_html(query: str) -> str:
    """검색 HTML 을 가져온다 — 테스트·preflight·실배포 교체가 이 봉합선을 쓴다."""
    import httpx

    r = httpx.get(
        _SEARCH_URL,
        params={"q": query},
        timeout=config.WEB_SEARCH_TIMEOUT,
        headers={"User-Agent": "RAPV-Assistant/1.0"},
        follow_redirects=True,
    )
    r.raise_for_status()
    return r.text


def _clean(fragment: str) -> str:
    """태그 제거 + HTML 엔티티 복원 + 공백 정규화."""
    return " ".join(_html.unescape(_TAG_RE.sub("", fragment)).split())


def _resolve_url(href: str) -> str:
    """DDG 의 리다이렉트 링크(//duckduckgo.com/l/?uddg=…)에서 원 URL 을 푼다."""
    href = _html.unescape(href)
    if "uddg=" in href:
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
        if qs.get("uddg"):
            return qs["uddg"][0]
    if href.startswith("//"):
        return "https:" + href
    return href


def parse_results(html_text: str, top_n: int) -> list[dict]:
    """검색 HTML → [{"title","url","snippet","origin":"web"}...] (최대 top_n)."""
    anchors = _RESULT_A_RE.findall(html_text)
    snippets = [_clean(s) for s in _SNIPPET_RE.findall(html_text)]
    out: list[dict] = []
    for i, (href, title_html) in enumerate(anchors[: max(1, min(top_n, 5))]):
        out.append(
            {
                "title": _clean(title_html),
                "url": _resolve_url(href),
                "snippet": snippets[i] if i < len(snippets) else "",
                # 결과 항목 단위에도 origin 을 박는다 — 항목이 목록에서 떼어져
                # 단독으로 소비돼도(출처 카드 등) 웹 유래임이 따라다닌다.
                "origin": "web",
            }
        )
    return out


def search(query: str, top_n: int = 3) -> dict:
    """인터넷 검색을 수행한다 — 항상 origin="web" 이 박힌 dict 를 반환.

    호출자(MCP 도구 계층)가 query 마스킹을 이미 마쳤다는 전제다(다른 도구와
    동일한 계층 분담). 여기서는 수행·파싱·에러 계약만 담당한다.
    """
    if not query.strip():
        return {"error": "query 가 비어 있음", "expected": "비어 있지 않은 검색어"}
    if not config.WEB_SEARCH_ENABLED:
        return {
            "error": "인터넷 검색이 비활성화되어 있음 (WEB_SEARCH=0)",
            "expected": "관리자가 WEB_SEARCH 를 켠 환경에서만 사용 가능",
        }
    try:
        html_text = _fetch_html(query)
        results = parse_results(html_text, top_n)
    except Exception as e:  # noqa: BLE001 - 외부 네트워크 실패는 에러 계약으로 흡수
        # 예외 타입명만 싣는다 — 외부 라이브러리 에러 문구에는 요청 URL(질의)이
        # 에코될 수 있다(LLM API 실패 안내와 같은 규율).
        return {
            "error": f"인터넷 검색 실패: {type(e).__name__}",
            "expected": "네트워크 연결과 외부 접근 정책(프록시·방화벽) 확인",
        }
    return {
        "query": query,
        "origin": "web",
        "notice": WEB_NOTICE,
        "results": results,
    }
