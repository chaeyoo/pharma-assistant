"""LLM 모드 tool-use 루프의 계약 테스트 — 스텁 LLM 으로 실제 루프를 태운다.

LLM 모드의 안전 배선(신뢰 소스 수집·이력 허용 집합·검증 게이트 통과)은
오프라인 모드와 코드 경로가 다른데, 실 API 키 없이는 실행 자체가 안 되어
테스트 커버리지가 0이었다 — '두 모드는 같은 계약'이라는 문서 주장이 한쪽
모드에서는 실행으로 확인된 적이 없던 셈이다. anthropic 클라이언트만 스텁으로
바꾸고 MCP·RAG·검증 게이트는 전부 실물을 태워, LLM 모드 경로의 계약을
실행 가능한 형태로 고정한다.
"""
from __future__ import annotations

import sys
import types

import pytest

from src import config
from src.agent.agent import RaAgent


class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _stub_anthropic(monkeypatch, responses: list, calls: list | None = None) -> None:
    """messages.create 가 준비된 응답을 순서대로 돌려주는 가짜 anthropic 모듈.

    calls 리스트를 넘기면 create 호출의 kwargs 를 순서대로 담아 준다 —
    모델에게 '실제로 어떤 도구 목록이 노출됐는가'를 검증하는 용도.
    """
    state = {"i": 0}

    class _Messages:
        async def create(self, **kwargs):
            if calls is not None:
                calls.append(kwargs)
            r = responses[state["i"]]
            state["i"] += 1
            return r

    fake = types.ModuleType("anthropic")
    fake.AsyncAnthropic = lambda api_key=None: types.SimpleNamespace(messages=_Messages())
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.setattr(config, "LLM_AVAILABLE", True)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-key")


@pytest.mark.asyncio
async def test_llm_history_search_allows_only_returned_superseded_docs(monkeypatch):
    """as_of(이력) 검색이 '실제로 반환한' 폐지본 인용은 경고에서 면제된다 —
    허용이 응답 전역 스위치가 아니라 문서 단위 집합으로 작동하는지의 배선 확인."""
    _stub_anthropic(monkeypatch, [
        types.SimpleNamespace(stop_reason="tool_use", content=[
            _Block(type="tool_use", id="t1", name="search_regulations",
                   input={"query": "중대한 이상사례 보고 기한", "as_of": "2025-01-01"}),
        ]),
        types.SimpleNamespace(stop_reason="end_turn", content=[
            # 기준일(as_of)을 답변에 재서술하지 않는 형태 — as_of 는 사용자 입력
            # 에코라 신뢰 소스에서 제거되므로, 재서술하면 미확인 날짜 경고가
            # 붙는 것이 의도된 보수 동작이다(사용자 지정 기준일은 근거가 아니다).
            _Block(type="text",
                   text="그 시점 당시 현행이던 구판 기준으로 보고 기한은 30일 이내였습니다."),
        ]),
    ])
    r = await RaAgent().chat("2025년 1월 시점의 이상사례 보고 기한은?")
    assert r.mode == "llm"
    assert any(c.get("doc_id") == "REG-013" for c in r.citations), r.citations
    # 이력 검색이 반환한 문서의 폐지본 인용 — 경고 면제, 수치(30일)도 구판 근거로 지원
    assert r.verification["superseded_cited"] == []
    assert r.verification["ok"], r.verification


@pytest.mark.asyncio
async def test_llm_case_echo_support_is_labeled(monkeypatch):
    """모델이 케이스 서술의 수치를 재서술하면 — 지지 근거가 도구 출력의 케이스
    에코뿐이므로 from_case(case_origin) 라벨이 붙는다. 규정·도구 계산에서 온
    수치(15일)는 라벨이 붙지 않는다: '사용자 서술 근거'와 '규정 근거'의 구분."""
    _stub_anthropic(monkeypatch, [
        types.SimpleNamespace(stop_reason="tool_use", content=[
            _Block(type="tool_use", id="t1", name="assess_adverse_event",
                   input={"case_description": "환자가 A정을 30일간 복용 후 두드러기로 입원했습니다"}),
        ]),
        types.SimpleNamespace(stop_reason="end_turn", content=[
            _Block(type="text",
                   text="복용 기간 30일의 입원 케이스로 중대에 해당하며, 보고 기한은 15일 이내입니다."),
        ]),
    ])
    r = await RaAgent().chat("이 케이스 언제까지 보고해야 하나요?")
    assert r.mode == "llm"
    assert r.verification["ok"], r.verification          # 경고는 없다
    assert r.verification["case_origin"] == ["30일"]     # 케이스 유래 지지 라벨
    assert "15일" not in r.verification["case_origin"]   # 규정(REG-005) 근거 지지


@pytest.mark.asyncio
async def test_llm_failed_as_of_does_not_open_allowance(monkeypatch):
    """형식이 틀린 as_of 호출(에러 계약)은 허용 집합을 채우지 못한다 — 시점
    검색이 성공한 적 없는데 폐지본 경고만 꺼지는 경로의 차단. 에러 계약 응답이
    신뢰 소스로 승격되지 않는 것도 함께 고정한다(에러 문구 속 사용자 입력 에코)."""
    _stub_anthropic(monkeypatch, [
        types.SimpleNamespace(stop_reason="tool_use", content=[
            _Block(type="tool_use", id="t1", name="search_regulations",
                   input={"query": "이상사례 보고 기한", "as_of": "2025/01/01"}),  # 형식 오류
        ]),
        types.SimpleNamespace(stop_reason="end_turn", content=[
            _Block(type="text", text="보고 기한 규정을 확인하지 못했습니다."),
        ]),
    ])
    r = await RaAgent().chat("2025년 1월 시점의 이상사례 보고 기한은?")
    assert r.mode == "llm"
    assert r.citations == []          # 성공한 검색이 없다
    assert r.verification["ok"]       # 수치 클레임 없는 안내 — 자명 통과
    # 에러 계약이 신뢰 소스로 승격됐다면 '2025' 같은 에코 값이 지원 근거가 됐을 것
    assert all(not c["supported"] for c in r.verification["checks"]) or not r.verification["checks"]
    # 에러 계약 응답만 있는 턴 — 신뢰 소스가 비어 grounded=False (근거 보증 아님)
    assert r.grounded is False


@pytest.mark.asyncio
async def test_llm_no_tool_answer_is_not_grounded(monkeypatch):
    """모델이 도구를 한 번도 부르지 않고 답하면 grounded=False — 이전에는
    dataclass 기본값 True 가 그대로 노출되어, 출처 0건 답변이 '근거로 뒷받침됨'
    배지를 달고 나갔다(LLM 모드에는 문턱 기반 abstention 이 없으므로 grounded 는
    '성공한 도구 출력이 신뢰 소스로 확보되었는가'의 신호여야 한다 — v7 발견)."""
    _stub_anthropic(monkeypatch, [
        types.SimpleNamespace(stop_reason="end_turn", content=[
            _Block(type="text", text="일반적으로 신속히 보고하는 것이 좋습니다."),
        ]),
    ])
    r = await RaAgent().chat("이상사례는 어떻게 보고하나요?")
    assert r.mode == "llm"
    assert r.citations == [] and r.tool_calls == []
    assert r.grounded is False


@pytest.mark.asyncio
async def test_llm_tool_grounded_answer_is_grounded(monkeypatch):
    """성공한 도구 출력이 신뢰 소스로 확보된 답변은 grounded=True — 위 테스트의
    반대 방향(도구 근거 답변까지 False 로 뒤집는 과보수 회귀 방지)."""
    _stub_anthropic(monkeypatch, [
        types.SimpleNamespace(stop_reason="tool_use", content=[
            _Block(type="tool_use", id="t1", name="search_regulations",
                   input={"query": "중대한 이상사례 보고 기한"}),
        ]),
        types.SimpleNamespace(stop_reason="end_turn", content=[
            _Block(type="text", text="중대한 이상사례는 15일 이내에 보고합니다."),
        ]),
    ])
    r = await RaAgent().chat("중대한 이상사례 보고 기한은?")
    assert r.mode == "llm"
    assert r.grounded is True and r.citations


@pytest.mark.asyncio
async def test_llm_empty_search_result_is_not_grounded(monkeypatch):
    """도구를 호출해 '성공했으나 결과 0건'인 검색은 grounded=True 로 만들지
    않는다 — citations=[] 인데 grounded=True 인 모순 차단. (v10)

    빈 검색 결과({"results": []}, as_of 가 전 문서 이전)는 에러도 에러 계약도
    아니라 신뢰 소스에 stringify 되어 쌓인다 — grounded 를 bool(trusted_texts)
    로 내면 v7 이 봉합한 '출처 0건+근거 배지'가 '도구 호출·성공하나 결과 0건'
    경로로 재발한다(오프라인은 빈 검색에 grounded=False 로 강등 — 그 대칭)."""
    _stub_anthropic(monkeypatch, [
        types.SimpleNamespace(stop_reason="tool_use", content=[
            _Block(type="tool_use", id="t1", name="search_regulations",
                   input={"query": "중대한 이상사례 보고 기한", "as_of": "1900-01-01"}),
        ]),
        types.SimpleNamespace(stop_reason="end_turn", content=[
            _Block(type="text", text="해당 시점에는 확인되는 규정이 없습니다."),
        ]),
    ])
    r = await RaAgent().chat("1900년 시점의 이상사례 보고 기한은?")
    assert r.mode == "llm"
    assert r.citations == []          # 빈 검색 — 출처 0건
    assert r.grounded is False        # 출처 0건이면 근거 보증 배지도 없다


_DDG_HTML = """
<div class="result">
  <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fnews%2F1&amp;rut=abc">MFDS 이상사례 보고 고시 개정</a>
  <a class="result__snippet" href="#">개정 고시는 2026-05-01 시행 예정이다.</a>
</div>
"""


@pytest.mark.asyncio
async def test_llm_web_tool_hidden_without_explicit_request(monkeypatch):
    """명시 요청이 없는 턴에는 search_web 이 도구 목록에서 아예 빠진다 —
    프롬프트 지시(어겨질 수 있는 규칙)가 아니라 도구 가용성으로 강제하는
    '명시' 쪽 절반의 배선 확인."""
    calls: list = []
    _stub_anthropic(monkeypatch, [
        types.SimpleNamespace(stop_reason="end_turn", content=[
            _Block(type="text", text="사내 규정 기준으로 답변드립니다."),
        ]),
    ], calls)
    await RaAgent().chat("중대한 이상사례 보고 기한은?")
    tool_names = [t["name"] for t in calls[0]["tools"]]
    assert "search_web" not in tool_names
    assert "search_regulations" in tool_names


@pytest.mark.asyncio
async def test_llm_web_results_are_separated_when_requested(monkeypatch):
    """명시 요청 턴 — search_web 이 노출·호출되고, 결과는 사내 근거와 분리된다:
    web_results 별도 필드, citations 비오염, grounded=False(사내 근거 아님),
    웹 유래 수치는 미확인 경고 대신 web_origin 라벨."""
    import src.websearch as websearch

    monkeypatch.setattr(websearch, "_fetch_html", lambda q: _DDG_HTML)
    calls: list = []
    _stub_anthropic(monkeypatch, [
        types.SimpleNamespace(stop_reason="tool_use", content=[
            _Block(type="tool_use", id="t1", name="search_web",
                   input={"query": "이상사례 보고 고시 개정 뉴스"}),
        ]),
        types.SimpleNamespace(stop_reason="end_turn", content=[
            _Block(type="text", text=(
                "🌐 인터넷 검색 결과에 따르면 개정 고시는 2026-05-01 시행 예정입니다 "
                "(출처: https://example.com/news/1 — 사내 규정 근거 아님)."
            )),
        ]),
    ], calls)
    r = await RaAgent().chat("이상사례 보고 고시 개정 소식 인터넷에서 검색해줘")
    assert "search_web" in [t["name"] for t in calls[0]["tools"]]
    assert r.web_results and r.web_results[0]["origin"] == "web"
    assert r.citations == []                     # 사내 출처 카드 비오염
    assert r.grounded is False                   # 웹 결과는 사내 근거 배지 대상이 아니다
    assert r.verification["ok"], r.verification  # 웹 재서술 날짜에 미확인 오탐이 없다
    assert "2026-05-01" in r.verification["web_origin"]


@pytest.mark.asyncio
async def test_llm_max_tokens_is_configured_not_hardcoded(monkeypatch):
    """생성 상한이 config.LLM_MAX_TOKENS 로 배선된다 — 1024 하드코딩은 2구역
    (사내 근거 + 🌐 웹) 답변이 중간에 잘리는 값이었다(실배포 실측)."""
    from src import config

    calls: list = []
    _stub_anthropic(monkeypatch, [
        types.SimpleNamespace(stop_reason="end_turn", content=[
            _Block(type="text", text="사내 규정 기준으로 답변드립니다."),
        ]),
    ], calls)
    await RaAgent().chat("중대한 이상사례 보고 기한은?")
    assert calls[0]["max_tokens"] == config.LLM_MAX_TOKENS >= 4096


@pytest.mark.asyncio
async def test_llm_truncated_answer_is_marked_not_silent(monkeypatch):
    """stop_reason=max_tokens 로 잘린 답변은 조용히 나가지 않는다 — 문장이
    중간에 끊겨도 완결된 답처럼 읽히므로("…필요합" 실측) 절단 표시를 붙인다."""
    _stub_anthropic(monkeypatch, [
        types.SimpleNamespace(stop_reason="max_tokens", content=[
            _Block(type="text", text="보고 기한 규정 확인이 필요합"),
        ]),
    ])
    r = await RaAgent().chat("이상사례 보고 절차를 아주 자세히 설명해줘")
    assert "길이 제한" in r.answer and "잘렸습니다" in r.answer


@pytest.mark.asyncio
async def test_llm_web_marker_backstop_when_model_omits_it(monkeypatch):
    """웹 결과가 사용된 턴인데 답변 본문에 🌐 표시가 없으면(프롬프트 미준수·
    길이 절단으로 웹 구역 소실) 분리 고지를 결정론적으로 덧붙인다 — '분리'가
    모델의 협조에만 의존하지 않게 하는 백스톱."""
    import src.websearch as websearch

    monkeypatch.setattr(websearch, "_fetch_html", lambda q: _DDG_HTML)
    _stub_anthropic(monkeypatch, [
        types.SimpleNamespace(stop_reason="tool_use", content=[
            _Block(type="tool_use", id="t1", name="search_web",
                   input={"query": "이상사례 보고 고시 개정 뉴스"}),
        ]),
        types.SimpleNamespace(stop_reason="end_turn", content=[
            _Block(type="text", text="개정 고시는 2026-05-01 시행 예정입니다."),  # 🌐 누락
        ]),
    ])
    r = await RaAgent().chat("이상사례 보고 고시 개정 소식 인터넷에서 검색해줘")
    assert "🌐" in r.answer and "인터넷 검색 결과가 사용되었습니다" in r.answer


@pytest.mark.asyncio
async def test_llm_api_failure_is_explicit_not_500(monkeypatch):
    """LLM API 호출 실패(잘못된 키·네트워크·모델명)는 예외 전파(HTTP 500)가
    아니라 명시적 안내 답변이 된다 — 가장 흔한 온보딩 실패 경로의 시끄럽고
    '원인을 말해주는' 실패. 안내문에는 예외 타입만 싣고 예외 메시지는 싣지
    않는다(외부 라이브러리 에러 문구의 요청 정보 에코 차단)."""
    class _Boom:
        async def create(self, **kwargs):
            raise RuntimeError("secret-request-detail")

    fake = types.ModuleType("anthropic")
    fake.AsyncAnthropic = lambda api_key=None: types.SimpleNamespace(messages=_Boom())
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.setattr(config, "LLM_AVAILABLE", True)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "bad-key")

    r = await RaAgent().chat("중대한 이상사례 보고 기한은?")
    assert r.mode == "llm"
    assert "LLM API 호출에 실패" in r.answer and "RuntimeError" in r.answer
    assert "secret-request-detail" not in r.answer   # 예외 메시지 비에코
    assert r.grounded is False
    assert r.verification  # 실패 안내도 게이트를 지난다(모든 응답이 게이트 통과)
