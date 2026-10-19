"""LLM judge 2차 검증의 '결정론적 부분' 계약 테스트 — 스텁으로 배선을 핀 고정한다.

judge 의 판정 능력(확률)은 eval/judge_eval.py 실측과 preflight canary 의 몫이고,
여기서는 실행 없이도 참이어야 하는 계약만 고정한다: 기본 비활성(호출 0회),
비차단(flags 가 있어도 ok 불변), malformed·예외의 unavailable 흡수(크래시·에코
금지), 결정론 축 정본의 무오염(summary 에 judge 축이 새지 않음), 설정 fail-closed.
스텁 방식은 test_agent_llm 의 _stub_anthropic 과 동일 — llm_judge 의 lazy
`import anthropic` 이 이 테스트의 전제다(모듈 상단 import 로 바꾸면 깨진다).
"""
from __future__ import annotations

import json
import sys
import types

import pytest

from src import config
from src.agent.agent import RaAgent
from src.verify.llm_judge import judge_answer
from src.verify.verifier import verify_answer

_FLAGGED_JSON = json.dumps({
    "flags": [{"kind": "negation_flip", "claim": "보고 대상이 아닙니다",
               "reason": "근거는 보고 대상이라고 말한다"}]
}, ensure_ascii=False)


def _stub_anthropic(monkeypatch, reply: str | None = None,
                    error: Exception | None = None, calls: list | None = None) -> None:
    """동기 anthropic 클라이언트 스텁 — judge 의 lazy import 를 가로챈다."""

    class _Messages:
        def create(self, **kwargs):
            if calls is not None:
                calls.append(kwargs)
            if error is not None:
                raise error
            return types.SimpleNamespace(
                content=[types.SimpleNamespace(type="text", text=reply)]
            )

    fake = types.ModuleType("anthropic")
    fake.Anthropic = lambda api_key=None: types.SimpleNamespace(messages=_Messages())
    monkeypatch.setitem(sys.modules, "anthropic", fake)


@pytest.mark.asyncio
async def test_judge_disabled_by_default(monkeypatch):
    """기본(비활성) 경로 — judge 는 호출 자체가 없어야 한다(opt-in 계약)."""
    def _boom(api_key=None):
        raise AssertionError("JUDGE_ENABLED=False 인데 anthropic 클라이언트가 생성됐다")

    fake = types.ModuleType("anthropic")
    fake.Anthropic = _boom
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    r = await RaAgent().chat("중대한 이상사례 보고 기한은 며칠 이내인가요?")
    assert "judge" not in r.verification


@pytest.mark.asyncio
async def test_judge_flag_is_nonblocking_and_visible(monkeypatch):
    """flags 부착 시: 경고문은 답변에, judge 는 verification 서브딕셔너리에 —
    그러나 결정론 게이트의 ok 는 불변이다(비차단 계약)."""
    monkeypatch.setattr(config, "JUDGE_ENABLED", True)
    calls: list = []
    _stub_anthropic(monkeypatch, reply=_FLAGGED_JSON, calls=calls)
    r = await RaAgent().chat("중대한 이상사례 보고 기한은 며칠 이내인가요?")
    j = r.verification["judge"]
    assert j["status"] == "flagged" and j["flags"][0]["kind"] == "negation_flip"
    assert "⚠ LLM 2차 검증" in r.answer
    assert r.verification["ok"] is True, "judge flags 가 결정론 게이트 판정을 바꿨다(차단 금지 위반)"
    assert calls, "judge 호출이 기록되지 않았다"
    assert calls[0]["model"] == config.JUDGE_MODEL


@pytest.mark.asyncio
async def test_judge_clean_verdict_adds_no_warning(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_ENABLED", True)
    _stub_anthropic(monkeypatch, reply='{"flags": []}')
    r = await RaAgent().chat("중대한 이상사례 보고 기한은 며칠 이내인가요?")
    assert r.verification["judge"]["status"] == "ok"
    assert "⚠ LLM 2차 검증" not in r.answer


@pytest.mark.asyncio
async def test_judge_malformed_output_degrades_to_unavailable(monkeypatch):
    """malformed 출력은 통과(ok)로 위장하지 않고 unavailable 로 가시화 —
    답변 본문은 무손상이어야 한다(judge 고장이 응답을 죽이면 안 된다)."""
    monkeypatch.setattr(config, "JUDGE_ENABLED", True)
    _stub_anthropic(monkeypatch, reply="죄송합니다, 판정할 수 없습니다.")
    r = await RaAgent().chat("중대한 이상사례 보고 기한은 며칠 이내인가요?")
    j = r.verification["judge"]
    assert j["status"] == "unavailable" and j["reason"] == "MalformedJudgeOutput"
    assert "⚠ LLM 2차 검증" not in r.answer


@pytest.mark.asyncio
async def test_judge_api_error_is_absorbed_without_echo(monkeypatch):
    """API 예외는 타입명만 남긴다 — 메시지 에코 금지(비밀·내부 정보 유출 경로)."""
    monkeypatch.setattr(config, "JUDGE_ENABLED", True)
    _stub_anthropic(monkeypatch, error=RuntimeError("sk-secret-do-not-echo"))
    r = await RaAgent().chat("중대한 이상사례 보고 기한은 며칠 이내인가요?")
    j = r.verification["judge"]
    assert j["status"] == "unavailable" and j["reason"] == "RuntimeError"
    assert "sk-secret-do-not-echo" not in json.dumps(r.verification)
    assert "sk-secret-do-not-echo" not in r.answer


def test_judge_rejects_unknown_kind(monkeypatch):
    """계약 밖 kind 가 섞이면 전체를 unavailable 로 — 절반 파싱된 판정은 잡음이다."""
    _stub_anthropic(monkeypatch, reply='{"flags": [{"kind": "grammar", "claim": "x", "reason": "y"}]}')
    j = judge_answer("답변", ["근거"])
    assert j["status"] == "unavailable" and j["reason"] == "MalformedJudgeOutput"


def test_judge_accepts_fenced_json(monkeypatch):
    """코드펜스·전후 텍스트가 섞인 JSON 도 파싱한다(모델 출력의 현실 형태)."""
    _stub_anthropic(monkeypatch, reply=f"판정 결과입니다:\n```json\n{_FLAGGED_JSON}\n```")
    j = judge_answer("답변", ["근거"])
    assert j["status"] == "flagged"


def test_check_config_fails_closed_without_key(monkeypatch):
    """LLM_JUDGE=1 + 키 없음 = 고장난 안전장치 배포 — preflight 가 기동을 거부한다."""
    from src import preflight

    monkeypatch.setattr(config, "JUDGE_ENABLED", True)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    problems = preflight.check_config()
    assert any("LLM_JUDGE" in p for p in problems)


def test_judge_group_absent_when_disabled(monkeypatch):
    """비활성(기본)이면 preflight 에 judge 그룹 자체가 없다 — 오프라인 부팅 무영향."""
    from src import preflight

    monkeypatch.setattr(config, "JUDGE_ENABLED", False)
    assert "LLM judge 자가 테스트" not in preflight.run_preflight(role="api")


def test_verifier_summary_is_not_polluted_by_judge():
    """결정론 축 정본의 무오염 — judge 는 agent 계층 부착이지 verifier 축이 아니다.
    (여기가 깨지면 test_preflight 의 축 커버리지 메타테스트 전제가 무너진다.)"""
    assert "judge" not in verify_answer("보고 기한은 15일 이내입니다", ["보고 기한은 15일 이내"]).summary()
