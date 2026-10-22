"""LLM 호출 게이트웨이 — 비교 실험이 모델을 부르는 유일한 통로.

실험의 공정성은 '모든 모델에 같은 요청'에서 나오므로, 요청을 만드는 코드를 한 곳에
모으고 실제 보낸 파라미터(request_params)를 결과에 그대로 기록한다. 샘플링 파라미터
(temperature·thinking·effort)는 **보내지 않는다** — Sonnet 5.5/Opus 5.5 는 비기본
temperature 를 거부하고 Opus 5.5 는 thinking 을 끌 수 없어, 세 모델에 '같은 값'을
보내는 유일한 방법이 '아무것도 보내지 않고 각 모델의 API 기본값을 쓰는 것'이다.

테스트는 같은 Protocol 을 만족하는 FakeGateway 를 주입한다(tests/test_model_compare.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Protocol

SAMPLING_NOTE = "api_default (temperature/thinking/effort not sent)"


@dataclass
class Completion:
    """한 번의 모델 호출 결과 — 성공·실패를 같은 형태로 돌려준다(error 로 구분)."""
    text: str = ""
    tool_uses: list[dict] = field(default_factory=list)  # [{"id","name","input"}]
    stop_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_ms: float = 0.0
    request_params: dict = field(default_factory=dict)
    error: str | None = None  # 예외 타입명만(메시지 원문 비에코 — agent 의 규율과 동일)


class LLMGateway(Protocol):
    def complete(
        self, *, model: str, system: str, messages: list[dict], max_tokens: int,
        tools: list[dict] | None = None,
    ) -> Completion: ...


def request_params(model: str, system: str, max_tokens: int, tools: list[dict] | None) -> dict:
    """결과 JSONL 에 남길 '실제 보낸 것' — 프롬프트 원문 대신 길이·도구명만 기록."""
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system_chars": len(system),
        "tools": [t["name"] for t in (tools or [])],
        "sampling": SAMPLING_NOTE,
    }


class AnthropicGateway:
    """anthropic SDK 동기 클라이언트 래퍼. timeout·max_retries 는 설정에서 온다."""

    def __init__(self, api_key: str, timeout_s: float, max_retries: int) -> None:
        import anthropic  # 지연 import — 키 없는 환경·테스트 스텁 배려

        self._client = anthropic.Anthropic(
            api_key=api_key, timeout=timeout_s, max_retries=max_retries
        )

    def complete(
        self, *, model: str, system: str, messages: list[dict], max_tokens: int,
        tools: list[dict] | None = None,
    ) -> Completion:
        params = request_params(model, system, max_tokens, tools)
        kwargs: dict = {
            "model": model, "max_tokens": max_tokens, "system": system, "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        t0 = perf_counter()
        try:
            resp = self._client.messages.create(**kwargs)
        except Exception as e:  # noqa: BLE001 - 실패도 '결과'로 기록한다(실패율 지표)
            return Completion(
                latency_ms=(perf_counter() - t0) * 1000, request_params=params,
                error=type(e).__name__,
            )
        latency = (perf_counter() - t0) * 1000
        text = "".join(getattr(b, "text", "") for b in resp.content if b.type == "text")
        tool_uses = [
            {"id": b.id, "name": b.name, "input": dict(b.input or {})}
            for b in resp.content if b.type == "tool_use"
        ]
        u = resp.usage
        return Completion(
            text=text, tool_uses=tool_uses, stop_reason=str(resp.stop_reason or ""),
            input_tokens=int(u.input_tokens or 0), output_tokens=int(u.output_tokens or 0),
            cache_read_tokens=int(getattr(u, "cache_read_input_tokens", 0) or 0),
            cache_write_tokens=int(getattr(u, "cache_creation_input_tokens", 0) or 0),
            latency_ms=latency, request_params=params,
        )
