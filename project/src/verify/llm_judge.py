"""LLM judge 2차 검증 — 결정론 게이트가 스스로 선언한 경계 밖을 보는 계층.

1차 게이트(verifier.py)는 결정론이 닿는 축(수치 존재·방향·역할·버전)을 전부
결정론으로 가져갔고, 그 경계를 문서에 남겼다: **부정문 뒤집기·주어 바꿔치기·
수치 없는 의미 왜곡**은 닫힌 어휘 집합으로 환원되지 않아 규칙의 범위 밖이다.
이 모듈이 그 자리를 채운다 — 답변과 신뢰 소스를 LLM 에 주고 '의미 관계가
뒤집혔는가'만 판정하게 한다(액터-크리틱: 답을 만든 경로와 다른 모델이 비판).

설계 원칙 — 1차 게이트와의 관계가 이 모듈의 핵심이다:

  1. **결정론 축 정본(canon) 밖.** 판정을 VerificationResult 의 축으로 넣지
     않고 verification["judge"] 서브딕셔너리로 부착한다. 축 정본(WARNING_AXES)
     에 등록하면 preflight 자가 테스트가 그 축의 planted 케이스를 구조로
     강제하는데(메타테스트), judge 는 API 키 없이 실행이 불가능하다 — 키 없는
     오프라인 부팅이 구조적으로 깨진다. '자가 테스트를 강제하는 구조'는
     결정론 축의 규율이고, 확률 계층은 다른 규율(아래 3)을 따른다.
  2. **비차단·opt-in.** flags 가 있어도 verification["ok"] 는 불변이다(경고
     부착만 — 1차 게이트와 같은 실패 방향: 시끄럽되 차단하지 않는다). 기본
     비활성(LLM_JUDGE=1 opt-in)인 이유: 전 응답마다 도는 유료 API 콜이고,
     확률적 판정이라 같은 답변에 다른 판정이 나올 수 있다 — 켜는 것은 비용과
     비결정론을 받아들이는 명시적 결정이어야 한다.
  3. **검증의 검증은 이원화.** 확률 검증자는 '검증의 검증' 순환을 만든다
     (dictionary.md 의 self-critique 순환) — 이 데모의 답: (a) 배선·파싱
     계약(비활성 기본·malformed 흡수·비차단)은 결정론이므로 pytest 스텁이
     핀 고정하고(tests/test_judge.py), (b) 살아있는 판정 능력은 planted
     의미 변조 평가셋(eval/judge_eval.py — 키 필요 실측)과 enabled 부팅 시
     preflight canary 가 잰다. 생성 모델(LLM_MODEL)과 판정 모델(JUDGE_MODEL,
     기본 저비용 모델)을 분리하는 것도 같은 순환의 부분 완화다.
  4. **1차 게이트의 몫은 판정하지 않는다.** 수치 존재·방향·역할은 결정론
     축이 이미 재현 가능하게 잡는다 — judge 가 같은 축을 중복 판정하면
     두 계층의 판정이 어긋날 때 어느 쪽이 정본인지 모호해진다(프롬프트로
     명시 배제). judge 의 관할은 결정론이 못 보는 의미 관계뿐이다.
  5. **실패는 unavailable 로 가시화.** API 예외·malformed 출력은 절대
     크래시하지 않고 status="unavailable" 로 응답에 남긴다 — '2차 검증이
     안 돌았다'와 '돌았는데 이상 없음'(ok)을 구분하지 않으면 judge 고장이
     통과처럼 보인다(조용한 비검증 금지 — 1차 게이트와 같은 원칙). 예외는
     타입명만 기록한다(메시지 에코 금지 — agent 의 오류 비에코 규율).

확장 지점: _finalize 가 동기 함수라 동기 클라이언트를 쓴다(응답당 +1콜의
지연은 opt-in 의 수용된 비용). 처리량이 문제가 되면 세 백엔드의 _finalize
호출부를 async 로 올리는 것이 교체 지점이다.
"""
from __future__ import annotations

import json
import re

from .. import config

# judge 가 판정하는 왜곡 유형의 정본 — 평가셋(judge_eval)·테스트와 공유한다.
JUDGE_KINDS = ("negation_flip", "subject_swap", "meaning_inversion")

# 입력 절단 상한 — judge 는 요약이 아니라 왜곡 탐지라 앞부분 우선 절단으로
# 충분하다(신뢰 소스 뒷부분이 잘리면 못 보는 왜곡이 생길 수 있으나, 상한이
# 없으면 긴 컨텍스트 비용·지연이 응답마다 발생한다 — 상한을 명시된 한계로 둔다).
_MAX_ANSWER_CHARS = 4000
_MAX_TRUSTED_CHARS = 8000

_SYSTEM = (
    "당신은 제약 규제(RA·PV) 도메인 답변의 2차 검증자(critic)다. "
    "답변이 근거 텍스트의 의미 관계를 왜곡했는지만 판정한다.\n"
    "판정 대상 (이 세 가지만):\n"
    "- negation_flip: 근거의 긍정/부정 관계가 답변에서 뒤집힘 (예: 근거 '보고 대상이다' → 답변 '보고 대상이 아니다')\n"
    "- subject_swap: 두 대상의 속성·의무가 서로 뒤바뀜 (예: 근거 '허가는 심의 필요, 신고는 불요' → 답변이 반대로)\n"
    "- meaning_inversion: 수치 없이 의미가 반전된 그 밖의 왜곡\n"
    "판정하지 말 것: 수치·날짜의 존재/방향/역할 (1차 결정론 게이트의 몫이다), "
    "표현 차이·요약·생략 (왜곡이 아니다), 근거에 없는 주제 (별도 축이 있다).\n"
    "확신이 없으면 flag 하지 않는다 (오탐은 경고 피로를 낳아 검증 계층 전체를 죽인다).\n"
    "반드시 아래 JSON 형식으로만 답한다. 다른 텍스트 금지:\n"
    '{"flags": [{"kind": "<negation_flip|subject_swap|meaning_inversion>", '
    '"claim": "<답변 속 문제 구절>", "reason": "<근거와 어긋나는 이유 한 문장>"}]}\n'
    "왜곡이 없으면: {\"flags\": []}"
)


def _extract_json(text: str) -> dict | None:
    """모델 출력에서 JSON 객체를 꺼낸다 — 코드펜스·전후 텍스트를 흡수한다."""
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _valid_flags(data: dict) -> list[dict] | None:
    """flags 배열을 계약 형태로 정제 — 계약 위반 항목은 버리지 않고 전체를
    무효(None)로 본다: 절반만 파싱된 판정은 '판정'이 아니라 잡음이다."""
    flags = data.get("flags")
    if not isinstance(flags, list):
        return None
    out: list[dict] = []
    for f in flags:
        if not isinstance(f, dict) or f.get("kind") not in JUDGE_KINDS:
            return None
        out.append({
            "kind": f["kind"],
            "claim": str(f.get("claim", ""))[:200],
            "reason": str(f.get("reason", ""))[:300],
        })
    return out


def judge_answer(answer: str, trusted_texts: list[str], question: str = "") -> dict:
    """답변의 의미 왜곡을 LLM 으로 판정한다 — verification["judge"] 의 값.

    반환 스키마(항상 이 형태 — 호출자가 분기하지 않도록 status 로 수렴):
      {"status": "ok" | "flagged" | "unavailable", "model": str,
       "flags": [{"kind", "claim", "reason"}], "reason": str(불가 사유, 타입명만)}
    """
    import anthropic  # 지연 import — 테스트 스텁(sys.modules 교체)과 오프라인 환경 배려

    trusted = "\n---\n".join(trusted_texts)[:_MAX_TRUSTED_CHARS]
    user = (
        f"[근거 텍스트]\n{trusted}\n\n"
        + (f"[사용자 질문]\n{question[:1000]}\n\n" if question else "")
        + f"[검증할 답변]\n{answer[:_MAX_ANSWER_CHARS]}"
    )
    try:
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=config.JUDGE_MODEL,
            max_tokens=1024,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(getattr(b, "text", "") for b in resp.content)
    except Exception as e:  # noqa: BLE001 - judge 실패가 응답을 죽이면 안 된다
        return {"status": "unavailable", "model": config.JUDGE_MODEL,
                "flags": [], "reason": type(e).__name__}

    data = _extract_json(text)
    flags = _valid_flags(data) if data is not None else None
    if flags is None:
        # malformed 출력 — '판정 없음'을 통과(ok)로 위장하지 않는다
        return {"status": "unavailable", "model": config.JUDGE_MODEL,
                "flags": [], "reason": "MalformedJudgeOutput"}
    return {"status": "flagged" if flags else "ok", "model": config.JUDGE_MODEL,
            "flags": flags, "reason": ""}


_KIND_LABEL = {
    "negation_flip": "부정 관계 뒤집기",
    "subject_swap": "주어(대상) 바꿔치기",
    "meaning_inversion": "의미 반전",
}


def judge_warning_text(judge: dict) -> str:
    """judge flags → 답변에 부착할 경고문(1차 게이트 warning_text 와 같은 형식)."""
    parts = []
    for f in judge.get("flags", []):
        label = _KIND_LABEL.get(f["kind"], f["kind"])
        parts.append(
            f"⚠ LLM 2차 검증({label}): '{f['claim']}' — {f['reason']} "
            "(확률적 판정이므로 규정 원문과 직접 대조하세요.)"
        )
    return "\n".join(parts)
