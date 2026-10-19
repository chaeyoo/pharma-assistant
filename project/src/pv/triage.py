"""이상사례(Adverse Event) 트리아지 — 중대성 판정 + 보고 경로/기한 계산.

PV 담당자의 첫 업무는 접수된 케이스를 '어떤 경로로 언제까지 보고해야 하는가'로
분류(triage)하는 것이다. 이 판정을 LLM에게 맡기지 않고 규칙 기반으로 구현한 이유:

  - 보고기한 계산은 컴플라이언스 그 자체다. 하루라도 틀리면 사고인데,
    LLM의 날짜 연산·기준 적용은 확률적이라 감사(audit) 대상이 될 수 없다.
  - 중대성(Serious) 기준은 규정(REG-005)에 닫힌 목록으로 정의되어 있어
    결정론적 규칙으로 충분히 표현된다 — 판단이 아니라 '대조'다.
  - 에이전트 설계 원칙: LLM은 도구 선택·설명(orchestration)을 맡고,
    규정이 정한 계산은 결정론적 도구가 맡는다. 결과가 항상 재현·검증된다.

판정 기준·기한은 사내 규제문서 REG-005(시판 후 안전관리 및 약물감시 기준)를
코드로 옮긴 것이며, MCP 도구 계층에서 해당 근거 문단을 함께 반환한다.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

from .redactor import redact

# 중대성(Serious) 기준 — REG-005 §2 의 닫힌 목록을 (기준명, 감지 키워드) 로 코드화.
# 키워드는 케이스 서술(자유 텍스트)에서 기준 충족을 감지하는 데 쓴다.
# 사망은 **미탐이 치명적**인 축이다 — 놓친 사망(비중대→정기보고 라우팅)은
# 데모의 핵심 컴플라이언스 주장("하루 틀리면 사고")을 정면으로 훼손한다.
# 사망 어휘는 형태 규칙 하나로 다 덮이지 않는 **열린 어휘 집합**이라(사망·별세·
# 소천·타계·서거·영면·작고…), 여기서는 흔한 표현을 **열거로 보강**한다. 이것은
# NFKC 같은 클래스 봉합이 아니라 **열거 + recall 규율**이다 — 자동 확정 계층이라
# 미탐 시 caveat 조차 없으므로 넓게 잡아 사람 확정이 tail 을 받게 한다(과탐 걷어
# 내기 < 미탐 사고). 잔여 tail(신조 완곡어 등)은 남으며 근본 해결은 NER/사전이다.
# 단 **짧은 마커는 형태소 확정형으로 좁힌다**(v12) — '급사'가 지급사유·공급사,
# '별세'가 특별세·개별세대, '운명하'가 '운명하에'에 substring 오발화해 정상 케이스를
# D+0(지체없이)로 오라우팅했다(장애 마커를 결합형으로 좁힌 v8 과 동형: 열거를
# 넓히되 짧은 항목은 경계로 좁혀 과탐을 막는다).
SERIOUSNESS_CRITERIA: dict[str, list[str]] = {
    "사망": [
        "사망", "숨졌", "죽었", "사망함",
        "돌연사", "유명을 달리", "숨을 거두", "숨을 거뒀", "숨을 거둔",
        # 짧은 마커 — 형태소 확정형으로만(급사/별세/운명하/임종의 substring 오발화 차단)
        "급사하", "급사한", "급사함", "급사했", "급사로",
        "별세하", "별세한", "별세함", "별세했", "별세로", "향년",
        "운명하셨", "운명하였", "운명했", "운명하신",
        "임종하", "임종했",
        # 흔한 완곡 사망어 보강(열거 — 저충돌 어휘, 충돌형은 활용형으로)
        "소천", "타계", "서거", "영면하", "작고하", "절명",
    ],
    "생명 위협": ["생명", "생명위협", "아나필락시스", "쇼크", "심정지", "소생술", "중환자실"],
    "입원 또는 입원기간 연장": ["입원", "입원기간 연장", "재입원"],
    # '장애' 단독 키워드는 동음이의 과탐(v8) — 한국어 의무기록에서 "위장 장애"·
    # "수면장애"의 '장애'는 disorder(질환 분류)이지 disability(신체 장애)가
    # 아닌데, 단독 매칭은 이를 전부 중대(15일 신속보고)로 밀어 올렸다.
    # disability 의미가 확정되는 결합형으로만 좁힌다(실명·청력 상실은 그 자체로 확정).
    "지속적/중대한 장애": [
        "영구 장애", "영구적 장애", "영구적인 장애", "장애 판정", "장애가 남",
        "지속적 장애", "지속적인 장애", "중대한 장애", "일상생활 장애",
        "실명", "청력 상실", "영구적",
    ],
    "선천적 기형": ["기형", "선천성", "선천적"],
    "기타 의학적으로 중요한 사건": ["의학적으로 중요"],
}

# 즉시(지체 없이) 신속보고 대상 기준 — REG-005 §2 표의 '사망·생명위협' 행
_IMMEDIATE = {"사망", "생명 위협"}

# 예상 여부(expectedness) 감지 — 허가사항(첨부문서)에 이미 기재된 반응인가.
# unexpected 판정이 expected 매칭보다 우선한다(_detect_expectedness) — "허가사항에
# 기재되어 있지 않은"은 expected 마커("허가사항에 기재")를 부분 문자열로 포함하므로,
# 부정형을 unexpected 마커에 두지 않으면 부정문이 expected 로 뒤집힌다(v9).
# unexpected 오판은 보수(15일 트래킹 유지) 방향이라 과탐이 미탐보다 싸다.
_EXPECTED_MARKERS = ["허가사항에 기재", "첨부문서에 기재", "알려진 부작용", "예상된"]
_UNEXPECTED_MARKERS = [
    "예상치 못한", "예상하지 못한", "허가사항에 없", "알려지지 않은",
    # "아닌"·"아님"은 음절 단위 완성형이라 "아니"의 부분 문자열이 아니다 —
    # 종결형(아니다)·관형형(아닌)·명사형(아님)이 서로 부분문자열이 아니라 각각
    # 필요하다(v11 — v9 가 아니/아닌만 열거해 명사형 "…아님"이 expected 로
    # 뒤집혔다). 근본 해결은 부정 어간 규칙(causality 의 _NEGATION_TAIL_RE 계열)의
    # 이식이나, expectedness 오판은 보수 방향(15일 유지)이라 최소 보강에 그친다.
    "기재되어 있지 않", "기재되지 않", "기재돼 있지 않",
    "알려진 부작용이 아니", "알려진 부작용이 아닌", "알려진 부작용이 아님",
    "기재된 반응이 아니", "기재된 반응이 아닌", "기재된 반응이 아님",
]


@dataclass
class TriageResult:
    is_serious: bool
    criteria_met: list[str]            # 충족한 중대성 기준
    expectedness: str                  # expected | unexpected | unknown
    route: str                         # 보고 경로 요약
    deadline_days: int | None          # 인지일로부터 며칠(즉시=0, 정기보고=None)
    awareness_date: str                # 인지일(YYYY-MM-DD)
    deadline_date: str | None          # 계산된 보고 마감일
    rationale: str                     # 판정 사유(사람이 읽는 설명)
    caveats: list[str] = field(default_factory=list)


def _detect_criteria(case_text: str) -> list[str]:
    low = case_text.lower()
    return [
        name
        for name, keywords in SERIOUSNESS_CRITERIA.items()
        if any(k in low for k in keywords)
    ]


def _detect_expectedness(case_text: str) -> str:
    if any(m in case_text for m in _UNEXPECTED_MARKERS):
        return "unexpected"
    if any(m in case_text for m in _EXPECTED_MARKERS):
        return "expected"
    return "unknown"


def assess_case(case_text: str, awareness_date: str = "") -> TriageResult:
    """케이스 서술을 트리아지한다.

    Args:
        case_text: 이상사례 케이스 서술(자유 텍스트).
        awareness_date: 회사가 케이스를 인지한 날(YYYY-MM-DD). 없으면 오늘.
            형식이 잘못되면 오늘로 폴백하되 caveat 로 명시한다(조용한 폴백 금지).

    판정 규칙(REG-005 §2):
      - 사망·생명위협           → 지체 없이(신속보고, D+0)
      - 그 외 중대 기준 충족     → 인지일로부터 15일 이내 신속보고
      - 비중대                  → 정기보고(PSUR)에 포함
    예상 여부를 판별할 수 없으면 '예상치 못한 사례'로 보수적으로 취급한다
    (기한을 놓치는 쪽보다 이르게 잡는 쪽이 안전한 실패).

    기한의 '일' 단위는 역일(calendar day)이다 — ICH 신속보고 관행과 같고,
    timedelta(days=15) 산술도 역일이다. 근무일(working day)로 읽으면 실제
    달력 기한이 뒤로 밀린다(근무일↔역일 혼동은 검증 게이트가 답변 축에서
    잡는 바로 그 오류라, 계산 계층도 단위를 명시한다 — v8).
    """
    # 인지일 형식 오류는 '조용히' 오늘로 대체하지 않는다 — 보고기한이 잘못된
    # 기준일로 계산되는 것은 컴플라이언스 리스크인데, 폴백이 조용하면 사용자는
    # 기한이 틀렸다는 신호를 받을 길이 없다(시끄러운 실패 원칙).
    caveats: list[str] = []
    try:
        aware = _dt.date.fromisoformat(awareness_date) if awareness_date else _dt.date.today()
    except ValueError:
        aware = _dt.date.today()
        # 형식이 틀린 인지일은 임의의 자유 텍스트일 수 있다 — 원문을 그대로
        # 에코하면 caveat 가 비마스킹 PII 유출 경로가 된다(도구 계층이 as_of·
        # 필터 에코를 redact 로 감싸는 것과 대칭, v9).
        caveats.append(
            f"인지일 '{redact(awareness_date).text}' 이(가) YYYY-MM-DD 형식이 아니어서 "
            f"오늘({aware.isoformat()}) 기준으로 기한을 계산했습니다. "
            "실제 인지일로 기한 재계산이 필요합니다."
        )
    # 형식은 유효해도 '오늘보다 미래'인 인지일은 오타(연도·월 뒤바뀜)일 가능성이
    # 높다 — 형식 검사만 하고 타당성(plausibility)은 안 보면, 2027 로 잘못 친
    # 인지일이 기한을 1년 뒤로 조용히 밀어도 아무 신호가 없다(형식 오류에는
    # 시끄러운 caveat 를 달아 놓고 값 오류는 조용히 통과시키는 비대칭).
    # 계산은 입력값대로 수행한다 — 자동 정정은 또 다른 조용한 폴백이다.
    if aware > _dt.date.today():
        caveats.append(
            f"인지일 {aware.isoformat()} 이(가) 오늘 이후의 미래 날짜입니다 — "
            "입력 오류(연도·월 오타)일 수 있으니 실제 인지일을 확인하세요. "
            "기한은 입력된 값 기준으로 계산되어 있습니다."
        )

    criteria = _detect_criteria(case_text)
    expectedness = _detect_expectedness(case_text)

    if not criteria:
        return TriageResult(
            is_serious=False,
            criteria_met=[],
            expectedness=expectedness,
            route="비중대 이상사례 → 정기보고(PSUR)에 포함",
            deadline_days=None,
            awareness_date=aware.isoformat(),
            deadline_date=None,
            rationale="케이스 서술에서 중대성(Serious) 기준(사망·생명위협·입원·장애·기형 등)이 감지되지 않았다.",
            caveats=[*caveats, "규칙 기반 1차 분류입니다. 최종 중대성 판단은 PV 담당자가 확정해야 합니다."],
        )

    if _IMMEDIATE & set(criteria):
        deadline_days = 0
        route = "신속보고 — 지체 없이(사망·생명위협)"
        rationale = f"사망·생명위협 기준 충족({', '.join(sorted(_IMMEDIATE & set(criteria)))}) → 지체 없이 보고."
    else:
        deadline_days = 15
        route = "신속보고 — 인지일로부터 15일(역일) 이내"
        rationale = f"중대성 기준 충족({', '.join(criteria)}) → 15일(역일, calendar day) 이내 신속보고."
        if expectedness == "expected":
            caveats.append(
                "허가사항에 기재된(예상된) 반응으로 보이나, 보수적으로 15일 트래킹을 권고합니다. "
                "정기보고 전환 여부는 규정 원문과 PV 담당자 확인이 필요합니다."
            )
        elif expectedness == "unknown":
            caveats.append("예상 여부(expectedness)를 판별하지 못해 '예상치 못한 사례'로 보수 적용했습니다.")

    caveats.append("규칙 기반 1차 분류입니다. 최종 중대성·인과성 판단은 PV 담당자가 확정해야 합니다.")
    return TriageResult(
        is_serious=True,
        criteria_met=criteria,
        expectedness=expectedness,
        route=route,
        deadline_days=deadline_days,
        awareness_date=aware.isoformat(),
        deadline_date=(aware + _dt.timedelta(days=deadline_days)).isoformat(),
        rationale=rationale,
        caveats=caveats,
    )
