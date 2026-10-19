"""인과성 평가 (Causality Assessment) — WHO-UMC 척도의 규칙 기반 '제안'.

PV 트리아지 다음 단계는 "이 이상사례가 정말 약 때문인가"를 평가하는 것이다.
REG-005 §2가 명시하듯 실무는 WHO-UMC 또는 Naranjo 척도를 쓴다.

트리아지(중대성 판정)와 달리 인과성은 규칙만으로 '확정'할 수 없다:
중대성 기준은 닫힌 목록(대조)이지만, 인과성은 임상 정보(중단/재투여 경과,
병용약물, 기저질환)를 종합하는 '판단'이라 정보가 부족하면 답이 없다.

그래서 이 모듈의 출력은 판정이 아니라 **제안(suggested) + 근거 신호 + 부족 정보 질문**이다:
  - 케이스 서술에서 WHO-UMC 판단 요소 4가지 신호를 감지한다
    (시간적 선후관계 · dechallenge · rechallenge · 대체 원인).
  - 감지된 신호 조합으로 도달 '가능한 최대' 등급을 제안하되, 정보가 없는
    요소는 충족으로 간주하지 않는다(보수적 — 등급을 올려주지 않는다).
  - 판단에 필요한데 서술에 없는 정보를 `missing_info` 질문 목록으로 돌려줘,
    PV 담당자가 보고자에게 무엇을 되물어야 하는지(follow-up query) 안내한다.

이 경계 설계가 핵심이다: 규칙은 '신호 감지와 질문 생성'까지, 확정은 사람.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# WHO-UMC 판단 요소별 감지 키워드 (케이스 자유 서술 대상)
_TEMPORAL_MARKERS = ["복용 후", "투여 후", "접종 후", "복용 직후", "투여 직후", "맞은 후", "먹은 후", "복용하고", "투여하고"]

# 부정 어미 일반 규칙(v9) — "호전되지 않았다"·"재발 없음"처럼 긍정 어휘 뒤에
# 부정이 곧바로 붙는 활용형은 열린 집합이라(되지 않/하지는 않/은 없/없음/이 아니…)
# 닫힌 목록 열거로는 새 활용형마다 같은 결함이 재발한다("호전되지 않"이 목록에
# 없어 IMPROVE 의 "호전"이 먼저 매칭 → 반증이 양성으로 뒤집힘). 긍정 어휘 매칭
# '직후 N자 이내'의 부정 어미(않·없·아니)를 검사해, 부정이 확인되면 '판단 불가'가
# 아니라 반대 상태(정보가 있는 반증)로 매핑한다 — 기존 3상태 체계 유지.
# "아닌"은 완성형 음절이라 "아니"를 부분 문자열로 포함하지 않는다 — 별도 나열.
_NEGATION_TAIL_RE = re.compile(r"않|없|아니|아닌")
# "되지 않"·"하지는 않"·"은 없음" 이 들어오는 최소 폭 — 더 넓히면 다음 절의
# 무관한 부정("호전됐고, 발열은 없다")까지 삼켜 반대 방향 오탐이 된다.
_NEGATION_SPAN = 6
# 절(clause) 경계 — 부정 어미는 긍정 어휘와 '같은 절'에 붙는 활용형(회복되지
# 않)일 때만 그 어휘의 부정이다. 창이 쉼표·마침표를 넘어가면 다음 절의 무관한
# 부정("중단하니 회복, 이상없음"의 '없음'은 회복이 아니라 이상의 부정)을 삼켜
# 양성 경과를 반증으로 뒤집는다 — span=6 만으로는 짧은 쉼표 절을 못 막았다(v10).
# 경계 집합은 **절 종결자만** 둔다(v11) — 가운뎃점 '·'(동의어 병렬 "호전·회복")·
# 줄바꿈 '\n'은 **양의적**이라 이분법으로 넣거나 빼면 반대편에서 진다(v12):
#   · 용언 내부 랩("호전\n되지 않음")에서 경계로 보면 부정을 놓쳐 과대평가(v11 제거 이유)
#   · 절 구분("회복\n이상없음")에서 경계로 안 보면 다음 절의 '없'이 회복을 부정해
#     rationale 이 사실과 반대로 단언(v11 제거가 새로 만든 반대편 미탐)
# 그래서 \n 을 절 종결자에 두되, **직후가 용언 연속(되·지·않·은·는·안·못)이면**
# 같은 절의 랩으로 보고 이어붙인다 — 연속 인지 경계(경계 봉합을 클래스로).
# 말줄임표 '…'·가운뎃점 '·'은 어절 내부(동의어 병렬·주저)라 경계에서 제외.
_CLAUSE_BOUNDARY_RE = re.compile(r"[,.;:!?、。，．；：！？\t　\n]")
# 용언 내부 줄바꿈 랩 — \n 직후가 용언 연속이면 절 경계가 아니라 같은 술어의 이어짐
_PREDICATE_WRAP_RE = re.compile(r"\n(?=\s*(?:되|지|않|은|는|안|못|였|았))")


def _negated_after(text: str, end: int) -> bool:
    """text[end:] 기준 N자 이내(단, 절 경계 전까지)에 부정 어미가 있는가."""
    win = text[end : end + _NEGATION_SPAN]
    win = _PREDICATE_WRAP_RE.sub("", win)   # 용언 내부 줄바꿈은 이어붙여 같은 절로
    boundary = _CLAUSE_BOUNDARY_RE.search(win)
    if boundary:
        win = win[: boundary.start()]   # 다음 절의 부정은 이 어휘의 부정이 아니다
    return bool(_NEGATION_TAIL_RE.search(win))


def _polarity(win: str, positive_words: list[str]) -> str:
    """창 안의 긍정 어휘를 부정 어미 규칙으로 판정 — positive | negated | absent.

    부정된 출현과 부정 없는 출현이 공존하면 positive 를 우선한다
    ("처음엔 호전되지 않다가 이후 호전" — 마지막에 성립한 긍정 서술이 경과다).
    """
    negated = False
    for w in positive_words:
        start = 0
        while (idx := win.find(w, start)) >= 0:
            if _negated_after(win, idx + len(w)):
                negated = True
            else:
                return "positive"
            start = idx + 1
    return "negated" if negated else "absent"

# 경과(outcome) 창 기반 감지 (v8) — "중단하니"·"다시 복용하니" 같은 결과
# 미포함 마커는 뒤따르는 경과가 호전이든 악화든 무조건 양성이 됐다.
# negative dechallenge("중단하니 오히려 악화")와 negative rechallenge
# ("다시 복용하니 아무 증상이 없었다")는 인과성의 '반증'인데, 마커가 결과를
# 안 보니 신호가 반대로 최고 등급(Certain)을 지지하고 rationale 문장까지
# 사실과 어긋났다. 마커는 '맥락의 시작'만 표시하고, 경과 어휘는 그 마커
# 이후의 창(window) 안에서 따로 판정한다 — 창의 끝은 다음 맥락 마커
# (중단 서술 뒤에 재투여 서술이 이어지는 케이스에서 경과가 섞이지 않도록).
_DECHALLENGE_CONTEXT = ["중단", "끊은", "끊으니", "끊었", "중지"]
_RECHALLENGE_CONTEXT = ["재투여", "다시 복용", "다시 투여", "재복용"]
_IMPROVE_WORDS = ["호전", "회복", "소실", "사라", "좋아", "가라앉"]
_NOT_IMPROVE_WORDS = ["악화", "에도 지속", "그대로", "계속되", "여전히"]
_RECUR_WORDS = ["재발", "다시 나타", "다시 발생", "또 나타", "같은 증상", "동일 증상", "재현"]
_NO_RECUR_WORDS = ["재발하지 않", "재발은 없", "재발이 없", "증상이 없", "증상은 없", "아무 증상", "나타나지 않", "발생하지 않"]

# 대체 원인은 3상태(v8): 존재(present) / 명시 배제(excluded) / 미언급(unknown).
# '미언급'을 '없음 충족'으로 치면 docstring 의 원칙("정보가 없는 요소는 등급을
# 올려주지 않는다")과 정면 모순 — 대체원인만은 not alternative 로 위로 열려
# 있었다. 또 "병용약물은 없었다" 같은 명시 부정문이 '병용' 부분 매칭으로
# 대체원인 '있음'이 되어 등급이 반대 방향으로 밀리고 사유 문장도 틀렸다.
_ALTERNATIVE_CAUSE = ["병용", "함께 복용", "다른 약", "기저질환", "원질환", "원래 앓", "기존 질환", "지병"]
_ALTERNATIVE_NEGATED = [
    "병용약물은 없", "병용약은 없", "병용약 없", "병용 약물은 없", "병용약물이나 기저질환은 없",
    "기저질환은 없", "기저질환 없", "기저질환이나 병용약물은 없", "다른 약은 없", "다른 약물은 없",
    "지병은 없", "특이 병력 없", "특이 기저질환 없", "대체 원인 없",
]

# WHO-UMC 등급 (영문 원어 병기 — KAERS/해외 보고 시 그대로 쓰는 용어)
CERTAIN = "확실함(Certain)"
PROBABLE = "상당히 확실함(Probable/Likely)"
POSSIBLE = "가능함(Possible)"
UNLIKELY = "가능성 적음(Unlikely)"
UNASSESSABLE = "평가곤란(Unassessable)"


@dataclass
class CausalityResult:
    suggested: str                       # 제안 등급(WHO-UMC)
    signals: dict[str, bool]             # 감지된 판단 요소 신호
    rationale: str                       # 등급 제안 사유
    missing_info: list[str] = field(default_factory=list)  # 보고자에게 되물을 질문


def _detect(case_text: str, markers: list[str]) -> bool:
    return any(m in case_text for m in markers)


def _window(text: str, start_markers: list[str], stop_markers: list[str]) -> str | None:
    """첫 (부정되지 않은) start 마커부터 다음 stop 마커 직전까지의 경과 서술 창.

    맥락 마커 자체가 부정된 서술("약을 중단하지 않았는데도 호전")에서는 그
    경과 자체가 일어난 적이 없다 — 부정된 마커로 창을 열면 뒤따르는 "호전"이
    존재하지 않는 dechallenge 의 양성 신호로 둔갑한다(v9). 마커 직후의
    부정 어미(않·없)를 같은 일반 규칙으로 검사해 그 출현은 건너뛴다.
    """
    starts: list[int] = []
    for m in start_markers:
        idx = text.find(m)
        while idx >= 0:
            if not _negated_after(text, idx + len(m)):
                starts.append(idx)
                break
            idx = text.find(m, idx + 1)
    if not starts:
        return None
    start = min(starts)
    stops = [i for m in stop_markers if (i := text.find(m, start + 1)) > start]
    return text[start : min(stops) if stops else len(text)]


def _dechallenge_state(case_text: str) -> str:
    """중단(dechallenge) 경과 — improved | not_improved | unknown.

    창의 끝을 재투여 맥락에서 끊는다 — "중단 후 호전, 재투여하니 악화"에서
    악화(재투여 경과)가 중단 경과를 오염시키지 않도록.
    """
    win = _window(case_text, _DECHALLENGE_CONTEXT, _RECHALLENGE_CONTEXT)
    if win is None:
        return "unknown"
    if any(w in win for w in _NOT_IMPROVE_WORDS):
        return "not_improved"
    polarity = _polarity(win, _IMPROVE_WORDS)
    if polarity == "positive":
        return "improved"
    if polarity == "negated":
        # "호전되지 않았다"는 판단 불가가 아니라 '호전 안 됨'이 확인된 반증이다
        return "not_improved"
    return "unknown"


def _rechallenge_state(case_text: str) -> str:
    """재투여(rechallenge) 경과 — recurred | no_recurrence | unknown."""
    win = _window(case_text, _RECHALLENGE_CONTEXT, [])
    if win is None:
        return "unknown"
    if any(w in win for w in _NO_RECUR_WORDS):
        return "no_recurrence"
    polarity = _polarity(win, _RECUR_WORDS)
    if polarity == "positive":
        return "recurred"
    if polarity == "negated":
        # "재발 없음"(조사 생략형)처럼 닫힌 목록 밖의 부정 활용 — 부정이 곧
        # 반대 상태(재발하지 않음)이므로 no_recurrence 로 매핑한다
        return "no_recurrence"
    return "unknown"


def _alternative_state(case_text: str) -> str:
    """대체 원인 — present | excluded | unknown. 명시 부정문을 먼저 본다."""
    if any(m in case_text for m in _ALTERNATIVE_NEGATED):
        return "excluded"
    if _detect(case_text, _ALTERNATIVE_CAUSE):
        return "present"
    return "unknown"


def assess_causality(case_text: str) -> CausalityResult:
    """케이스 서술에서 WHO-UMC 신호를 감지해 인과성 등급을 '제안'한다.

    등급 규칙(WHO-UMC 기준을 감지 가능한 신호로 근사, v8):
      - 시간관계 + 중단 후 호전 + 재투여 후 재발 + 대체원인 '명시 배제' → Certain
      - 위와 같으나 대체원인 미언급                                   → Probable
        (미언급은 배제 확인이 아니다 — Certain 은 대체원인 배제가 '확인'될 때만)
      - 재투여 후 '재발하지 않음'(반증 신호)                           → Possible 상한
      - 시간관계 + 중단 후 호전 (대체원인 미확인/배제)                 → Probable
      - 시간관계 있음(대체원인 존재 포함)                              → Possible
      - 시간관계 없음 + 중단/재투여 양성 경과(시간관계 함의)           → Possible
        (경과가 신호인데 투여 시점 서술이 없다 — 확인 질문과 함께 낮게 제안)
      - 시간관계 없음 + 대체원인 존재                                  → Unlikely
      - 신호 자체가 없음                                               → Unassessable
    정보가 서술에 없으면 '충족'으로 치지 않는다 — 등급은 아래로만 보수 적용된다.
    """
    temporal = _detect(case_text, _TEMPORAL_MARKERS)
    de_state = _dechallenge_state(case_text)
    re_state = _rechallenge_state(case_text)
    alt_state = _alternative_state(case_text)

    dechallenge = de_state == "improved"
    rechallenge = re_state == "recurred"
    alternative = alt_state == "present"

    # 표시용 신호는 '양성 확인'만 True — 반증/미언급은 rationale·질문이 나른다.
    signals = {
        "시간적 선후관계": temporal,
        "중단 후 호전(dechallenge)": dechallenge,
        "재투여 후 재발(rechallenge)": rechallenge,
        "대체 원인 가능성(병용약·기저질환)": alternative,
    }

    # 부족 정보 → follow-up 질문. '미언급(unknown)'일 때만 묻는다 —
    # 반증(악화·미재발)이나 명시 배제는 이미 '정보가 있는' 상태다.
    missing: list[str] = []
    if not temporal:
        missing.append("의심약 투여와 증상 발생의 시간적 선후관계(투여 후 언제 발생했는가)")
    if de_state == "unknown":
        missing.append("의심약 중단(dechallenge) 후 증상 경과(호전/지속)")
    if re_state == "unknown":
        missing.append("재투여(rechallenge) 여부와 재발 여부")
    if alt_state == "unknown":
        missing.append("병용약물·기저질환 등 대체 원인 유무")

    if temporal and dechallenge and rechallenge and alt_state == "excluded":
        suggested = CERTAIN
        rationale = "시간관계·중단 후 호전·재투여 후 재발이 모두 확인되고 대체 원인이 명시적으로 배제되었다."
    elif temporal and dechallenge and rechallenge:
        suggested = PROBABLE
        rationale = (
            "시간관계·중단 후 호전·재투여 후 재발은 확인되나, 대체 원인의 배제가 "
            "서술로 확인되지 않아 Certain 은 보류한다(미언급은 배제 확인이 아니다)."
        )
    elif temporal and re_state == "no_recurrence":
        suggested = POSSIBLE
        rationale = "재투여 후 재발하지 않아(반증 신호) 시간관계만으로 Possible 이상 올리지 않는다."
    elif temporal and dechallenge and alt_state != "present":
        suggested = PROBABLE
        rationale = (
            "시간관계와 중단 후 호전이 확인되고 대체 원인이 명시적으로 배제되었다(재투여 정보는 없음)."
            if alt_state == "excluded"
            else "시간관계와 중단 후 호전이 확인된다(대체 원인 미언급 — 재투여 정보 없이 이 이상 올리지 않는다)."
        )
    elif temporal:
        suggested = POSSIBLE
        rationale = (
            "시간적 선후관계는 확인되나 "
            + (
                "병용약·기저질환 등 대체 원인 가능성이 있다."
                if alternative
                else (
                    "중단 후 호전이 확인되지 않아(악화/지속) 그 이상 올릴 수 없다."
                    if de_state == "not_improved"
                    else "중단/재투여 경과 정보가 없어 그 이상 올릴 수 없다."
                )
            )
        )
    elif dechallenge or rechallenge:
        # 중단/재투여 양성 경과는 감지됐는데 투여 시점 서술이 없는 케이스 —
        # 이대로 Unassessable 로 떨어뜨리면 rationale("감지되지 않는다")이
        # 감지된 신호와 모순된다(v9). WHO-UMC 근사의 보수 원칙대로 낮은 등급
        # (Possible)에 머물고, 시간관계 확인 질문(missing_info)으로 넘긴다.
        suggested = POSSIBLE
        rationale = (
            "중단/재투여 후 경과(호전·재발)가 시간관계를 함의하나 투여 시점 서술이 "
            "없어 Possible 이상 올리지 않는다 — 투여~발생의 시간적 선후관계 확인이 필요하다."
        )
    elif alternative:
        suggested = UNLIKELY
        rationale = "시간적 선후관계가 서술에 없고 대체 원인(병용약·기저질환) 가능성이 있다."
    elif de_state != "unknown" or re_state != "unknown" or alt_state != "unknown":
        # 반증(악화·미재발)이나 대체원인 배제만 있는 케이스 — 신호가 '없는' 것이
        # 아니므로 rationale 이 감지 사실을 반영해야 한다(모순 문장 방지, v9).
        suggested = UNASSESSABLE
        rationale = (
            "시간적 선후관계가 서술에서 감지되지 않아 평가가 곤란하다 — 중단/재투여 "
            "경과 등 일부 신호는 있으나 투여~발생 시간관계 확인이 선행되어야 한다."
        )
    else:
        suggested = UNASSESSABLE
        rationale = "인과성 판단 요소(시간관계·경과·대체원인)가 서술에서 감지되지 않는다."

    return CausalityResult(
        suggested=suggested,
        signals=signals,
        rationale=rationale,
        missing_info=missing,
    )
