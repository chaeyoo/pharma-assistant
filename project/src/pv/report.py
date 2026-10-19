"""이상사례 개별사례보고(ICSR) 초안 — 최소보고요건 검증 + KAERS 초안 조립.

트리아지(triage)·인과성 제안(causality)·용어 코딩(coding)을 한 번에 묶어
PV 담당자가 KAERS(한국 의약품 이상사례 보고시스템)에 올릴 보고서의
'초안'을 만든다. 사람이 빈칸을 채워 확정하는 것이 전제다.

핵심은 초안 생성보다 **최소보고요건(minimum reporting criteria) 검증**이다.
ICH E2D가 정의하는 유효한 케이스의 4요소:
  ① 식별 가능한 환자  ② 식별 가능한 보고자  ③ 의심 의약품  ④ 이상사례
하나라도 없으면 '보고 가능한 케이스'가 아니라 '정보 보완 대상'이다.
→ 이 도구는 4요소 충족 여부를 판정하고, 빠진 요소를 follow-up 항목으로 안내한다.

PII 마스킹과의 관계(설계상 긴장 지점):
  ①의 요건은 '환자를 특정할 수 있는 정보가 존재하는가'이지 원문 값이 아니다.
  마스킹은 값을 지우지만 존재 신호([이름]님, 환자번호 [번호], "45세 남성")는
  남기므로, 마스킹된 텍스트 위에서도 요건 판정이 가능하다.
  즉 '외부 API 경계에서는 값을 감추고, 요건 판정은 신호로 한다'로 양립시켰다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .causality import CausalityResult, assess_causality
from .coding import (
    CandidateTerm,
    CodedTerm,
    code_terms,
    flag_uncoded_expressions,
    suggest_candidates,
)
from .triage import TriageResult, assess_case

# 최소보고요건 감지 신호 (마스킹된 텍스트에서도 남는 '존재 신호' 위주)
_PATIENT_MARKERS = ["환자", "환아", "수진자"]
# 나이 정규식: '세' 뒤가 '대'로 이어지는 "3세대(세팔로스포린)"는 세대(generation)
# 표기이지 환자 나이가 아니다 — 후방 경계로 배제한다(v9).
# 소아 명사 "남아"(男兒)는 동사 "남아"(남다+아)와 동형이라, "증상이 남아 있음"
# 같은 잔존 서술에 오발화해 환자정보 없는 케이스의 ①요건을 조용히 충족시켰다.
# v11 은 동사 활용을 **부정 열거**(있|계|서|났|나서)로 배제했으나, 열거 밖 활용
# ("약이 남아도"·"남아만"·종결 남아)이 그대로 새어 조용한 통과가 재발했다(v12 —
# 열거는 다음 반례에서 진다). 배제를 열거하는 대신, **명사 용법의 긍정 경계**로
# 뒤집는다: 명사 남아는 주격/목적격/관형 조사(가·는·을·를·의·와·과…)나 환자
# 명사가 직후에 오고, 동사 남아는 연결어미(도·서·만·아…)가 온다.
_PATIENT_RE = re.compile(
    r"\d{1,3}\s*세(?!대)|[남여]성"
    r"|남아(?=\s*(?:가|는|를|의|와|과|랑|이나|한테|에게)|\s*환)|여아"
)
# 마스킹 토큰 [이름] 은 환자·보고자를 구분하지 않는다 — 보고자 직역(의사·약사 등)
# 바로 뒤(공백 0~1자)에 붙은 [이름] 은 보고자 성명일 개연성이 높은데, 이를 환자
# 요건 ①의 신호로 전용하면 보고자만 있는 케이스가 ①을 조용히 통과한다(v9).
# 직역에 직결된 [이름] 만 좁게 제외한다 — 독립 [이름](PV-009 등)의 ①신호는 유지.
_REPORTER_NAME_TOKEN_RE = re.compile(
    r"(?:보고자|의사|약사|간호사|의료진|보호자)\s?\[이름\]"
)
# "병원에서" 단독 마커는 치료 문맥("병원에서 치료받았다")을 보고자로 오탐해
# 보고자 정보가 없는 케이스를 reportable=True 로 '조용히 통과'시켰다(v8) —
# 이 모듈의 실패 방향은 '시끄러운 보완 요청'이어야 하므로, 병원 언급은
# 보고 행위가 결합된 형태만 인정한다.
# 전문의·주치의(v11) — '의사' 어절이 없어(전문+의·주치+의) v9 이전부터의 갭 보강.
# (한의사·한약사는 아래 _REPORTER_RE 로 옮겼다 — substring 마커로 두면 "대한한의사
# 협회"·"한약사회" 기관명에 오발화해 보고자 없는 케이스가 조용히 통과했다, v12.)
_REPORTER_MARKERS = ["보고자", "간호사", "보호자", "본인이 직접", "의료진", "병원에서 보고", "병원이 보고",
                     "전문의", "주치의"]
# "의사"·"약사"는 합성어 부분 매칭이 잦다 — "의사소통"·"의사결정"·"의사표시"의
# '의사'는 intent, "약사법"의 '약사'는 법령명이지 사람(보고자)이 아닌데, 단순
# substring 은 이를 전부 ②요건 충족으로 밀어 '조용한 통과'가 됐다(v9).
# v9 는 후행 어절을 네거티브 룩어헤드로 열거해 막았으나, 이는 '의사'가 접두인
# 합성어(의사소통)만 가릴 뿐 '의사'가 어절 중간·후미에 박힌 최빈 합성어를
# 놓쳤다 — "주(注)+의사(意事)+항"=주의사항, 유의사항, "제(製)+약사"=제약사
# 는 그대로 보고자로 발화해 v9 가 없애려던 '조용한 통과'가 재발했다(v10).
# 열거로 좁히는 방향은 새 합성어마다 진다(조사 열거를 어절 경계로 대체한
# redactor·검증기의 교훈과 같은 계열) — 사람(보고자)로서의 '의사/약사'는 늘
# 어절의 첫 음절이므로, 앞이 한글이면(합성어 내부) 비매칭하는 룩비하인드
# 경계로 대체한다. 후행 룩어헤드(소통/결정/표시/표현·법)는 "의사가 소통"류
# 정상 매칭 뒤 활용까지 막지 않도록 어절 첫 음절 조건과 함께 남긴다.
# 한의사·한약사도 같은 어절 경계 클래스로(v12) — 한글 직결 정당 보고자("한의사가"·
# "한약사가")는 인정하되, 앞이 한글인 기관명("대한한의사협회"→한 앞에 대한)과 후행
# "협회/회/법"은 배제한다. v11 이 이 마커를 substring 으로 둬 기관명에 오발화하던
# 비대칭을, 의사/약사에 이미 쓴 룩비하인드 봉합을 신규 마커에 전파해 대칭화.
_REPORTER_RE = re.compile(
    r"(?<![가-힣])의사(?!소통|결정|표시|표현)|(?<![가-힣])약사(?!법)"
    r"|(?<![가-힣])한의사(?!협회|회)|(?<![가-힣])한약사(?!회|법)"
)
# 의심약 감지: "OO정/캡슐/주사 + 복용/투여" 처럼 노출(exposure) 맥락이 따라올 때만
# 매칭한다 — '규정·판정·일정' 같은 '-정'으로 끝나는 일반 명사의 오탐을 막기 위해.
# 제품명 토큰은 비숫자 문자로 시작해야 한다 — "아스피린 1정을 복용"의 "1정"은
# 수량(dose)이지 제품명이 아닌데, 수량 토큰만으로 ③요건이 충족되면 제품명·성분
# 없는 케이스가 보완 요청 없이 통과한다(v9). 수량뿐이면 미충족(시끄러운 보완 요청).
_DRUG_RE = re.compile(
    r"[가-힣A-Za-z][가-힣A-Za-z0-9]*\s*(?:정|캡슐|주사제|주사|시럽|연고|패치)"
    r"(?=\s*(?:을|를)?\s*(?:복용|투여|접종|먹|맞))"
)


@dataclass
class ReportDraft:
    reportable: bool                     # 최소보고요건 4요소 충족 여부
    missing: list[str]                   # 빠진 요소(보완 안내)
    triage: TriageResult
    causality: CausalityResult
    coded_terms: list[CodedTerm]         # 1계층: 확정 코딩(집계 대상)
    candidate_terms: list[CandidateTerm] = field(default_factory=list)  # 2계층: 사람 확정 대기
    uncoded_expressions: list[str] = field(default_factory=list)        # 3계층: 감지만(PT 없음)
    draft_markdown: str = ""
    followups: list[str] = field(default_factory=list)   # 보고자에게 되물을 질문


def _check_minimum_criteria(
    case_text: str, suspected_drug: str, reporter: str, patient_info: str,
    coded: list[CodedTerm], triage: TriageResult,
    candidates: list[CandidateTerm], uncoded: list[str],
) -> tuple[list[str], dict[str, str]]:
    """ICH E2D 최소보고요건 4요소를 점검한다. (빠진 요소 목록, 채워진 값) 반환."""
    fields: dict[str, str] = {}
    missing: list[str] = []

    # 보고자 직역에 직결된 [이름](예: "담당 약사 [이름]님")을 걷어낸 뒤 남는
    # [이름] 만 환자 존재 신호로 인정한다 — 역할 무구분 전용 방지(위 주석, v9).
    patient_name_token = "[이름]" in _REPORTER_NAME_TOKEN_RE.sub("", case_text)
    if patient_info.strip():
        fields["환자"] = patient_info.strip()
    elif (
        any(m in case_text for m in _PATIENT_MARKERS)
        or patient_name_token
        or _PATIENT_RE.search(case_text)
    ):
        fields["환자"] = "케이스 서술에서 확인(비식별 처리됨) — 나이/성별/이니셜 보완 권장"
    else:
        missing.append("① 식별 가능한 환자 (나이·성별·이니셜 등 최소 식별 정보)")

    if reporter.strip():
        fields["보고자"] = reporter.strip()
    elif any(m in case_text for m in _REPORTER_MARKERS) or _REPORTER_RE.search(case_text):
        fields["보고자"] = "케이스 서술에서 확인 — 보고자 자격(의사/약사/소비자 등) 명시 권장"
    else:
        missing.append("② 식별 가능한 보고자 (보고자 자격과 연락 경로)")

    drug_match = _DRUG_RE.search(case_text)
    if suspected_drug.strip():
        fields["의심 의약품"] = suspected_drug.strip()
    elif drug_match:
        fields["의심 의약품"] = f"서술에서 감지: {drug_match.group(0).strip()} — 제품명·성분 확정 필요"
    else:
        missing.append("③ 의심 의약품 (제품명 또는 성분명)")

    # ④요소의 본질은 '구체적 이상사례 서술이 존재하는가'이지 '코딩에 성공했는가'가
    # 아니다(ICH E2D). 확정 코딩이 없어도 후보(2계층)·미코딩 감지(3계층)가 있으면
    # 요건은 충족으로 보고, 코딩 확정은 follow-up 으로 넘긴다 — 코딩 사전의 빈틈이
    # '보고 불가' 오판으로 연쇄되는 것을 여기서 끊는다. 막연한 서술("몸이 좋지
    # 않다")은 세 계층 모두 잡지 않으므로 여전히 미충족(specificity 요구).
    if coded or triage.criteria_met:
        fields["이상사례"] = ", ".join(t.pt for t in coded) if coded else "서술에서 확인(코딩 필요)"
    elif candidates:
        fields["이상사례"] = (
            "후보 감지(확정 필요): " + ", ".join(f"{c.pt}({c.pt_en})?" for c in candidates)
        )
    elif uncoded:
        fields["이상사례"] = f"증상 서술 감지(미코딩): {', '.join(uncoded)} — PT 부여 필요"
    else:
        missing.append("④ 이상사례 (구체적 증상/사건)")

    return missing, fields


def _render_markdown(
    case_text: str, fields: dict[str, str], missing: list[str],
    triage: TriageResult, causality: CausalityResult, coded: list[CodedTerm],
    candidates: list[CandidateTerm], uncoded: list[str],
) -> str:
    """KAERS 개별사례안전성보고(ICSR) 항목 구조를 따르는 사람이 읽는 초안."""
    lines = ["# 이상사례 개별사례보고(ICSR) 초안 — KAERS 제출용", ""]
    status = "✅ 최소보고요건 충족(초안 검토 후 제출 가능)" if not missing else "⛔ 정보 보완 필요(최소보고요건 미충족)"
    lines += [f"**상태: {status}**", ""]

    lines.append("## 1. 최소보고요건 (ICH E2D 4요소)")
    for name in ["환자", "보고자", "의심 의약품", "이상사례"]:
        lines.append(f"- {name}: {fields.get(name, '**(미확인 — 보완 필요)**')}")
    lines.append("")

    if coded or candidates or uncoded:
        lines.append("## 2. 이상사례 표준 용어 코딩 (MedDRA 방식)")
        lines.append("| 서술 표현 | PT(표준 용어) | SOC(기관계) | 상태 |")
        lines.append("|---|---|---|---|")
        for t in coded:
            lines.append(f"| {t.verbatim} | {t.pt} ({t.pt_en}) | {t.soc} | 확정 |")
        for c in candidates:
            lines.append(
                f"| {c.verbatim} | {c.pt} ({c.pt_en}) | {c.soc} | ⚠ 후보(승인/기각 필요) |"
            )
        for u in uncoded:
            lines.append(f"| {u} | (미코딩 — PT 부여 필요) | - | ⚠ 감지만 |")
        lines.append("")

    lines.append("## 3. 중대성 및 보고 기한 (규칙 기반 판정)")
    if triage.is_serious:
        lines.append(f"- 중대성: **중대(Serious)** — 기준: {', '.join(triage.criteria_met)}")
    else:
        lines.append("- 중대성: 비중대 (중대성 기준 미감지)")
    lines.append(f"- 보고 경로: {triage.route}")
    if triage.deadline_date:
        lines.append(f"- 보고 기한: **{triage.deadline_date}** (인지일 {triage.awareness_date} 기준, 역일(calendar day) 계산)")
    lines.append("")

    lines.append("## 4. 인과성 평가 (WHO-UMC · 제안)")
    lines.append(f"- 제안 등급: **{causality.suggested}**")
    lines.append(f"- 사유: {causality.rationale}")
    detected = [k for k, v in causality.signals.items() if v]
    if detected:
        lines.append(f"- 감지된 판단 요소: {', '.join(detected)}")
    lines.append("")

    lines.append("## 5. 경과 서술 (개인정보 비식별 처리본)")
    lines.append(f"> {case_text.strip()}")
    lines.append("")

    lines.append("---")
    lines.append("⚠ 본 초안은 규칙 기반 자동 생성입니다. 중대성·인과성·코딩의 최종 확정과 제출 책임은 PV 담당자에게 있습니다.")
    return "\n".join(lines)


def build_report(
    case_text: str,
    suspected_drug: str = "",
    reporter: str = "",
    patient_info: str = "",
    awareness_date: str = "",
) -> ReportDraft:
    """케이스 서술(+선택 필드)로 ICSR 초안을 조립한다.

    입력 텍스트는 이미 PII 마스킹된 상태를 전제한다(MCP 도구 계층에서 수행).
    """
    triage = assess_case(case_text, awareness_date)
    causality = assess_causality(case_text)
    coded = code_terms(case_text)
    candidates = suggest_candidates(case_text, coded)
    uncoded = flag_uncoded_expressions(case_text, coded, candidates)
    missing, fields = _check_minimum_criteria(
        case_text, suspected_drug, reporter, patient_info, coded, triage,
        candidates, uncoded,
    )

    # follow-up: 최소요건 누락 + 코딩 확정 대기 + 인과성 부족 정보를 한 목록으로
    followups = [f"최소보고요건 보완: {m}" for m in missing]
    followups += [
        f"코딩 확정: '{c.verbatim}' → {c.pt}({c.pt_en}) 후보 승인/기각 "
        "(LLT 참조 매칭 — 자동 확정 금지)"
        for c in candidates
    ]
    followups += [f"용어 코딩: 미코딩 증상 표현 '{u}' 에 PT 부여 필요" for u in uncoded]
    followups += [f"인과성 평가 보완: {q}" for q in causality.missing_info]

    draft = _render_markdown(
        case_text, fields, missing, triage, causality, coded, candidates, uncoded
    )
    return ReportDraft(
        reportable=not missing,
        missing=missing,
        triage=triage,
        causality=causality,
        coded_terms=coded,
        candidate_terms=candidates,
        uncoded_expressions=uncoded,
        draft_markdown=draft,
        followups=followups,
    )
