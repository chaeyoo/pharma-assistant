"""이상사례 용어 표준화 코딩 (MedDRA 방식의 경량 사전).

PV 케이스는 "숨쉬기 힘들다", "온몸에 두드러기" 같은 구어 서술(verbatim)로
들어오지만, 규제 보고(KAERS/E2B)는 표준 용어(MedDRA PT: Preferred Term)로
코딩해야 한다. 코딩이 흔들리면 같은 이상사례가 다른 용어로 흩어져
시그널 탐지(집계)가 무너진다 — PV 데이터 품질의 출발점이 코딩이다.

구현 원칙:
  - 실제 MedDRA는 유료 라이선스 용어집(수만 개 PT)이라 데모에 담을 수 없다.
    여기서는 '구어 → PT(+SOC 기관계 분류)' 매핑 구조와 코딩 파이프라인의
    자리를 소사전으로 증명하고, 실무에선 사전만 MedDRA 본체로 교체한다.
  - 질의확장 사전(rag/synonyms.py)과 같은 철학: 결정론적(감사 가능)·저지연.
    사전이 못 잡는 롱테일 표현은 LLM 보조 코딩 후보 제시 → 사람 확정으로 확장.
  - 같은 PT가 여러 표현으로 감지되면 1건으로 dedupe한다(집계 왜곡 방지).

3계층 구조 (확정/후보/미코딩 — 실무 MedDRA 오토코딩의 축소판):
  1계층 code_terms      : 사람이 검수한 '구어 키워드 → PT' 소사전. 매칭은 자동
                          확정(정밀도 무관용 — 오탐은 시그널 집계를 오염시킨다).
  2계층 suggest_candidates: LLT(Lowest Level Term) 스타일 참조 테이블. 검수
                          전의 넓은 매핑이라 자동 확정하지 않고 '후보'로만
                          제시한다(needs_confirmation) — 사람이 승인해야 확정.
                          검수를 통과한 항목은 1계층 사전으로 '승격'된다.
  3계층 flag_uncoded_expressions: 어느 사전에도 없는 증상 서술을 '미코딩
                          표현'으로 감지만 한다(PT 없음). 최소보고요건 ④
                          (이상사례 존재)는 '구체적 증상 서술이 있는가'이지
                          '코딩에 성공했는가'가 아니므로, 이 신호로 코딩
                          실패가 보고요건 미충족으로 연쇄되는 것을 끊는다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# (PT 한글, PT 영문, SOC 기관계, 감지 키워드) — MedDRA 스타일 소사전.
# 키워드는 케이스 자유 서술에서 해당 PT를 감지하는 데 쓴다(부분 일치).
_TERM_DICT: list[tuple[str, str, str, list[str]]] = [
    ("아나필락시스 반응", "Anaphylactic reaction", "면역계 장애", ["아나필락시스"]),
    ("두드러기", "Urticaria", "피부 및 피하조직 장애", ["두드러기", "담마진"]),
    ("발진", "Rash", "피부 및 피하조직 장애", ["발진"]),
    ("소양증", "Pruritus", "피부 및 피하조직 장애", ["가려움", "소양감", "가려워"]),
    ("호흡곤란", "Dyspnoea", "호흡기, 흉곽 및 종격 장애", ["호흡곤란", "숨쉬기 힘", "숨이 차", "숨을 쉬기 어려"]),
    ("두통", "Headache", "신경계 장애", ["두통", "머리가 아프", "머리가 아파"]),
    ("어지러움", "Dizziness", "신경계 장애", ["어지러움", "어지럼", "현기증", "어지러워"]),
    ("실신", "Syncope", "신경계 장애", ["실신", "의식을 잃", "의식 소실"]),
    # "발작"은 다의어라 1계층(자동 확정) 키워드에서 뺀다(v11) — "공황발작"(정신)·
    # "협심증 발작"(심장)·"천식 발작"(호흡기)은 발작(seizure)이 아닌 발작
    # (attack/paroxysm)인데, 부분 일치가 이를 신경계 경련으로 자동 확정해 SOC 를
    # 가로질러 시그널 집계를 오염시켰다(위경련≠경련·황달≠간손상과 같은 계열).
    # 접두 배제 열거는 다음 질환명에서 재발하므로(패턴5), "발작"은 2계층 후보로
    # 내려 사람이 seizure 여부를 확정하게 한다(_LLT_REFERENCE). "경련"은 유지.
    ("경련", "Seizure", "신경계 장애", ["경련"]),
    ("오심", "Nausea", "위장관 장애", ["오심", "메스꺼움", "메스꺼워", "메스껍", "구역감"]),
    ("구토", "Vomiting", "위장관 장애", ["구토", "토했", "토를 하"]),
    ("설사", "Diarrhoea", "위장관 장애", ["설사"]),
    ("발열", "Pyrexia", "전신 장애 및 투여부위 병태", ["발열", "고열", "열이 나", "열이 났"]),
    ("안면부종", "Face oedema", "전신 장애 및 투여부위 병태", ["얼굴이 붓", "안면부종", "얼굴 부종", "입술이 붓"]),
    ("저혈압", "Hypotension", "혈관 장애", ["저혈압", "혈압이 떨어", "혈압 저하"]),
    ("심정지", "Cardiac arrest", "심장 장애", ["심정지"]),
    ("간효소 상승", "Hepatic enzyme increased", "임상검사", ["간수치 상승", "간수치가 올", "ast 상승", "alt 상승", "간효소"]),
    ("간손상", "Liver injury", "간담도 장애", ["간손상", "간독성"]),
    # 황달은 독립 PT(v8) — MedDRA 에서 Jaundice 는 간담도 장애의 독립 PT 이고,
    # 용혈성·폐쇄성 황달은 간손상이 아니다. '검수 사전=자동 확정' 계층에
    # 오매핑이 있으면 집계가 구조적으로 오염되므로 분리한다.
    ("황달", "Jaundice", "간담도 장애", ["황달"]),
]


# LLT(Lowest Level Term) 스타일 참조 테이블 — 2계층(후보 제시) 전용.
# 실제 MedDRA에서 "청력 상실" 같은 하위 표현(LLT)은 PT "난청"으로 묶인다.
# 이 테이블은 '기계 적재된 참조본'이라는 설정: 1계층 소사전과 달리 항목별
# 사람 검수를 거치지 않았으므로 자동 확정하지 않고 후보로만 낸다.
# (LLT 표현, PT 한글, PT 영문, SOC)
_LLT_REFERENCE: list[tuple[str, str, str, str]] = [
    ("저혈당", "저혈당", "Hypoglycaemia", "대사 및 영양 장애"),
    ("혈당이 떨어", "저혈당", "Hypoglycaemia", "대사 및 영양 장애"),
    ("청력 상실", "난청", "Deafness", "귀 및 미로 장애"),
    ("청력 저하", "난청", "Deafness", "귀 및 미로 장애"),
    ("귀가 안 들리", "난청", "Deafness", "귀 및 미로 장애"),
    ("울렁거리", "오심", "Nausea", "위장관 장애"),
    ("속이 울렁", "오심", "Nausea", "위장관 장애"),
    ("이명", "이명", "Tinnitus", "귀 및 미로 장애"),
    ("귀에서 소리", "이명", "Tinnitus", "귀 및 미로 장애"),
    ("불면", "불면증", "Insomnia", "정신 장애"),
    ("잠이 오지 않", "불면증", "Insomnia", "정신 장애"),
    ("탈모", "탈모증", "Alopecia", "피부 및 피하조직 장애"),
    # "발작"은 seizure(경련)일 수도, attack/paroxysm(공황·협심증·천식 발작)일
    # 수도 있어 자동 확정 불가 — 후보로 제시하고 사람이 신경계 발작인지 확정(v11).
    ("발작", "경련", "Seizure", "신경계 장애"),
]

# 3계층: 어느 사전에도 없는 '증상 서술' 감지 패턴 — 신체 감각을 구체적으로
# 서술하는 표현만 좁게 매칭한다. "몸이 좋지 않다" 같은 막연한 서술은 일부러
# 잡지 않는다 — ICH E2D의 ④요소는 '구체적(specific) 이상사례'를 요구하므로,
# 막연한 호소만 있는 케이스는 ④ 미충족(보완 요청)이 올바른 판정이다.
_UNCODED_SYMPTOM_RE = re.compile(
    r"[가-힣]*(?:저릿|저리|쑤시|욱신|따끔|화끈|뻐근|결리|시큰)[가-힣]*"
)

# 1계층 키워드의 '전방 합성어' 배제(v9) — "위경련"(위장관 경련성 복통)·
# "근육경련"(근경련)의 '경련'은 신경계 발작(Seizure)이 아닌데, 부분 일치는
# 이를 자동 '확정'해 버린다. 1계층은 자동 확정 계층이라 오매핑이 곧 시그널
# 집계의 구조적 오염이다(황달≠간손상 분리와 같은 원칙). 키워드별로 앞에
# 붙으면 다른 의미가 확정되는 접두 어절을 닫힌 목록으로 배제한다.
_FALSE_COMPOUND_PREFIXES: dict[str, tuple[str, ...]] = {
    "경련": ("위", "근육"),
}


def _keyword_position(low: str, keyword: str) -> int:
    """키워드의 첫 유효 출현 위치 — 배제 접두가 붙은 출현은 건너뛴다. 없으면 -1."""
    prefixes = _FALSE_COMPOUND_PREFIXES.get(keyword, ())
    start = 0
    while (idx := low.find(keyword, start)) >= 0:
        if not any(low[max(0, idx - len(p)) : idx] == p for p in prefixes):
            return idx
        start = idx + 1
    return -1


@dataclass
class CodedTerm:
    verbatim: str      # 서술에서 감지된 원 표현(첫 매칭 키워드)
    pt: str            # 표준 용어(PT) 한글
    pt_en: str         # PT 영문 (KAERS/E2B 국제 보고용)
    soc: str           # 기관계 대분류(SOC)

    def as_dict(self) -> dict:
        return {"verbatim": self.verbatim, "pt": self.pt, "pt_en": self.pt_en, "soc": self.soc}


def code_terms(case_text: str) -> list[CodedTerm]:
    """케이스 서술에서 이상사례 표현을 감지해 표준 용어(PT)로 코딩한다.

    같은 PT는 1건으로 dedupe하고, 서술에 등장한 순서를 유지한다
    (보고서의 '주요 이상사례'가 서술 흐름과 일치하게).
    """
    low = case_text.lower()
    hits: list[tuple[int, CodedTerm]] = []
    seen_pt: set[str] = set()
    for pt, pt_en, soc, keywords in _TERM_DICT:
        if pt in seen_pt:
            continue
        positions = [
            (p, k) for k in keywords if (p := _keyword_position(low, k.lower())) >= 0
        ]
        if not positions:
            continue
        pos, keyword = min(positions)
        seen_pt.add(pt)
        hits.append((pos, CodedTerm(verbatim=keyword, pt=pt, pt_en=pt_en, soc=soc)))
    hits.sort(key=lambda h: h[0])
    return [t for _, t in hits]


@dataclass
class CandidateTerm:
    """2계층(LLT 참조) 매칭 결과 — 사람 확정 전까지는 '후보'다.

    확정 코드(CodedTerm)와 타입을 분리한 이유: 후보가 확정 코드 목록에
    섞여 들어가 시그널 집계에 잡히는 실수를 타입 수준에서 막기 위해서다.
    """
    verbatim: str      # 서술에서 감지된 LLT 표현
    pt: str            # 제안 PT 한글
    pt_en: str         # 제안 PT 영문
    soc: str           # SOC 기관계
    needs_confirmation: bool = True   # 항상 True — 자동 확정 금지

    def as_dict(self) -> dict:
        return {
            "verbatim": self.verbatim, "pt": self.pt, "pt_en": self.pt_en,
            "soc": self.soc, "needs_confirmation": self.needs_confirmation,
        }


def suggest_candidates(case_text: str, confirmed: list[CodedTerm]) -> list[CandidateTerm]:
    """1계층 사전이 놓친 표현을 LLT 참조 테이블로 '후보 제시'한다.

    - 이미 확정된 PT는 제외한다(같은 사례의 이중 집계 방지).
    - 같은 PT 후보는 1건으로 dedupe, 서술 등장 순서 유지.
    - 자동 확정하지 않는다: 잘못 붙은 코드는 집계를 오염시키므로,
      검수 안 된 매핑은 '승인/기각' 결정을 사람에게 넘긴다. 승인된 매핑을
      1계층 사전에 승격하는 것이 사전 성장의 운영 루프다.
    """
    low = case_text.lower()
    confirmed_pts = {t.pt for t in confirmed}
    hits: list[tuple[int, CandidateTerm]] = []
    seen_pt: set[str] = set()
    for llt, pt, pt_en, soc in _LLT_REFERENCE:
        if pt in confirmed_pts or pt in seen_pt:
            continue
        pos = low.find(llt.lower())
        if pos < 0:
            continue
        seen_pt.add(pt)
        hits.append((pos, CandidateTerm(verbatim=llt, pt=pt, pt_en=pt_en, soc=soc)))
    hits.sort(key=lambda h: h[0])
    return [t for _, t in hits]


def symptom_keywords() -> tuple[str, ...]:
    """사전이 아는 증상 표면형 전체 — 라우팅 등 '감지' 용도 (v8).

    오프라인 라우터(agent._route_intent)가 케이스 서술을 PV 도구로 보낼지
    판단할 때 이 사전과 같은 어휘를 쓰게 해, '코딩 사전은 아는데 라우팅
    마커는 모르는' 어휘 불일치를 구조적으로 없앤다 — 마커가 중대 어휘
    (사망·입원 등)뿐이면 "두드러기" 같은 일반 증상 케이스가 검색으로 빠져
    회피 응답이 된다(가이드 3-2 예시가 실제로 실패하던 경로). 사전에 항목을
    추가하면 라우팅 어휘도 자동으로 따라온다.
    """
    words: list[str] = []
    for _pt, _pt_en, _soc, synonyms in _TERM_DICT:
        words.extend(synonyms)
    words.extend(llt for llt, _pt, _pt_en, _soc in _LLT_REFERENCE)
    return tuple(dict.fromkeys(words))


def flag_uncoded_expressions(
    case_text: str, coded: list[CodedTerm], candidates: list[CandidateTerm]
) -> list[str]:
    """어느 사전에도 안 잡힌 '구체적 증상 서술'을 미코딩 표현으로 감지한다.

    PT를 붙이지 않는다(3계층은 감지만) — '무엇인지 모르지만 증상 서술이
    있다'는 신호로, 최소보고요건 ④ 판정과 코딩 follow-up 질문에 쓰인다.
    이미 코딩(확정/후보)된 표현과 겹치는 매칭은 제외한다.
    """
    known = {t.verbatim for t in coded} | {c.verbatim for c in candidates}
    out: list[str] = []
    for m in _UNCODED_SYMPTOM_RE.finditer(case_text):
        expr = m.group(0)
        if any(k in expr or expr in k for k in known):
            continue
        if expr not in out:
            out.append(expr)
    return out
