"""답변 사후 검증(post-generation verification) — 생성 경계의 마지막 관문.

검색 평가(evaluate)·신뢰성 평가(faithfulness)는 **오프라인 품질 측정**이다.
운영에서 LLM 모드의 답변은 매 요청마다 새로 생성되는데, 그 답변 속 수치가
근거와 일치하는지 평가셋은 보증해 주지 않는다. 규제 도메인에서 답변 속
"15일"이 "30일"로 바뀌는 오류는 문장이 아무리 유창해도 컴플라이언스 사고다.

그래서 **모든 응답에 대해 런타임으로 실행되는 결정론적 검증 계층**을 둔다:

  1. 수치 클레임 검증 — 답변에서 '숫자+단위'(15일, 6개월, 120 근무일, 90% …),
     고유어 수량 표현(보름, 이틀, 한 달 …), 날짜(YYYY-MM-DD·"2026년 7월 25일"·
     연도 없는 부분 표기 "7월 25일")를 추출해, 각각이 **신뢰 소스(trusted
     sources)** 안에 실제로 존재하는지 대조한다. 부분 날짜는 월-일 접미로,
     연도 단독 표기("2025년")는 완전한 날짜의 연도 성분으로 대조한다 — 둘 다
     값이 같은 표기 변형이지 환산이 아니다(단위 엄격성과 충돌하지 않음).
  2. 방향 한정어 검증 — 수치가 근거에 있어도 **한정어의 방향이 뒤집히면**
     ("15일 이내" → "15일 이후") 별도 경고를 낸다. 방향 한정어는
     닫힌 어휘 집합(이내·이하·미만·까지 / 이상·이후·초과)이라 기계 검증이
     가능하다 — '관계 왜곡은 전부 LLM judge 몫'이라는 초기 경계 설정을
     재심사해 결정론으로 끌어온 부분이다. 같은 검증을 **날짜에도**
     ("2026-07-25까지" → "2026-07-25 이후"), **고유어 수사에도**("보름 이후")
     대칭 적용한다 — 방향 검증이 일부 표기에만 있으면, 같은 기한 왜곡이
     표기에 따라 한쪽만 잡히는 축 간 비대칭(사각지대)이 된다. 방향·역할
     충돌의 **판정 기준은 strict 계층(규정 근거·도구 출력)** 이다 — 케이스
     서술까지 합쳐 판정하면 케이스의 "15일 이후 증상"이 규정의 "15일 이내"
     방향 경고를 조용히 무력화한다(케이스에 같은 방향이 있으면 경고를 끄는
     대신 from_case 라벨로 모호성을 가시화한다).
  3. 날짜 역할 검증 — 도구 출력에 날짜가 여러 개면(인지일·마감일·오늘)
     답변이 두 날짜의 **역할을 맞바꿔도**("보고 기한은 <인지일>입니다") 각
     날짜가 신뢰 소스에 존재하므로 존재 대조(1)는 통과한다. 결정론적 도구는
     날짜를 역할 키(deadline_date·due_date·awareness_date)로 라벨링해
     출력하므로, 답변에서 역할 키워드(기한/마감·인지일)에 직접 붙은 날짜를
     그 역할의 라벨 집합과 대조한다 — 방향 한정어와 같은 원리로, 닫힌
     키워드·라벨 기반이라 기계 검증이 가능한 축이다.
  4. 인용 버전 검증 — 답변의 출처(citation)에 폐지(superseded)된 문서가
     섞였는지 확인한다(이력 조회를 명시하지 않았다면 그 자체가 결함).

신뢰 소스의 정의가 이 모듈의 핵심 설계다:
  신뢰 소스 = 검색된 근거 문단 ∪ 결정론적 도구의 출력 − 질문 에코
  근거 문단만 보면 도구가 '계산해 낸' 값(예: 인지일+15일=마감일)이 근거
  원문에 없다는 이유로 오탐된다. 규칙 기반 도구의 출력은 테스트로 검증된
  결정론이므로 근거와 같은 신뢰 등급으로 취급한다.
  반대로 도구 출력에 에코된 **사용자 질의(query)는 신뢰 소스에서 제외**한다
  — 포함하면 사용자가 틀린 수치를 전제로 물었을 때 모델이 맞장구쳐도
  통과하는 구멍이 생긴다(전제의 승격). 대신 질문에 있던 수치가 미확인으로
  판정되면 `from_question` 라벨로 구분해, 경고 문구가 '환각'이 아니라
  '전제 확인 필요'를 가리키게 한다(부정·정정 맥락의 오탐 완화).
  케이스 서술 에코(case)는 **2계층**이다: 지지 근거로는 인정하되(케이스
  재서술은 정당 — 경고하면 초안마다 오탐), 그 계층에서만 지지되는 클레임은
  `from_case` 라벨로 노출한다. "케이스는 사실"에서 "케이스가 규정 클레임의
  근거"로 건너뛰는 비약(사용자 서술의 승격)을 차단이 아니라 가시화로 다룬다
  — 기계는 '케이스 재서술'과 '케이스 수치가 우연히 규정 클레임을 지지'를
  구분할 수 없기 때문이다(구분은 사람의 몫, 라벨은 그 판단을 빠르게).
  인터넷 검색 결과(web_texts)는 **3번째 계층**이다: 사용자가 명시적으로
  요청한 웹 검색의 결과를 답변이 재서술하는 것은 정당하므로 지지 근거로
  인정하되, 그 계층에서'만' 지지되는 클레임은 `from_web` 라벨로 노출한다 —
  웹 결과가 사내 규정 근거로 조용히 승격되는 것(명시와 분리 원칙의 위반)을
  게이트를 끄는 대신 라벨로 가시화한다. 검증 게이트 자체는 웹 답변에도
  그대로 돈다: 어느 계층에도 없는 수치는 여전히 '미확인' 경고다. 방향·역할
  축의 판정 기준은 종전대로 strict 계층뿐이다(웹 계층은 존재 축에만 참여).

단위의 엄격성:
  '근무일'과 '일'은 **다른 단위**다 — "120 근무일"을 "120일"로 옮기면 실제
  달력 기한이 달라진다. '주(週)' 환산("15일"→"약 2주")과 마찬가지로, 근거에
  없는 단위 환산은 '지원되지 않는 클레임'으로 취급한다. 고유어 수사는 값이
  정확히 같은 표기 변형(보름=15일)만 사전으로 동치 처리한다 — 환산이 아니라
  표기 정규화이므로 엄격성과 충돌하지 않는다.

실패 방향의 설계:
  검증 실패 시 답변을 차단하지 않고 **경고를 부착**한다. 검증기 자신도
  오탐 가능성이 있고(패러프레이즈 등), 이 도구의 원칙은 '사람의 최종
  확정을 빠르게'이지 자동 차단이 아니다. 단, 경고는 조용히 숨기지 않고
  답변 본문·API 필드·UI 배지에 모두 노출한다(시끄러운 실패). 지원된
  클레임에는 근거 위치 스니펫(evidence)을 붙여 사람이 즉시 대조하게 한다.

왜 LLM 재검증(self-critique)이 아니라 규칙인가:
  검증자가 확률적이면 '검증의 검증'이 다시 필요해지는 순환이 생긴다.
  결정론적 검증은 같은 답변에 항상 같은 판정을 내려 감사 가능하고,
  그 자체를 평가셋(eval/verify_eval.py)으로 측정할 수 있다.

이 계층이 잡지 **않는** 것(경계의 명시):
  부정문("30일이 아니라 15일"), 주어 바꿔치기, 수치 없는 의미 왜곡은
  결정론 규칙의 범위 밖이다 — opt-in 2차 계층(llm_judge.py)이 담당한다.
  2차 계층은 기본 비활성·비차단이며, 이 파일의 결정론 축 정본(WARNING_AXES
  등)에 **등록되지 않는다** — 확률적 판정을 결정론 축과 같은 정본에 섞으면
  '축 = 자가 테스트 강제'라는 구조 계약이 키 없는 부팅에서 깨진다
  (경계와 이원화의 근거는 llm_judge.py docstring 참고).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# 단위 목록: 긴 단위가 먼저 와야 한다 ('근무일'이 '일'보다 먼저 매칭되도록).
# '근무일'을 별도 단위로 두는 이유: 코퍼스의 핵심 처리기한(120·75·60 근무일…)이
# '일'만 인식하는 정규식에는 아예 안 잡혀 검증 사각지대였다 — 그리고
# 근무일→역일 환산은 실제 달력 기한을 바꾸는 오류라 단위를 구분해 대조해야 한다.
# '주(週)'의 lookahead: LLM이 "15일"을 "약 2주"로 패러프레이즈하면 근거에 없는
# 환산값이 생기고, 마감일 환산 오차는 그 자체가 리스크다.
# lookahead 는 전면 한글 배제(`(?![가-힣])`)가 아니라 **비단위 합성어만** 배제한다
# (v9): 한국어에서 '주'는 조사·어미·'간(間)'이 직결되는 표기("2주간 보관",
# "약 2주입니다", "2주로 연장")가 가장 흔한데, 전면 배제는 그 표기의 수집을
# 통째로 차단했다 — "15일 → 약 2주간" 환산 위조가 그 표기로만 오면 조용히
# 통과하고, 근거가 "2주간"이라 쓰면 옳은 답변 "2주"에 미확인 오탐이 붙는
# 양방향 결함(\b 가 한글 직결에서 경계가 아니던 v6 교훈이 검증기 자신의 단위
# 표기에는 미전파된 형태). '주년·주기·주차·주말·주택·주주'만 좁게 배제한다.
# '달'(고유어 개월)·'퍼센트'(% 철자 표기)도 같은 계열의 미수집 표기라 단위로
# 수집하고 canonical(개월·%)로 정규화한다 — 환산이 아니라 표기 변형이다.
_UNIT_ALT = r"근무일|영업일|개월|퍼센트|주일|시간|일|년|주(?!년|기|차|말|택|주)|달(?!러)|회|세|%"
# 표기 변형 단위 → canonical 단위 ("2주일"="2주", "6달"="6개월", "90 퍼센트"="90%").
_UNIT_ALIASES = {"주일": "주", "달": "개월", "퍼센트": "%"}


def _norm_unit(unit: str) -> str:
    """단위 표기 변형을 canonical 로 접는다 — 모든 수집 지점이 이 한 함수를
    거쳐야 답변·근거·질문의 대조가 표기와 무관하게 대칭이 된다."""
    return _UNIT_ALIASES.get(unit, unit)


# 선행 룩비하인드 (?<![\d.\-]) : 날짜에 조사처럼 '일'이 붙은 표기("2026-07-25일이다")
# 에서 '25일'이 기간 클레임으로 오추출되는 오염을 막는다 — 근거 쪽에서 오추출되면
# 답변의 지어낸 '25일 기한'이 그 오염된 값에 지지되어 통과한다(한국어 날짜
# 표기를 _normalize 로 접는 것과 같은 계열의 구멍인데, ISO+접미 표기는 정규화
# 대상이 아니라 정규식 경계로 막아야 한다).
_NUM_UNIT_RE = re.compile(rf"(?<![\d.\-])(\d+(?:\.\d+)?)\s*({_UNIT_ALT})")
# 날짜 경계에 \b 를 쓰면 안 된다 — 한글도 \w 라서 "2026-07-25입니다"처럼 조사가
# 바로 붙는(한국어에서 가장 흔한) 표기의 날짜가 통째로 추출을 벗어난다.
# 검증 대칭성 덕에 조용히 지나가던 사각지대: 답변·근거 양쪽에서 똑같이 안
# 잡히므로 오탐이 없어 아무 신호도 없었다 — '검증할 클레임이 아예 수집되지
# 않는' 실패는 통과처럼 보인다. 숫자·하이픈 연속만 경계로 배제한다.
_DATE_RE = re.compile(r"(?<![\d-])(\d{4}-\d{2}-\d{2})(?![\d-])")
# 한국어 날짜 표기("2026년 7월 25일") — _normalize 에서 ISO 로 정규화한다.
_KDATE_RE = re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일")
# 연도 없는 월-일 표기("7월 25일") — ISO 로 접을 수 없으므로 반복 날짜 표기
# (ISO 8601 --MM-DD)로 정규화한다. 이 정규화가 없으면 **이중 결함**이 된다:
# (1) '25일' 성분이 기간 클레임으로 오추출되어, 옳은 답변("기한은 7월 25일")에
#     '25일 미확인' 오탐이 붙고(alert fatigue), 근거에 우연히 기간 '25일'이
#     있으면 틀린 날짜가 그 값에 지지되어 통과한다(오염).
# (2) 부분 날짜 클레임 자체는 아예 추출되지 않는다 — 수집되지 않는 클레임은
#     검증 사각지대다(조용한 비검증은 통과처럼 보인다).
# 검증은 월-일 접미(suffix) 대조로 한다: "--07-25" 는 신뢰 소스의 완전한 날짜
# "2026-07-25" 의 월-일 성분과 값이 같은 **표기 변형**이다(환산 아님 — 연도
# 성분을 지어내지 않으므로 단위 엄격성과 충돌하지 않는다).
_PARTIAL_KDATE_RE = re.compile(r"(?<!\d)(\d{1,2})\s*월\s*(\d{1,2})\s*일")
_PARTIAL_DATE_RE = re.compile(r"--(\d{2}-\d{2})(?![\d-])")
# 범위 표기("15~30일")의 하한 — 주 정규식은 상한(30일)만 잡아 하한이 검증을
# 벗어난다. 구분자에 '-'를 넣지 않는 이유: 날짜(2026-07-25)와 충돌한다.
_RANGE_RE = re.compile(rf"(?<![\d.\-])(\d+(?:\.\d+)?)\s*[~∼〜–—]\s*\d+(?:\.\d+)?\s*({_UNIT_ALT})")
# 하이픈 범위("10-15일")는 상한·하한이 **모두** 미수집이었다(v8) — '-' 를
# 구분자에서 뺀 대가로 하한은 안 잡히고, 상한("15일")마저 앞이 '-' 라
# _NUM_UNIT_RE 의 룩비하인드에 걸려 표현 전체가 사각지대가 됐다(위조된
# 하한을 포함해 전부 조용한 통과). 날짜(4-2-2 자릿수)와는 '각 변이 1~3자리
# + 앞에 숫자·하이픈 금지'로 구분한다: "2026-07-25일"의 "07-25일"은 앞이
# '-' 라 배제되고, "10-15일"은 앞이 공백이라 잡힌다. 두 변을 모두 수집한다.
_HYPHEN_RANGE_RE = re.compile(
    rf"(?<![\d.\-])(\d{{1,3}}(?:\.\d+)?)-(\d{{1,3}}(?:\.\d+)?)\s*({_UNIT_ALT})"
)

# 고유어 수량 표현 → 허용 canonical (값, 단위) 형태들.
# '환산'이 아니라 값이 정확히 같은 '표기 변형'만 담는다(보름=15일).
# 사전이 보수적인 이유: 오탐 없는 어휘만 넣는다 — '하루'는 "하루빨리" 같은
# 관용구 오탐 위험이 있어 제외(코딩 사전과 같은 철학: 확신 없는 항목은 안 넣는다).
_NATIVE_NUMERALS: dict[str, tuple[tuple[str, str], ...]] = {
    "이틀": (("2", "일"),),
    "사흘": (("3", "일"),),
    "나흘": (("4", "일"),),
    "닷새": (("5", "일"),),
    "열흘": (("10", "일"),),
    "보름": (("15", "일"),),
    "일주일": (("7", "일"), ("1", "주")),
    "한 달": (("1", "개월"),),
    "한달": (("1", "개월"),),
    "두 달": (("2", "개월"),),
    "두달": (("2", "개월"),),
    "석 달": (("3", "개월"),),
    "세 달": (("3", "개월"),),
    "반년": (("6", "개월"),),
    "반 년": (("6", "개월"),),
}
_NATIVE_RE = re.compile("|".join(sorted((re.escape(w) for w in _NATIVE_NUMERALS), key=len, reverse=True)))

# 방향 한정어 — 닫힌 어휘 집합이라 결정론적 대조가 가능하다.
# 수치+단위 바로 뒤(마크다운 강조 등 브리지 문자 허용)에 붙은 경우만 본다.
# '이전'·'부터'(v8): '이전'은 '이내/까지'와 같은 급의 고빈도 상한 어휘인데
# 빠져 있어 "15일 이전"↔"15일 이후" 뒤집기가 어휘 하나 차이로 통과했다.
# '부터'는 기산점(하한) — "2026-07-25까지"를 "…부터"로 뒤집는 왜곡을 잡는다.
_UPPER_WORDS = ("이내", "안에", "이하", "미만", "까지", "이전")
_LOWER_WORDS = ("이상", "이후", "초과", "경과", "부터")
_QUAL_RE = re.compile(
    rf"(?<![\d.\-])(\d+(?:\.\d+)?)\s*({_UNIT_ALT})[\s*_)\]】'\"”]*({'|'.join(_UPPER_WORDS + _LOWER_WORDS)})"
)
# 고유어 수사에 붙은 방향 한정어("보름 이내") — 존재 대조는 고유어를 canonical
# (15, 일)로 접어 대칭 처리하면서 방향 대조는 숫자 표기(_QUAL_RE)에만 있으면,
# "보름 이후"라는 방향 뒤집기가 존재 축(보름=15일, 근거에 실존)을 통과하고
# 방향 축(숫자 없음, 미수집)도 지나친다 — 표기에 따라 같은 왜곡이 한쪽만
# 잡히는 축 간 비대칭(날짜 방향 축을 추가했던 것과 동일한 원리의 사각지대).
_NATIVE_QUAL_RE = re.compile(
    rf"({_NATIVE_RE.pattern})[\s*_)\]】'\"”]*({'|'.join(_UPPER_WORDS + _LOWER_WORDS)})"
)
# 날짜에 붙는 방향 한정어("2026-07-25까지" ↔ "2026-07-25 이후") — 방향 검증이
# 수치에만 있고 날짜에는 없는 것은 축 간 비대칭이었다: 근거가 "…까지 제출"인
# 기한 날짜를 답변이 "… 이후 제출"로 뒤집어도 날짜 자체는 근거에 실존해
# 존재 대조를 통과한다. 수치 방향 뒤집기와 같은 등급의 오류가 표기(기간 vs
# 날짜)에 따라 한쪽만 잡히던 사각지대. 대조 규칙은 수치와 동일하게 보수적 —
# 근거가 그 날짜에 대해 '반대 방향만' 말할 때만 플래그한다.
_DATE_QUAL_RE = re.compile(
    rf"(?<![\d-])(\d{{4}}-\d{{2}}-\d{{2}})[\s*_)\]】'\"”]*({'|'.join(_UPPER_WORDS + _LOWER_WORDS)})"
)
# 부분 날짜(--MM-DD 정규화 표기)에 붙는 방향 한정어(v8) — 존재 축은 접미
# 대조로 부분 표기를 지지해 주면서 방향 축은 완전한 ISO 만 수집하면,
# "7월 25일 이후"라는 뒤집기가 존재 축(월-일 성분 일치)의 지지를 받은 채
# 방향 축을 통째로 우회한다 — '지지된 왜곡'은 미수집보다 나쁘다. 이 파일이
# 스스로 제거했다고 선언한 "같은 왜곡이 표기에 따라 한쪽만 잡히는 축 간
# 비대칭"이 부분 날짜 표기에 남아 있던 형태.
_PARTIAL_DATE_QUAL_RE = re.compile(
    rf"--(\d{{2}}-\d{{2}})[\s*_)\]】'\"”]*({'|'.join(_UPPER_WORDS + _LOWER_WORDS)})"
)
_OPPOSITE = {"상한": "하한", "하한": "상한"}

# 날짜 역할 대조 — 결정론적 도구의 역할 라벨(직렬화된 JSON 키) ↔ 답변의 역할 키워드.
# 답변 쪽은 키워드가 날짜에 '직접' 붙은 경우만 본다("기한: 2026-07-25",
# "보고 기한은 2026-07-25", "(인지일 2026-07-10") — 사이에 다른 단어가 끼면
# ("기한 규정은 2025-04-01 시행…") 역할 주장으로 보지 않는다. 보수성:
# 신뢰 소스에 해당 역할 라벨이 없으면 판단 근거가 없으므로 플래그하지 않고,
# 존재 대조를 통과한 날짜만 본다(미확인 날짜는 존재 대조 축이 먼저 잡는다).
_ROLE_BRIDGE = r"[은는이가]?\s*[:：]?[\s*_\"'(（]*"
_ROLE_LABEL_RE: dict[str, re.Pattern[str]] = {
    "기한": re.compile(r'"(?:deadline_date|due_date)"\s*:\s*"(\d{4}-\d{2}-\d{2})"'),
    "인지일": re.compile(r'"awareness_date"\s*:\s*"(\d{4}-\d{2}-\d{2})"'),
}
_ROLE_ANSWER_RE: dict[str, re.Pattern[str]] = {
    "기한": re.compile(rf"(?:기한|마감)일?{_ROLE_BRIDGE}(\d{{4}}-\d{{2}}-\d{{2}})"),
    "인지일": re.compile(rf"인지일{_ROLE_BRIDGE}(\d{{4}}-\d{{2}}-\d{{2}})"),
}
# 역할 대조의 부분 날짜 표기(v8) — "보고 기한은 7월 10일입니다 (인지일 7월
# 25일)"처럼 연도 없는 표기로 역할을 뒤바꾸면, 두 날짜 모두 접미 대조로
# 존재 축을 통과하면서 역할 축(완전한 ISO 만 매칭)은 아예 발화하지 않았다.
_ROLE_ANSWER_PARTIAL_RE: dict[str, re.Pattern[str]] = {
    "기한": re.compile(rf"(?:기한|마감)일?{_ROLE_BRIDGE}--(\d{{2}}-\d{{2}})"),
    "인지일": re.compile(rf"인지일{_ROLE_BRIDGE}--(\d{{2}}-\d{{2}})"),
}


def _qual_class(word: str) -> str:
    return "상한" if word in _UPPER_WORDS else "하한"


def _normalize(text: str) -> str:
    """추출 전 정규화 — 천단위 콤마 제거 + 한국어 날짜 표기의 ISO 정규화.

    "2026년 7월 25일"을 ISO(2026-07-25)로 접지 않으면 두 가지가 깨진다:
    (1) 답변·질문의 한국어 날짜가 근거의 ISO 날짜와 표기만 달라 대조를 벗어난다
        — 검증 사각지대이자, 질문 속 한국어 날짜가 from_question 라벨을 못 받아
        올바른 답변에 '환각' 경고가 붙는 오탐 경로.
    (2) 날짜의 일(日) 성분("…7월 25일")이 기간 클레임("25일")으로 오추출된다
        — 날짜 문맥의 수사를 기간으로 읽는 것은 단위 엄격성의 위반이다.
    환산이 아니라 표기 정규화이므로(보름=15일과 같은 철학) 엄격성과 충돌하지
    않는다. 답변·신뢰 소스·질문에 대칭 적용된다(모두 이 함수를 거친다).
    """
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    text = _KDATE_RE.sub(
        lambda m: f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}", text
    )

    # 연도 없는 월-일 표기("7월 25일" → "--07-25"): 완전한 K-날짜를 먼저 접은 뒤
    # 남은 부분 날짜만 잡는다. 달력상 불가능한 값(13월·32일)은 날짜가 아니므로
    # 건드리지 않는다(그대로 두면 기간 클레임으로 추출·대조된다 — 보수적).
    def _pd(m: re.Match[str]) -> str:
        mm, dd = int(m.group(1)), int(m.group(2))
        if not (1 <= mm <= 12 and 1 <= dd <= 31):
            return m.group(0)
        return f"--{mm:02d}-{dd:02d}"

    return _PARTIAL_KDATE_RE.sub(_pd, text)


@dataclass
class ClaimCheck:
    claim: str        # 정규화된 클레임 표기 (예: "15일", "보름", "2026-07-25", "15일 이후")
    kind: str         # "numeric" | "date" | "direction" | "role"
    supported: bool   # 신뢰 소스에서 확인됐는가 (direction/role 은 항상 False=충돌)
    evidence: str = ""        # 지원 시 신뢰 소스의 해당 위치 스니펫(사람 대조용)
    from_question: bool = False  # 미확인 수치가 사용자 질문에 있던 값인가(전제 에코)
    from_case: bool = False      # 지지 근거가 '사용자 케이스 서술'뿐인가(승격 라벨)
    from_web: bool = False       # 지지 근거가 '인터넷 검색 결과'뿐인가(분리 라벨)

    def as_dict(self) -> dict:
        return {
            "claim": self.claim,
            "kind": self.kind,
            "supported": self.supported,
            "evidence": self.evidence,
            "from_question": self.from_question,
            "from_case": self.from_case,
            "from_web": self.from_web,
        }


@dataclass
class VerificationResult:
    checks: list[ClaimCheck] = field(default_factory=list)
    superseded_cited: list[str] = field(default_factory=list)  # 폐지본 인용 doc_id

    @property
    def unsupported(self) -> list[str]:
        """근거에서 확인되지 않은 수치·날짜 클레임(방향·역할 충돌은 별도 축)."""
        return [c.claim for c in self.checks if not c.supported and c.kind in ("numeric", "date")]

    @property
    def direction_conflicts(self) -> list[str]:
        """수치는 근거에 있으나 한정어 방향이 뒤집힌 클레임."""
        return [c.claim for c in self.checks if c.kind == "direction"]

    @property
    def role_conflicts(self) -> list[str]:
        """날짜는 근거에 있으나 역할(기한↔인지일)이 도구 라벨과 어긋난 클레임."""
        return [c.claim for c in self.checks if c.kind == "role"]

    @property
    def question_origin(self) -> list[str]:
        """미확인 수치·날짜 클레임 중 사용자 질문에 있던 값(전제 에코 — 경고 문구를
        달리 한다). **numeric·date 로 한정**한다(v10): 방향·역할 축의 from_question
        완화는 warning_text 의 각 축 블록이 직접 처리하는데, kind 미필터이면 방향·
        역할 클레임(supported=False 라 항상 충돌)이 이 축으로 새어 '⚠ 전제 확인
        필요: 수치 …' 블록에 오분류·이중 경고되고 계기판 by_axis 도 이중 집계됐다."""
        return [
            c.claim for c in self.checks
            if not c.supported and c.from_question and c.kind in ("numeric", "date")
        ]

    @property
    def case_origin(self) -> list[str]:
        """케이스 서술 유래 신호(from_case)가 붙은 클레임 — 경고와 등급이 다르다.

        '케이스는 사실'이므로 답변이 케이스를 재서술하는 것은 정당하다(경고하면
        보고서 초안마다 오탐 — alert fatigue). 그러나 '사실'이라는 성질이 그 값을
        규정 클레임의 근거로 승격시키지는 않는다 — 케이스의 "30일간 복용"이
        답변의 "보고 기한 30일"을 지지하면 그것은 사용자 서술의 승격이다.
        기계는 두 경우를 구분할 수 없으므로 차단 대신 **라벨**로 노출한다:
        감사 로그·UI 가 '규정/도구 근거'와 '사용자 서술 근거'를 구분해 읽는다.

        존재 축(supported=True, 케이스 계층만 지지)만이 아니라 **방향 충돌의
        from_case**(케이스에 같은 방향 표현이 있어 '재서술 vs 왜곡'이 모호한
        경고)도 포함한다(v9) — 종전에는 supported=True 를 요구해 방향축 라벨이
        summary·계기판(case_labeled)·감사 로그 어디에도 집계되지 않았다:
        라벨은 죽어도 소리가 없다는 이 라벨 자신의 원칙이 방향축에는 적용되지
        않던 빈틈이다."""
        return [c.claim for c in self.checks if c.from_case]

    @property
    def web_origin(self) -> list[str]:
        """인터넷 검색 결과 계층에서'만' 지지된 클레임(from_web) — 분리 라벨.

        사용자가 명시적으로 요청한 웹 검색의 재서술은 정당하므로 경고하지
        않는다(경고하면 웹 검색 답변마다 오탐 — alert fatigue). 그러나 웹
        결과는 사내 규제문서 근거가 아니다 — 이 라벨이 없으면 웹 유래 수치가
        '수치 검증 통과' 배지 아래에서 사내 근거처럼 읽힌다(명시와 분리
        원칙이 답변 표시 계층에만 있고 검증 계층에는 없는 비대칭). 감사
        로그·UI 가 '규정/도구 근거'·'사용자 서술 근거'·'웹 근거'를 구분해
        읽도록 case_origin 과 같은 등급 라벨로 노출한다."""
        return [c.claim for c in self.checks if c.from_web]

    @property
    def ok(self) -> bool:
        # ok 는 '차단 축'(BLOCKING_AXES)에서 파생한다(v10) — 종전에는 축을 손으로
        # 나열해, 새 차단 축을 ok 에 배선하면서 WARNING_AXES 에 등록하지 않으면
        # warn_rate 만 오르고 by_axis 귀속·자가 테스트가 없는 축이 배포됐다
        # (7-7 패턴3 '정본 앵커'가 봉합하지 않은 마지막 결합). 정본을 한 곳
        # (BLOCKING_AXES)에 두고, BLOCKING⊆WARNING 은 메타테스트가 강제한다.
        return not any(getattr(self, axis) for axis in BLOCKING_AXES)

    def summary(self) -> dict:
        """API/UI 노출용 요약."""
        return {
            "ok": self.ok,
            "checked": len([c for c in self.checks if c.kind in ("numeric", "date")]),
            "unsupported": self.unsupported,
            "direction_conflicts": self.direction_conflicts,
            "role_conflicts": self.role_conflicts,
            "question_origin": self.question_origin,
            "case_origin": self.case_origin,
            "web_origin": self.web_origin,
            "superseded_cited": self.superseded_cited,
            "checks": [c.as_dict() for c in self.checks],
        }


# 검증 축의 정본 목록(v9) — 계기판(GateStats)과 preflight 자가 테스트의 축
# 우주는 이 목록에서 파생된다. 종전에는 observability.GateStats._AXES 라는
# 수동 복제 튜플이 앵커였는데, 그러면 검증기에 새 축을 추가하고 observability
# 를 안 건드리는 가장 개연성 높은 부분 변경에서 메타 검사가 통과한 채 자가
# 테스트 없는 축이 배포된다 — "새 축 추가 시 자가 테스트가 구조로 강제된다"
# 는 주장의 숨은 전제('새 축은 반드시 _AXES 에도 등록된다')가 검사되지 않는
# 형태였다. 정본을 축의 산지(검증기)에 두고, summary() 키와의 일치는
# 테스트(test_preflight)가 고정한다.
#   WARNING_AXES: 응답을 경고 상태로 만들거나 경고 문구를 바꾸는 축 —
#     preflight 자가 테스트는 이 전 축에 '심은 오류' 케이스를 가져야 한다.
#   LABEL_AXES: ok 를 유지한 채 등급만 구분하는 라벨 축 — 라벨은 죽어도
#     소리가 없으므로 역시 자가 테스트 커버리지를 강제한다.
WARNING_AXES = (
    "unsupported", "direction_conflicts", "role_conflicts",
    "question_origin", "superseded_cited",
)
LABEL_AXES = ("case_origin", "web_origin")
# ok 를 실패(False)시키는 '차단 축'의 정본(v10) — VerificationResult.ok 는 이
# 목록에서 파생한다. WARNING_AXES 의 부분집합이어야 하며(전제 라벨
# question_origin 은 경고 문구만 바꾸는 비차단 축이라 제외), 그 결합은
# test_verifier 의 메타테스트가 강제한다 — 새 차단 축을 추가하면 WARNING_AXES
# 등록(→ by_axis 집계·preflight 자가 테스트)이 구조로 강제된다.
BLOCKING_AXES = (
    "unsupported", "direction_conflicts", "role_conflicts", "superseded_cited",
)


@dataclass(frozen=True)
class _Occurrence:
    display: str                          # 답변 속 원문 표기
    forms: tuple[tuple[str, str], ...]    # 허용 canonical (값, 단위) — 고유어는 복수 가능
    kind: str                             # "numeric" | "date"


def _numeric_forms(text: str) -> set[tuple[str, str]]:
    """텍스트의 모든 수치 클레임을 canonical (값, 단위) 집합으로.

    수치는 (값, 단위) 튜플로 정규화한다 — "15일 이내"와 "15 일"이 같은
    클레임으로 대조되도록. 값은 불필요한 선행 0만 제거한다("03년"="3년").
    '주일' 단위는 '주'로 정규화한다("2주일"="2주").
    고유어 수사(보름 등)는 canonical 형태 전부를 집합에 더한다 — 추출이
    답변·신뢰 소스 양쪽에 대칭으로 적용되므로, 근거가 "보름"이라 쓰고
    답변이 "15일"이라 써도(또는 그 반대) 동치로 대조된다.
    """
    text = _normalize(text)
    forms = {
        (m.group(1).lstrip("0") or "0", _norm_unit(m.group(2)))
        for m in _NUM_UNIT_RE.finditer(text)
    }
    for m in _RANGE_RE.finditer(text):  # 범위 하한
        unit = _norm_unit(m.group(2))
        forms.add((m.group(1).lstrip("0") or "0", unit))
    for m in _HYPHEN_RANGE_RE.finditer(text):  # 하이픈 범위 — 두 변 모두(v8)
        unit = _norm_unit(m.group(3))
        forms.add((m.group(1).lstrip("0") or "0", unit))
        forms.add((m.group(2).lstrip("0") or "0", unit))
    for m in _NATIVE_RE.finditer(text):
        forms |= set(_NATIVE_NUMERALS[m.group(0)])
    return forms


def _qualifier_map(text: str) -> dict[tuple[str, str], set[str]]:
    """(값, 단위) → 그 수치에 붙어 등장한 방향 한정어 클래스 집합.

    고유어 수사 표기("보름 이내")도 canonical (15, 일)로 접어 수집한다 —
    존재 대조와 마찬가지로 방향 대조도 표기와 무관하게 대칭이어야 한다
    (근거가 "보름 이내"라 쓰고 답변이 "15일 이후"라 써도, 그 반대여도 잡힌다).
    """
    text = _normalize(text)
    out: dict[tuple[str, str], set[str]] = {}
    for m in _QUAL_RE.finditer(text):
        unit = _norm_unit(m.group(2))
        key = (m.group(1).lstrip("0") or "0", unit)
        out.setdefault(key, set()).add(_qual_class(m.group(3)))
    for m in _NATIVE_QUAL_RE.finditer(text):
        cls = _qual_class(m.group(2))
        for form in _NATIVE_NUMERALS[m.group(1)]:
            out.setdefault(form, set()).add(cls)
    return out


def _date_qualifier_map(text: str) -> dict[str, set[str]]:
    """날짜(ISO) → 그 날짜에 붙어 등장한 방향 한정어 클래스 집합.

    한국어 날짜 표기는 _normalize 가 먼저 ISO 로 접으므로("2026년 7월 25일까지"
    → "2026-07-25까지") 표기와 무관하게 대칭으로 수집된다.
    """
    text = _normalize(text)
    out: dict[str, set[str]] = {}
    for m in _DATE_QUAL_RE.finditer(text):
        out.setdefault(m.group(1), set()).add(_qual_class(m.group(2)))
    return out


def _partial_date_qualifier_map(text: str) -> dict[str, set[str]]:
    """월-일 접미("MM-DD") → 방향 한정어 클래스 집합 (v8).

    완전한 날짜의 한정어("2026-07-25까지")를 접미(07-25)로도 투영한다 —
    존재 축이 부분 표기를 완전한 날짜의 월-일 성분으로 지지하는 것과 같은
    대칭: 답변이 "7월 25일 이후"라 쓰고 근거가 "2026-07-25까지"라 써도,
    그 반대여도 방향 대조가 성립해야 한다.
    """
    text = _normalize(text)
    out: dict[str, set[str]] = {}
    for m in _DATE_QUAL_RE.finditer(text):
        out.setdefault(m.group(1)[5:], set()).add(_qual_class(m.group(2)))
    for m in _PARTIAL_DATE_QUAL_RE.finditer(text):
        out.setdefault(m.group(1), set()).add(_qual_class(m.group(2)))
    return out


def extract_claims(text: str) -> tuple[set[tuple[str, str]], set[str]]:
    """텍스트에서 (수치, 단위) 클레임 집합과 날짜 집합을 추출한다."""
    return _numeric_forms(text), set(_DATE_RE.findall(_normalize(text)))


def _partial_dates(text: str) -> set[str]:
    """연도 없는 월-일 표기의 접미(suffix, "MM-DD") 집합을 추출한다."""
    return set(_PARTIAL_DATE_RE.findall(_normalize(text)))


def _occurrences(answer: str) -> list[_Occurrence]:
    """답변 쪽 추출 — 표기(display) 단위로 클레임을 나열한다(경고 문구용)."""
    answer = _normalize(answer)
    occs: list[_Occurrence] = []
    seen: set[str] = set()

    def _add(display: str, forms: tuple[tuple[str, str], ...], kind: str) -> None:
        if display not in seen:
            seen.add(display)
            occs.append(_Occurrence(display, forms, kind))

    for m in _NUM_UNIT_RE.finditer(answer):
        num = m.group(1).lstrip("0") or "0"
        unit = _norm_unit(m.group(2))
        _add(f"{num}{unit}", ((num, unit),), "numeric")
    for m in _RANGE_RE.finditer(answer):
        num = m.group(1).lstrip("0") or "0"
        unit = _norm_unit(m.group(2))
        _add(f"{num}{unit}", ((num, unit),), "numeric")
    for m in _HYPHEN_RANGE_RE.finditer(answer):  # 하이픈 범위 — 두 변을 별개 클레임으로(v8)
        unit = _norm_unit(m.group(3))
        for g in (m.group(1), m.group(2)):
            num = g.lstrip("0") or "0"
            _add(f"{num}{unit}", ((num, unit),), "numeric")
    for m in _NATIVE_RE.finditer(answer):
        _add(m.group(0), _NATIVE_NUMERALS[m.group(0)], "numeric")
    for d in _DATE_RE.findall(answer):
        _add(d, ((d, ""),), "date")
    for suffix in _PARTIAL_DATE_RE.findall(answer):
        mm, dd = suffix.split("-")
        # 표시는 사람이 읽는 원 표기로 되돌린다("--07-25" → "7월 25일")
        _add(f"{int(mm)}월 {int(dd)}일", ((suffix, ""),), "partial_date")
    return occs


def _snippet(trusted: str, form: tuple[str, str], kind: str, width: int = 34) -> str:
    """신뢰 소스에서 클레임이 등장한 위치의 스니펫 — 사람의 대조를 빠르게."""
    if kind == "date":
        pat = re.escape(form[0])
    elif kind == "partial_date":
        # 신뢰 소스의 완전한 날짜(YYYY-MM-DD) 또는 같은 부분 표기(--MM-DD)
        pat = rf"\d{{4}}-{re.escape(form[0])}|--{re.escape(form[0])}"
    else:
        num, unit = form
        # canonical 단위로 지지된 클레임의 스니펫은 신뢰 소스의 '표기 변형'
        # 원문에서도 찾아야 한다(근거가 "6달"이라 쓰고 답변이 "6개월"인 경우).
        unit_pat = {
            "주": "주일?",
            "개월": "(?:개월|달)",
            "%": "(?:%|퍼센트)",
        }.get(unit, re.escape(unit))
        pat = rf"(?<!\d)0*{re.escape(num)}\s*{unit_pat}"
    m = re.search(pat, trusted)
    if not m:  # 고유어 표기로 지원된 경우
        for word, canon in _NATIVE_NUMERALS.items():
            if form in canon:
                m = re.search(re.escape(word), trusted)
                if m:
                    break
    if not m:
        return ""
    lo, hi = max(0, m.start() - width), min(len(trusted), m.end() + width)
    return " ".join(trusted[lo:hi].split())


def verify_answer(
    answer: str,
    trusted_texts: list[str],
    citations: list[dict] | None = None,
    allow_superseded: bool = False,
    question: str = "",
    allowed_superseded_ids: set[str] | None = None,
    user_fact_texts: list[str] | None = None,
    web_texts: list[str] | None = None,
) -> VerificationResult:
    """답변의 수치·날짜 클레임과 방향 한정어를 신뢰 소스와 대조하고 인용 버전을 점검한다.

    Args:
        answer: 사용자에게 나가는 최종 답변 텍스트.
        trusted_texts: 신뢰 소스 — 검색 근거 문단 + 결정론적 도구 출력의
            직렬화 문자열(질문 에코 필드는 호출자가 제거). 이 밖에서 온
            수치는 전부 '미확인'으로 판정한다.
        citations: 답변에 부착된 출처 메타(status 포함 시 버전 점검).
        allow_superseded: 사용자가 명시적으로 이력(폐지본) 조회를 요청한
            경우 True — 폐지본 인용이 결함이 아니라 목적이 된다. 응답 전체가
            하나의 이력 검색에서 나온 경우(오프라인 라우터)에만 쓴다.
        question: 사용자 질문 원문. 미확인 수치가 질문에 있던 값이면
            from_question 으로 라벨링해 경고 문구를 '전제 확인'으로 조정한다
            (부정·정정 답변의 오탐 완화 — 신뢰하지도, 조용히 넘기지도 않는다).
        allowed_superseded_ids: 이력(as_of·include_superseded) 검색이 **실제로
            반환한** 문서의 doc_id 집합 — 이 문서들의 폐지본 인용만 허용한다.
            전역 bool 스위치는 한 턴에 이력 검색과 현행 검색이 섞일 때(LLM
            모드) 현행 검색 쪽에 상류 결함으로 섞여 든 폐지본 인용까지 경고를
            꺼 버린다 — 안전장치를 끄는 스위치의 면적은 근거가 성립하는
            범위('이력 조회가 그 문서를 반환했다')로 좁혀야 한다.
        user_fact_texts: 사용자 제공 사실(케이스 서술의 도구 에코 등) — 지지
            근거로는 인정하되(재서술은 정당 — 경고하면 초안마다 오탐),
            이 계층에서만 지지되는 클레임은 from_case 로 라벨링한다.
            "케이스는 사실"이 "케이스가 규정 클레임의 근거"로 비약하는 지점을
            차단 대신 가시화하는 2계층 신뢰 소스 설계다.
        web_texts: 인터넷 검색 결과(search_web 도구 출력) — 사용자가 명시적으로
            요청한 웹 검색의 재서술은 정당하므로 지지 근거로 인정하되, 이
            계층에서만 지지되는 클레임은 from_web 으로 라벨링한다(3계층).
            웹 결과를 trusted_texts 에 섞으면 사내 규정 근거로 조용히 승격되고
            (분리의 소멸), 아예 빼면 웹 검색 답변마다 '미확인' 오탐이 붙는다
            (alert fatigue) — 라벨 계층이 그 둘 사이의 유일한 정합 지점이다.
    """
    result = VerificationResult()

    strict = _normalize("\n".join(trusted_texts))
    facts = _normalize("\n".join(user_fact_texts)) if user_fact_texts else ""
    web = _normalize("\n".join(web_texts)) if web_texts else ""
    # base_facts(strict+케이스)가 종전의 '신뢰 소스', trusted 는 웹 계층까지의
    # 전체 결합 — 지지 판정·스니펫에 쓴다(방향·역할 판정 기준은 strict 유지).
    base_facts = f"{strict}\n{facts}" if facts else strict
    trusted = f"{base_facts}\n{web}" if web else base_facts
    src_nums, src_dates = extract_claims(trusted)
    # strict 계층(규정 근거·도구 출력)만의 클레임 — from_case 라벨의 기준선
    strict_nums, strict_dates = (
        extract_claims(strict) if (facts or web) else (src_nums, src_dates)
    )
    # strict+케이스 계층까지의 클레임 — from_web 라벨의 기준선(여기 없고
    # 전체 결합에만 있으면 지지 근거는 웹뿐이다)
    bf_nums, bf_dates = extract_claims(base_facts) if web else (src_nums, src_dates)
    # 부분 날짜(월-일) 접미 — 완전한 날짜의 월-일 성분도 지지 근거가 된다
    src_partials = _partial_dates(trusted) | {d[5:] for d in src_dates}
    strict_partials = (
        (_partial_dates(strict) | {d[5:] for d in strict_dates})
        if (facts or web)
        else src_partials
    )
    bf_partials = (
        (_partial_dates(base_facts) | {d[5:] for d in bf_dates}) if web else src_partials
    )
    # 방향·역할 대조의 기준은 strict 계층이다 — 케이스 서술(facts)까지 합친
    # 지도로 판정하면, 케이스의 "15일 이후 증상 발생"이 규정의 "15일 이내"
    # 방향 뒤집기 경고를 **조용히 무력화**한다(존재 축에는 2계층을 만들어 놓고
    # 방향 축은 단층으로 둔 비대칭). 반대로 케이스에만 있는 한정어가 충돌을
    # '만들어내는' 경로도 함께 사라진다(규정 근거 없는 판정 금지 — 보수성).
    strict_quals = _qualifier_map(strict)
    facts_quals = _qualifier_map(facts) if facts else {}
    q_nums, q_dates = extract_claims(question) if question else (set(), set())
    q_partials = (_partial_dates(question) | {d[5:] for d in q_dates}) if question else set()
    # 질문의 방향·역할 지도(v9) — from_question 완화는 존재 축에만 있었다:
    # 사용자의 틀린 전제("15일 이후에 하면 되나요?")를 **정정**하는 옳은 답변
    # ("15일 이후가 아니라 이내입니다")은 그 전제를 재서술할 수밖에 없는데,
    # 방향·역할 충돌에는 전제 라벨이 배선되지 않아 완화 없는 '컴플라이언스
    # 오류' 경고가 붙었다(축 × 완화 라벨 매트릭스의 빈 칸 — 옳은 답변에 붙는
    # 오탐은 미탐과 같은 등급의 결함이다). 존재 축과 동일한 원칙으로, 경고를
    # 끄지 않고 **종류만 조정**한다(from_question 라벨 + '전제 확인' 문구).
    q_quals = _qualifier_map(question) if question else {}
    q_date_quals = _date_qualifier_map(question) if question else {}
    q_partial_quals = _partial_date_qualifier_map(question) if question else {}
    norm_question = _normalize(question) if question else ""
    q_roles: dict[str, set[str]] = {
        role: {m.group(1) for m in rre.finditer(norm_question)}
        for role, rre in _ROLE_ANSWER_RE.items()
    } if question else {}
    q_roles_partial: dict[str, set[str]] = {
        role: {m.group(1) for m in rre.finditer(norm_question)}
        for role, rre in _ROLE_ANSWER_PARTIAL_RE.items()
    } if question else {}

    for occ in _occurrences(answer):
        kind = occ.kind
        # 계층 라벨의 공통 규칙 — strict 에 있으면 무라벨, strict 에는 없고
        # strict+케이스(base_facts)에 있으면 from_case, 거기에도 없이 전체
        # 결합(웹 포함)에서만 지지되면 from_web(계층 우선순위: 규정·도구 >
        # 케이스 서술 > 웹 — 상위 계층이 지지하면 하위 라벨은 붙지 않는다).
        from_web = False
        if occ.kind == "date":
            supported = occ.display in src_dates
            from_q = (not supported) and occ.display in q_dates
            from_case = supported and occ.display not in strict_dates and occ.display in bf_dates
            from_web = supported and occ.display not in bf_dates
            evidence = _snippet(trusted, (occ.display, ""), "date") if supported else ""
        elif occ.kind == "partial_date":
            # 부분 날짜(연도 없음)는 월-일 접미로 대조한다 — 완전한 날짜의
            # 성분 재서술은 값이 같은 표기 변형이지 환산이 아니다. 외부
            # 계약(kind)은 "date" 로 노출한다(부분/완전은 표기의 차이).
            suffix = occ.forms[0][0]
            supported = suffix in src_partials
            from_q = (not supported) and suffix in q_partials
            from_case = supported and suffix not in strict_partials and suffix in bf_partials
            from_web = supported and suffix not in bf_partials
            evidence = _snippet(trusted, (suffix, ""), "partial_date") if supported else ""
            kind = "date"
        else:
            hit = next((f for f in occ.forms if f in src_nums), None)
            supported = hit is not None
            evidence = _snippet(trusted, hit, "numeric") if hit else ""
            in_strict = any(f in strict_nums for f in occ.forms)
            in_bf = any(f in bf_nums for f in occ.forms)
            from_case = supported and not in_strict and in_bf
            from_web = supported and not in_bf
            if not supported:
                # 연도 단독 표기("2025년"): 신뢰 소스의 완전한 날짜(2025-04-01)의
                # 연도 성분과 값이 같으면 지지로 본다 — 표기 변형이지 환산이
                # 아니다. 이 폴백이 없으면 근거가 날짜로만 말하는 연도를 언급한
                # 옳은 답변마다 '미확인 수치' 오탐이 붙는다(alert fatigue 경로).
                num, unit = occ.forms[0]
                if unit == "년" and re.fullmatch(r"(19|20)\d{2}", num):
                    year_date = next((d for d in src_dates if d.startswith(num + "-")), "")
                    if year_date:
                        supported = True
                        evidence = _snippet(trusted, (year_date, ""), "date")
                        y_strict = (num, "년") in strict_nums or any(
                            d.startswith(num + "-") for d in strict_dates
                        )
                        y_bf = (num, "년") in bf_nums or any(
                            d.startswith(num + "-") for d in bf_dates
                        )
                        from_case = not y_strict and y_bf
                        from_web = not y_bf
            from_q = (not supported) and any(f in q_nums for f in occ.forms)
        result.checks.append(
            ClaimCheck(
                occ.display, kind, supported,
                evidence=evidence, from_question=from_q,
                from_case=from_case, from_web=from_web,
            )
        )

    # 방향 한정어 대조 — 수치가 지원된 클레임에 한해, 답변의 한정어 방향이
    # strict 계층(규정 근거·도구 출력)과 뒤집혔는지 본다. strict 에 한정어
    # 없이 값만 있으면 판단 근거가 없으므로 플래그하지 않는다(보수적 — 오탐
    # 방지). 답변의 방향 표현이 케이스 서술(facts)에는 존재하면 — 케이스
    # 재서술("복용 15일 이후 증상")일 수도, 규정 왜곡일 수도 있다 — 기계는
    # 구분할 수 없으므로 경고를 끄는 대신 from_case 라벨로 그 모호성을
    # 가시화한다(존재 축의 2계층 설계와 같은 원리: 차단도 침묵도 아닌 라벨).
    norm_answer = _normalize(answer)
    seen_dir: set[str] = set()
    # (허용 canonical 형태들, 한정어 단어, 표시용 표기) — 숫자 표기 + 고유어 표기
    answer_quals: list[tuple[tuple[tuple[str, str], ...], str, str]] = []
    for m in _QUAL_RE.finditer(norm_answer):
        unit = _norm_unit(m.group(2))
        key = (m.group(1).lstrip("0") or "0", unit)
        answer_quals.append(((key,), m.group(3), f"{key[0]}{key[1]}"))
    for m in _NATIVE_QUAL_RE.finditer(norm_answer):
        answer_quals.append((_NATIVE_NUMERALS[m.group(1)], m.group(2), m.group(1)))
    for forms, qual_word, base in answer_quals:
        key = next((f for f in forms if f in src_nums), None)
        if key is None:
            continue  # 값 자체가 미확인 — 위에서 이미 unsupported 로 잡혔다
        cls = _qual_class(qual_word)
        strict_cls = set().union(*(strict_quals.get(f, set()) for f in forms))
        if cls not in strict_cls and _OPPOSITE[cls] in strict_cls:
            display = f"{base} {qual_word}"
            if display in seen_dir:
                continue
            seen_dir.add(display)
            in_facts = any(cls in facts_quals.get(f, set()) for f in forms)
            in_q = any(cls in q_quals.get(f, set()) for f in forms)
            result.checks.append(
                ClaimCheck(display, "direction", False,
                           evidence=_snippet(trusted, key, "numeric"),
                           from_question=in_q, from_case=in_facts)
            )

    # 날짜 방향 한정어 대조 — 존재 대조를 통과한 날짜에 한해, 답변의 한정어
    # 방향("2026-07-25까지" ↔ "이후")이 strict 계층과 뒤집혔는지 본다. 수치
    # 방향 대조와 같은 규칙: strict 에 그 날짜의 한정어가 없으면 판단하지
    # 않고(보수성), 케이스 서술에 같은 방향이 있으면 from_case 로 가시화한다.
    src_date_quals = _date_qualifier_map(strict)
    facts_date_quals = _date_qualifier_map(facts) if facts else {}
    for m in _DATE_QUAL_RE.finditer(norm_answer):
        d = m.group(1)
        if d not in src_dates:
            continue  # 날짜 자체가 미확인 — 존재 대조 축이 이미 잡았다
        cls = _qual_class(m.group(2))
        trusted_cls = src_date_quals.get(d, set())
        if cls not in trusted_cls and _OPPOSITE[cls] in trusted_cls:
            display = f"{d} {m.group(2)}"
            if display in seen_dir:
                continue
            seen_dir.add(display)
            result.checks.append(
                ClaimCheck(display, "direction", False,
                           evidence=_snippet(trusted, (d, ""), "date"),
                           from_question=cls in q_date_quals.get(d, set()),
                           from_case=cls in facts_date_quals.get(d, set()))
            )

    # 부분 날짜(연도 없는 표기)의 방향 대조(v8) — 존재 축이 접미로 지지하는
    # 표기는 방향 축도 접미로 대조한다. 규칙은 완전한 날짜와 동일: strict 에
    # 그 접미의 한정어가 '반대 방향만' 있을 때 플래그, 케이스 방향은 라벨.
    src_partial_quals = _partial_date_qualifier_map(strict)
    facts_partial_quals = _partial_date_qualifier_map(facts) if facts else {}
    for m in _PARTIAL_DATE_QUAL_RE.finditer(norm_answer):
        suffix = m.group(1)
        if suffix not in src_partials:
            continue  # 존재 축(부분 날짜 접미 대조)이 이미 잡았다
        cls = _qual_class(m.group(2))
        trusted_cls = src_partial_quals.get(suffix, set())
        if cls not in trusted_cls and _OPPOSITE[cls] in trusted_cls:
            mm, dd = suffix.split("-")
            display = f"{int(mm)}월 {int(dd)}일 {m.group(2)}"
            if display in seen_dir:
                continue
            seen_dir.add(display)
            result.checks.append(
                ClaimCheck(display, "direction", False,
                           evidence=_snippet(trusted, (suffix, ""), "partial_date"),
                           from_question=cls in q_partial_quals.get(suffix, set()),
                           from_case=cls in facts_partial_quals.get(suffix, set()))
            )

    # 날짜 역할 대조 — 존재 대조를 통과한 날짜에 한해, 답변이 그 날짜에 부여한
    # 역할(기한/인지일)이 결정론적 도구의 역할 라벨과 일치하는지 본다.
    # 신뢰 소스에 해당 역할 라벨이 없으면(검색 근거만 있는 경우 등) 판단
    # 근거가 없으므로 플래그하지 않는다(보수적 — 오탐 방지).
    seen_role: set[str] = set()
    for role, answer_re in _ROLE_ANSWER_RE.items():
        # 역할 라벨은 strict 계층에서만 수집한다 — 라벨은 결정론적 도구의 JSON
        # 직렬화 키에서 오므로 케이스 서술에 있을 수 없지만, '판정 기준은
        # strict'라는 방향·역할 축의 공통 규칙을 코드에서도 동일하게 지킨다.
        labels = set(_ROLE_LABEL_RE[role].findall(strict))
        if not labels:
            continue
        for m in answer_re.finditer(norm_answer):
            d = m.group(1)
            if d in labels or d not in src_dates:
                continue  # 역할 일치, 또는 미확인 날짜(존재 대조 축이 이미 잡았다)
            display = f"{role} {d}"
            if display in seen_role:
                continue
            seen_role.add(display)
            result.checks.append(
                ClaimCheck(display, "role", False, evidence=", ".join(sorted(labels)),
                           from_question=d in q_roles.get(role, set()))
            )
        # 부분 날짜 표기의 역할 대조(v8) — 라벨(완전한 ISO)을 월-일 접미로
        # 투영해 비교한다. 존재 축을 접미로 통과한 표기가 역할 축(ISO 전용
        # 매칭)을 우회하던 사각지대의 봉합. 보수성 규칙은 동일하다.
        label_suffixes = {l[5:] for l in labels}
        for m in _ROLE_ANSWER_PARTIAL_RE[role].finditer(norm_answer):
            suffix = m.group(1)
            if suffix in label_suffixes or suffix not in src_partials:
                continue
            mm, dd = suffix.split("-")
            display = f"{role} {int(mm)}월 {int(dd)}일"
            if display in seen_role:
                continue
            seen_role.add(display)
            result.checks.append(
                ClaimCheck(display, "role", False, evidence=", ".join(sorted(labels)),
                           from_question=suffix in q_roles_partial.get(role, set()))
            )

    if citations and not allow_superseded:
        allowed = allowed_superseded_ids or set()
        result.superseded_cited = [
            c.get("doc_id") or c.get("source") or "?"
            for c in citations
            if c.get("status") == "superseded" and c.get("doc_id") not in allowed
        ]
    return result


def warning_text(v: VerificationResult) -> str:
    """검증 실패 시 답변에 부착할 경고문(시끄러운 실패)."""
    parts: list[str] = []
    q_origin = set(v.question_origin)
    hallucinated = [c for c in v.unsupported if c not in q_origin]
    if hallucinated:
        parts.append(
            "⚠ 자동 검증 경고: 답변 속 수치·날짜 "
            + ", ".join(f"'{c}'" for c in hallucinated)
            + " 이(가) 검색 근거·도구 결과에서 확인되지 않았습니다. "
            "제출·회신 전 규정 원문 대조가 필요합니다."
        )
    if q_origin:
        parts.append(
            "⚠ 전제 확인 필요: 수치 "
            + ", ".join(f"'{c}'" for c in sorted(q_origin))
            + " 은(는) 질문에 포함되어 있던 값으로, 검색 근거에서는 확인되지 않았습니다. "
            "답변이 이를 정정하는 맥락인지 포함해 질문의 전제 자체를 규정 원문과 대조하세요."
        )
    for c in (x for x in v.checks if x.kind == "direction"):
        # 문구 우선순위(v9): 전제 에코 > 케이스 에코 > 무조건 경고 — 존재 축의
        # '환각 vs 전제 확인' 구분과 같은 원리다. 질문의 틀린 전제를 정정하는
        # 답변("15일 이후가 아니라 이내")은 전제를 재서술할 수밖에 없는데,
        # 완화 없는 '컴플라이언스 오류' 단정이 붙으면 옳은 정정마다 오탐이다
        # (경고 자체는 유지 — 끄는 것이 아니라 종류를 조정한다).
        if c.from_question:
            note = (
                " 이 방향 표현은 질문의 전제에 있던 것입니다 — 답변이 그 전제를"
                " 정정·재서술하는 맥락인지 확인하세요(근거 자체는 반대 방향입니다)."
            )
        elif c.from_case:
            note = (
                " 케이스 서술에는 같은 방향 표현이 있어 재서술일 수 있으나, 규정"
                " 근거와는 방향이 반대입니다 — 어느 쪽인지 확인이 필요합니다."
            )
        else:
            note = " 기한·범위의 방향이 뒤집히면 수치가 맞아도 컴플라이언스 오류입니다."
        parts.append(
            f"⚠ 방향 한정어 경고: 답변의 '{c.claim}' 은(는) 근거와 방향이 반대입니다"
            + (f" (근거: \"…{c.evidence}…\")" if c.evidence else "")
            + "." + note
        )
    for c in (x for x in v.checks if x.kind == "role"):
        role, date = c.claim.split(" ", 1)
        premise = (
            " 이 역할-날짜 조합은 질문의 전제에 있던 것입니다 — 답변이 전제를"
            " 정정하는 맥락인지 확인하세요."
            if c.from_question else ""
        )
        parts.append(
            f"⚠ 날짜 역할 경고: 답변이 '{role}'(으)로 제시한 {date} 은(는) 근거에 존재하지만, "
            f"도구가 해당 역할로 계산한 날짜({c.evidence})와 다릅니다. "
            "날짜들의 역할(인지일↔마감일)이 서로 뒤바뀌지 않았는지 대조하세요." + premise
        )
    if v.superseded_cited:
        parts.append(
            "⚠ 버전 경고: 폐지(superseded)된 규정 "
            + ", ".join(v.superseded_cited)
            + " 이(가) 출처에 포함되어 있습니다. 현행 규정 기준으로 재확인하세요."
        )
    return "\n".join(parts)
