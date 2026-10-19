"""LLM judge 2차 검증의 평가 — 확률 계층은 '핀'이 아니라 '실측'으로 잰다.

verify_eval 이 결정론 게이트를 메타모픽 변조로 재는 것과 같은 철학을, judge 의
관할(수치 없는 의미 왜곡)에 적용한다: 정상 답변에 **의미 변조**를 심어 탐지를
재고(planted), 정상 답변의 오탐을 함께 감시한다(clean). 변조 3유형은 judge 의
판정 대상 정본(JUDGE_KINDS)과 1:1 이다:

  - negation_flip   : 부정 관계 뒤집기 ("보고 대상이다" → "보고 대상이 아니다")
  - subject_swap    : 두 대상의 의무·속성 맞바꾸기 (변경허가 ↔ 변경신고)
  - meaning_inversion: 수치 없는 그 밖의 의미 반전 (면제 ↔ 의무)

결정론 게이트 평가(verify_eval)와의 결정적 차이 — **CI 핀이 없다**:
  judge 는 확률적이라 같은 케이스에 다른 판정이 나올 수 있다. 탐지율을 CI 로
  핀하면 (a) 플레이키 실패로 핀 전체의 신뢰가 죽거나 (b) 통과를 위해 케이스를
  쉽게 만드는 압력이 생긴다. 그래서 이 스크립트는 측정·기록만 한다(키 필요
  수동 실행). judge 의 **결정론적 부분**(배선·파싱·비차단·기본 비활성)은
  tests/test_judge.py 가 스텁으로 핀 고정하고, enabled 부팅의 생존 canary 는
  preflight(check_judge)가 맡는다 — 검증의 검증을 세 조각으로 이원화한 것이
  self-critique 순환에 대한 이 데모의 답이다(llm_judge.py docstring).

실행:  python -m eval.judge_eval   (ANTHROPIC_API_KEY 필요 — 없으면 안내 후 종료)
"""
from __future__ import annotations

from src import config

# 케이스 테이블 — trusted 는 코퍼스 문체의 근거 발췌, answer 는 판정 대상.
# corrupt=True 케이스의 answer 는 근거의 '의미 관계'만 뒤집은 것이다(수치·날짜
# 왜곡은 넣지 않는다 — 그것은 1차 결정론 게이트의 관할이라 여기 섞으면 judge 가
# 경계를 침범해도 점수가 오르는 잘못된 보상이 된다).
_CASES: tuple[dict, ...] = (
    # --- negation_flip (4) ---
    {"kind": "negation_flip", "corrupt": True,
     "trusted": ["중대한 이상사례는 신속보고 대상이다."],
     "answer": "중대한 이상사례는 신속보고 대상이 아닙니다."},
    {"kind": "negation_flip", "corrupt": True,
     "trusted": ["임상시험 중 발생한 중대하고 예상치 못한 이상반응은 식약처 보고 대상이다."],
     "answer": "임상시험 중 발생한 중대하고 예상치 못한 이상반응은 식약처에 보고할 필요가 없습니다."},
    {"kind": "negation_flip", "corrupt": True,
     "trusted": ["시판 후에도 안전성 정보의 수집·평가 의무는 유지된다."],
     "answer": "시판 후에는 안전성 정보의 수집·평가 의무가 없습니다."},
    {"kind": "negation_flip", "corrupt": True,
     "trusted": ["원료의약품 등록(DMF) 대상 성분은 등록 없이 완제 허가 신청에 사용할 수 없다."],
     "answer": "원료의약품 등록(DMF) 대상 성분은 등록 없이도 완제 허가 신청에 사용할 수 있습니다."},
    # --- subject_swap (4) ---
    {"kind": "subject_swap", "corrupt": True,
     "trusted": ["변경허가는 사전 심사·승인이 필요하고, 변경신고는 수리 통보로 완료된다."],
     "answer": "변경신고는 사전 심사·승인이 필요하고, 변경허가는 수리 통보로 완료됩니다."},
    {"kind": "subject_swap", "corrupt": True,
     "trusted": ["의약품은 품목허가 대상이고, 의약외품은 품목신고 대상이다."],
     "answer": "의약품은 품목신고 대상이고, 의약외품은 품목허가 대상입니다."},
    {"kind": "subject_swap", "corrupt": True,
     "trusted": ["제조업자는 GMP 적합판정을 유지할 의무가 있고, 수입자는 해외 제조소의 적합성을 확인할 의무가 있다."],
     "answer": "수입자는 GMP 적합판정을 유지할 의무가 있고, 제조업자는 해외 제조소의 적합성을 확인할 의무가 있습니다."},
    {"kind": "subject_swap", "corrupt": True,
     "trusted": ["신속심사 대상은 희귀의약품이고, 우선심사 대상은 혁신형 제약기업의 신약이다."],
     "answer": "신속심사 대상은 혁신형 제약기업의 신약이고, 우선심사 대상은 희귀의약품입니다."},
    # --- meaning_inversion (4) ---
    {"kind": "meaning_inversion", "corrupt": True,
     "trusted": ["조건부 허가는 시판 후 확증 임상 자료 제출을 조건으로 한다."],
     "answer": "조건부 허가를 받으면 시판 후 확증 임상 자료 제출이 면제됩니다."},
    {"kind": "meaning_inversion", "corrupt": True,
     "trusted": ["표시기재 사항의 변경은 변경절차를 거친 뒤에 적용해야 한다."],
     "answer": "표시기재 사항은 먼저 적용한 뒤에 변경절차를 밟으면 됩니다."},
    {"kind": "meaning_inversion", "corrupt": True,
     "trusted": ["실태조사에서 보완 요구를 받으면 기한 내 보완 후 재평가를 받는다."],
     "answer": "실태조사에서 보완 요구를 받아도 재평가 없이 종결됩니다."},
    {"kind": "meaning_inversion", "corrupt": True,
     "trusted": ["인과성 평가가 '가능함' 이상인 케이스는 규제기관 보고 여부 판단에 포함한다."],
     "answer": "인과성 평가가 '가능함' 이상인 케이스는 규제기관 보고 판단에서 제외합니다."},
    # --- clean (8) — 재서술·요약은 왜곡이 아니다(오탐 감시) ---
    {"kind": "clean", "corrupt": False,
     "trusted": ["중대한 이상사례는 신속보고 대상이다."],
     "answer": "중대한 이상사례는 신속보고 대상입니다."},
    {"kind": "clean", "corrupt": False,
     "trusted": ["변경허가는 사전 심사·승인이 필요하고, 변경신고는 수리 통보로 완료된다."],
     "answer": "변경허가는 사전에 심사와 승인을 받아야 하고, 변경신고는 수리 통보로 끝납니다."},
    {"kind": "clean", "corrupt": False,
     "trusted": ["임상시험 중 발생한 중대하고 예상치 못한 이상반응은 식약처 보고 대상이다."],
     "answer": "임상시험 중 중대하고 예상치 못한 이상반응이 생기면 식약처에 보고해야 합니다."},
    {"kind": "clean", "corrupt": False,
     "trusted": ["조건부 허가는 시판 후 확증 임상 자료 제출을 조건으로 한다."],
     "answer": "조건부 허가는 시판 후에 확증 임상 자료를 내는 것을 조건으로 합니다."},
    {"kind": "clean", "corrupt": False,
     "trusted": ["의약품은 품목허가 대상이고, 의약외품은 품목신고 대상이다."],
     "answer": "의약외품은 품목신고 대상이고, 의약품은 품목허가 대상입니다."},
    {"kind": "clean", "corrupt": False,
     "trusted": ["시판 후에도 안전성 정보의 수집·평가 의무는 유지된다."],
     "answer": "허가를 받은 뒤에도 안전성 정보를 계속 수집하고 평가해야 합니다."},
    {"kind": "clean", "corrupt": False,
     "trusted": ["실태조사에서 보완 요구를 받으면 기한 내 보완 후 재평가를 받는다."],
     "answer": "실태조사 보완 요구를 받으면 기한 안에 보완하고 재평가를 받습니다."},
    {"kind": "clean", "corrupt": False,
     "trusted": ["표시기재 사항의 변경은 변경절차를 거친 뒤에 적용해야 한다."],
     "answer": "표시기재를 바꾸려면 변경절차부터 거쳐야 합니다."},
)


def evaluate() -> dict:
    from src.verify.llm_judge import judge_answer

    per_kind: dict[str, list[bool]] = {}
    clean_pass = clean_n = unavailable = 0
    failures: list[str] = []
    for i, case in enumerate(_CASES):
        j = judge_answer(case["answer"], case["trusted"])
        if j["status"] == "unavailable":
            unavailable += 1
            failures.append(f"case{i}({case['kind']}) unavailable: {j['reason']}")
            continue
        if case["corrupt"]:
            detected = j["status"] == "flagged"
            per_kind.setdefault(case["kind"], []).append(detected)
            if not detected:
                failures.append(f"case{i}({case['kind']}) 미탐지: {case['answer']!r}")
        else:
            clean_n += 1
            if j["status"] == "ok":
                clean_pass += 1
            else:
                failures.append(f"case{i}(clean) 오탐: {j['flags']}")
    return {
        "per_kind": {k: {"hit": sum(v), "n": len(v)} for k, v in sorted(per_kind.items())},
        "clean_n": clean_n, "clean_pass": clean_pass,
        "unavailable": unavailable, "failures": failures,
        "model": config.JUDGE_MODEL,
    }


def main() -> None:
    if not config.ANTHROPIC_API_KEY:
        print("judge 평가는 ANTHROPIC_API_KEY 가 필요합니다 — 이 평가는 확률 계층의")
        print("'실측 기록'용 수동 스크립트라 CI 에도 들어가지 않습니다(핀은 tests/test_judge.py).")
        return
    from eval.stats import fmt_ci

    res = evaluate()
    print("=" * 66)
    print(f"LLM judge 2차 검증 평가 · model={res['model']} · planted 의미 변조 + clean · Wilson 95% CI")
    print("=" * 66)
    for kind, v in res["per_kind"].items():
        print(f"DetectionRate[{kind:<17}] : {fmt_ci(v['hit'], v['n'])}  (n={v['n']})")
    print(f"CleanPassRate (재서술 오탐 없음)   : {fmt_ci(res['clean_pass'], res['clean_n'])}  (n={res['clean_n']})")
    if res["unavailable"]:
        print(f"※ unavailable {res['unavailable']}건 — 표본에서 제외(은폐가 아니라 명시)")
    if res["failures"]:
        print("-" * 66)
        for f in res["failures"]:
            print(f"  - {f}")
    print("-" * 66)
    print("해석: 이 수치는 '핀'이 아니라 '실측'이다 — 확률 계층을 CI 로 핀하면")
    print("      플레이키 실패가 핀 전체의 신뢰를 죽이거나 케이스를 쉽게 만드는")
    print("      압력이 생긴다. 탐지율의 갭은 judge 프롬프트·모델 선택의 개선 여지")
    print("      크기이고, CleanPassRate 는 경고 피로(alert fatigue)의 반대 방향")
    print("      감시다. 결정론적 배선 계약은 tests/test_judge.py 가 핀 고정한다.")


if __name__ == "__main__":
    main()
