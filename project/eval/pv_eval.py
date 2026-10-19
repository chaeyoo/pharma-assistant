"""PV 워크플로 수치 평가 — 규칙 기반이라도 '측정 없이 신뢰 없음'.

트리아지→인과성→코딩→최소보고요건 전 단계를 라벨 데이터셋(pv_dataset.json,
22케이스)으로 평가한다. 검색(evaluate)·신뢰성(faithfulness)과 같은 철학:
PV 도구도 회귀가 수치로 잡혀야 사전·규칙을 안심하고 고칠 수 있다.

라벨은 '도메인 정답'이지 코드 출력의 복사가 아니다. 사전이 못 잡는 롱테일
구어(저혈당·울렁거림·난청)와 2계층(LLT 참조)조차 못 잡는 심층 롱테일
(저릿저릿)을 일부러 포함해, 확정 재현율(1계층)과 후보 재현율(2계층)이 모두
정직하게 1.0 미만이 되게 했다 — 각 갭이 사전 검수 큐와 MedDRA 본체 교체라는
확장 지점의 크기다.

측정 지표
  - SeriousnessAcc : 중대성(Serious) 판정 정확도 — 닫힌 목록 대조라 1.0이어야 정상
  - DeadlineAcc    : 보고기한 계산 정확도(일수 + 날짜 연산) — 컴플라이언스 그 자체
  - CausalityAcc   : WHO-UMC 제안 등급 일치율
  - Coding P/R/F1  : 1계층 '확정' 코딩 micro 정밀도/재현율/F1
                     (재현율<1 = 검수된 사전의 롱테일 크기 — 후보로 메워도 확정
                     지표는 섞지 않는다: 확정과 후보는 집계 신뢰도가 다르다)
  - CandidateRecall: 확정∪후보(2계층 LLT 제안)까지 합친 재현율 — '사람 확정
                     큐에 올라오는가'를 측정. 확정 재현율과의 갭 = 검수 대기량
  - CandidatePrecision: 제시된 후보 중 정답 비율 — 후보는 사람이 거르지만
                     노이즈가 많으면 검수 비용이 커지므로 따로 감시한다
  - AEDetectionRecall: 이상사례 존재를 어느 계층으로든(확정/후보/미코딩 감지)
                     알아챈 비율 — 최소보고요건 ④ 판정의 상한
  - ReportableAcc  : 최소보고요건(ICH E2D 4요소) 판정 정확도
  - MissingCountAcc: 빠진 요소 개수까지 정확히 짚는가(라벨이 있는 문항만)

실행:  python -m eval.pv_eval
"""
from __future__ import annotations

import json

from src import config
from src.pv.report import build_report


def _load() -> list[dict]:
    path = config.BASE_DIR / "eval" / "pv_dataset.json"
    return json.loads(path.read_text(encoding="utf-8"))["items"]


def evaluate() -> dict:
    items = _load()
    n = len(items)
    serious_ok = deadline_ok = causality_ok = reportable_ok = 0
    missing_total = missing_ok = 0
    tp = fp = fn = 0
    cand_tp = cand_fp = 0            # 2계층 후보 채점(확정과 분리 집계)
    ae_cases = ae_detected = 0       # 이상사례 '존재 감지'(계층 무관)
    failures: list[str] = []

    for item in items:
        exp = item["expect"]
        args = item.get("args", {})
        r = build_report(item["case"], **args)

        if r.triage.is_serious == exp["serious"]:
            serious_ok += 1
        else:
            failures.append(f"{item['id']} 중대성: got={r.triage.is_serious} want={exp['serious']}")

        dd_ok = r.triage.deadline_days == exp["deadline_days"]
        if "deadline_date" in exp:
            dd_ok = dd_ok and r.triage.deadline_date == exp["deadline_date"]
        if dd_ok:
            deadline_ok += 1
        else:
            failures.append(
                f"{item['id']} 기한: got={r.triage.deadline_days}({r.triage.deadline_date}) "
                f"want={exp['deadline_days']}({exp.get('deadline_date', '-')})"
            )

        # 라벨은 WHO-UMC 영문 키워드(Possible 등), 출력은 '가능함(Possible)' 형식
        if exp["causality"] in r.causality.suggested:
            causality_ok += 1
        else:
            failures.append(f"{item['id']} 인과성: got={r.causality.suggested} want={exp['causality']}")

        got_pts = {t.pt for t in r.coded_terms}
        want_pts = set(exp["pts"])
        tp += len(got_pts & want_pts)
        fp += len(got_pts - want_pts)
        fn += len(want_pts - got_pts)
        if want_pts - got_pts:
            failures.append(f"{item['id']} 확정코딩 미검출(사전 롱테일): {sorted(want_pts - got_pts)}")
        if got_pts - want_pts:
            failures.append(f"{item['id']} 확정코딩 오탐: {sorted(got_pts - want_pts)}")

        # 2계층 후보: 확정이 놓친 정답을 사람 확정 큐에 올렸는가 + 후보 노이즈
        got_cand = {t.pt for t in r.candidate_terms}
        cand_tp += len(got_cand & (want_pts - got_pts))
        cand_fp += len(got_cand - want_pts)
        want_cand = set(exp.get("candidate_pts", []))
        if got_cand != want_cand:
            failures.append(
                f"{item['id']} 후보 불일치: got={sorted(got_cand)} want={sorted(want_cand)}"
            )
        if want_pts - got_pts - got_cand:
            failures.append(
                f"{item['id']} 후보에도 없음(심층 롱테일 — 미코딩 감지"
                f"{'됨' if r.uncoded_expressions else ' 안 됨'}): "
                f"{sorted(want_pts - got_pts - got_cand)}"
            )
        if exp.get("uncoded_detected") and not r.uncoded_expressions:
            failures.append(f"{item['id']} 미코딩 감지 실패(3계층)")

        # 이상사례 '존재' 감지 — 계층 무관(확정/후보/미코딩 중 하나라도)
        if want_pts:
            ae_cases += 1
            if got_pts or got_cand or r.uncoded_expressions:
                ae_detected += 1

        if r.reportable == exp["reportable"]:
            reportable_ok += 1
        else:
            failures.append(
                f"{item['id']} 보고요건: got={r.reportable} want={exp['reportable']} "
                f"(missing={r.missing})"
            )

        if "missing_n" in exp:
            missing_total += 1
            if len(r.missing) == exp["missing_n"]:
                missing_ok += 1
            else:
                failures.append(f"{item['id']} 누락개수: got={len(r.missing)} want={exp['missing_n']}")

    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "n": n,
        "counts": {
            "serious_ok": serious_ok, "deadline_ok": deadline_ok,
            "reportable_ok": reportable_ok, "coding_tp": tp, "coding_fn": fn,
        },
        "SeriousnessAcc": round(serious_ok / n, 3),
        "DeadlineAcc": round(deadline_ok / n, 3),
        "CausalityAcc": round(causality_ok / n, 3),
        "CodingPrecision": round(precision, 3),
        "CodingRecall": round(recall, 3),
        "CodingF1": round(f1, 3),
        "CandidateRecall": round((tp + cand_tp) / (tp + fn), 3) if tp + fn else 1.0,
        "CandidatePrecision": round(cand_tp / (cand_tp + cand_fp), 3) if cand_tp + cand_fp else 1.0,
        "AEDetectionRecall": round(ae_detected / ae_cases, 3) if ae_cases else 1.0,
        "ReportableAcc": round(reportable_ok / n, 3),
        "MissingCountAcc": round(missing_ok / missing_total, 3) if missing_total else None,
        "failures": failures,
    }


def main() -> None:
    res = evaluate()
    print("=" * 62)
    print(f"PV 워크플로 평가 · 라벨 케이스 {res['n']}건 (트리아지→인과성→코딩→보고요건)")
    print("=" * 62)
    print(f"SeriousnessAcc  (중대성 판정)          : {res['SeriousnessAcc']:.3f}")
    print(f"DeadlineAcc     (보고기한 일수+날짜 연산): {res['DeadlineAcc']:.3f}")
    print(f"CausalityAcc    (WHO-UMC 제안 등급)     : {res['CausalityAcc']:.3f}")
    print(
        f"Coding P/R/F1   (1계층 확정 코딩)       : "
        f"{res['CodingPrecision']:.3f} / {res['CodingRecall']:.3f} / {res['CodingF1']:.3f}"
    )
    print(
        f"CandidateRecall (확정∪후보 — 검수 큐 도달): {res['CandidateRecall']:.3f}"
        f"  (후보 정밀도 {res['CandidatePrecision']:.3f})"
    )
    print(f"AEDetectionRecall(존재 감지 — ④요건 상한): {res['AEDetectionRecall']:.3f}")
    print(f"ReportableAcc   (최소보고요건 4요소)    : {res['ReportableAcc']:.3f}")
    if res["MissingCountAcc"] is not None:
        print(f"MissingCountAcc (빠진 요소 개수 정확도) : {res['MissingCountAcc']:.3f}")
    from eval.stats import fmt_ci

    c = res["counts"]
    print("-" * 62)
    print("통계적 정직성 (Wilson 95% CI — n=22 표본에서 1.000의 의미):")
    print(f"  SeriousnessAcc {fmt_ci(c['serious_ok'], res['n'])} · DeadlineAcc {fmt_ci(c['deadline_ok'], res['n'])}")
    print(f"  ReportableAcc  {fmt_ci(c['reportable_ok'], res['n'])} · CodingRecall {fmt_ci(c['coding_tp'], c['coding_tp'] + c['coding_fn'])}")
    if res["failures"]:
        print("-" * 62)
        print("불일치 상세 (의도된 롱테일 감점 포함):")
        for f in res["failures"]:
            print(f"  - {f}")
    print("-" * 62)
    print("해석: 중대성·기한은 규정의 닫힌 목록 대조 + 날짜 연산이라 1.0이 정상이며,")
    print("      1.0이 깨지면 규칙 회귀다(CI가 잡는다). 확정 코딩 재현율(<1.0)은 검수된")
    print("      사전의 롱테일 크기 — 2계층(LLT 참조 후보)이 이 갭을 '사람 확정 큐'로")
    print("      올리고(CandidateRecall), 어느 사전에도 없는 심층 롱테일은 3계층이")
    print("      '미코딩 감지'로 표시한다(AEDetectionRecall). 확정 정밀도는 무관용(1.0):")
    print("      잘못 붙은 코드는 집계를 오염시키므로 후보를 확정에 섞지 않는다.")
    print("      보고요건 ④는 '증상 서술의 존재'로 판정하므로 코딩 실패가 reportable")
    print("      오판으로 연쇄되지 않는다(막연한 서술은 여전히 미충족 — specificity).")


if __name__ == "__main__":
    main()
