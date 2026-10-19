"""계획 능력(plan) 평가 — 의도가 '올바른 도구 호출'로 변환되는가.

검색 평가(evaluate)·신뢰성 평가(faithfulness)는 도구가 '무엇을 돌려주는가'를
재지만, 에이전트 품질의 앞 절반은 그 전에 결정된다: **어떤 도구를, 어떤
인자로, 무엇은 부르지 않기로 했는가**(계획). 잘못된 도구 선택은 이후 단계가
아무리 정확해도 회복되지 않는다 — 규정 질문에 업무 일정 도구를 부르면
검색 정확도 1.0 이 무의미하다.

측정 지표 (항목 단위 채점 — Wilson CI 병기 가능하도록 비율이 아니라 건수)
  - ToolRecall    : 기대한 도구를 전부 호출한 항목 비율. tools_ordered 항목은
                    호출 순서까지 일치해야 한다(체크리스트 실패→검색 폴백처럼
                    순서가 곧 회복 로직인 케이스).
  - ToolPrecision : 불필요한 호출이 없는 항목 비율 — forbidden 도구 미호출
                    **그리고** 기대 집합 밖 잉여 호출 없음. 특히 search_web 은
                    외부 송신 스위치라 오호출이 일반 오호출보다 비싸다.
  - ParamAccuracy : 인자 검사(equals / contains / absent)를 전부 통과한 항목
                    비율. 검사가 있는 항목만 분모로 한다(없는 항목을 자동
                    통과로 섞으면 지표가 부풀려진다).

핀과 실측의 구분 (verify_eval 의 규율을 그대로 적용):
  기본 실행은 **오프라인 결정론 라우터**를 강제한다(환경에 API 키가 있어도
  config.LLM_AVAILABLE 을 잠시 내려 같은 입력이 항상 같은 라우팅이 되게).
  결정론 경로의 지표는 1.0 이 정상이며 tests/test_plan_eval.py 가 CI 실패로
  핀 고정한다 — 깨지는 순간이 곧 라우팅 회귀다. LLM 모드(--llm, 키 필요)는
  비결정론이라 핀이 불가능하므로 수동 실측으로만 기록한다(expect_llm 라벨).

실행:  python -m eval.plan_eval          # 오프라인 결정론 라우터 (CI 핀 대상)
       python -m eval.plan_eval --llm    # LLM 모드 실측 (키 필요, 핀 아님)
"""
from __future__ import annotations

import asyncio
import json
import sys

from src import config
from src.agent.agent import RaAgent

# MCP 도구 전집 — 데이터셋 무결성 검사(test_plan_eval)와 공유하는 정본.
# 새 도구가 추가되면 여기와 데이터셋 라벨을 함께 재검토한다.
TOOL_UNIVERSE = frozenset({
    "search_regulations", "get_ra_deadlines", "get_submission_checklist",
    "assess_adverse_event", "draft_ae_report", "search_web",
    "list_regulation_documents",
})


def _load_items(name: str = "plan_dataset.json") -> list[dict]:
    return json.loads((config.BASE_DIR / "eval" / name).read_text(encoding="utf-8"))["items"]


def _check_args(spec: dict, args: dict) -> list[str]:
    """인자 검사 1건 — equals(정확 일치) / contains(부분 문자열) / absent(부재)."""
    problems: list[str] = []
    for key, want in spec.get("equals", {}).items():
        got = args.get(key)
        if got != want:
            problems.append(f"{key}: got={got!r} want={want!r}")
    for key, substrings in spec.get("contains", {}).items():
        got = str(args.get(key, ""))
        for s in substrings:
            if s not in got:
                problems.append(f"{key}: {s!r} 미포함 (got={got!r})")
    for key in spec.get("absent", []):
        if key in args:
            problems.append(f"{key}: 부재해야 하는데 존재 (got={args[key]!r})")
    return problems


def _score_item(item: dict, called: list[tuple[str, dict]], llm_labels: bool) -> dict:
    """한 항목 채점. called = [(도구명, 인자), ...] 호출 순서대로."""
    exp = item.get("expect_llm") if llm_labels and item.get("expect_llm") else item["expect"]
    called_names = [n for n, _ in called]
    failures: list[str] = []

    # Recall: 기대 도구 전부 호출(+ tools_ordered 면 그 도구들 간 순서 일치)
    recall_ok = all(t in called_names for t in exp["tools"])
    if recall_ok and exp.get("tools_ordered"):
        expected_set = set(exp["tools"])
        recall_ok = [n for n in called_names if n in expected_set] == exp["tools"]
    if not recall_ok:
        failures.append(f"recall: called={called_names} want={exp['tools']}")

    # Precision: forbidden 미호출 + 기대 집합 밖 잉여 호출 없음
    forbidden_hit = [t for t in exp.get("forbidden", []) if t in called_names]
    extras = [n for n in called_names if n not in set(exp["tools"])]
    precision_ok = not forbidden_hit and not extras
    if not precision_ok:
        failures.append(f"precision: forbidden={forbidden_hit} extras={extras}")

    # Parameter accuracy: 검사가 있는 항목만 (없으면 None — 분모에서 제외)
    param_ok: bool | None = None
    if exp.get("args_contains"):
        param_ok = True
        for tool, spec in exp["args_contains"].items():
            args = next((a for n, a in called if n == tool), None)
            if args is None:
                param_ok = False
                failures.append(f"param[{tool}]: 도구가 호출되지 않아 인자 검사 불가")
                continue
            problems = _check_args(spec, args)
            if problems:
                param_ok = False
                failures.append(f"param[{tool}]: " + "; ".join(problems))

    return {"recall_ok": recall_ok, "precision_ok": precision_ok,
            "param_ok": param_ok, "failures": failures}


async def _run(dataset: str, llm_labels: bool) -> dict:
    agent = RaAgent()
    items = _load_items(dataset)
    recall = precision = param = param_n = 0
    failures: list[str] = []
    for item in items:
        r = await agent.chat(item["question"], history=item.get("history", []))
        called = [(tc.name, tc.args) for tc in r.tool_calls]
        s = _score_item(item, called, llm_labels)
        recall += s["recall_ok"]
        precision += s["precision_ok"]
        if s["param_ok"] is not None:
            param_n += 1
            param += s["param_ok"]
        failures.extend(f"{item['id']} {f}" for f in s["failures"])
    n = len(items)
    return {
        "n": n,
        "counts": {"recall_ok": recall, "precision_ok": precision,
                   "param_ok": param, "param_n": param_n},
        "ToolRecall": round(recall / n, 3),
        "ToolPrecision": round(precision / n, 3),
        "ParamAccuracy": round(param / param_n, 3) if param_n else None,
        "failures": failures,
    }


def evaluate(dataset: str = "plan_dataset.json", llm: bool = False) -> dict:
    """계획 평가 실행. llm=False(기본)면 오프라인 라우터를 **강제**한다.

    환경에 ANTHROPIC_API_KEY 가 있으면 chat() 이 LLM 경로로 새는데, 그러면
    'CI 핀 = 결정론 라우터의 계약'이라는 전제가 실행 환경에 따라 조용히
    무너진다 — 평가가 무엇을 재는지는 환경이 아니라 코드가 결정해야 한다.
    """
    saved = config.LLM_AVAILABLE
    if not llm:
        config.LLM_AVAILABLE = False
    try:
        return asyncio.run(_run(dataset, llm_labels=llm))
    finally:
        config.LLM_AVAILABLE = saved


def main() -> None:
    from eval.stats import fmt_ci
    from src.mcp_server.http_harness import run_mcp_http_server

    llm = "--llm" in sys.argv
    if llm and not config.LLM_AVAILABLE:
        print("--llm 모드는 ANTHROPIC_API_KEY 가 필요합니다 — 오프라인 핀 평가는 인자 없이 실행하세요.")
        return

    with run_mcp_http_server() as url:
        config.MCP_SERVER_URL = url
        res = evaluate(llm=llm)

    c, n = res["counts"], res["n"]
    mode = "LLM 모드 실측 (핀 아님)" if llm else "오프라인 결정론 라우터 (CI 핀)"
    print("=" * 62)
    print(f"계획 능력 평가 · {mode} · {n}항목 · Wilson 95% CI")
    print("=" * 62)
    print(f"ToolRecall    (기대 도구 전부 호출)   : {fmt_ci(c['recall_ok'], n)}  (n={n})")
    print(f"ToolPrecision (불필요 호출 없음)      : {fmt_ci(c['precision_ok'], n)}  (n={n})")
    print(f"ParamAccuracy (인자 정확)             : {fmt_ci(c['param_ok'], c['param_n'])}  (n={c['param_n']})")
    if res["failures"]:
        print("-" * 62)
        print("불일치 상세:")
        for f in res["failures"]:
            print(f"  - {f}")
    print("-" * 62)
    print("해석: 오프라인 라우터는 결정론이므로 세 지표 모두 1.0 이 정상이며, 깨지는")
    print("      순간이 곧 라우팅 회귀다(tests/test_plan_eval.py 가 CI 실패로 핀 고정).")
    print("      1.000은 '계획이 완벽하다'가 아니라 '18개 계약이 지켜진다'는 뜻이다 —")
    print("      멀티 인텐트의 단일 처리(PLAN-013) 같은 알려진 한계는 라벨(note)에")
    print("      정직하게 남기고, LLM 모드 기대(expect_llm)는 --llm 실측으로만 본다.")


if __name__ == "__main__":
    main()
