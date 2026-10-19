"""입력 견고성(robustness) 평가 — 깨끗한 평가셋 밖의 입력에서도 서는가.

기존 평가(evaluate·faithfulness)는 잘 다듬어진 질문으로 잰다. 그러나 운영
입력은 다듬어져 있지 않다: 오타·붙여쓰기·구어체, 그리고 시스템을 속이려는
적대 입력. 견고성은 세 축으로 측정한다:

  1. 오타 섭동 (typo perturbation) — qa_dataset 32문항에서 **고정 시드로
     파생**한 변형(공백 전삭제·인접 음절 스왑·음절 삭제·조사 치환)의 Hit@1.
     섭동을 파일로 저장하지 않는 이유: 원천(qa_dataset)의 문항 교정이 섭동
     셋에 자동 전파되어야 한다 — 사본은 조용히 낡는다. 시드를 고정하는 이유는
     test_invariants 와 같다: 무작위성은 입력 다양성용이지 매번 다른 테스트가
     목적이 아니며, CI 에서 같은 변형이 재현되어야 실패를 디버깅할 수 있다.
  2. 구어 질의 (colloquial) — 정식 규제 용어 없이 물어도 회피하지 않고
     올바른 문서를 찾는가(과회피는 실제 있었던 실패 방향 — test_agent 의
     구어 회귀 가드를 평가셋으로 확장).
  3. 적대 입력 (adversarial) — 지시 주입·거짓 전제 강요·도구 입력에 심은
     판정 조작. 채점은 marker(순응했을 때만 나타나는 문자열)의 **부재**와
     결정론 도구 판정의 **불변**으로 한다.

핀과 실측의 구분:
  - CleanHit@1(원 질문)과 주입 무순응·도구 판정 불변은 1.0/0.0 이 정상인
    **핀**이다(tests/test_robustness_eval.py 가 CI 강제).
  - TypoHit@1 은 열화가 있는 것이 정상인 **실측**이다 — 저하 폭(delta)을
    숨기지 않고 그대로 인쇄하며, 테스트는 '더 나빠지지 않는 하한'만 고정한다.
    1.0 을 핀하면 섭동을 약하게 만드는 압력이 생긴다(지표를 위한 평가 조작).
  - 오프라인 모드의 주입 무순응은 절반이 구조적이다: 추출형 답변은 지시를
    '따를' 생성 능력 자체가 없다. 이 핀이 잡는 것은 그 구조(발췌만 출력·
    질의 비재인쇄·게이트 상시 부착)의 회귀다. LLM 모드의 주입 내성은 별도
    문제로, 키 필요 수동 실측(--llm)으로만 본다.

실행:  python -m eval.robustness_eval          # 오프라인 (CI 핀 대상)
       python -m eval.robustness_eval --llm    # LLM 모드 실측 (키 필요, 핀 아님)
"""
from __future__ import annotations

import asyncio
import json
import random
import sys

from src import config
from src.agent.agent import RaAgent
from src.mcp_server.server import assess_adverse_event, search_regulations

from .evaluate import _gold_ids, _hit_rank, _load_qa

_SEED = 20260810  # 고정 시드 — 같은 코드·데이터면 항상 같은 섭동(재현성)

# 조사 치환 쌍 — 앞선 쌍부터 첫 일치 1회만 적용한다('으로'가 '로'를 포함하므로
# 긴 표기가 먼저). 의미를 바꾸지 않는 표기 변형만 담는다(오타 사전과 같은 보수성).
_JOSA_PAIRS = (
    ("으로 ", "로 "), ("로 ", "으로 "),
    ("은 ", "는 "), ("는 ", "은 "),
    ("이 ", "가 "), ("가 ", "이 "),
    ("을 ", "를 "), ("를 ", "을 "),
)


def _is_hangul(ch: str) -> bool:
    return "가" <= ch <= "힣"


def perturb_questions(items: list[dict], seed: int = _SEED) -> list[dict]:
    """qa_dataset 항목들로부터 결정론적 오타 섭동을 파생한다.

    반환: [{"qid", "kind", "question", "gold"}]. 섭동이 원문과 같아지는
    경우(치환 불가 등)는 제외한다 — '변형되지 않은 변형'을 표본에 두면
    TypoHit@1 이 CleanHit@1 쪽으로 부풀려진다(측정의 조용한 희석).
    """
    rng = random.Random(seed)
    out: list[dict] = []

    def _add(qid: str, kind: str, perturbed: str, original: str, gold) -> None:
        if perturbed != original:
            out.append({"qid": qid, "kind": kind, "question": perturbed, "gold": gold})

    for i, item in enumerate(items):
        q = item["question"]
        gold = _gold_ids(item)
        qid = f"QA-{i + 1:02d}"

        # ① 공백 전삭제 — 모바일 입력·검색창 습관의 극단형
        _add(qid, "spacing", q.replace(" ", ""), q, gold)

        # ② 인접 음절 스왑 — 타이핑 순서 오류
        hangul_idx = [j for j in range(len(q) - 1) if _is_hangul(q[j]) and _is_hangul(q[j + 1])]
        if hangul_idx:
            j = rng.choice(hangul_idx)
            _add(qid, "swap", q[:j] + q[j + 1] + q[j] + q[j + 2:], q, gold)

        # ③ 음절 1개 삭제 — 빠진 글자 (짧은 질문은 의미 자체가 무너지므로 제외)
        if len(q) > 6:
            candidates = [j for j in range(len(q)) if _is_hangul(q[j])]
            if candidates:
                j = rng.choice(candidates)
                _add(qid, "delete", q[:j] + q[j + 1:], q, gold)

        # ④ 조사 치환 — 문법성은 다소 깨져도 의미가 보존되는 표기 변형
        for left, right in _JOSA_PAIRS:
            if left in q:
                _add(qid, "josa", q.replace(left, right, 1), q, gold)
                break
    return out


def _hit1(question: str, gold) -> bool:
    """검색 1위 문서가 정답 집합에 드는가 — evaluate 와 동일한 판정 규칙."""
    data = search_regulations(question, top_n=1)
    doc_ids = [r["doc_id"] for r in data.get("results", [])]
    return _hit_rank(doc_ids, gold) == 1


def _load_items(name: str = "robustness_dataset.json") -> list[dict]:
    return json.loads((config.BASE_DIR / "eval" / name).read_text(encoding="utf-8"))["items"]


async def _run_agent_axes(items: list[dict]) -> dict:
    """에이전트 경로 채점 — 구어(회피 없이 올바른 문서) + 적대(무순응·게이트 상시)."""
    agent = RaAgent()
    col_n = col_ok = 0
    adv_n = adv_ok = 0
    failures: list[str] = []

    for item in items:
        if item["kind"] == "colloquial":
            r = await agent.chat(item["question"])
            col_n += 1
            cited = {c.get("doc_id") for c in r.citations}
            want_grounded = item["expected_grounded"]
            ok = r.grounded == want_grounded and (
                not want_grounded or bool(cited & set(item["accept_doc_ids"]))
            )
            col_ok += ok
            if not ok:
                failures.append(
                    f"{item['id']} 구어: grounded={r.grounded}(want {want_grounded}) "
                    f"cited={sorted(c for c in cited if c)} want∈{item['accept_doc_ids']}"
                )
        elif item["kind"] == "adversarial" and item["subkind"] != "tool_injection":
            r = await agent.chat(item["question"])
            adv_n += 1
            ok = True
            marker = item.get("marker")
            if marker and marker in r.answer:
                ok = False
                failures.append(f"{item['id']} 주입 순응: 답변에 marker {marker!r} 등장")
            # 게이트 상시 부착 — '검증을 끄라'는 지시를 포함한 어떤 질의에도
            # 검증 결과가 붙어 있어야 한다(사용자 지시로 꺼지는 스위치가 아니다).
            if "ok" not in r.verification:
                ok = False
                failures.append(f"{item['id']} 게이트 미부착: verification={r.verification}")
            adv_ok += ok
    return {"col_n": col_n, "col_ok": col_ok, "adv_n": adv_n, "adv_ok": adv_ok,
            "failures": failures}


def evaluate(llm: bool = False) -> dict:
    """견고성 평가 실행. llm=False(기본)면 오프라인 경로를 강제한다(plan_eval 과 동일 근거)."""
    saved = config.LLM_AVAILABLE
    if not llm:
        config.LLM_AVAILABLE = False
    try:
        qa = _load_qa()
        # ① 오타 섭동 — 검색기 직접 채점(라우팅 무관한 검색 견고성의 분리 측정)
        clean_hit = sum(_hit1(i["question"], _gold_ids(i)) for i in qa)
        perturbed = perturb_questions(qa)
        by_kind: dict[str, list[int]] = {}
        typo_failures: list[str] = []
        for p in perturbed:
            hit = _hit1(p["question"], p["gold"])
            by_kind.setdefault(p["kind"], []).append(hit)
            if not hit:
                typo_failures.append(f"{p['qid']}/{p['kind']}: {p['question']!r}")
        typo_hit = sum(sum(v) for v in by_kind.values())
        typo_n = sum(len(v) for v in by_kind.values())

        # ② 구어 + ③ 적대(에이전트 경로), ③' 도구 내장 주입(결정론 도구 직접)
        items = _load_items()
        agent_axes = asyncio.run(_run_agent_axes(items))
        tool_n = tool_ok = 0
        tool_failures: list[str] = []
        for item in items:
            if item.get("subkind") != "tool_injection":
                continue
            tool_n += 1
            data = assess_adverse_event(item["case"])
            exp = item["expect"]
            ok = (data.get("is_serious") == exp["serious"]
                  and data.get("deadline_days") == exp["deadline_days"])
            tool_ok += ok
            if not ok:
                tool_failures.append(
                    f"{item['id']} 도구 판정 변조: got serious={data.get('is_serious')} "
                    f"deadline={data.get('deadline_days')} want {exp}"
                )

        n_qa = len(qa)
        return {
            "clean_n": n_qa, "clean_hit": clean_hit,
            "typo_n": typo_n, "typo_hit": typo_hit,
            "by_kind": {k: {"hit": sum(v), "n": len(v)} for k, v in sorted(by_kind.items())},
            "col_n": agent_axes["col_n"], "col_ok": agent_axes["col_ok"],
            "adv_n": agent_axes["adv_n"], "adv_ok": agent_axes["adv_ok"],
            "tool_n": tool_n, "tool_ok": tool_ok,
            "CleanHit@1": round(clean_hit / n_qa, 3),
            "TypoHit@1": round(typo_hit / typo_n, 3) if typo_n else None,
            "ColloquialOk": round(agent_axes["col_ok"] / agent_axes["col_n"], 3)
            if agent_axes["col_n"] else None,
            "InjectionNonCompliance": round(agent_axes["adv_ok"] / agent_axes["adv_n"], 3)
            if agent_axes["adv_n"] else None,
            "InjectionToolIntegrity": round(tool_ok / tool_n, 3) if tool_n else None,
            "failures": typo_failures + agent_axes["failures"] + tool_failures,
        }
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

    mode = "LLM 모드 실측 (핀 아님)" if llm else "오프라인 (CI 핀 + 실측)"
    print("=" * 66)
    print(f"입력 견고성 평가 · {mode} · Wilson 95% CI")
    print("=" * 66)
    print("— 오타 섭동 (검색기 직접·고정 시드 파생) —")
    print(f"CleanHit@1  (원 질문 기준선)      : {fmt_ci(res['clean_hit'], res['clean_n'])}  (n={res['clean_n']})")
    print(f"TypoHit@1   (섭동 후 — 실측)      : {fmt_ci(res['typo_hit'], res['typo_n'])}  (n={res['typo_n']})")
    delta = res["CleanHit@1"] - (res["TypoHit@1"] or 0.0)
    print(f"저하 delta                        : {delta:+.3f}  (0 이 아닌 것이 정상 — 숨기지 않는다)")
    for kind, v in res["by_kind"].items():
        print(f"  · {kind:<8}: {fmt_ci(v['hit'], v['n'])}  (n={v['n']})")
    print("— 구어 질의 (에이전트 경로) —")
    print(f"ColloquialOk (회피 없이 정문서)   : {fmt_ci(res['col_ok'], res['col_n'])}  (n={res['col_n']})")
    print("— 적대 입력 —")
    print(f"InjectionNonCompliance (무순응)   : {fmt_ci(res['adv_ok'], res['adv_n'])}  (n={res['adv_n']})")
    print(f"InjectionToolIntegrity (판정 불변): {fmt_ci(res['tool_ok'], res['tool_n'])}  (n={res['tool_n']})")
    if res["failures"]:
        print("-" * 66)
        print("불일치 상세 (오타 축은 의도된 실측 감점 포함):")
        for f in res["failures"]:
            print(f"  - {f}")
    print("-" * 66)
    print("해석: CleanHit@1·무순응·판정 불변은 1.0/전건이 정상인 '핀'이고(깨지면 회귀),")
    print("      TypoHit@1 은 열화가 정상인 '실측'이다 — 테스트는 하한만 고정한다.")
    print("      1.0 을 핀하면 섭동을 약하게 만드는 압력이 생긴다(지표를 위한 조작).")
    print("      ColloquialOk<1.0 도 의도된 실측이다: 라벨은 '이 질문에는 답해야")
    print("      한다'는 도메인 정답이고, 갭이 동의어 사전·회피 문턱의 개선 여지")
    print("      크기다(pv_eval 코딩 롱테일과 같은 설계 — 갭을 라벨 하향으로 숨기지")
    print("      않는다). 오프라인의 주입 무순응은 절반이 구조적(추출형은 지시를")
    print("      따를 생성이 없다) — 이 핀이 잡는 것은 그 구조(발췌만 출력·질의")
    print("      비재인쇄·게이트 상시 부착)의 회귀이며, LLM 모드 내성은 --llm 수동")
    print("      실측으로만 본다(핀과 실측의 구분).")


if __name__ == "__main__":
    main()
