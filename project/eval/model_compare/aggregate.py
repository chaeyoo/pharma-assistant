"""집계 — 질문별 레코드 → (단계, 층, 모델)별 요약 행. 순수 함수만 둔다(테스트 고정 입력).

분모 규칙(정직성): 결정론 지표(핵심어·출처·검증기·회피·주입 무순응·도구선택)는 **시도 전수**가
분모이고 실패한 호출은 '틀림'으로 센다. 심판 지표는 심판이 유효한 판정을 돌려준 건만 분모로
하며 표에 '(심판·참고)' 라벨을 붙여 결정론 지표와 구분한다. p95 는 nearest-rank 방식이다.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from eval.stats import wilson_interval

TIER_LABEL = {
    "simple": "A1. 단순 (QA 32 · 정답이 컨텍스트에 있음)",
    "complex": "A2. 복합 (근거 2개 결합)",
    "no_evidence": "A3. 근거 제거 (QA 문항의 정답 문서를 뺀 컨텍스트 · 기대: 회피)",
    "out_of_scope": "A4. 범위 밖 (사내 문서로 답 불가 · 기대: 회피)",
    "adversarial": "A5. 적대 입력 (지시 주입·거짓 전제 · marker 부재가 정상)",
    "tool_selection": "B. 도구 선택 (plan 18문항 · 1스텝)",
}
# 층별 대표 지표(계기판 요약용) — 결정론 지표만 대표로 둔다.
PRIMARY = {
    "simple": ("correctness_keywords", "핵심어 일치"),
    "complex": ("correctness_keywords", "핵심어 일치"),
    "no_evidence": ("abstain_ok", "회피 정확"),
    "out_of_scope": ("abstain_ok", "회피 정확"),
    "adversarial": ("injection_resisted", "주입 무순응"),
    "tool_selection": ("tool_choice_ok", "도구 선택 정확"),
}


def median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return float(s[mid]) if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def p95(xs: list[float]) -> float | None:
    """nearest-rank: 정렬 후 ceil(0.95·n) 번째 값(1-based)."""
    if not xs:
        return None
    s = sorted(xs)
    rank = max(1, math.ceil(0.95 * len(s)))
    return float(s[rank - 1])


def cost_usd(input_tokens: int, output_tokens: int, cache_read: int, cache_write: int,
             price: dict | None) -> float | None:
    """단가(USD per 1M tokens)로 호출 비용 계산. 단가가 없으면 None(추정 금지)."""
    if not price or price.get("input") is None or price.get("output") is None:
        return None
    total = (
        input_tokens * float(price["input"])
        + output_tokens * float(price["output"])
        + cache_read * float(price.get("cache_read") or 0.0)
        + cache_write * float(price.get("cache_write") or 0.0)
    )
    return round(total / 1_000_000, 6)


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "p": round(k / n, 3) if n else None, "lo": lo, "hi": hi}


def _count(rs: list[dict], key: str) -> int:
    return sum(1 for r in rs if r.get(key))


def aggregate(records: list[dict]) -> list[dict]:
    """(stage, tier, model) 별 요약 행. 행 순서는 레코드 등장 순(run 이 stage·모델 순으로 정렬해 둔다)."""
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for r in records:
        groups.setdefault((r["stage"], r.get("tier", r["stage"]), r["model"]), []).append(r)
    rows: list[dict] = []
    for (stage, tier, model), rs in groups.items():
        n = len(rs)
        ok = [r for r in rs if r["status"] == "ok"]
        lat = [float(r["latency_ms"]) for r in rs if r.get("latency_ms") is not None]
        costs = [r["cost_usd"] for r in rs if r.get("cost_usd") is not None]
        row: dict = {
            "stage": stage, "tier": tier, "model": model, "n": n,
            "failures": n - len(ok),
            "failure_rate": round((n - len(ok)) / n, 3) if n else None,
            "failure_kinds": sorted({r["status"] for r in rs if r["status"] != "ok"}),
            "latency_p50_ms": round(median(lat), 1) if lat else None,
            "latency_p95_ms": round(p95(lat), 1) if lat else None,
            "input_tokens": sum(int(r.get("input_tokens") or 0) for r in rs),
            "output_tokens": sum(int(r.get("output_tokens") or 0) for r in rs),
            "cost_total_usd": round(sum(costs), 4) if costs else None,
            "cost_mean_usd": round(sum(costs) / len(costs), 5) if costs else None,
            # 길이 상한 절단(stop_reason=max_tokens) — 핵심어 누락이 능력이 아니라 절단 때문일 수 있어 드러낸다
            "truncated": sum(1 for r in rs if r.get("stop_reason") == "max_tokens"),
        }
        if stage == "generation":
            judged_f = [r for r in rs if r.get("faithfulness_judge")]
            row["faithfulness_all_supported"] = _rate(
                sum(1 for r in judged_f if r["faithfulness_judge"]["all_supported"]), len(judged_f))
            sup = sum(r["faithfulness_judge"]["supported"] for r in judged_f)
            tot = sum(r["faithfulness_judge"]["total"] for r in judged_f)
            row["faithfulness_claim_support"] = {"k": sup, "n": tot, "p": round(sup / tot, 3) if tot else None}
            row["verifier_ok"] = _rate(sum(1 for r in rs if (r.get("verifier") or {}).get("ok")), n)
            row["citation_present"] = _rate(_count(rs, "citation_present"), n)
            if tier in ("simple", "complex"):
                row["correctness_keywords"] = _rate(_count(rs, "correctness_keywords"), n)
                judged_c = [r for r in rs if r.get("correctness_judge")]
                row["correctness_judge"] = _rate(
                    sum(1 for r in judged_c if r["correctness_judge"]["match"]), len(judged_c))
                if tier == "complex":
                    row["context_covers_gold"] = _rate(_count(rs, "context_covers_gold"), n)
            if tier in ("no_evidence", "out_of_scope"):
                # no_evidence 의 누출 문항(정답 문서를 뺐는데 다른 문서에 핵심어가 전부 남음)은
                # '근거 없음'이 아니므로 회피 분모에서 뺀다. n_leaked 로 개수를 드러낸다.
                scored = [r for r in rs if not r.get("evidence_leaked")]
                row["n_leaked"] = n - len(scored)
                row["abstain_ok"] = _rate(_count(scored, "abstain_ok"), len(scored))
                row["hallucinated_numbers"] = _rate(
                    sum(1 for r in scored if (r.get("verifier") or {}).get("hallucinated_numbers")), len(scored))
                if tier == "no_evidence":
                    row["asserted_without_evidence"] = _rate(_count(scored, "asserted_without_evidence"), len(scored))
            if tier == "adversarial":
                row["marker_present"] = _rate(_count(rs, "marker_present"), n)
                row["hallucinated_numbers"] = _rate(
                    sum(1 for r in rs if (r.get("verifier") or {}).get("hallucinated_numbers")), n)
                row["injection_resisted"] = _rate(_count(rs, "injection_resisted"), n)
        else:
            row["tool_choice_ok"] = _rate(_count(rs, "tool_choice_ok"), n)
            row["tool_choice_lenient"] = _rate(_count(rs, "tool_choice_lenient"), n)
            with_param = [r for r in rs if r.get("tool_param_ok") is not None]
            row["tool_param_ok"] = _rate(sum(1 for r in with_param if r["tool_param_ok"]), len(with_param))
        key, label = PRIMARY.get(tier, PRIMARY["tool_selection"] if stage != "generation" else ("verifier_ok", "검증기 통과"))
        row["primary"] = {"label": label, **row.get(key, {"k": 0, "n": 0, "p": None})}
        rows.append(row)
    return rows


def _fmt(rate: dict | None) -> str:
    if not rate or rate.get("p") is None:
        return "—"
    if "lo" in rate:
        return f"{rate['p']:.3f} [{rate['lo']:.3f}, {rate['hi']:.3f}] ({rate['k']}/{rate['n']})"
    return f"{rate['p']:.3f} ({rate['k']}/{rate['n']})"


def _num(x, fmt: str = "{:.1f}") -> str:
    return "—" if x is None else fmt.format(x)


def _tail(r: dict) -> str:
    return (f"| {_num(r['failure_rate'], '{:.3f}')} ({r['failures']}/{r['n']}) | {r.get('truncated', 0)} "
            f"| {_num(r['latency_p50_ms'])} / {_num(r['latency_p95_ms'])} "
            f"| {_num(r['cost_total_usd'], '{:.4f}')} / {_num(r['cost_mean_usd'], '{:.5f}')} |")


_TAIL_HDR = "| 실패율 | 절단(max_tokens) | 지연 p50 / p95 (ms) | 비용 합계 / 건당 (USD) |"
_TAIL_SEP = "|---|---|---|---|"


def to_markdown(rows: list[dict]) -> str:
    """층별 집계 표 — 결정론 열과 '(심판·참고)' 열을 분리해 표시."""
    out: list[str] = []
    by_tier: dict[str, list[dict]] = {}
    for r in rows:
        by_tier.setdefault(r["tier"], []).append(r)
    for tier in ("simple", "complex", "no_evidence", "out_of_scope", "adversarial", "tool_selection"):
        rs = by_tier.get(tier)
        if not rs:
            continue
        out.append(f"### {TIER_LABEL[tier]} — 결정론 지표")
        out.append("")
        if tier in ("simple", "complex"):
            extra = " 컨텍스트가 정답 문서 포함 |" if tier == "complex" else ""
            out.append("| 모델 | n | 핵심어 일치 | 출처 표기 | 검증기 통과 |" + extra + _TAIL_HDR[1:])
            out.append("|---|---|---|---|---|" + ("---|" if extra else "") + _TAIL_SEP[1:])
            for r in rs:
                mid = f"| {_fmt(r.get('context_covers_gold'))} " if tier == "complex" else ""
                out.append(f"| {r['model']} | {r['n']} | {_fmt(r['correctness_keywords'])} | {_fmt(r['citation_present'])} "
                           f"| {_fmt(r['verifier_ok'])} {mid}" + _tail(r))
            out.append("")
            out.append(f"#### {tier} — 심판 지표 (심판·참고: 고정 심판 모델의 확률적 판정)")
            out.append("")
            out.append("| 모델 | gold_answer 의미 일치 (심판·참고) | 주장 전부 지지 (심판·참고) | 주장 단위 지지율 (심판·참고) |")
            out.append("|---|---|---|---|")
            for r in rs:
                out.append(f"| {r['model']} | {_fmt(r['correctness_judge'])} | {_fmt(r['faithfulness_all_supported'])} "
                           f"| {_fmt(r['faithfulness_claim_support'])} |")
        elif tier in ("no_evidence", "out_of_scope"):
            if tier == "no_evidence":
                out.append("| 모델 | n (누출 제외) | 회피 정확 | 미확인 수치 발생 ↓ | 근거 없이 정답 단정 ↓ | 주장 전부 지지 (심판·참고) |" + _TAIL_HDR[1:])
                out.append("|---|---|---|---|---|---|" + _TAIL_SEP[1:])
            else:
                out.append("| 모델 | n | 회피 정확 | 미확인 수치 발생 ↓ | 주장 전부 지지 (심판·참고) |" + _TAIL_HDR[1:])
                out.append("|---|---|---|---|---|" + _TAIL_SEP[1:])
            for r in rs:
                n_txt = f"{r['n']} ({r['n'] - r.get('n_leaked', 0)})" if tier == "no_evidence" else str(r["n"])
                mid = f"| {_fmt(r.get('asserted_without_evidence'))} " if tier == "no_evidence" else ""
                out.append(f"| {r['model']} | {n_txt} | {_fmt(r['abstain_ok'])} | {_fmt(r['hallucinated_numbers'])} {mid}"
                           f"| {_fmt(r['faithfulness_all_supported'])} " + _tail(r))
        elif tier == "adversarial":
            out.append("| 모델 | n | 주입 무순응 (marker 없음 또는 반박 동반) | marker 등장 (엄격·참고) | 미확인 수치 발생 ↓ (질문 에코 제외) | 출처 표기 | 주장 전부 지지 (심판·참고) |" + _TAIL_HDR[1:])
            out.append("|---|---|---|---|---|---|---|" + _TAIL_SEP[1:])
            for r in rs:
                out.append(f"| {r['model']} | {r['n']} | {_fmt(r['injection_resisted'])} | {_fmt(r['marker_present'])} | {_fmt(r['hallucinated_numbers'])} "
                           f"| {_fmt(r['citation_present'])} | {_fmt(r['faithfulness_all_supported'])} " + _tail(r))
        else:
            out.append("| 모델 | n | 도구 선택 정확 (엄격: 라우터 계약 일치) | 도구 선택 관대 (기대 도구 포함·forbidden 없음) | 인자 정확 |" + _TAIL_HDR[1:])
            out.append("|---|---|---|---|---|" + _TAIL_SEP[1:])
            for r in rs:
                out.append(f"| {r['model']} | {r['n']} | {_fmt(r['tool_choice_ok'])} | {_fmt(r['tool_choice_lenient'])} | {_fmt(r['tool_param_ok'])} " + _tail(r))
        out.append("")
    return "\n".join(out)


_CSV_COLS = [
    "stage", "tier", "model", "n", "failures", "failure_rate", "failure_kinds", "truncated",
    "correctness_keywords", "correctness_judge", "context_covers_gold",
    "n_leaked", "abstain_ok", "asserted_without_evidence", "hallucinated_numbers", "marker_present", "injection_resisted",
    "faithfulness_all_supported", "faithfulness_claim_support", "citation_present", "verifier_ok",
    "tool_choice_ok", "tool_choice_lenient", "tool_param_ok",
    "latency_p50_ms", "latency_p95_ms", "input_tokens", "output_tokens",
    "cost_total_usd", "cost_mean_usd",
]


def _flat(v):
    if isinstance(v, dict):
        return "" if v.get("p") is None else f"{v['p']} ({v['k']}/{v['n']})"
    if isinstance(v, list):
        return ";".join(map(str, v))
    return "" if v is None else v


def write_outputs(rows: list[dict], out_dir: Path, meta: dict) -> dict[str, Path]:
    """summary.csv / summary.md / summary.json 을 쓴다. 반환: 경로 dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, md_path, json_path = out_dir / "summary.csv", out_dir / "summary.md", out_dir / "summary.json"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _flat(r.get(k)) for k in _CSV_COLS})
    md_path.write_text(to_markdown(rows), encoding="utf-8")
    json_path.write_text(
        json.dumps({**meta, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"csv": csv_path, "md": md_path, "json": json_path}
