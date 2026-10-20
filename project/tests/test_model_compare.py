"""모델 비교 실험(eval/model_compare) 테스트 — LLM 없이 전체 파이프라인을 돌린다.

FakeGateway 가 모델별 스크립트 응답(정상 텍스트·tool_use·타임아웃·빈 답변)을 돌려주고,
검색·도구 스키마·검증기는 실물을 태운다. 집계 계산(중앙값·p95·비용·CI)은 고정 입력으로 핀 고정.
"""
from __future__ import annotations

import json
import pathlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src import config
from eval.model_compare import aggregate as agg
from eval.model_compare import run as mc_run
from eval.model_compare import scoring
from eval.model_compare.gateway import Completion, request_params
from eval.model_compare.report import build_report


class _TimeoutErr(Exception):
    pass


class FakeGateway:
    """모델명·도구 유무에 따라 스크립트 응답을 돌려주는 가짜 게이트웨이.

    scripts[model] = {"text": str | Exception | "", "tool": [tool_use dicts]}
    심판 호출(모델 == judge)은 judge_text 로 답한다. calls 에 kwargs 를 기록한다.
    """

    def __init__(self, scripts: dict, judge: str, judge_text: str, latency_by_model: dict | None = None):
        self.scripts, self.judge, self.judge_text = scripts, judge, judge_text
        self.latency = latency_by_model or {}
        self.calls: list[dict] = []

    def complete(self, *, model, system, messages, max_tokens, tools=None) -> Completion:
        self.calls.append({"model": model, "system": system, "messages": messages,
                           "max_tokens": max_tokens, "tools": tools})
        params = request_params(model, system, max_tokens, tools)
        lat = float(self.latency.get(model, 100.0))
        if model == self.judge:
            return Completion(text=self.judge_text, stop_reason="end_turn", input_tokens=50,
                              output_tokens=20, latency_ms=30.0, request_params=params)
        s = self.scripts[model]
        if tools:
            return Completion(text="", tool_uses=s.get("tool", []), stop_reason="tool_use",
                              input_tokens=300, output_tokens=40, latency_ms=lat, request_params=params)
        text = s["text"]
        if isinstance(text, Exception):
            return Completion(latency_ms=lat, request_params=params, error=type(text).__name__)
        return Completion(text=text, stop_reason="end_turn", input_tokens=1000, output_tokens=100,
                          latency_ms=lat, request_params=params)


def _cfg(models: list[str]) -> dict:
    cfg = mc_run.load_config()
    cfg["models"] = models
    cfg["judge_model"] = "fake-judge"
    cfg["pricing"] = {
        "fake-good": {"input": 1.0, "output": 5.0, "cache_read": 0.1, "cache_write": 1.25},
        "fake-bad": {"input": 2.0, "output": 10.0, "cache_read": 0.1, "cache_write": 2.5},
        "fake-abstain": {"input": 1.0, "output": 5.0, "cache_read": 0.1, "cache_write": 1.25},
        "fake-judge": {"input": 2.0, "output": 10.0, "cache_read": 0.2, "cache_write": 2.5},
    }
    return cfg


def _unreviewed(rel: str) -> dict:
    return json.loads((config.BASE_DIR / rel).read_text(encoding="utf-8"))


GOOD_ANSWER = "신약 품목허가 심사는 신청일로부터 120 근무일 이내에 처리됩니다 (출처: 의약품 품목허가 신청 가이드 · 4. 심사 기간)."
ABSTAIN_ANSWER = "제공된 근거에서 확인되지 않습니다."
JUDGE_OK = json.dumps({"claims": [{"claim": "120 근무일", "supported": True, "reason": "근거 4절"}],
                       "match": True, "reason": "일치"}, ensure_ascii=False)


def test_full_pipeline_with_fake_gateway(tmp_path, pipeline, mcp_server_url):
    """A(5개 층)·B 전 단계가 LLM 없이 돌고, 산출물 4종이 생기며, 실패·오답·회피가 상태로 기록된다."""
    APITimeoutError = type("APITimeoutError", (Exception,), {})  # 실제 SDK 예외 타입명과 동일
    tool_ok = [{"id": "t1", "name": "search_regulations", "input": {"query": "품목허가 심사"}}]
    gw = FakeGateway(
        scripts={
            "fake-good": {"text": GOOD_ANSWER, "tool": tool_ok},
            "fake-bad": {"text": APITimeoutError(), "tool": []},
            "fake-abstain": {"text": ABSTAIN_ANSWER, "tool": tool_ok},
        },
        judge="fake-judge", judge_text=JUDGE_OK,
        latency_by_model={"fake-good": 100.0, "fake-bad": 900.0, "fake-abstain": 50.0},
    )
    cfg = _cfg(["fake-good", "fake-bad", "fake-abstain"])
    tools = mc_run.fetch_tool_schemas()
    assert {t["name"] for t in tools} >= {"search_regulations", "get_ra_deadlines", "search_web"}
    gold = {it["id"]: it for it in _unreviewed(cfg["datasets"]["gold_answers"])["items"]}
    complex_items = _unreviewed(cfg["datasets"]["complex"])["items"]

    res = mc_run.run_experiment(cfg, gw, tmp_path, pipeline=pipeline, tools=tools,
                                gold=gold, complex_items=complex_items, workers=2)
    recs = res["records"]
    sizes = res["meta"]["tier_sizes"]
    assert sizes == {"simple": 32, "complex": len(complex_items), "no_evidence": 32,
                     "out_of_scope": 8, "adversarial": 5, "tool_selection": 18}
    assert len(recs) == sum(sizes.values()) * 3
    for name in ("results.jsonl", "summary.csv", "summary.md", "summary.json"):
        assert (tmp_path / name).exists(), name
    assert len((tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()) == len(recs)

    def pick(tier, model, qid=None):
        rs = [r for r in recs if r.get("tier") == tier and r["model"] == model]
        return next(r for r in rs if r["question_id"] == qid) if qid else rs

    # 컨텍스트 고정: 같은 층·문항은 모델이 달라도 같은 청크 id
    by_q = {}
    for r in recs:
        if r["stage"] == "generation":
            by_q.setdefault((r["tier"], r["question_id"]), set()).add(tuple(r["context_chunk_ids"]))
    assert all(len(v) == 1 for v in by_q.values())
    assert all(r["request_params"]["sampling"].startswith("api_default") for r in recs)

    # simple
    q1 = pick("simple", "fake-good", "Q01")
    assert q1["status"] == "ok" and q1["correctness_keywords"] is True and q1["citation_present"] is True
    assert q1["correctness_judge"] == {"match": True, "reason": "일치"}
    assert q1["faithfulness_judge"]["all_supported"] is True
    assert q1["cost_usd"] == pytest.approx((1000 * 1.0 + 100 * 5.0) / 1e6)
    assert q1["judge_usage"]["cost_usd"] == pytest.approx(2 * (50 * 2.0 + 20 * 10.0) / 1e6)
    bad = pick("simple", "fake-bad")
    assert all(r["status"] == "timeout" and r["error"] == "APITimeoutError" for r in bad)
    assert bad[0]["correctness_judge"] is None and bad[0]["correctness_judge_status"] == "skipped"
    assert bad[0]["correctness_keywords"] is False and bad[0]["cost_usd"] == 0.0
    assert pick("simple", "fake-abstain", "Q01")["correctness_keywords"] is False

    # complex: sub_queries 합집합 컨텍스트, 정답 문서 포함 여부 기록
    c1 = pick("complex", "fake-good", "C01")
    assert len(c1["context_chunk_ids"]) >= 2 and "context_covers_gold" in c1
    assert c1["correctness_judge"] is not None  # gold 가 있는 층은 심판 정답 일치가 돈다

    # no_evidence: 정답 문서 청크가 컨텍스트에서 빠져 있다
    ne1 = pick("no_evidence", "fake-good", "NE01")
    assert "REG-001" in ne1["removed_doc_ids"] and "REG-001" not in ne1["context_doc_ids"]
    assert ne1["abstain_ok"] is False and "evidence_leaked" in ne1
    # 누출이 아닌 문항에서만 '근거 없이 단정'이 성립한다
    assert ne1["asserted_without_evidence"] is (not ne1["evidence_leaked"])
    # (주의: '120 근무일' 은 REG-007 의 비교 문장에도 실재해 미확인 수치로 잡히지 않을 수 있다 —
    #  회피 실패 판정은 회피 표현 부재만으로 충분하다)
    ne_abs = pick("no_evidence", "fake-abstain", "NE01")
    assert ne_abs["abstain_ok"] is True and ne_abs["asserted_without_evidence"] is False
    assert ne_abs["correctness_judge"] is None  # gold 없는 층은 정답 일치 심판을 돌리지 않는다

    # out_of_scope / adversarial
    assert pick("out_of_scope", "fake-abstain", "OOS01")["abstain_ok"] is True
    assert pick("out_of_scope", "fake-good", "OOS01")["abstain_ok"] is False
    adv = pick("adversarial", "fake-good")
    assert {r["question_id"] for r in adv} == {"ROB-A01", "ROB-A03", "ROB-A04", "ROB-A05", "ROB-A06"}
    assert all(r["injection_resisted"] is True for r in adv)
    assert all(r["injection_resisted"] is False for r in pick("adversarial", "fake-bad"))

    # B 단계: search_web 은 명시 요청 턴에만 노출(운영 규칙)
    tool_calls = [c for c in gw.calls if c["tools"]]
    web_exposed = [c for c in tool_calls if any(t["name"] == "search_web" for t in c["tools"])]
    assert 0 < len(web_exposed) < len(tool_calls)
    p1 = pick("tool_selection", "fake-good", "PLAN-001")
    assert p1["tool_choice_ok"] is True and p1["tool_param_ok"] is True
    assert pick("tool_selection", "fake-good", "PLAN-002")["tool_choice_ok"] is False
    assert all(r["status"] == "schema_error" and r["tool_choice_ok"] is False
               for r in pick("tool_selection", "fake-bad"))

    # 집계
    rows = {(r["tier"], r["model"]): r for r in res["rows"]}
    g = rows[("simple", "fake-good")]
    assert g["failure_rate"] == 0.0 and g["latency_p50_ms"] == 100.0 and g["latency_p95_ms"] == 100.0
    assert g["correctness_keywords"]["n"] == 32 and g["correctness_judge"]["n"] == 32
    assert g["primary"]["label"] == "핵심어 일치" and g["primary"]["k"] == g["correctness_keywords"]["k"]
    b = rows[("simple", "fake-bad")]
    assert b["failure_rate"] == 1.0 and b["failure_kinds"] == ["timeout"]
    assert b["correctness_judge"]["p"] is None  # 판정 없음 → 분모 0
    assert rows[("no_evidence", "fake-abstain")]["abstain_ok"]["p"] == 1.0
    # 고정 답변은 Q01 핵심어(120·근무일)만 담으므로 '근거 없이 단정'은 최대 NE01 1건(누출이면 0)
    ne_row = rows[("no_evidence", "fake-good")]
    assert ne_row["asserted_without_evidence"]["k"] == (0 if ne1["evidence_leaked"] else 1)
    assert ne_row["abstain_ok"]["n"] == 32 - ne_row["n_leaked"]
    assert rows[("tool_selection", "fake-good")]["tool_choice_lenient"]["k"] >= rows[("tool_selection", "fake-good")]["tool_choice_ok"]["k"]
    assert rows[("adversarial", "fake-good")]["injection_resisted"]["p"] == 1.0
    assert rows[("tool_selection", "fake-bad")]["failure_kinds"] == ["schema_error"]
    assert "correctness_keywords" not in rows[("out_of_scope", "fake-good")]
    md = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "심판·참고" in md and "A3. 근거 제거" in md and "B. 도구 선택" in md
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["judge_model"] == "fake-judge" and len(summary["rows"]) == 6 * 3

    # 보고서: (d) 는 RECOMMENDATION.md 없으면 미작성으로 남는다(추천을 지어내지 않음)
    report = build_report(tmp_path, cfg, res["rows"], recs, res["meta"])
    text = report.read_text(encoding="utf-8")
    assert "## (b) 모델별 집계" in text and "## (e) 한계" in text and "미작성" in text
    assert "PLAN-002" in text and "APITimeoutError" in text and "근거 없이 정답 단정" in text


def test_rescore_recomputes_without_api(tmp_path, pipeline, mcp_server_url):
    """--rescore: 저장된 답변으로 결정론 지표를 재계산하고 집계·요약을 다시 쓴다(게이트웨이 호출 없음)."""
    gw = FakeGateway({"fake-good": {"text": GOOD_ANSWER, "tool": [
        {"id": "t", "name": "search_regulations", "input": {"query": "품목허가"}}]}}, "fake-judge", JUDGE_OK)
    cfg = _cfg(["fake-good"])
    cfg["tiers"] = ["simple", "adversarial"]
    gold = {it["id"]: it for it in _unreviewed(cfg["datasets"]["gold_answers"])["items"]}
    tools = mc_run.fetch_tool_schemas()
    mc_run.run_experiment(cfg, gw, tmp_path, pipeline=pipeline, tools=tools, gold=gold)
    n_calls = len(gw.calls)
    # 저장된 답변을 손으로 바꿔 재채점이 '저장본'을 쓰는지 확인
    recs = [json.loads(l) for l in (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    q1 = next(r for r in recs if r["question_id"] == "Q01")
    q1["answer"] = "약 넉 달 걸립니다."
    (tmp_path / "results.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8")
    res = mc_run.rescore(cfg, tmp_path, pipeline=pipeline)
    assert len(gw.calls) == n_calls  # API(게이트웨이) 재호출 없음
    q1b = next(r for r in res["records"] if r["question_id"] == "Q01")
    assert q1b["correctness_keywords"] is False and q1b["missing_keywords"] == ["120", "근무일"]
    assert q1b["correctness_judge"] == {"match": True, "reason": "일치"}  # 심판 결과는 보존
    assert res["meta"].get("rescored_at") and (tmp_path / "summary.md").exists()


def test_tier_subset_runs_only_selected(tmp_path, pipeline):
    """config.tiers 에서 뺀 층은 실행되지 않는다(모델·층 추가/삭제는 설정만으로)."""
    gw = FakeGateway({"fake-good": {"text": GOOD_ANSWER, "tool": []}}, "fake-judge", JUDGE_OK)
    cfg = _cfg(["fake-good"])
    cfg["tiers"] = ["out_of_scope", "adversarial"]
    res = mc_run.run_experiment(cfg, gw, tmp_path, stage="a", pipeline=pipeline)
    assert res["meta"]["tier_sizes"] == {"out_of_scope": 8, "adversarial": 5}
    assert {r["tier"] for r in res["records"]} == {"out_of_scope", "adversarial"}


def test_run_refuses_unreviewed_gold(tmp_path, monkeypatch):
    cfg = mc_run.load_config()
    for key, loader in (("gold_answers", mc_run.load_gold), ("complex", mc_run.load_complex_items)):
        data = _unreviewed(cfg["datasets"][key])
        data["reviewed"] = False
        p = tmp_path / f"{key}.json"
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        c2 = {**cfg, "datasets": {**cfg["datasets"], key: str(p.relative_to(config.BASE_DIR)) if p.is_relative_to(config.BASE_DIR) else str(p)}}
        monkeypatch.setattr(config, "BASE_DIR", config.BASE_DIR)
        c2["datasets"][key] = str(p)
        monkeypatch.setattr(mc_run, "_read", lambda rel, _p=p: json.loads(pathlib.Path(rel).read_text(encoding="utf-8")) if rel == str(_p) else json.loads((config.BASE_DIR / rel).read_text(encoding="utf-8")))
        with pytest.raises(RuntimeError):
            loader(c2)
        assert len(loader(c2, require_reviewed=False)) > 0


def test_canary_and_failure_report(tmp_path):
    """키가 틀리면 canary 가 모델별 오류를 모으고, 실패 보고서는 숫자 없이 사실만 남기며 옛 집계를 지운다."""
    AuthenticationError = type("AuthenticationError", (Exception,), {})
    gw = FakeGateway({"fake-good": {"text": GOOD_ANSWER, "tool": []},
                      "fake-bad": {"text": AuthenticationError(), "tool": []}}, "fake-judge", JUDGE_OK)
    cfg = _cfg(["fake-good", "fake-bad"])
    probe = mc_run.canary(cfg, gw)
    assert probe == {"fake-good": None, "fake-bad": "AuthenticationError", "fake-judge": None}
    assert all(c["max_tokens"] == 5 for c in gw.calls)
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    path = mc_run.write_failure_report(tmp_path, cfg, "canary 실패", probe)
    text = path.read_text(encoding="utf-8")
    assert "실행 실패" in text and "AuthenticationError" in text and "| fake-good | OK |" in text
    assert not (tmp_path / "summary.json").exists()


def test_config_rejects_unknown_tier(tmp_path):
    cfg = mc_run.load_config()
    cfg["tiers"] = ["simple", "nope"]
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError):
        mc_run.load_config(p)


def test_config_rejects_judge_in_models(tmp_path):
    cfg = mc_run.load_config()
    cfg["judge_model"] = cfg["models"][0]
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError):
        mc_run.load_config(p)


def test_gold_answers_align_with_qa_dataset():
    cfg = mc_run.load_config()
    qa = mc_run.load_qa_items(cfg)
    gold = mc_run.load_gold(cfg, require_reviewed=False)
    assert [q["id"] for q in qa] == list(gold)
    assert all(gold[q["id"]]["question"] == q["question"] for q in qa)
    assert all(gold[q["id"]]["gold_answer"].strip() for q in qa)


def test_complex_dataset_integrity():
    cfg = mc_run.load_config()
    items = mc_run.load_complex_items(cfg, require_reviewed=False)
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids)) >= 10
    corpus = "\n".join(p.read_text(encoding="utf-8") for p in (config.BASE_DIR / "data" / "regulations").glob("*.md"))
    for it in items:
        assert len(it["sub_queries"]) >= 2 and it["doc_ids"] and it["keywords"] and it["gold_answer"].strip()
        # 핵심어는 코퍼스에 실재해야 한다(존재하지 않는 핵심어는 채점 불가능)
        for k in it["keywords"]:
            assert k in corpus, (it["id"], k)


# ---------------------------------------------------------------------------
# 집계 계산 — 고정 입력
# ---------------------------------------------------------------------------
def test_median_and_p95_nearest_rank():
    assert agg.median([]) is None and agg.p95([]) is None
    assert agg.median([5.0]) == 5.0 and agg.p95([5.0]) == 5.0
    assert agg.median([3.0, 1.0, 2.0]) == 2.0
    assert agg.median([4.0, 1.0, 3.0, 2.0]) == 2.5
    xs = [float(i) for i in range(1, 21)]  # 1..20 → ceil(0.95*20)=19 번째 = 19
    assert agg.p95(xs) == 19.0
    assert agg.p95([1.0, 2.0, 3.0]) == 3.0  # ceil(2.85)=3


def test_cost_usd_fixed_inputs():
    price = {"input": 2.0, "output": 10.0, "cache_read": 0.2, "cache_write": 2.5}
    assert agg.cost_usd(1_000_000, 0, 0, 0, price) == 2.0
    assert agg.cost_usd(1000, 500, 0, 0, price) == pytest.approx(0.007)
    assert agg.cost_usd(0, 0, 1_000_000, 1_000_000, price) == pytest.approx(2.7)
    assert agg.cost_usd(10, 10, 0, 0, None) is None  # 단가 없음 → 추정 금지
    assert agg.cost_usd(10, 10, 0, 0, {"input": 1.0}) is None


def test_aggregate_rates_and_ci():
    recs = []
    for i in range(4):
        recs.append({"stage": "generation", "tier": "simple", "model": "m", "status": "ok", "latency_ms": 100.0 * (i + 1),
                     "cost_usd": 0.001, "input_tokens": 10, "output_tokens": 5,
                     "correctness_keywords": i < 3, "citation_present": True,
                     "verifier": {"ok": i != 0},
                     "correctness_judge": {"match": True, "reason": ""} if i < 2 else None,
                     "faithfulness_judge": {"supported": 2, "total": 2, "all_supported": True, "claims": []}})
    recs.append({"stage": "generation", "tier": "simple", "model": "m", "status": "api_error", "latency_ms": 50.0,
                 "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "correctness_keywords": False,
                 "citation_present": False, "verifier": {"ok": False}, "correctness_judge": None,
                 "faithfulness_judge": None})
    (row,) = agg.aggregate(recs)
    assert row["n"] == 5 and row["failures"] == 1 and row["failure_rate"] == 0.2
    assert row["correctness_keywords"] == {"k": 3, "n": 5, "p": 0.6, "lo": 0.231, "hi": 0.882}
    assert row["correctness_judge"]["n"] == 2 and row["correctness_judge"]["p"] == 1.0
    assert row["faithfulness_claim_support"] == {"k": 8, "n": 8, "p": 1.0}
    assert row["latency_p50_ms"] == 200.0 and row["latency_p95_ms"] == 400.0
    assert row["cost_total_usd"] == 0.004 and row["cost_mean_usd"] == 0.0008
    assert row["input_tokens"] == 40 and row["truncated"] == 0
    assert row["primary"] == {"label": "핵심어 일치", **row["correctness_keywords"]}


def test_aggregate_abstention_tier_fixed_inputs():
    base = {"stage": "generation", "tier": "no_evidence", "model": "m", "status": "ok", "latency_ms": 10.0,
            "cost_usd": 0.0, "input_tokens": 1, "output_tokens": 1, "citation_present": False,
            "faithfulness_judge": None}
    recs = [
        {**base, "abstain_ok": True, "asserted_without_evidence": False, "verifier": {"ok": True, "hallucinated_numbers": []}},
        {**base, "abstain_ok": False, "asserted_without_evidence": True, "verifier": {"ok": False, "hallucinated_numbers": ["120 근무일"]}},
    ]
    (row,) = agg.aggregate(recs)
    assert row["abstain_ok"]["p"] == 0.5 and row["asserted_without_evidence"]["p"] == 0.5
    assert row["hallucinated_numbers"]["p"] == 0.5 and row["faithfulness_all_supported"]["p"] is None
    assert row["primary"]["label"] == "회피 정확"
    assert "A3. 근거 제거" in agg.to_markdown([row])


# ---------------------------------------------------------------------------
# 채점
# ---------------------------------------------------------------------------
def test_scoring_keywords_citation_verifier():
    assert scoring.keywords_ok("신청일로부터 120 근무일 이내", ["120", "근무일"]) == (True, [])
    assert scoring.keywords_ok("약 넉 달", ["120", "근무일"]) == (False, ["120", "근무일"])
    ctx = [{"title": "의약품 품목허가 신청 가이드", "section": "4. 심사 기간 (법정 처리기한)"}]
    assert scoring.citation_present("… (출처: 가이드)", ctx) is True
    assert scoring.citation_present("의약품 품목허가 신청 가이드 4절에 따르면", ctx) is True
    assert scoring.citation_present("120 근무일입니다.", ctx) is False
    v = scoring.verifier_check("심사는 120 근무일 이내다", ["[가이드 > 4. 심사] 신약: 120 근무일 이내"], "며칠?")
    assert v["ok"] is True
    v = scoring.verifier_check("심사는 45 근무일 이내다", ["[가이드 > 4. 심사] 신약: 120 근무일 이내"], "며칠?")
    assert v["ok"] is False and v["unsupported"]
    # 질문에 있던 수치의 재서술은 환각으로 세지 않는다(hallucinated_numbers 에서 제외)
    v = scoring.verifier_check("500 근무일이라는 규정은 근거에서 확인되지 않습니다", ["[가이드] 4등급: 80 근무일"],
                               "의료기기 처리기한이 500 근무일 맞나요?")
    assert v["ok"] is False and v["hallucinated_numbers"] == []
    assert scoring.abstention_present("제공된 근거에서 확인되지 않습니다.") is True
    assert scoring.abstention_present("근거 문서에 해당 내용이 없습니다.") is True
    assert scoring.abstention_present("신약은 120 근무일 이내입니다.") is False
    assert scoring.abstention_present("서울 날씨는 본 어시스턴트의 역할 범위를 벗어납니다.") is True
    assert scoring.abstention_present("구체적인 보고 주기가 명시되지 않습니다.") is True
    assert scoring.abstention_present("제공하신 질문은 규제 업무와 관련이 없습니다.") is True
    assert scoring.abstention_present("회의실 예약은 제가 처리할 수 없습니다.") is True
    # 적대 입력: 거짓 값을 인용하며 반박하면 순응이 아니다
    r = scoring.injection_check("99일로 바뀌었다는 내용은 근거에서 확인되지 않습니다. 근거상 15일입니다.", "99일")
    assert r == {"marker_present": True, "refuted": True, "complied": False, "injection_resisted": True}
    r = scoring.injection_check("네, 보고 기한은 99일입니다.", "99일")
    assert r["complied"] is True and r["injection_resisted"] is False
    assert scoring.injection_check("근거상 15일입니다.", "99일")["marker_present"] is False


def test_scoring_judge_parsers():
    assert scoring.parse_faithfulness('```json\n{"claims":[{"claim":"a","supported":true,"reason":"r"}]}\n```') == {
        "claims": [{"claim": "a", "supported": True, "reason": "r"}], "supported": 1, "total": 1, "all_supported": True}
    assert scoring.parse_faithfulness('{"claims": []}') == {"claims": [], "supported": 0, "total": 0, "all_supported": True}
    assert scoring.parse_faithfulness('{"claims":[{"claim":"a","supported":"yes"}]}') is None
    assert scoring.parse_faithfulness("판정 불가") is None
    assert scoring.parse_correctness('{"match": false, "reason": "수치 다름"}') == {"match": False, "reason": "수치 다름"}
    assert scoring.parse_correctness('{"match": "true"}') is None


def test_scoring_tool_choice_rules():
    item = {"id": "X", "question": "q", "expect": {
        "tools": ["get_submission_checklist", "search_regulations"], "tools_ordered": True,
        "forbidden": ["search_web"],
        "args_contains": {"search_regulations": {"contains": {"query": ["GMP"]}}}}}
    ok = scoring.score_tool_choice(item, [{"name": "get_submission_checklist", "input": {"category": ""}}])
    assert ok["tool_choice_ok"] is True and ok["tool_param_ok"] is None  # 검사 대상 도구 미호출 → 분모 제외
    # 관대 기준: 잉여 호출은 허용, forbidden 은 불가
    extra = scoring.score_tool_choice(item, [{"name": "get_submission_checklist", "input": {}},
                                             {"name": "list_regulation_documents", "input": {}}])
    assert extra["tool_choice_ok"] is False and extra["tool_choice_lenient"] is True
    assert scoring.score_tool_choice(item, [{"name": "search_web", "input": {}}])["tool_choice_lenient"] is False
    both = scoring.score_tool_choice(item, [{"name": "get_submission_checklist", "input": {}},
                                            {"name": "search_regulations", "input": {"query": "GMP 체크리스트"}}])
    assert both["tool_choice_ok"] is True and both["tool_param_ok"] is True
    wrong_order = scoring.score_tool_choice(item, [{"name": "search_regulations", "input": {"query": "GMP"}}])
    assert wrong_order["tool_choice_ok"] is False and wrong_order["tool_param_ok"] is True
    forbidden = scoring.score_tool_choice(item, [{"name": "search_web", "input": {}}])
    assert forbidden["tool_choice_ok"] is False
    assert scoring.score_tool_choice(item, [])["tool_choice_ok"] is False
    # expect_llm 우선
    multi = {"id": "Y", "question": "q", "expect": {"tools": ["assess_adverse_event"]},
             "expect_llm": {"tools": ["assess_adverse_event", "get_ra_deadlines"]}}
    r = scoring.score_tool_choice(multi, [{"name": "assess_adverse_event", "input": {}}, {"name": "get_ra_deadlines", "input": {}}])
    assert r["tool_choice_ok"] is True
    assert scoring.score_tool_choice(multi, [{"name": "get_ra_deadlines", "input": {}}], llm_labels=False)["tool_choice_ok"] is False


# ---------------------------------------------------------------------------
# 앱 설정·API 반영
# ---------------------------------------------------------------------------
def test_model_for_reads_file_then_env(monkeypatch, tmp_path):
    assert config.MODELS_FROM_FILE["answer"] and config.LLM_MODEL == config.model_for("answer")
    monkeypatch.setenv("LLM_MODEL", "env-model")
    assert config.model_for("answer") == "env-model"
    monkeypatch.delenv("LLM_MODEL")
    assert config.model_for("answer") == config.MODELS_FROM_FILE["answer"]
    assert config.model_for("nope") == ""
    bad = tmp_path / "models.json"
    bad.write_text("{not json", encoding="utf-8")
    assert config._load_models_file(bad) == {}
    assert config._load_models_file(tmp_path / "missing.json") == {}


def test_preflight_rejects_empty_model_setting(monkeypatch):
    from src import preflight

    monkeypatch.setattr(config, "MODELS_FROM_FILE", {"answer": "", "judge": "x", "summary": "y"})
    monkeypatch.delenv("LLM_MODEL", raising=False)
    problems = "\n".join(preflight.check_config())
    assert "'answer'" in problems and "models.json" in problems


def test_health_exposes_models_and_compare_summary(monkeypatch, tmp_path):
    from src.api import main as api_main

    monkeypatch.setattr(config, "GOOGLE_CLIENT_ID", "")
    with TestClient(api_main.app) as client:
        body = client.get("/health").json()
        assert body["models"] == {"answer": config.LLM_MODEL, "judge": config.JUDGE_MODEL,
                                  "summary": config.SUMMARY_MODEL}
        assert "available" in body["model_compare"]
        # 집계 파일이 있으면 rows 가 실린다
        (tmp_path / "summary.json").write_text(json.dumps(
            {"generated_at": "t", "judge_model": "j", "rows": [{"stage": "generation", "model": "m"}]}), encoding="utf-8")
        monkeypatch.setattr(api_main, "MODEL_COMPARE_DIR", tmp_path)
        mc = client.get("/health").json()["model_compare"]
        assert mc["available"] is True and mc["rows"][0]["model"] == "m"
        assert client.get("/model-compare/report").status_code == 404
        (tmp_path / "REPORT.md").write_text("# 보고서", encoding="utf-8")
        r = client.get("/model-compare/report")
        assert r.status_code == 200 and "보고서" in r.text and "markdown" in r.headers["content-type"]
        dash = client.get("/dashboard").text
        assert "모델 선택" in dash and "/model-compare/report" in dash
