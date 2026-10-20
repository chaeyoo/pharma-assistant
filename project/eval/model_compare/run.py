"""모델 비교 실험 실행기 — 같은 입력으로 모델 N개를 돌려 질문별 원시 결과(JSONL)와
모델별 집계(CSV·MD·JSON)·보고서(REPORT.md)를 만든다.

단계·층
  A. generation (난이도 층 5개, 전 모델에 같은 시스템 프롬프트·같은 [근거] 블록, 도구 없음)
     simple       : QA 32문항, 검색 top_n 컨텍스트(정답이 들어 있음).
     complex      : 복합 12문항, sub_queries 각각의 검색 결과를 합친 컨텍스트(근거 2개 결합).
     no_evidence  : QA 문항 중 '정답 문서를 뺀' 컨텍스트 변형 — 기대는 회피. 새 라벨 없이
                    기존 라벨에서 파생(환각 압박 테스트).
     out_of_scope : 범위 밖 8문항(abstention_dataset) — 기대는 회피.
     adversarial  : 지시 주입·거짓 전제(robustness_dataset 의 marker 있는 항목) — marker 부재가 정상.
  B. tool_selection : plan 18문항. 운영 SYSTEM_PROMPT + MCP 도구 스키마(운영과 같은 노출 규칙:
     search_web 은 명시 요청 턴에만)로 1스텝만 호출해 첫 tool_use 를 채점한다.

정직성 규칙(코드로 강제)
  - ANTHROPIC_API_KEY 가 없으면 숫자를 만들지 않고 안내 후 종료한다.
  - gold_answers.json·complex_dataset.json 은 reviewed=true 일 때만 쓴다.
  - 프롬프트는 전 모델 공통 파일 1개이며 모델별 분기가 없다.
  - 실제 보낸 요청 파라미터(request_params)를 레코드마다 기록한다.
  - 단가가 설정에 없는 모델의 비용은 null 로 남긴다(추정 금지).

실행:  python -m eval.model_compare.run [--stage a|b|all] [--runs N] [--workers N] [--out DIR]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from src import config
from eval.model_compare import aggregate as agg
from eval.model_compare import scoring
from eval.model_compare.gateway import AnthropicGateway, Completion, LLMGateway

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "prompts"
CONFIG_FILE = HERE / "config.json"
STAGE_GENERATION = "generation"
STAGE_TOOL = "tool_selection"
ALL_TIERS = ("simple", "complex", "no_evidence", "out_of_scope", "adversarial")


# ---------------------------------------------------------------------------
# 입력 로딩
# ---------------------------------------------------------------------------
def load_config(path: Path = CONFIG_FILE) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for key in ("models", "judge_model", "runs_per_question", "max_tokens", "context_top_n", "pricing", "datasets"):
        if key not in cfg:
            raise ValueError(f"config.json 에 '{key}' 가 없다")
    if not cfg["models"]:
        raise ValueError("config.json 의 models 목록이 비어 있다")
    if cfg["judge_model"] in cfg["models"]:
        raise ValueError("judge_model 은 비교 대상 models 와 달라야 한다(self-critique 순환 방지)")
    cfg.setdefault("tiers", list(ALL_TIERS))
    unknown = [t for t in cfg["tiers"] if t not in ALL_TIERS]
    if unknown:
        raise ValueError(f"알 수 없는 tier: {unknown} (가능: {ALL_TIERS})")
    return cfg


def load_prompt(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8").strip()


def _read(rel: str) -> dict:
    return json.loads((config.BASE_DIR / rel).read_text(encoding="utf-8"))


def load_qa_items(cfg: dict) -> list[dict]:
    """qa_dataset 에는 id 가 없다 — 순번으로 Q01..Qnn 을 부여한다(gold_answers 와 같은 규칙)."""
    return [{"id": f"Q{i:02d}", **it} for i, it in enumerate(_read(cfg["datasets"]["generation"])["items"], 1)]


def load_plan_items(cfg: dict) -> list[dict]:
    return _read(cfg["datasets"]["tool_selection"])["items"]


def _require_reviewed(rel: str, data: dict, require: bool) -> None:
    if require and data.get("reviewed") is not True:
        raise RuntimeError(
            f"{rel} 가 아직 검토되지 않았다(reviewed=false) — 발췌 정답을 사람이 확인한 뒤 "
            "reviewed=true 로 바꿔야 실행한다"
        )


def load_gold(cfg: dict, require_reviewed: bool = True) -> dict[str, dict]:
    rel = cfg["datasets"]["gold_answers"]
    data = _read(rel)
    _require_reviewed(rel, data, require_reviewed)
    return {it["id"]: it for it in data["items"]}


def load_complex_items(cfg: dict, require_reviewed: bool = True) -> list[dict]:
    rel = cfg["datasets"]["complex"]
    data = _read(rel)
    _require_reviewed(rel, data, require_reviewed)
    return data["items"]


def load_out_of_scope_items(cfg: dict) -> list[dict]:
    return [{"id": f"OOS{i:02d}", **it} for i, it in enumerate(_read(cfg["datasets"]["out_of_scope"])["items"], 1)]


def load_adversarial_items(cfg: dict) -> list[dict]:
    """robustness_dataset 의 적대 항목 중 marker 가 있고 질문형인 것(도구 입력 주입 제외)."""
    return [it for it in _read(cfg["datasets"]["adversarial"])["items"]
            if it.get("kind") == "adversarial" and it.get("marker") and it.get("question")]


# ---------------------------------------------------------------------------
# 고정 입력 구성 — 검색 결과(컨텍스트)와 도구 스키마는 모델과 무관하게 1회만 만든다
# ---------------------------------------------------------------------------
def _ctx_from_chunks(scored: list) -> dict:
    lines = [
        f"[근거 {i}] (출처: {s.chunk.title} · {s.chunk.source} · {s.chunk.section})\n{s.chunk.text.strip()}"
        for i, s in enumerate(scored, 1)
    ]
    return {
        "chunk_ids": [s.chunk.chunk_id for s in scored],
        "doc_ids": sorted({s.chunk.doc_id for s in scored}),
        "block": "\n\n".join(lines) if lines else "(제공된 근거 없음)",
        "texts": [s.chunk.text.strip() for s in scored],
        "meta": [{"doc_id": s.chunk.doc_id, "title": s.chunk.title, "section": s.chunk.section} for s in scored],
    }


def build_contexts(items: list[dict], pipeline, top_n: int) -> dict[str, dict]:
    """simple·out_of_scope·adversarial: 질문 1회 검색(운영 search_regulations top_n 과 같은 경로)."""
    return {it["id"]: _ctx_from_chunks(pipeline.retrieve(it["question"], rerank_n=top_n).chunks) for it in items}


def build_complex_contexts(items: list[dict], pipeline, top_n: int) -> dict[str, dict]:
    """complex: sub_queries 각각 top_n 검색 → chunk_id 로 중복 제거해 합친다(운영에서 검색 2회에 해당)."""
    out: dict[str, dict] = {}
    for it in items:
        seen: set[str] = set()
        merged = []
        for q in it["sub_queries"]:
            for s in pipeline.retrieve(q, rerank_n=top_n).chunks:
                if s.chunk.chunk_id not in seen:
                    seen.add(s.chunk.chunk_id)
                    merged.append(s)
        out[it["id"]] = _ctx_from_chunks(merged)
    return out


def _gold_ids(item: dict) -> set[str]:
    return set(item.get("accept_doc_ids") or []) | {item["relevant_doc_id"]}


def build_no_evidence_contexts(items: list[dict], pipeline, top_n: int, candidates: int) -> dict[str, dict]:
    """no_evidence: 넓게 회수(candidates)한 뒤 정답 문서(relevant_doc_id·accept_doc_ids)의 청크를
    전부 제거하고 top_n 만 남긴다 — '관련은 있어 보이지만 답은 없는' 컨텍스트."""
    out: dict[str, dict] = {}
    for it in items:
        gold = _gold_ids(it)
        scored = pipeline.retrieve(it["question"], top_k=max(candidates, top_n), rerank_n=candidates).chunks
        kept = [s for s in scored if s.chunk.doc_id not in gold][:top_n]
        out[it["id"]] = _ctx_from_chunks(kept)
        out[it["id"]]["removed_doc_ids"] = sorted(gold)
    return out


async def _fetch_tool_schemas() -> list[dict]:
    from src.agent.agent import _mcp_client, _to_anthropic_tools

    async with _mcp_client() as client:
        return _to_anthropic_tools(await client.list_tools())


def fetch_tool_schemas() -> list[dict]:
    """운영 에이전트와 같은 경로(MCP list_tools → Anthropic tools 포맷)로 도구 스키마를 받는다."""
    return asyncio.run(_fetch_tool_schemas())


def tools_for_question(tools: list[dict], question: str) -> list[dict]:
    """운영과 같은 노출 규칙 — search_web 은 명시 요청 턴에만(agent._wants_web_search)."""
    from src.agent.agent import _wants_web_search

    if _wants_web_search(question):
        return tools
    return [t for t in tools if t["name"] != "search_web"]


# ---------------------------------------------------------------------------
# 호출 1건 → 레코드 1건
# ---------------------------------------------------------------------------
def _status(c: Completion, expect_tool: bool) -> str:
    if c.error:
        return "timeout" if "Timeout" in c.error else "api_error"
    if expect_tool:
        return "ok" if c.tool_uses else "schema_error"
    return "ok" if c.text.strip() else "empty"


def _usage(c: Completion, model: str, pricing: dict) -> dict:
    return {
        "input_tokens": c.input_tokens, "output_tokens": c.output_tokens,
        "cache_read_tokens": c.cache_read_tokens, "cache_write_tokens": c.cache_write_tokens,
        "cost_usd": agg.cost_usd(c.input_tokens, c.output_tokens, c.cache_read_tokens,
                                 c.cache_write_tokens, pricing.get(model)),
        "latency_ms": round(c.latency_ms, 1),
    }


def _judge(gateway: LLMGateway, model: str, system: str, user: str, max_tokens: int,
           parser: Callable[[str], dict | None], pricing: dict) -> tuple[dict | None, str, dict]:
    c = gateway.complete(model=model, system=system,
                         messages=[{"role": "user", "content": user}], max_tokens=max_tokens)
    usage = _usage(c, model, pricing)
    if c.error:
        return None, "unavailable", usage
    parsed = parser(c.text)
    return parsed, ("ok" if parsed is not None else "schema_error"), usage


def deterministic_fields(tier: str, item: dict, ctx: dict, answer: str, status: str) -> dict:
    """저장된 답변만으로 다시 계산할 수 있는 결정론 지표 전부 — --rescore 가 재사용한다."""
    ok = status == "ok"
    keywords = list(item.get("keywords", []))
    kw_ok, missing = scoring.keywords_ok(answer, keywords) if ok else (False, keywords)
    verifier = (scoring.verifier_check(answer, ctx["texts"], item["question"]) if ok
                else {"ok": False, "unsupported": [], "question_origin": [], "hallucinated_numbers": [],
                      "direction_conflicts": [], "role_conflicts": [], "skipped": "no_answer"})
    abstained = scoring.abstention_present(answer) if ok else False
    out: dict = {
        "keywords": keywords, "correctness_keywords": kw_ok, "missing_keywords": missing,
        "citation_present": scoring.citation_present(answer, ctx["meta"]) if ok else False,
        "abstained": abstained, "verifier": verifier,
    }
    if tier == "complex":
        out["context_covers_gold"] = set(item.get("doc_ids", [])) <= set(ctx["doc_ids"])
    if tier in ("no_evidence", "out_of_scope"):
        out["abstain_ok"] = ok and abstained and not verifier["hallucinated_numbers"]
        if tier == "no_evidence":
            out["removed_doc_ids"] = ctx.get("removed_doc_ids", [])
            ctx_text = "\n".join(ctx["texts"]).lower()
            # 누출: 정답 문서를 뺐는데도 다른 문서의 참조문에 핵심어가 전부 남아 있음 — 이 문항은
            # '근거 없음'이 아니므로 회피 분모에서 제외한다(코퍼스 상호참조의 실측 한계).
            out["evidence_leaked"] = bool(keywords) and all(k.lower() in ctx_text for k in keywords)
            # 근거 없이 정답을 단정 — 핵심어를 전부 말하면서 회피 표현이 없음(누출 문항 제외).
            out["asserted_without_evidence"] = ok and kw_ok and not abstained and not out["evidence_leaked"]
    if tier == "adversarial":
        out["marker"] = item.get("marker")
        inj = scoring.injection_check(answer, item.get("marker")) if ok else \
            {"marker_present": False, "refuted": False, "complied": False, "injection_resisted": False}
        out.update(inj)
    return out


def generation_record(cfg: dict, tier: str, model: str, run: int, item: dict, ctx: dict,
                      gold_answer: str | None, gateway: LLMGateway, judge_gateway: LLMGateway,
                      prompts: dict) -> dict:
    user = f"[근거]\n{ctx['block']}\n\n[질문]\n{item['question']}"
    c = gateway.complete(model=model, system=prompts["generation"],
                         messages=[{"role": "user", "content": user}], max_tokens=int(cfg["max_tokens"]))
    status = _status(c, expect_tool=False)
    answer = c.text.strip()
    keywords = list(item.get("keywords", []))
    kw_ok, missing = scoring.keywords_ok(answer, keywords) if status == "ok" else (False, keywords)
    verifier = (scoring.verifier_check(answer, ctx["texts"], item["question"]) if status == "ok"
                else {"ok": False, "unsupported": [], "question_origin": [], "hallucinated_numbers": [],
                      "direction_conflicts": [], "role_conflicts": [], "skipped": "no_answer"})
    rec: dict = {
        "stage": STAGE_GENERATION, "tier": tier, "model": model, "run": run,
        "question_id": item["id"], "question": item["question"],
        "context_chunk_ids": ctx["chunk_ids"], "context_doc_ids": ctx["doc_ids"],
        "request_params": c.request_params,
        "status": status, "error": c.error, "stop_reason": c.stop_reason,
        "answer": answer, **_usage(c, model, cfg["pricing"]),
        "keywords": keywords, "correctness_keywords": kw_ok, "missing_keywords": missing,
        "judge_model": cfg["judge_model"],
        "correctness_judge": None, "correctness_judge_status": "skipped",
        "faithfulness_judge": None, "faithfulness_judge_status": "skipped",
        "judge_usage": None,
    }
    rec.update(deterministic_fields(tier, item, ctx, answer, status))
    if status != "ok":
        return rec
    jmax = int(cfg.get("judge_max_tokens", 1024))
    f_res, f_status, f_usage = _judge(judge_gateway, cfg["judge_model"], prompts["judge_faithfulness"],
                                      f"[근거]\n{ctx['block']}\n\n[답변]\n{answer}", jmax,
                                      scoring.parse_faithfulness, cfg["pricing"])
    rec["faithfulness_judge"], rec["faithfulness_judge_status"] = f_res, f_status
    c_usage: dict | None = None
    if gold_answer is not None:
        c_res, c_status, c_usage = _judge(
            judge_gateway, cfg["judge_model"], prompts["judge_correctness"],
            f"[질문]\n{item['question']}\n\n[기대 답]\n{gold_answer}\n\n[모델 답변]\n{answer}", jmax,
            scoring.parse_correctness, cfg["pricing"])
        rec["correctness_judge"], rec["correctness_judge_status"] = c_res, c_status
    costs = [f_usage["cost_usd"]] + ([c_usage["cost_usd"]] if c_usage else [])
    rec["judge_usage"] = {
        "input_tokens": f_usage["input_tokens"] + (c_usage or {}).get("input_tokens", 0),
        "output_tokens": f_usage["output_tokens"] + (c_usage or {}).get("output_tokens", 0),
        "cost_usd": None if any(x is None for x in costs) else round(sum(costs), 6),
    }
    return rec


def tool_record(cfg: dict, model: str, run: int, item: dict, tools: list[dict],
                gateway: LLMGateway, system_prompt: str) -> dict:
    exposed = tools_for_question(tools, item["question"])
    messages = [*item.get("history", []), {"role": "user", "content": item["question"]}]
    c = gateway.complete(model=model, system=system_prompt, messages=messages,
                         max_tokens=int(cfg["max_tokens"]), tools=exposed)
    status = _status(c, expect_tool=True)
    return {
        "stage": STAGE_TOOL, "tier": STAGE_TOOL, "model": model, "run": run,
        "question_id": item["id"], "question": item["question"],
        "context_chunk_ids": [], "request_params": c.request_params,
        "status": status, "error": c.error, "stop_reason": c.stop_reason,
        "answer": c.text.strip(), "tool_uses": c.tool_uses, **_usage(c, model, cfg["pricing"]),
        **tool_fields(item, c.tool_uses, status),
    }


def tool_fields(item: dict, tool_uses: list[dict], status: str) -> dict:
    score = scoring.score_tool_choice(item, tool_uses) if status == "ok" else \
        {"tool_choice_ok": False, "tool_choice_lenient": False, "tool_param_ok": None, "failures": [f"status={status}"]}
    return {
        "tool_choice_ok": score["tool_choice_ok"], "tool_choice_lenient": score["tool_choice_lenient"],
        "tool_param_ok": score["tool_param_ok"], "tool_failures": score["failures"],
        "expected_tools": (item.get("expect_llm") or item["expect"])["tools"],
    }


# ---------------------------------------------------------------------------
# 전체 실행
# ---------------------------------------------------------------------------
def prepare_generation(cfg: dict, pipeline, gold: dict[str, dict] | None = None,
                       complex_items: list[dict] | None = None) -> dict[str, dict]:
    """층별 (items, contexts, gold_by_id) — 검색은 여기서 1회만 돈다."""
    top_n = int(cfg["context_top_n"])
    tiers: dict[str, dict] = {}
    qa = load_qa_items(cfg)
    if "simple" in cfg["tiers"]:
        gold = gold if gold is not None else load_gold(cfg)
        tiers["simple"] = {"items": qa, "contexts": build_contexts(qa, pipeline, top_n),
                           "gold": {k: v["gold_answer"] for k, v in gold.items()}}
    if "complex" in cfg["tiers"]:
        items = complex_items if complex_items is not None else load_complex_items(cfg)
        tiers["complex"] = {"items": items, "contexts": build_complex_contexts(items, pipeline, top_n),
                            "gold": {it["id"]: it["gold_answer"] for it in items}}
    if "no_evidence" in cfg["tiers"]:
        items = [{**it, "id": it["id"].replace("Q", "NE", 1)} for it in qa]
        tiers["no_evidence"] = {"items": items, "gold": {},
                                "contexts": build_no_evidence_contexts(items, pipeline, top_n,
                                                                       int(cfg.get("no_evidence_candidates", 12)))}
    if "out_of_scope" in cfg["tiers"]:
        items = load_out_of_scope_items(cfg)
        tiers["out_of_scope"] = {"items": items, "contexts": build_contexts(items, pipeline, top_n), "gold": {}}
    if "adversarial" in cfg["tiers"]:
        items = load_adversarial_items(cfg)
        tiers["adversarial"] = {"items": items, "contexts": build_contexts(items, pipeline, top_n), "gold": {}}
    return tiers


def run_experiment(
    cfg: dict, gateway: LLMGateway, out_dir: Path, *, stage: str = "all",
    runs: int | None = None, workers: int = 1, pipeline=None, tools: list[dict] | None = None,
    judge_gateway: LLMGateway | None = None, gold: dict[str, dict] | None = None,
    complex_items: list[dict] | None = None, progress: Callable[[str], None] | None = None,
) -> dict:
    """실험 전체를 돌리고 산출물을 out_dir 에 쓴다. 반환: {"records", "rows", "paths", "meta"}.

    gateway/judge_gateway/pipeline/tools/gold/complex_items 를 주입할 수 있어 테스트는 LLM 없이 돈다.
    """
    runs = int(runs or cfg["runs_per_question"])
    judge_gateway = judge_gateway or gateway
    prompts = {
        "generation": load_prompt("generation_system.txt"),
        "judge_faithfulness": load_prompt("judge_faithfulness.txt"),
        "judge_correctness": load_prompt("judge_correctness.txt"),
    }
    say = progress or (lambda s: None)
    tasks: list[Callable[[], dict]] = []
    tier_sizes: dict[str, int] = {}

    if stage in ("all", "a"):
        from src.rag.pipeline import RagPipeline

        pipeline = pipeline or RagPipeline().build()
        tiers = prepare_generation(cfg, pipeline, gold=gold, complex_items=complex_items)
        for tier, t in tiers.items():
            tier_sizes[tier] = len(t["items"])
            for model in cfg["models"]:
                for run in range(1, runs + 1):
                    for it in t["items"]:
                        tasks.append(lambda tr=tier, m=model, r=run, i=it, tt=t: generation_record(
                            cfg, tr, m, r, i, tt["contexts"][i["id"]], tt["gold"].get(i["id"]),
                            gateway, judge_gateway, prompts))
        say("검색 고정 완료: " + ", ".join(f"{k}={v}" for k, v in tier_sizes.items()))
    if stage in ("all", "b"):
        from src.agent.agent import SYSTEM_PROMPT

        plan_items = load_plan_items(cfg)
        tools = tools if tools is not None else fetch_tool_schemas()
        for model in cfg["models"]:
            for run in range(1, runs + 1):
                for it in plan_items:
                    tasks.append(lambda m=model, r=run, i=it: tool_record(cfg, m, r, i, tools, gateway, SYSTEM_PROMPT))
        tier_sizes[STAGE_TOOL] = len(plan_items)

    say(f"호출 시작: {len(tasks)}건 (workers={workers})")
    records: list[dict] = []
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for k, rec in enumerate(ex.map(lambda t: t(), tasks), 1):
                records.append(rec)
                if k % 20 == 0:
                    say(f"  {k}/{len(tasks)}")
    else:
        for k, t in enumerate(tasks, 1):
            records.append(t())
            if k % 20 == 0:
                say(f"  {k}/{len(tasks)}")
    tier_order = {t: i for i, t in enumerate((*ALL_TIERS, STAGE_TOOL))}
    model_order = {m: i for i, m in enumerate(cfg["models"])}
    records.sort(key=lambda r: (tier_order[r["tier"]], model_order[r["model"]], r["question_id"], r["run"]))

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rows = agg.aggregate(records)
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": stage, "runs_per_question": runs, "workers": workers,
        "tier_sizes": tier_sizes, "models": cfg["models"], "judge_model": cfg["judge_model"],
        "judge_cost_usd": round(sum((r.get("judge_usage") or {}).get("cost_usd") or 0.0 for r in records), 4),
        "pricing_source_url": cfg.get("pricing_source_url"), "pricing_checked_on": cfg.get("pricing_checked_on"),
    }
    paths = agg.write_outputs(rows, out_dir, meta)
    paths["jsonl"] = out_dir / "results.jsonl"
    return {"records": records, "rows": rows, "paths": paths, "meta": meta}


def canary(cfg: dict, gateway: LLMGateway) -> dict[str, str | None]:
    """본 실행 전 모델별 1회 최소 호출 — 키·모델명·권한 문제를 321건이 아니라 4건에서 잡는다.
    반환: {model: None(정상) | 예외 타입명}."""
    out: dict[str, str | None] = {}
    for m in [*cfg["models"], cfg["judge_model"]]:
        c = gateway.complete(model=m, system="ping", messages=[{"role": "user", "content": "ping"}], max_tokens=5)
        out[m] = c.error
    return out


def write_failure_report(out_dir: Path, cfg: dict, reason: str, details: dict[str, str | None]) -> Path:
    """실행하지 못한 사실만 담은 REPORT.md — 숫자 없는 보고서. 이전 실행의 집계 파일이 남아
    계기판·README 가 옛 숫자를 '최근 결과'로 읽지 않도록 함께 지운다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("results.jsonl", "summary.csv", "summary.md", "summary.json"):
        (out_dir / name).unlink(missing_ok=True)
    lines = [
        "# 모델 성능 비교 → 모델 선택 실험 보고서",
        "",
        f"**실행 실패 — 측정값 없음.** {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        f"사유: {reason}",
        "",
        "| 모델 | canary 결과 |",
        "|---|---|",
        *[f"| {m} | {e or 'OK'} |" for m, e in details.items()],
        "",
        "실행하지 않은 숫자는 쓰지 않는다. 키·모델명을 바로잡은 뒤 `python -m eval.model_compare.run` 을 다시 실행하면 이 파일이 실측 보고서로 대체된다.",
        "",
    ]
    path = out_dir / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def rescore(cfg: dict, out_dir: Path, *, pipeline=None, judge_gateway: LLMGateway | None = None,
            progress: Callable[[str], None] | None = None) -> dict:
    """results.jsonl 의 저장된 답변으로 결정론 지표만 다시 계산한다(API 호출 없음) — 채점 규칙을
    고친 뒤 재호출 비용 없이 재집계하기 위한 경로. 검색은 결정론이라 컨텍스트를 같은 코드로 재구성한다.
    judge_gateway 를 주면 심판 상태가 schema_error/unavailable 인 레코드만 다시 판정한다(현재 judge_max_tokens).
    """
    say = progress or (lambda s: None)
    path = out_dir / "results.jsonl"
    records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    meta = json.loads((out_dir / "summary.json").read_text(encoding="utf-8")) if (out_dir / "summary.json").exists() else {}
    gen = [r for r in records if r["stage"] == STAGE_GENERATION]
    if gen:
        from src.rag.pipeline import RagPipeline

        pipeline = pipeline or RagPipeline().build()
        tiers = prepare_generation(cfg, pipeline)
        prompts = {"judge_faithfulness": load_prompt("judge_faithfulness.txt"),
                   "judge_correctness": load_prompt("judge_correctness.txt")}
        jmax = int(cfg.get("judge_max_tokens", 1024))
        rejudged = 0
        for r in gen:
            t = tiers[r["tier"]]
            item = next(i for i in t["items"] if i["id"] == r["question_id"])
            ctx = t["contexts"][item["id"]]
            assert ctx["chunk_ids"] == r["context_chunk_ids"], f"컨텍스트 불일치: {r['tier']} {r['question_id']}"
            r.update(deterministic_fields(r["tier"], item, ctx, r["answer"], r["status"]))
            if judge_gateway is not None and r["status"] == "ok":
                extra_cost = 0.0
                if r.get("faithfulness_judge_status") in ("schema_error", "unavailable"):
                    res, st, u = _judge(judge_gateway, cfg["judge_model"], prompts["judge_faithfulness"],
                                        f"[근거]\n{ctx['block']}\n\n[답변]\n{r['answer']}", jmax,
                                        scoring.parse_faithfulness, cfg["pricing"])
                    r["faithfulness_judge"], r["faithfulness_judge_status"] = res, st
                    extra_cost += u["cost_usd"] or 0.0
                    rejudged += 1
                gold = t["gold"].get(item["id"])
                if gold is not None and r.get("correctness_judge_status") in ("schema_error", "unavailable"):
                    res, st, u = _judge(judge_gateway, cfg["judge_model"], prompts["judge_correctness"],
                                        f"[질문]\n{item['question']}\n\n[기대 답]\n{gold}\n\n[모델 답변]\n{r['answer']}",
                                        jmax, scoring.parse_correctness, cfg["pricing"])
                    r["correctness_judge"], r["correctness_judge_status"] = res, st
                    extra_cost += u["cost_usd"] or 0.0
                    rejudged += 1
                if extra_cost and r.get("judge_usage"):
                    r["judge_usage"]["cost_usd"] = round((r["judge_usage"].get("cost_usd") or 0.0) + extra_cost, 6)
        say(f"재채점 {len(gen)}건, 재판정 {rejudged}건")
    for r in records:
        if r["stage"] == STAGE_TOOL:
            item = next(i for i in load_plan_items(cfg) if i["id"] == r["question_id"])
            r.update(tool_fields(item, r.get("tool_uses", []), r["status"]))
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rows = agg.aggregate(records)
    meta = {k: v for k, v in meta.items() if k != "rows"}
    meta["rescored_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta["judge_cost_usd"] = round(sum((r.get("judge_usage") or {}).get("cost_usd") or 0.0 for r in records), 4)
    paths = agg.write_outputs(rows, out_dir, meta)
    paths["jsonl"] = path
    return {"records": records, "rows": rows, "paths": paths, "meta": meta}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="모델 비교 실험")
    ap.add_argument("--stage", choices=["a", "b", "all"], default="all")
    ap.add_argument("--runs", type=int, default=None, help="질문당 모델당 실행 횟수(기본: config)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--rescore", action="store_true",
                    help="API 재호출 없이 저장된 답변으로 결정론 지표만 재계산·재집계·보고서 재생성")
    ap.add_argument("--rejudge-failed", action="store_true",
                    help="--rescore 와 함께: 심판 판정이 schema_error/unavailable 인 레코드만 다시 판정(키 필요)")
    args = ap.parse_args(argv)

    cfg = load_config()
    if args.rescore:
        from eval.model_compare.report import build_report

        jg = None
        if args.rejudge_failed:
            if not config.ANTHROPIC_API_KEY:
                print("--rejudge-failed 는 ANTHROPIC_API_KEY 가 필요하다")
                return 1
            jg = AnthropicGateway(config.ANTHROPIC_API_KEY, float(cfg.get("timeout_s", 120)), int(cfg.get("max_retries", 0)))
        res = rescore(cfg, args.out, judge_gateway=jg, progress=print)
        report = build_report(args.out, cfg, res["rows"], res["records"], res["meta"])
        print(agg.to_markdown(res["rows"]))
        print(f"보고서: {report}")
        return 0
    if not config.ANTHROPIC_API_KEY:
        print("ANTHROPIC_API_KEY 가 없다 — 실험을 실행하지 않는다(숫자를 만들지 않음). "
              "project/.env 에 키를 넣고 `set -a; . ./.env; set +a` 후 다시 실행할 것.")
        return 1
    try:
        if "simple" in cfg["tiers"]:
            load_gold(cfg)
        if "complex" in cfg["tiers"]:
            load_complex_items(cfg)
    except RuntimeError as e:
        print(str(e))
        return 1
    gateway = AnthropicGateway(config.ANTHROPIC_API_KEY, float(cfg.get("timeout_s", 120)),
                               int(cfg.get("max_retries", 0)))
    from src.mcp_server.http_harness import run_mcp_http_server
    from eval.model_compare.report import build_report

    probe = canary(cfg, gateway)
    bad = {m: e for m, e in probe.items() if e}
    if bad:
        path = write_failure_report(args.out, cfg, "본 실행 전 canary 호출 실패 — " + ", ".join(f"{m}: {e}" for m, e in bad.items()), probe)
        print(f"canary 실패 — 실험을 실행하지 않는다(숫자를 만들지 않음): {bad}\n"
              f"ANTHROPIC_API_KEY(유효성·길이)와 config.json 의 모델명을 확인할 것. 실패 기록: {path}")
        return 1
    print("canary 통과: " + ", ".join(probe))

    with run_mcp_http_server() as url:
        config.MCP_SERVER_URL = url
        res = run_experiment(cfg, gateway, args.out, stage=args.stage, runs=args.runs,
                             workers=args.workers, progress=print)
    report = build_report(args.out, cfg, res["rows"], res["records"], res["meta"])
    n_fail = sum(1 for r in res["records"] if r["status"] != "ok")
    if res["records"] and n_fail == len(res["records"]):
        print(f"경고: 전 호출({n_fail}건) 실패 — 집계 표의 0 은 측정값이 아니라 실패다. 보고서 (c) 의 오류 타입을 확인할 것.")
        return 2
    print(agg.to_markdown(res["rows"]))
    print(f"원시 결과: {res['paths']['jsonl']}\n집계: {res['paths']['csv']} · {res['paths']['md']}\n보고서: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
