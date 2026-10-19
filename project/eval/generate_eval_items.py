"""LLM 기반 평가셋 후보 생성 — 평가셋을 '정적 게이트'에서 '성장하는 자산'으로.

기존 평가셋(qa·holdout·plan·robustness)은 수동 작성 + 실패 분석 마이닝으로
자랐다. 이 스크립트는 세 번째 확장 경로를 연다: 코퍼스와 기존 문항을 LLM 에
주고 **후보**를 생성하게 한다. 전략 4종은 평가 방법론의 목표지향 생성 기법과
1:1 이다:

  - paraphrase   : 기존 문항의 재표현 — 같은 정답, 다른 어휘(어휘 불일치 견고성)
  - ambiguous    : 모호화 — 정보를 덜 주고 물어 회수 난도를 올린 변형
  - mixed_intent : 분포 보간 — 두 의도(RA 기한 + PV 케이스 등)를 한 질문에 결합
  - adversarial  : 거짓 전제·유도 변형 — robustness 셋의 후보

**리뷰 게이트가 이 스크립트의 핵심 설계다** — 생성물은 어떤 경로로도 평가셋에
자동 병합되지 않는다:

  1. 산출은 eval/generated_candidates.json **한 곳뿐**이고, 모든 항목이
     status="pending_review" 로 태어난다. qa/holdout/plan/robustness 파일은
     이 스크립트가 열지도 쓰지도 않는다(읽기는 중복 제거용뿐).
  2. 병합은 사람이 한다 — 검토 후 항목을 대상 평가셋으로 **옮기고**(복사가
     아니라 이동: 이 파일에서 제거) note 에 검토 근거를 남긴다. 홀드아웃
     규율(_comment)과 같은 원칙: 평가셋의 라벨은 사람이 책임지는 계약이고,
     LLM 생성 라벨을 무검토로 편입하면 '평가로 평가를 오염'시키는 순환이
     생긴다(측정 도구의 눈금을 측정 대상이 그리는 형태).
  3. 자가 검증은 **라벨이지 필터가 아니다** — target_doc_id 실존 검사(불실존은
     탈락: 코퍼스에 없는 정답 라벨은 검토 불가능한 후보다)를 빼면, 검색 1위
     일치 여부(retrieval_verified)는 참/거짓 그대로 기록만 한다. false 후보를
     자동 폐기하지 않는 이유: 검색이 못 찾는 문항이야말로 가장 가치 있는
     후보일 수 있다(홀드아웃의 존재 이유) — 판단은 사람의 몫이다.

CI 미등록·키 필요 수동 도구다. 파싱·검증·no-auto-merge 계약(결정론 부분)은
tests/test_generate_eval.py 가 스텁으로 핀 고정한다(judge 와 같은 이원화).

실행:  python -m eval.generate_eval_items [--n 5] [--out eval/generated_candidates.json]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re

from src import config

STRATEGIES = ("paraphrase", "ambiguous", "mixed_intent", "adversarial")

_STRATEGY_BRIEF = {
    "paraphrase": "기존 문항을 같은 정답 문서를 겨냥한 채 완전히 다른 어휘(실무 구어 포함)로 재표현한 질문",
    "ambiguous": "기존 문항에서 핵심 용어를 일부 빼거나 두루뭉술하게 만들어 회수 난도를 올린 질문(여전히 정답 문서는 특정 가능해야 함)",
    "mixed_intent": "두 의도를 한 질문에 결합한 것 — 예: RA 마감 확인 + PV 케이스 판정, 규정 검색 + 체크리스트. target_doc_id 는 주 의도의 문서 하나만",
    "adversarial": "거짓 전제·특정 답 유도('~라고 들었는데 맞지?')를 심은 변형 — 견고성 평가셋(robustness) 후보",
}


def _load_json(name: str) -> dict:
    return json.loads((config.BASE_DIR / "eval" / name).read_text(encoding="utf-8"))


def _norm_question(q: str) -> str:
    """중복 판정용 정규화 — 공백·구두점 차이만으로 '새 문항'이 되지 않게."""
    return re.sub(r"[\s?.!,~'\"]+", "", q)


def existing_questions() -> set[str]:
    """기존 평가셋 + 후보 파일의 질문 전집(정규화) — 중복 생성 방지의 기준선."""
    out: set[str] = set()
    for name in ("qa_dataset.json", "holdout_dataset.json", "abstention_dataset.json",
                 "plan_dataset.json", "robustness_dataset.json"):
        for item in _load_json(name)["items"]:
            q = item.get("question") or item.get("case") or ""
            if q:
                out.add(_norm_question(q))
    try:
        for item in _load_json("generated_candidates.json")["items"]:
            out.add(_norm_question(item["question"]))
    except FileNotFoundError:
        pass
    return out


def _corpus_index() -> dict[str, str]:
    """{doc_id: title} — target_doc_id 실존 검사와 프롬프트 문서 목록의 원천."""
    from src.rag.loader import load_documents

    return {d.doc_id: d.title for d in load_documents(config.REG_DIR)}


def _parse_candidates(text: str) -> list[dict] | None:
    """모델 출력 → candidates 리스트. malformed 는 None(부분 파싱 잡음 금지 — judge 와 동일)."""
    m = re.search(r"\{.*\}", text.strip(), re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    cands = data.get("candidates") if isinstance(data, dict) else None
    if not isinstance(cands, list):
        return None
    return [c for c in cands if isinstance(c, dict)]


def _validate(cand: dict, strategy: str, doc_ids: set[str],
              existing: set[str]) -> tuple[dict | None, str]:
    """후보 1건 검증 — 반환 (정제된 후보 | None, 탈락 사유)."""
    q = str(cand.get("question", "")).strip()
    if len(q) < 8:
        return None, "질문이 비었거나 너무 짧음"
    if _norm_question(q) in existing:
        return None, f"기존 문항과 중복: {q!r}"
    target = str(cand.get("target_doc_id", "") or "").strip()
    if target and target not in doc_ids:
        # 코퍼스에 없는 정답 라벨은 사람이 검토할 방법이 없다 — 유일한 자동 탈락 사유
        return None, f"실존하지 않는 target_doc_id={target!r}: {q!r}"
    return {
        "question": q,
        "strategy": strategy,
        "target_doc_id": target or None,
        "source_question": str(cand.get("source_question", "") or "").strip() or None,
        "rationale": str(cand.get("rationale", "") or "").strip(),
    }, ""


def _verify_retrieval(cand: dict) -> dict:
    """검색 1위 일치 라벨 — 필터가 아니라 검토자를 위한 참고 정보다."""
    from src.mcp_server.server import search_regulations

    data = search_regulations(cand["question"], top_n=1)
    results = data.get("results", [])
    top = results[0]["doc_id"] if results else None
    return {
        "retrieved_doc_id": top,
        "retrieval_verified": (top == cand["target_doc_id"]) if cand["target_doc_id"] else None,
    }


def generate(n_per_strategy: int = 5) -> dict:
    """전략별 후보 생성 + 검증 라벨링. 반환 {"items", "rejected"} — 파일 쓰기는 main 만."""
    import anthropic  # 지연 import — 스텁·오프라인 배려(judge 와 동일)

    docs = _corpus_index()
    existing = existing_questions()
    qa_sample = [i["question"] for i in _load_json("qa_dataset.json")["items"][:12]]
    doc_list = "\n".join(f"- {d}: {t}" for d, t in sorted(docs.items()))

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    today = _dt.date.today().isoformat()
    items: list[dict] = []
    rejected: list[str] = []
    for strategy in STRATEGIES:
        prompt = (
            "제약 규제(RA·PV) 사내 문서 검색 도구의 평가 문항 후보를 생성한다.\n"
            f"[전략] {strategy}: {_STRATEGY_BRIEF[strategy]}\n"
            f"[코퍼스 문서 목록]\n{doc_list}\n"
            f"[기존 문항 예시 — 이것과 중복 금지]\n" + "\n".join(f"- {q}" for q in qa_sample) + "\n\n"
            f"위 전략으로 서로 다른 후보 {n_per_strategy}건을 만들어라. 반드시 아래 JSON 만 출력:\n"
            '{"candidates": [{"question": "...", "target_doc_id": "REG-0NN 또는 빈 문자열", '
            '"source_question": "변형의 원본 문항(없으면 빈 문자열)", "rationale": "이 후보가 재는 것 한 문장"}]}'
        )
        try:
            resp = client.messages.create(
                model=config.LLM_MODEL, max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(getattr(b, "text", "") for b in resp.content)
        except Exception as e:  # noqa: BLE001 - 한 전략의 실패가 전체를 죽이면 안 된다
            rejected.append(f"[{strategy}] 생성 호출 실패: {type(e).__name__}")
            continue
        cands = _parse_candidates(text)
        if cands is None:
            rejected.append(f"[{strategy}] malformed 출력 — 전략 전체 건너뜀")
            continue
        for cand in cands[:n_per_strategy]:
            ok, reason = _validate(cand, strategy, set(docs), existing)
            if ok is None:
                rejected.append(f"[{strategy}] {reason}")
                continue
            existing.add(_norm_question(ok["question"]))  # 같은 실행 내 중복도 차단
            ok.update(_verify_retrieval(ok))
            ok.update({"status": "pending_review", "generated_at": today,
                       "model": config.LLM_MODEL})
            items.append(ok)
    return {"items": items, "rejected": rejected}


_OUT_COMMENT = (
    "LLM 생성 평가 문항 후보(eval/generate_eval_items.py 산출) — 전 항목 status=pending_review. "
    "이 파일의 항목은 사람 검토 없이 어떤 평가셋에도 병합하지 않는다. 병합 절차: 검토자가 라벨"
    "(target_doc_id 등)을 확정하고, 대상 평가셋으로 옮기면서(복사가 아니라 이동) note 에 검토자·"
    "근거를 남기고 이 파일에서 제거한다. retrieval_verified 는 필터가 아니라 참고 라벨이다 — "
    "false(검색이 못 찾는 문항)가 오히려 가치 있는 후보일 수 있다(판단은 사람의 몫)."
)


def main(argv: list[str] | None = None, out_path=None) -> None:
    parser = argparse.ArgumentParser(description="LLM 기반 평가셋 후보 생성(리뷰 게이트)")
    parser.add_argument("--n", type=int, default=5, help="전략당 후보 수(기본 5)")
    parser.add_argument("--out", default=str(config.BASE_DIR / "eval" / "generated_candidates.json"))
    args = parser.parse_args(argv)
    out = out_path or args.out

    if not config.ANTHROPIC_API_KEY:
        print("평가셋 후보 생성은 ANTHROPIC_API_KEY 가 필요합니다 — CI 미등록 수동 도구이며,")
        print("파싱·리뷰 게이트 계약은 tests/test_generate_eval.py 가 키 없이 핀 고정합니다.")
        return

    res = generate(n_per_strategy=args.n)
    # 기존 후보 파일과 병합(추가만) — 이전 pending 검토물을 덮어쓰지 않는다
    from pathlib import Path

    out = Path(out)
    prior: list[dict] = []
    if out.exists():
        prior = json.loads(out.read_text(encoding="utf-8")).get("items", [])
    merged = prior + res["items"]
    out.write_text(
        json.dumps({"_comment": _OUT_COMMENT, "items": merged}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"후보 {len(res['items'])}건 생성 → {out} (누적 {len(merged)}건, 전건 pending_review)")
    by = {}
    for i in res["items"]:
        by[i["strategy"]] = by.get(i["strategy"], 0) + 1
    for s in STRATEGIES:
        print(f"  - {s:<13}: {by.get(s, 0)}건")
    verified = sum(1 for i in res["items"] if i["retrieval_verified"] is True)
    labeled = sum(1 for i in res["items"] if i["retrieval_verified"] is not None)
    print(f"  retrieval_verified: {verified}/{labeled} (false 는 폐기가 아니라 검토 우선순위)")
    if res["rejected"]:
        print("탈락/건너뜀 (은폐가 아니라 명시):")
        for r in res["rejected"]:
            print(f"  - {r}")
    print("다음 단계: 사람이 검토 후 대상 평가셋으로 '이동'한다 — 자동 병합 경로는 없다.")


if __name__ == "__main__":
    main()
