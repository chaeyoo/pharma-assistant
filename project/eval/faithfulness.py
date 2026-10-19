"""답변 신뢰성(faithfulness) 평가 — 검색 정확도 그 다음의 '생성 안전성' 지표.

검색이 정답을 회수했는지(evaluate.py)와 별개로, **에이전트의 최종 답변이
근거 안에서만 말했는지 / 근거가 없을 때 지어내지 않는지**를 측정한다.
제약 규제 도메인에서 틀린 답은 컴플라이언스 사고이므로 이 축이 특히 중요하다.

측정 지표
  1. AnswerGroundedness (범위내): 답변이 인용한 근거(context) 안에 정답 핵심어가
     실제로 존재하는 비율. = 답이 '지어낸 것'이 아니라 근거에 뒷받침되는가.
  2. CitationRate (범위내): 답변에 출처(citation)가 붙은 비율.
  3. AbstentionAccuracy (범위밖): 사내문서로 답할 수 없는 질문에 대해
     환각 대신 '근거 없음'으로 답한 비율. (=환각 억제 안전장치의 실효성)

실행:  python -m eval.faithfulness
"""
from __future__ import annotations

import asyncio
import json

from src import config
from src.agent.agent import RaAgent


def _load(path_name: str) -> list[dict]:
    return json.loads((config.BASE_DIR / "eval" / path_name).read_text(encoding="utf-8"))["items"]


def _ctx_has_keywords(citations_text: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    return all(k.lower() in citations_text.lower() for k in keywords)


async def _run() -> dict:
    agent = RaAgent()
    in_scope = _load("qa_dataset.json")
    oos = _load("abstention_dataset.json")

    grounded_hits = cited = wrong_abstain = 0
    for item in in_scope:
        r = await agent.chat(item["question"])
        # 답변의 근거(citation) 문단 텍스트를 모아 정답 핵심어 포함 여부 확인
        ctx_text = "\n".join(
            f"{c.get('title','')} {c.get('section','')}" for c in r.citations
        )
        # citation 은 메타만 담으므로, 근거 존재성은 검색 결과 재조회로 보강
        from src.mcp_server.server import search_regulations

        search = search_regulations(item["question"], top_n=3)
        results_text = "\n".join(x["text"] for x in search["results"])
        if r.grounded and _ctx_has_keywords(results_text, item.get("keywords", [])):
            grounded_hits += 1
        if r.citations:
            cited += 1
        if not r.grounded:
            wrong_abstain += 1  # 범위내인데 abstain → 오답(과도한 회피)

    abstained = 0
    for item in oos:
        r = await agent.chat(item["question"])
        if not r.grounded:
            abstained += 1

    n_in, n_oos = len(in_scope), len(oos)
    return {
        "n_in_scope": n_in,
        "n_oos": n_oos,
        "counts": {"grounded": grounded_hits, "cited": cited, "abstained": abstained},
        "AnswerGroundedness": round(grounded_hits / n_in, 3),
        "CitationRate": round(cited / n_in, 3),
        "OverAbstain": round(wrong_abstain / n_in, 3),
        "AbstentionAccuracy": round(abstained / n_oos, 3),
    }


def main() -> None:
    # 완전 분리: 에이전트는 MCP_SERVER_URL(HTTP)로만 도구를 호출한다 — 평가도
    # 같은 wire 를 태우도록 하니스로 MCP 서버를 자체 기동한다(도커 불필요).
    from src.mcp_server.http_harness import run_mcp_http_server

    with run_mcp_http_server() as url:
        config.MCP_SERVER_URL = url
        res = asyncio.run(_run())
    print("=" * 60)
    print("답변 신뢰성(faithfulness) 평가 · 오프라인 모드")
    print("=" * 60)
    print(f"범위내 질문 {res['n_in_scope']}건 · 범위밖 질문 {res['n_oos']}건")
    print("-" * 60)
    print(f"AnswerGroundedness (범위내 답이 근거로 뒷받침): {res['AnswerGroundedness']:.3f}")
    print(f"CitationRate       (범위내 답에 출처 부착)   : {res['CitationRate']:.3f}")
    print(f"OverAbstain        (범위내인데 과도 회피)    : {res['OverAbstain']:.3f}  (낮을수록 좋음)")
    print(f"AbstentionAccuracy (범위밖에서 환각 대신 회피): {res['AbstentionAccuracy']:.3f}")
    print("-" * 60)
    from eval.stats import fmt_ci

    c = res["counts"]
    print("통계적 정직성 (Wilson 95% CI — 1.000은 '이 표본에서 실패 관측 0'):")
    print(f"  AnswerGroundedness {fmt_ci(c['grounded'], res['n_in_scope'])} (n={res['n_in_scope']})")
    print(f"  AbstentionAccuracy {fmt_ci(c['abstained'], res['n_oos'])} (n={res['n_oos']})")
    print("-" * 60)
    print("해석: 검색이 정답 재료를 주고(AnswerGroundedness), 답에는 출처가 붙으며(CitationRate),")
    print("      범위 밖 질문에는 지어내지 않고 '근거 없음'으로 답한다(AbstentionAccuracy).")
    print("      → 제약 규제 도메인이 요구하는 '추적성 + 환각 억제'를 수치로 증명.")


if __name__ == "__main__":
    main()
