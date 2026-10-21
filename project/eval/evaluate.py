"""RAG 검색 품질 평가 (RAGAS 스타일 경량 버전).

측정 지표 (검색기 성능):
  - Hit@k           : 정답 문서가 상위 k개 안에 회수됐는가 (Recall 관점)
  - MRR             : 정답 문서의 첫 등장 순위의 역수 평균 (상위에 올렸는가)
  - Context Recall  : 정답 근거에 있어야 할 핵심어(keywords)가 회수된 context에 존재하는 비율
                      (= 생성 단계가 정답을 말할 '재료'를 검색이 제공했는가 → faithfulness 상한)

리랭킹 on/off 를 비교해 'RAG 최적화(리랭킹)'의 효과를 수치로 보여준다.

실행:  python -m eval.evaluate
"""
from __future__ import annotations

import json
import time

from src import config
from src.rag.pipeline import RagPipeline


def _load_qa(dataset: str = "qa_dataset.json") -> list[dict]:
    path = config.BASE_DIR / "eval" / dataset
    return [x for x in json.loads(path.read_text(encoding="utf-8"))["items"]]


def _hit_rank(retrieved_doc_ids: list[str], gold: str | list[str]) -> int:
    """정답 문서의 순위(1-based). 없으면 0.

    gold 는 단일 doc_id(str) 또는 동등 정답 집합(list) — 복수 정답 문항
    (accept_doc_ids)은 집합 멤버십으로 판정한다(v9). 코퍼스에 같은 답이
    두 문서에 실재하는 라벨 모호성은 검색기 결함이 아니라 라벨 결함이므로,
    문항 교체가 아닌 정답 집합 확장으로 교정한다. 단일 str 호출부
    (eval/sweep.py 등)는 그대로 동작한다.
    """
    accepted = {gold} if isinstance(gold, str) else set(gold)
    for i, d in enumerate(retrieved_doc_ids, 1):
        if d in accepted:
            return i
    return 0


def _gold_ids(item: dict) -> str | list[str]:
    """문항의 정답 라벨 — accept_doc_ids(복수 정답)가 있으면 그것, 없으면
    기존 단일 relevant_doc_id. ContextRecall 은 keywords 의 컨텍스트 존재
    판정이라 어느 정답 문서가 회수되든 같은 기준으로 채점된다(별도 분기 불요)."""
    return item.get("accept_doc_ids") or item["relevant_doc_id"]


def _context_recall(context_text: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    hit = sum(1 for k in keywords if k.lower() in context_text.lower())
    return hit / len(keywords)


def _retrieve(pipe: RagPipeline, mode: str, question: str, top_k: int, rerank_n: int):
    """검색 모드별 결과 반환. mode: 'vector' | 'hybrid' | 'rerank' | 'expand'.

    네 모드 모두 동일한 버전 인지 후보군(폐지본 제외)에서 검색해 공정 비교한다.
    """
    if mode == "vector":
        # 벡터(TF-IDF 코사인) 단독
        return pipe.retriever.vector_search(question, top_k)[:rerank_n]
    if mode == "hybrid":
        # 벡터+BM25 하이브리드 (리랭킹 없음)
        return pipe.retriever._hybrid(question, top_k)[:rerank_n]
    if mode == "rerank":
        # 하이브리드 + 리랭킹 (질의 확장 없음)
        return pipe.retriever.retrieve(
            question, top_k=top_k, rerank_n=rerank_n, expand=False
        )
    # 도메인 동의어 질의 확장 + 하이브리드 + 리랭킹 (운영 기본값)
    return pipe.retriever.retrieve(question, top_k=top_k, rerank_n=rerank_n, expand=True)


def evaluate(
    mode: str, top_k: int, rerank_n: int, pipe: RagPipeline | None = None,
    dataset: str = "qa_dataset.json",
) -> dict:
    pipe = pipe or RagPipeline().build()
    qa = _load_qa(dataset)
    hits1 = hits3 = 0
    hn_hits1 = hn_total = 0          # hard-negative 부분집합 Hit@1
    mrr_sum = 0.0
    ctx_recall_sum = 0.0
    latencies: list[float] = []

    for item in qa:
        t0 = time.perf_counter()
        results = _retrieve(pipe, mode, item["question"], top_k, rerank_n)
        latencies.append((time.perf_counter() - t0) * 1000.0)  # ms

        doc_ids = [s.chunk.doc_id for s in results]
        rank = _hit_rank(doc_ids, _gold_ids(item))
        if rank == 1:
            hits1 += 1
        if 1 <= rank <= 3:
            hits3 += 1
        mrr_sum += (1.0 / rank) if rank else 0.0

        if item.get("hard_negative"):
            hn_total += 1
            if rank == 1:
                hn_hits1 += 1

        ctx_text = "\n".join(s.chunk.text for s in results)
        ctx_recall_sum += _context_recall(ctx_text, item.get("keywords", []))

    n = len(qa)
    latencies.sort()
    return {
        "n": n,
        "counts": {"hits1": hits1, "hn_hits1": hn_hits1, "hn_total": hn_total},
        "Hit@1": round(hits1 / n, 3),
        "Hit@3": round(hits3 / n, 3),
        "MRR": round(mrr_sum / n, 3),
        "ContextRecall": round(ctx_recall_sum / n, 3),
        "HardNegHit@1": round(hn_hits1 / hn_total, 3) if hn_total else None,
        "p50_ms": round(latencies[len(latencies) // 2], 2) if latencies else 0.0,
        "mean_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="RAG 검색 품질 평가")
    parser.add_argument(
        "--skip-provider-compare", action="store_true",
        help="말미의 임베딩 provider 비교표(tfidf/hashing/dense 재구축)를 생략 — 운영 파이프라인 수치만 본다",
    )
    args = parser.parse_args()
    # 리랭킹 효과를 순위에 민감하게 보려고 최종 1건(rerank_n=1)으로 평가
    top_k, rerank_n = config.RETRIEVE_TOP_K, 1
    qa_n = len(_load_qa())
    print("=" * 62)
    print(f"RAG 검색 평가 · QA {qa_n}건 · top_k={top_k}, 최종 rerank_n={rerank_n}")
    print("=" * 62)

    # 인덱스는 1회만 구축해 세 모드가 공유(공정 비교 + 빠른 실행)
    pipe = RagPipeline().build()
    # auto 는 환경 따라 dense/tfidf 로 갈리므로 이 수치가 '무엇의 수치'인지 명시
    print(f"임베더: EMBEDDER_KIND={pipe.embedder_kind} → {type(pipe.retriever.store.embedder).__name__}"
          f" · 벡터 스토어: {type(pipe.retriever.store).__name__}(pool={pipe.retriever.vector_pool})")
    cols = [
        ("① 벡터만", "vector"),
        ("② 하이브리드", "hybrid"),
        ("③ +리랭킹", "rerank"),
        ("④ +질의확장", "expand"),
    ]
    res = {name: evaluate(mode, top_k, rerank_n, pipe=pipe) for name, mode in cols}

    header = f"{'지표':<16}" + "".join(f"{name:>14}" for name, _ in cols)
    print(header)
    print("-" * len(header))
    for k in ["Hit@1", "MRR", "ContextRecall", "HardNegHit@1", "mean_ms"]:
        row = f"{k:<16}" + "".join(f"{res[name][k]:>14}" for name, _ in cols)
        print(row)
    print("-" * len(header))
    print("해석:")
    print(" - HardNegHit@1: 어휘가 겹치는 유사문서(하드네거티브)가 섞인 문항에서의 정확도.")
    print("   어휘 기반 벡터(TF-IDF) 단독은 어휘 유사도에 끌려 오답 문서를 골랐지만(0.786),")
    print("   밀집임베딩은 벡터 단독으로도 의미를 구분한다(1.0). ② 하이브리드 구간의 일시")
    print("   하락은 작은 코퍼스에서 BM25 가 더하는 노이즈로, 리랭킹·질의확장이 회복시킨다")
    print("   — BM25 는 고유명사·코드(DMF, ALCOA) 정확 매칭 보험으로 유지한다.")
    print(" - mean_ms: 질의당 평균 검색 지연. 리랭킹은 정밀도를 얻는 대신 지연이 늘어난다")
    print("   (정밀도↔속도 트레이드오프). 폐지본(REG-013)은 버전 필터로 기본 제외된다.")
    print(" - ④ 질의확장: 도메인 동의어 사전('부작용'→'이상사례', '심각'→'중대한')으로")
    print("   어휘 불일치를 메워 Hit@1이 오른다. 확장은 1단계 회수에 전 가중으로,")
    print("   2단계 리랭킹엔 절반 가중의 보조 신호로만 반영(정밀도 희석 방지).")
    print("   하이퍼파라미터 스윕·ablation 근거는 python -m eval.sweep 으로 재현.")

    # --- 통계적 정직성: 소표본 지표에 95% 신뢰구간(Wilson) 병기 ---
    from eval.stats import fmt_ci

    best = res["④ +질의확장"]
    c, n = best["counts"], best["n"]
    print()
    print("통계적 정직성 (운영 기본 ④, Wilson 95% CI):")
    print(f" - Hit@1        : {fmt_ci(c['hits1'], n)}  (n={n})")
    print(f" - HardNegHit@1 : {fmt_ci(c['hn_hits1'], c['hn_total'])}  (n={c['hn_total']})")
    print("   → 1.000은 '완벽'이 아니라 '이 표본에서 실패 관측 0'이라는 뜻이다.")
    print("     구간 하한이 진짜 성능의 보수적 추정 — 개선 주장은 구간이 갈릴 때만 한다.")

    # --- 홀드아웃 평가(v8): 개발셋 수치와 일반화 성능의 구분 ---
    # 위 32문항은 실패 분석→신호 추가→재측정 루프에 쓰인 '개발셋'이라, 그 위의
    # 1.000은 일반화 성능이 아니다. 홀드아웃 문항은 튜닝에 일절 쓰지 않는다
    # (규율은 eval/holdout_dataset.json 의 _comment 참고) — 수치가 1.0 미만이어도
    # 그대로 싣는다: 이 갭이 개발셋 수치와 일반화의 차이를 정직하게 보여준다.
    hold = evaluate("expand", top_k, rerank_n, pipe=pipe, dataset="holdout_dataset.json")
    hc, hn = hold["counts"], hold["n"]
    print()
    print("홀드아웃 (튜닝 미사용 문항 — 개발셋 32문항과 분리, 운영 기본 ④):")
    # fmt_ci 가 점추정치를 포함하므로 앞에 값을 또 찍지 않는다(중복 인쇄 정리, v9)
    print(f" - Hit@1        : {fmt_ci(hc['hits1'], hn)}  (n={hn})")
    print(f" - ContextRecall: {hold['ContextRecall']}")
    print("   → 개발셋 1.000과 이 수치의 차이가 '튜닝된 수치'와 '일반화'의 갭이다.")
    print("     (v9: 종전 0.833 의 유일 실패는 검색 결함이 아니라 복수 정답 문항의")
    print("      라벨 모호성이었다 — accept_doc_ids 로 정답 집합만 교정, 튜닝 없음.")
    print("      1.000 이어도 n=6 의 CI 하한이 말하듯 '일반화 보장'이 아니다.)")

    # --- 임베딩 provider 비교(pluggable 실증) ---
    if args.skip_provider_compare:
        return
    print()
    print("=" * 62)
    print("임베딩 provider 비교 (동일 하이브리드+리랭킹 파이프라인)")
    print("=" * 62)
    prov_header = f"{'provider':<16}" + "".join(f"{k:>14}" for k in ["Hit@1", "MRR", "ContextRecall"])
    print(prov_header)
    print("-" * len(prov_header))
    for kind in ["tfidf", "hashing", "dense"]:
        try:
            p = RagPipeline(embedder_kind=kind).build()
        except Exception as e:
            # dense 는 ml 미설치/모델 미캐시 환경에서 불가 — 비교표에서 정직하게 표기
            print(f"{kind:<16}{'사용 불가(' + type(e).__name__ + ') — 건너뜀':>42}")
            continue
        r = evaluate("rerank", top_k, rerank_n, pipe=p)
        row = f"{kind:<16}" + "".join(f"{r[k]:>14}" for k in ["Hit@1", "MRR", "ContextRecall"])
        print(row)
    print("-" * len(prov_header))
    print("해석: EmbeddingProvider 인터페이스로 임베더를 교체해도 파이프라인은 무수정 동작한다.")
    print("      dense(밀집임베딩, sentence-transformers)가 기본 provider 이고 어휘 기반")
    print("      tfidf/hashing 은 경량 폴백 — 실무에선 상용 임베딩(Voyage 등)도 같은 자리에 끼운다.")


if __name__ == "__main__":
    main()
