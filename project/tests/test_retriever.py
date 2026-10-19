"""리트리버 구성요소 단위 테스트: BM25 · 정규화 · 하이브리드 · 리랭킹 불변식."""
from __future__ import annotations

from src.rag.retriever import BM25Index, _minmax


def test_minmax_normalizes_to_unit_range():
    out = _minmax([1.0, 3.0, 5.0])
    assert out[0] == 0.0 and out[-1] == 1.0
    assert all(0.0 <= v <= 1.0 for v in out)
    # 모두 동일하면 0으로(0분모 방지)
    assert _minmax([2.0, 2.0, 2.0]) == [0.0, 0.0, 0.0]
    assert _minmax([]) == []


def test_bm25_ranks_matching_doc_highest():
    idx = BM25Index()
    docs = [
        "품목허가 심사 기간은 신약 120 근무일이다",
        "임상시험 계획 승인 IND 처리기한",
        "화장품 전성분 표시 광고 기준",
    ]
    idx.index(docs)
    scores = idx.scores("신약 품목허가 심사 기간")
    assert scores[0] == max(scores), "질의와 가장 겹치는 문서가 최고점"
    assert scores[0] > 0.0


def test_bm25_idf_penalizes_common_terms():
    idx = BM25Index()
    # '공통' 은 모든 문서에, '희귀' 는 한 문서에만 → 희귀가 더 높은 IDF
    idx.index(["공통 희귀", "공통 단어", "공통 단어"])
    assert idx.idf["희귀"] > idx.idf["공통"]


def test_hybrid_and_rerank_return_within_bounds(pipeline):
    q = "GMP 데이터 완전성 ALCOA 원칙"
    hybrid = pipeline.retriever._hybrid(q, top_k=8)
    assert 0 < len(hybrid) <= 8
    # 하이브리드 점수는 내림차순 정렬
    assert all(hybrid[i].score >= hybrid[i + 1].score for i in range(len(hybrid) - 1))
    reranked = pipeline.retriever.retrieve(q, top_k=8, rerank_n=3)
    assert 0 < len(reranked) <= 3


def test_no_signal_query_returns_empty(pipeline):
    """무신호 계약(v9): 전 후보의 벡터·BM25 원점수가 0이면 빈 결과.

    종전에는 안정 정렬이 '전부 0점'을 코퍼스 첫 문서 순서로 위장해 빈 질의가
    임의 문서 3청크를 score 0.0 의 검색 결과로 달고 나갔다(가짜 순위) —
    관련도 신호가 전혀 없으면 결과가 없다고 답하는 것이 맞다."""
    r = pipeline.retriever
    assert r.retrieve("", top_k=8, rerank_n=3) == []
    # 완전 무관 어휘(코퍼스 OOV)도 동일 — 벡터·BM25 모두 0점
    assert r.retrieve("zxqv wxyj kqzp", top_k=8, rerank_n=3) == []
    # 유신호 질의는 계속 정상 반환(무신호 판정이 과확장되지 않았는지의 반대면)
    assert r.retrieve("품목허가 심사 기간", top_k=8, rerank_n=3)


def test_dense_no_signal_floor_contract():
    """무신호 계약 v13: 밀집 임베더는 무관 질의에도 코사인 0 이 안 나오므로
    no_signal_floor 속성으로 자기 바닥값을 선언한다 — floor 이하 = 벡터 무신호."""
    from src.rag.chunker import Chunk
    from src.rag.embedder import SentenceTransformerEmbedder
    from src.rag.retriever import HybridRetriever
    from src.rag.vectorstore import InMemoryVectorStore
    from tests.test_embedder import _StubSTModel

    chunks = [
        Chunk(chunk_id="A-1", doc_id="A", source="a.md", title="허가",
              section="1. 절차", text="품목허가 심사 기간", effective_date="2024-01-01"),
        Chunk(chunk_id="B-1", doc_id="B", source="b.md", title="보고",
              section="1. 기한", text="이상사례 보고 기한", effective_date="2024-01-01"),
    ]
    emb = SentenceTransformerEmbedder(model=_StubSTModel())
    r = HybridRetriever(InMemoryVectorStore(emb))
    r.index(chunks)
    # 기호-only 질의: 토큰이 없으면 임베딩 호출 전에 무신호로 잘린다
    assert r.retrieve("!!! ???", top_k=8, rerank_n=3) == []
    # BM25 0(코퍼스 OOV) + 스텁 코사인 > 0: floor 0.0 이면 벡터 신호로 결과 반환
    emb.no_signal_floor = 0.0
    assert r.retrieve("zxqv wxyj", top_k=8, rerank_n=3)
    # floor 1.0(코사인 상한) 이면 같은 질의가 무신호 → 빈 결과
    emb.no_signal_floor = 1.0
    assert r.retrieve("zxqv wxyj", top_k=8, rerank_n=3) == []


def test_rerank_fixes_hard_negative(pipeline):
    """어휘가 겹치는 하드네거티브(GMP 실태조사 REG-011)를 물리치고
    ALCOA 데이터 완전성(REG-003)을 최상위로 올려야 한다."""
    q = "GMP 데이터 완전성 ALCOA 원칙은 무엇인가요?"
    top = pipeline.retriever.retrieve(q, top_k=8, rerank_n=1)[0]
    assert top.chunk.doc_id == "REG-003"


def test_colloquial_query_resolved_by_expansion(pipeline):
    """구어 질의('부작용이 심각', '섞이는 것')는 원 질의 토큰이 문서에 거의
    없어도, 동의어 확장(심각→중대한, 섞이→교차오염)으로 정답 문서를 찾아야
    한다 — eval 오류 분석에서 사전을 보강한 회귀 가드."""
    top = pipeline.retriever.retrieve(
        "부작용이 심각하게 나타났을 때 당국에 얼마나 빨리 알려야 하나요?",
        top_k=8, rerank_n=1,
    )[0]
    assert top.chunk.doc_id == "REG-005"

    top = pipeline.retriever.retrieve(
        "약을 만드는 시설에서 서로 다른 제품이 섞이는 것을 막는 기본 원칙은?",
        top_k=8, rerank_n=1,
    )[0]
    assert top.chunk.doc_id == "REG-003"


def test_aux_terms_keep_reranker_robust_without_prior(pipeline):
    """확장 토큰 보조신호(aux)의 존재 이유: 1차 prior 를 끄면(rerank_weight=1.0)
    완전 어휘 불일치 질의에서 aux 없는 리랭커는 판별력을 잃는다."""
    r = pipeline.retriever
    q = "부작용이 심각하게 나타났을 때 당국에 얼마나 빨리 알려야 하나요?"
    saved = r.rerank_weight
    try:
        r.rerank_weight = 1.0
        with_aux = r.retrieve(q, top_k=8, rerank_n=1, aux_in_rerank=True)[0]
        assert with_aux.chunk.doc_id == "REG-005", "aux 켜면 prior 없이도 정답"
    finally:
        r.rerank_weight = saved


def test_rerank_title_field_signal(pipeline):
    """제목 정합(title) 신호: 같은 본문 매칭이라도 문서 제목이 질의와 겹치는
    청크가 더 높은 점수를 받아야 한다(BM25F 감각의 필드 분리)."""
    from src.rag.chunker import Chunk

    def mk(title: str) -> Chunk:
        return Chunk(
            chunk_id="T-1", doc_id="T", source="t.md", title=title,
            section="1. 개요", text="의약품 용기 기재사항에 대한 내용",
        )

    r = pipeline.retriever
    q = "의약품 용기 기재사항은?"
    s_match = r._rerank_score(q, mk("의약품 표시기재 기준"))
    s_mismatch = r._rerank_score(q, mk("화장품 표시광고 규정"))
    assert s_match > s_mismatch


# ---------------------------------------------------------------------------
# 리랭커 v3 — 섹션 타입 prior (질의 의도 게이트)
# ---------------------------------------------------------------------------
def test_contrast_section_penalty_fixes_hard_negative(pipeline):
    """'X와의 차이(주의)' 대조 섹션은 X 도메인 어휘를 통째로 인용해 어휘 신호가
    정답 문서를 이긴다(잔여 하드네거티브 실패의 형태). 질의가 비교를 묻지
    않으면 대조 섹션을 감점해 정답(REG-004 본문 조항)이 1순위여야 한다."""
    top = pipeline.retriever.retrieve(
        "의약품 용기에 반드시 기재해야 하는 사항은?", top_k=8, rerank_n=1
    )[0]
    assert top.chunk.doc_id == "REG-004"
    assert "차이" not in top.chunk.section


def test_contrast_penalty_gated_off_for_comparison_query(pipeline):
    """질의 의도 게이트: 사용자가 '차이'를 물으면 대조 섹션이 곧 정답이므로
    페널티를 끈다 — 페널티가 '섹션 삭제'가 아니라 '의도 조건부 신호'라는 증거."""
    top = pipeline.retriever.retrieve(
        "화장품 표시기재는 의약품 표시기재와 뭐가 다른가요?", top_k=8, rerank_n=1
    )[0]
    assert top.chunk.doc_id == "REG-008"
    assert "차이" in top.chunk.section


def test_preamble_demotion_prefers_operative_section(pipeline):
    """서두(목적/개요) 섹션은 문서 주제 어휘를 요약해 담아 운영 질문에서
    본문 조항을 이기는 과대평가가 난다 — 감쇠 후 본문 조항이 1순위."""
    top = pipeline.retriever.retrieve(
        "제품 설명서에 경고 문구는 어디에 표시하나요?", top_k=8, rerank_n=1
    )[0]
    assert top.chunk.doc_id == "REG-004"
    assert "첨부문서" in top.chunk.section


def test_preamble_demotion_gated_off_for_definition_query(pipeline):
    """정의/취지 질문('~란 무엇')은 서두가 곧 정답 — 게이트가 감쇠를 꺼서
    개요 섹션이 그대로 1순위여야 한다."""
    top = pipeline.retriever.retrieve(
        "시판 후 안전관리란 무엇을 하는 활동인가요?", top_k=8, rerank_n=1
    )[0]
    assert top.chunk.doc_id == "REG-005"
    assert "개요" in top.chunk.section


def test_section_priors_disabled_reproduces_old_failure(pipeline):
    """ablation 가드: prior 를 끄면(0.0) 대조 섹션이 다시 이긴다 — 개선이
    이 신호에서 왔음을 테스트로 고정(우연한 다른 변경으로 통과하는 것 방지)."""
    r = pipeline.retriever
    saved = (r.contrast_penalty, r.preamble_penalty)
    try:
        r.contrast_penalty = r.preamble_penalty = 0.0
        top = r.retrieve("의약품 용기에 반드시 기재해야 하는 사항은?", top_k=8, rerank_n=1)[0]
        assert top.chunk.doc_id == "REG-008" and "차이" in top.chunk.section
    finally:
        r.contrast_penalty, r.preamble_penalty = saved
