"""임베더 provider 테스트: TF-IDF · 해싱 · 밀집(sentence-transformers) · 코사인."""
from __future__ import annotations

import math

import pytest

from src.rag.embedder import (
    HashingEmbedder,
    SentenceTransformerEmbedder,
    TfidfEmbedder,
    cosine,
    get_embedder,
)


class _StubSTModel:
    """`.encode()` 만 흉내내는 결정적 스텁 — torch/모델 다운로드 없이 계약 검증."""

    def __init__(self) -> None:
        self.calls: list = []

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            self.calls.append(texts)
            return self._vec(texts)
        self.calls.append(list(texts))
        return [self._vec(t) for t in texts]

    @staticmethod
    def _vec(text: str) -> list[float]:
        # 텍스트 바이트합 기반 결정적 3차원 벡터(전 성분 양수 → 코사인 항상 > 0)
        s = sum(text.encode("utf-8"))
        return [1.0 + (s % 7), 1.0 + (len(text) % 5), 1.0 + (s % 3)]


def test_tfidf_embeds_unit_norm():
    emb = TfidfEmbedder()
    emb.fit(["품목허가 심사 기간", "이상사례 보고 기한"])
    v = emb.embed("품목허가 심사")
    assert v, "임베딩이 비어있지 않아야 함"
    norm = math.sqrt(sum(w * w for w in v.values()))
    assert abs(norm - 1.0) < 1e-6, "L2 정규화되어야 함"


def test_tfidf_requires_fit():
    emb = TfidfEmbedder()
    try:
        emb.embed("x")
        assert False, "fit 전 embed 는 예외여야 함"
    except RuntimeError:
        pass


def test_hashing_is_deterministic_and_handles_oov():
    emb = HashingEmbedder(n_buckets=256)
    emb.fit(["품목허가 심사", "이상사례 보고"])
    a = emb.embed("완전히 처음 보는 미등록 단어들")   # OOV 여도 예외 없이 벡터 생성
    b = emb.embed("완전히 처음 보는 미등록 단어들")
    assert a == b, "동일 입력은 동일 벡터(재현성)"
    assert a, "OOV 입력도 해싱으로 벡터화 가능"


def test_cosine_bounds_and_self_similarity():
    emb = TfidfEmbedder()
    emb.fit(["품목허가 심사 기간 신약", "화장품 전성분 표시"])
    v = emb.embed("품목허가 심사 기간")
    assert abs(cosine(v, v) - 1.0) < 1e-6, "자기 자신과의 코사인은 1"
    assert cosine(v, {}) == 0.0, "빈 벡터와는 0"


def test_factory_returns_distinct_providers():
    assert isinstance(get_embedder("tfidf"), TfidfEmbedder)
    assert isinstance(get_embedder("hashing"), HashingEmbedder)
    try:
        get_embedder("nope")
        assert False
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# 밀집임베딩 (SentenceTransformerEmbedder) — 스텁 주입으로 torch 없이 계약 검증
# ---------------------------------------------------------------------------
def test_dense_wraps_vector_as_index_dict():
    """dense 벡터는 {차원 인덱스: 값} dict 로 담겨 cosine()/vectorstore 와 호환(Voyage 선례)."""
    emb = SentenceTransformerEmbedder(model=_StubSTModel())
    emb.fit(["문서 하나", "문서 둘"])
    v = emb.embed("문서 하나")
    assert set(v.keys()) == {"0", "1", "2"}
    assert all(isinstance(x, float) for x in v.values())
    assert abs(cosine(v, v) - 1.0) < 1e-6


def test_dense_prefix_asymmetry_and_passage_cache():
    """e5 비대칭 계약: fit=passage 배치 인코딩+캐시, embed=캐시 히트(색인 경로)
    또는 query prefix 인코딩(질의 경로) — prefix 누락은 조용한 품질 저하라
    스텁 호출 기록으로 회귀 고정한다."""
    stub = _StubSTModel()
    emb = SentenceTransformerEmbedder(model=stub)
    emb.fit(["규정 본문 텍스트"])
    assert stub.calls[0] == ["passage: 규정 본문 텍스트"]
    emb.embed("규정 본문 텍스트")  # 색인 경로: passage 캐시 히트
    assert len(stub.calls) == 1, "코퍼스 원문 재-embed 는 인코딩 재호출 없이 캐시에서"
    emb.embed("품목허가 질의")     # 질의 경로
    assert stub.calls[-1] == "query: 품목허가 질의"


def test_dense_requires_fit():
    emb = SentenceTransformerEmbedder(model=_StubSTModel())
    try:
        emb.embed("x")
        assert False, "fit 전 embed 는 예외여야 함"
    except RuntimeError:
        pass


def test_auto_falls_back_to_tfidf_without_ml(monkeypatch):
    """auto 폴백 계약: 밀집임베딩 로드 불가(미설치/모델 미캐시) → TF-IDF."""
    import src.rag.embedder as em

    def _boom(model_name, local_files_only):
        raise ImportError("sentence_transformers 미설치")

    monkeypatch.setattr(em, "_load_st_model", _boom)
    assert isinstance(em.get_embedder("auto"), TfidfEmbedder)


def test_dense_explicit_fails_fast(monkeypatch):
    """dense 명시 설정은 폴백 없이 실패를 전파한다(시끄러운 실패 원칙)."""
    import src.rag.embedder as em

    def _boom(model_name, local_files_only):
        raise ImportError("sentence_transformers 미설치")

    monkeypatch.setattr(em, "_load_st_model", _boom)
    try:
        em.get_embedder("dense")
        assert False, "dense 는 폴백 없이 예외여야 함"
    except ImportError:
        pass


def test_dense_real_model_semantic_ranking():
    """실모델 통합 스모크(1건) — ml 미설치/모델 미캐시 환경에선 skip.
    어휘가 안 겹치는 의미 유사 문장이 무관 문장보다 높은 코사인을 받아야 한다."""
    pytest.importorskip("sentence_transformers")
    from src import config
    from src.rag.embedder import _load_st_model

    try:
        _load_st_model(config.EMBEDDING_MODEL, True)
    except Exception:
        pytest.skip("임베딩 모델 미캐시 — scripts/download_embedding_model.py 로 받아야 함")
    emb = get_embedder("dense")
    corpus = ["약물 부작용은 규제기관에 신속히 보고해야 한다", "화장품 광고는 표시 기준을 따라야 한다"]
    emb.fit(corpus)
    q = emb.embed("이상사례 발생 시 당국 통보 기한")
    related = cosine(q, emb.embed(corpus[0]))
    unrelated = cosine(q, emb.embed(corpus[1]))
    assert related > unrelated
