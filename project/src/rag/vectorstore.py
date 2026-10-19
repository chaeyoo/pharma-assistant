"""인메모리 벡터 스토어 (Vector Store).

청크와 그 임베딩을 보관하고 코사인 유사도로 top-k 검색을 제공한다.
실무의 Vector DB(pgvector/Qdrant/Chroma) 자리에 해당하는 경량 대체물.
"""
from __future__ import annotations

from dataclasses import dataclass

from .chunker import Chunk
from .embedder import EmbeddingProvider, SparseVec


@dataclass
class Scored:
    """검색 결과 1건 (청크 + 점수)."""
    chunk: Chunk
    score: float


class InMemoryVectorStore:
    def __init__(self, embedder: EmbeddingProvider) -> None:
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.vectors: list[SparseVec] = []

    def index(self, chunks: list[Chunk]) -> None:
        """청크 코퍼스로 임베더를 학습시키고 전체를 벡터화해 저장."""
        self.chunks = chunks
        self.embedder.fit([c.text for c in chunks])
        self.vectors = [self.embedder.embed(c.text) for c in chunks]

    # 주: 순수 코사인 검색 메서드는 v12 에서 제거했다 — 호출부가 없는 데드코드였고,
    # 버전 필터(폐지본·미래 시행일 제외)와 무신호 계약을 우회하는 경로라, 향후 누가
    # 배선하면 폐지 구판이 그대로 새는 유지보수 함정이었다. 실제 검색은 HybridRetriever
    # 가 store.chunks/store.vectors 를 직접 읽어 _candidate_indices(버전 필터) 뒤에서
    # 수행한다(retriever.py). cosine 은 그 경로가 embedder 에서 직접 쓴다.
