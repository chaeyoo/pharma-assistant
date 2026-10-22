"""RAG 파이프라인 오케스트레이터.

로딩 → 청킹 → 임베딩/인덱싱 을 부팅 시 1회 수행하고(pgvector 스토어면 바뀐 청크만
증분 임베딩하고 나머지는 DB 에 저장된 벡터를 재사용한다),
질의 시 검색(하이브리드+리랭킹) → 근거(context) 조립을 담당한다.
생성(Generation)은 agent 계층에서 이 검색 결과를 받아 수행한다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .. import config
from ..observability import flow, logger
from .chunker import Chunk, chunk_documents
from .embedder import get_embedder
from .loader import load_documents
from .retriever import HybridRetriever
from .vectorstore import Scored, get_vector_store


@dataclass
class RetrievedContext:
    """검색 결과를 생성단계로 넘기기 위한 묶음."""
    query: str
    chunks: list[Scored]

    def to_prompt_block(self) -> str:
        """LLM 프롬프트에 넣을 근거 블록(출처 번호 포함)."""
        lines = []
        for i, s in enumerate(self.chunks, 1):
            lines.append(
                f"[근거 {i}] (출처: {s.chunk.title} · {s.chunk.source} · {s.chunk.section})\n"
                f"{s.chunk.text.strip()}"
            )
        return "\n\n".join(lines)

    def citations(self) -> list[dict]:
        """UI/응답용 출처 메타데이터."""
        return [
            {
                "n": i,
                "doc_id": s.chunk.doc_id,
                "title": s.chunk.title,
                "source": s.chunk.source,
                "section": s.chunk.section,
                "version": s.chunk.version,
                "effective_date": s.chunk.effective_date,
                "score": round(s.score, 4),
            }
            for i, s in enumerate(self.chunks, 1)
        ]


class RagPipeline:
    def __init__(
        self,
        reg_dir: Path | None = None,
        embedder_kind: str | None = None,
        vector_store: str | None = None,
    ) -> None:
        self.reg_dir = reg_dir or config.REG_DIR
        self.embedder_kind = embedder_kind or config.EMBEDDER_KIND
        self.vector_store_kind = vector_store or config.VECTOR_STORE
        # 콜드스타트 분해용 — 밀집임베딩이면 여기서 torch 임포트·모델 로드가 일어난다
        t0 = time.perf_counter()
        store = get_vector_store(
            get_embedder(self.embedder_kind), self.vector_store_kind, self.embedder_kind
        )
        self.prep_ms = round((time.perf_counter() - t0) * 1000)
        self.retriever = HybridRetriever(
            store,
            alpha=config.HYBRID_ALPHA,
            rerank_weight=config.RERANK_WEIGHT,
            idf_power=config.RERANK_IDF_POWER,
            contrast_penalty=config.RERANK_CONTRAST_PENALTY,
            preamble_penalty=config.RERANK_PREAMBLE_PENALTY,
        )
        self.n_docs = 0
        self.n_chunks = 0

    def build(self) -> "RagPipeline":
        """인덱스 구축 (부팅 시 1회).

        끝에 단계별 소요 시간을 INFO 한 줄로 남긴다 — Cloud Run 처럼 0대로 스케일되는
        환경에서 콜드스타트가 어디에 쓰이는지(모델 로드 vs 임베딩/색인) 로그만으로
        읽을 수 있게 하고, pgvector 증분 색인(재사용/신규 임베딩 수)의 효과를 같은 줄에서
        확인한다.
        """
        t0 = time.perf_counter()
        docs = load_documents(self.reg_dir)
        chunks: list[Chunk] = chunk_documents(
            docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP
        )
        load_ms = round((time.perf_counter() - t0) * 1000)
        t1 = time.perf_counter()
        self.retriever.index(chunks)
        index_ms = round((time.perf_counter() - t1) * 1000)
        self.n_docs, self.n_chunks = len(docs), len(chunks)
        self.build_ms = self.prep_ms + load_ms + index_ms
        store = self.retriever.store
        sync = getattr(store, "last_sync", None) or {}
        sync_note = (
            f" · 재사용 {sync.get('reused', 0)} · 신규/변경 임베딩 {sync.get('embedded', 0)}"
            if sync else ""
        )
        logger.info(
            "RAG 인덱스 구축: 문서 %d · 청크 %d · 임베더 %s · 스토어 %s%s · "
            "모델로드 %dms · 로드/청킹 %dms · 색인 %dms · 합계 %dms",
            self.n_docs, self.n_chunks, type(store.embedder).__name__, type(store).__name__,
            sync_note, self.prep_ms, load_ms, index_ms, self.build_ms,
        )
        return self

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        rerank_n: int | None = None,
        as_of: str = "",
        include_superseded: bool = False,
    ) -> RetrievedContext:
        # '미지정'(None) 판정은 is None 으로 — `or` 는 0 도 미지정으로 취급해
        # top_k=0(후보 0개 요청)이 조용히 기본값 8로 승격되는 falsy 함정이 된다
        # (호출자는 0건을 요청했는데 8건 검색이 도는 조용한 확대).
        top_k = config.RETRIEVE_TOP_K if top_k is None else top_k
        rerank_n = config.RERANK_TOP_N if rerank_n is None else rerank_n
        flow(
            "RagPipeline.retrieve()",
            "2단계 검색 실행 — 하이브리드로 후보 top_k 를 넓게 모은 뒤 리랭커가 rerank_n 으로 좁힌다",
            top_k=top_k, rerank_n=rerank_n, expand=config.QUERY_EXPANSION,
            next="HybridRetriever.retrieve() 호출(src/rag/retriever.py) — 버전 필터→벡터+BM25 결합→리랭킹 순으로 진행",
        )
        results = self.retriever.retrieve(
            query,
            top_k=top_k,
            rerank_n=rerank_n,
            as_of=as_of,
            include_superseded=include_superseded,
            expand=config.QUERY_EXPANSION,
        )
        return RetrievedContext(query=query, chunks=results)
