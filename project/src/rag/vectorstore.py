"""벡터 스토어 (Vector Store) — 청크·임베딩 보관 + 버전 필터 하의 유사도 검색.

두 구현이 같은 계약(`VectorStore` Protocol)을 만족한다:
  - InMemoryVectorStore: 기본. 프로세스 메모리에 청크·벡터를 들고 전수 코사인 계산.
    희소(TF-IDF/해싱)·밀집 임베더 모두 수용. 재시작하면 처음부터 다시 임베딩한다.
  - PgVectorStore: Postgres + pgvector. 청크 메타·임베딩을 테이블에 영속화하고
    유사도 검색을 DB 가 수행한다(HNSW 인덱스, 버전 필터는 SQL WHERE 로 pre-filter).
    부팅 시 **증분 색인** — 텍스트 해시가 같은 청크는 재임베딩하지 않고, 바뀐
    청크만 인코딩·갱신, 사라진 청크는 삭제한다. 밀집 임베더 전용(고정 차원).

리트리버(HybridRetriever)는 어느 구현이든 `chunks`(코퍼스 순서)·`version_filter`
(BM25 후보 마스킹용)·`search()`(벡터 후보 + 점수)만 쓴다. BM25·리랭킹은 스토어와
무관하게 파이썬에서 수행한다 — Postgres FTS 에는 한국어 형태소 분석이 없어 BM25 를
DB 로 옮기면 어휘 신호의 토크나이즈가 리랭커와 갈라지기 때문이다.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import threading
from dataclasses import dataclass
from typing import Protocol

from .. import config
from ..observability import logger
from .chunker import Chunk
from .embedder import EmbeddingProvider, SentenceTransformerEmbedder, SparseVec, VoyageEmbedder, cosine
from .versionfilter import VersionFilter


@dataclass
class Scored:
    """검색 결과 1건 (청크 + 점수)."""
    chunk: Chunk
    score: float


class VectorStore(Protocol):
    """리트리버가 의존하는 스토어 계약."""

    embedder: EmbeddingProvider
    chunks: list[Chunk]
    version_filter: VersionFilter

    def index(self, chunks: list[Chunk]) -> None: ...

    def search(
        self, query_vec: SparseVec, cutoff: _dt.date, include_superseded: bool, limit: int
    ) -> dict[int, float]:
        """버전 필터를 통과한 청크의 {코퍼스 위치: 코사인 유사도}.

        limit>0 이면 유사도 상위 limit 개만(후보 풀), 0 이면 전수. 반환에 없는
        후보는 '반환된 최솟값 이하'라는 것만 보장된다(리트리버가 바닥값으로 채움).
        """
        ...


class InMemoryVectorStore:
    def __init__(self, embedder: EmbeddingProvider) -> None:
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.vectors: list[SparseVec] = []
        self.version_filter = VersionFilter()

    def index(self, chunks: list[Chunk]) -> None:
        """청크 코퍼스로 임베더를 학습시키고 전체를 벡터화해 저장."""
        self.chunks = chunks
        self.version_filter = VersionFilter.from_chunks(chunks)
        self.embedder.fit([c.text for c in chunks])
        self.vectors = [self.embedder.embed(c.text) for c in chunks]

    def search(
        self, query_vec: SparseVec, cutoff: _dt.date, include_superseded: bool, limit: int = 0
    ) -> dict[int, float]:
        # limit 은 무시하고 항상 전수를 돌려준다 — 메모리 경로는 정확 계산이 공짜고,
        # 리트리버의 min-max 정규화가 전 후보군 기준이어야 종전 수치와 동일하다.
        idxs = self.version_filter.candidates(cutoff, include_superseded)
        return {i: cosine(query_vec, self.vectors[i]) for i in idxs}


# ---------------------------------------------------------------------------
# pgvector
# ---------------------------------------------------------------------------
DENSE_PROVIDERS = (SentenceTransformerEmbedder, VoyageEmbedder)


def _dense_list(vec: SparseVec) -> list[float]:
    """밀집 임베더의 {차원 인덱스: 값} 표현 → 순서 리스트."""
    return [vec[str(i)] for i in range(len(vec))]


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _model_tag(embedder: EmbeddingProvider) -> str:
    """임베딩 공간 식별자 — 모델이 바뀌면 저장된 벡터는 모두 무효라 전체 재색인한다."""
    name = getattr(embedder, "model_name", None) or getattr(embedder, "model", "")
    return f"{type(embedder).__name__}:{name}"


# 버전 필터의 SQL 표현 — versionfilter.VersionFilter.candidates() 와 동치여야 한다
# (tests/test_vectorstore.py 가 실 DB 로 동치를 고정). succ 는 doc_id 별 시행일
# (한 문서의 청크는 시행일을 공유; 미상은 NULL → MIN 이 NULL → fail-closed).
_FILTER_SQL = """
    c.effective_date IS NOT NULL AND c.effective_date <= %(cutoff)s
    AND (
        c.status <> 'superseded'
        OR %(include_superseded)s
        OR (succ.eff IS NOT NULL AND succ.eff > %(cutoff)s)
    )
"""


class PgVectorStore:
    """Postgres + pgvector 영속 벡터 스토어 (밀집 임베더 전용).

    테이블 `<table>` 에 청크 메타·임베딩을, `<table>_meta` 에 임베딩 공간(모델·차원)을
    둔다. 색인은 한 트랜잭션 + advisory lock 으로 수행해 동시 부팅하는 복제본이
    서로 덮어쓰지 않고, 중간에 죽어도 직전 일관 상태가 남는다.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        url: str | None = None,
        pool: int | None = None,
        table: str = "rag_chunks",
    ) -> None:
        if not isinstance(embedder, DENSE_PROVIDERS):
            raise ValueError(
                f"pgvector 스토어는 밀집 임베더 전용이다(고정 차원 vector 컬럼) — "
                f"{type(embedder).__name__} 는 코퍼스 전체로 사전을 학습하는 희소 벡터라 영속화 대상이 아니다"
            )
        import psycopg  # lazy — 선택 기능(requirements.txt 의 psycopg[binary])

        self._psycopg = psycopg
        self.embedder = embedder
        self.url = url or config.PGVECTOR_URL
        if not self.url:
            raise ValueError("VECTOR_STORE=pgvector 인데 PGVECTOR_URL 이 없다")
        self.pool = config.PGVECTOR_POOL if pool is None else pool
        if not table.replace("_", "").isalnum():
            raise ValueError(f"테이블명 오류: {table!r}")
        self.table = table
        self.chunks: list[Chunk] = []
        self.version_filter = VersionFilter()
        self.dim: int | None = None
        self.last_sync: dict[str, int] = {}
        self._conn = None
        self._lock = threading.Lock()  # 질의 연결 1개 공유(MCP 도구 호출은 스레드풀에서 동시 실행 가능)

    # ---- 연결 ----
    def _connect(self):
        return self._psycopg.connect(self.url, connect_timeout=5, autocommit=True)

    def _query_conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = self._connect()
        return self._conn

    # ---- 색인(증분) ----
    def index(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.version_filter = VersionFilter.from_chunks(chunks)
        tag = _model_tag(self.embedder)
        t, meta = self.table, f"{self.table}_meta"
        with self._connect() as conn, conn.transaction():
            # 동일 테이블을 동시 색인하는 부팅 복제본 직렬화(키는 테이블명 해시)
            conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (t,))
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {meta} (key text PRIMARY KEY, value text NOT NULL)"
            )
            stored = dict(conn.execute(f"SELECT key, value FROM {meta}").fetchall())
            table_exists = conn.execute("SELECT to_regclass(%s) IS NOT NULL", (t,)).fetchone()[0]
            if table_exists and stored.get("model") != tag:
                # 임베딩 공간이 바뀜(모델 교체) → 저장 벡터 전부 무효. 차원도 바뀔 수 있어 재생성.
                logger.warning("pgvector: 임베딩 모델 변경(%s → %s) — 전체 재색인", stored.get("model"), tag)
                conn.execute(f"DROP TABLE {t}")
                table_exists = False
            existing: dict[str, str] = {}
            if table_exists:
                existing = dict(conn.execute(f"SELECT chunk_id, text_sha256 FROM {t}").fetchall())

            shas = {c.chunk_id: _text_sha(c.text) for c in chunks}
            changed_ids = {c.chunk_id for c in chunks if existing.get(c.chunk_id) != shas[c.chunk_id]}
            to_embed = [(i, c) for i, c in enumerate(chunks) if c.chunk_id in changed_ids]
            unchanged = [(i, c) for i, c in enumerate(chunks) if c.chunk_id not in changed_ids]
            stale = sorted(set(existing) - set(shas))

            # 바뀐 청크만 인코딩. fit 은 e5 passage 캐시 채우기(ST)/no-op(Voyage).
            self.embedder.fit([c.text for _, c in to_embed])
            vectors = [_dense_list(self.embedder.embed(c.text)) for _, c in to_embed]
            dim = len(vectors[0]) if vectors else int(stored["dim"]) if "dim" in stored else None
            if dim is None:
                # 코퍼스가 비었고 저장된 공간도 없음 — 만들 것이 없다
                self.dim = None
                self.last_sync = {"reused": 0, "embedded": 0, "deleted": 0}
                return
            if not table_exists:
                conn.execute(
                    f"""CREATE TABLE {t} (
                        chunk_id       text PRIMARY KEY,
                        position       int  NOT NULL,
                        doc_id         text NOT NULL,
                        source         text NOT NULL,
                        title          text NOT NULL,
                        section        text NOT NULL,
                        text           text NOT NULL,
                        version        text NOT NULL,
                        effective_date date,
                        status         text NOT NULL,
                        superseded_by  text NOT NULL,
                        metadata       jsonb NOT NULL,
                        text_sha256    text NOT NULL,
                        embedding      vector({dim}) NOT NULL,
                        updated_at     timestamptz NOT NULL DEFAULT now()
                    )"""
                )
                conn.execute(f"CREATE INDEX {t}_doc_id_idx ON {t} (doc_id)")
                conn.execute(
                    f"CREATE INDEX {t}_embedding_idx ON {t} USING hnsw (embedding vector_cosine_ops)"
                )
            for k, v in (("model", tag), ("dim", str(dim))):
                conn.execute(
                    f"INSERT INTO {meta} (key, value) VALUES (%s, %s) "
                    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                    (k, v),
                )
            if stale:
                conn.execute(f"DELETE FROM {t} WHERE chunk_id = ANY(%s)", (stale,))
            with conn.cursor() as cur:
                # 텍스트가 바뀐 청크: 메타+임베딩 전체 갱신
                cur.executemany(
                    f"""INSERT INTO {t} (chunk_id, position, doc_id, source, title, section, text,
                            version, effective_date, status, superseded_by, metadata,
                            text_sha256, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            position = EXCLUDED.position, doc_id = EXCLUDED.doc_id,
                            source = EXCLUDED.source, title = EXCLUDED.title,
                            section = EXCLUDED.section, text = EXCLUDED.text,
                            version = EXCLUDED.version, effective_date = EXCLUDED.effective_date,
                            status = EXCLUDED.status, superseded_by = EXCLUDED.superseded_by,
                            metadata = EXCLUDED.metadata, text_sha256 = EXCLUDED.text_sha256,
                            embedding = EXCLUDED.embedding, updated_at = now()""",
                    [
                        self._row(i, c, shas[c.chunk_id]) + (_vector_literal(v),)
                        for (i, c), v in zip(to_embed, vectors)
                    ],
                )
                # 텍스트가 같은 청크: 임베딩은 그대로, 메타만 갱신 — frontmatter(status·
                # 시행일·superseded_by)는 본문 해시에 안 잡히므로 반드시 매 부팅 동기화한다.
                cur.executemany(
                    f"""UPDATE {t} SET position = %s, doc_id = %s, source = %s, title = %s,
                            section = %s, version = %s, effective_date = %s, status = %s,
                            superseded_by = %s, metadata = %s
                        WHERE chunk_id = %s""",
                    [self._meta_row(i, c) for i, c in unchanged],
                )
        self.dim = dim
        self.last_sync = {"reused": len(unchanged), "embedded": len(to_embed), "deleted": len(stale)}
        logger.info(
            "pgvector 색인 동기화: 청크 %d개 중 재사용 %d · 신규/변경 임베딩 %d · 삭제 %d (dim=%d)",
            len(chunks), len(unchanged), len(to_embed), len(stale), dim,
        )

    def _row(self, pos: int, c: Chunk, sha: str) -> tuple:
        # effective_date 는 파싱된 date(미상=NULL)로 저장 — SQL 필터가 문자열 비교를 하지 않도록
        return (
            c.chunk_id, pos, c.doc_id, c.source, c.title, c.section, c.text, c.version,
            self.version_filter.chunk_eff[pos], c.status, c.superseded_by,
            json.dumps(c.metadata, ensure_ascii=False, default=str), sha,
        )

    def _meta_row(self, pos: int, c: Chunk) -> tuple:
        return (
            pos, c.doc_id, c.source, c.title, c.section, c.version,
            self.version_filter.chunk_eff[pos], c.status, c.superseded_by,
            json.dumps(c.metadata, ensure_ascii=False, default=str), c.chunk_id,
        )

    # ---- 검색 ----
    def search(
        self, query_vec: SparseVec, cutoff: _dt.date, include_superseded: bool, limit: int = 0
    ) -> dict[int, float]:
        if self.dim is None:
            return {}
        t = self.table
        sql = f"""
            WITH succ AS (SELECT doc_id, MIN(effective_date) AS eff FROM {t} GROUP BY doc_id)
            SELECT c.position, 1 - (c.embedding <=> %(q)s::vector) AS sim
            FROM {t} c LEFT JOIN succ ON succ.doc_id = c.superseded_by
            WHERE {_FILTER_SQL}
            ORDER BY c.embedding <=> %(q)s::vector
        """
        params = {
            "q": _vector_literal(_dense_list(query_vec)),
            "cutoff": cutoff,
            "include_superseded": include_superseded,
        }
        if limit > 0:
            sql += " LIMIT %(limit)s"
            params["limit"] = limit
        with self._lock:
            for attempt in (1, 2):
                try:
                    conn = self._query_conn()
                    with conn.transaction():
                        if limit > 0:
                            # HNSW 는 ef_search 개의 후보만 보고 필터를 적용하므로, 풀보다
                            # 작으면 LIMIT 보다 적게 돌아온다 — 풀 크기에 맞춘다(상한 1000).
                            conn.execute(
                                "SELECT set_config('hnsw.ef_search', %s, true)",
                                (str(min(max(40, limit * 2), 1000)),),
                            )
                        rows = conn.execute(sql, params).fetchall()
                    break
                except self._psycopg.OperationalError:
                    # 끊어진 연결(DB 재시작·유휴 타임아웃)은 1회 재접속 후 재시도
                    self._conn = None
                    if attempt == 2:
                        raise
        return {int(pos): float(sim) for pos, sim in rows}


def get_vector_store(
    embedder: EmbeddingProvider, kind: str | None = None, embedder_kind: str | None = None
) -> VectorStore:
    """스토어 팩토리. VECTOR_STORE 설정으로 교체한다.

    - memory(기본): 인메모리.
    - pgvector: Postgres. 밀집 임베더가 전제다 — EMBEDDER_KIND=auto 가 TF-IDF 로 폴백한
      경우는 '명시 설정이 아닌 환경 폴백'이므로 warning 과 함께 인메모리로 함께 폴백하고,
      tfidf/hashing 을 명시한 조합은 시끄럽게 실패한다(명시 설정은 시끄럽게 실패 원칙).
      PGVECTOR_URL 부재·DB 접속 실패도 전파한다 — 기동 전 점검은 preflight 가 맡는다.
    """
    kind = kind or config.VECTOR_STORE
    embedder_kind = embedder_kind or config.EMBEDDER_KIND
    if kind == "memory":
        return InMemoryVectorStore(embedder)
    if kind == "pgvector":
        if embedder_kind == "auto" and not isinstance(embedder, DENSE_PROVIDERS):
            logger.warning(
                "VECTOR_STORE=pgvector 이지만 밀집임베딩이 폴백(%s)됨 — 인메모리 스토어로 함께 폴백",
                type(embedder).__name__,
            )
            return InMemoryVectorStore(embedder)
        return PgVectorStore(embedder)
    raise ValueError(f"알 수 없는 벡터 스토어: {kind}")
