"""벡터 스토어 테스트 — 인메모리/pgvector 계약, 증분 색인, SQL 버전 필터 동치.

pgvector 테스트는 PGVECTOR_TEST_URL 이 설정된 환경에서만 실 DB 로 돌고(CI 기본은
skip), 밀집 임베더는 `.encode()` 만 흉내내는 결정적 스텁을 주입해 torch 없이 검증한다.
"""
from __future__ import annotations

import datetime as _dt
import os
import uuid

import pytest

from src import config
from src.rag.chunker import Chunk
from src.rag.embedder import SentenceTransformerEmbedder, TfidfEmbedder, cosine, get_embedder
from src.rag.retriever import HybridRetriever
from src.rag.vectorstore import InMemoryVectorStore, PgVectorStore, get_vector_store
from src.rag.versionfilter import VersionFilter, parse_date, resolve_cutoff

PG_URL = os.environ.get("PGVECTOR_TEST_URL", "")
needs_pg = pytest.mark.skipif(not PG_URL, reason="PGVECTOR_TEST_URL 미설정 — 실 Postgres+pgvector 필요")


class _StubSTModel:
    """결정적 8차원 스텁(전 성분 양수). 호출 기록으로 '무엇을 인코딩했는가'를 검증한다."""

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
        b = text.encode("utf-8")
        raw = [1.0 + sum(b[i::8]) % 13 for i in range(8)]
        n = sum(x * x for x in raw) ** 0.5
        return [x / n for x in raw]


def _chunk(cid: str, text: str, *, doc_id: str = "", eff: str = "2024-01-01",
           status: str = "active", succ: str = "", title: str = "제목", section: str = "본문") -> Chunk:
    return Chunk(
        chunk_id=cid, doc_id=doc_id or cid.split("::")[0], source=f"{cid}.md", title=title,
        section=section, text=text, version="v1", effective_date=eff, status=status, superseded_by=succ,
    )


# ---------------------------------------------------------------------------
# 버전 필터(공유 구현) · 팩토리
# ---------------------------------------------------------------------------
def test_version_filter_rules():
    chunks = [
        _chunk("A::0", "현행", eff="2024-01-01"),
        _chunk("F::0", "미래 시행", eff="2030-01-01"),
        _chunk("N::0", "시행일 미상", eff="not-a-date"),
        _chunk("S::0", "폐지본(후속 A)", eff="2020-01-01", status="superseded", succ="A"),
        _chunk("T::0", "폐지본(후속 F, 아직 시행 전)", eff="2020-01-01", status="superseded", succ="F"),
        _chunk("U::0", "폐지본(후속 없음)", eff="2020-01-01", status="superseded", succ="ZZZ"),
    ]
    vf = VersionFilter.from_chunks(chunks)
    today = _dt.date(2025, 6, 1)
    assert vf.candidates(today, False) == [0, 4]            # T 는 '오늘의 현행'(후속본 미시행)
    assert vf.candidates(today, True) == [0, 3, 4, 5]       # 폐지본 포함 — 미래·미상은 여전히 제외
    assert vf.candidates(_dt.date(2022, 1, 1), False) == [3, 4]  # 당시 현행 폐지본, A 는 시행 전
    assert parse_date("") is None and parse_date("2024-13-01") is None
    assert resolve_cutoff("") == _dt.date.today()
    with pytest.raises(ValueError):
        resolve_cutoff("2024/01/01")


def test_memory_store_search_matches_direct_cosine():
    emb = TfidfEmbedder()
    store = InMemoryVectorStore(emb)
    chunks = [_chunk("A::0", "품목허가 심사 기간"), _chunk("B::0", "이상사례 보고 기한", eff="2030-01-01")]
    store.index(chunks)
    qv = emb.embed("품목허가 심사")
    sims = store.search(qv, _dt.date(2025, 1, 1), False, limit=1)
    assert set(sims) == {0}, "미래 시행 청크 제외 · 인메모리는 limit 무시(전수)"
    assert sims[0] == pytest.approx(cosine(qv, store.vectors[0]))


def test_factory_memory_default_and_pgvector_contracts(monkeypatch):
    assert isinstance(get_vector_store(TfidfEmbedder(), "memory"), InMemoryVectorStore)
    with pytest.raises(ValueError):
        get_vector_store(TfidfEmbedder(), "nope")
    # 명시 희소 임베더 + pgvector 는 시끄럽게 실패
    with pytest.raises(ValueError, match="밀집 임베더 전용"):
        get_vector_store(TfidfEmbedder(), "pgvector", embedder_kind="tfidf")
    # auto 가 tfidf 로 폴백한 환경이면 스토어도 인메모리로 함께 폴백(경고)
    assert isinstance(get_vector_store(TfidfEmbedder(), "pgvector", embedder_kind="auto"), InMemoryVectorStore)
    # URL 부재
    pytest.importorskip("psycopg")
    monkeypatch.setattr(config, "PGVECTOR_URL", "")
    with pytest.raises(ValueError, match="PGVECTOR_URL"):
        PgVectorStore(SentenceTransformerEmbedder(model=_StubSTModel()))


def test_retriever_fills_pool_gap_with_floor():
    """스토어가 풀(limit)만 돌려줄 때 풀 밖 후보는 풀 최솟값으로 채워진다."""

    class _PooledStore(InMemoryVectorStore):
        def search(self, qv, cutoff, incl, limit=0):
            full = super().search(qv, cutoff, incl, 0)
            top = sorted(full.items(), key=lambda kv: kv[1], reverse=True)[:limit] if limit else full.items()
            return dict(top)

    emb = TfidfEmbedder()
    chunks = [_chunk(f"D{i}::0", t) for i, t in enumerate(
        ["품목허가 신청 서류", "품목허가 심사 기간", "이상사례 보고", "GMP 실사", "라벨 표시기재"])]
    exact = HybridRetriever(InMemoryVectorStore(emb), vector_pool=0)
    exact.index(chunks)
    pooled = HybridRetriever(_PooledStore(emb), vector_pool=2)
    pooled.index(chunks)
    qv = emb.embed("품목허가")
    idxs = list(range(5))
    ex = exact._vector_scores(qv, idxs, "", False)
    po = pooled._vector_scores(qv, idxs, "", False)
    top2 = sorted(range(5), key=lambda i: ex[i], reverse=True)[:2]
    assert all(po[i] == ex[i] for i in top2)
    floor = min(po[i] for i in top2)
    assert all(po[i] == floor for i in idxs if i not in top2)
    # 풀 ≥ 후보 수면 전수와 동일
    big = HybridRetriever(_PooledStore(emb), vector_pool=99); big.index(chunks)
    assert big._vector_scores(qv, idxs, "", False) == ex


# ---------------------------------------------------------------------------
# pgvector (실 DB)
# ---------------------------------------------------------------------------
@pytest.fixture
def pg_table():
    """테스트별 고유 테이블 — 끝나면 삭제."""
    import psycopg

    name = f"test_rag_{uuid.uuid4().hex[:12]}"
    yield name
    with psycopg.connect(PG_URL, autocommit=True) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {name}")
        conn.execute(f"DROP TABLE IF EXISTS {name}_meta")


def _pg_store(stub: _StubSTModel, table: str, pool: int = 0) -> PgVectorStore:
    return PgVectorStore(SentenceTransformerEmbedder(model=stub), url=PG_URL, pool=pool, table=table)


@needs_pg
def test_pg_incremental_index(pg_table):
    corpus = [_chunk("A::0", "품목허가 신청 서류"), _chunk("B::0", "이상사례 보고 기한"), _chunk("C::0", "GMP 실사")]
    stub = _StubSTModel()
    s1 = _pg_store(stub, pg_table); s1.index(corpus)
    assert s1.last_sync == {"reused": 0, "embedded": 3, "deleted": 0} and s1.dim == 8
    assert stub.calls == [[f"passage: {c.text}" for c in corpus]]

    # 재부팅(새 임베더·새 스토어, 같은 코퍼스) → 인코딩 0회
    stub2 = _StubSTModel()
    s2 = _pg_store(stub2, pg_table); s2.index(corpus)
    assert s2.last_sync == {"reused": 3, "embedded": 0, "deleted": 0}
    assert stub2.calls == [[]], "바뀐 청크가 없으면 passage 인코딩은 빈 배치 1회(= 실질 0회)"

    # 1건 변경 + 1건 삭제 + 1건 추가, 그리고 메타만 바뀐 청크(status) → 임베딩 재사용·메타는 갱신
    corpus2 = [
        _chunk("A::0", "품목허가 신청 서류", status="superseded", succ="D"),
        _chunk("B::0", "이상사례 보고 기한 (개정)"),
        _chunk("D::0", "변경허가 신고", eff="2024-06-01"),
    ]
    stub3 = _StubSTModel()
    s3 = _pg_store(stub3, pg_table); s3.index(corpus2)
    assert s3.last_sync == {"reused": 1, "embedded": 2, "deleted": 1}
    assert stub3.calls == [["passage: 이상사례 보고 기한 (개정)", "passage: 변경허가 신고"]]
    import psycopg
    with psycopg.connect(PG_URL) as conn:
        rows = dict(conn.execute(f"SELECT chunk_id, status FROM {pg_table}").fetchall())
        assert rows == {"A::0": "superseded", "B::0": "active", "D::0": "active"}
        positions = dict(conn.execute(f"SELECT chunk_id, position FROM {pg_table}").fetchall())
        assert positions == {"A::0": 0, "B::0": 1, "D::0": 2}

    # 질의: 폐지본 A 는 후속 D(2024-06-01 시행)로 대체됨 → 기본 제외, 2024-03-01 시점엔 현행
    qv = s3.embedder.embed("품목허가")
    assert set(s3.search(qv, _dt.date(2025, 1, 1), False, 0)) == {1, 2}
    assert set(s3.search(qv, _dt.date(2025, 1, 1), True, 0)) == {0, 1, 2}
    assert set(s3.search(qv, _dt.date(2024, 3, 1), False, 0)) == {0, 1}
    assert len(s3.search(qv, _dt.date(2025, 1, 1), True, 2)) == 2, "limit=풀 크기"


@needs_pg
def test_pg_model_change_rebuilds(pg_table):
    corpus = [_chunk("A::0", "텍스트")]
    s1 = PgVectorStore(SentenceTransformerEmbedder(model=_StubSTModel(), model_name="m1"), url=PG_URL, table=pg_table)
    s1.index(corpus)
    s2 = PgVectorStore(SentenceTransformerEmbedder(model=_StubSTModel(), model_name="m2"), url=PG_URL, table=pg_table)
    s2.index(corpus)
    assert s2.last_sync["embedded"] == 1, "모델이 바뀌면 저장 벡터는 무효 → 전체 재임베딩"


@needs_pg
def test_pg_sql_filter_equals_python_filter(pg_table):
    """SQL WHERE(_FILTER_SQL) 와 VersionFilter.candidates() 의 동치 — 엣지 전수."""
    chunks = [
        _chunk("A::0", "현행 a", eff="2024-01-01"),
        _chunk("A::1", "현행 a 둘째", eff="2024-01-01"),
        _chunk("F::0", "미래", eff="2030-01-01"),
        _chunk("N::0", "미상", eff=""),
        _chunk("S::0", "폐지 후속 A", eff="2020-01-01", status="superseded", succ="A"),
        _chunk("T::0", "폐지 후속 F", eff="2020-01-01", status="superseded", succ="F"),
        _chunk("U::0", "폐지 후속 없음", eff="2020-01-01", status="superseded", succ="ZZZ"),
        _chunk("V::0", "폐지 후속 미상", eff="2020-01-01", status="superseded", succ="N"),
    ]
    store = _pg_store(_StubSTModel(), pg_table); store.index(chunks)
    qv = store.embedder.embed("질의")
    for cutoff in (_dt.date(2019, 1, 1), _dt.date(2020, 1, 1), _dt.date(2022, 6, 1), _dt.date(2024, 1, 1),
                   _dt.date(2025, 1, 1), _dt.date(2030, 1, 1), _dt.date(2031, 1, 1)):
        for incl in (False, True):
            sql_set = set(store.search(qv, cutoff, incl, 0))
            py_set = set(store.version_filter.candidates(cutoff, incl))
            assert sql_set == py_set, (cutoff, incl)


@needs_pg
def test_pg_similarity_matches_python_cosine(pg_table):
    corpus = [_chunk(f"D{i}::0", t) for i, t in enumerate(["품목허가 신청", "이상사례 보고", "GMP 실사", "표시기재"])]
    pg = _pg_store(_StubSTModel(), pg_table); pg.index(corpus)
    mem = InMemoryVectorStore(SentenceTransformerEmbedder(model=_StubSTModel())); mem.index(corpus)
    qv = pg.embedder.embed("허가 신청")
    a, b = pg.search(qv, _dt.date.today(), False, 0), mem.search(qv, _dt.date.today(), False, 0)
    assert a.keys() == b.keys()
    for i in a:
        assert a[i] == pytest.approx(b[i], abs=1e-5)


@needs_pg
def test_pg_pipeline_equals_memory_pipeline_on_real_corpus(pg_table, monkeypatch):
    """실 코퍼스 + 평가 질의: pgvector(전수 풀) 와 인메모리의 최종 순위가 동일."""
    import json

    from src.rag.chunker import chunk_documents
    from src.rag.loader import load_documents

    docs = load_documents(config.REG_DIR)
    chunks = chunk_documents(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    mem = HybridRetriever(InMemoryVectorStore(SentenceTransformerEmbedder(model=_StubSTModel())), vector_pool=0)
    mem.index(chunks)
    pg = HybridRetriever(_pg_store(_StubSTModel(), pg_table), vector_pool=0)
    pg.index(chunks)
    assert pg.store.last_sync["embedded"] == len(chunks)
    items = json.loads((config.BASE_DIR / "eval" / "qa_dataset.json").read_text(encoding="utf-8"))["items"]
    for it in items:
        q = it["question"]
        a = mem.retrieve(q, top_k=8, rerank_n=3)
        b = pg.retrieve(q, top_k=8, rerank_n=3)
        assert [s.chunk.chunk_id for s in a] == [s.chunk.chunk_id for s in b], q
        # 점수는 float4 저장·DB 측 거리 계산의 부동소수 오차(≈1e-7)만 허용
        assert [s.score for s in a] == pytest.approx([s.score for s in b], abs=1e-5), q
        va = [s.chunk.chunk_id for s in mem.vector_search(q, 3)]
        vb = [s.chunk.chunk_id for s in pg.vector_search(q, 3)]
        assert va == vb, q
    # 두 번째 부팅은 전부 재사용
    pg2 = HybridRetriever(_pg_store(_StubSTModel(), pg_table), vector_pool=0); pg2.index(chunks)
    assert pg2.store.last_sync == {"reused": len(chunks), "embedded": 0, "deleted": 0}


@needs_pg
def test_pg_store_via_factory_and_reconnect(pg_table, monkeypatch):
    monkeypatch.setattr(config, "PGVECTOR_URL", PG_URL)
    store = get_vector_store(SentenceTransformerEmbedder(model=_StubSTModel()), "pgvector", embedder_kind="dense")
    assert isinstance(store, PgVectorStore)
    store.table = pg_table
    store.index([_chunk("A::0", "a"), _chunk("B::0", "b")])
    qv = store.embedder.embed("a")
    assert len(store.search(qv, _dt.date.today(), False, 0)) == 2
    store._conn.close()  # 끊어진 연결은 1회 재접속으로 복구
    assert len(store.search(qv, _dt.date.today(), False, 0)) == 2
    assert get_embedder("tfidf") is not None
