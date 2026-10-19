"""버전 인지 검색 테스트: 폐지본 제외 · 이력 조회 · 과거 시점(as_of)."""
from __future__ import annotations

# 서버 내부 단위 테스트 — 도구 함수를 직접 호출한다(HTTP 클라이언트 경로는 test_mcp/test_agent 가 커버)
from src.mcp_server.server import search_regulations


def test_superseded_excluded_by_default(pipeline):
    # 폐지 구판(REG-013)은 기본 검색에서 제외되어야 한다.
    res = pipeline.retriever.retrieve("이상사례 보고 기한", top_k=8, rerank_n=8)
    assert all(s.chunk.status != "superseded" for s in res)
    assert all(s.chunk.doc_id != "REG-013" for s in res)


def test_include_superseded_surfaces_old_version(pipeline):
    res = pipeline.retriever.retrieve(
        "이상사례 보고 기한", top_k=12, rerank_n=12, include_superseded=True
    )
    assert any(s.chunk.doc_id == "REG-013" for s in res), "이력 조회 시 폐지본 노출"


def test_current_answer_prefers_effective_doc(pipeline):
    # 현행(REG-005, 15일)이 폐지본(REG-013, 30일)보다 우선.
    top = pipeline.retriever.retrieve("중대한 이상사례 보고 기한", top_k=8, rerank_n=1)[0]
    assert top.chunk.doc_id == "REG-005"
    assert top.chunk.status == "active"


def test_as_of_excludes_not_yet_effective(pipeline):
    # 2024-06-01 시점에는 2025 시행 문서들이 아직 유효하지 않다.
    res = pipeline.retriever.retrieve(
        "품목허가 심사 기간", top_k=20, rerank_n=20, as_of="2024-06-01"
    )
    for s in res:
        if s.chunk.effective_date:
            assert s.chunk.effective_date <= "2024-06-01"


def test_as_of_includes_then_active_superseded(pipeline):
    """as_of 시점 조회의 핵심 의미론: '그 시점의 현행'을 반환한다.

    REG-013(구판, 2021-01-01 시행)은 지금은 폐지본이지만 2025-01-01 시점에는
    현행이었다(후속본 REG-005는 2025-04-01 시행). '폐지본 기본 제외'를 as_of에도
    그대로 적용하면 신·구판이 모두 걸러져 시점 조회가 0건이 된다 — 개정 이력이
    있는(=시점 조회가 필요한) 규정에서만 무너지는 사각지대였다."""
    res = pipeline.retriever.retrieve(
        "이상사례 보고 기한", top_k=8, rerank_n=3, as_of="2025-01-01"
    )
    assert res, "시점 조회 결과가 0건 — 당시 현행 버전이 필터에서 걸러졌다"
    assert any(s.chunk.doc_id == "REG-013" for s in res), "당시 현행이던 구판이 포함되어야 함"
    assert all(s.chunk.doc_id != "REG-005" for s in res), "그 시점에 시행 전인 현행본은 제외"


def test_as_of_after_revision_returns_current(pipeline):
    # 후속본 시행(2025-04-01) 이후 시점에는 구판이 다시 제외된다.
    res = pipeline.retriever.retrieve(
        "이상사례 보고 기한", top_k=8, rerank_n=8, as_of="2025-06-01"
    )
    assert any(s.chunk.doc_id == "REG-005" for s in res)
    assert all(s.chunk.doc_id != "REG-013" for s in res)


def test_mcp_search_exposes_version_fields():
    out = search_regulations("신약 품목허가 심사 기간", top_n=3)
    r = out["results"][0]
    assert "version" in r and "effective_date" in r and "status" in r


def test_malformed_effective_date_fails_closed_under_as_of():
    """시행일을 해석할 수 없는 문서는 시점(as_of) 조회에서 제외된다(fail-closed)
    — 예외 전파로 검색 전체가 죽지도, 무필터로 통과해 오답에 섞이지도 않는다.
    (데이터 결함 자체는 preflight 코퍼스 무결성 검사가 파일명과 함께 보고한다)"""
    from src.rag.chunker import Chunk
    from src.rag.embedder import get_embedder
    from src.rag.retriever import HybridRetriever
    from src.rag.vectorstore import InMemoryVectorStore

    chunks = [
        Chunk(chunk_id="ok::s0::w0", doc_id="OK", source="ok.md", title="정상 규정",
              section="본문", text="[정상 규정 > 본문]\n심사 기간은 60일이다.",
              effective_date="2024-01-01"),
        Chunk(chunk_id="bad::s0::w0", doc_id="BAD", source="bad.md", title="결함 규정",
              section="본문", text="[결함 규정 > 본문]\n심사 기간은 999일이다.",
              effective_date="2025.08.01"),  # 형식 오류(점 표기)
    ]
    r = HybridRetriever(InMemoryVectorStore(get_embedder("tfidf")))
    r.index(chunks)
    res = r.retrieve("심사 기간", top_k=5, rerank_n=5, as_of="2026-01-01")  # 크래시 없어야 함
    ids = {s.chunk.doc_id for s in res}
    assert "OK" in ids and "BAD" not in ids
    # 기본 검색도 'as_of=오늘'과 동치라(v9) 시행일 해석 불가 청크는 동일하게
    # fail-closed 제외된다 — 실코퍼스에서는 preflight 가 시행일 유효성을 기동
    # 전에 보증하므로, 이 경로는 데이터 결함이 게이트를 뚫었을 때 그 문서가
    # '현행'으로 나가는 것을 막는 마지막 방어다.
    res_all = r.retrieve("심사 기간", top_k=5, rerank_n=5)
    ids_all = {s.chunk.doc_id for s in res_all}
    assert "OK" in ids_all and "BAD" not in ids_all


def test_default_search_equals_as_of_today():
    """기본 검색(무 as_of)은 'as_of=오늘'과 동치다(v9).

    종전 무 as_of 경로는 status 만 보고 시행일을 안 봐서, (a) 미래 시행일의
    active 문서가 '현행'으로 반환되고 (b) 그 문서로 대체될 폐지본 — 후속본이
    아직 시행 전이라 오늘의 실제 현행 — 은 은폐됐다. v8 이 as_of 경로에서
    봉합한 조합 결함과 동형이 기본 경로에 남아 있던 것. 현 코퍼스(REG-013 의
    후속본 REG-005 는 이미 시행)에서는 발화하지 않으므로 합성 청크로 고정한다."""
    from src.rag.chunker import Chunk
    from src.rag.embedder import get_embedder
    from src.rag.retriever import HybridRetriever
    from src.rag.vectorstore import InMemoryVectorStore

    chunks = [
        # 오늘의 실제 현행: 폐지본이지만 후속본(FUT)이 아직 시행 전
        Chunk(chunk_id="cur::s0::w0", doc_id="CUR", source="cur.md", title="보고 기한 규정",
              section="본문", text="[보고 기한 규정 > 본문]\n심사 기간은 60일이다.",
              effective_date="2020-01-01", status="superseded", superseded_by="FUT"),
        # 미래 시행 active: 아직 오늘의 현행이 아니다
        Chunk(chunk_id="fut::s0::w0", doc_id="FUT", source="fut.md", title="보고 기한 규정 개정판",
              section="본문", text="[보고 기한 규정 개정판 > 본문]\n심사 기간은 30일이다.",
              effective_date="2100-01-01", status="active"),
    ]
    r = HybridRetriever(InMemoryVectorStore(get_embedder("tfidf")))
    r.index(chunks)

    # (a) 미래 시행 active 는 기본 검색에서 제외 +
    # (b) 후속본 미시행 구간의 폐지본은 include_superseded=False 여도
    #     '오늘의 현행'으로 포함
    ids = {s.chunk.doc_id for s in r.retrieve("심사 기간", top_k=5, rerank_n=5)}
    assert ids == {"CUR"}, f"기본 검색이 '오늘의 현행'과 다르다: {ids}"

    # (c) 기존 as_of 동작 불변: 구간 판정 [시행일, 후속본 시행일) 그대로
    assert {s.chunk.doc_id for s in r.retrieve("심사 기간", top_k=5, rerank_n=5,
                                               as_of="2099-12-31")} == {"CUR"}
    assert {s.chunk.doc_id for s in r.retrieve("심사 기간", top_k=5, rerank_n=5,
                                               as_of="2100-01-01")} == {"FUT"}
    assert r.retrieve("심사 기간", top_k=5, rerank_n=5, as_of="2019-12-31") == []


def test_as_of_two_tier_chain_returns_exactly_then_active(tmp_path):
    """2단 폐지 체인(v1→v2→현행 v3)에서 as_of 는 '그 시점의 현행' 정확히
    한 버전만 반환한다. (v8)

    superseded_by 는 '직전 후속본' 규약이다 — 구간 판정 [시행일, 후속본
    시행일)이 성립하는 전제. preflight 의 체인 검사(다단 순회)와 같은 규약을
    리트리버 쪽에서도 회귀 테스트로 고정한다(둘이 반대 규약을 요구하면 다단
    이력을 넣는 순간 어느 쪽으로 써도 한쪽이 깨진다)."""
    from src.rag.pipeline import RagPipeline

    def _doc(name: str, doc_id: str, eff: str, extra: str = "") -> None:
        (tmp_path / name).write_text(
            f"---\ndoc_id: {doc_id}\ntitle: 보고 기한 규정\nversion: '{doc_id[-1]}'\n"
            f"effective_date: {eff}\n{extra}---\n\n## 보고 기한\n이상사례 보고 기한 규정 본문.\n",
            encoding="utf-8",
        )

    _doc("v1.md", "R-V1", "2020-01-01", "status: superseded\nsuperseded_by: R-V2\n")
    _doc("v2.md", "R-V2", "2022-01-01", "status: superseded\nsuperseded_by: R-V3\n")
    _doc("v3.md", "R-V3", "2024-01-01")
    p = RagPipeline(reg_dir=tmp_path).build()

    for as_of, expected in [("2021-06-01", "R-V1"), ("2023-06-01", "R-V2"), ("2025-06-01", "R-V3")]:
        res = p.retriever.retrieve("이상사례 보고 기한", top_k=8, rerank_n=8, as_of=as_of)
        ids = {s.chunk.doc_id for s in res}
        assert ids == {expected}, f"as_of={as_of}: {ids} (기대: {expected} 단독)"
