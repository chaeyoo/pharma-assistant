"""로더·청커·검색 파이프라인 기본 동작(pytest).

실행:  pytest            (권장)
       python -m tests.test_rag   (독립 실행도 유지)
"""
from __future__ import annotations

from src import config
from src.rag.chunker import chunk_documents
from src.rag.loader import load_documents
from src.rag.pipeline import RagPipeline


def test_loader_and_chunker():
    docs = load_documents(config.REG_DIR)
    assert len(docs) >= 12, "확장된 규제문서(하드네거티브 포함) 12건 이상 로드"
    assert all(d.title and d.text for d in docs), "모든 문서에 제목·본문 존재"
    # frontmatter 메타(버전/시행일)가 파싱되는지
    assert any(d.metadata.get("version") for d in docs), "버전 메타 파싱"
    chunks = chunk_documents(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    assert len(chunks) > len(docs), "청킹으로 청크 수가 문서 수보다 많아야 함"
    # 버전 필드가 청크로 전파되는지
    assert all(c.title and c.text for c in chunks)
    assert any(c.status == "superseded" for c in chunks), "폐지본 청크가 존재(버전 테스트용)"


def test_retrieval_relevance(pipeline: RagPipeline):
    # 자연어 질문형 — 도메인 정답 문서가 1순위로 나와야 한다.
    cases = [
        ("신약 품목허가 심사는 며칠 이내에 처리되나요?", "REG-001"),
        ("중대한 이상사례는 며칠 이내에 보고해야 하나요?", "REG-005"),
        ("GMP 데이터 완전성 ALCOA 원칙은 무엇인가요?", "REG-003"),
        ("변경신고는 며칠 이내에 수리 통보되나요?", "REG-002"),
        ("의료기기 4등급 신개발 제품의 품목허가 심사는 며칠 이내인가요?", "REG-007"),
    ]
    for query, gold in cases:
        ctx = pipeline.retrieve(query)
        assert ctx.chunks, f"'{query}' 결과 없음"
        assert ctx.chunks[0].chunk.doc_id == gold, (
            f"'{query}' → 기대 {gold}, 실제 {ctx.chunks[0].chunk.doc_id}"
        )
        assert ctx.citations(), "출처가 비어있지 않아야 함"


def test_citations_carry_version(pipeline: RagPipeline):
    ctx = pipeline.retrieve("신약 품목허가 심사는 며칠 이내에 처리되나요?")
    cite = ctx.citations()[0]
    assert cite["doc_id"] and cite["title"]
    assert "version" in cite and "effective_date" in cite


if __name__ == "__main__":
    # 독립 실행(하위호환): fixture 없이 파이프라인 직접 구축
    pipe = RagPipeline().build()
    test_loader_and_chunker()
    test_retrieval_relevance(pipe)
    test_citations_carry_version(pipe)
    print("스모크 테스트 통과 ✅ (자세한 검증은 pytest 로 실행)")


# ---------------------------------------------------------------------------
# v8 — 동의어 표제어 경계·HashingEmbedder 부호 누적
# ---------------------------------------------------------------------------
def test_expand_query_english_terms_need_word_boundary():
    """영문 표제어는 단어 경계에서만 발화한다 — 합성어 내부 매칭은 주제 표류. (v8)

    "audit finding"의 'ind'가 임상시험 확장을, "PVC 포장재"의 'pv'가 약물감시
    확장을 일으켜 무관 문서로 순위가 밀리던 오탐."""
    from src.rag.synonyms import expand_query
    assert expand_query("audit finding 대응 방안") == "audit finding 대응 방안"
    assert expand_query("PVC 포장재 기준") == "PVC 포장재 기준"
    # 경계가 있으면 여전히 확장된다
    assert "약물감시" in expand_query("PV 업무 범위")
    assert "임상시험계획 승인" in expand_query("IND 제출 일정")


def test_expand_query_longest_term_wins_over_substring():
    """최장 일치 우선(v9): "라벨링"이 매칭한 구간에서 부분 문자열 표제어
    "라벨"은 발화하지 않는다 — 라벨링과 무관한 확장어(첨부문서)가 실려
    주제가 흐려지는 오탐 방지. 짧은 표제어 단독 질의의 확장은 불변이다."""
    from src.rag.synonyms import expand_query
    e = expand_query("라벨링 문구 기준은?")
    assert "표시기재" in e and "첨부문서" not in e
    e2 = expand_query("라벨 문구 기준은?")
    assert "표시기재" in e2 and "첨부문서" in e2


def test_expand_query_korean_terms_keep_substring_match():
    """한글 표제어는 조사·활용 직결("심각하게")이 정상이라 부분 매칭을 유지한다. (v8)"""
    from src.rag.synonyms import expand_query
    assert "중대한" in expand_query("부작용이 심각하게 나타나면")


def test_hashing_embedder_is_order_invariant():
    """HashingEmbedder 는 bag-of-words — 토큰 순서가 벡터를 바꾸면 안 된다. (v8)

    종전에는 버킷 부호를 '마지막 토큰'이 덮어써서, 반대 부호 토큰이 충돌한
    버킷의 성분이 순서에 따라 반전됐다(signed hashing 의 상쇄 원리 위반)."""
    from src.rag.embedder import HashingEmbedder
    e = HashingEmbedder(n_buckets=1)  # 모든 토큰이 한 버킷에 충돌하는 극단 케이스
    e.fit(["tok0 tok1"])
    assert e.embed("tok0 tok1") == e.embed("tok1 tok0")


def test_sweep_propagates_accept_doc_ids(pipeline, monkeypatch):
    """스윕 채점기도 복수 정답 교정(accept_doc_ids)을 전파한다 — evaluate 에만
    배선하면, 하이퍼파라미터를 흔든 설정에서 동등 정답 문항이 miss 로 갈려
    라벨 아티팩트가 파라미터 효과로 오귀속된다(v9 패턴4 의 sweep 미전파, v10).

    한 문항의 top1 문서를 primary 라벨에서 빼고 accept 집합에만 넣는다 —
    단일 라벨(구버전)이면 miss(Hit@1=0), _gold_ids 전파면 hit(Hit@1=1)."""
    from eval import sweep
    qa = sweep._load_qa()
    item = dict(qa[0])
    res = pipeline.retriever.retrieve(item["question"], top_k=config.RETRIEVE_TOP_K, rerank_n=1)
    top1 = res[0].chunk.doc_id
    item["relevant_doc_id"] = "REG-DOES-NOT-EXIST"
    item["accept_doc_ids"] = ["REG-DOES-NOT-EXIST", top1]
    monkeypatch.setattr(sweep, "_load_qa", lambda: [item])
    out = sweep.run_config(pipeline, rerank_n=1)
    assert out["Hit@1"] == 1.0


def test_tokenize_nfkc_folds_compatibility_chars():
    """tokenize 가 NFKC 로 호환·전각 문자를 접어 색인 가능하게 만든다(v12).

    실 규제문서에 흔한 ㎎·전각 영숫자가 종전에는 [A-Za-z0-9]·[가-힣] 어디에도
    안 걸려 통째로 탈락(검색 불가)했다. 코퍼스·질의가 같은 tokenize 를 지나므로
    양측 대칭 정규화된다."""
    from src.rag.textutil import tokenize
    assert "500mg" in tokenize("500㎎ 이하")        # ㎎→mg 접혀 토큰화
    assert "gmp" in tokenize("ＧＭＰ 실태조사")       # 전각→반각
