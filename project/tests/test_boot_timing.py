"""부팅 타이밍 로그 — RagPipeline.build() 가 콜드스타트 분해 한 줄을 남기는지.

Cloud Run 처럼 0대로 스케일되는 환경에서 '모델 로드 / 로드·청킹 / 색인' 중 어디에
시간이 쓰이는지, pgvector 증분 색인이 임베딩을 얼마나 생략했는지를 로그만으로
읽기 위한 계약이다(deploy/gcp/README.md 4차 절이 이 줄을 참조한다).
"""
from __future__ import annotations

import logging
import re

from src.rag.pipeline import RagPipeline


def test_build_logs_cold_start_breakdown(caplog):
    with caplog.at_level(logging.INFO, logger="rapv_assistant"):
        p = RagPipeline(embedder_kind="tfidf", vector_store="memory").build()
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("RAG 인덱스 구축:")]
    assert len(lines) == 1, caplog.text
    line = lines[0]
    assert f"문서 {p.n_docs} · 청크 {p.n_chunks}" in line
    assert "임베더 TfidfEmbedder · 스토어 InMemoryVectorStore" in line
    # 수치 필드 4종이 모두 ms 단위로 찍힌다
    for field in ("모델로드", "로드/청킹", "색인", "합계"):
        assert re.search(rf"{field} \d+ms", line), line
    # 인메모리 스토어는 증분 색인 정보가 없다(pgvector 만 재사용/신규 수치를 붙인다)
    assert "재사용" not in line
    assert p.build_ms >= p.prep_ms
