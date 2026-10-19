"""PDF 인제스트 경로(scripts/ingest_pdf.py) — 실데이터 반입 파이프의 회귀 가드. (v8)

코퍼스가 md 샘플이라는 것과 별개로, 실무의 PDF 문서가 loader/chunker 가
소화하는 규격으로 변환되는 경로가 '작동하는 코드'로 존재함을 보증한다.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ingest_pdf import ingest
from src.rag.chunker import chunk_documents
from src.rag.loader import load_documents


def _make_minimal_pdf(path: Path) -> None:
    """텍스트 1줄이 든 최소 유효 PDF 를 손으로 조립한다(외부 의존성 없이)."""
    content = b"BT /F1 14 Tf 50 750 Td (1. Scope) Tj 0 -20 Td (This guideline covers post-market safety reporting.) Tj ET"
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objs)+1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n").encode()
    path.write_bytes(out)


def test_pdf_ingest_produces_loader_compatible_markdown(tmp_path):
    pdf = tmp_path / "guideline.pdf"
    _make_minimal_pdf(pdf)
    out = ingest(pdf, doc_id="REG-901", title="시판후 안전성 보고 가이드(수입)",
                 effective_date="2026-01-01", outdir=tmp_path / "imported")
    text = out.read_text(encoding="utf-8")
    assert text.startswith("---\n") and "doc_id: REG-901" in text
    assert "disclaimer:" in text                    # 검토 전 반입 금지 신호
    assert "## 1. Scope" in text                    # 헤딩 휴리스틱 적용
    # 기존 파이프라인(loader → chunker)이 그대로 소화한다
    docs = load_documents(out.parent)
    assert len(docs) == 1 and docs[0].doc_id == "REG-901"
    chunks = chunk_documents(docs, 500, 80)
    assert chunks and chunks[0].doc_id == "REG-901"
    assert any("post-market" in c.text for c in chunks)


def test_pdf_ingest_rejects_bad_effective_date(tmp_path):
    pdf = tmp_path / "g.pdf"
    _make_minimal_pdf(pdf)
    with pytest.raises(ValueError):
        ingest(pdf, doc_id="R-1", title="t", effective_date="2026/01/01",
               outdir=tmp_path / "imported")
