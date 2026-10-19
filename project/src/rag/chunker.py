"""청킹 (Chunking).

문서를 검색 단위(chunk)로 분할한다. RAG 품질을 좌우하는 핵심 단계.
- 구조 인식: markdown 헤딩(##)을 경계로 우선 분할하여 문맥 보존.
- 크기 제한 + 겹침(overlap): 긴 섹션은 문자 기준으로 자르되, 경계에서 정보가
  끊기지 않도록 overlap 만큼 겹쳐서 자른다.
각 청크는 원문 섹션 제목을 함께 보관해 검색·출처표시에 활용한다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .loader import Document


@dataclass
class Chunk:
    """검색 단위 청크."""
    chunk_id: str
    doc_id: str
    source: str
    title: str          # 문서 제목
    section: str        # 소속 섹션(헤딩) 제목
    text: str
    # 버전 인지 검색용 메타(규제 산업 필수: 개정/폐지 이력 추적)
    version: str = ""
    effective_date: str = ""   # ISO(YYYY-MM-DD) 시행일
    status: str = "active"     # active | superseded
    superseded_by: str = ""    # 폐지된 경우 대체 문서 doc_id
    metadata: dict[str, Any] = field(default_factory=dict)


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _split_by_heading(text: str) -> list[tuple[str, str]]:
    """(섹션제목, 섹션본문) 리스트로 분할. 헤딩 기준 구조 청킹.

    첫 헤딩 이전(또는 헤딩 없는 문서)의 기본 섹션명은 "(본문)" — "개요"로
    두면 리트리버의 서두 섹션 감쇠(_PREAMBLE_SECTION_RE: 목적|개요|총칙)에
    걸려, 헤딩 없는 문서의 **모든** 청크가 정의형 질의가 아닌 한 일괄
    감점되는 계통 편향이 생긴다(현 코퍼스는 전 문서에 헤딩이 있어 잠복
    상태였지만, 실무 문서 투입 시 바로 드러난다 — v8).
    """
    sections: list[tuple[str, str]] = []
    cur_title = "(본문)"
    cur_lines: list[str] = []
    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            if cur_lines:
                sections.append((cur_title, "\n".join(cur_lines).strip()))
                cur_lines = []
            cur_title = m.group(2).strip()
        else:
            cur_lines.append(line)
    if cur_lines:
        sections.append((cur_title, "\n".join(cur_lines).strip()))
    return [(t, b) for t, b in sections if b]


def _window_split(text: str, size: int, overlap: int) -> list[str]:
    """긴 텍스트를 size 크기, overlap 겹침으로 슬라이딩 분할."""
    if len(text) <= size:
        return [text]
    step = max(1, size - overlap)
    windows = []
    for start in range(0, len(text), step):
        piece = text[start:start + size]
        if piece.strip():
            windows.append(piece)
        if start + size >= len(text):
            break
    return windows


def chunk_documents(
    docs: list[Document], chunk_size: int, overlap: int
) -> list[Chunk]:
    """모든 문서를 청크로 변환."""
    chunks: list[Chunk] = []
    for doc in docs:
        for si, (section, body) in enumerate(_split_by_heading(doc.text)):
            for wi, window in enumerate(_window_split(body, chunk_size, overlap)):
                cid = f"{doc.doc_id}::s{si}::w{wi}"
                chunks.append(
                    Chunk(
                        chunk_id=cid,
                        doc_id=doc.doc_id,
                        source=doc.source,
                        title=doc.title,
                        section=section,
                        # 검색 정확도를 위해 문서/섹션 제목을 본문 앞에 덧붙임
                        # (Contextual Retrieval 아이디어의 경량 버전)
                        text=f"[{doc.title} > {section}]\n{window}",
                        version=str(doc.metadata.get("version", "")),
                        effective_date=str(doc.metadata.get("effective_date", "")),
                        status=str(doc.metadata.get("status", "active")),
                        superseded_by=str(doc.metadata.get("superseded_by", "")),
                        metadata={**doc.metadata, "section": section},
                    )
                )
    return chunks
