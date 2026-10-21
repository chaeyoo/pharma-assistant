"""버전 인지 필터 (Version-aware filter) — 검색 후보군 판정의 단일 구현.

"기준일(as_of)에 시행 중이던 버전만" 을 고르는 규칙이다. 벡터 스토어가 둘
(인메모리·pgvector)이고 BM25 도 같은 후보군을 써야 하므로, 판정 로직을 한 곳에
두고 모두가 이것을 참조한다. pgvector 스토어는 같은 규칙을 SQL WHERE 절로
표현하는데(vectorstore.py `_FILTER_SQL`), 두 표현의 동치는 tests/test_vectorstore.py
가 실 DB 로 고정한다.

규칙(HybridRetriever 의 종전 _candidate_indices 와 동일 — v9 설계 근거는 그 docstring):
  - as_of 미지정은 as_of=오늘과 동치.
  - 시행일 ≤ cutoff 인 청크만(시행 전 제외). 시행일 미상(해석 불가)은 제외(fail-closed).
  - 폐지본(superseded)은 include_superseded 가 아니면, 후속본 시행일 > cutoff 인
    '당시 현행' 구간일 때만 포함. 후속본 시행일 미상·후속본 부재는 제외(fail-closed).
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

from .chunker import Chunk


def parse_date(s: str) -> _dt.date | None:
    """ISO 날짜 파싱 — 실패·공백은 None. 호출부가 fail-closed 로 처리한다."""
    if not s:
        return None
    try:
        return _dt.date.fromisoformat(s)
    except ValueError:
        return None


def resolve_cutoff(as_of: str) -> _dt.date:
    """as_of → 기준일. 형식 오류는 즉시 ValueError(직접 호출자는 시끄럽게 깨지는 게 옳다)."""
    return _dt.date.fromisoformat(as_of) if as_of else _dt.date.today()


@dataclass
class VersionFilter:
    """청크 코퍼스의 시행일 메타를 인덱싱 시 1회 파싱해 두고 후보군을 판정한다."""

    chunk_eff: list[_dt.date | None] = field(default_factory=list)   # 청크별 시행일
    doc_eff: dict[str, _dt.date | None] = field(default_factory=dict)  # doc_id → 시행일(후속본 조회)
    status: list[str] = field(default_factory=list)
    superseded_by: list[str] = field(default_factory=list)

    @classmethod
    def from_chunks(cls, chunks: list[Chunk]) -> "VersionFilter":
        return cls(
            chunk_eff=[parse_date(c.effective_date) for c in chunks],
            doc_eff={c.doc_id: parse_date(c.effective_date) for c in chunks},
            status=[c.status for c in chunks],
            superseded_by=[c.superseded_by for c in chunks],
        )

    def candidates(self, cutoff: _dt.date, include_superseded: bool) -> list[int]:
        """필터를 통과한 청크 위치(코퍼스 순서 인덱스) 목록."""
        out: list[int] = []
        for i, eff in enumerate(self.chunk_eff):
            if eff is None or eff > cutoff:
                continue  # 시행일 미상(fail-closed) 또는 그 시점에 시행 전
            if self.status[i] == "superseded" and not include_superseded:
                succ = self.doc_eff.get(self.superseded_by[i])
                if succ is None or succ <= cutoff:
                    continue  # 후속본 시행일 미상이거나 이미 대체된 시점
            out.append(i)
        return out
