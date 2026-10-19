"""문서 로더 (Document Loader).

규제문서(markdown + YAML frontmatter)를 읽어 메타데이터와 본문으로 분리한다.
실무의 PDF/HWP 로더 자리에 해당하며, 여기서는 데모를 위해 markdown을 사용한다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Document:
    """로드된 원문 문서 1건."""
    doc_id: str
    source: str                       # 파일명
    title: str
    metadata: dict[str, Any] = field(default_factory=dict)
    text: str = ""


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    """아주 단순한 YAML frontmatter 파서(외부 의존성 없이 데모용).

    key: value / key: [a, b] 형태만 처리한다.
    """
    m = _FRONTMATTER_RE.match(raw)
    if not m:
        return {}, raw
    body = raw[m.end():]
    meta: dict[str, Any] = {}
    for line in m.group(1).splitlines():
        line = line.rstrip()
        if not line or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()
        if val.startswith("[") and val.endswith("]"):
            items = [x.strip().strip('"').strip("'") for x in val[1:-1].split(",")]
            meta[key] = [x for x in items if x]
        else:
            meta[key] = val.strip('"').strip("'")
    return meta, body


def load_documents(reg_dir: Path) -> list[Document]:
    """디렉터리 내 모든 .md 규제문서를 로드한다."""
    docs: list[Document] = []
    for path in sorted(reg_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw)
        docs.append(
            Document(
                doc_id=str(meta.get("doc_id", path.stem)),
                source=path.name,
                title=str(meta.get("title", path.stem)),
                metadata=meta,
                text=body.strip(),
            )
        )
    return docs
