"""실데이터 인제스트 경로 시연 — PDF → 코퍼스 마크다운(+frontmatter) 변환 (v8).

이 데모의 코퍼스는 md+frontmatter 로 손질된 샘플이지만, 실제 현장의 병목은
식약처 고시·사내 SOP 같은 **PDF/HWP 문서의 수집·파싱**이다. 이 스크립트는 그
"확장 지점"이 가설이 아니라 작동하는 경로임을 증명한다:

    PDF → (pypdf 텍스트 추출) → 헤딩 휴리스틱 → frontmatter 부착 →
    data/imported/<doc_id>.md → 기존 loader/chunker/retriever 가 그대로 소화

사용:
    .venv/bin/python -m scripts.ingest_pdf 문서.pdf \
        --doc-id REG-101 --title "OO 가이드라인" --effective-date 2026-01-01
    # → data/imported/REG-101.md 생성. 검토 후 data/regulations/ 로 옮기고
    #   preflight 로 frontmatter 계약을 검사한 뒤 서버를 재기동한다(부팅 1회
    #   인덱싱 — 핫리로드는 없다). 평가셋(qa/holdout)에는 자동 포함되지 않는다.

정직한 경계:
  - pypdf 는 텍스트 '추출'만 한다 — 표·2단 조판·스캔본(OCR 필요)은 범위 밖이고,
    실무에선 이 자리를 상용 파서(예: Upstage Document Parse, HWP 변환기)로
    교체한다. 이 스크립트는 그 교체 자리의 인터페이스를 고정하는 최소 구현이다.
  - 산출물은 사람 검토를 전제로 data/imported/ 에 떨어진다 — 검토 없이 코퍼스에
    바로 넣지 않는다(코퍼스는 검색 근거의 신뢰 소스라, 파싱 오류가 곧 오답의
    근거가 된다).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys
from pathlib import Path

# 헤딩 휴리스틱: "1. 개요" / "제1조" / 짧은 단독 대문자 줄 등을 섹션 경계로 본다.
_HEADING_LINE_RE = re.compile(r"^(\d+(?:\.\d+)*[.)]\s+\S|제\s*\d+\s*[조장절]|[A-Z][A-Za-z ]{2,40}$)")


def _extract_text(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:  # 명확히 실패 — 조용한 폴백 금지
        raise SystemExit(
            "pypdf 가 필요합니다: .venv/bin/pip install -r requirements-dev.txt"
        )
    reader = PdfReader(str(pdf_path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _to_markdown(text: str) -> str:
    """추출 텍스트에 헤딩 휴리스틱을 적용해 '## 섹션' 구조를 입힌다."""
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            out.append("")
            continue
        if _HEADING_LINE_RE.match(stripped):
            out.append(f"## {stripped}")
        else:
            out.append(stripped)
    return "\n".join(out).strip() + "\n"


def ingest(pdf_path: Path, doc_id: str, title: str, effective_date: str,
           version: str = "1.0", outdir: Path | None = None) -> Path:
    """PDF 1건을 코퍼스 규격 마크다운으로 변환해 저장하고 경로를 반환한다."""
    _dt.date.fromisoformat(effective_date)  # 형식 오류는 여기서 시끄럽게
    body = _to_markdown(_extract_text(pdf_path))
    if not body.strip():
        raise SystemExit(f"{pdf_path}: 추출된 텍스트가 없습니다(스캔본이면 OCR 필요)")
    outdir = outdir or Path(__file__).resolve().parent.parent / "data" / "imported"
    outdir.mkdir(parents=True, exist_ok=True)
    md = (
        "---\n"
        f"doc_id: {doc_id}\n"
        f"title: {title}\n"
        f"version: \"{version}\"\n"
        f"effective_date: {effective_date}\n"
        f"source_file: {pdf_path.name}\n"
        "disclaimer: PDF 자동 변환본 — 사람 검토 전 코퍼스 반입 금지\n"
        "---\n\n"
        f"{body}"
    )
    out = outdir / f"{doc_id}.md"
    out.write_text(md, encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="PDF → 코퍼스 마크다운 인제스트")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--effective-date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--version", default="1.0")
    args = ap.parse_args(argv)
    out = ingest(args.pdf, args.doc_id, args.title, args.effective_date, args.version)
    print(f"생성: {out}")
    print("다음 단계: 내용 검토 → data/regulations/ 로 이동 → "
          ".venv/bin/python -m src.preflight → 서버 재기동(부팅 1회 인덱싱)")


if __name__ == "__main__":
    main(sys.argv[1:])
