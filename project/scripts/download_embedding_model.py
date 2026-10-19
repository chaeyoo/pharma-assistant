"""밀집임베딩 모델 1회 다운로드 스크립트.

서버 기동(EMBEDDER_KIND=auto)은 다운로드를 하지 않는다 — 미캐시면 TF-IDF 로
폴백할 뿐이다(기동이 수백 MB 다운로드로 조용히 블록되는 것을 막는 설계).
밀집임베딩을 쓰려면 이 스크립트로 명시적으로 받아둔다:

    .venv/bin/pip install -r requirements-ml.txt
    .venv/bin/python scripts/download_embedding_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402


def main() -> int:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers 미설치 — 먼저 실행: .venv/bin/pip install -r requirements-ml.txt")
        return 1
    print(f"모델 다운로드: {config.EMBEDDING_MODEL} (허깅페이스 캐시에 저장)")
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    dim = model.get_sentence_embedding_dimension()
    print(f"완료 — 차원={dim}. 이후 기동은 EMBEDDER_KIND=auto 로 밀집임베딩을 사용한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
