"""임베더 (Embedder) — 텍스트를 벡터로.

핵심 설계: **provider 교체 가능(pluggable)** 구조.
- 기본값(auto): 로컬 밀집임베딩(sentence-transformers, 의미 매칭) — 미설치/모델
  미캐시 환경에선 TF-IDF 희소 벡터로 자동 폴백해 데모가 항상 돌아간다.
- 실무 확장 지점: `EmbeddingProvider` 를 구현하면 상용 임베딩 API
  (예: OpenAI/Cohere/Voyage 임베딩)로 그대로 교체 가능하다.

벡터는 {key: weight} 형태의 dict 로 표현하고, 코사인 유사도로 비교한다.
희소 임베더는 key=단어, 밀집 임베더는 key=차원 인덱스 문자열로 담아
(VoyageEmbedder 선례) cosine()/vectorstore 계약을 공유한다.
"""
from __future__ import annotations

import functools
import hashlib
import math
import os
from collections import Counter
from typing import Protocol

from .. import config
from ..observability import logger
from .textutil import tokenize

SparseVec = dict[str, float]


class EmbeddingProvider(Protocol):
    """임베딩 제공자 인터페이스. 실무에선 이 자리에 상용 임베딩 API를 끼운다."""

    def fit(self, corpus: list[str]) -> None: ...
    def embed(self, text: str) -> SparseVec: ...


class TfidfEmbedder:
    """오프라인 TF-IDF 임베더 (기본 provider)."""

    def __init__(self) -> None:
        self._idf: dict[str, float] = {}
        self._n_docs = 0
        self._fitted = False

    def fit(self, corpus: list[str]) -> None:
        """코퍼스로 IDF(역문서빈도)를 학습한다."""
        self._n_docs = len(corpus)
        df: Counter[str] = Counter()
        for text in corpus:
            for term in set(tokenize(text)):
                df[term] += 1
        # smooth IDF
        self._idf = {
            term: math.log((self._n_docs + 1) / (freq + 1)) + 1.0
            for term, freq in df.items()
        }
        self._fitted = True

    def embed(self, text: str) -> SparseVec:
        """텍스트를 L2 정규화된 TF-IDF 희소 벡터로 변환."""
        if not self._fitted:
            raise RuntimeError("embed() 전에 fit()을 먼저 호출해야 한다.")
        tf = Counter(tokenize(text))
        if not tf:
            return {}
        max_tf = max(tf.values())
        vec: SparseVec = {}
        for term, freq in tf.items():
            idf = self._idf.get(term)
            if idf is None:
                continue  # 학습 코퍼스에 없던 미지의 단어는 무시
            vec[term] = (0.5 + 0.5 * freq / max_tf) * idf  # 증강 TF × IDF
        # L2 정규화
        norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
        return {t: w / norm for t, w in vec.items()}


def _stable_hash(token: str) -> int:
    """프로세스 간 재현 가능한 해시(파이썬 내장 hash()는 salt 때문에 부적합)."""
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")


class HashingEmbedder:
    """해싱 트릭(feature hashing) 기반 임베더 — TF-IDF와 다른 두 번째 provider.

    사전(vocabulary)을 만들지 않고 토큰을 고정 개수(n_buckets)의 버킷으로 해싱한다.
    - 장점: 미등록 단어(OOV)에 강하고 메모리가 고정적이다(sklearn HashingVectorizer,
      Vowpal Wabbit 등이 쓰는 실전 기법).
    - 충돌 편향을 줄이려 signed hashing(부호 해시)을 적용한다.
    - fit()에서 버킷별 IDF를 학습해 TF-IDF 를 해시공간에서 근사한다.

    TfidfEmbedder 와 동일한 EmbeddingProvider 인터페이스 → 파이프라인 무수정 교체.
    """

    def __init__(self, n_buckets: int = 4096) -> None:
        self.n_buckets = n_buckets
        self._idf: dict[int, float] = {}
        self._n_docs = 0
        self._fitted = False

    def _bucket(self, token: str) -> tuple[int, float]:
        h = _stable_hash(token)
        idx = h % self.n_buckets
        sign = 1.0 if (h >> 63) & 1 else -1.0   # 부호 해시(충돌 상쇄)
        return idx, sign

    def fit(self, corpus: list[str]) -> None:
        self._n_docs = len(corpus)
        df: Counter[int] = Counter()
        for text in corpus:
            seen = {self._bucket(t)[0] for t in tokenize(text)}
            for b in seen:
                df[b] += 1
        self._idf = {
            b: math.log((self._n_docs + 1) / (freq + 1)) + 1.0 for b, freq in df.items()
        }
        self._fitted = True

    def embed(self, text: str) -> SparseVec:
        if not self._fitted:
            raise RuntimeError("embed() 전에 fit()을 먼저 호출해야 한다.")
        # 부호 해시의 정석: 버킷마다 sign*count 를 '누적'한다 — 반대 부호로
        # 충돌한 토큰이 서로 상쇄되는 것이 signed hashing 의 존재 이유다.
        # 종전 구현은 |tf| 를 합산하고 부호는 '마지막 토큰'이 덮어써서, 같은
        # bag-of-words 인데 토큰 순서에 따라 벡터 성분의 부호가 뒤집혔다
        # (n_buckets=1 에서 "tok0 tok1" 과 "tok1 tok0" 의 코사인이 -1 — v8).
        signed: dict[int, float] = {}
        for tok in tokenize(text):
            idx, sign = self._bucket(tok)
            signed[idx] = signed.get(idx, 0.0) + sign
        vec: SparseVec = {}
        for idx, s in signed.items():
            if s == 0.0:  # 완전 상쇄된 버킷은 성분 없음
                continue
            idf = self._idf.get(idx, 1.0)
            vec[str(idx)] = math.copysign((1.0 + math.log(abs(s))) * idf, s)
        if not vec:
            return {}
        norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
        return {t: w / norm for t, w in vec.items()}


class VoyageEmbedder:
    """실제 상용 임베딩 API(Voyage AI) provider — 'pluggable' 의 실전 경로.

    VOYAGE_API_KEY 가 설정된 경우에만 사용 가능하다(없으면 명확히 실패).
    무거운 SDK 의존성 없이 표준 라이브러리(urllib)로 REST 호출한다.
    fit() 은 no-op(사전학습 모델이라 코퍼스 학습 불필요) — 인터페이스만 맞춘다.

    데모는 오프라인이 기본이므로 실행되진 않지만, 확장 지점이 '가설'이 아니라
    실제 동작 코드로 존재함을 보여준다.
    """

    def __init__(self, model: str = "voyage-3", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("VOYAGE_API_KEY", "").strip()
        self._fitted = True  # 사전학습 모델 → fit 불필요

    def fit(self, corpus: list[str]) -> None:  # noqa: D401 - 인터페이스 호환용 no-op
        return None

    def embed(self, text: str) -> SparseVec:
        if not self.api_key:
            raise RuntimeError("VoyageEmbedder 는 VOYAGE_API_KEY 가 필요하다.")
        import json
        import urllib.request

        req = urllib.request.Request(
            "https://api.voyageai.com/v1/embeddings",
            data=json.dumps({"input": [text], "model": self.model}).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        dense = payload["data"][0]["embedding"]
        # dense 벡터를 {index: value} 희소표현으로 담아 cosine() 과 호환
        return {str(i): float(v) for i, v in enumerate(dense)}


@functools.lru_cache(maxsize=4)
def _load_st_model(model_name: str, local_files_only: bool):
    """sentence-transformers 모델 로더 — 프로세스당 1회만 로드(지배 비용).

    lru_cache 싱글턴이라 evaluate 의 provider 비교·sweep 의 파이프라인 다중
    재구축·테스트 세션이 같은 모델 객체를 공유한다. lazy import 라 ml 미설치
    환경에선 ImportError 가 나고, auto 는 이를 폴백 신호로 쓴다.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, local_files_only=local_files_only)


class SentenceTransformerEmbedder:
    """로컬 밀집임베딩 provider (sentence-transformers) — 의미 매칭 담당.

    BM25/TF-IDF 는 어휘(정확 단어) 신호라 동의어·의역은 synonyms.py 사전에
    의존했다. 밀집임베딩은 그 갭(사전에 없는 의미 유사)을 벡터 공간에서 메운다.

    e5 계열은 비대칭 검색용 prefix 를 요구한다(query:/passage:). Protocol 은
    fit/embed 만 있어 질의·문서를 구분할 수 없으므로: fit() 에서 코퍼스를
    passage prefix 로 배치 인코딩해 {원문: 벡터} 캐시에 담고, embed() 는
    캐시 히트 = 색인 경로(passage 벡터 반환), 미스 = 질의 경로(query prefix
    인코딩)로 판정한다. vectorstore.index() 가 fit 직후 같은 텍스트로 embed
    를 재호출하는 기존 흐름에 무수정으로 맞아떨어진다. (질의가 청크 원문과
    완전 동일한 엣지는 청크에 '[제목 > 섹션]' prefix 가 붙어 실질 발생 불가)
    """

    def __init__(
        self,
        model_name: str | None = None,
        model=None,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name or config.EMBEDDING_MODEL
        # 테스트는 .encode() 만 가진 스텁을 model 로 주입해 torch 없이 검증한다.
        self._model = model if model is not None else _load_st_model(self.model_name, local_files_only)
        self._passage_cache: dict[str, SparseVec] = {}
        self._fitted = False
        # 무신호 계약용 바닥값 — 밀집 코사인은 무관 질의에도 0 이 안 나온다.
        # retriever._hybrid 가 getattr 로 읽는다(희소 임베더는 속성 없음 = 0.0).
        self.no_signal_floor = config.DENSE_SIGNAL_FLOOR

    @staticmethod
    def _wrap(vec) -> SparseVec:
        # dense 벡터를 {index: value} 로 담아 cosine()/vectorstore 와 호환(Voyage 선례)
        return {str(i): float(v) for i, v in enumerate(vec)}

    def fit(self, corpus: list[str]) -> None:
        """코퍼스를 passage 로 배치 인코딩해 캐시 — 색인 경로의 사실상 임베딩 캐시."""
        vecs = self._model.encode(
            [f"passage: {t}" for t in corpus],
            normalize_embeddings=True,
            batch_size=32,
        )
        self._passage_cache = {t: self._wrap(v) for t, v in zip(corpus, vecs)}
        self._fitted = True

    def embed(self, text: str) -> SparseVec:
        if not self._fitted:
            raise RuntimeError("embed() 전에 fit()을 먼저 호출해야 한다.")
        cached = self._passage_cache.get(text)
        if cached is not None:
            return cached  # 색인 경로: fit 에서 만든 passage 벡터
        vec = self._model.encode(f"query: {text}", normalize_embeddings=True)
        return self._wrap(vec)


def cosine(a: SparseVec, b: SparseVec) -> float:
    """두 희소 벡터의 코사인 유사도. (이미 L2 정규화됐다면 내적과 동일)"""
    if not a or not b:
        return 0.0
    # 더 작은 쪽을 순회
    if len(a) > len(b):
        a, b = b, a
    dot = sum(w * b.get(t, 0.0) for t, w in a.items())
    na = math.sqrt(sum(w * w for w in a.values())) or 1.0
    nb = math.sqrt(sum(w * w for w in b.values())) or 1.0
    return dot / (na * nb)


def get_embedder(kind: str = "auto") -> EmbeddingProvider:
    """임베더 팩토리. EMBEDDER_KIND 환경변수/설정으로 provider 를 교체한다.

    - auto(기본): 밀집임베딩을 '이미 받아둔 것만'(local_files_only) 시도하고,
      미설치/모델 미캐시면 TF-IDF 로 폴백한다 — 기동이 수백 MB 다운로드로
      블록되거나 오프라인 환경에서 죽는 일이 없다. 폴백은 warning 1줄로 남긴다.
    - dense: 명시 요구 → 폴백 없이 실패를 전파한다(다운로드는 허용).
      '명시 설정은 시끄럽게 실패' — voyage 의 API 키 계약과 같은 원칙.
    """
    if kind == "tfidf":
        return TfidfEmbedder()
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "voyage":
        return VoyageEmbedder()
    if kind == "dense":
        # 받아둔 모델이 있으면 허브 확인 없이 바로 쓴다(기동 지연의 주범이 네트워크
        # 왕복이다). 없을 때만 다운로드를 허용 — '명시 설정은 시끄럽게 실패' 원칙은
        # 그대로(다운로드도 실패하면 예외 전파).
        try:
            return SentenceTransformerEmbedder(local_files_only=True)
        except ImportError:
            raise
        except Exception:
            return SentenceTransformerEmbedder()
    if kind == "auto":
        try:
            return SentenceTransformerEmbedder(local_files_only=True)
        except Exception as e:  # ImportError(미설치)·OSError(모델 미캐시) 등
            logger.warning(
                "밀집임베딩 사용 불가(%s: %s) — TF-IDF 로 폴백. 활성화: "
                "pip install -r requirements-ml.txt && python scripts/download_embedding_model.py",
                type(e).__name__, e,
            )
            return TfidfEmbedder()
    raise ValueError(f"알 수 없는 임베더: {kind}")
