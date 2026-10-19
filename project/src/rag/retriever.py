"""하이브리드 리트리버 + 리랭커 (RAG '최적화'의 핵심).

2단계 검색으로 정밀도를 끌어올린다.
  1단계 (많이 회수):  하이브리드 검색 = 벡터(밀집임베딩 코사인, TF-IDF 폴백) + 키워드(BM25)
                     → 의미 유사(동의어)와 정확 용어(고유명사/코드) 모두 커버.
  2단계 (정밀 재정렬): 리랭커가 질의-청크 관련도를 다시 점수화해 상위 N개만 남김.
                     → Bi-Encoder(빠름) 로 넓게, Cross-Encoder 감각(정밀) 로 좁게.

리랭커는 오프라인에서 (질의 토큰 커버리지 + 정확 구문 매칭 + 섹션제목 가중)
으로 근사한다. 실무에선 이 자리에 Cross-Encoder 리랭커나 LLM 리랭커를 끼운다.
"""
from __future__ import annotations

import datetime as _dt
import math
import re
from collections import Counter

from .chunker import Chunk
from .embedder import cosine
from .synonyms import expand_query
from .textutil import tokenize
from .vectorstore import InMemoryVectorStore, Scored


def _parse_date(s: str) -> _dt.date | None:
    """ISO 날짜 파싱 — 실패·공백은 None. 호출부가 fail-closed 로 처리한다."""
    if not s:
        return None
    try:
        return _dt.date.fromisoformat(s)
    except ValueError:
        return None


class BM25Index:
    """BM25 키워드 검색 인덱스 (Okapi BM25)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs_tokens: list[list[str]] = []
        self.doc_len: list[int] = []
        self.avgdl = 0.0
        self.idf: dict[str, float] = {}
        self.tf: list[Counter[str]] = []

    def index(self, texts: list[str]) -> None:
        self.docs_tokens = [tokenize(t) for t in texts]
        self.tf = [Counter(toks) for toks in self.docs_tokens]
        self.doc_len = [len(toks) for toks in self.docs_tokens]
        n = len(texts)
        self.avgdl = (sum(self.doc_len) / n) if n else 0.0
        df: Counter[str] = Counter()
        for toks in self.docs_tokens:
            for term in set(toks):
                df[term] += 1
        self.idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def scores(self, query: str) -> list[float]:
        q_terms = tokenize(query)
        out: list[float] = []
        for i, tf in enumerate(self.tf):
            dl = self.doc_len[i] or 1
            s = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                idf = self.idf.get(term, 0.0)
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += idf * (freq * (self.k1 + 1)) / (denom or 1)
            out.append(s)
        return out


# ---- 섹션 타입 감지 (리랭커 v3: 질의 의도로 게이트되는 구조 prior) ----
# 규제문서에는 "X와의 차이/구분(주의)" 같은 '대조 섹션'이 반복적으로 등장한다
# (이 코퍼스에만 5곳: 화장품↔의약품 표시기재, DMF↔완제, 실태조사↔데이터완전성,
# 희귀↔일반허가, 의약외품↔의약품). 대조 섹션은 다른 도메인의 어휘를 통째로
# 인용하므로 어휘 기반 신호(coverage)로는 'X를 다루는 문서'와 구분이 불가능
# — 섹션 '제목'의 대조 구문("~와의/과의 + 차이/구분/관계/비교")이 유일하게
# 남는 구조 신호다. 질의 자체가 비교를 물으면(gate) 페널티를 끈다.
_CONTRAST_SECTION_RE = re.compile(r"[와과]의\s*.{0,8}?(차이|구분|관계|비교)")
_CONTRAST_QUERY_MARKERS = ("차이", "다르", "다른", "비교", "구분", "관계", "vs")

# '목적/개요/총칙' 서두 섹션은 문서의 모든 주제 어휘를 요약적으로 담고 있어
# coverage 신호가 과대평가되기 쉽다(운영 질문의 답은 대개 본문 조항에 있다).
# 질의가 정의/취지를 물을 때는(gate) 서두가 곧 정답이므로 감쇠를 끈다.
_PREAMBLE_SECTION_RE = re.compile(r"(목적|개요|총칙)")
_PREAMBLE_QUERY_MARKERS = ("무엇", "뭐", "목적", "개요", "취지", "정의", "어떤 활동", "어떤 문서")


def _is_contrast_section(section: str) -> bool:
    return bool(_CONTRAST_SECTION_RE.search(section))


def _is_preamble_section(section: str) -> bool:
    return bool(_PREAMBLE_SECTION_RE.search(section))


def _minmax(values: list[float]) -> list[float]:
    """0~1 정규화 (하이브리드 결합 전 스케일 정렬)."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


class HybridRetriever:
    """벡터 + BM25 하이브리드 1차 검색 후 리랭킹."""

    def __init__(
        self,
        store: InMemoryVectorStore,
        alpha: float = 0.5,
        rerank_weight: float = 0.9,
        idf_power: float = 0.5,
        contrast_penalty: float = 0.3,
        preamble_penalty: float = 0.055,
    ) -> None:
        self.store = store
        self.alpha = alpha              # 벡터 가중(1-alpha 는 BM25 가중)
        self.rerank_weight = rerank_weight  # 리랭커 신호 vs 1차 점수 prior 결합 비율
        self.idf_power = idf_power      # 리랭커 토큰 가중 = idf^p (0=균등, 1=IDF 그대로)
        # 섹션 타입 prior (질의 의도 게이트, 0이면 비활성 — sweep ablation 용)
        self.contrast_penalty = contrast_penalty
        self.preamble_penalty = preamble_penalty
        self.bm25 = BM25Index()
        # 리랭커용 사전 계산 캐시 (질의마다 청크를 재토크나이징하면 지연이 4배로 는다)
        self._chunk_tf: list[Counter[str]] = []
        self._sec_tokens: list[set[str]] = []
        self._title_w: list[dict[str, float]] = []   # 제목 토큰별 가중(질의 무관 → 사전 계산)
        self._title_total: list[float] = []
        self._chunk_ix: dict[int, int] = {}
        self._chunk_eff: list[_dt.date | None] = []      # 청크별 파싱된 시행일
        self._doc_eff: dict[str, _dt.date | None] = {}   # doc_id → 시행일(as_of 구간 판정)
        self._max_idf = 1.0

    def index(self, chunks: list[Chunk]) -> None:
        self.store.index(chunks)
        self.bm25.index([c.text for c in chunks])
        self._chunk_tf = list(self.bm25.tf)  # BM25 인덱스와 동일 토크나이즈 재사용
        self._max_idf = max(self.bm25.idf.values()) if self.bm25.idf else 1.0
        # 섹션 신호는 '섹션 제목만' 쓴다 — 문서 제목은 title 신호가 따로 담당
        # (섞으면 같은 문서의 모든 청크가 제목 토큰을 공유해 섹션 간 변별이 죽는다)
        self._sec_tokens = [set(tokenize(c.section)) for c in chunks]
        self._sec_contrast = [_is_contrast_section(c.section) for c in chunks]
        self._sec_preamble = [_is_preamble_section(c.section) for c in chunks]
        self._title_w = [
            {t: self._token_weight(t) for t in set(tokenize(c.title))} for c in chunks
        ]
        self._title_total = [sum(w.values()) for w in self._title_w]
        self._chunk_ix = {id(c): i for i, c in enumerate(chunks)}
        # 시행일은 인덱싱 시 1회 파싱해 둔다(질의마다 청크 수만큼 재파싱하지 않는다
        # — 리랭커 캐시와 같은 이유). 해석 불가능한 날짜는 None 으로 보존해
        # 시점(as_of) 조회에서 fail-closed 로 처리한다. doc 단위 맵은 폐지본의
        # '당시 현행' 구간 [시행일, 후속본 시행일) 판정에서 후속본 조회용.
        self._chunk_eff = [_parse_date(c.effective_date) for c in chunks]
        self._doc_eff = {c.doc_id: _parse_date(c.effective_date) for c in chunks}

    def _candidate_indices(self, as_of: str, include_superseded: bool) -> list[int]:
        """버전 인지 필터를 통과한 청크 인덱스만 반환(세 검색 모드 공통 후보군).

        - **기본 경로(as_of 미지정)는 as_of=오늘과 동치다(v9)** — 종전에는
          무 as_of 경로가 status 만 보고 시행일을 보지 않아, 미래 시행일의
          active 문서가 '현행'으로 반환되고, 그 문서로 대체될 폐지본(후속본이
          아직 시행 전이라 **오늘의 실제 현행**)은 폐지 필터에 걸려 은폐됐다.
          v8 이 as_of 경로에서 봉합한 '두 필터의 교집합이 공집합' 조합 결함과
          동형이 무 as_of 경로에 남아 있던 것 — 경로를 하나의 구간 판정으로
          통일해 코드 갈래 자체를 줄인다(같은 결함이 갈래마다 재발하지 않도록).
        - as_of(기준일) 기준 "그 시점에 시행 중이던 버전"을 반환한다:
            · 그 시점에 아직 시행 전인 문서는 제외하고,
            · **당시 시행 중이던 폐지본은 포함한다** — 폐지본이라도 시행일 ≤ as_of
              < 후속본 시행일 구간이면 그때의 현행이다. '폐지본 기본 제외'를
              as_of 에도 그대로 적용하면 개정된 적 있는 규정은 과거 시점 조회에서
              신·구판이 모두 걸러져 **아무 버전도 안 나온다** — "시점 조회를
              지원한다"는 주장이 정작 개정 이력이 있는(=시점 조회가 필요한)
              규정에서만 무너지는 논리적 비약이었다.
        - fail-closed: 시행일을 해석할 수 없는 청크(또는 후속본 시행일 미상)는
          제외한다 — '언제부터 유효한지 모르는 문서'를 특정 시점의 현행으로
          제시하지 않는다. 크래시도, 조용한 통과도 아니다: 데이터 결함 자체는
          preflight(코퍼스 무결성)가 파일명과 함께 보고하고, 여기서는 그 결함이
          검색 전체를 죽이거나(예외 전파) 오답에 섞이는(무필터 통과) 두 나쁜
          방향을 모두 차단한다. 경로 통일의 트레이드오프로 이 fail-closed 가
          **기본 검색에도 적용된다**(종전에는 무 as_of 경로에서 시행일 결함
          문서도 검색됐다) — preflight 가 시행일 유효성을 기동 전에 보증하므로
          실질 위험은 없고, 결함이 배포를 뚫었을 때 '유효기간 미상 문서'가
          현행으로 나가는 것보다 안전한 방향이다.
        - 잠재 상호작용(현 코퍼스 미발생): 후속본이 아직 시행 전인 폐지본은
          include_superseded=False 여도 '오늘의 현행'으로 기본 검색에 반환된다.
          답변 검증 게이트(superseded_cited)는 이력 조회가 아닐 때 폐지본 인용을
          경고하므로, 그런 코퍼스에서는 '오늘 현행인 폐지본' 인용에 경고가 붙을
          수 있다 — 현 코퍼스는 유일한 폐지본 REG-013 의 후속본 REG-005 가 이미
          시행 중이라 이 경로가 발화하지 않으며(회귀는 tests/test_versioning.py
          의 합성 청크 테스트로 고정), 발화하는 코퍼스를 들일 때는 게이트의
          허용 규약(src/agent·src/verify 관할)을 함께 조정해야 한다.

        as_of 형식 오류는 여기서 즉시 ValueError 다 — 사용자 경로는 MCP 도구
        경계(search_regulations)가 검증해 명시적 에러로 답하고, 직접 호출자
        (eval·테스트)의 형식 오류는 조용히 무시하는 것보다 시끄럽게 깨지는 것이 옳다.
        """
        cutoff = _dt.date.fromisoformat(as_of) if as_of else _dt.date.today()
        out: list[int] = []
        for i, c in enumerate(self.store.chunks):
            eff = self._chunk_eff[i]
            if eff is None or eff > cutoff:
                continue  # 시행일 미상(fail-closed) 또는 그 시점에 시행 전
            if c.status == "superseded" and not include_superseded:
                succ = self._doc_eff.get(c.superseded_by)
                if succ is None or succ <= cutoff:
                    continue  # 후속본 시행일 미상이거나 이미 대체된 시점
            out.append(i)
        return out

    # ---- 벡터 단독 검색(버전 필터 공유; eval 비교·폴백용) ----
    def vector_search(
        self, query: str, top_k: int, as_of: str = "", include_superseded: bool = False
    ) -> list[Scored]:
        qv = self.store.embedder.embed(query)
        idxs = self._candidate_indices(as_of, include_superseded)
        scored = [
            Scored(chunk=self.store.chunks[i], score=cosine(qv, self.store.vectors[i]))
            for i in idxs
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    # ---- 1단계: 하이브리드 검색 ----
    def _hybrid(
        self, query: str, top_k: int, as_of: str = "", include_superseded: bool = False
    ) -> list[Scored]:
        # 빈/기호-only 질의는 provider 무관 무신호 — 임베딩 호출 전에 자른다
        # (밀집 임베더는 빈 질의도 벡터를 만들어 무관 문서에 코사인 점수를 준다).
        if not tokenize(query):
            return []
        qv = self.store.embedder.embed(query)
        bm_scores_all = self.bm25.scores(query)
        idxs = self._candidate_indices(as_of, include_superseded)
        # 정규화는 '후보군 안에서' 수행해야 스케일이 왜곡되지 않는다.
        vec_scores = [cosine(qv, self.store.vectors[i]) for i in idxs]
        bm_scores = [bm_scores_all[i] for i in idxs]
        # 무신호 계약(v9): 전 후보의 벡터·BM25 '원점수'가 모두 무신호면 빈 결과.
        # 안정 정렬은 전부 0점인 후보를 코퍼스 첫 문서 순서 그대로 내보내
        # 빈 질의·완전 무관 질의가 임의 문서를 score 0.0 의 '검색 결과'로 달고
        # 나가는 가짜 순위가 됐다(에이전트는 SCORE_FLOOR 로 회피하지만, 직접
        # 호출자·eval 은 그 순위를 그대로 소비한다). 관련도 신호가 전혀 없으면
        # 결과가 없다고 답하는 것이 맞다 — 판정은 정규화 전 원점수 기준이다
        # (_minmax 는 '전원 동점'을 전원 0으로 만들어 유신호와 무신호를 구분 못 한다).
        # v13: '무신호' 판정을 provider 별 바닥값으로 일반화 — 희소(TF-IDF/해싱)는
        # 겹치는 단어가 없으면 코사인이 정확히 0이라 바닥값 0.0 그대로지만, 밀집
        # 임베딩은 무관 질의에도 0.3~0.6대 코사인이 나와 '0 이하 = 무신호'가 성립
        # 하지 않는다. 밀집 임베더는 no_signal_floor 속성(config.DENSE_SIGNAL_FLOOR,
        # eval 보정)으로 자기 바닥값을 선언하고, 속성이 없는 희소 임베더는 0.0.
        vec_floor = getattr(self.store.embedder, "no_signal_floor", 0.0)
        if max(vec_scores, default=0.0) <= vec_floor and max(bm_scores, default=0.0) <= 0.0:
            return []
        vn, bn = _minmax(vec_scores), _minmax(bm_scores)
        combined = [
            Scored(chunk=self.store.chunks[i], score=self.alpha * vn[j] + (1 - self.alpha) * bn[j])
            for j, i in enumerate(idxs)
        ]
        combined.sort(key=lambda s: s.score, reverse=True)
        return combined[:top_k]

    # ---- 2단계: 리랭킹 ----
    # 리랭커 컴포넌트 가중치. eval 스윕에서 이 데이터셋 지표는 title 0~0.15,
    # idf_power 0~1에 둔감했다 — 잔여 하드네거티브(대조 섹션) 실패는 이 4신호의
    # 어떤 재가중으로도 안 뒤집혔고, 결국 '신호 추가'(섹션 타입 prior, v3)로
    # 풀렸다. 그래서 가중은 '수치가 좋아서'가 아니라 신호의 역할 분리
    # (본문/구문/섹션/제목) 기준으로 고정 — 코퍼스가 커지면 sweep 으로 재보정.
    RERANK_COMPONENT_WEIGHTS: dict[str, float] = {
        "coverage": 0.55,   # 본문: 질의 토큰 커버리지(idf^p 가중)
        "phrase": 0.20,     # 구문: 질의 원문 정확 매칭
        "section": 0.15,    # 섹션: 정답 '섹션' 선택 신호
        "title": 0.10,      # 제목: 문서 정합(BM25F 감각의 필드 신호)
    }

    def _token_weight(self, term: str) -> float:
        """리랭커용 토큰 가중 = idf^p (BM25 인덱스의 IDF 재사용).

        p=0 이면 균등(v1과 동일), p=1 이면 IDF 그대로. 매칭 밀도(tf 포화)
        신호와 결합한 초기 실험에서 p=1은 '의약품'처럼 흔하지만 문서 구분에
        필수인 토큰을 지나치게 죽여 하드네거티브를 악화시켰다(0.846→0.769).
        밀도 신호 제거 후에는 p에 둔감해졌지만, 큰 코퍼스에서 판별 토큰
        우대가 필요해질 것에 대비해 완만한 중간값 0.5를 기본으로 남겼다.
        코퍼스에 없는 토큰(OOV)은 가장 희귀한 수준으로 취급한다.
        """
        if not self.bm25.idf:
            return 1.0
        idf = self.bm25.idf.get(term) or self._max_idf
        return idf ** self.idf_power

    def _rerank_score(self, query: str, chunk: Chunk, aux_terms: set[str] | None = None) -> float:
        """질의-청크 관련도 재점수 (Cross-Encoder 근사).

        네 가지 신호의 가중합:
          (1) coverage — 청크 텍스트가 질의 토큰을 얼마나 덮는가(idf^p 가중).
              여기서 '청크 텍스트'는 컨텍스트 prefix("[제목 > 섹션]")를 포함한다
              (BM25 인덱스와 동일 토큰) — 즉 제목·섹션 토큰도 coverage 에
              기여하며, 이는 title/section 신호와의 부분 중복을 감수한
              **의도된 트레이드오프**다(v8 에서 prefix 제외 본문만으로 바꿔
              측정한 결과 Hit@1 1.000→0.969, HardNeg 1.000→0.929 로 후퇴 —
              현 가중·prior 가 prefix 포함 coverage 기준으로 보정되어 있고,
              소코퍼스에서는 제목 문맥이 coverage 에 실리는 편이 유리했다.
              재보정 없이 이 정의만 바꾸면 안 된다).
          (2) phrase   — 질의 원문(공백 정리·소문자)이 '청크 텍스트'에 통째로
              등장하는가. 여기서 '청크 텍스트'도 coverage 와 동일하게 컨텍스트
              prefix("[제목 > 섹션]")를 **포함한 대조**다 — 즉 질의가 문서
              제목·섹션 헤더 문자열과 일치해도 구문 신호가 켜지며, 이는
              coverage 의 prefix 포함과 같은 **의도된 트레이드오프**다
              (제목형 질의가 해당 문서 청크에 정확 매칭 보너스를 받는 방향.
              현 가중·prior 와 개발셋 Hit@1 1.000 이 이 정의 기준으로 고정되어
              있으므로, '본문만' 대조로 바꾸려면 coverage 정의 변경과 마찬가지로
              재보정 없이는 안 된다).
          (3) section  — 질의 토큰이 '섹션 제목'에 있는가(같은 문서 안에서
              정답 섹션을 고르는 신호 — 문서 제목은 섞지 않는다)
          (4) title    — 문서 제목 정합: 제목 토큰 중 질의에 등장하는 '비율'.
              하드네거티브 문서(예: 화장품 규정 속 '의약품 표시기재와의 차이'
              섹션)는 본문이 질의 어휘를 많이 공유해 coverage 로는 못 거르므로,
              문서 제목('화장품 표시·광고')이 질의와 겹치지 않는 것을 반증
              신호로 쓴다 — coverage 의 제목 토큰 '가산'과 달리 이 신호는
              제목 쪽 기준 비율이라 반대 방향(불일치의 감점)을 담당한다.

        aux_terms: 질의확장으로 '추가된' 토큰 집합(선택). 원 질의 토큰의
        절반 가중으로만 반영한다. 완전 어휘 불일치 질의("부작용이 심각…"은
        원 질의 토큰이 규정 문서에 하나도 없다)에서 리랭커가 판별력을 잃는
        것을 막는 안전망 — rerank_weight=1.0(1차 prior 없음) ablation 에서
        aux 유무가 Hit@1 0.969 vs 0.906 차이를 만든다(32문항).
        """
        q_terms = tokenize(query)
        if not q_terms:
            return 0.0
        q_set = set(q_terms)

        ix = self._chunk_ix.get(id(chunk))
        if ix is None:  # 인덱스 밖 청크(단위 테스트 등) — 캐시 없이 즉석 계산
            c_set = set(tokenize(chunk.text))
            sec_terms = set(tokenize(chunk.section))
            title_w = {t: self._token_weight(t) for t in set(tokenize(chunk.title))}
            title_total = sum(title_w.values())
            is_contrast = _is_contrast_section(chunk.section)
            is_preamble = _is_preamble_section(chunk.section)
        else:
            c_set = set(self._chunk_tf[ix])
            sec_terms = self._sec_tokens[ix]
            title_w = self._title_w[ix]
            title_total = self._title_total[ix]
            is_contrast = self._sec_contrast[ix]
            is_preamble = self._sec_preamble[ix]

        # 토큰 가중치: 원 질의는 idf^p, 확장 토큰은 그 절반(보조 신호)
        weights = {t: self._token_weight(t) for t in q_set}
        for t in (aux_terms or set()) - q_set:
            weights[t] = 0.5 * self._token_weight(t)
        total_w = sum(weights.values()) or 1.0

        coverage = sum(w for t, w in weights.items() if t in c_set) / total_w
        phrase = 1.0 if query.strip() and query.strip().lower() in chunk.text.lower() else 0.0
        section_hit = sum(w for t, w in weights.items() if t in sec_terms) / total_w
        # 제목 정합은 '제목 쪽 기준' 비율(제목 토큰이 질의로 얼마나 설명되는가)
        title_match = (
            sum(w for t, w in title_w.items() if t in weights) / title_total
            if title_total
            else 0.0
        )

        w = self.RERANK_COMPONENT_WEIGHTS
        score = (
            w["coverage"] * coverage
            + w["phrase"] * phrase
            + w["section"] * section_hit
            + w["title"] * title_match
        )

        # 섹션 타입 prior (v3): 질의 의도로 게이트되는 구조 신호.
        #  - 대조 섹션("X와의 차이/구분")은 X 도메인 어휘를 통째로 인용해
        #    coverage 가 정답 문서를 이긴다(하드네거티브의 최빈 실패 형태).
        #    질의가 비교를 묻지 않으면 강하게 감점한다. 크기 0.3은 스윕에서
        #    '뒤집히는 최소값(≈0.24) 이상 & 게이트 문항 불변' 플랫 구간의 값.
        #  - 서두 섹션(목적/개요/총칙)은 문서 주제 어휘를 요약해 담고 있어
        #    운영 질문("어디에 표시?")에서 본문 조항을 이기는 과대평가가 난다.
        #    감쇠 0.055는 유효 밴드(0.04~0.07: 아래는 미교정, 위는 정답 서두
        #    문항 역전)의 중앙 — 근거는 eval/sweep.py 로 재현.
        low_q = query.lower()
        if is_contrast and self.contrast_penalty > 0:
            if not any(m in low_q for m in _CONTRAST_QUERY_MARKERS):
                score -= self.contrast_penalty
        if is_preamble and self.preamble_penalty > 0:
            if not any(m in low_q for m in _PREAMBLE_QUERY_MARKERS):
                score -= self.preamble_penalty
        return score

    def retrieve(
        self,
        query: str,
        top_k: int,
        rerank_n: int,
        as_of: str = "",
        include_superseded: bool = False,
        expand: bool = True,
        aux_in_rerank: bool = True,
    ) -> list[Scored]:
        """최종 검색: (질의 확장) → 하이브리드 top_k → 리랭킹 → 상위 rerank_n 반환.

        as_of / include_superseded 로 버전 인지 검색을 제어한다. as_of 미지정
        기본 검색은 as_of=오늘과 동치이고(구간 판정·fail-closed 의 트레이드오프는
        _candidate_indices 참고), 무신호 질의(전 후보 원점수 0)는 빈 결과다.

        질의 확장은 '1단계 회수'에 전 가중으로 적용하고("부작용"→"이상사례"
        같은 어휘 불일치를 메워 recall 확보), 2단계 리랭킹은 원 질의 토큰을
        기준으로 재점수하되 확장 토큰을 절반 가중의 보조 신호로만 쓴다.
        원 질의만으로 재점수하면 완전 어휘 불일치 질의에서 리랭커가 판별력을
        잃는다(rerank_weight=1.0 ablation: aux 유무 = Hit@1 0.969 vs 0.906).
        """
        q1 = expand_query(query) if expand else query
        first = self._hybrid(q1, top_k, as_of, include_superseded)
        if not first:
            return []
        # 확장으로 '추가된' 토큰만 보조 신호로 리랭커에 전달
        # (aux_in_rerank=False 는 eval/sweep.py 의 ablation 용 스위치)
        aux = (
            (set(tokenize(q1)) - set(tokenize(query)))
            if (q1 != query and aux_in_rerank)
            else None
        )
        # 1차 하이브리드 점수를 prior 로 블렌딩(순수 재정렬은 쉬운 질의를 오히려 떨어뜨림).
        # 실무 Cross-Encoder 리랭커도 first-stage 점수와 결합해 안정화하는 관행을 반영.
        first_scores = _minmax([s.score for s in first])
        reranked = [
            Scored(
                chunk=s.chunk,
                score=self.rerank_weight * self._rerank_score(query, s.chunk, aux_terms=aux)
                + (1 - self.rerank_weight) * first_scores[i],
            )
            for i, s in enumerate(first)
        ]
        reranked.sort(key=lambda s: s.score, reverse=True)
        return reranked[:rerank_n]
