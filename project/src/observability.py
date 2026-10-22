"""경량 관측성(observability) — 스텝별 지연·에러를 구조화 로그로 남긴다.

엔터프라이즈 에이전트에서 '무슨 도구를 몇 ms에, 성공/실패로 호출했는가'는
디버깅·SLA·비용 관리의 기본이다. 실무에선 이 자리에 LangSmith/OpenTelemetry를
끼우지만, 여기서는 의존성 없이 동일한 개념(span·trace)을 순수 파이썬으로 구현한다.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .verify.verifier import WARNING_AXES

logger = logging.getLogger("rapv_assistant")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# 학습용 흐름 로그 (flow log)
# ---------------------------------------------------------------------------
# UI 에서 질문 하나가 들어왔을 때 '어떤 파일의 어떤 함수를 어떤 순서로 거치는지'
# 사람이 따라 읽도록 남기는 로그다. span 로그(JSON, 기계 집계용)와 별개로
# [흐름 NN] 순번 · 파일명 · 함수명 · 이 단계의 목적 · 주요 파라미터를 한 줄에 담는다.
# FLOW_LOG=0 환경변수로 끈다.
# 주의: 원문 개인정보를 싣지 않는다 — 메시지 텍스트는 마스킹(redact) '이후'
# 계층에서만 로그에 올리고, 그 전 계층(API 진입)은 길이만 기록한다.
FLOW_LOG_ENABLED = os.environ.get("FLOW_LOG", "1") not in ("0", "false", "off")
_flow_n = 0


def flow_reset() -> None:
    """요청 단위 순번 초기화 — API 진입점(POST /chat)에서 부른다.

    데모는 단일 사용자 전제라 전역 카운터로 충분하다. 동시 요청이 겹치면
    순번이 섞일 수 있다 — 실무라면 요청별 correlation id 로 대체할 자리.
    """
    global _flow_n
    _flow_n = 0


def _flow_val(v: Any) -> str:
    """파라미터 값을 로그 한 줄에 들어가게 접는다(개행 제거·100자 절단)."""
    s = v if isinstance(v, str) else repr(v)
    s = s.replace("\n", " ")
    return s if len(s) <= 100 else s[:100] + "…"


def flow(func: str, purpose: str, **params: Any) -> None:
    """흐름 로그 한 줄: [흐름 NN] 파일:라인 :: 함수 — 목적 | k=v, ...

    파일 경로와 라인 번호는 호출 지점의 스택 프레임에서 자동으로 얻는다 —
    수동으로 적으면 코드가 수정되어 줄이 밀릴 때마다 로그가 거짓말을 하게
    된다(로그의 위치 정보가 틀리면 없는 것보다 나쁘다). 라인 번호는 flow()
    를 부른 그 줄이므로, 에디터에서 해당 파일:라인으로 점프해 앞뒤 코드를
    읽으면 그 단계가 코드 어디인지 바로 찾아진다.
    """
    global _flow_n
    if not FLOW_LOG_ENABLED:
        return
    _flow_n += 1
    fr = sys._getframe(1)  # 호출자 프레임 — 파일 경로·라인 번호의 출처
    path = fr.f_code.co_filename
    i = path.rfind("/src/")  # 절대경로를 프로젝트 상대경로(src/…)로 접는다
    loc = f"{path[i + 1:] if i != -1 else os.path.basename(path)}:{fr.f_lineno}"
    # next= 는 '왜 다음 호출이 일어나는가'의 인과 설명 — 데이터 값과 달리
    # 잘리면 의미가 깨지므로 절단 없이 맨 뒤에 화살표로 구분해 붙인다.
    nxt = params.pop("next", None)
    tail = (
        " | " + ", ".join(f"{k}={_flow_val(v)}" for k, v in params.items())
        if params
        else ""
    )
    if nxt:
        tail += f"  ▶ 다음: {nxt}"
    logger.info(f"[흐름 {_flow_n:02d}] {loc} :: {func} — {purpose}{tail}")


@dataclass
class Span:
    """단일 작업 구간(도구 호출·LLM 스텝·검색 등)."""
    name: str
    kind: str            # tool | llm | pipeline | agent
    duration_ms: float
    ok: bool = True
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "ms": round(self.duration_ms, 2),
            "ok": self.ok,
            **self.detail,
        }


@dataclass
class Trace:
    """한 요청의 span 모음(에이전트 실행 트레이스)."""
    spans: list[Span] = field(default_factory=list)

    def add(self, span: Span) -> None:
        self.spans.append(span)

    @property
    def total_ms(self) -> float:
        """요청 총 지연(wall-clock).

        스텝 span(tool·llm)은 최상위 agent span '안에서' 실행되므로 전부
        합산하면 같은 시간이 두 번 세어진다 — 계기판이 실제의 ~2배 지연을
        보고하면 SLA 판단·병목 진단이 함께 틀린다(관측이 틀리면 관측이 없는
        것보다 나쁘다). 그래서 최상위 agent span 이 있으면 그 wall-clock 을
        쓰고, 없으면(부분 트레이스) 합산으로 폴백한다.
        """
        for s in self.spans:
            if s.kind == "agent":
                return round(s.duration_ms, 2)
        return round(sum(s.duration_ms for s in self.spans), 2)

    def to_list(self) -> list[dict]:
        return [s.to_dict() for s in self.spans]


@dataclass
class GateStats:
    """검증 게이트 운영 계기판 — 프로세스 수명 동안의 경고율을 상시 집계한다.

    검증 계층의 운영 리스크는 오탐 그 자체보다 **경고율의 추이**다: 경고가
    잦아지면 담당자가 경고를 무시하기 시작하고(alert fatigue) 그 순간 계층
    전체가 죽는다. 배치 후 이 신호를 볼 수단이 없으면 죽는 순간을 모른 채
    지나간다 — 그래서 게이트 통과/경고를 축별로 세어 /health 에 노출한다
    (운영자가 배치 다음 날 아침에 확인하는 계기판). 집계는 응답 단위다:
    한 응답에 미확인 수치가 3개여도 unsupported 축 +1 — 계기판이 재는 것은
    '경고가 붙은 응답의 비율'이지 클레임 개수가 아니다.

    검증 결과는 응답 단위 감사 로그(JSON)로도 남는다 — 규제 도메인에서
    "그 답변이 그때 검증을 통과했는가"는 사후 감사의 질문이기 때문이다.
    답변 원문이 아니라 판정 요약만 남긴다(로그에 클레임 수치·판정만, PII 없음).

    분모의 규율 — 전체 경고율(warn_rate)과 함께 **클레임이 있던 응답만의
    경고율(warn_rate_checked)** 을 병기한다. "warn_rate 상승 = 품질 회귀 또는
    오탐 증가"라는 해석에는 트래픽 믹스가 일정하다는 숨은 전제가 있다:
    회피·무클레임 응답(checked=0)은 자명하게 통과하므로, 범위 밖 질문의
    비중이 늘면 품질 변화가 없어도 warn_rate 가 내려간다(좋아지는 착시) —
    분모가 섞인 지표는 추이 해석이 목적일수록 위험하다. warn_rate_checked 는
    '검증할 것이 있던 응답'만 분모로 잡아 믹스 변화에 흔들리지 않는다.
    """
    responses: int = 0
    warned: int = 0
    checked_responses: int = 0   # 수치·날짜 클레임이 1개 이상 있던 응답
    warned_checked: int = 0      # 그중 경고가 붙은 응답
    case_labeled: int = 0        # 케이스 서술 유래(from_case) 라벨이 붙은 응답
    web_labeled: int = 0         # 인터넷 검색 유래(from_web) 라벨이 붙은 응답
    by_axis: dict[str, int] = field(default_factory=dict)

    # 축 목록의 정본은 검증기(verifier.WARNING_AXES)다 — 여기서 수동 복제
    # 튜플을 유지하면, 검증기에 새 축을 추가하고 이 파일을 안 건드리는 부분
    # 변경에서 자가 테스트 커버리지 메타 검사(축 우주가 이 튜플이었다)가
    # 통과한 채 새 축이 계기판·자가 테스트 양쪽에서 조용히 빠진다(v9).
    _AXES = WARNING_AXES

    def record(self, summary: dict) -> None:
        self.responses += 1
        warned = not summary.get("ok", True)
        if warned:
            self.warned += 1
        if summary.get("checked"):
            self.checked_responses += 1
            if warned:
                self.warned_checked += 1
        # 경고가 아닌 '등급 라벨'도 추이를 본다 — 케이스 서술 유래 지지가 갑자기
        # 늘면 답변이 규정 근거 대신 사용자 서술에 기대기 시작했다는 신호다
        # (경고율만 보면 이 이동은 보이지 않는다 — 라벨은 ok=True 이므로).
        # 집계 기준은 from_case 라벨이 붙은 '모든' 체크다(v9) — summary 의
        # case_origin 은 존재·방향 축의 from_case 를 모두 포함하도록 검증기
        # 쪽에서 정의된다(docstring "from_case 라벨이 붙은 응답"과 집계의 일치).
        if summary.get("case_origin"):
            self.case_labeled += 1
        # 웹 유래 라벨의 추이 — 웹 지지 응답이 갑자기 늘면 답변이 사내 규정
        # 근거 대신 인터넷 결과에 기대기 시작했다는 신호다(명시와 분리 원칙의
        # 침식 — case_labeled 와 같은 원리로, 경고율만 보면 이 이동은 보이지 않는다).
        if summary.get("web_origin"):
            self.web_labeled += 1
        for axis in self._AXES:
            if summary.get(axis):
                self.by_axis[axis] = self.by_axis.get(axis, 0) + 1

    def snapshot(self) -> dict:
        return {
            "responses": self.responses,
            "warned": self.warned,
            "warn_rate": round(self.warned / self.responses, 4) if self.responses else 0.0,
            "checked_responses": self.checked_responses,
            "warn_rate_checked": (
                round(self.warned_checked / self.checked_responses, 4)
                if self.checked_responses else 0.0
            ),
            "case_labeled": self.case_labeled,
            "web_labeled": self.web_labeled,
            "by_axis": dict(self.by_axis),
        }

    def reset(self) -> None:
        self.responses = 0
        self.warned = 0
        self.checked_responses = 0
        self.warned_checked = 0
        self.case_labeled = 0
        self.web_labeled = 0
        self.by_axis = {}


gate_stats = GateStats()


def record_verification(summary: dict) -> None:
    """검증 게이트 결과 1건을 계기판에 집계하고 감사 로그를 남긴다."""
    gate_stats.record(summary)
    logger.info(
        "verification "
        + json.dumps(
            {
                "ok": summary.get("ok"),
                "checked": summary.get("checked"),
                "unsupported": summary.get("unsupported"),
                "direction_conflicts": summary.get("direction_conflicts"),
                "role_conflicts": summary.get("role_conflicts"),
                "question_origin": summary.get("question_origin"),
                # case_origin 은 경고가 아니라 등급 라벨(지지 근거가 사용자
                # 케이스 서술뿐인 클레임) — 감사에서 '규정 근거'와 구분해 읽는다.
                "case_origin": summary.get("case_origin"),
                # web_origin 도 등급 라벨 — 지지 근거가 인터넷 검색 결과뿐인
                # 클레임(사내 규정 근거 아님 — 감사에서 별도 계층으로 읽는다).
                "web_origin": summary.get("web_origin"),
                "superseded_cited": summary.get("superseded_cited"),
            },
            ensure_ascii=False,
        )
    )


# ---------------------------------------------------------------------------
# 요청 단위 진행 이벤트 (UI 실시간 표시용)
# ---------------------------------------------------------------------------
# span 은 '끝난 뒤' trace 에 쌓이므로 UI 는 답변이 완성될 때까지 아무것도 못 본다.
# 콜드스타트(검색 서버 연결 수십 초)·LLM 스텝처럼 긴 구간을 진행 중에 보여 주려고,
# timed() 의 시작·종료를 요청별 큐로도 내보낸다. 큐는 ContextVar 라 동시 요청이
# 섞이지 않고(flow 로그의 전역 카운터와 달리), 싱크가 없으면(테스트·CLI·/chat 동기
# 경로) 완전한 no-op 이다. 이벤트는 span 과 같은 데이터(이름·kind·ms·마스킹 이후의
# args)만 싣는다 — trace 의 PII 규율이 그대로 적용된다.
_progress: contextvars.ContextVar[asyncio.Queue | None] = contextvars.ContextVar(
    "rapv_progress", default=None
)


def progress_emit(event: dict) -> None:
    """현재 요청의 진행 큐에 이벤트를 넣는다. 싱크가 없으면 아무것도 하지 않는다."""
    q = _progress.get()
    if q is not None:
        q.put_nowait(event)


@contextmanager
def progress_scope(queue: asyncio.Queue):
    """이 블록(과 블록 안에서 만든 asyncio task)의 timed() 이벤트를 queue 로 보낸다."""
    token = _progress.set(queue)
    try:
        yield
    finally:
        _progress.reset(token)


@contextmanager
def timed(trace: Trace, name: str, kind: str, detail: dict | None = None):
    """with 블록의 실행시간을 측정해 trace에 span으로 기록하고 구조화 로그를 남긴다.

    블록에서 예외가 나도 span(ok=False)을 남긴 뒤 예외를 다시 던진다.
    진행 싱크가 있으면 시작·종료 이벤트도 내보낸다(progress_scope 참고).
    """
    t0 = time.perf_counter()
    ok = True
    err_name = ""
    progress_emit({"type": "start", "name": name, "kind": kind, **(detail or {})})
    try:
        yield
    except Exception as e:  # noqa: BLE001 - 관측 목적상 모든 예외 기록 후 재전파
        ok = False
        err_name = type(e).__name__
        raise
    finally:
        dt = (time.perf_counter() - t0) * 1000.0
        d = dict(detail or {})
        if err_name:
            d["error"] = err_name
        trace.add(Span(name=name, kind=kind, duration_ms=dt, ok=ok, detail=d))
        progress_emit({"type": "end", "name": name, "kind": kind, "ms": round(dt, 2), "ok": ok, **d})
        logger.info(
            "span " + json.dumps({"name": name, "kind": kind, "ms": round(dt, 2), "ok": ok, **d}, ensure_ascii=False)
        )
