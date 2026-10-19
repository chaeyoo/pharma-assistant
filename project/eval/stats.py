"""평가 지표의 통계적 정직성 — 소표본 비율 지표에 신뢰구간을 붙인다.

이 데모의 평가셋은 32문항(QA)·22케이스(PV)로 작다. 그 표본에서 나온
"Hit@1 1.000"은 '완벽한 검색기'가 아니라 **'이 표본에서 실패가 관측되지
않았다'**는 뜻이고, 둘의 차이를 숫자로 말하는 도구가 신뢰구간이다.
(n=32, 성공 32/32 의 95% 구간은 [0.893, 1.000] — 진짜 성능이 0.9여도
이 표본에서 1.000이 나올 수 있다.)

Wilson score interval 을 쓰는 이유:
  - 이항 비율 전용 폐쇄형 공식이라 **결정론적**이다(부트스트랩과 달리
    난수가 없어 CI 로그가 매 실행 동일 — 회귀 비교가 가능).
  - 소표본·극단 비율(0.0/1.0)에서 정규근사(Wald)처럼 구간이 0폭으로
    붕괴하지 않는다 — 정확히 우리가 있는 영역이다.

용법: 평가 스크립트들이 핵심 비율 지표 출력에 [lo, hi]를 병기한다.
개선을 주장할 때의 규율: 두 구간이 크게 겹치면 '개선'이 아니라
'구분 불가'로 읽는다.
"""
from __future__ import annotations

import math


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """이항 비율의 Wilson score 신뢰구간(기본 95%).

    Args:
        successes: 성공 횟수 (0 <= successes <= n).
        n: 시행 횟수. 0이면 정보가 없으므로 [0, 1] 전 구간을 반환.
        z: 정규분위수 (95% → 1.96).
    """
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, round(center - half, 3)), min(1.0, round(center + half, 3)))


def fmt_ci(successes: int, n: int) -> str:
    """'0.938 [0.799, 0.982]' 형태의 출력 문자열."""
    lo, hi = wilson_interval(successes, n)
    p = successes / n if n else 0.0
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}]"
