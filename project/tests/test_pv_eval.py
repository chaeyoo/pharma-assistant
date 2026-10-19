"""PV 워크플로 평가(eval/pv_eval.py)의 회귀 가드.

규칙·사전을 고치다 조용히 깨지는 것을 막는다:
  - 중대성·기한·인과성은 규정의 닫힌 목록 대조 + 날짜 연산 → 정확도 1.0 이 정상.
    1.0 미만이면 규칙 회귀(또는 라벨 수정 필요)다.
  - 코딩 정밀도 1.0: 잘못 붙은 PT는 시그널 집계를 오염시킨다(무관용).
  - 코딩 재현율은 의도적으로 1.0 미만(소사전 롱테일 문항 포함) — 하한만 지킨다.
  - 2계층(후보)·3계층(미코딩 감지)이 확정 코딩의 갭을 메워, 이상사례 '존재 감지'와
    보고요건 판정은 1.0 을 지켜야 한다(코딩 실패가 reportable 오판으로 연쇄 금지).
"""
from __future__ import annotations

import json

from src import config
from eval.pv_eval import evaluate


def test_pv_dataset_integrity():
    items = json.loads(
        (config.BASE_DIR / "eval" / "pv_dataset.json").read_text(encoding="utf-8")
    )["items"]
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids)), "케이스 id 중복"
    for it in items:
        exp = it["expect"]
        assert isinstance(exp["serious"], bool)
        assert exp["deadline_days"] in (None, 0, 15)
        assert exp["causality"] in ("Certain", "Probable", "Possible", "Unlikely", "Unassessable")
        assert isinstance(exp["pts"], list)
        assert isinstance(exp["reportable"], bool)


def test_pv_eval_regression_thresholds():
    res = evaluate()
    assert res["SeriousnessAcc"] == 1.0, res["failures"]
    assert res["DeadlineAcc"] == 1.0, res["failures"]
    assert res["CausalityAcc"] == 1.0, res["failures"]
    assert res["CodingPrecision"] == 1.0, "코딩 오탐은 집계 오염 — 무관용"
    # 확정 재현율 하한: 심층 롱테일 2문항(PV-021·022) 추가로 0.792 가 기대값.
    # 이 하한이 깨지면 '검수된 사전'의 회귀다(후보 티어로는 메워지지 않아야 한다).
    assert res["CodingRecall"] >= 0.75, "사전 회귀로 확정 재현율이 과도하게 떨어짐"
    # 2계층: 확정이 놓친 정답이 사람 확정 큐에 올라와야 하고, 후보 노이즈는 무관용
    assert res["CandidateRecall"] >= res["CodingRecall"], "후보 티어가 갭을 못 메움"
    assert res["CandidateRecall"] >= 0.9
    assert res["CandidatePrecision"] == 1.0, "후보 노이즈는 검수 비용 — 무관용 유지"
    # 3계층까지 합친 '존재 감지'와 보고요건 판정은 1.0 — 코딩 실패의 연쇄 차단이 목적
    assert res["AEDetectionRecall"] == 1.0, res["failures"]
    assert res["ReportableAcc"] == 1.0, res["failures"]
    assert res["MissingCountAcc"] == 1.0
