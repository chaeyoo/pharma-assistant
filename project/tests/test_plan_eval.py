"""계획 능력 평가(eval/plan_eval.py)의 CI 핀 — 라우팅 계약을 실패로 강제한다.

eval 스크립트 자체는 측정·출력만 하고 exit 0 이므로, 스크립트만 CI 에 두면
'핀이라 부르지만 아무것도 고정하지 않는' 상태가 된다(verify_eval 과 같은
원칙). 오프라인 라우터는 결정론이라 세 지표 모두 1.0 핀이 가능하다 —
여기가 깨지면 새 발견이 아니라 라우팅(_route_intent·인자 승격) 회귀다.
"""
from __future__ import annotations

import json

import pytest

from eval.plan_eval import TOOL_UNIVERSE, _load_items, evaluate


def test_plan_dataset_integrity():
    """평가셋 자체의 계약 — 지표가 재기 전에 라벨이 유효해야 한다."""
    items = _load_items()
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids)), "id 중복"
    for item in items:
        exp = item["expect"]
        assert item.get("note"), f"{item['id']}: 라벨 근거(note) 누락"
        assert exp["tools"], f"{item['id']}: 기대 도구 비어 있음"
        unknown = set(exp["tools"]) - TOOL_UNIVERSE
        assert not unknown, f"{item['id']}: 미지의 도구 라벨 {unknown}"
        forbidden = set(exp.get("forbidden", []))
        assert forbidden <= TOOL_UNIVERSE, f"{item['id']}: 미지의 forbidden {forbidden - TOOL_UNIVERSE}"
        assert not (forbidden & set(exp["tools"])), f"{item['id']}: tools∩forbidden 교차"
        # 인자 검사 라벨은 기대 도구에만 붙는다(호출 안 되는 도구의 인자 검사는 항상 실패)
        assert set(exp.get("args_contains", {})) <= set(exp["tools"]), (
            f"{item['id']}: args_contains 가 기대 도구 밖 도구를 가리킴"
        )


def test_plan_dataset_is_json_clean():
    """파일이 표준 json 으로 재직렬화 가능해야 한다(수작업 편집 보호)."""
    from src import config

    raw = (config.BASE_DIR / "eval" / "plan_dataset.json").read_text(encoding="utf-8")
    data = json.loads(raw)
    assert "_comment" in data and "items" in data


@pytest.fixture(scope="module")
def res() -> dict:
    # conftest 의 세션 MCP 서버(autouse)가 config.MCP_SERVER_URL 을 이미 주입했다
    return evaluate()


# 표본 수 하한 — 항목 삭제·라벨 축소로 지표가 '조용히 수축'하는 실패 차단.
# 현재 실측(n=18, param_n=15)에서 보수적 여유를 둔 값 — 늘어나는 방향만 정상이다.
_MIN_N = {"n": 15}
_MIN_PARAM_N = 10


def test_plan_eval_sample_floors(res):
    assert res["n"] >= _MIN_N["n"], f"n={res['n']} < {_MIN_N['n']} — 평가셋이 조용히 수축했다"
    assert res["counts"]["param_n"] >= _MIN_PARAM_N, (
        f"param_n={res['counts']['param_n']} < {_MIN_PARAM_N} — 인자 검사 라벨이 조용히 수축했다"
    )


def test_plan_eval_pins(res):
    """오프라인 결정론 라우터의 계약 핀 — 비율이 아니라 카운트 동등으로 고정한다."""
    c = res["counts"]
    assert c["recall_ok"] == res["n"], f"ToolRecall 회귀: {res['failures']}"
    assert c["precision_ok"] == res["n"], f"ToolPrecision 회귀: {res['failures']}"
    assert c["param_ok"] == c["param_n"], f"ParamAccuracy 회귀: {res['failures']}"
