"""평가셋 후보 생성기(eval/generate_eval_items.py)의 결정론 계약 테스트.

생성 품질(확률)은 사람 검토의 몫이고, 여기서는 실행 없이 참이어야 하는 계약만
핀 고정한다: 파싱·검증(실존 doc·중복 탈락), retrieval_verified 는 필터가 아니라
라벨, malformed 의 무크래시 흡수, 그리고 **no-auto-merge** — 어떤 실행 경로도
기존 평가셋 파일을 건드리지 않는다(리뷰 게이트의 구조적 강제).
"""
from __future__ import annotations

import json
import sys
import types

import pytest

from src import config
from eval.generate_eval_items import (
    STRATEGIES,
    _parse_candidates,
    _validate,
    existing_questions,
    generate,
    main,
)


def _stub_anthropic(monkeypatch, reply_by_call: list[str]) -> None:
    """전략별 create 호출에 순서대로 응답하는 동기 anthropic 스텁."""
    state = {"i": 0}

    class _Messages:
        def create(self, **kwargs):
            text = reply_by_call[min(state["i"], len(reply_by_call) - 1)]
            state["i"] += 1
            return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=text)])

    fake = types.ModuleType("anthropic")
    fake.Anthropic = lambda api_key=None: types.SimpleNamespace(messages=_Messages())
    monkeypatch.setitem(sys.modules, "anthropic", fake)


def _reply(cands: list[dict]) -> str:
    return json.dumps({"candidates": cands}, ensure_ascii=False)


def test_parse_absorbs_fences_and_rejects_malformed():
    assert _parse_candidates('```json\n{"candidates": [{"question": "q"}]}\n```') == [{"question": "q"}]
    assert _parse_candidates("후보를 만들 수 없습니다") is None
    assert _parse_candidates('{"candidates": "not-a-list"}') is None


def test_validate_rejects_nonexistent_doc_and_duplicates():
    doc_ids = {"REG-001", "REG-005"}
    existing = {"신약품목허가심사는며칠이내에처리되나요"}
    ok, reason = _validate(
        {"question": "존재하지 않는 문서를 겨냥한 질문입니다", "target_doc_id": "REG-999"},
        "paraphrase", doc_ids, existing,
    )
    assert ok is None and "REG-999" in reason
    ok, reason = _validate(  # 구두점·공백 차이는 중복이다
        {"question": "신약 품목허가 심사는 며칠 이내에 처리되나요??"},
        "paraphrase", doc_ids, existing,
    )
    assert ok is None and "중복" in reason
    ok, reason = _validate(
        {"question": "허가 갱신 주기는 어떻게 되나요", "target_doc_id": "REG-001",
         "rationale": "재표현"},
        "paraphrase", doc_ids, existing,
    )
    assert ok is not None and ok["target_doc_id"] == "REG-001" and ok["strategy"] == "paraphrase"


def test_generate_labels_without_filtering(monkeypatch):
    """retrieval_verified 는 참/거짓 그대로 라벨링만 — false 후보도 살아남는다."""
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-key")
    _stub_anthropic(monkeypatch, [_reply([
        {"question": "신약 품목허가 심사 기간은 얼마나 되나요",
         "target_doc_id": "REG-001", "rationale": "재표현"},
        {"question": "완전히 무관한 주제의 질문입니다 오늘 날씨 알려주세요",
         "target_doc_id": "REG-005", "rationale": "검색이 못 찾는 후보"},
    ])])
    res = generate(n_per_strategy=2)
    assert len(res["items"]) >= 2
    labels = {i["retrieval_verified"] for i in res["items"][:2]}
    assert True in labels and False in labels, "false 후보가 자동 폐기됐다(라벨≠필터 위반)"
    for item in res["items"]:
        assert item["status"] == "pending_review"
        assert item["strategy"] in STRATEGIES
        assert item["model"] and item["generated_at"]


def test_generate_absorbs_malformed_strategy(monkeypatch):
    """한 전략의 malformed 출력은 그 전략만 건너뛴다 — 크래시·부분 파싱 금지."""
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-key")
    _stub_anthropic(monkeypatch, ["판정 불가"])
    res = generate(n_per_strategy=1)
    assert res["items"] == []
    assert len(res["rejected"]) == len(STRATEGIES)
    assert all("malformed" in r for r in res["rejected"])


def test_no_auto_merge_into_eval_datasets(monkeypatch, tmp_path):
    """리뷰 게이트의 구조 핀 — main 실행 전후로 기존 평가셋 파일이 불변이어야 한다."""
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-key")
    _stub_anthropic(monkeypatch, [_reply([
        {"question": "허가 갱신은 몇 년마다 해야 하나요", "target_doc_id": "REG-001",
         "rationale": "재표현"},
    ])])
    guarded = ["qa_dataset.json", "holdout_dataset.json", "abstention_dataset.json",
               "plan_dataset.json", "robustness_dataset.json"]
    before = {n: (config.BASE_DIR / "eval" / n).read_bytes() for n in guarded}
    out = tmp_path / "candidates.json"
    main(argv=["--n", "1"], out_path=out)
    for n in guarded:
        assert (config.BASE_DIR / "eval" / n).read_bytes() == before[n], f"{n} 이 변조됐다"
    written = json.loads(out.read_text(encoding="utf-8"))
    assert all(i["status"] == "pending_review" for i in written["items"])
    assert "병합하지 않는다" in written["_comment"]


def test_main_without_key_writes_nothing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    out = tmp_path / "candidates.json"
    main(argv=[], out_path=out)
    assert not out.exists()
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().out


def test_existing_questions_covers_all_datasets():
    """중복 기준선이 5개 평가셋을 전부 포괄해야 '이미 있는 문항의 재생성'이 걸러진다."""
    existing = existing_questions()
    assert "신약품목허가심사는며칠이내에처리되나요" in existing  # qa_dataset
    assert len(existing) >= 70  # qa32+holdout6+abstention8+plan18+robustness15
