"""배포 전 점검(src/preflight) 테스트 — 게이트가 실제 결함을 잡는지.

preflight 는 '부팅은 통과하지만 운영 중에 오답으로 나타나는' 데이터·설정
결함을 기동 전에 차단하는 배포 게이트다. 여기서는 두 방향을 모두 고정한다:
(1) 리포의 실제 데이터·기본 설정은 통과해야 하고(오탐 감시 — run.sh/CI가
이 게이트 뒤에 있으므로 오탐은 곧 배포 불능), (2) 대표 결함(메타 누락·
doc_id 충돌·폐지 체인 단절·날짜 오타·모순 설정)은 잡혀야 한다.
"""
from __future__ import annotations

from pathlib import Path

from src import config, preflight


def test_repo_data_passes_preflight():
    """리포 상태 그대로는 전 그룹 통과 — 게이트 오탐은 곧 배포 불능이다."""
    report = preflight.run_preflight()
    assert all(not problems for problems in report.values()), report


def test_preflight_role_group_composition():
    """롤별 점검 그룹 분담 — 완전 분리 배포에서 각 컨테이너가 자기 책임만 진다.

    mcp 롤은 원격 연결 점검이 없어야 하고(자기 자신이 서버), api 롤은 파이프
    라인을 소유하지 않으므로 코퍼스·스모크가 없어야 한다. 그룹 구성만 고정한다
    — api 롤의 'MCP 연결' 실행 자체는 conftest 가 띄운 실서버로 검증한다."""
    mcp_groups = set(preflight.run_preflight(role="mcp"))
    assert "MCP 연결" not in mcp_groups
    assert {"코퍼스 무결성", "스모크(canary)"} <= mcp_groups

    api_report = preflight.run_preflight(role="api")
    assert "MCP 연결" in api_report
    assert not api_report["MCP 연결"], api_report["MCP 연결"]  # conftest 의 실서버에 실연결 성공
    assert {"코퍼스 무결성", "스모크(canary)"}.isdisjoint(api_report)


def _write_doc(dirpath: Path, name: str, meta: dict, body: str = "본문 내용") -> None:
    fm = "\n".join(f"{k}: {v}" for k, v in meta.items())
    (dirpath / name).write_text(f"---\n{fm}\n---\n\n{body}\n", encoding="utf-8")


def test_corpus_check_catches_meta_defects(tmp_path):
    _write_doc(tmp_path, "a.md", {"doc_id": "R-1", "title": "t"})  # version·effective_date 누락
    _write_doc(tmp_path, "b.md", {"doc_id": "R-2", "title": "t", "version": "1.0",
                                  "effective_date": "2025-13-01"})  # 존재하지 않는 달
    _write_doc(tmp_path, "c.md", {"doc_id": "R-2", "title": "t", "version": "1.1",
                                  "effective_date": "2025-01-01"})  # doc_id 중복
    text = "\n".join(preflight.check_corpus(tmp_path))
    assert "필수 필드 누락" in text
    assert "YYYY-MM-DD" in text
    assert "중복" in text


def test_corpus_check_catches_broken_supersede_chain(tmp_path):
    # 폐지본이 가리키는 후속 문서가 코퍼스에 없다 — 버전 인지 검색의 전제 붕괴
    _write_doc(tmp_path, "old.md", {"doc_id": "R-OLD", "title": "t", "version": "0.9",
                                    "effective_date": "2020-01-01", "status": "superseded",
                                    "superseded_by": "R-GONE"})
    problems = preflight.check_corpus(tmp_path)
    assert any("R-GONE" in p for p in problems)
    # superseded_by 자체가 없는 경우도 잡는다
    _write_doc(tmp_path, "old2.md", {"doc_id": "R-OLD2", "title": "t", "version": "0.9",
                                     "effective_date": "2020-01-01", "status": "superseded"})
    problems = preflight.check_corpus(tmp_path)
    assert any("superseded_by 가 없다" in p for p in problems)


def test_corpus_check_catches_chain_date_inversion(tmp_path):
    """폐지 체인의 시행일 역전 — as_of 시점 조회의 구간 판정([구판 시행일,
    후속본 시행일))이 빈 구간이 되어 시점 조회를 조용히 망가뜨리는 결함."""
    _write_doc(tmp_path, "old.md", {"doc_id": "R-OLD", "title": "t", "version": "0.9",
                                    "effective_date": "2025-01-01", "status": "superseded",
                                    "superseded_by": "R-NEW"})
    _write_doc(tmp_path, "new.md", {"doc_id": "R-NEW", "title": "t", "version": "1.0",
                                    "effective_date": "2024-01-01"})  # 구판보다 이른 시행일
    problems = preflight.check_corpus(tmp_path)
    assert any("늦지 않다" in p for p in problems)


def test_tasks_check_catches_bad_schema(tmp_path):
    f = tmp_path / "ra_tasks.json"
    f.write_text(
        '{"deadlines": [{"item": "x", "due_date": "다음주", "type": "t", "owner": "o", "status": "s"},'
        ' {"item": "y"}], "checklists": {"a": []}}',
        encoding="utf-8",
    )
    text = "\n".join(preflight.check_tasks(f))
    assert "YYYY-MM-DD" in text          # 날짜 형식
    assert "필수 필드 누락" in text        # deadlines[1]
    assert "비어 있다" in text             # 빈 체크리스트


def test_tasks_check_catches_stale_deadlines(tmp_path):
    """시한부 샘플 데이터의 부패 — 마감일이 전부 과거가 되면 '임박한 마감' 데모가
    전항목 연체 목록으로 조용히 죽는다(스키마는 유효하므로 형식 검사만으로는
    영원히 통과). 전건 과거일 때만 결함으로 센다 — 과거 1건은 '지남/긴급' 연출용
    의도된 데이터라 오탐이 되면 안 된다(리포 데이터 통과는 별도 테스트가 고정)."""
    import datetime as dt
    import json as _json

    past = (dt.date.today() - dt.timedelta(days=30)).isoformat()
    future = (dt.date.today() + dt.timedelta(days=30)).isoformat()
    row = {"item": "x", "type": "t", "owner": "o", "status": "s"}
    f = tmp_path / "ra_tasks.json"

    f.write_text(_json.dumps({
        "deadlines": [{**row, "due_date": past}, {**row, "due_date": past}],
        "checklists": {"a": ["1"]},
    }), encoding="utf-8")
    assert any("모두 과거" in p for p in preflight.check_tasks(f))

    f.write_text(_json.dumps({
        "deadlines": [{**row, "due_date": past}, {**row, "due_date": future}],
        "checklists": {"a": ["1"]},
    }), encoding="utf-8")
    assert not any("모두 과거" in p for p in preflight.check_tasks(f))


def test_mcp_reachable_waits_out_rolling_deploy_version_skew():
    """무중단 롤링 배포 중에는 구버전 MCP 인스턴스가 새 인스턴스 준비 전까지
    정상 응답한다 — '연결 성공 + 기대 도구 없음'을 즉시 실패로 처리하면
    새 도구를 추가한 커밋마다 API preflight 가 과도기에 기동을 거부한다
    (search_web 추가 배포에서 실측). 도구 누락도 deadline 안에서 재시도해
    스큐가 해소되면 통과해야 한다."""
    old = {"search_regulations", "get_ra_deadlines", "get_submission_checklist",
           "assess_adverse_event", "draft_ae_report", "list_regulation_documents"}
    seq = [old, old, old | {"search_web"}]   # 구버전 2회 응답 후 새 버전 전환
    calls = {"n": 0}

    def probe():
        tools = seq[min(calls["n"], len(seq) - 1)]
        calls["n"] += 1
        return tools

    problems = preflight.check_mcp_reachable(timeout_s=5.0, probe=probe, poll_interval_s=0.01)
    assert problems == [] and calls["n"] >= 3


def test_mcp_reachable_reports_missing_tools_after_deadline():
    """deadline 이 지나도록 도구가 누락이면 — 버전 스큐 안내가 담긴 명시적
    실패로 보고한다(기다림이 침묵 통과로 바뀌면 안 된다: 진짜 계약 위반은
    여전히 기동 차단)."""
    old = {"search_regulations"}
    problems = preflight.check_mcp_reachable(timeout_s=0.05, probe=lambda: old, poll_interval_s=0.01)
    assert problems and "도구 누락" in problems[0]
    assert "search_web" in problems[0] and "버전 스큐" in problems[0]


def test_mcp_reachable_reports_connection_failure_after_deadline():
    """연결 자체가 끝내 안 되면 — 종전과 같은 '연결 실패 + 주소 출처 진단'
    경로가 유지된다(누락 재시도 추가가 이 경로를 밀어내지 않는다)."""
    def probe():
        raise ConnectionError("boom")

    problems = preflight.check_mcp_reachable(timeout_s=0.05, probe=probe, poll_interval_s=0.01)
    assert problems and "연결 실패" in problems[0] and "ConnectionError" in problems[0]


def test_gate_self_tests_cover_every_warning_axis():
    """자가 테스트의 축 대칭성 — 런타임 게이트의 모든 경고 축에 대해 '심은
    오류' 케이스가 존재해야 한다. 처음에는 수치 존재 축만 자가 테스트했는데,
    그러면 방향·역할·버전 축은 고장난 채 배포될 수 있다 — 새 축을 추가하면
    이 테스트가 자가 테스트 추가를 강제한다.

    축 우주의 앵커는 검증기 자신(verifier.WARNING_AXES)이다(v9) — 종전에는
    계기판(GateStats._AXES)의 수동 복제 튜플이 앵커라, 검증기에만 축을
    추가하는 부분 변경에서 이 메타 검사가 통과한 채 자가 테스트 없는 축이
    배포될 수 있었다('구조로 강제한다'는 주장의 숨은 전제가 미검사)."""
    from src.verify.verifier import LABEL_AXES, WARNING_AXES

    planted_axes = {t["axis"] for t in preflight._GATE_SELF_TESTS if not t["expect_ok"]}
    assert planted_axes == set(WARNING_AXES), (
        f"자가 테스트 미커버 축: {set(WARNING_AXES) - planted_axes}"
    )
    # 정상(clean) 케이스도 최소 1건 이상 — 오탐 방향의 자가 테스트
    assert any(t["expect_ok"] for t in preflight._GATE_SELF_TESTS)
    # 라벨 축(ok 를 유지하는 등급 라벨)도 자가 테스트가 있어야 한다 — 라벨은
    # 죽어도 응답이 계속 ok 라 고장 신호 자체가 없다.
    labeled = {t["label"] for t in preflight._GATE_SELF_TESTS if t.get("label")}
    assert set(LABEL_AXES) <= labeled, f"라벨 축 자가 테스트 누락: {set(LABEL_AXES) - labeled}"


def test_axis_registry_matches_verifier_summary():
    """축 정본(WARNING_AXES∪LABEL_AXES)과 검증기 summary() 키의 일치 — 검증기에
    새 축(summary 키)을 추가하면 정본 등록·자가 테스트 추가가 연쇄로 강제된다.
    계기판의 축 목록도 정본에서 파생됐는지 함께 고정한다(수동 복제 금지)."""
    from src.observability import GateStats
    from src.verify.verifier import LABEL_AXES, WARNING_AXES, verify_answer

    summary_keys = set(verify_answer("답", ["근거"]).summary().keys())
    non_axis_keys = {"ok", "checked", "checks"}
    assert set(WARNING_AXES) | set(LABEL_AXES) == summary_keys - non_axis_keys, (
        "검증기 summary 키와 축 정본이 어긋난다 — 새 축을 추가했으면 WARNING_AXES/"
        f"LABEL_AXES 에 등록하고 preflight 자가 테스트를 추가할 것: {summary_keys - non_axis_keys}"
    )
    assert tuple(GateStats._AXES) == tuple(WARNING_AXES)


def test_gate_self_test_asserts_direction_from_case_label():
    """케이스 간섭 자가 테스트가 축 발화만이 아니라 from_case 라벨(case_origin)
    의 생존까지 단언하는지 — 라벨만 죽으면 케이스 재서술에 '컴플라이언스 오류'
    단정 문구가 붙는 오탐 방향으로 조용히 퇴행한다(v9)."""
    case_tests = [
        t for t in preflight._GATE_SELF_TESTS
        if t.get("user_facts") and not t["expect_ok"] and t.get("axis") == "direction_conflicts"
    ]
    assert case_tests and all(t.get("label") == "case_origin" for t in case_tests)


def test_corpus_check_catches_future_effective_active_without_predecessor(tmp_path):
    """미래 시행일의 현행(active) 문서가 '시행 중인 구판' 없이 단독 존재하면
    시행일까지 그 주제가 오늘 기준 검색에서 통째로 사라진다 — 스키마가 전부
    유효한 채 주제 하나가 조용히 죽는 형태(형식/값 타당성은 다른 층). (v9)"""
    import datetime as dt

    future = (dt.date.today() + dt.timedelta(days=90)).isoformat()
    past = (dt.date.today() - dt.timedelta(days=400)).isoformat()

    # (a) 단독 미래 시행 active → 결함
    _write_doc(tmp_path, "new.md", {"doc_id": "R-NEW", "title": "t", "version": "2.0",
                                    "effective_date": future})
    problems = preflight.check_corpus(tmp_path)
    assert any("시행 중인 구판이 없다" in p for p in problems)

    # (b) 시행 중인 구판이 함께 있으면 정상(개정 예고본 + 현행 구판의 정당한 공존)
    _write_doc(tmp_path, "old.md", {"doc_id": "R-OLD", "title": "t", "version": "1.0",
                                    "effective_date": past, "status": "superseded",
                                    "superseded_by": "R-NEW"})
    assert not any("시행 중인 구판" in p for p in preflight.check_corpus(tmp_path))


def test_smoke_catches_broken_gate(monkeypatch):
    """게이트가 모든 답변을 통과시키는 고장(항상 ok) 상태로는 배포되지 않는다."""
    from src.verify import verifier

    monkeypatch.setattr(
        verifier, "verify_answer", lambda *a, **k: verifier.VerificationResult()
    )
    problems = preflight.smoke_checks()
    assert any("검증 게이트 자가 테스트 실패" in p for p in problems)


def test_smoke_catches_broken_redactor(monkeypatch):
    """안전장치 자가 테스트의 대칭성: 검증 게이트뿐 아니라 PII 마스킹도 고장난
    채로는 배포되지 않는다 — 심은 개인정보가 살아남으면 smoke 가 잡아야 한다."""
    from src.pv import redactor

    monkeypatch.setattr(
        redactor, "redact",
        lambda text: redactor.RedactionReport(text=text, counts={}),  # 아무것도 안 지우는 고장
    )
    problems = preflight.smoke_checks()
    assert any("PII 마스킹 자가 테스트 실패" in p for p in problems)


def test_smoke_catches_broken_history_block_masking(monkeypatch):
    """이력 마스킹의 표기 변형 자가 테스트 — 블록 리스트 표기(content=[{type:
    text}])의 마스킹이 죽은 채로는 배포되지 않는다. 문자열 표기 자가 테스트만
    있으면 블록 경로의 고장은 신호 없이 통과한다(표기 변형 커버리지의 대칭)."""
    from src.agent import agent as agent_mod

    monkeypatch.setattr(agent_mod, "_redact_history", lambda history: history)  # 원문 통과 고장
    problems = preflight.smoke_checks()
    assert any("블록 표기" in p for p in problems)


def test_config_check_catches_unauthenticated_public_mcp(monkeypatch):
    """공개 배포 인증 가드 — MCP_REQUIRE_AUTH=1 인데 토큰이 비면 기동 거부.
    공개 web 서비스에서 토큰이 실수로 지워져도 무인증 공개 상태로 조용히
    돌아가지 않게 하는 fail-closed 가드다."""
    monkeypatch.setattr(config, "MCP_REQUIRE_AUTH", True)
    monkeypatch.setattr(config, "MCP_AUTH_TOKEN", "")
    assert any("MCP_AUTH_TOKEN" in p for p in preflight.check_config())
    monkeypatch.setattr(config, "MCP_AUTH_TOKEN", "sekrit")
    assert not any("MCP_AUTH_TOKEN" in p for p in preflight.check_config())


def test_config_check_catches_contradictions(monkeypatch):
    """각 값은 유효해도 조합이 모순인 경우 — 조용한 품질 붕괴의 형태."""
    monkeypatch.setattr(config, "RERANK_TOP_N", 99)  # top_k(8)보다 큼
    problems = preflight.check_config()
    assert any("RERANK_TOP_N" in p for p in problems)
    monkeypatch.setattr(config, "RERANK_TOP_N", 3)
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 700)  # chunk_size(500)보다 큼
    problems = preflight.check_config()
    assert any("CHUNK_OVERLAP" in p for p in problems)


def test_config_check_embedder_kind_values(monkeypatch):
    """EMBEDDER_KIND 유효값에 auto/dense 포함 — 오타는 기동 전에 잡는다."""
    monkeypatch.setattr(config, "EMBEDDER_KIND", "nope")
    assert any("EMBEDDER_KIND" in p for p in preflight.check_config())
    monkeypatch.setattr(config, "EMBEDDER_KIND", "auto")
    assert not any("EMBEDDER_KIND" in p for p in preflight.check_config())


def test_config_check_agent_backend_values(monkeypatch):
    """AGENT_BACKEND 유효값 검증 — direct|pydantic_ai|langgraph 외는 기동 전에 잡는다."""
    monkeypatch.setattr(config, "AGENT_BACKEND", "nope")
    assert any("AGENT_BACKEND" in p for p in preflight.check_config())
    monkeypatch.setattr(config, "AGENT_BACKEND", "langgraph")  # 설치돼 있으므로 통과
    assert not any("AGENT_BACKEND" in p for p in preflight.check_config())


def test_config_check_dense_requires_loadable_model(monkeypatch):
    """dense 명시(폴백 없음)인데 모델 로드 불가면 기동 전에 실패해야 한다 —
    기동 후 수백 MB 다운로드 블록/런타임 예외보다 preflight 실패가 낫다."""
    from src.rag import embedder as em

    monkeypatch.setattr(config, "EMBEDDER_KIND", "dense")

    def _boom(model_name, local_files_only):
        raise ImportError("sentence_transformers 미설치")

    monkeypatch.setattr(em, "_load_st_model", _boom)
    assert any("dense" in p for p in preflight.check_config())


def test_corpus_check_accepts_valid_two_tier_chain(tmp_path):
    """정당한 2단 폐지 체인(구판→중간판(폐지)→현행)은 결함이 아니다. (v8)

    superseded_by 규약은 '직전 후속본'이다 — 리트리버의 as_of 구간 판정
    [시행일, 후속본 시행일)이 이를 전제하는데, 종전 preflight 는 '후속본이
    폐지본이면 결함'이라 정반대 규약을 요구했다(다단 이력을 넣는 순간 어느
    쪽으로 써도 한쪽이 깨지는 잠복 상충)."""
    _write_doc(tmp_path, "v1.md", {"doc_id": "R-V1", "title": "t", "version": "1.0",
                                   "effective_date": "2020-01-01", "status": "superseded",
                                   "superseded_by": "R-V2"})
    _write_doc(tmp_path, "v2.md", {"doc_id": "R-V2", "title": "t", "version": "2.0",
                                   "effective_date": "2022-01-01", "status": "superseded",
                                   "superseded_by": "R-V3"})
    _write_doc(tmp_path, "v3.md", {"doc_id": "R-V3", "title": "t", "version": "3.0",
                                   "effective_date": "2024-01-01"})
    assert preflight.check_corpus(tmp_path) == []


def test_corpus_check_catches_chain_cycle_and_dangling_tail(tmp_path):
    """순환 체인과, 중간에서 끊기는 체인(폐지 종점)은 결함으로 잡는다. (v8)"""
    _write_doc(tmp_path, "a.md", {"doc_id": "R-A", "title": "t", "version": "1",
                                  "effective_date": "2020-01-01", "status": "superseded",
                                  "superseded_by": "R-B"})
    _write_doc(tmp_path, "b.md", {"doc_id": "R-B", "title": "t", "version": "2",
                                  "effective_date": "2021-01-01", "status": "superseded",
                                  "superseded_by": "R-A"})  # 순환
    problems = preflight.check_corpus(tmp_path)
    assert any("순환" in p for p in problems)


def test_corpus_check_catches_mid_chain_date_inversion(tmp_path):
    """다단 체인 중간 링크의 시행일 역전도 잡는다. (v8)"""
    _write_doc(tmp_path, "v1.md", {"doc_id": "R-V1", "title": "t", "version": "1",
                                   "effective_date": "2020-01-01", "status": "superseded",
                                   "superseded_by": "R-V2"})
    _write_doc(tmp_path, "v2.md", {"doc_id": "R-V2", "title": "t", "version": "2",
                                   "effective_date": "2023-01-01", "status": "superseded",
                                   "superseded_by": "R-V3"})
    _write_doc(tmp_path, "v3.md", {"doc_id": "R-V3", "title": "t", "version": "3",
                                   "effective_date": "2022-01-01"})  # 중간 링크 역전
    problems = preflight.check_corpus(tmp_path)
    assert any("늦지 않다" in p for p in problems)
