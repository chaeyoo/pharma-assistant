"""PV(Pharmacovigilance·약물감시) 도메인 로직.

PV 케이스 처리 워크플로를 순수 로직 모듈로 구현한다:
  redactor(PII 비식별화) → triage(중대성 판정·보고기한 계산)
  → causality(WHO-UMC 인과성 제안) → coding(표준 용어 코딩)
  → report(최소보고요건 검증 + KAERS 보고서 초안 조립)
MCP 도구(src/mcp_server)가 이 로직을 감싸 에이전트에 노출한다.
"""
