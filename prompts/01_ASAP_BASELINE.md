# M1 — 공식 baseline 실제 실행
`docs/08_ASAP_INTEGRATION_PLAN.md`에 따라 actual README/환경/entrypoint를 읽어줘.
첫 데이터는 beam_assembly/original, 방법은 dfs+heur-out, 목표는 원본 그대로야.
physics smoke, native sequence planning, setup/stats/tree, 가능하면 plan/path export를 실제 확인해.
stdout/stderr/exit/commit/asset hash/설정을 run별로 보존해.
False/timeout을 collision으로 만들지 말고 종료 코드와 native plan success를 구분해.
실패를 mock로 대체하지 마. 사용자 허용 없이 기존 환경을 지우거나 전역 시스템을 바꾸지 마.
결과를 IMPLEMENTATION_STATUS.md와 run 보고서에 기록해.
