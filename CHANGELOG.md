# Changelog

## 2026-11-22 · Astra 순위와 이미지 기반 해결지침

- 기본 실행 경로에 STEP/XCAF·부품별 OBJ 입력, 두 초기 뷰, 상태를 이어가는 전체 후보 순위 계약, 공식 DFS/RedMax 실행을 연결했다.
- 판정 직전 관측, 목표 중심 실패 이미지, 같은 채택 경로의 해결 추적, 다음 회차부터 사용하는 고정 인덱스와 실제 DINOv2 검색을 추가했다.
- 분해 영상, 부분·완료를 구분하는 export v1, 해시 검사 및 전달 ZIP을 추가했다.
- 공식 Beam 기준선과 새 고정 순위 실행, 연구실 배터리 한 부품, 진단 조립체의 회차 간 검색을 검증했다. 실제 Astra 생성 호출은 0회다. [전체 결과와 명령](reports/MASTER_DESIGN_DELIVERY_20261122.md)

## 2026-11-21 — Initial public research snapshot

- Published existing planner, worker, tools, tests and selected design/evidence files without changing their bytes.
- Added Korean entry documentation, known limitations, source inventory and offline verification details.
- Linked the official ASAP commit as a submodule instead of uploading its CAD, weights or build products.
- Preserved all source experiments; excluded credentials, private attachments, raw provider logs and the full experience library.
- Recorded the F-drive portability failures separately from passing offline tests.
