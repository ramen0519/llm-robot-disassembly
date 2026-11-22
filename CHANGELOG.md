# Changelog

## 2026-11-22 · 이전 지식 보관과 운영 사례 초기화

- 새 실행기의 기본 DB를 `.cases/astra_20261122_clean/index.sqlite`로 변경했다. DB, 동결 목록, 새 사례가 같은 수집분인지 확인한다. `--store`로 만든 새 저장소와 회차 간 학습은 계속 지원한다.
- 예전 전체 지식·선별본·등록 스킬·모의 시험 기억을 F에 별도로 보관했다. 321개 원본 해시 대조, 운영 사례 및 동일 과거 이미지 검색 결과 0건, 오프라인 29건 통과. [상태](IMPLEMENTATION_STATUS.md)

## 2026-11-22 · 운영 기억과 이전 실험 안내 정리

기본 README를 현재 실행 프로필 중심으로 정리하고 11월 21일 공개 당시 안내를 [이전 버전 문서](docs/history/README_20261121.md)로 보관했다. F 작업본 AGENTS의 조립 전달 파일 설명을 구현된 export v1 기준으로 갱신했다. 실행 코드·라이브러리·실험 원본 변경은 없다.

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
