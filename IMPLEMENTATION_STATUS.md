# 구현 상태

## 2026-11-22 · 기본 실행 경로와 과거 기억 분리 점검

기본 실행기 `--run`의 `caseplanner` 경로, `--run-dir` 호환 경로, 현재 SQLite DB와 이전 동결 JSON 원문을 대조했다. 새 프로필에서 예전 순서 스킬을 자동으로 읽거나 실행하는 연결은 없다.

| 읽기 전용으로 확인한 자료 | 실측 |
|---|---|
| 운영 `.cases/index.sqlite` | `research`, 사례 0건 |
| 마지막 커밋 고정 두 회차의 모의 시험 DB | `offline_contract`, 사례 2건. 지침 작성·인용은 모의 응답 |
| 이전 `migration_v6/training_runtime_library.json` | 704개 카드. 경험 642·교훈 52·스킬 10. 그중 범위가 정해진 등록 Python 스킬 원문 2개 확인 |

검사에 사용한 SQL은 `SELECT name FROM library_domain WHERE singleton=1`, `SELECT COUNT(*) FROM cases`다. 과거 JSON은 `cards[].kind`로 집계했다. WSL ASAP 환경에서 `python -B -m unittest discover -s tests -p test_astra_cases_v1.py -v`를 실행해 26건이 통과했다. 운영 저장소에서 모의 DB를 거부하는 경우와 동결 뒤 추가된 사례를 제외하는 경우가 포함된다. 현재·과거 안내의 로컬 링크 22개와 과거 안내 본문의 보존을 확인했다.

[현재 사용 안내](README.md), [당시 사용 안내](docs/history/README_20261121.md), AGENTS.md, 변경 기록, 결정 로그를 갱신했다. 이번 점검에서 실행 코드·DB·기존 실험 자료 변경, 유료 호출, 새 물리 실험은 0건이다.

## 2026-11-22 · T0–T5 연결 및 최종 검증

상태와 실행 명령, 모델·입력·해시, 주요 함수, 파일별 변경, 실패와 미실행 내역을 [최종 보고서](reports/MASTER_DESIGN_DELIVERY_20261122.md)에 정리했다. [실행 설명서](docs/ASTRA_CASES_V1.md)에서 같은 기본 실행기로 재현할 수 있다.

| 검증 | 실제 결과 |
|---|---|
| 오프라인 계약·이전 회귀 | 새 26건과 이전 39건 통과 |
| 공식 Beam 기준선 | 실제 DFS + heur-out, 부품 4/4 제거 |
| 새 실행기 + Beam | 고정 순위, 실제 RedMax, 4/4 제거·경로 24검사·안정성 4검사·추가 지지 0개 |
| 연구실 배터리 STEP | 13개 실제 부품·형상 5종, 뚜껑 12의 경로 통과, 잔여 부품 안정성 실패, 1시도에서 0개 제거 |
| 진단 조립체 | 실제 트레이 연결성 실패 → 기둥 2 제거 → 원래 트레이 1 제거. 모의 순위 사용. 3/3 제거 |
| 실제 이미지·기억 연결 | 실제 PNG와 DINOv2로 다음 회차 검색 통과. 모의 모델이 작성·인용한 지침은 별도 시험 DB에만 등록 |
| 실제 Astra | 계정에서 모델 ID 조회 성공. 순위·해결지침 생성·실제 모델에 의한 재사용은 미실행. 생성 0회 |
| 조립 전달 | 완주 Beam의 연속 상태·역순 후보·원본 좌표·영상·해시를 포함한 [전달 ZIP](evidence/beam_transfer_complete.zip). 조립 로봇 실행은 미실행 |

전달용 ZIP을 실제로 다시 열어 물리검사와 영상의 바이트 해시를 검사했다. [트레이 실패·해결 전달 파일](evidence/tray_case_transfer.zip)과 [배터리 부분 실행 전달 파일](evidence/battery_transfer_complete.zip)도 보존했다. 초기부터 부품이 2개인 입력에서 마지막 native 선택과 6회 경로·1회 안정성 검사까지 따로 통과했다.

초기 실패도 남겼다. STEP의 면 없는 그룹을 처리했고, 초기화에 365초가 걸린 Beam 시도와 통신 오류 모의 시험을 별도 보존했다. 통신 오류 시험에는 물리 전이나 추가 요청이 없었다. 공개 원본 및 과거 실험이 있는 작업본은 보존했다.

코드 커밋 `7c86738`을 고정한 뒤 Windows 기본 진입점에서 공식 Beam을 다시 완주했다. 같은 커밋으로 트레이 수집·조회 두 회차도 재현했다. 세 실행 각각에서 소스·설정 15개 해시를 Git 원문과 대조했다. 전달 ZIP 세 개의 내부 파일 총 237개 해시가 일치했다. [커밋 고정 재현 기록](evidence/postcommit_verification.json) · [압축 파일 검증](evidence/release_verification.json)

## 2026-11-22 · 설계 v1 통합 중간 확인

F의 독립 브랜치에서 공식 Beam 원본 기준선, 실제 STEP 13개 인스턴스, 새 Beam 연속 제거 4개, 배터리 덮개의 경로 통과 후 안정성 실패 및 실제 DINOv2 벡터를 확인했다. 4부품 진단에서는 트레이 실패 뒤 기둥과 트레이 제거가 같은 경로에서 검증됐다. 실제 Astra 생성 호출은 0회다. 다음 확인 지점은 분리된 두 회차의 저장·검색 연결과 전달 파일 무결성이다.

## 2026-11-21 공개 당시 상태

## Reviewed publication snapshot

This is a reviewed source snapshot of the existing research implementation. Original C-drive source and all historical results remain unchanged. The publishing checkout and new logs are on F drive. No E-drive access, paid model call, or new physics experiment was performed for publication.

## Included implementation

- Native ASAP sequence adapter, external candidate ordering and recorded physics budgets.
- Host model services, IPC, LangGraph workflows and budget/transport guards.
- Experience/lesson/Python-skill representations, relation-aware retrieval, restricted execution and provenance checks.
- Older explicit-action and recovery pipelines retained separately.
- Fastener-state logic; this is not screwdriver control or real bolt removal.
- Two model-generated scoped registered order skills and three selected historical execution reports.

## Actual checks performed for publication

- Destination repository and push permission: verified; empty public repository.
- Exact allowlist: 577 pre-existing code/test/config/document files, 6,501,588 bytes.
- No original CAD, full experience library, raw provider logs, credentials or user attachments included.
- Secret/privacy scan: no genuine credentials or personal paths/email/phone found in the allowlist. One explicit fake token in a redaction unit test was reviewed and retained.
- API-free tests on F checkout: 59 tests, 49 passed, 10 errors. All 10 errors are C-only memory/ledger path guards on the F checkout. The original guards were preserved.
- Portable subset: 39 tests passed; these overlap the 59-test run and are not 39 additional independent tests.
- Syntax, source preservation and Git remote checks: final results in PUBLICATION_CHECKS.md.

## Historical evidence, not new runs

- Official Beam baseline: 4 removals, native completion. See runs/m1_wsl_20261016/report.md.
- 00335 sequence-native Sol run: 15/15 removals, part-level PASS. See its included report.
- 2026-11-19 knowledge integration: scoped offline evidence; actual model choice/native performance must be distinguished from injected responses.
- Battery-pack failure: see reports/BATTERY_PACK_DIAGNOSIS.md. No whole-assembly impossibility claim.

## NOT_RUN / NOT_IMPLEMENTED / UNVERIFIED

- New paid API, RedMax full run, fresh PC install, robot simulation and hardware: NOT_RUN.
- UR3e/2F-85 disassembly execution and new assembly sequence handoff contract: NOT_IMPLEMENTED (design only).
- General superiority over ASAP and general benefit of memory/skills: UNVERIFIED.
- Automatic portability of historical hard-coded Windows/WSL paths: UNVERIFIED; known C-only guards remain.
- Full frozen 704-card library: intentionally not included in public snapshot.
