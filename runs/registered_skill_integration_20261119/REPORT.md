# 기존 등록 순서 스킬 2개의 새 검색·실행 통합

2026-11-19. 사용자 요청: 기존 등록 스킬의 출처와 실행 형식을 연결하고, 새 검색→선택→실행을 API 없이 확인한다. 기존 결과를 덮어쓰지 않고 E드라이브는 접근하지 않는다.

## 실제로 고친 누락

이전 v5 보관본에는 두 등록 스킬이 있었으나 익명 학습 경험의 출처를 풀지 못해 `unclassified`였고 실행용 663개 목록에는 없었다. 실행용에 보이던 8개 Python은 별개의 미등록 후보였다. 또한 기존 등록 형식은 부품별 수치 점수를 반환하고 현재 native 순서 실행기는 전체 후보 순열을 받으므로, 검색 카드만 바꿔서는 실행되지 않았다.

이번에는 원본 생성 packet의 39개 익명 관측을 실제 제품·관측·CAD에 연결하고, 두 스킬의 생성·검증·등록 근거를 확인했다. 원래 Python·등록 상태·검증 범위는 변경하지 않았다. 제한 VM으로 원래 점수를 계산하고 전체 후보 순위로 변환하는 adapter를 추가했다.

| 기존 등록 | 원본 파일 | 유지한 상태 |
|---|---|---|
| Luna 작성 순서 스킬 | `runs/scored_order_skill_pilot_20261111/registry/skill.json` | REGISTERED_SCOPED |
| Astra 작성 순서 스킬 | `runs/scored_order_astra_author_20261112/registry/skill.json` | REGISTERED_SCOPED |

두 스킬의 원래 물리검증 범위는 `path_only_sequence_proposal/v1`의 학습 제품 두 개다. 이번 연결을 `sequence_native_v1` 물리검증 성공이나 신규 스킬 등록으로 보고하지 않는다. 두 생성 기록에는 별도 교훈 생성물이 없으므로 없는 교훈은 만들어 넣지 않았다.

## 저장 및 실행 흐름

1. `scored_skill_provenance.load_verified_scored_provenance`: 생성 packet·실측 로그·등록 내용·코드 해시를 검증한다. 인용한 경험뿐 아니라 생성 때 본 39개 경험과 전체 5제품을 기록한다.
2. `Migrator.convert`: 기존 카드의 원문을 보존하면서 출처·기하·학습 분류를 연결한다. 결과는 새 `migration_v6`에 저장한다.
3. `build_bridge/validate_bridge`: 두 등록과 원래 검증 기록, VM 코드의 해시를 동결한다. 미등록 후보를 실행 가능하게 승격하지 않는다.
4. `SequenceRankService.prepare` → `searched_bridges`: 새 관계 검색에서 등록 카드가 발견되고 동결된 실행 코드와 출처가 일치할 때 선택 목록에 포함한다. 접촉 정보가 없는 경우는 이유를 기록하고 직접 순위 판단을 남긴다.
5. 모델 응답 계약은 `direct` 또는 제공된 `skill_id`를 허용한다. 전체 Python은 한 번만 제공하고 관계 카드는 이를 가리킨다. 제공·제외 이유와 입력 용량을 기록한다.
6. `SequenceRankService.decide` → `execute_bridge`: 요청/상태/코드 결합을 검사하고 제한 VM에서 원본 코드를 실행한다. 고정 베이스를 포함한 전체 잔여 부품으로 특징을 계산한 뒤 베이스만 제거 후보에서 제외한다. 보류/거절은 이유와 선택 ID를 보존하고 실행을 중단한다.
7. 결과는 부품 시도 순서다. DFS·자세·방향·지지·경로·안정성은 계속 원래 ASAP 책임이며 이번에는 물리 실행을 하지 않았다.

## 라이브러리 변화와 보존

| 항목 | 이전 v5 | 새 v6 |
|---|---:|---:|
| 전체 보관 카드 | 2,983 | 2,983 |
| 실행용 경험 | 603 | 642 |
| 실행용 교훈 | 52 | 52 |
| 실행용 미등록 참조 스킬 | 8 | 8 |
| 실행용 기존 등록 스킬 | 0 | 2 |
| 실행용 합계 | 663 | 704 |

전체 카드 중 41개의 연결 메타데이터만 바뀌었다. 기존 실행용 663개 카드와 전체 원문·Python·상태는 동일하다. 기존 실험 결과493개, v5 근거1,098개, v6 근거1,143개를 대조했으며 중복을 제외한1,621경로 모두 해시가 일치했다. `migration_comparison.json`, `preservation_final.json`에 근거가 있다. 기존 운영 소스의 수정 전 사본은 `before/`와 `before_root.json`에 보존했다.

## 검증 종류를 구분한 결과

실제 저장 CAD의 초기/중간 상태를 읽었다. **모델 응답만 테스트용으로 주입했고 등록된 Python은 실제 제한 VM에서 실행했다.** 외부 네트워크와 프로세스 실행을 차단했고 차단된 시도도 0이었다. 이는 실제 모델이 자유롭게 선택했다는 결과가 아니다.

| 저장 CAD 상태 | 잔여 부품 | 직접 순서 / Luna 코드 / Astra 코드 |
|---|---:|---|
| 20014 초기 | 20 | 3/3 PASS |
| 20014 연결된 분해 체인 이후 | 12 | 3/3 PASS |
| 00709 초기 | 21 | 3/3 PASS |
| 00709 연결된 분해 체인 이후 | 15 | 3/3 PASS |
| 07009 초기 | 5 | 3/3 PASS |

15개 provider 계약 재생 시험에서 두 스킬이 최종 본문·ID schema에 존재하고 코드 바이트가 원본과 같은지 확인했다. 스킬 실행10건은 구 VM과 순서·특징 해시·실행 단계 수가 일치했다. 직접 선택5건은 스킬을 실행하지 않았다. 07009는 두 원본의 부품별 수치가 완전히 같은 것을 확인한 후 기존에 측정된 접촉 그래프를 연결했으며 원본 파일은 변경하지 않았다.

`provider_replay_01/` 및 최종 `provider_replay_final/`에 실제 준비 본문, 주입 응답, 실행 결과, 원래 VM 대조, 입력 해시와 요약을 보존했다. 같은15건의 반복 점검이며 독립30실험으로 집계하지 않는다. 본문은 상태별 약30~42KB였고 원문 코드를 자르지 않았다. 입력 준비 PASS는 물리 분해 성공을 뜻하지 않는다.

**최종 고정 코드 검사: 183개 전부 PASS, 실패/skip 0, 136.577초.** 출처10검사·bridge18검사·선택/실행 통합16검사와 기존 scored/native/검색/학습 계약 회귀를 포함한다. `tests_final/command.json`, `stdout.log`, `stderr.log`에 전체 명령과 결과가 있다. 최종 provider 재생15건도 PASS이며 네트워크/외부 프로세스 시도0, 입력·실행 코드 해시 변경0이었다.

## 실제 명령과 로그

아래 명령은 모두 `.venv-orchestrator/Scripts/python.exe -X utf8 -B`로 실행했다. 기록 래퍼 `tools/run_recorded.py`가 각 명령의 전체 argv·시작/종료·종료코드·stdout/stderr를 새 폴더에 남겼다.

| 명령 | 로그 위치 |
|---|---|
| `tools/migrate_relational_knowledge.py --output runs/relational_knowledge_migration_20261119/migration_v6` | `migration_command/` |
| `tools/prepare_scored_sequence_bridges.py --registry <Luna 원본> --registry <Astra 원본> --output runs/registered_skill_integration_20261119/frozen_bridges.json` | `freeze_command/` |
| `-m unittest tests.test_registered_sequence_integration tests.test_sequence_ranker tests.test_sequence_runner -v` | `integration_tests_01/02/03`, `integration_tests_final/` |
| 출처·scored VM·관계 검색·기존 native 학습의 회귀 검사 | `regression_tests_01/` |
| `tools/verify_registered_sequence_integration.py --library <migration_v6/training_runtime_library.json> --bridges <frozen_bridges.json> --output <새 replay 폴더>` | `provider_replay_command/`, `provider_replay_01/` |
| 최종 동일13개 테스트 모듈 통합 실행 | `tests_final/` |
| 최종 source pin 확대 후 provider 재생15건 | `provider_replay_final_command/`, `provider_replay_final/` |
| `tools/run_sequence_native_experiment.py --help` | `runner_help_command/` |

초기 연결 테스트 실패 로그도 보존했다. 합성 입력에 접촉 그래프가 없어서 스킬이 정상 거절된 사례와, 오류 응답의 기존 입력 ID 반환을 실행으로 잘못 해석한 테스트 기대값을 바로잡았다. 접촉 미관측은 운영 코드에서 유효한 그래프로 꾸미지 않으며 제공 단계에서 제외한다. 선택한 스킬의 거절 진단 누락도 수정했다.

넓은 회귀 검사 첫 실행에서는 143개 중 반복 학습 2개가 중단됐다. 당시 검사 도중 운영 `.py`를 편집했다. 이후 수정 전 5모듈을 별도 프로세스에 주입한 동일 2검사와, 편집을 멈춘 현재 코드의 동일 2검사는 모두 통과했고 25작업 연결도 완료했다. 최초 원문 예외가 없어 소스 변경 감시 중단이라고 확정하지 않는다. `iterative_regression_diagnosis.json`과 `ITERATIVE_REGRESSION_DIAGNOSIS.md`에 초기 실패·백업 대조·재실행을 구분했다. 진단 첫 시동의 import 경로 오류도 보존했다.

별도 읽기 전용 검토에서 지적한 선택 거절의 진단 유실, 미관측 접촉 입력, 제공/제외 목록 모순을 보완했다. 전체 생성 노출 제품은 실행 허용 목록뿐 아니라 참조 카드의 검색·읽기에서도 제외한다. 최종 검사 중에는 소스를 고정한다.

## 이번에 하지 않은 작업

- 실제 API 호출·전송·비용: **0회 / $0**. 테스트 ledger의 토큰·금액은 주입 응답/가상 단가이며 실제 청구가 아니다.
- 새 ASAP/RedMax 물리검증, 새 전체 분해, 로봇 실행: **NOT_RUN**.
- 새 LLM 교훈/Python 생성·신규 등록: **NOT_RUN**.
- 실제 LLM이 스킬을 자율 선택하는지, 분해 성공률·지지 점수·시간이 개선되는지: **UNVERIFIED**.
- DFS 제어 정체 등 앞서 감사한 별도 문제: 이번 범위에서 수정하지 않음.

다음 실제 시험은 위 두 동결 입력을 명시하고 동일 물리 설정·예산으로 선택/비선택/실행/물리 성공을 분리 집계한다. 별도의 API/물리 실행을 자동으로 시작하지 않았다.
