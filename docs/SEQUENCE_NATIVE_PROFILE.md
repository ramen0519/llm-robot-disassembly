# 부품 순서만 판단하는 새 경로 — sequence_native_v1

2026-11-10 사용자 지시에 따라 구현했다. 기존 명시 행동·계층 계획 경로는 보존하며 새 실행기를 통해서만 활성화한다. 로봇 실행 모듈이나 조립 프로젝트의 계약을 구현한 것은 아니다.

## 역할

1. 기존 CAD 입력에서 남은 부품과 기하·접촉 정보를 만든다. 기본 후보 나열은 문자열 정렬이며 heuristic/GNN 추천 순서가 아니다.
2. 동결한 경험·교훈·호환되는 순서 스킬을 조건·출처와 함께 제공한다. 평가 제품00335와 현재 제품에서 나온 과거 지식은 기본 제외한다. 현재 회차의 실제 관측은 별도 제공한다.
3. LLM은 모든 후보가 한 번씩 들어가는 **시도 우선순위**를 반환하거나 등록된 순서 Python 스킬을 선택한다. 한 번 반환한 순위를 ASAP이 소비하는 동안 방향·지지별 API 호출을 하지 않는다. 새 탐색 상태에서 다시 순위를 요청할 수 있다.
4. 원래 ASAP DFS가 후보를 시도한다. 원래 안정 자세 생성·확률 순서, 6축 경로 검사, 떨어지는 부품에 따른 greedy 지지 탐색을 사용한다. 실패 후 다른 자세/후보 및 backtracking도 ASAP이 소유한다.
5. 원래 RedMax 결과와 시간 내 수락 여부를 기록한다. 다음 상태로 이어지는 **연결된 계획 체인**만 완료 수준으로 집계한다.
6. 회차 종료 후 실제 native 관측을 별도 동결 파일에 저장한다. 다음 회차가 명시적으로 이 파일을 읽으면 순서 판단 참고자료로 사용할 수 있다. 평가 중 장기 메모리를 자동 변경하지 않는다.

**예외:** 원래 ASAP에서 마지막 두 부품의 제거 선택은 generator를 호출하지 않고 질량·고정 베이스 규칙을 사용한다. 이 경로도 원형을 유지했다. 따라서 ‘GNN이 하던 순위 역할을 LLM으로 교체’라고 설명하며 ‘마지막 제거까지 전부 LLM이 선택’이라고 설명하지 않는다.

## 입력·출력 계약

- 입력: `rank_request`, `features.protocol=sequence_native_v1`; 실제 CAD, 출처가 있는 동결 관측·교훈, 현재 회차 관측 최대12건.
- 직접 판단 응답: `request_id`, `state_id`, `decision_kind=direct`, 전체 순열 `candidate_ids`, `selected_skill_id=null`, 제공된 `evidence_ids`.
- 스킬 선택 응답: `decision_kind=skill`, `candidate_ids=null`, 실제 제공한 `selected_skill_id`. 호스트가 제한 VM으로 실행해 전체 순열을 검증한다. LLM이 코드 실행 결과를 추측해서 중복 제출하지 않는다.
- 자세·방향·지지·물리 판정을 출력하거나 후보를 누락·중복·추가하면 거절한다.
- 잘못된 응답, 연결 실패, 예산 소진은 **중단**한다. canonical/heuristic 순서로 몰래 계속하지 않는다. 자동 HTTP 재시도0회.
- 본문 전체65,536 UTF-8 bytes 제한. 과거 항목은 내용과 출처를 함께 유지한 채 항목 단위로 줄이고 그 이유를 기록한다. 인용 ID·현재 회차 피드백·응답 스키마도 크기 계산에 포함한다. 현재 기하를 일부 삭제해 제한을 우회하지 않는다.

## 메모리와 스킬의 실제 상태

- 기존 `explicit_action_v1` 관측은 출처가 다른 제품의 **조건부 참고**로 제공한다. 과거 PASS는 현재 물리 PASS가 아니며 현재 부품 ID로 치환하지 않는다.
- 새 `sequence_native_observation_v1`은 최종 worker 결과의 attempts를 보존한다. native 한 시도에는 여러 방향/지지 검사가 포함될 수 있으므로 기존 단일 행동 형식으로 위장하지 않는다. `accepted_for_transition`, 마감, raw 결과, 입력·소스 해시를 유지한다.
- `frozen_lessons_v1`은 출처 제품을 확인할 수 있는 경우 참고 가설로 제공한다. 출처 미상은 제외한다.
- 기존 자동 생성 Python은 물리 복구 행동용이다. **순서 스킬과 호환되지 않아 실행하지 않고 원본을 보존한다.** 관련 교훈·관측은 참고할 수 있다.
- 새 순서 스킬은 `sequence_skill_v1` 카드와 `python_policy_v1` 프로그램으로 표현한다. `propose(ctx)`가 전체 후보 순열과 근거 ID만 반환한다. 파일·네트워크·검증기·장면 수정은 허용하지 않는다.
- 현재 실제 모델이 생성·검증한 순서 스킬은 **아직 없다**. 선택→VM 실행→순열 검사는 fixture 프로그램으로 검증했다. 실제 생성·등록·성능효과는 별도 유료/물리 검증 대상이다.

## 구현 파일

| 경로 | 주요 함수/역할 |
|---|---|
| `worker/sequence_native.py` | `SequenceNativeSession.generate`, `installed_sequence`, `connected_path`, `completed_in_budget` |
| `worker/run_sequence_native.py` | `main`; 원래 DFS 호출, 물리 계수·입력/소스 보존, `--prepare-only` |
| `orchestrator/sequence_ranker.py` | `SequenceRankService.prepare/rank/decide`, `FixtureSequenceService`; 기존 영속 예산 장부 재사용 |
| `orchestrator/sequence_knowledge.py` | `build_sequence_knowledge`, `build_native_snapshot`, `validate_sequence_skill`, `execute_sequence_skill` |
| `tools/run_sequence_native_experiment.py` | Windows/WSL IPC, 기존 LangGraph 단계, 새 run, 동결 파일 해시, 결과 집계 |

## 실행

### 선택 사항 — 체결 상태를 먼저 확인하는 순서 계획

후면 하우징과 프레임, 볼트 네 곳을 STEP의 개별 배치 경로에 연결한 설정은 `configs/lab_motor_rear_fasteners.json`이다. 아래 명령은 우리 순서 플래너의 체결 단계만 실행한다. 사건 파일을 생략하면 모든 볼트가 체결된 초기 상태를 출력한다.

```powershell
& ".\.venv-orchestrator\Scripts\python.exe" -B worker/run_sequence_native.py fasteners --joint configs/lab_motor_rear_fasteners.json --bom runs/lab_total_structure_v1/bom/bom.json --events runs/lab_motor_fasteners_20261117/events.json
```

`unfasten`을 확인하면 체결이 풀리지만 볼트는 아직 구멍에 남는다. `recover`를 확인하면 볼트가 회수된다. 실패·불명은 마지막 확인 상태를 유지한다. 되감기는 사건을 덮어쓰지 않고 해당 분기의 이력을 재생한다. 체결된 볼트와 두 부품은 계획상의 한 묶음이고, 모두 풀린 뒤에도 회수되지 않은 볼트가 있으면 덮개는 후보에서 제외한다. 이 단계의 프레임은 이번 작업에서 남겨 둘 부품이며 실제 지그 고정의 증거가 아니다.

일반 물리 worker에도 같은 세 옵션을 명시할 수 있다. 체결이나 미회수 볼트가 남아 있으면 순위 요청과 물리 장면 생성 전에 중단한다. 네 볼트를 회수한 뒤에는 해당 볼트들을 뺀 OBJ와 그 조립 폴더의 `fastener_occurrences.json`이 필요하다. 파일 형식은 `{"source_step_sha256":"…","placements":{"<남은 STEP 배치 경로>":{"obj":"0.obj","sha256":"…"}}}`이며 모든 잔여 배치와 OBJ 해시가 일치해야 한다. 후보가 생겨도 바로 제거 성공으로 기록하지 않는다. [이 모터의 검증 기록](../runs/lab_motor_fasteners_20261117/REPORT.md)

기본은 `--mode prepare`로 모델·물리 호출 없이 CAD만 검사한다. `--mode fixture`는 실제 ASAP/RedMax를 사용하지만 순서 응답은 테스트용이다. 실제 모델은 `--mode llm --allow-live-api --llm-config <별도 승인한 설정>`에서만 활성화한다. `--use-registered-key`는 live에서만 기존 사용자 암호화 키를 읽고 오프라인에서는 거절한다.

초기 API 없는 실행 명령은 `runs/sequence_native_prepare_20261110/*/execute/command.json` 및 `cad_command/command.json`, `offline_tests/command.json`에 기록했다. 이후 승인된 실제00335 유료 실행은 `runs/sequence_native_00335_live_20261110/REPORT.md`에 기록했으며15/15 제거 PASS, Sol16회였다. 초기 `api_proposal.json`은 당시 제안 기록이고 새 실행 승인이 아니다. `llm_config.disabled.json`은 호출 상한0의 비활성 설정이다.

## 검증 한계

공식5부품 Beam의 API 없는 고정 응답 검사와 이후00335의 실제 Sol15/15 제거는 구분한다. 한 제품의 성공은 메모리·스킬 효과나 GNN 대비 일반 우월성의 증거가 아니다. 이후 순서 학습 연결은 `docs/SEQUENCE_LEARNING_PROFILE.md`와 `runs/sequence_learning_prepare_20261110/REPORT.md`를 따른다. 실제 모델이 생성한 순서 스킬의 효과, 로봇 파지·이동·실물 성공은 미검증이다.
