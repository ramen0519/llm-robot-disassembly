# 조립 프로젝트에 넘길 순서 제안 모듈의 범위

> 2026-11-06 후속 설명: [현재 구현과 향후 역할 분리](ROBOT_ROLE_BOUNDARY.md)를 함께 읽는다. LLM 작업 판단과 파지·추출·로봇 모션 계산의 경계를 명확히 한 권고안이다. 기존 ASAP 계약은 변경하지 않았으며, 검증 전 순서 후보를 완료 계획으로 취급하지 않는다. 새 로봇 기능/API/실험은 아직 미실행이다.


작성일: 2026-11-06. **제안 계약 v1 / 아직 schema·변환기·새 실행 옵션을 구현하지 않음.**

## 1. 한 문장 계약

“이 부품들과 이 조건에서 확인한 분해 순서 후보와 근거를 준다. 조립 순서로 채택할지, 로봇이 잡고 넣을 수 있는지는 조립 프로젝트가 결정하고 검증한다.”

| 포함 | 제외 |
|---|---|
| 부품 ID·asset/단위·좌표 대응, 전체/선택적 목표 | 카메라 인식·자동 CAD 대응 |
| 선택된 분해 계획 체인, 순서, 남은 부품 | DFS의 모든 탐색 시도를 한 순서로 합친 결과 |
| 단계별 조건: 남은 부품·지지·자세, 실제 verifier 출처 | 지지/자세가 로봇으로 실현 가능하다는 보증 |
| 완료/부분/불확실/오류 구분, 실패와 예산 정보 | 팔 관절 명령·그리퍼 명령·삽입 제어·실물 명령 |
| 검증된 사실과 계획 후보 구분 | 장기 Memory/교훈/스킬 DB 전체, API 키/원시 비공개 prompt |

조립 프로젝트는 이 모듈 없이도 자체 순서를 받을 수 있어야 한다. 모듈 실패/미완료를 조립 성공으로 바꾸지 않으며, 자체 fallback을 쓰면 정책과 근거를 별도 기록한다.

## 2. 현존하는 연결 코드는 두 종류다

확인 경로 기준: `C:\dev\llm-robot-assembly`.

| 생산/변환 경로 | 현재 형식과 조건 | 소비 측 |
|---|---|---|
| `scripts/disassembly_adapter.py::candidate_from_report` | `explicit_action_v1` 보고서의 완료된 `selected_plan`과 PASS/수락·연속성 확인 → `kind=assembly_candidate`, schema_version 1. `robot_execution_validated=False`, `coordinate_binding_validated=False` | 좌표 바인딩은 별도. 아래 소비 함수와 같은 입력이 아님 |
| `scripts/run_asap_native.py::assess_candidate` | native ASAP 보고서와 정규화/mesh 출처 → `native_assembly_candidate/v1` | `scripts/planner_candidate_input.py::load_robot_candidate`가 이 형식과 부속 파일/hash/부품 대응을 검증 |

`load_robot_candidate`는 보고서, normalization, mesh, CAD/part map을 대조한다. 단순 ID 배열만 주면 현재 기능이 그대로 작동한다고 설명하면 안 된다. 문서 `docs/STEP_16E_PLANNER_TO_ROBOT.md`의 작은 외부 순서 연결 시험과 전체 10부품 조립 기록도 구분한다.

**향후 변경 지점:** 분해 쪽에 versioned export 변환기를 추가하고, 조립 쪽에는 별도 승인된 작업에서 그 계약을 읽는 좁은 adapter를 추가한다. 현재 두 형식은 보존한다. 이번에는 어느 쪽 코드도 수정하지 않는다.

## 3. 제안 입력 계약

우선 로컬 파일 JSON으로 교환한다. HTTP 서버나 상시 서비스는 필요해졌을 때 추가한다.

| 필드 | 의미와 조건 |
|---|---|
| `schema` / `request_id` | 예: `sequence_request/v1`; 추적 가능한 요청 ID |
| `assembly_id`, `asset_manifest` | 부품별 ID와 mesh hash, 단위 스케일, 좌표 정의. 부품 ID 자동 재해석 금지 |
| `initial_state` | 초기 부품 pose, fixed base/fixture, 부품 목록. 실제 지원하지 않는 고정/지지 관계는 명시 보류 |
| `goal` | `full_disassembly` 또는 `target_recovery`와 목표 ID. 두 완료 조건을 혼용하지 않음 |
| `planning_profile` | 원래 ASAP heuristic/GNN 또는 제안 선택기 등 명시한 방식·버전. 알고리즘/가중치/seed 출처 |
| `verification_profile` | verifier·설정 hash·solver/seed·예산. 로봇 검증을 자동 포함하지 않음 |
| `budget` | wall/physics 상한; LLM 방식 선택 시 별도 승인된 호출/비용 한도. 숨은 유료 호출 금지 |

좌표변환 convention과 길이 단위를 필수로 선언한다. 단위가 없는 OBJ 크기를 임의로 미터로 해석하지 않는다. 입력 부품 목록·mesh가 달라지면 이전 계획을 정확 일치 결과로 재사용하지 않는다.

## 4. 제안 출력 계약

| 필드 | 의미 |
|---|---|
| `schema`, `request_id`, `result_id` | `sequence_proposal/v1` 등 명시 버전과 요청 연결 |
| `status` | `COMPLETE`, `PARTIAL`, `UNKNOWN`, `ERROR`. 구조적 유효성과 물리 성공을 별도로 기록 |
| `goal`, `goal_satisfied` | 무엇을 완료했는지. 선택적 회수 성공은 전체 분해 완료가 아님 |
| `disassembly_order`, `remaining_part_ids` | 실제 수락된 **선택 계획 체인**의 제거 순서와 잔여 부품 |
| `steps` | 제거 부품, before/after 부품 집합, 조건부 support/pose, 근거 참조. 근거가 없으면 null/unknown |
| `evidence_scope` | 부품 수준/로봇 시뮬레이션/실물 중 실제 검사한 범위 |
| `source` | planner/verifier/code 버전, 입력·설정·보고서 hash, seed, 검증된 evidence 참조 |
| `limitations` | 미검증 접근·잡기·지그 변경·삽입, timeout/부분 종료 이유 등 |
| `metrics` | 계획 시간·물리 호출·모델 호출·추정비용, 준비 비용 별도 |
| `robot_execution_validated` | 기본 false. 이 출력 자체는 로봇 명령/성공 보증이 아님 |

계획 탐색 중 검사한 성공 edge와 실제 로봇 실행 관측도 서로 구분한다. report 전체의 성공 action들을 모아서 순서로 만들지 않는다. backtracking 이후 선택 체인 밖의 성공은 출력 순서에 넣지 않는다.

예: 부품 A/B/C와 고정 베이스에서 “C→B→A 제거”를 완성했다면 조립 쪽은 “A→B→C 설치”를 **검토 후보**로 만들 수 있다. 그러나 반대 방향의 접촉·끼움·그리퍼 접근이 가능한지는 조립 쪽에서 다시 검사한다. 특정 순서 하나가 성공했다는 이유로 “다른 모든 순서는 불가능”한 선후 제약을 만들지 않는다.

## 5. 소비 측 정책

1. schema, 입력/보고서 hash, ID 대응, 단위/좌표, 실행 조건을 검사한다. 다른 제품의 source ID를 그대로 쓰지 않는다.
2. `full_disassembly + COMPLETE + goal_satisfied`이면서 선택 체인이 완전하고 연속일 때만 전체 역순 후보를 생성한다. fixed base/최종 잔여 seed는 별도 명시한다.
3. PARTIAL/UNKNOWN/ERROR와 선택적 회수 결과는 전체 조립 순서로 승격하지 않는다. 확인된 부분 정보로만 보여주거나 자체 계획에 참고한다.
4. 각 단계의 지지/자세 조건은 출처 근거로 전달한다. UR3e 1대와 실제 지그로 구현 가능한지 조립 실행기에서 별도 판단한다.
5. 조립의 grasp/IK/충돌/운반/삽입/최종 검사 후에만 로봇 성공을 기록한다. LLM의 설명, source PASS, 프로세스 exit 0으로 대체하지 않는다.

## 6. 구현 전 정한 검증 항목

| 검사 | 기대 결과 |
|---|---|
| 연속·완전한 저장된 선택 체인 | 같은 제거 순서·잔여 seed·출처 보존; robot validated는 false |
| 같은 부품 두 번 제거, 중간 상태 누락, branch 혼합 | 거절/ERROR, 임의 복구해서 완료 처리하지 않음 |
| PASS지만 deadline 초과/상태 전이에 미수락 | 완료된 계획 체인으로 채택하지 않음; 원시 PASS는 보존 |
| unknown/부분/선택적 회수 | 전체 조립 순서 생성 안 함 |
| ID/mesh hash/단위/좌표 불일치 또는 누락 | 실행 입력으로 바인딩 보류 |
| 지원하지 않는 버전/oversize/nonfinite/중복 JSON key | 명시 오류, 기존 소비 경로에 무조건 전달하지 않음 |
| API 오류·유료 예산 미승인 | 무단 추가 호출 0; 이유와 실제 사용량 기록 |
| 기존 native/explicit 형식 | 기존 결과/소비 코드를 유지; 새 adapter와 각각 회귀검사 |

이번에는 위 테스트를 실행하지 않았다. **문서 내 계약 설계**이며 구현 후 통과해야 할 조건이다. 조립 저장소의 새 adapter 작업은 해당 프로젝트 범위에서 따로 진행한다.
