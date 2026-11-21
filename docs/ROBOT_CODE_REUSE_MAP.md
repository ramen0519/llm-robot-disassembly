# 조립 코드 재사용과 분해 신규 기능의 경계

작성일: 2026-11-06. **소스 검토에 근거한 이식 설계이며 이식 완료 목록이 아니다.**
원본 조립 저장소는 읽기만 했다. 확인한 HEAD: `9786b09428e231124f3e55c3f841b8511b82fcd5`.
원본 파일별 SHA-256은 `runs/robot_disassembly_design_20261106/assembly_before.json`에 기록한다. HEAD만으로 로컬 소스가 같다고 가정하지 않는다.

## 1. 기반으로 재사용할 후보

아래 조립 경로는 모두 `C:\dev\llm-robot-assembly` 기준이다. 함수 본문과 호출 제약을 확인했으며 새로운 분해 장면에서의 실행은 아직 NOT_RUN이다.

| 기존 파일·주요 함수 | 가져올 기반 | 남은 확인/변경 |
|---|---|---|
| `models/universal_robots_ur3e/`, `models/robotiq_2f85/`; `scripts/ur3e_with_gripper.py::build_robot_spec`, `load_robot_with_gripper` | UR3e/2F-85 모델·장착 관계 | 원본 README/라이선스/asset hash 보존. 단순화 모델이며 실제 로봇 교정값으로 간주하지 않음 |
| `scripts/move_tcp.py::ArmPositionController.apply`; `view_tcp.py::read_tcp_pose`, `check_tcp_frame` | 관절 제어, TCP 좌표 확인 | 새 workcell 좌표·관절/actuator 이름·단위·timestep 대조 |
| `scripts/solve_tcp_ik.py::solve_tcp_pose` | 목표 TCP의 IK와 조건 검사 | 새 장면/접근 방향에서 도달성·충돌·관절 한계 회귀검사 |
| `scripts/workcell_motion.py::WorkcellMotion.plan`, `execute`, `step` | Cartesian 구간 계획, 제어 실행, 중간 관측 | verifier와 stage의 분해용 계약 필요. `_candidate`의 운반 부품 예측은 복사 장면에만 적용하며 live teleport 금지 |
| `scripts/joint_path_planner.py::plan_joint_path`, `plan_tcp_path` | 충돌 검사 기반 RRT-Connect 자유공간 계획 | 기존 edge 표본 검사는 연속 접촉 분리 가능성의 증명이 아님. 잡은 부품 형상/지그/나머지 제품을 모두 검사 |
| `scripts/failure_report.py::FailureCode`, `FailureEvent`, `TrialRecorder` | 실패 기록·근거·회차 산출물 패턴 | 조립 전용 실패 코드를 분해 판정에 대응; 원인 미확인은 UNKNOWN으로 기록 |

기존 RRT 공급 코드는 `third_party/motion_planners/`와 그 라이선스·상류 출처를 함께 보존한다. 로봇 모델 README도 함께 이식한다. 모델 외형이 있다고 접촉/마찰/실물 정확도가 검증된 것은 아니다.

## 2. 수정해서 재사용해야 하는 코드

| 기존 파일·함수 | 현재 전제 | 분해에서는 필요한 변경 |
|---|---|---|
| `scripts/assembly_skills.py::AssemblySkills.approach`, `pick` | 지지면 위 자유 부품, 정해진 크기/자세 범위, 닫기 후 들어 올리기 | 조립된 대상의 접촉 관계와 주변 부품을 관측. `잡기`와 `분리 이동`을 분리; 단순 lift로 끼워진 부품을 뽑지 않음 |
| `scripts/empty_approach.py::plan_empty_approach` | 빈 그리퍼, 안정된 지지 부품, `approach_hover` 단계 | 조립체 내부 대상의 접근/접촉 조건. 기존 guard를 통째로 제거하는 방식 금지 |
| `scripts/loaded_transport.py::_observe_free_grasp`, `plan_loaded_transfer` | 이미 free grasp, 지지 접촉 없음, 바닥 간격 최소 0.05m, mating 비활성 | 분리 완료와 안전 이격 확인 후에만 재사용. 분리 전 경로를 이 함수에 우회 전달하지 않음 |
| `scripts/workcell_verification.py::WorkcellVerifier.contacts`, `physical_state`, `check_held`, `verify_placement` | 특정 작업대/부품과 조립 동작 단계 | 여러 잔여 부품·지그·제거 부품·그리퍼에 맞는 contact policy. 허용 접촉과 금지 충돌을 단계별 구분 |
| `scripts/frame_assembly_skills.py::FrameVerifier`, `FrameSkills.check_grasp` | 설치된 부품 목록, 삽입 방향/깊이 검증 | 제거/잔존 목록과 추출 가능성 검사로 재구성. 기존 `check_grasp`의 삽입 경로 검사를 분해 검사라고 부르지 않음 |
| `scripts/post_grasp.py::numeric_grasp`, `world_grasp`, `grasp_geometry` | 프레임 기하와 seating 조건 | 좌표/치수 계산 원칙 재사용, 분리 전 파지면/손가락 간섭 조건 재검증 |
| `scripts/frame_product.py::product_parts`, `solid_for`, `export_product`; `frame_assembly_scene.py::build_frame_scene` | 프레임 제품 생성 및 조립 작업 장면 | CAD 생성기 후보로 활용. 조립된 초기 pose, 물리적 끼움·지그·분리 관계를 새 장면으로 명시 |
| `scripts/assembly_graph.py::run_assembly_graph` | 조립 action 목록, 삽입 완료와 조립 복구 | 직렬 실행·관측·revision 패턴만 참고. 새 분해 상태/행동과 기존 학습 orchestrator의 소유권을 맞춤; graph 두 개가 같은 로봇을 제어하지 않음 |

`scripts/insertion_skills.py::InsertionSkills.insert`, `InsertionVerifier.verify_insertion`과 `insertion_recovery.py::recover_alignment`는 조립 전용 절차다. 낮은 수준의 제어/관측 패턴은 참고할 수 있으나 **삽입 궤적을 역재생한 것을 검증된 분해 스킬로 등록하지 않는다.**

## 3. 메모리·스킬 코드의 재사용 판단

| 코드 | 실제 범위 | 결정 |
|---|---|---|
| 조립 `scripts/experience_memory.py::condition_fingerprint`, `experience_from_run`, `retrieve_experiences` | `four_post_frame_first_joint/v1`의 정확한 simulation 조건과 삽입 증거를 요구 | 형식 검증·출처·조건 대응 원칙만 참고. 범용 분해 Memory로 그대로 복사하거나 기존 DB를 합치지 않음 |
| 조립 `scripts/frozen_knowledge.py::freeze_knowledge`, `FrozenKnowledge` | 조립 경험/recipe를 동결하여 조회 | 회차 전 동결 원칙 유지. 분해의 기존 동결 체계와 명시 변환 필요 |
| 조립 `scripts/skill_recipes.py::create_candidate`, `validate_candidate`, `RecipeReplayPolicy` | 성공한 `recover_alignment → insert`를 고정 템플릿으로 추출. `contains_generated_python=False` | 이를 자동 Python 생성 구현으로 보고하지 않음. 생성 Python은 분해 저장소의 기존 VM/생성·등록 기반 유지 |
| 분해 `orchestrator/python_skill_program.py::evaluate`, `step_program`; `python_skill_vm.py` | 제한 Python으로 ASAP 행동을 제안. 기본 문맥은 부품·자세·지지·경로 | 언어 제한·유한 실행·검사 패턴 유지. 로봇 행동 문맥/반환값은 새 버전으로 구현해야 함 |
| 분해 `orchestrator/lesson_memory.py`, `explicit_action_memory.py`, `knowledge_store.py` | 근거/교훈/관측/버전 지식 저장 | 기존 저장소를 기반으로 쓰되 verifier 도메인·실행 수락·실제 로봇 조건 보완. 기존 기록은 불변 |

## 4. 새로 만들어야 하는 분해 기능

아래 명칭은 기능 설명이며 **현재 존재하는 함수명이 아니다**.

1. **조립 상태 입력/장면 생성:** 부품별 asset, 단위·좌표, 질량/마찰, mating 관계, fixture, 제거/잔여 ID, 목표 트레이.
2. **조립체 내부 파지 검증:** 열린/닫힌 손가락과 주변 부품 간섭, 대상에 대한 실제 파지 유지 확인.
3. **분리 경로 실행:** 접촉 가능한 구간과 자유공간 운반을 구분. 초기에는 검증 가능한 단순 직선/제한된 구간부터 지원하고 unsupported 조인트를 명시.
4. **분리·잔여 안정성 판정:** 상대 위치·관계 해제·접촉·속도·기울기 등 실제 근거 사용. 임의로 constraint를 끄거나 부품을 숨겨 성공 처리하지 않음.
5. **분해 복구:** 재관측·안전한 후퇴·재파지/방향 변경 후보. 이미 움직인 장면에서 가능한 행동인지 확인; 리셋은 복구가 아님.
6. **로봇 행동/학습 연결:** 상태 revision과 명령 ID, domain별 경험·교훈·스킬, 현재 회차/동결 장기 지식 분리.
7. **순서 제안 출력 변환:** 선택된 계획 체인만 내보내는 계약. 조립 실행 명령을 출력하지 않음.

## 5. 이식 방법과 검증 순서

- 조립 저장소를 실시간으로 import하거나 수정하지 않는다. 다음 구현 시 필요한 최소 파일/asset을 출처·hash·라이선스와 함께 **분해 저장소 내 독립 모듈**로 고정해 가져오는 방법을 우선한다. 저장소 간 범용 공통 패키지 분리는 실제 중복 유지 비용이 확인된 뒤 검토한다.
- 조립은 Python 3.11 계열 프로젝트, ASAP worker는 Python 3.8 환경이다. 현재 orchestrator 환경까지 세 환경의 의존성을 조사하고 새 로봇 worker와 명시 IPC 계약을 사용한다. ASAP에 MuJoCo/현대 패키지를 몰아 설치하거나 다른 프로젝트 환경을 바꾸지 않는다.
- 로봇 제어는 로봇 worker, LLM/예산/동결 지식은 orchestrator, 부품 수준 검증은 기존 ASAP worker가 맡는 구성을 목표로 한다. 환경이나 프로세스를 새로 만들지는 않았다.
- 순서: asset/TCP 동일성 → 빈 팔 충돌/제어 → 자유 부품 파지·운반 → 조립 부품 잡기·분리 → LLM/학습. 앞 단계 성공을 뒤 단계 성공으로 보고하지 않는다.
- 기존 조립 성과를 회귀 근거로 보관하되 이식한 버전은 자체 테스트를 통과해야 한다. 이식 이후 변경은 분해 저장소 안에서만 한다.

이번 검토로 “그대로 가져올 수 있음”이 확정된 것은 설계 기반의 존재다. **새 장면에서의 호환성·분해 성능은 아직 검증하지 않았다.**
