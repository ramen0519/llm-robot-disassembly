# LLM Robot Disassembly Research

LLM이 부품의 **시도 우선순위**를 정하고, 공식 ASAP의 DFS·자세·방향·지지 탐색과 RedMax 물리검증으로 분해 계획을 확인하는 연구 프로토타입입니다. 실제 관측에서 경험·교훈·제한된 Python 스킬을 만들고 다음 작업에서 재사용하는 효과를 연구합니다.

**공개 시점: 2026-11-21. 이 저장소는 코드 검토용 연구 스냅샷입니다.** 원본 실험 폴더와 운영 환경은 보존했습니다. 다운로드만으로 모든 과거 실험을 재현하는 배포 패키지는 아닙니다. 파일 이름에 `robot`이 있지만 UR3e/2F-85 로봇 분해·실물 실행은 아직 구현 완료 범위가 아닙니다.

## 현재 흐름

1. 부품별 CAD, 조립 위치와 고정 조건을 입력합니다.
2. 형상·주변 관계·검사 조건으로 관련 경험을 검색하고 연결된 교훈·스킬을 제공합니다.
3. LLM은 전체 후보 우선순위를 제시하거나 허가된 순서 스킬을 선택합니다.
4. ASAP이 DFS와 자세·방향·지지 탐색을 수행하고 RedMax가 부품 수준 물리를 검사합니다.
5. 실제 수락된 연속 분해 체인과 실패·시간 초과·비용을 기록합니다.
6. 별도 학습 단계에서 교훈과 Python 후보를 생성·검증·등록합니다. 평가 전 라이브러리는 동결합니다.

LLM이 모든 물리 판단을 대신하지 않습니다. 마지막 두 부품 선택에는 원래 ASAP의 질량/베이스 규칙이 남아 있습니다. 순서 전용 공식 경로는 여섯 축 방향의 단일 힘 시도를 사용하며 BFS 경로 생성이나 중간 방향 전환을 연결한 것으로 주장하지 않습니다. 과거의 명시 행동·경로 복구 코드는 별도로 보존되어 있습니다.

## 어디부터 읽을까

| 목적 | 파일/폴더 |
|---|---|
| 최신 공개 범위와 한계 | [구현 상태](IMPLEMENTATION_STATUS.md), [공개 검증](PUBLICATION_CHECKS.md) |
| 순서 계획의 역할 분담 | [순서 전용 프로필](docs/SEQUENCE_NATIVE_PROFILE.md) |
| 경험·교훈·스킬과 검색 | [관계 기반 지식 설계](docs/RELATIONAL_KNOWLEDGE.md) |
| 모델 호출·검색·제한 Python 실행 | `orchestrator/` |
| ASAP 연동·검증·관측 기록 | `worker/` |
| 실행 및 자료 준비 도구 | `tools/` |
| 계약·회귀 테스트 | `tests/` |
| 실제 생성된 두 스킬 원문 | [Luna 스킬](runs/scored_order_skill_pilot_20261111/registry/skill.json), [Astra 스킬](runs/scored_order_astra_author_20261112/registry/skill.json) |
| 로봇 확장 설계(미구현) | [로봇 분해 설계](docs/ROBOT_DISASSEMBLY_DESIGN.md) |
| 최근 배터리팩 진단 | [접촉·안정성 진단 요약](reports/BATTERY_PACK_DIAGNOSIS.md) |

과거 문서는 작성 당시 상태를 설명합니다. 특히 순서 프로필의 초기 '실제 생성 스킬 없음'은 11월 10일 상태이고, 이후 생성된 두 등록 스킬 및 11월 19일 통합과 구분해야 합니다. 현재 상태는 이 README와 IMPLEMENTATION_STATUS를 먼저 확인하세요.

## 확인한 성과와 제한

- 공식 Beam `dfs + heur-out` baseline: 전체 분해 성공. [실행 기록](runs/m1_wsl_20261016/report.md)
- 공식 16부품 00335: 순서 전용 Sol + 원래 ASAP으로 15개 제거 성공. [실행 기록](runs/sequence_native_00335_live_20261110/REPORT.md)
- 새 관계 기반 검색과 등록 스킬의 연결: 저장 입력·주입 응답을 사용하는 오프라인 통합 검증이 존재합니다. [검증 보고서](runs/registered_skill_integration_20261119/REPORT.md)
- 로컬 동결 자료는 경험 642개·교훈 52개·스킬 카드 10개(등록 2개, 미등록 참조 8개)입니다. **전체 704개 카드 원문은 이번 공개에 포함하지 않았습니다.** 등록 스킬 2개는 코드와 검증 범위를 포함합니다.
- 한 제품 성공이나 등록 사실은 메모리·스킬의 일반적 성능 향상, ASAP보다 우수함, 실물 성공의 증거가 아닙니다.

## API 없이 코드 검사

현대 orchestrator와 ASAP worker는 별도 Python 환경입니다. 기존 확인 환경은 orchestrator Python 3.12, ASAP Python 3.8입니다. `requirements-orchestrator.lock.txt`는 당시 환경 기록이며 새 PC에서 설치 재현을 이번 공개 과정에서 검증하지 않았습니다.

저장소 루트에서 아래 핵심 테스트는 표준 라이브러리만으로 실행할 수 있습니다. 모델·CAD·RedMax를 호출하지 않습니다.

```bash
python -B -m unittest tests.test_relational_knowledge tests.test_relational_retrieval tests.test_relational_migration tests.test_scored_order_skill.ScoredOrderSkillTests.test_id_independent_numeric_program_and_complete_ties tests.test_scored_order_skill.ScoredOrderSkillTests.test_no_io_no_part_ids_no_shapes_leak -v
```

공개 복사본에서 39개 검사를 통과했습니다. 전체 테스트에는 비공개 CAD/과거 run이나 Windows의 특정 경로를 필요로 하는 항목이 있습니다. F드라이브에서 더 넓은 59개 검사를 실행했을 때 49개 통과·10개 오류였으며, 오류는 기존 C드라이브 전용 경로 제한 때문입니다. 이를 우회하거나 실행 코드를 바꾸지 않았습니다.

## 실제 물리 실험 환경

ASAP은 원본 저장소와 commit을 고정한 Git submodule로 연결합니다.

```bash
git submodule update --init --recursive external/ASAP
```

원본 commit: `3727e716b57e0b508c4e9ecd9fefa6953e8f72da`. 빌드와 공식 데이터·GNN 가중치는 [ASAP 공식 안내](https://github.com/yunshengtian/ASAP)를 따릅니다. 모델 가중치가 없으면 GNN 실행이라고 보고할 수 없습니다.

`tools/run_sequence_native_experiment.py`의 `--mode prepare`는 입력 준비, `--mode fixture`는 주입된 순위 + 실제 물리검증, `--mode llm`은 실제 API 사용입니다. Windows 실행기는 기존 WSL 이름·Linux Python 경로·프로젝트 경로를 가정하므로 **다른 PC에서는 먼저 경로 설정과 제한을 검토해야 합니다.** 현재 공개본을 범용 설치 완료 제품으로 설명하지 않습니다.

유료 실행에는 별도의 키·모델·횟수·비용·시간 상한 설정이 필요합니다. 키와 인증 파일은 포함하지 않습니다. 코드에 남은 모델 ID·과거 가격은 역사적 실험 설정이므로 현재 가용성과 단가를 별도로 확인해야 합니다.

## 공개하지 않은 자료

원본 CAD/STEP/OBJ, GNN 가중치·빌드 엔진, 가상환경, 전체 경험 원문, API 원시 요청/응답, 인증·키, 사용자 첨부 파일, 대용량 경로·시각화 로그를 제외했습니다. 따라서 문서에 있는 일부 과거 run 링크는 이 공개본에 없고 로컬 증거 위치를 나타냅니다.

[제3자 자료와 라이선스](THIRD_PARTY.md) · [공개 파일 명세](PUBLICATION_MANIFEST.json)
