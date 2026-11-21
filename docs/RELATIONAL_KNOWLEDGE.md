# 형상·주변 관계를 연결한 경험 / 교훈 / Python 스킬

2026-11-19. 기존 자료를 새 구조로 변환하고 검색·모델 입력을 연결했다. 원본 관측, 교훈 문장, 생성 Python, 검증·등록 상태는 보존한다. 이번 변경은 새로운 모델 학습이나 물리검증 결과가 아니다.

## 무엇이 달라졌나

| 구분 | 새 구조와 역할 |
|---|---|
| 경험 | 당시 제거 대상·잔여 부품·관측 결과에, 원본 CAD에서 확인 가능한 크기·비율·중심·접촉 이웃·검사 조건을 연결한다. |
| 교훈 | 기존 문장을 보존하고, 실제 인용한 경험을 고유 ID로 연결한다. 다른 조건에서 항상 성립하는 법칙으로 승격하지 않는다. |
| Python 스킬 | 원래 코드·작성 모델·해시·검증/등록 상태를 보존하며 연결된 교훈·경험을 함께 조회한다. 검색되었다는 이유로 실행 권한을 주지 않는다. |
| 카드 | 위 세 자료를 모델이 함께 읽을 수 있게 보여주는 표현이다. 독립된 새 실험이나 추가 지식이 아니다. |

형상에는 `extents_xyz`, `ratio_xyz`, `ratio_sorted`, 중심, 단위·좌표계가 들어간다. 축에 정렬된 경계상자의 비율이며 정확한 표면 모양이나 임의 회전에 대한 불변 표현은 아니다. 단위가 없으면 절대 크기를 같은 단위의 값처럼 비교하지 않는다.

주변 관계에는 기록된 접촉 이웃, 이웃 크기·비율, 대상과 이웃의 중심 위치 차이, 연결 성분 수 및 대상 제거 시 그래프 변화가 들어간다. 이는 **기준 CAD의 정적 접촉 그래프**다. 실제 이동 중 접촉 위치·법선·면적, 체결 관계, 누가 누구를 지지하거나 경로를 막는지는 원본에 없으면 `null`이다. 정적 그래프 변화만으로 붕괴·충돌을 판정하지 않는다.

전체 회차의 최대·누적 지지 점수와 단계별 관측을 구분한다. 한 회차에서 추출한 여러 단계는 같은 `evidence_group_id`를 사용하므로 독립된 여러 성공 사례처럼 계산하지 않는다. 원본 PASS, 상태 전이 수락, 시간 초과도 따로 보존한다.

## 검색과 모델 전달

1. 현재 상태의 **각 제거 후보**에 대해 같은 형상·주변 관계·조건 표현을 만든다.
2. Python이 형상 유사성, 주변 관계 유사성, 상황/검사 조건의 세 채널을 각각 검색한다. 순위 결합에는 RRF를 사용한다. 부품 번호나 제품 번호 자체는 유사성 특징이 아니다.
3. 같은 회차의 중복 결과가 상위를 차지하지 않도록 경험 묶음을 분산한다. 연결된 교훈·스킬을 찾아 함께 읽는다. 누락과 조건 차이를 표시한다.
4. 모델에는 작은 요약만 전달한다. 원본 전체 로그·중복 출처는 로컬에 보존하고, 선택된 원문 교훈과 Python은 그대로 전달한다. Python을 중간에서 잘라 전달하지 않는다. 이웃·근거가 생략되면 수와 이유를 표시한다.
5. 순서 경로에서는 모델이 제공된 카드를 참고하거나 사용하지 않을 수 있고, `relational_card_ids`로 인용한다. `agent_tools_v1` 경로에서는 모델이 `shape:`, `relations:`, `situation:` 검색 및 카드 읽기를 요청할 수 있다. 검색 결과가 모든 라이브러리를 읽었다는 뜻은 아니다.
6. 모델은 부품 순서를 제안한다. 기존 실행기는 원래 등록된 스킬의 권한과 물리검증 계약을 계속 적용한다. 관련 카드의 검색·열람은 새 코드 실행이나 물리 PASS가 아니다.

## 전체 보존본과 실행용 자료

현재 변환본은 `runs/relational_knowledge_migration_20261119/migration_v6/`다. v5까지의 결과도 그대로 보존했다. v6는 이전에 보관만 되고 실행용에서 빠졌던 등록 스킬 두 개의 출처 연결을 보완한다.

| 파일 | 용도 |
|---|---|
| `library.json` | 식별한 기존 지식 자료 전체를 보존한 감사용 변환본. 과거 평가·fixture·미분류 자료를 포함하므로 통째로 평가 입력에 사용하지 않는다. |
| `training_runtime_library.json` | 출처가 확인된 학습 경험 및 그 경험만 인용하는 교훈·스킬의 동결본. 현재 평가 제품과 연결된 파생 지식도 실행 시 제외한다. |
| `inventory.json` | 조사한 파일별 가져오기/제외/지원하지 않는 형식의 상태와 원본 해시. |
| `preservation.json` | v6가 읽은 기존 지식·기하·등록 근거 1,143파일의 보존 검증. |
| `summary.json` | 변환 수·미해결 연결·미실행 범위. |

원본의 고유 경험 2,277건·교훈 89건·스킬 기록 134건을 변환했다. 단계별 파생 경험 483건을 더해 카드 2,983개다. **서로 독립된 물리실험 2,983회를 뜻하지 않는다.** 저장된 버전·후보·fixture도 포함한다.

경험 카드 2,760개 중 형상 수치가 있는 것은 2,297개, 정적 접촉 근거가 있는 것은 642개다. 1,374개의 인용 연결은 출처 부족/모호성 때문에 미해결로 남겼다. 연결을 임의로 맞추지 않는다. 실행용 동결본은 경험 642개·교훈 52개·스킬 10개, 총 704개다. **스킬 10개 중 8개는 미등록 참조 후보, 2개는 기존 `REGISTERED_SCOPED` 스킬**이다. 신규 생성·등록은 아니다. 기존 663개 실행 카드는 완전히 동일하다.

추가로 연결한 39개 경험은 두 스킬을 생성할 때 사용한 실제 익명 관측이다. 생성 packet의 별칭을 원본 관측·CAD·검증 로그에 대조해서 연결했으며, 전체 보존본의 카드 수는 증가하지 않았다. 스킬이 인용한 경험과 생성 때 본 전체 39개 경험/5제품을 구분한다. 이 두 생성 기록에는 별도의 교훈 파일이 없으므로 교훈을 임의로 만들지 않았다.

학습 60회차의 원본과 같은 자료를 인용한 기록을 명시적으로 학습 출처로 인정했다. 과거 다른 자료의 train/eval 역할이 확실하지 않으면 `unclassified`로 보존하고 새 실행용 검색에서는 제외한다. 원본 수치를 보강했다고 이전에 모델이 그 정보를 보았던 것으로 소급하지 않는다.

## 사용 및 구현 위치

기존 실행 명령에 다음 옵션을 명시하면 새 자료를 사용한다. 옵션을 생략한 기존 baseline/과거 경로는 유지한다. 이 옵션 자체는 API 사용 승인이 아니다.

```powershell
--relational-knowledge runs/relational_knowledge_migration_20261119/migration_v6/training_runtime_library.json
```

- `tools/run_sequence_native_experiment.py`: 순서 전용 경로. `SequenceRankService`가 현재 후보별 검색 결과를 기존 provider 입력에 결합한다.
- `tools/run_hierarchical_episode.py`, `tools/run_generated_skills.py`: `agent_tools_v1`과 함께 사용할 때 지식 검색/읽기를 연결한다.
- `orchestrator/knowledge_import.py`: `discover_sources`, `import_sources` — 기존 형식·출처 읽기.
- `tools/migrate_relational_knowledge.py`: `Migrator.experience_context`, `Migrator.convert`, `resolve_versions` — 상태·기하·인용 연결 및 동결본 생성.
- `orchestrator/relational_knowledge.py`: `safe_path`, `context_from_geometry`, `make_card`, `freeze_library`, `load_library`.
- `orchestrator/relational_retrieval.py`: `search_cards` — 세 채널 검색, 근거 묶음 분산, 누락·조건 차이 표시.
- `orchestrator/relational_runtime.py`: `RelationalRuntime.search/read/packet`, `compact_card` — 제한된 입력 크기의 실제 모델 전달.
- `orchestrator/knowledge_tools.py`, `knowledge_retrieval.py`, `sequence_ranker.py`: 기존 실행 입력과 조회 계약 연결.
- `orchestrator/hierarchical_service.py`, `generated_skill_service.py`: 기존 서비스에 선택적 동결 라이브러리 전달. 로봇 실행 권한 추가 없음.

변환 재실행은 반드시 **새 출력 폴더**를 지정한다. 기존 결과를 덮어쓰지 않는다.

```powershell
& "./.venv-orchestrator/Scripts/python.exe" -X utf8 -B tools/migrate_relational_knowledge.py --output runs/<새_출력_폴더>
```

## 확인한 범위와 남은 범위

### 기존 등록 Python 두 개의 순서 실행 연결

새 관계 검색과 현재 순서 실행기를 함께 쓰려면 `tools/run_sequence_native_experiment.py`에 두 옵션을 명시한다. 아래는 옵션 조각이며 실행/API 승인 명령이 아니다.

```powershell
--relational-knowledge runs/relational_knowledge_migration_20261119/migration_v6/training_runtime_library.json
--scored-skills runs/registered_skill_integration_20261119/frozen_bridges.json
```

원래 레지스트리와 Python은 변경하지 않는다. `scored_sequence_bridge.py`가 기존 수치 점수 출력을 현재 후보 전체 순서로 연결한다. 잔여 전체 부품(고정 베이스 포함)으로 원래 특징을 계산하고 원래 제한 VM을 실행한 뒤, 베이스만 제거 후보에서 제외한다. 동점 처리도 원래 scorer의 규칙을 유지한다. 물리 PASS를 재사용하거나 후보를 잘라내지 않는다.

`scored_sequence_runtime.py`는 출처가 확인된 관계 카드와 명시적으로 동결한 등록 스킬이 모두 맞을 때만 검색 결과를 실행 선택 목록에 넣는다. 모델은 `direct` 또는 제공된 스킬을 선택한다. 스킬이 있다는 이유로 선택을 강제하지 않는다. 현재 상태의 접촉 정보가 미관측이면 실행할 수 없는 스킬은 제외 사유를 남기고 직접 순위 판단은 유지한다.

원문 코드는 `sequence_skills.program.source`에 한 번만 온전히 넣고 관계 카드는 같은 코드를 가리킨다. 등록 스킬 참조에 입력 공간을 예약하되, 예산이 부족하면 명시적으로 거절한다. 출력 ID·상태·코드·등록 원본의 해시를 확인하며, 보류/거절도 선택 ID와 실행 진단을 남긴다.

과거 두 스킬의 실제 검증 범위는 `path_only_sequence_proposal/v1`이다. 이번 연결은 **순서 계산의 호환성 검사**이며 `sequence_native_v1`의 새 물리검증/성능 검증은 `NOT_RUN`이다. 신규 로봇 스킬도 아니다. [통합 검증 기록](../runs/registered_skill_integration_20261119/REPORT.md)을 확인한다.

단위/기존 계약 회귀와 실제 저장 CAD·지식을 사용한 모델 입력 구성은 API 없이 확인한다. 최종 명령·로그·결과는 [이번 작업 보고서](../runs/relational_knowledge_migration_20261119/REPORT.md)에 기록한다.

이번에 기존 교훈의 의미나 Python 알고리즘을 모델로 다시 생성하지 않았다. 새 형상 정보가 연결되었어도 기존 코드가 그 필드를 자동으로 사용하게 되는 것은 아니다. 새로운 지식 생성·등록, 실제 모델의 적절한 선택, 다른 조립품의 성능 개선은 별도 검증이 남아 있다. 원본에 없는 동적 물리 근거를 복원하거나 MuJoCo/로봇 기능을 구현한 변경도 아니다.
