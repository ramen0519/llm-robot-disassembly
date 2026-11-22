# 분해 설계 v1 통합 검증 보고

## 결론

분해 기본 실행기에서 CAD 입력, Astra의 대화형 순위 계약, 공식 ASAP DFS·RedMax, 실패 관측, 해결 사건과 DINOv2 검색, 영상 및 조립 전달을 연결했다. 실제 부품 물리와 실제 이미지 검색은 실행했다. 실제 Astra는 계정에서 모델 ID를 조회했다. 생성 호출은 0회다. 실제 Astra 작성 지침의 등록·자율 재사용·성능 효과는 아직 실험하지 않았다.

## 실제로 확인한 단위

| 대상 | 순위·지침 출처 | 물리 결과 | 사건·기억 |
|---|---|---|---|
| 공식 5부품 Beam 기준선 | 원래 DFS + heur-out | 1 → 0 → 3 → 2. 베이스 6, 4개 제거. 원본 통계 93.93초 | 기존 비교 기준 |
| 공식 Beam 통합 실행 | 고정 순위 | 0 → 1 → 2 → 3. 경로 24회·안정성 4회. 추가 지지 최대 0, 합계 0 | 목표 실패 사건 0 |
| 연구실 배터리팩 | 진단용으로 덮개 12를 선택, 베이스 0 | 허용한 한 부품 시도에서 경로 6회·안정성 1회. 경로 통과, 안정성 실패, 제거 0 | 원시 안정성 반환 부품 8·2. 대상이 빠진 실제 잔여 12개를 촬영·인코딩. 미해결 1, 등록 0 |
| 컵·삽입물·뚜껑 진단 | 고정 순위 | 실제 RedMax가 1 → 2를 받아들임 | 예상한 내부 부품 실패는 관측되지 않음. 등록 0 |
| 트레이와 기둥 진단 | 고정 순위 | 목표 1 실패 → 2 채택 → 1 채택 → 3 채택. 24회 경로·5회 안정성. 추가 지지 [0,1,0] | 실제 국소 해결 1건. 고정 순위 실행은 지침 생성 skip |
| 같은 트레이 수집·조회 두 회차 | **모의 Responses 응답**, 실제 물리·PNG·DINOv2 | 회차별 3개 연속 제거. 원시 검사와 대상 부재를 검증 | 첫 회차 사례 등록 → 다음 회차 동결 목록 → 실제 이미지 질의 반환 → 모의 응답의 인용. 회차 중 생성한 사례는 자기 조회에서 제외 |
| 통신 오류 주입 | 모의 연결 timeout | 한 번 시도 뒤 종료. 경로 0·안정성 0·채택 0 | 등록 0, 전체 조립 후보 null |

트레이는 메커니즘 검증을 위해 제작한 네 부품 진단 자료다. 모의 응답이 선택한 순서와 인용을 실제 Astra의 판단으로 집계하지 않았다. 같은 제품의 후속 조회는 새 제품 일반화 평가가 아니다. 실무용 연구 라이브러리에는 이 시험의 모의 지침이 들어가지 않는다.

## 입력과 시각 자료

실제 배터리 STEP의 선언 단위는 mm다. 발생 인스턴스 13개가 5개 형상 정의를 공유한다. 홀더 4개와 배터리 셀 6개는 독립적인 위치를 가진다. 면 없는 보조 복합체 2개를 별도 기록했다. 전체 78쌍의 삼각형 표면 거리를 계산했고, 0.1 mm 이내가 22쌍이었다. 기존 별도 OpenCascade 입력 분석과 13개 부품의 bbox 크기·중심 차이는 최대 0.000000072 mm였다. 메시 부피를 신뢰하기 어려운 부품 0과 11은 null로 남겼다. 기존 배터리 실행에서 적용했던 바닥 이동을 이번 입력에 자동으로 더하지 않았다.

- 배터리 [실제 기하](../evidence/battery_transfer_final/input/geometry.json) · [부품·공유 형상 매핑](../evidence/battery_transfer_final/input/part_mapping.json) · [상면](../evidence/battery_transfer_final/input/top.png) · [대각선](../evidence/battery_transfer_final/input/oblique.png) · [실제 실패 PNG](../evidence/battery_transfer_final/checks/check_00007_target.png) · [부분 실행 영상](../evidence/battery_transfer_final/execution.mp4).
- 공식 Beam [완주 영상](../evidence/beam_transfer_final/execution.mp4) · [핵심 프레임](../evidence/beam_transfer_final/execution_frame_000.png) · [연속 경로](../evidence/beam_transfer_final/result.json) · [export v1](../evidence/beam_transfer_final/export_v1.json) · [전달 ZIP](../evidence/beam_transfer_complete.zip).
- 두 회차 시험 [생성·재사용 대조](../evidence/contract_pair_transfer/verification.json) · [첫 회차 사건](../evidence/contract_pair_transfer/collect/incidents.json) · [두 번째 회차 조회·인용](../evidence/contract_pair_transfer/query/retrieval.json) · [실패부터 해결까지의 클립 목록](../evidence/contract_pair_transfer/query/media.json).

공식 Beam 전달 ZIP에는 명세를 포함해 86개 파일이 들어 있다. 트레이의 [실패·해결 전달 ZIP](../evidence/tray_case_transfer.zip)은 명세 포함 90개 파일을 담는다. 검사기는 압축 파일 내부를 직접 읽어 원본 SHA-256과 대조한다. 부품별 검사, 분기에서 채택되지 않은 시도, 삼각형 표면 거리표, 실패 이미지, 재생 프레임 목록도 함께 확인할 수 있다. 각 회차의 영상은 이번에 기록한 RedMax 물체 변환을 8 fps로 표시하며 시도 사이를 컷으로 잇는다.

- [Beam 실행 설정](../evidence/beam_transfer_final/settings.json) · [코드 버전과 해시](../evidence/beam_transfer_final/code_versions.json) · [RedMax·ASAP 고정 버전과 설정](../evidence/beam_transfer_final/physics.json) · [실제 입력 OBJ 해시](../evidence/beam_transfer_final/native_input.json).
- [연구실 배터리의 단계별 검사와 매체를 포함한 전달 ZIP](../evidence/battery_transfer_complete.zip) · [배터리 부품 쌍 78개의 실측 거리](../evidence/battery_transfer_final/input/surface_distances.json).
- [베이스와 트레이만 있는 초기 두 부품 실행](../evidence/two_part_terminal/result.json). 원래 ASAP의 마지막 부품 규칙과 실제 경로 6회·안정성 1회로 트레이가 제거됐다. 모델 호출은 0회였다.

## 코드와 검증 경계

| 단계 | 구현 파일·주요 동작 |
|---|---|
| T0·실행 | `tools/run_sequence_native_experiment.py` 기존 진입점, `tools/run_disassembly.py::main`, `tools/astra_cases_windows.py`, `configs/astra_cases_v1.json` |
| T1 | `caseplanner/step_xcaf.py::convert`, `geometry.py::prepare`, `images.py::Scene`. 실제 XCAF 발생·배치, FCL 삼각형 거리, 직교 깊이 렌더링 |
| T2 | `astra.py::Astra.rank/call`, `run.py::Session.rank/simulate/accept`. 이전 대화 ID·현재 상태 ID, 계약 교정·비용, 원래 native DFS 및 마지막 두 부품 검증 |
| T3 | `observe.py::Observer`, `run.py::Session.close_target/learn`, `memory.py::Incidents`. reset 전 검사 상태, 사건 경계, 시작 노드 고유 ID·시간 순서·원래 목표 검증 |
| T4 | `memory.py::Encoder/Library`. 사례 중복 방지, 원시 근거 복사·해시, 실제 이미지와 고정 모델, stage/outcome 선필터, 상위 3건, 평가 계열 제외·동결 |
| T5 | `deliver.py::finalize/video`, `tools/check_disassembly_export.py`. 연속 체인·역순 후보·근거 해시·영상 프레임 점검, 이전 동작/버린 가지 분리 |

F의 main 스냅샷은 034006a7bc0de91cf6370eff0c07e6796f6e2877이었고 미커밋 변경은 없었다. C 작업본은 Git 저장소가 아니다. 공개 시 복사한 577개 파일을 C 원본과 직전 공개본에 대조해 모두 일치했다. C의 주요 코드 폴더에 공개본보다 추가된 소스는 없었다. C에 없는 PUBLICATION_CHECKS는 F 공개본에서 읽었다. 새 작업 위치는 F의 research/astra-cases-v1 브랜치다. [보존 확인](../evidence/preservation_review.json).

최종 새 계약 **26개**, 이전 공개판의 휴대 가능 회귀 **39개**가 통과했다. 후자는 전체 과거 테스트 모음의 부분집합이다. 새 기본 실행기는 과거 C 전용 경로를 호출하지 않는다. 과거 모든 프로필의 F 이식이나 기존 공개본의 10개 경로 제한 오류 해소를 이번 결과로 주장하지 않는다. [새 테스트](../evidence/contract_tests_release.txt) · [기존 회귀](../evidence/legacy_regression.txt) · [산출물·영상·좌표 검증](../evidence/release_verification.json).

## 커밋 고정 후 재현

구현 커밋 `7c867386946e3fce64181626775e0fb82d67ecd9`에서 Windows 기본 실행기로 Beam 4/4를 다시 확인했고, 같은 소스로 트레이의 수집 및 후속 조회 두 회차도 재현했다. 회차마다 소스·설정 15개 파일의 SHA-256을 실제 Git 객체와 대조했다. 세 ZIP은 압축 내부 237개 파일까지 해시가 일치했다. Beam 자세 계산에서 trimesh의 제곱근 수치 경고가 출력됐고 실제 채택 경로 4개와 물리 근거 4세트가 생성됐다. [커밋·세 실행·소스 해시 검증](../evidence/postcommit_verification.json) · [최종 Windows 실행 결과](../evidence/committed_beam/result.json) · [동일 소스로 재현한 두 회차 기록](../evidence/committed_case_pair/verification.json)

## 실패와 미실행 기록

- 첫 STEP 읽기는 CAD 보조 객체에 삼각형이 없어 중단됐다. 원본을 유지하고 면 없는 발생 항목과 실제 형상을 구분해 재실행했다. [최초 오류](../evidence/battery_step_01/error.json).
- 첫 통합 Beam은 F 마운트에서 네이티브 SDF를 직접 쓰는 초기화가 지연되어 검증 0회로 끝났다. 원래 코드가 예외로 기록한 종료와 실제 365.28초를 보존했다. 검증기·임계값은 유지하고 정확히 같은 메시와 배치 파일을 회차 전용 Linux RAM에 복사해 재실행했다. [최초 결과](../evidence/beam_chain_01/result.json) · [실제 입력 해시](../evidence/beam_transfer_final/native_input.json).
- 계정에 등록한 키로 공식 모델 메타데이터 GET은 성공했다. 설정의 `gpt-6-astra`와 반환 ID가 일치했다. 생성 요청, 연구 데이터 전송과 실제 청구 사용량은 0이다. [모델 확인](../evidence/model_access.json).
- 실제 Astra가 자유롭게 순위를 고르고 근거 있는 지침을 생성·인용하는지, 학습 제품과 다른 제품에서 도움이 되는지, 실제 로봇 grasp·운반·조립이 가능한지는 미검증이다. 혼합 통합시험의 실제 이미지는 RedMax의 기록 행렬을 렌더링한 것이다. 모의 지침, 이전 영상이나 임의 벡터를 실운영 사례로 보고하지 않는다.

## 실행 재현과 다음 유료 범위

[시작 방법·입력 형식·설정](../docs/ASTRA_CASES_V1.md). 새 실행은 기존과 겹치지 않는 폴더에서 시작한다. 아래는 WSL 작업 루트에서 실제 사용한 명령 형식이다. `$PY`는 `/opt/asap-research/envs/asap/bin/python`이고, `$STEP`은 사용자에게서 받은 `all_in_one.stp`의 절대경로다. 재현 시 환경변수 `TMPDIR`, `MPLCONFIGDIR`, `XDG_CACHE_HOME`은 F의 `.runtime` 하위로 지정한다.

```bash
$PY -B /opt/asap-research/ASAP/plan_sequence/run_seq_plan.py --dir beam_assembly --id original --planner dfs --generator heur-out --base-part 6 --seed 0 --budget 16 --max-gripper 3 --max-pose 3 --timeout 90 --early-term --disable-save-sdf --debug 1 --log-dir /mnt/f/disassembly_research/design_v1_20261122/evidence/baseline
$PY -B tools/run_sequence_native_experiment.py --mode fixture --input /opt/asap-research/ASAP/assets/beam_assembly/original --assembly official_beam --family beam --unit cm --base 6 --run evidence/beam_transfer_final --max-attempts 16 --max-path 96 --max-stability 64 --seconds 180
$PY -B tools/run_sequence_native_experiment.py --mode fixture --input "$STEP" --assembly lab_battery --family lab_battery --base 0 --fixture-order 12 --run evidence/battery_transfer_final --max-attempts 1 --max-path 6 --max-stability 4 --seconds 120
$PY -B tests/make_support_diagnostic_cad.py evidence/diagnostic_tray_meshes
$PY -B tests/run_cases_physics_contract.py evidence/contract_pair_transfer
$PY -B tests/run_cases_error_contract.py evidence/transport_failure
$PY -B -m unittest discover -s tests -p test_astra_cases_v1.py -v
```

실제 Astra 단계의 제안은 위 트레이를 처음부터 자유롭게 선택하는 최대 두 회차다. **회차당 생성 6회·프로그램 추정 $1.50, 합계 최대 12회·$3.00**이다. 회차당 탐색 180초, 부품+자세 시도 6회·경로 36회·안정성 16회로 제한한다. 준비·미디어 시간은 별도다. 첫 회차가 검색 가능한 실운영 사례를 생성했을 때만 두 번째 회차를 실행한다. 회차 간 같은 기하를 쓰는 재사용 시연이며 성능 비교가 아니다. 현재 테스트 첫 요청의 프로그램 예약 추정은 $0.18169다. 이후 실제 문맥·토큰에 따라 금액이 달라진다.

전송 대상은 진단 트레이의 두 전체 PNG·수치 기하, 순위 후보와 관측, 필요할 때 기존에 등록한 지침의 요약이다. 공식 `https://api.openai.com/v1/responses`를 사용한다. 연구실 배터리 CAD·PNG는 이 제안에 포함하지 않는다. 다른 제품에 대해 과거 승인한 예산은 사용하지 않았다. PowerShell에서 **새 승인 뒤** 실행할 첫 명령은 다음과 같다.

```powershell
& $py -B tools/run_sequence_native_experiment.py --mode astra --input evidence/diagnostic_tray_meshes --assembled-obj --unit mm --family diagnostic_tray --assembly diagnostic_tray --base 0 --run evidence/astra_collect_01 --store evidence/astra_research_cases_01 --max-calls 6 --max-usd 1.50 --rank-attempts 2 --max-attempts 6 --max-path 36 --max-stability 16 --seconds 180
```

첫 회차에 PNG·임베딩을 가진 사례가 등록되면 동일 명령에서 출력 이름만 `evidence/astra_query_01`로 바꾸고 같은 기억 폴더를 사용한다. 두 번째 회차에 실패가 발생하지 않거나 Astra가 지침을 인용하지 않으면 그 결과를 그대로 기록한다.
