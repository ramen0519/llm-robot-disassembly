# Astra 순위와 해결 사례를 쓰는 분해 실행기

## 시작

일반 실행 진입점은 `tools/run_sequence_native_experiment.py`다. Windows에서 기존 Python으로 실행하면 전용 WSL의 Python 3.8과 공식 ASAP로 연결한다. 새 프로필의 기본 모델은 `configs/astra_cases_v1.json`의 `gpt-6-astra`다. 기존 `--run-dir` 인자를 쓰는 호출은 과거 프로필을 그대로 사용한다.

아래 PowerShell 명령은 F 작업본에서 실행한다. 각 실행에는 새 출력 폴더 이름을 쓴다. `--mode prepare`는 기하·이미지만 생성하고 `--mode fixture`는 명시한 시험 순위로 실제 RedMax까지 실행한다.

```powershell
Set-Location F:/disassembly_research/design_v1_20261122
$py = 'C:/dev/disassembly_codex_ready_v2/.venv-orchestrator/Scripts/python.exe'
& $py -B tools/run_sequence_native_experiment.py --mode fixture --input /opt/asap-research/ASAP/assets/beam_assembly/original --assembly official_beam --family beam --unit cm --base 6 --run evidence/my_beam --max-attempts 16 --max-path 96 --max-stability 64 --seconds 180
```

STEP은 `--input 모델.stp`로 지정한다. 부품별 OBJ에는 단위가 필요하다. ASAP `config.json`이 있으면 완성 배치인 `final_state`를 읽는다. 이미 조립 좌표에 있는 OBJ에는 `--assembled-obj --unit mm`를 명시한다. 개별 로컬 좌표의 OBJ는 아래 구조의 JSON과 같은 폴더에 둔다. 행렬의 병진 성분은 mm다. 실제 물리 입력은 상세 로컬 메시와 완성 배치를 cm로 환산한다.

```json
{"unit":"mm","parts":[{"part_id":"pin","geometry_id":"pin_shape","mesh":"pin.obj","cad_name":"Pin","mesh_coordinate_frame":"part_local","assembly_matrix_mm":[[1,0,0,10],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}]}
```

위 한 행은 형식 예시다. 실행 조립품에는 두 개 이상의 부품이 필요하다. STEP에서는 XCAF 정의 하나를 여러 발생 인스턴스가 공유할 수 있다. 조립품 계층을 따라 배치를 누적하고 한 부품의 여러 솔리드를 함께 메시화한다. 면이 없는 참조는 `source/occurrences.json`에 별도 기록한다. OBJ의 원래 부품 ID와 ASAP에 필요한 숫자 ID의 대응은 `part_mapping.json`에 있다. 부피는 닫히고 일관되게 배향된 메시에서 산출한다. 확인할 수 없는 값은 null이다.

## 한 회차에서 일어나는 일

1. Python이 상세 메시에서 완성 위치의 크기·중심·부피와 삼각형 표면 간 거리를 계산한다. 초기 근접 허용값은 0.1 mm다. +Z 상면과 (+X,+Y,+Z) 대각선의 직교투영 PNG를 생성한다.
2. 첫 Responses 요청에는 정적 기하, 유효 후보, Python 상태, PNG 두 장이 들어간다. 후속 요청은 `previous_response_id`와 최신 상태·실패·짧은 지침을 쓴다. 계획 대화는 서버 저장을 사용한다. 지침 생성은 별도의 `store=false` 요청이다. 근접은 체결·지지력·원인의 증명이 아니다.
3. 순위 응답의 요청 ID, 상태 ID, 전체 후보 순열, 인용 사례 ID를 검사한다. 원래 DFS가 순서에 따라 자세·단일 힘 방향·지지·안정성을 검사한다. 고정 베이스는 제거 후보에서 제외한다. 백트래킹 때 베이스 외 미시도 자식이 없는 프레임을 반환한다. 마지막 두 부품은 원래 질량·베이스 규칙으로 고르고 실제 검사를 수행한다.
4. 각 방향 검사와 지지 검사의 종료 행렬을 다음 reset 전에 복제한다. 한 부품에 허용된 자세·방향·지지 검사를 마치고도 실패한 경계에서 사건을 연다. 마지막 원시 실패와 일치하는 상태만 대표 장면이 된다. 대상이 사라진 안정성 장면에서 그 부품의 직전 위치는 카메라 중심에만 쓰인다. 소요 시간 초과·전송 오류·근거 누락을 충돌로 분류하지 않는다.
5. 같은 시작 노드에서 연속 채택된 경로로 원래 부품 제거가 통과하면 국소 해결이다. Astra가 실패·선행 전이·목표 제거를 인용하는 조건부 순위 지침을 만들면 경험과 함께 등록한다. 지침이 skip이거나 유효하지 않으면 원자료만 남는다. 근거 ID와 관측 연결을 검사하지만 자연어 문장의 과학적 인과성까지 자동 증명하지 않는다.
6. 검색은 실제 검사 단계와 결과가 일치하는 동결 사례만 대상으로 한다. 768×768 대상 중심 PNG 전체를 224×224로 줄여 고정 DINOv2 ViT-S/14의 정규화된 384차원 CLS 특징을 계산한다. 코사인 유사도로 최대 세 사례를 반환한다. 모델 입력에는 적용 조건과 전략을 싣고 원시 벡터와 과거 이미지는 첨부하지 않는다. 조회·전달·모델 인용은 별개다.
7. 채택한 가지의 행렬에서 영상과 핵심 그림을 생성한다. 확인된 완주에는 잔여 베이스와 역순 조립 후보가 있다. 부분 결과에는 검증된 부분 체인만 있다. `robot_verified=false`다.

## 기억과 실행 예산

회차 시작에 사례 ID·원본 해시를 `frozen_index.json`에 고정한다. 새 연구 라이브러리는 `.cases/`이며, 같은 사건은 한 번만 등록한다. 이미 등록된 이전 세대 Python 스킬·교훈 카드는 기존 실행기에서 유지한다. 새 프로필은 해당 카드를 자동으로 해결 사례로 변환하지 않는다. 시험용 라이브러리와 운영 라이브러리는 데이터베이스 자체에서 구분한다.

평가에는 `--evaluation --family 제품계열 --exclude-family 다른평가계열`을 사용한다. 평가용 신규 지침은 운영 인덱스에 들어가지 않는다. 사전 고정한 모집단은 `--frozen-index 이전회차/frozen_index.json`으로 지정한다. paired OFF는 `--memory-off`를 더한다. 새 대화와 동일한 물리·시간 예산을 사용해야 한다.

`--max-calls`와 `--max-usd`는 순위·교정·지침을 합친 회차 상한이다. 준비와 도움말에는 키가 필요 없다. 유료 모드에는 양수 상한이 필요하다. 네트워크 자동 재전송은 없으며 출력 계약 교정은 기본적으로 최초 포함 두 시도까지다. 오류 후 독립 대화나 휴리스틱으로 계속하지 않는다. 토큰·예약·실제 응답 사용량을 분리한다. 금액은 설정 단가로 계산한 프로그램 추정치다.

## 파일을 확인하는 방법

| 확인 목적 | 회차 폴더 안의 파일 |
|---|---|
| 최초 입력·원본 배치 | `input/geometry.json`, `input/part_mapping.json`, `input/surface_distances.json`, `input/top.png`, `input/oblique.png` |
| 코드·입력·조건의 재현 | `settings.json`, `code_versions.json`, `physics.json`, `native_input.json` |
| 실제 요청·조건별 실패와 수락 | `api.jsonl`, `events.jsonl`, `attempts.json`, `checks/` |
| 문제 장면·임베딩·동결 목록·지침 인용 | `incidents.json`, `checks/*_target.png`, `checks/*_embedding.npy`, `frozen_index.json`, `retrieval.json` |
| 결과·영상·조립 전달 | `result.json`, `media.json`, `execution.mp4`, `execution_frames.json`, `export_v1.json` |

최종 영상은 실제 실행에서 표본화한 월드 행렬을 다시 그린 것이다. 단계 사이에는 장면 전환이 있다. 동적 프레임이 없는 검사에는 실제 정지 장면만 남긴다. F의 입력 파일과 해시가 같은 바이트만 Linux RAM에 잠시 올려 RedMax가 자체 SDF를 만들게 한다. 영구 파일은 F에 기록한다.

```powershell
& $py -B tools/check_disassembly_export.py evidence/beam_delivery/export_v1.json --zip evidence/another_beam_export.zip
```

전달 도구는 압축된 파일을 직접 읽어 해시를 다시 계산한다. `export_v1.json`과 같은 폴더에 기록된 원본 좌표, 부품 메시와 거리표, 분기별 검사·대표 실패 이미지·영상 프레임까지 패키지에 포함한다. 연구실 데이터가 들어 있는 패키지는 F에 보관한다.

## 이 장비에서 사용한 의존성

공식 ASAP commit `3727e716b57e0b508c4e9ecd9fefa6953e8f72da`, WSL `ASAP-Ubuntu`, ASAP Python 3.8.0, 별도 STEP helper Python 3.10.12를 사용했다. 로컬 `.runtime/`의 OpenCascade 휠은 `cadquery-ocp-novtk 7.9.3.1.1`이다. DINOv2 모델 revision은 `ed25f3a31f01632728cabb09d1542f84ab7b0056`이고 safetensors SHA-256은 `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`이다. 사용한 Transformers는 4.39.3, Torch는 1.13.1+cpu다. 기존 연구 환경의 패키지는 교체하지 않았다. 의존성이 빠진 새 작업본에서만 아래를 사용한다.

```bash
/usr/bin/python3 -B tools/setup_design_assets.py ocp
/opt/asap-research/envs/asap/bin/python -B tools/setup_design_assets.py vision
```
