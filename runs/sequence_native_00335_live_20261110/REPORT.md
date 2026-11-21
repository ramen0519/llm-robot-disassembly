# 00335 순서 전용 Sol + 원래 ASAP 실제 검증

작성2026-11-10. 범위: 실제 모델·ASAP/RedMax를 사용하는 부품 수준 분해 계획. Mock 결과가 아니며 로봇 실행 성공이 아니다.

사용자가 제안한00335 16부품 **최대25회/$4·본60분/외부65분**에 ‘가자’로 동의한 한 회차를 실행했다. 결과는 **전체 부품 수준 분해 PASS**, 최대 연결 제거 **15/15개**다. 분기별 PASS를 합하지 않았다. 실행 중 운영 코드 변경·자동 HTTP 재시도·실패 회차 재실행·새 스킬 생성은0이다.

| 항목 | 원래 ASAP DFS+공식GNN | 이전 넓은 판단 범위 LLM | 이번 순서 전용 Sol+원래ASAP |
|---|---:|---:|---:|
| 최대 연결 제거/목표 | 15/15 | 2/15 | **15/15** |
| 전체 완료 | PASS | UNKNOWN | **PASS** |
| 경로 검사 | 234 | 127 | 162 |
| 안정성 검사 | 80 | 102 | 59 |
| Worker 시간(초) | 919.29 | 2299.77 | 1093.80 |
| 실제 API 시도 | 0 | 25 | 16 |
| 토큰 사용×설정단가 비용 | $0 | $3.6657648 | $0.689712 |

원래GNN과 동일 CAD·핵심 물리·전역 검사/시간 상한을 유지하고 이번에는 자세·6축 방향·greedy 지지·DFS 탐색도 원래ASAP에 맡겼다. 원래GNN은 이번에 재실행하지 않고 직전 실제 비교 결과를 사용했다. 같은 seed라도 GNN 샘플링 등 RNG 소비가 다를 수 있고, 모델 호출시간·지식 입력은 다르다. 이미 개발에 사용한 같은제품의1회 예비비교로, 새제품 일반화나 메모리/스킬의 인과 효과를 입증하지 않는다.

**해석:** 이전2/15에서 전체15/15로 실행 능력은 개선됐다. 원래GNN 대비 경로 검사30.8%·안정성 검사26.2% 감소했지만 Worker 시간은19.0% 증가했다. 검사1회의 비용이 같지 않으며, 검사 횟수 감소를 시간 감소로 표현하지 않는다.

- 실제 모델: gpt-6.1-sol. 정상 응답 16건, 호출 실패 0건, 미확정 요청 0건. 최종 중단 근거 `solution found`.
- 과거 관측 명시 인용 0회, 교훈 인용 0회, 이번 회차 관측 인용 15회. 인용과 효과는 다르며, 무인용도 모델이 읽지 않았다는 증거는 아니다.
- 호환 순서 Python 스킬은0개 제공, 선택/실행0. 기존 물리 복구 스킬은 보존했지만 이번 순서용으로 실행하지 않았다. **새 순서 스킬 생성·등록·효과는 여전히 미검증**이다.
- 기존 결과 17,443파일·운영 Python 263파일·동결 지식·입력 보존 확인. E드라이브 사용·조사 없음. 로봇 파지/실물 실행 없음.
- 새 실제 관측 27건을 별도 저장: PASS/수락15건, UNKNOWN/미수락12건, 마감 초과0. 순서·수락·인용·사용량·관측 해시는 독립 검토 두 건에서도 일치했다.

상세: `runs/sequence_native_00335_live_20261110/REPORT.md`, `analysis.json`, `comparison.json`, `trial/episode/worker_result.json`, `trial/api_ledger.sqlite`, `trial/execute/command.json`·stdout/stderr. 새 실제 관측은 `trial/native_observations.json`에 분리 저장했으며 기존 동결 메모리에는 합치지 않았다.

## 호출·예산·물리 상세

- API endpoint: `https://api.openai.com/v1/responses`, 요청 모델 `gpt-6.1-sol`, reasoning low, 출력 cap2048, 본문 cap65536bytes.
- 승인상한25회/$4.00. 실제 예약 16회/$2.292434, usage 기준 설정단가 추정 $0.689712. 실청구서 미확인. 실패/미확정 예약은 임의 환급 처리하지 않는다.
- 입력 토큰 325621, 출력 토큰 3847, 합계 329468.
- 모델 오류: `[]`. HTTP 재시도 0.
- 복합 후보·자세 시도 27회, 수락된 물리 시도 15회. 이를 연결된 제거 수와 혼동하지 않는다.
- 경로 원시 결과: True 78, False 84, 오류 0. 안정성: True 15, False 44, 오류 0.
- 전체 순서: `['15', '6', '7', '1', '3', '0', '10', '14', '5', '4', '2', '12', '13', '8', '9']`. 최대 부분 체인: `['15', '6', '7', '1', '3', '0', '10', '14', '5', '4', '2', '12', '13', '8', '9']`.
- Worker 1093.803초, 외부 1112.812초, check_success 내부물리 499.901초, 순위 IPC 대기 합계 148.642초. 초기화/기하/기타 시간은 물리 check_success 시간과 같지 않다.
- 내부물리 시간도 GNN 452.924초 대비 이번 499.901초로 늘었다. 더 적은 검사가 반드시 더 짧은 계산을 뜻하지 않으며 전체 시간에는 모델 대기도 포함된다.
- 원래GNN과 공통7개 물리 소스/바이너리 hash 일치: `True`. 모든 물리상수 일치: `True`. 원래GNN/heuristic 호출 방지 감시값 0.
- terminal은 `unchanged_upstream_mass_or_base_rule`. 마지막 두 부품도 LLM이 직접 골랐다고 설명하지 않는다.

## 성공·실패·미실행 구분

- 실행됨: 새 한 회차, 실제 모델 호출과 원래 ASAP 탐색, 결과·비용·원시 물리 기록·경험 별도 저장.
- 완료 여부: **전체 부품 수준 분해 PASS**. 실패한 개별 시도는 불가능한 부품/충돌 증명으로 바꾸어 저장하지 않는다.
- 미실행: GNN 새 재실행, 반복 seed, memory off 대조, sequence skill 자동생성/등록, 스킬 효과 대조, 새제품 일반화, 로봇 파지·모션/실물 검증.
- 보존 확인: `{"protected_files": 17443, "prior_changed": [], "production_source_files": 263, "source_changed": [], "input_changed": [], "frozen_changed": [], "E_drive_access": false}`.
- 당시 사전검사는 검증 완료 코드·최종194개 회귀 결과의 해시 일치와 입력/지식 복사해시를 확인했다. 이번에194개 검사를 새로 실행한 것으로 보고하지 않는다. 기존 준비 결과는 `runs/sequence_native_prepare_20261110`에 보존했다.

## 실행 명령·로그

Windows 실행 진입 명령은 승인된 `tools/run_sequence_native_experiment.py --mode llm --allow-live-api --use-registered-key`다. 전체 정책/인자는 `trial/configuration.json`, WSL 실제 argv는 `trial/execute/command.json`에 있다. 기존 사용자 암호화 키는 인증에만 사용했고 파일에 출력/기록하지 않았다.

```text
[
  "C:/Windows/System32/wsl.exe",
  "-d",
  "ASAP-Ubuntu",
  "-u",
  "root",
  "--cd",
  "/opt/asap-research/ASAP",
  "--",
  "/usr/bin/timeout",
  "--signal=TERM",
  "--kill-after=10s",
  "3900s",
  "/opt/asap-research/envs/asap/bin/python",
  "-B",
  "-u",
  "/mnt/c/dev/disassembly_codex_ready_v2/worker/run_sequence_native.py",
  "--asap-root",
  "/opt/asap-research/ASAP",
  "--assembly-dir",
  "/mnt/c/dev/disassembly_codex_ready_v2/runs/sequence_native_00335_live_20261110/input/00335",
  "--output-dir",
  "/mnt/c/dev/disassembly_codex_ready_v2/runs/sequence_native_00335_live_20261110/trial/episode",
  "--run-id",
  "trial",
  "--seed",
  "0",
  "--budget",
  "256",
  "--path-limit",
  "256",
  "--stability-limit",
  "256",
  "--timeout",
  "3600",
  "--response-timeout",
  "120",
  "--max-gripper",
  "3",
  "--max-pose",
  "128"
]
```

준비: `prepare.py`, 진단 조회: `poll.py`, 최종 읽기 전용 분석: `analyze.py`. 원시 로그·SQLite·IPC 파일은 수정하지 않았다. 다음 실행은 새 run과 별도 사용자 범위에서 진행한다.
