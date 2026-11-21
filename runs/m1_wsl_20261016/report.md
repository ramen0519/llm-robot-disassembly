# M1 공식 beam baseline — PASS

2026-10-16 KST. 연구 runtime에 LLM/새 DFS/새 physics를 추가하지 않았다.

## 실제 결과

- 공식 원본 commit: 3727e716b57e0b508c4e9ecd9fefa6953e8f72da.
- 입력: 공식 beam_assembly/original, OBJ 0/1/2/3/6. 원본 mesh/pose 무수정.
- planner=dfs, generator=heur-out, base-part=6, max-gripper=2, early-term=True.
- 기본값 유지: seed=0, budget=400, max-pose=3, pose-reuse=0, num-proc=1, timeout=None.
- **계획 성공: 1 → 0 → 3 → 2 제거, 최종 부품 6.**
- native stats: success=true, stop_msg='solution found', time=69.81초,
  n_eval=3, total_n_eval=3, n_gripper=1. 전체 프로세스 경과 82.79초.
- n_eval은 upstream 탐색 통계다. 마지막 leaf 검증 및 내부 방향별 연산 전체를 세지 않으므로
  총 verifier 호출 수/geometry 연산 수로 해석하지 않는다. 해당 metric은 null.
- 실제 계획 경로의 네 edge 모두 feasible=True, 유효한 action/support, 최종 (6,)을 검사해 PASS.
  이는 공식 native 전체 분해 종료이며 임의 target의 선택적 회수 기능은 구현하지 않았다.

## 환경 / 원본 유지

- WSL ASAP-Ubuntu, Ubuntu22.04.5, Python3.8.0.
- Linux 실행 소스 /opt/asap-research/ASAP; worker /opt/asap-research/envs/asap/bin/python.
- NumPy1.21.4, SciPy1.7.3, trimesh3.10.8, torch1.13.1+cpu, PyG2.2.0.
- 공식 C++ 원본을 CMake Release로 빌드. 메모리를 위해 -j8 대신 -j2 사용.
- Windows 보존 원본과 Linux 실행 소스 모두 tracked diff 없음. 런타임 SDF 캐시는 생성됨.
- 설치/실패/재시도와 freeze: ../setup_20261016_wsl/report.md.

## 명령

아래는 Linux ASAP 루트에서 실행한 내부 명령이며, 전체 WSL argv 및 Windows cwd는
각 command.json에 기록되어 있다. stdout/stderr/exit_code 원본 보존.

```bash
/opt/asap-research/envs/asap/bin/python -u plan_sequence/run_seq_plan.py \
  --dir beam_assembly --id original --planner dfs --generator heur-out \
  --max-gripper 2 --base-part 6 \
  --log-dir /mnt/c/dev/disassembly_codex_ready_v2/runs/m1_wsl_20261016/beam/native \
  --early-term
```

공식 smoke의 test_sim/test_simple_sim.py를 worker/smoke_replay_once.py로 실행했다.
물리 2000단계는 원본 그대로이며, 뷰어 loop/infinite만 False로 설정해 자동 종료했다.
Xvfb + software OpenGL에서 replay 완료, q/qdot/variables 유한값 확인, exit0.
이는 '공식 entrypoint + 단회 재생 옵션 wrapper'이며 CLI 바이트 단위 무변경 실행이라고 부르지 않는다.

공식 exporter play_logged_plan.py --save-all을 Xvfb에서 실행했다.
카메라 pos=3.15,-1.24,1.6, lookat=2.59,-0.55,1.16. exit0, 경과125.29초.
export는 선택된 경로를 다시 계산/재생한 후 저장한다. 비용은 원래 탐색 시간과 분리한다.

## 산출물

- beam/native/dfs-heur-out/s0/original/setup.json
- beam/native/dfs-heur-out/s0/original/stats.json
- beam/native/dfs-heur-out/s0/original/tree.pkl — 이번 로컬 실행 생성물만 신뢰하여 로딩
- validation.json — tree contract PASS, 물리 재실행은 별도 exporter 수행
- smoke/physics_summary.json — 2000 steps, viewer completed, finite=True
- export/results/path/{0_1,1_0,2_3,3_2}/ — 각 300개 시각 × 5개 부품의 4×4 행렬
- export/results/record/{0_1,1_0,2_3,3_2}.gif
- export/results/{mesh,part_fix,pose}/
- export_validation.json — 전체 6000개 행렬의 shape/유한값/마지막 행과 GIF header 검사 PASS

## 검증 범위 / 다음 단계

- 실제 ASAP 물리 및 공식 beam 계획 재현 완료. mock 성공으로 대체하지 않았다.
- 단일 예제/seed 성공이며 성능 개선·일반화·실물 로봇 실행은 입증하지 않는다.
- M2는 동일한 warm SDF 상태의 직접 실행과 no-op adapter를 별도로 비교한다.
- ranker, memory, skill, LangGraph, live LLM은 NOT_RUN.
