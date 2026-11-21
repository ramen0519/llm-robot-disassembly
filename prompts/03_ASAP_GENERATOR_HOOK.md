# 실제 ranker/verification hook
M1/M2 결과를 확인하고 docs/05, docs/08, docs/23을 읽어줘.
actual generator/DFS/feasibility call graph와 signatures를 파일·함수·commit 기준으로 조사해.
검사 순서, caching, pose/support loop, state별 generator 재호출 시점을 확인해.
먼저 no-op hook을 연결해 공식 baseline과 후보/검증 의미가 같은지 테스트해.
그다음 strict permutation ranker와 actual event hook을 추가해.
근거 없는 function name, blocker ID, status를 만들지 마.
수정 patch는 작게 유지하고 공식 원본 비교 run을 보존해.
