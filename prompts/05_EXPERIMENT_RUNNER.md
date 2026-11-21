# 공정한 비교 runner
M1/M2 증거와 실제 capabilities를 확인하고 docs/10을 읽어줘.
B0 native/B0w no-op/B1 rule+memory/B2 rule+same-skill/P LLM+same-skill을 가능한 범위에서 비교해.
현재 beam이 공식 설치/개발 예제라는 점을 명시하고 새 task 일반화라고 하지 마.
같은 tasks/goal/solver/budget/seed 및 동결 memory snapshot을 사용해.
skill 내부 solver 요청도 중앙에서 세고 no-LLM baseline을 유지해.
전체/성공-only 비용, offline 구축비용, zero-check 실패를 분리해.
성능 향상을 강제하지 말고 raw rows에서 summary를 재생성해.
