# Evidence memory와 등록형 executable strategy
docs/06, docs/21, docs/23에 맞춰 실제 verifier event를 append-only로 저장해줘.
MVP는 soft ranking only, hard pruning 비활성.
UNKNOWN/ERROR/진단 없는 bool을 물리적으로 불가능한 제약으로 승격하지 마.
mesh/pose/support/solver 설정을 기록하고 null을 허용해.
등록형 함수는 읽기 전용 state에서 candidate priority/strategy proposal을 반환하게 해.
팀원 code가 없으면 이식했다고 하지 말고 독립 구현임을 밝혀.
function 제공/검색/선택/실행/result를 나누어 계측해.
auto code evolution과 arbitrary runtime exec는 넣지 마.
