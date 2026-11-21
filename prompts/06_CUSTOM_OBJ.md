# 후속: custom OBJ / 다른 official assemblies
현재 first baseline은 반드시 official beam이다. 사용자 요청 또는 milestone 승인 뒤에만 시작해.
부품별 mesh hash와 assembled transforms, 단위, 좌표계, 상태/목표를 확인해.
각 부품을 개별 원점으로 옮기지 말고 contact graph를 무조건 precedence로 바꾸지 마.
공식 이미 처리된 데이터를 재정규화하지 마. custom data만 source 안내에 맞춰 전처리해.
선택적 목표를 추가할 경우 native goal과 다르다는 사실을 config/실험 표에 기록해.
렌더/기하 확인과 실제 physics plan 통과를 분리해 보고해.
