# LangGraph — native baseline 위의 얇은 실행 계층
M1 증거를 먼저 확인하고 docs/07을 읽어줘.
현재 upstream Python 환경과 LangGraph의 실제 Python 요구를 확인해; 무조건 하나로 섞지 마.
validate task → memory snapshot → launch worker → collect results → finalize graph를 구현하고 실행해.
DFS는 ASAP가 소유하고 LangGraph는 run/thread/artifact 참조를 소유하게 해.
whole-run wrapper인지 live per-candidate 연결인지 capability에 따로 표시해.
C++ DFS resume를 구현하지 않았다면 checkpointer만으로 지원한다고 주장하지 마.
실제 graph trace와 error/duplicate/fallback/END 테스트를 남겨.
