@echo off
chcp 65001 > nul
echo ADR 대시보드 시작 중...
streamlit run "%~dp0adr_app.py" --server.port 8502
pause
