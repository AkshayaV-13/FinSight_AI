@echo off
title FinSight AI — Financial Intelligence Copilot
cd /d "%~dp0"
set PYTHONUTF8=1
echo Starting FinSight AI on http://localhost:8501 ...
"C:\Users\Akshaya V\OneDrive\Desktop\Internship project\.venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501
pause
