@echo off
cd /d "%~dp0"
echo Starting MU 2.0 & 7-Asset Rotation Dashboard Server...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8501 " ^| findstr "LISTENING"') do (
    taskkill /PID %%a /F >nul 2>&1
)
start "" /B python src\mu_quant\dashboard\server.py
timeout /t 2 /nobreak >nul
start "" "http://127.0.0.1:8501/rotation"
