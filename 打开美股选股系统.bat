@echo off
cd /d "%~dp0"
echo [1/2] Syncing latest cloud screener & Alpaca trading data...
if exist "C:\Program Files\Git\cmd\git.exe" (
    "C:\Program Files\Git\cmd\git.exe" pull --rebase --autostash origin main >nul 2>&1
)
echo [2/2] Starting US Stock Screener & Trading Dashboard Server...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8501 " ^| findstr "LISTENING"') do (
    taskkill /PID %%a /F >nul 2>&1
)
start "US_Stock_Screener_8501" /MIN python src\mu_quant\dashboard\server.py
timeout /t 2 /nobreak >nul
start "" "http://127.0.0.1:8501/screener"
