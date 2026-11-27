@echo off
chcp 65001 >nul
cd /d E:\AI策略交易
echo ====================================================
echo   MU Specialist Quant System 2.0 本地网页控制台
echo   正在自动清理旧版 8501 端口残留进程...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8501 " ^| findstr "LISTENING"') do taskkill /F /PID %%a >nul 2>&1
echo   浏览器请访问: http://127.0.0.1:8501
echo   (无杠杆 · 禁止实盘自动下单 · 仅研究与纸面交易)
echo ====================================================
start "" http://127.0.0.1:8501
python src\mu_quant\dashboard\server.py --port 8501
pause
