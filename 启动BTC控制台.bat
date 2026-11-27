@echo off
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src"
echo Starting BTC-USDT-SWAP Quant Dashboard at http://127.0.0.1:8502 ...
start "" http://127.0.0.1:8502
python src\btc_quant\dashboard\server.py --port 8502
pause
