@echo off
chcp 65001 >nul
cd /d E:\AI策略交易
set PYTHONPATH=E:\AI策略交易\src
echo 正在启动 BTC-USDT-SWAP 永续合约量化控制台 (http://127.0.0.1:8502) ...
start "" http://127.0.0.1:8502
python src\btc_quant\dashboard\server.py --port 8502
pause
