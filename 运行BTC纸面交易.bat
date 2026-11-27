@echo off
chcp 65001 >nul
cd /d E:\AI策略交易
set PYTHONPATH=E:\AI策略交易\src
python -m btc_quant.paper_trading.runner --refresh
pause
