@echo off
chcp 65001 >nul
cd /d E:\AI策略交易
set PYTHONPATH=E:\AI策略交易\src
echo 默认以干跑预览模式检查 OKX 模拟盘 (x-simulated-trading: 1) BTC-USDT-SWAP 仓位...
python -m btc_quant.paper_trading.runner --refresh --sync-okx-demo
echo.
echo 如需真实向 OKX 模拟盘下单，请加上 --execute-demo 参数。
pause
