@echo off
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src"
echo Dry-run preview for OKX Demo (x-simulated-trading: 1) BTC-USDT-SWAP...
python -m btc_quant.paper_trading.runner --refresh --sync-okx-demo
pause
