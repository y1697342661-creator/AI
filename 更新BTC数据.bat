@echo off
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src"
python -m btc_quant.data.loader
pause
