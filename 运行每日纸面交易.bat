@echo off
chcp 65001 >nul
cd /d E:\AI绛栫暐浜ゆ槗
python src\mu_quant\paper_trading\daily_runner.py --force-refresh
pause
