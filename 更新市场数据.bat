@echo off
chcp 65001 >nul
cd /d E:\AI绛栫暐浜ゆ槗
python -c "import sys; sys.path.insert(0,'src'); from mu_quant.data.loader import refresh_all_market_data; import json; print(json.dumps(refresh_all_market_data(force=True), ensure_ascii=False, indent=2))"
pause
