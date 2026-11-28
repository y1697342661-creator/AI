# 美股攻守兼备 · 全市场选股与 Alpaca 模拟盘自动交易系统

全美股两阶段漏斗选股 + 市场状态攻守动态配仓 + **Alpaca Paper Trading（纯模拟盘、零杠杆）** 云端全自动执行系统。

## 核心架构
1. **选股与配仓大脑 (`src/mu_quant/screener/`)**：
   - 根据 `SPY` / `QQQ` 200 日年线与市场宽度自动切换三档状态：
     - **牛市进攻 (`BULL_OFFENSE`)**：进攻 80% + 防守 20% + 现金 0%
     - **震荡平衡 (`NEUTRAL_BALANCE`)**：进攻 45% + 防守 45% + 现金 10%
     - **熊市防守 (`BEAR_DEFENSE`)**：进攻 15% + 防守 65% + 现金 20%
   - 结合核心攻守池与全美股高流动性漏斗（股价 ≥ $5、市值 ≥ $30B、日成交 ≥ $50M），按趋势、相对 QQQ 超额动量、下行波动率（DVol21）综合打分，输出目标仓位至 `data/screener/latest_allocation.json`。
2. **模拟盘自动执行器 (`src/mu_quant/rotation/alpaca_paper.py`)**：
   - 接口硬编码为 `https://paper-api.alpaca.markets`（仅模拟盘，代码层绝不接触实盘）。
   - 严格零杠杆、先卖后买、单股权重硬上限 ≤ 40%，成交记录追加写入 `data/rotation/alpaca_paper_orders.csv`。

## 云端定时任务（GitHub Actions）
- **周二～周六 北京时间 `06:30`**（`.github/workflows/us_screener_daily.yml`）：全美股扫描并更新 `data/screener/latest_allocation.json`。
- **周一～周五 北京时间 `22:35`**（`.github/workflows/mu_daily_quant.yml`）：读取最新目标权重并在 Alpaca 模拟盘自动调仓。

## 本地快速启动
- **打开本地选股看板**：双击 `打开美股选股系统.bat`，浏览器访问 `http://127.0.0.1:8501/screener`。
- **运行自动化测试**：
  ```powershell
  $env:PYTHONPATH="src"; python -m unittest discover -s tests -p "test_*.py" -v
  ```
