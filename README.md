# MU Specialist Quant System 2.0
# 美光（NASDAQ: MU）专项双向量化波段交易系统

针对 **Micron Technology（NASDAQ: MU）** 正股的模块化双向量化研究与前瞻性纸面交易系统。

## 核心纪律
- 标的：NASDAQ: MU 正股（**严禁使用永续合约或以代币价格冒充正股**）
- 杠杆：**严格禁用（最大总仓位 ≤ 100%）**
- 实盘下单：**硬编码禁用自动下单**，仅输出模拟成交单与前瞻性纸面交易日志
- 成交时序：**T 日收盘出信号 → T+1 开盘成交**；计入双边佣金+滑点（默认 15bp/边）、做空按日融券费（默认 1.5%/年）及跨除息日分红代付
- 多空独立：做多与做空完全独立建模，下跌市允许空仓

## 快速启动
1. **一键打开本地网页控制台**：双击 `启动控制台.bat`，或命令行运行：
   ```powershell
   cd E:\AI策略交易
   python src\mu_quant\dashboard\server.py --port 8501
   ```
   浏览器打开 `http://127.0.0.1:8501` 即可查看行情、核心指标、市场环境、多空信号、止损价、净值对比曲线、绩效表与纸面交易面板。
2. **运行自动化测试**：
   ```powershell
   cd E:\AI策略交易
   python -m unittest discover -s tests -v
   ```

## 目录结构
- `configs/strategy_versions/`：不可变策略参数版本快照（当前 `v2.0.0.json`）
- `data/raw/`：NASDAQ: MU 正股及行业对照原始历史行情
- `data/paper_trading/`：只追加（append-only）纸面信号与交易账本
- `src/mu_quant/`：14 个核心功能模块（data / features / market_regime / long_strategy / short_strategy / portfolio / risk / backtest / execution_simulator / paper_trading / performance / dashboard / logs）
- `research/`：阶段 2 与阶段 3 研究脚本及原始 JSON 结果
- `tests/`：自动化单元测试、未来函数检查与订单时序测试
