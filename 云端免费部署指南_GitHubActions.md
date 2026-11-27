# 零成本 24/7 云端自动运行指南（GitHub Actions 免费版）

本指南让你在 **电脑彻底关机** 的情况下，利用 GitHub Actions 免费额度（私有仓库每月 2,000 分钟，本系统每月仅消耗约 90 分钟，占比 < 5%），实现 **每 4 小时自动拉取 OKX K 线 → 更新纸面账本 → 同步 OKX 模拟盘（带交易所端原生止损单）**。

---

## 5 分钟配置步骤

### 第 1 步：在 GitHub 新建一个 **Private（私有）** 仓库
- 登录 GitHub，点击右上角 `+` → `New repository`。
- 仓库名例如 `ai-quant-trading`，**务必选择 `Private`（私有）**。

### 第 2 步：把本地代码推送到私有仓库
> 本地 `configs/okx_demo_secrets.local.json` 已写入 `.gitignore`，**不会被上传到 GitHub**，可放心推送。

在 `E:\AI策略交易` 打开 PowerShell 执行：
```powershell
cd E:\AI策略交易
git init
git add .
git commit -m "feat: init MU & BTC quant systems"
git branch -M main
git remote add origin https://github.com/<你的用户名>/<你的私有仓库名>.git
git push -u origin main
```

### 第 3 步：在 GitHub 填入 OKX 模拟盘密钥（Secrets）
打开你的私有仓库 → `Settings` → `Secrets and variables` → `Actions` → `New repository secret`，添加以下 3 个加密变量（与你本地 `configs/okx_demo_secrets.local.json` 一致）：
- `OKX_DEMO_API_KEY`
- `OKX_DEMO_SECRET_KEY`
- `OKX_DEMO_PASSPHRASE`

> 安全锁说明：代码中已硬编码 `x-simulated-trading: 1` 与逐仓 `isolated` 模式（杠杆上限 3x），即使在云端运行也**绝对不会触碰实盘账户**。

### 第 4 步：选择干跑预览或自动提交模拟盘订单
- **默认模式（最安全）**：每 4 小时自动刷新数据、记录纸面账本，并以 **干跑预览（Dry-Run）** 打印模拟盘调仓计划。
- **开启自动同步 OKX 模拟盘**：在仓库 `Settings` → `Secrets and variables` → `Actions` → `Variables` 标签页，新增一个变量：
  - Name: `BTC_AUTO_EXECUTE_DEMO`
  - Value: `true`
- **随时手机/网页手动触发**：打开仓库的 `Actions` → `BTC 4H Perpetual Quant Cloud Runner` → 点击 `Run workflow` 即可立刻测试一次！

---

## 关机后的双重保护
1. **每 4 小时云端定时执行**：GitHub 云端 Linux 机器在 UTC `00:02 / 04:02 / 08:02 / 12:02 / 16:02 / 20:02` 自动跑完并将最新净值与信号写回 `data/btc/paper_trading/`。
2. **交易所端原生止损单**：开启模拟盘下单后，每次开仓/加仓都会通过 OKX `attachAlgoOrds` 在交易所撮合引擎挂上**标记价格（Mark Price）止损单**，即使在两根 4H K 线之间出现极端暴跌/插针，也能由 OKX 服务器毫秒级触发止损。
