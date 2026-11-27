# 一键注册 Windows 每日定时任务：
# 1) 周二至周六 07:00（美股收盘后）：自动更新美股正股数据并记录纸面信号
# 2) 周一至周五 22:35（美股开盘后 5 分钟，XMU-USDT 盘中点差最窄）：自动同步 OKX 模拟盘现货仓位（仅虚拟盘 x-simulated-trading=1）

$Action1 = New-ScheduledTaskAction -Execute "python.exe" -Argument "E:\AI策略交易\src\mu_quant\paper_trading\daily_runner.py" -WorkingDirectory "E:\AI策略交易"
$Trigger1 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Tuesday,Wednesday,Thursday,Friday,Saturday -At 7:00AM
Register-ScheduledTask -TaskName "MU_Quant_2_Daily_Data_Signal" -Action $Action1 -Trigger $Trigger1 -Description "MU 2.0 每日美股收盘后自动更新行情与记录纸面信号" -Force

$Action2 = New-ScheduledTaskAction -Execute "python.exe" -Argument "E:\AI策略交易\src\mu_quant\paper_trading\okx_demo_runner.py" -WorkingDirectory "E:\AI策略交易"
$Trigger2 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At 10:35PM
Register-ScheduledTask -TaskName "MU_Quant_2_OKX_Demo_Sync" -Action $Action2 -Trigger $Trigger2 -Description "MU 2.0 美股开盘时段自动同步 OKX 模拟盘现货 XMU-USDT（仅虚拟盘）" -Force

Write-Host "已成功注册两个 Windows 每日定时任务：MU_Quant_2_Daily_Data_Signal (07:00) & MU_Quant_2_OKX_Demo_Sync (22:35)"
