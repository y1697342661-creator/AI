"""CLI Runner for the US Stock Offense+Defense Full-Market Screener (v2.0.0).

Usage:
    python -m mu_quant.screener.runner --scan-mode full_market --preset core_momentum --refresh --write-report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mu_quant.screener.engine import SCREENER_DIR, run_screener

LATEST_REPORT_MD = SCREENER_DIR / "latest_report.md"
LATEST_SCAN_JSON = SCREENER_DIR / "latest_scan.json"


def _fmt_pct(v: float | None, signed: bool = True) -> str:
    if v is None:
        return "-"
    return f"{v * 100:+.1f}%" if signed else f"{v * 100:.1f}%"


def render_markdown_report(res: dict[str, Any], top_n: int = 15) -> str:
    mb = res["market_breadth"]
    b = res["benchmark"]
    reg = res["market_regime"]
    plan = res["portfolio_plan"]
    fun = res.get("funnel", {})

    mode_str = (
        f"全美股两阶段漏斗（扫描 {fun.get('us_market_total', 0)} 只 → 初筛活跃 {fun.get('funnel_passed', 0)} 只 + 核心攻守池）"
        if res.get("scan_mode") == "full_market"
        else "核心攻守精选池（48只+自选）"
    )

    lines = [
        f"# 美股攻守兼备·全市场选股与动态配置日报 ({res['signal_date']})",
        "",
        f"- **市场状态**：**`{reg['regime']}`（{reg['label']}）** — {reg['reason']}",
        f"- **调仓动作**：`{plan.get('rebalance_action', 'REBALANCE')}` — {plan.get('rebalance_reason', '')}",
        f"- **攻守仓位建议**：进攻 `{plan['offense_weight']*100:.1f}%` · 防守 `{plan['defense_weight']*100:.1f}%` · 现金 `{plan['cash_weight']*100:.1f}%`（总仓位 `{plan['total_invested_weight']*100:.1f}%`，严格零杠杆）",
        f"- **扫描模式**：{mode_str} | **基准 QQQ 60日**：`{_fmt_pct(b['mom60'])}` | **年线上占比**：`{mb['above_ma200']}/{mb['total']}` ({mb['above_ma200_pct']*100:.1f}%)",
        "",
        "## 🎯 攻守动态组合建议（可直接对接轮动执行器 `latest_allocation.json`）",
        "",
        "| 角色 | 代码 | 目标权重 | 综合分 | DVol21 | 2.5ATR止损 | 赛道 | 逻辑 |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for p in plan["positions"]:
        role_cn = "🔥 进攻" if p["sleeve"] == "OFFENSE" else "🛡️ 防守"
        stp = f"${p['stop_price']:.2f}" if p["stop_price"] is not None else "-"
        tag_new = " 🌐" if p.get("discovered") else ""
        lines.append(
            f"| {role_cn} | **{p['symbol']}**{tag_new} | **{p['weight']*100:.1f}%** | {p['score']:.1f} "
            f"| {_fmt_pct(p['dvol21'], False)} | {stp} | {p['sector']} | {p['theme']} |"
        )
    if plan["cash_weight"] > 0.005:
        lines.append(f"| 💵 现金 | **CASH** | **{plan['cash_weight']*100:.1f}%** | - | - | - | 现金防守 | 保留纯现金机动仓位 |")

    lines += [
        "",
        f"## 📋 当前预设榜单 (`{res['preset']}` · 命中 {mb['matched_count']} 只)",
        "",
        "| 排名 | 代码 | 综合分 | 评级 | 收盘价 | 20日 | 60日 | 超额RS60 | DVol21 | 2.5ATR止损 | 赛道 | 标签 |",
        "|---:|---|---:|:---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in res["stocks"][:top_n]:
        stp = f"${r['stop_price']:.2f}" if r["stop_price"] is not None else "-"
        tags = " / ".join(r["tags"][:3])
        lines.append(
            f"| {r['rank']} | **{r['symbol']}** | {r['composite_score']:.1f} | {r['grade']} "
            f"| ${r['close']:.2f} | {_fmt_pct(r['mom20'])} | {_fmt_pct(r['mom60'])} "
            f"| {_fmt_pct(r['rs60_vs_qqq'])} | {_fmt_pct(r['dvol21'], False)} | {stp} "
            f"| {r['sector']} | {tags} |"
        )

    if res.get("discovered_top5"):
        lines += [
            "",
            "## 🌐 全美股漏斗新发现黑马 Top-5",
            "",
            "| 代码 | 综合分 | 评级 | 60日动量 | DVol21 | 赛道 & 市值成交 |",
            "|---|---:|:---:|---:|---:|---|",
        ]
        for x in res["discovered_top5"]:
            lines.append(
                f"| **{x['symbol']}** | {x['composite_score']:.1f} | {x['grade']} "
                f"| {_fmt_pct(x['mom60'])} | {_fmt_pct(x['dvol21'], False)} | {x['sector']} · {x['theme']} |"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="US Stock Offense+Defense Full-Market Screener")
    parser.add_argument(
        "--preset",
        default="core_momentum",
        choices=["core_momentum", "defense_shield", "breakout_52w", "market_discovery", "all"],
    )
    parser.add_argument("--scan-mode", default="curated", choices=["curated", "full_market"])
    parser.add_argument("--sector", default="ALL")
    parser.add_argument("--refresh", action="store_true", help="Refresh live daily bars from EastMoney")
    parser.add_argument("--top", type=int, default=15, help="Number of top stocks to display")
    parser.add_argument("--write-report", action="store_true", help="Write latest_report.md, latest_scan.json & GitHub Step Summary")
    args = parser.parse_args()

    res = run_screener(
        preset=args.preset,
        sector=args.sector,
        refresh=args.refresh,
        scan_mode=args.scan_mode,
        record_history=args.refresh,
    )
    mb = res["market_breadth"]
    reg = res["market_regime"]
    plan = res["portfolio_plan"]
    fun = res.get("funnel", {})

    print(f"\n=== {res['name']} ({res['signal_date']}) ===")
    print(f"市场状态: [{reg['regime']}] {reg['label']} | {reg['reason']}")
    print(
        f"攻守配置: 进攻 {plan['offense_weight']*100:.1f}% + 防守 {plan['defense_weight']*100:.1f}% + "
        f"现金 {plan['cash_weight']*100:.1f}% | 总仓位 {plan['total_invested_weight']*100:.1f}%"
    )
    if fun.get("enabled"):
        print(f"全市场漏斗: 全美股 {fun.get('us_market_total')} 只 -> 活跃初筛 {fun.get('funnel_passed')} 只 -> 总深度扫描 {mb['total']} 只")

    print("\n[攻守动态目标组合 (已写入 data/screener/latest_allocation.json)]:")
    for p in plan["positions"]:
        tag = "进攻" if p["sleeve"] == "OFFENSE" else "防守"
        print(f"  [{tag}] {p['symbol']:<6} 权重 {p['weight']*100:5.1f}% | 分数 {p['score']:5.1f} | DVol {_fmt_pct(p['dvol21'], False):>6} | {p['sector']}")

    print(f"\n[预设榜单: {args.preset} ({mb['matched_count']}/{mb['total']})]:")
    for r in res["stocks"][: args.top]:
        print(
            f"  #{r['rank']:<2} {r['symbol']:<6} {r['composite_score']:5.1f} ({r['grade']}) "
            f"${r['close']:<8.2f} 60D={_fmt_pct(r['mom60']):>7} RS60={_fmt_pct(r['rs60_vs_qqq']):>7} "
            f"DVol={_fmt_pct(r['dvol21'], False):>6}  {' '.join(r['tags'][:3])}"
        )

    if res.get("discovered_top5"):
        print("\n[全市场漏斗新黑马 Top-5]:")
        for x in res["discovered_top5"]:
            print(f"  - {x['symbol']:<6} 分数 {x['composite_score']:.1f} | 60日 {_fmt_pct(x['mom60'])} | {x['theme']}")

    if args.write_report:
        md = render_markdown_report(res, top_n=args.top)
        LATEST_REPORT_MD.write_text(md, encoding="utf-8")
        compact = dict(res)
        compact["stocks"] = [{k: v for k, v in r.items() if k != "sparkline"} for r in res["stocks"]]
        LATEST_SCAN_JSON.write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")

        summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_file:
            with open(summary_file, "a", encoding="utf-8") as fh:
                fh.write(md)


if __name__ == "__main__":
    main()
