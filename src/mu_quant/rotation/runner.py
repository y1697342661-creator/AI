"""CLI Runner for 6-Asset Cross-Sector High-Elasticity Top-2 Rotation (Zero Leverage, OKX Demo Only)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mu_quant.rotation.alpaca_paper import (
    load_screener_allocation,
    sync_rotation_alpaca_paper,
    sync_screener_alpaca_paper,
)
from mu_quant.rotation.engine import (
    compute_rotation_signal,
    load_rotation_config,
    record_rotation_signal,
    sync_rotation_okx_demo,
    update_rotation_forward_scorecard,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="US Stock Screener-Linked / Rotation Executor (Paper/Simulated Only)")
    parser.add_argument("--from-screener", action="store_true", help="Execute directly from data/screener/latest_allocation.json")
    parser.add_argument("--no-refresh", action="store_true", help="Use cached daily CSVs without network fetch")
    parser.add_argument("--sync-okx-demo", action="store_true", help="Sync whitelisted X*-USDT spot positions on OKX Demo")
    parser.add_argument("--sync-alpaca-paper", action="store_true", help="Sync US stocks on Alpaca Paper Trading (paper-api ONLY)")
    parser.add_argument("--dry-run", action="store_true", help="Preview rebalance orders without submitting")
    args = parser.parse_args()

    if args.from_screener:
        alloc = load_screener_allocation()
        print("=" * 72)
        print(f"美股全市场选股联动执行器 ({alloc.get('version')} · {alloc.get('regime_label', alloc.get('regime'))})")
        print("=" * 72)
        print(
            f"信号日期: {alloc.get('signal_date')} | 进攻: {float(alloc.get('offense_weight', 0))*100:.1f}% "
            f"| 防守: {float(alloc.get('defense_weight', 0))*100:.1f}% | 现金: {float(alloc.get('cash_weight', 0))*100:.1f}%"
        )
        for p in alloc.get("positions", []):
            tag = "进攻" if p.get("sleeve") == "OFFENSE" else "防守"
            print(f"  [{tag}] {p['symbol']:<6} 目标={float(p['weight'])*100:5.1f}% | 综合分={p.get('score')} | {p.get('sector')} · {p.get('theme')}")
        if args.sync_alpaca_paper:
            print("-" * 72)
            res = sync_screener_alpaca_paper(alloc, dry_run=args.dry_run)
            print(json.dumps(res, ensure_ascii=False, indent=2))
            print("=" * 72)
            return 0 if res.get("ok") else 1
        print("=" * 72)
        return 0

    cfg = load_rotation_config()
    print("=" * 72)
    print(f"{cfg['name']} ({cfg['version']} · x-simulated-trading=1)")
    print("=" * 72)

    sig = compute_rotation_signal(cfg, refresh=not args.no_refresh)
    record_rotation_signal(sig)
    fwd = update_rotation_forward_scorecard(sig)

    print(f"最新交易日 : {sig['signal_date']} | 前瞻净值: ${fwd['nav']:,.2f}")
    print(f"选中 Top-{cfg['rules']['top_k']} : {', '.join(sig['selected_symbols']) if sig['selected_symbols'] else '全空仓(CASH)'} "
          f"| 总目标仓位: {sig['total_invested_weight']*100:.1f}% | 现金: {sig['cash_weight']*100:.1f}%")
    print("-" * 72)
    for s in sig["scores"]:
        st = "站上年线" if s["above200"] else "跌破年线"
        m60 = f"{s['mom60']*100:+.1f}%" if s["mom60"] is not None else "N/A"
        dv = f"{s['dvol21']*100:.1f}%" if s["dvol21"] is not None else "N/A"
        print(f"  {s['symbol']:<5} ({s['okx_inst_id']:<10}) 收盘=${s['close']:<8.2f} {st} 60日={m60:<7} 下行波动={dv:<6} 目标={s['target_weight']*100:5.1f}%")

    if args.sync_alpaca_paper:
        print("-" * 72)
        res = sync_rotation_alpaca_paper(sig, cfg, dry_run=args.dry_run)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        print("=" * 72)
        return 0 if res.get("ok") else 1

    if args.sync_okx_demo:
        print("-" * 72)
        res = sync_rotation_okx_demo(sig, cfg, dry_run=args.dry_run)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        print("=" * 72)
        return 0 if res.get("ok") else 1

    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
