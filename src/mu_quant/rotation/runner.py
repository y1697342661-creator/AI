"""CLI Runner for US Stock Screener-Linked Alpaca Paper Trading Auto-Executor (Zero Leverage, Paper API Only)."""
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
    sync_screener_alpaca_paper,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="US Stock Full-Market Screener -> Alpaca Paper Trading Auto-Executor (paper-api ONLY)"
    )
    parser.add_argument(
        "--from-screener",
        action="store_true",
        help="Read target weights from data/screener/latest_allocation.json (default behavior)",
    )
    parser.add_argument(
        "--sync-alpaca-paper",
        action="store_true",
        help="Sync US stocks on Alpaca Paper Trading (paper-api.alpaca.markets ONLY)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview rebalance orders without submitting",
    )
    args = parser.parse_args()

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
        print(
            f"  [{tag}] {p['symbol']:<6} 目标={float(p['weight'])*100:5.1f}% "
            f"| 综合分={p.get('score')} | {p.get('sector')} · {p.get('theme')}"
        )

    if args.sync_alpaca_paper:
        print("-" * 72)
        res = sync_screener_alpaca_paper(alloc, dry_run=args.dry_run)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        print("=" * 72)
        return 0 if res.get("ok") else 1

    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
