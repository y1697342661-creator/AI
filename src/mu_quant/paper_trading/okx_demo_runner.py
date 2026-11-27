"""OKX Simulated Trading (Demo Only) Auto-Sync Runner.

Workflow:
1. Auto-refresh NASDAQ: MU + benchmark market data (Yahoo 3mo -> Eastmoney fallback).
2. Verify frozen v2.0.0 strategy integrity & data quality.
3. Compute T-day close target weight on MU common stock.
4. Append signal to local paper signals_log.csv.
5. Sync OKX SIMULATED SPOT `XMU-USDT` position (`x-simulated-trading: 1`, zero leverage, cash only).
6. If an order executes, also append to local paper trades_ledger.csv for dashboard visibility.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mu_quant.config import load_active_config, load_strategy_version
from mu_quant.dashboard.server import build_dashboard_payload
from mu_quant.data.loader import refresh_all_market_data
from mu_quant.execution_simulator.okx_demo_channel import sync_demo_spot_position
from mu_quant.paper_trading.daily_runner import verify_version_freeze
from mu_quant.paper_trading.engine import record_daily_signal, record_paper_trade


def main() -> int:
    ac = load_active_config()
    ver = ac.get("active_version", "v2.0.0")
    cfg = load_strategy_version(ver)

    print("=" * 72)
    print(f"MU 2.0 · OKX 模拟盘自动同步演练 ({ver} · x-simulated-trading=1)")
    print("=" * 72)

    fz = verify_version_freeze(ver)
    print(f"[1/4] 冻结校验: ok={fz['ok']} | SHA256={fz['sha256'][:16]}...")

    ref = refresh_all_market_data(force=False, min_interval_hours=4.0)
    print(f"[2/4] 行情更新: ok={ref.get('ok')} | skipped={ref.get('skipped')}")

    payload = build_dashboard_payload(
        long_mode=cfg["long_strategy"]["default_mode"],
        portfolio_mode="long_only",  # OKX XMU-USDT spot supports long+cash only (no leverage/swap)
        start_date="2004-01-01",
    )
    if payload["quality"]["overall"] != "PASS":
        print(f"[阻断] 数据质检未通过，停止同步！")
        return 1

    m = payload["market"]
    tw = float(payload["signals"]["target_weight"])
    record_daily_signal({
        "signal_date": m["date"],
        "strategy_version": ver,
        "portfolio_mode": "okx_demo_spot_long_only",
        "close_price": m["close"],
        "regime": payload["regime"]["code"],
        "long_signal": payload["signals"]["long"]["signal"],
        "short_signal": payload["signals"]["short"]["signal"],
        "target_weight": tw,
        "stop_price": payload["risk_plan"]["stop_price"],
        "note": f"OKX模拟盘同步 | {payload['signals']['long']['reason']}",
    })
    print(f"[3/4] MU最新收盘({m['date']}): ${m['close']:.2f} | 目标仓位: {tw*100:.1f}%")

    sync_res = sync_demo_spot_position(
        target_weight=tw,
        strategy_version=ver,
        rebalance_tol=float(cfg["long_strategy"].get("rebalance_tolerance", 0.05)),
        mu_ref_close=m["close"],
    )
    print(f"[4/4] OKX 模拟盘执行结果: action={sync_res.get('action')} | ok={sync_res.get('ok')}")
    if sync_res.get("ok") and sync_res.get("action") in ("BUY", "SELL"):
        od = sync_res["order"]
        record_paper_trade({
            "trade_date": m["date"],
            "strategy_version": ver,
            "action": od["action"],
            "shares": od["est_shares"],
            "fill_price": od["ref_xmu_px"],
            "commission": round(od["est_shares"] * od["ref_xmu_px"] * 0.001, 2),
            "borrow_fee": 0.0,
            "note": f"OKX模拟盘自动同步 ordId={od['ord_id']}",
        })
    print(json.dumps({k: v for k, v in sync_res.items() if k != "raw_resp"}, ensure_ascii=False, indent=2))
    print("=" * 72)
    return 0 if sync_res.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
