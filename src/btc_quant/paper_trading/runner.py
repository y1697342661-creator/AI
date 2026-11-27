"""CLI Runner for BTC-USDT-SWAP 4H Paper Trading & Optional OKX Demo Sync.

Usage:
    python -m btc_quant.paper_trading.runner
    python -m btc_quant.paper_trading.runner --refresh
    python -m btc_quant.paper_trading.runner --refresh --sync-okx-demo
    python -m btc_quant.paper_trading.runner --refresh --sync-okx-demo --execute-demo
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from btc_quant.config import load_active_config
from btc_quant.data.loader import fetch_live_snapshot, load_btc_datasets, refresh_btc_market_data
from btc_quant.execution_simulator.okx_demo_swap import sync_demo_swap_target
from btc_quant.features.indicators import build_1d_features, build_4h_features
from btc_quant.market_regime.regime import latest_regime_summary
from btc_quant.paper_trading.engine import record_paper_step
from btc_quant.risk.manager import BtcRiskConfig
from btc_quant.strategies.perp_strategies import latest_signal_snapshot


def main() -> None:
    ap = argparse.ArgumentParser(description="BTC-USDT-SWAP 4H Paper Trading & OKX Demo Runner")
    ap.add_argument("--refresh", action="store_true", help="Refresh OKX 4H/1D/funding data first")
    ap.add_argument("--sync-okx-demo", action="store_true", help="Compute OKX Simulated Trading sync plan")
    ap.add_argument("--execute-demo", action="store_true", help="Submit order to OKX Simulated Trading (x-simulated-trading: 1 ONLY)")
    ap.add_argument("--allocation-usdt", type=float, default=None, help="Optional USDT cap for OKX demo allocation")
    args = ap.parse_args()

    cfg = load_active_config()
    if args.refresh:
        refresh_btc_market_data()

    ds = load_btc_datasets()
    df_1d_f = build_1d_features(ds["df_1d"])
    df_4h_f = build_4h_features(ds["df_4h"], df_1d_f)

    strat_cfg = cfg.get("strategy", {})
    risk_raw = cfg.get("risk", {})
    risk_cfg = BtcRiskConfig(
        max_long_leverage=float(strat_cfg.get("max_long_leverage", 2.0)),
        max_short_leverage=float(strat_cfg.get("max_short_leverage", 1.2)),
        hard_cap_leverage=float(risk_raw.get("hard_cap_leverage", 3.0)),
        single_trade_risk_pct=float(risk_raw.get("single_trade_risk_pct", 0.025)),
        dd_brake_start=float(risk_raw.get("dd_brake_start", 0.12)),
        dd_brake_full=float(risk_raw.get("dd_brake_full", 0.25)),
        dd_min_scale=float(risk_raw.get("dd_min_scale", 0.35)),
        dd_halt_threshold=float(risk_raw.get("dd_halt_threshold", 0.32)),
    )

    regime = latest_regime_summary(df_4h_f)
    signal = latest_signal_snapshot(df_4h_f, mode=strat_cfg.get("default_mode", "dual_horizon_ensemble"), params=strat_cfg)
    last_fr = float(df_4h_f["funding_rate"].iloc[-1])

    paper_res = record_paper_step(signal=signal, regime=regime, risk_cfg=risk_cfg, funding_rate_8h=last_fr)
    out = {
        "version": cfg.get("version", "v1.0.0"),
        "regime": regime,
        "signal": signal,
        "paper": paper_res,
    }

    if args.sync_okx_demo:
        live = fetch_live_snapshot()
        plan = paper_res["risk_plan"]
        out["okx_demo_sync"] = sync_demo_swap_target(
            target_leverage=plan["approved_leverage"],
            ref_price=live["last"],
            isolated_leverage=plan["isolated_margin_leverage"],
            allocation_usdt=args.allocation_usdt,
            stop_loss_px=plan["stop_px"],
            dry_run=(not args.execute_demo),
        )

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
