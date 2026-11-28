"""Multi-Coin (BTC + ETH + SOL) 4H Perpetual Portfolio Runner & OKX Demo Sync.

Usage:
    python -m btc_quant.paper_trading.portfolio_runner --refresh
    python -m btc_quant.paper_trading.portfolio_runner --refresh --sync-okx-demo
    python -m btc_quant.paper_trading.portfolio_runner --refresh --sync-okx-demo --execute-demo
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from btc_quant.config import PAPER_DIR, load_active_config
from btc_quant.data.loader import (
    COIN_SPECS,
    SUPPORTED_COINS,
    fetch_live_snapshot,
    load_coin_datasets,
    refresh_all_coins_data,
)
from btc_quant.execution_simulator.okx_demo_swap import get_demo_swap_account_state, sync_demo_swap_target
from btc_quant.features.indicators import build_1d_features, build_4h_features
from btc_quant.logs.logger import get_logger
from btc_quant.market_regime.regime import latest_regime_summary
from btc_quant.paper_trading.engine import record_paper_step
from btc_quant.risk.manager import BtcRiskConfig, compute_contract_risk_plan
from btc_quant.strategies.perp_strategies import latest_signal_snapshot

log = get_logger("btc_quant.portfolio_runner")
PORTFOLIO_LOG = PAPER_DIR / "portfolio_signals_log.csv"


def run_portfolio_cycle(
    refresh: bool = False,
    sync_okx_demo: bool = False,
    execute_demo: bool = False,
    allocation_usdt: float | None = None,
) -> dict[str, Any]:
    cfg = load_active_config()
    port_cfg = cfg.get("portfolio", {})
    weights: dict[str, float] = port_cfg.get("weights", {"BTC": 0.30, "ETH": 0.30, "SOL": 0.40})
    coins = tuple(c for c in weights.keys() if c in SUPPORTED_COINS)

    if refresh:
        refresh_all_coins_data(coins=coins)

    strat_cfg = cfg.get("strategy", {})
    risk_raw = cfg.get("risk", {})
    risk_cfg = BtcRiskConfig(
        max_long_leverage=float(strat_cfg.get("max_long_leverage", 2.5)),
        max_short_leverage=float(strat_cfg.get("max_short_leverage", 1.5)),
        hard_cap_leverage=float(risk_raw.get("hard_cap_leverage", 3.0)),
        single_trade_risk_pct=float(risk_raw.get("single_trade_risk_pct", 0.03)),
        dd_brake_start=float(risk_raw.get("dd_brake_start", 0.12)),
        dd_brake_full=float(risk_raw.get("dd_brake_full", 0.25)),
        dd_min_scale=float(risk_raw.get("dd_min_scale", 0.35)),
        dd_halt_threshold=float(risk_raw.get("dd_halt_threshold", 0.32)),
    )

    # Read total simulated USDT equity once so all 3 coins share the exact equity base cleanly
    total_demo_equity = allocation_usdt
    if sync_okx_demo and not total_demo_equity:
        acct = get_demo_swap_account_state("BTC-USDT-SWAP")
        if acct.get("ok"):
            total_demo_equity = float(acct["usdt_equity"])

    coin_results: dict[str, Any] = {}
    log_rows: list[dict[str, Any]] = []
    mode = strat_cfg.get("default_mode", "dual_horizon_ensemble")

    for coin in coins:
        spec = COIN_SPECS[coin]
        w = float(weights[coin])
        ds = load_coin_datasets(coin)
        df_4h_f = build_4h_features(ds["df_4h"], build_1d_features(ds["df_1d"]))
        regime = latest_regime_summary(df_4h_f)
        signal = latest_signal_snapshot(df_4h_f, mode=mode, params=strat_cfg)

        # Keep BTC single-coin paper ledger updated too for full dashboard compatibility
        if coin == "BTC":
            record_paper_step(signal=signal, regime=regime, risk_cfg=risk_cfg, funding_rate_8h=float(df_4h_f["funding_rate"].iloc[-1]))

        slice_eq = (total_demo_equity * w) if (total_demo_equity and total_demo_equity > 0) else (100_000.0 * w)
        plan = compute_contract_risk_plan(
            equity_usdt=slice_eq,
            price=float(signal["close_px"]),
            target_leverage=float(signal["target_leverage"]),
            stop_distance_pct=float(signal["stop_distance_pct"]) / 100.0,
            funding_7d_ann_pct=float(signal.get("funding_7d_ann_pct", 0.0)),
            cfg=risk_cfg,
            ct_val=spec["ct_val"],
        )

        sync_res = None
        if sync_okx_demo:
            live = fetch_live_snapshot(coin)
            sync_res = sync_demo_swap_target(
                target_leverage=plan["approved_leverage"],
                ref_price=float(live["last"]),
                isolated_leverage=plan["isolated_margin_leverage"],
                allocation_usdt=slice_eq,
                stop_loss_px=plan["stop_px"],
                dry_run=(not execute_demo),
                inst_id=spec["swap_inst"],
            )

        coin_results[coin] = {
            "weight": w,
            "inst_id": spec["swap_inst"],
            "regime": regime["regime"],
            "signal": signal,
            "risk_plan": plan,
            "okx_demo_sync": sync_res,
        }
        log_rows.append(
            {
                "recorded_at": dt.datetime.now().isoformat(timespec="seconds"),
                "bar_time_utc": signal["bar_time_utc"],
                "coin": coin,
                "weight": w,
                "close_px": signal["close_px"],
                "regime": regime["regime"],
                "direction": plan["side"],
                "approved_leverage": plan["approved_leverage"],
                "okx_contracts": plan["okx_contracts"],
                "stop_px": plan["stop_px"],
                "demo_action": sync_res["action"] if sync_res and sync_res.get("ok") else "PAPER_ONLY",
            }
        )

    # Append to multi-coin portfolio signals log (idempotent per bar_time_utc + coin)
    new_df = pd.DataFrame(log_rows)
    if PORTFOLIO_LOG.exists():
        old_df = pd.read_csv(PORTFOLIO_LOG, encoding="utf-8-sig")
        comb = pd.concat([old_df, new_df], ignore_index=True)
        comb = comb.drop_duplicates(subset=["bar_time_utc", "coin"], keep="last")
        comb.to_csv(PORTFOLIO_LOG, index=False, encoding="utf-8-sig")
    else:
        new_df.to_csv(PORTFOLIO_LOG, index=False, encoding="utf-8-sig")

    return {
        "version": cfg.get("version", "v2.0.0"),
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "total_demo_equity_usdt": total_demo_equity,
        "weights": weights,
        "coins": coin_results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="BTC + ETH + SOL 3-Coin 4H Perpetual Portfolio Runner")
    ap.add_argument("--refresh", action="store_true", help="Refresh OKX 4H/1D/funding data for all 3 coins")
    ap.add_argument("--sync-okx-demo", action="store_true", help="Sync OKX Simulated Trading positions")
    ap.add_argument("--execute-demo", action="store_true", help="Execute orders on OKX Simulated Trading (x-simulated-trading: 1 ONLY)")
    ap.add_argument("--allocation-usdt", type=float, default=None, help="Optional total USDT cap across all 3 coins")
    args = ap.parse_args()

    out = run_portfolio_cycle(
        refresh=args.refresh,
        sync_okx_demo=args.sync_okx_demo,
        execute_demo=args.execute_demo,
        allocation_usdt=args.allocation_usdt,
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
