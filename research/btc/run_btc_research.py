"""Refined BTC-USDT-SWAP Research & Parameter Plateau Validation (with Conservative Funding Imputation)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from btc_quant.backtest.engine import PerpBacktestConfig, run_perp_backtest
from btc_quant.data.loader import load_btc_datasets, validate_btc_data_quality
from btc_quant.features.indicators import build_1d_features, build_4h_features
from btc_quant.strategies.perp_strategies import generate_strategy_signals

OUT_DIR = ROOT / "research" / "btc"


def main() -> None:
    ds = load_btc_datasets()
    qa = validate_btc_data_quality(ds)
    df_1d_f = build_1d_features(ds["df_1d"])
    df_4h_f = build_4h_features(ds["df_4h"], df_1d_f)
    df_bt = df_4h_f.loc["2021-02-01":].copy()

    modes = [
        ("buy_and_hold", {}),
        ("mtf_regime_trend", {"vol_target": 0.55, "max_long_leverage": 1.8, "max_short_leverage": 1.2}),
        ("donchian_cta", {"vol_target": 0.55, "max_long_leverage": 1.8, "max_short_leverage": 1.2}),
        ("dual_horizon_ensemble", {"vol_target": 0.60, "max_long_leverage": 2.0, "max_short_leverage": 1.2}),
        ("cta_trend_ensemble", {"vol_target": 0.58, "max_long_leverage": 2.0, "max_short_leverage": 1.2}),
        ("long_only_control", {"vol_target": 0.60, "max_long_leverage": 2.0, "max_short_leverage": 0.0}),
    ]

    split_dt = "2024-06-30 23:59:59+00:00"
    cfg_std = PerpBacktestConfig(taker_fee_bps=5.0, slippage_bps=3.0, rebalance_tol=0.20)
    cfg_stress = PerpBacktestConfig(taker_fee_bps=10.0, slippage_bps=5.0, rebalance_tol=0.20)

    summary: dict[str, Any] = {"data_audit": qa, "strategies": {}, "plateau_check": []}
    print("=== FULL PERIOD (2021-02 to 2026-11) | 8bps/side + Real & Conservative Imputed 8h Funding ===")
    print(f"{'Strategy':<24} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>8} {'Calmar':>7} {'Trades':>7} {'Win%':>6} {'PF':>6} {'Fund$':>9} {'Fees$':>8} {'Liq':>4}")
    for mode, params in modes:
        tgt, sd, td = generate_strategy_signals(df_bt, mode=mode, params=params)
        r_full = run_perp_backtest(df_bt, tgt, sd, td, cfg_std)
        r_is = run_perp_backtest(df_bt.loc[:split_dt], tgt.loc[:split_dt], sd.loc[:split_dt], td.loc[:split_dt], cfg_std)
        r_oos = run_perp_backtest(df_bt.loc[split_dt:], tgt.loc[split_dt:], sd.loc[split_dt:], td.loc[split_dt:], cfg_std)
        r_str = run_perp_backtest(df_bt, tgt, sd, td, cfg_stress)
        m = r_full["metrics"]
        print(
            f"{mode:<24} {m['cagr_pct']:>7.2f} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
            f"{m['max_drawdown_pct']:>8.2f} {m['calmar']:>7.3f} {m['n_trades']:>7d} "
            f"{m['win_rate_pct']:>6.1f} {m['profit_factor']:>6.2f} {m['total_funding_usd']:>9.0f} {m['total_fees_usd']:>8.0f} {m['liquidations']:>4d}"
        )
        summary["strategies"][mode] = {
            "full_8bps": m,
            "is_2021_2024H1": r_is["metrics"],
            "oos_2024H2_2026": r_oos["metrics"],
            "stress_15bps": r_str["metrics"],
        }

    print("\n=== IS (2021-02..2024-06) vs OOS (2024-07..2026-11) & COST STRESS (15bps/side) ===")
    print(f"{'Strategy':<24} {'IS_CAGR':>8} {'IS_Shp':>7} {'IS_DD':>8} | {'OOS_CAGR':>8} {'OOS_Shp':>8} {'OOS_DD':>8} | {'Stress15_Shp':>12}")
    for mode, d in summary["strategies"].items():
        mi, mo, ms = d["is_2021_2024H1"], d["oos_2024H2_2026"], d["stress_15bps"]
        print(
            f"{mode:<24} {mi['cagr_pct']:>8.2f} {mi['sharpe']:>7.3f} {mi['max_drawdown_pct']:>8.2f} | "
            f"{mo['cagr_pct']:>8.2f} {mo['sharpe']:>8.3f} {mo['max_drawdown_pct']:>8.2f} | {ms['sharpe']:>12.3f}"
        )

    print("\n=== PARAMETER PLATEAU CHECK (cta_trend_ensemble) ===")
    for vt in (0.45, 0.55, 0.60, 0.70):
        for ew in (30, 42, 60):
            p = {"vol_target": vt, "donchian_entry_window": ew, "max_long_leverage": 2.0, "max_short_leverage": 1.2}
            tgt, sd, td = generate_strategy_signals(df_bt, mode="cta_trend_ensemble", params=p)
            m = run_perp_backtest(df_bt, tgt, sd, td, cfg_std)["metrics"]
            row = {"vol_target": vt, "donchian_entry": ew, "cagr_pct": m["cagr_pct"], "sharpe": m["sharpe"], "max_dd_pct": m["max_drawdown_pct"], "calmar": m["calmar"], "liq": m["liquidations"]}
            summary["plateau_check"].append(row)
            print(f"  vol_target={vt:.2f}, dc_entry={ew:2d} -> CAGR={m['cagr_pct']:>6.2f}%, Sharpe={m['sharpe']:.3f}, MaxDD={m['max_drawdown_pct']:>6.2f}%, Calmar={m['calmar']:.3f}, Liq={m['liquidations']}")

    (OUT_DIR / "btc_research_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
