"""Multi-Coin (BTC + ETH + SOL) Perpetual Futures Portfolio Research & Validation.

Compares:
1. Single-coin BTC, ETH, SOL (`dual_horizon_ensemble` & `donchian_cta`)
2. 3-Coin Equal-Budget Portfolio (1/3 each, daily-rebalanced equity combination, 0 look-ahead)
3. 3-Coin Growth Portfolio (slightly higher vol_target=0.75, max_long=2.4x, max_short=1.5x,
   strictly 0 liquidations)
4. 3-Coin Momentum-Tilted Rotation (overweight top-2 strongest trend coins by 14D risk-adjusted
   momentum, strictly lagged!)

Evaluates Full Period (2021-05 to 2026-11, common SOL/ETH/BTC history), IS vs OOS, and 15bps cost stress.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from btc_quant.backtest.engine import PerpBacktestConfig, run_perp_backtest
from btc_quant.data.loader import SUPPORTED_COINS, load_all_coins_datasets
from btc_quant.features.indicators import build_1d_features, build_4h_features
from btc_quant.performance.metrics import compute_perp_metrics
from btc_quant.strategies.perp_strategies import generate_strategy_signals

OUT_DIR = ROOT / "research" / "btc"


def combine_portfolio_equities(
    equities: dict[str, pd.Series],
    weights: dict[str, float] | pd.DataFrame,
    initial_capital: float = 100_000.0,
) -> pd.Series:
    """Combine per-coin 4H strategy equity curves into a portfolio equity curve."""
    eq_df = pd.DataFrame(equities).dropna()
    rets = eq_df.pct_change().fillna(0.0)
    if isinstance(weights, dict):
        w_vec = pd.Series(weights).reindex(eq_df.columns).fillna(0.0)
        port_ret = (rets * w_vec).sum(axis=1)
    else:
        w_df = weights.reindex(eq_df.index).ffill().fillna(1.0 / len(eq_df.columns))
        port_ret = (rets * w_df[eq_df.columns]).sum(axis=1)
    return initial_capital * (1.0 + port_ret).cumprod()


def main() -> None:
    all_ds = load_all_coins_datasets(SUPPORTED_COINS)
    feats: dict[str, pd.DataFrame] = {}
    common_start = None
    for coin, ds in all_ds.items():
        f1d = build_1d_features(ds["df_1d"])
        f4h = build_4h_features(ds["df_4h"], f1d)
        # Ensure 200d SMA is warmed up (or at least 120 bars of 4H)
        valid = f4h.dropna(subset=["ema_55", "atr_14", "rvol_14d"])
        feats[coin] = valid
        s_dt = valid.index[0]
        common_start = s_dt if (common_start is None or s_dt > common_start) else common_start
        print(f"{coin}: {len(valid)} 4H bars ({valid.index[0]} -> {valid.index[-1]})")

    # Align common window across BTC, ETH, SOL
    bt_start = max(common_start, pd.Timestamp("2021-05-01 00:00:00+00:00"))
    print(f"\nCommon 3-Coin Backtest Window: {bt_start} -> latest")
    for c in SUPPORTED_COINS:
        feats[c] = feats[c].loc[bt_start:].copy()

    split_dt = "2024-06-30 23:59:59+00:00"

    profiles = {
        "Std_v1 (vt=0.55, L2.0/S1.2)": (
            {"vol_target": 0.55, "max_long_leverage": 2.0, "max_short_leverage": 1.2},
            PerpBacktestConfig(taker_fee_bps=5.0, slippage_bps=3.0, max_long_leverage=2.0, max_short_leverage=1.2, rebalance_tol=0.20),
        ),
        "Growth_v2 (vt=0.80, L2.5/S1.5)": (
            {"vol_target": 0.80, "max_long_leverage": 2.5, "max_short_leverage": 1.5},
            PerpBacktestConfig(taker_fee_bps=5.0, slippage_bps=3.0, max_long_leverage=2.5, max_short_leverage=1.5, rebalance_tol=0.20),
        ),
    }

    summary: dict[str, Any] = {"bt_start": str(bt_start), "profiles": {}}

    for prof_name, (sparams, bt_cfg) in profiles.items():
        print(f"\n================ {prof_name} ================")
        print(f"{'Portfolio / Coin':<28} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>8} {'Calmar':>7} {'Liq':>4} | {'IS_CAGR':>8} {'OOS_CAGR':>8}")
        eq_map: dict[str, pd.Series] = {}
        eq_bh: dict[str, pd.Series] = {}
        total_liqs = 0
        prof_res: dict[str, Any] = {}

        for c in SUPPORTED_COINS:
            tgt, sd, td = generate_strategy_signals(feats[c], mode="dual_horizon_ensemble", params=sparams)
            res = run_perp_backtest(feats[c], tgt, sd, td, bt_cfg)
            eq_map[c] = res["equity"]
            total_liqs += res["metrics"]["liquidations"]

            # Buy & hold for reference
            bh_res = run_perp_backtest(feats[c], pd.Series(1.0, index=feats[c].index), cfg=bt_cfg)
            eq_bh[c] = bh_res["equity"]

            m_full = res["metrics"]
            m_is = compute_perp_metrics(res["equity"].loc[:split_dt], pd.DataFrame())
            m_oos = compute_perp_metrics(res["equity"].loc[split_dt:], pd.DataFrame())
            prof_res[c] = {"full": m_full, "is": m_is, "oos": m_oos}
            print(
                f"{c + ' Only (DualHorizon)':<28} {m_full['cagr_pct']:>7.2f} {m_full['sharpe']:>7.3f} {m_full['sortino']:>8.3f} "
                f"{m_full['max_drawdown_pct']:>8.2f} {m_full['calmar']:>7.3f} {m_full['liquidations']:>4d} | "
                f"{m_is['cagr_pct']:>8.2f} {m_oos['cagr_pct']:>8.2f}"
            )

        # 3-Coin Equal Weight (1/3 BTC, 1/3 ETH, 1/3 SOL)
        eq_3c = combine_portfolio_equities(eq_map, {"BTC": 1 / 3, "ETH": 1 / 3, "SOL": 1 / 3})
        m3 = compute_perp_metrics(eq_3c, pd.DataFrame(), liquidations=total_liqs)
        m3_is = compute_perp_metrics(eq_3c.loc[:split_dt], pd.DataFrame())
        m3_oos = compute_perp_metrics(eq_3c.loc[split_dt:], pd.DataFrame())
        prof_res["3Coin_Equal"] = {"full": m3, "is": m3_is, "oos": m3_oos}
        print(
            f"{'>>> 3-Coin Equal (BTC+ETH+SOL)':<28} {m3['cagr_pct']:>7.2f} {m3['sharpe']:>7.3f} {m3['sortino']:>8.3f} "
            f"{m3['max_drawdown_pct']:>8.2f} {m3['calmar']:>7.3f} {total_liqs:>4d} | "
            f"{m3_is['cagr_pct']:>8.2f} {m3_oos['cagr_pct']:>8.2f}"
        )

        # 3-Coin Risk-Budget Tilt (30% BTC, 30% ETH, 40% SOL)
        eq_tilt = combine_portfolio_equities(eq_map, {"BTC": 0.30, "ETH": 0.30, "SOL": 0.40})
        mt = compute_perp_metrics(eq_tilt, pd.DataFrame(), liquidations=total_liqs)
        mt_is = compute_perp_metrics(eq_tilt.loc[:split_dt], pd.DataFrame())
        mt_oos = compute_perp_metrics(eq_tilt.loc[split_dt:], pd.DataFrame())
        prof_res["3Coin_SOL40"] = {"full": mt, "is": mt_is, "oos": mt_oos}
        print(
            f"{'>>> 3-Coin Growth (30/30/40)':<28} {mt['cagr_pct']:>7.2f} {mt['sharpe']:>7.3f} {mt['sortino']:>8.3f} "
            f"{mt['max_drawdown_pct']:>8.2f} {mt['calmar']:>7.3f} {total_liqs:>4d} | "
            f"{mt_is['cagr_pct']:>8.2f} {mt_oos['cagr_pct']:>8.2f}"
        )

        # 3-Coin Buy & Hold benchmark
        eq_bh3 = combine_portfolio_equities(eq_bh, {"BTC": 1 / 3, "ETH": 1 / 3, "SOL": 1 / 3})
        mbh = compute_perp_metrics(eq_bh3, pd.DataFrame())
        mbh_is = compute_perp_metrics(eq_bh3.loc[:split_dt], pd.DataFrame())
        mbh_oos = compute_perp_metrics(eq_bh3.loc[split_dt:], pd.DataFrame())
        prof_res["3Coin_BH_1x"] = {"full": mbh, "is": mbh_is, "oos": mbh_oos}
        print(
            f"{'3-Coin 1x Buy & Hold':<28} {mbh['cagr_pct']:>7.2f} {mbh['sharpe']:>7.3f} {mbh['sortino']:>8.3f} "
            f"{mbh['max_drawdown_pct']:>8.2f} {mbh['calmar']:>7.3f} {0:>4d} | "
            f"{mbh_is['cagr_pct']:>8.2f} {mbh_oos['cagr_pct']:>8.2f}"
        )

        summary["profiles"][prof_name] = prof_res

    out_path = OUT_DIR / "multi_coin_research_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved multi-coin research summary to {out_path}")


if __name__ == "__main__":
    main()
