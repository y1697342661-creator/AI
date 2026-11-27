"""Portfolio combiner: Long-only (default), Short-only (experimental), and Long+Short independent combination."""
from __future__ import annotations

from typing import Any

import pandas as pd

from mu_quant.backtest.engine import BacktestCosts, ShortSwingParams, run_short_swing_backtest, run_weight_backtest
from mu_quant.long_strategy.strategies import compute_long_weights
from mu_quant.performance.metrics import compute_performance_metrics
from mu_quant.short_strategy.strategies import compute_short_signals


def run_all_portfolio_views(
    f: pd.DataFrame,
    cfg: dict[str, Any],
    earnings: pd.Series | None = None,
    *,
    start: str = "2004-01-01",
    end: str = "2026-12-31",
    long_mode: str | None = None,
    initial: float = 100_000.0,
) -> dict[str, dict[str, Any]]:
    ec = cfg.get("execution", {})
    costs = BacktestCosts(
        commission_bps=float(ec.get("commission_bps", 10.0)),
        slippage_bps=float(ec.get("slippage_bps", 5.0)),
        borrow_annual=float(ec.get("short_borrow_annual", 0.015)),
    )
    sc = cfg.get("short_strategy", {})
    sp = ShortSwingParams(
        stop_atr_mult=float(sc.get("stop_atr_mult", 2.0)),
        trail_atr_mult=float(sc.get("trail_atr_mult", 2.5)),
        target_rr=float(sc.get("target_rr", 2.0)),
        max_hold_days=int(sc.get("max_hold_days", 40)),
        target_vol=float(sc.get("vol_target", 0.35)),
        exit_before_earnings=bool(sc.get("exit_before_earnings", True)),
    )

    w_bh = pd.Series(1.0, index=f.index)
    w_long = compute_long_weights(f, cfg, mode=long_mode)
    w_def = compute_long_weights(f, cfg, mode="ma200_rvol45")
    s_in, s_out = compute_short_signals(f, cfg)

    res_bh = run_weight_backtest(f, w_bh, start=start, end=end, costs=costs, initial=initial)
    res_long = run_weight_backtest(f, w_long, start=start, end=end, costs=costs, initial=initial)
    res_def = run_weight_backtest(f, w_def, start=start, end=end, costs=costs, initial=initial)
    res_short = run_short_swing_backtest(f, s_in, s_out, start=start, end=end, params=sp, costs=costs, earnings_dates=earnings, initial=initial)

    # Independent Long + Short sleeve combination (short weighted conservatively at 0.5 by default)
    sw = float(sc.get("portfolio_weight", 0.5))
    rl = res_long["equity"].pct_change().fillna(0.0)
    rs = res_short["equity"].pct_change().fillna(0.0)
    eq_combo = (1.0 + rl + sw * rs).cumprod() * initial
    w_combo = (res_long["weights"] + sw * res_short["weights"]).clip(-1.0, 1.0)
    combo_trades = sorted(res_long["trades"] + res_short["trades"], key=lambda x: x["entry_date"])
    combo_cost = res_long["metrics"]["total_cost"] + sw * res_short["metrics"]["total_cost"]
    res_combo = {
        "equity": eq_combo,
        "weights": w_combo,
        "trades": combo_trades,
        "metrics": compute_performance_metrics(eq_combo, w_combo, combo_trades, combo_cost, initial),
    }

    return {
        "buy_hold": res_bh,
        "long_primary": res_long,
        "long_defensive": res_def,
        "short_experimental": res_short,
        "long_short_combo": res_combo,
    }
