"""Performance & risk attribution metrics for BTC Perpetual Futures strategies."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_perp_metrics(
    equity: pd.Series,
    trades: pd.DataFrame,
    funding_pnl_series: pd.Series | None = None,
    fee_series: pd.Series | None = None,
    liquidations: int = 0,
    bars_per_year: int = 365 * 6,
) -> dict[str, Any]:
    eq = equity.dropna()
    if len(eq) < 2:
        return {}

    rets = eq.pct_change().fillna(0.0)
    total_ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    days = max((eq.index[-1] - eq.index[0]).total_seconds() / 86400.0, 1.0)
    years = days / 365.25
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0) if eq.iloc[-1] > 0 else -1.0

    # Compute daily resampled returns for standard Sharpe/Sortino comparability
    daily_eq = eq.resample("1D").last().ffill().dropna()
    d_rets = daily_eq.pct_change().dropna()
    vol_ann = float(d_rets.std() * np.sqrt(365.0)) if len(d_rets) > 1 else 0.0
    sharpe = float((d_rets.mean() * 365.0) / vol_ann) if vol_ann > 1e-12 else 0.0
    down_rets = d_rets[d_rets < 0]
    down_vol = float(np.sqrt((down_rets ** 2).mean()) * np.sqrt(365.0)) if len(down_rets) > 0 else 0.0
    sortino = float((d_rets.mean() * 365.0) / down_vol) if down_vol > 1e-12 else 0.0

    running_max = eq.cummax()
    dd = eq / running_max - 1.0
    max_dd = float(dd.min())
    calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 1e-12 else 0.0

    n_trades = len(trades) if trades is not None else 0
    if n_trades > 0 and "pnl_pct" in trades.columns:
        wins = trades[trades["pnl_pct"] > 0]
        losses = trades[trades["pnl_pct"] <= 0]
        win_rate = float(len(wins) / n_trades)
        gross_win = float(wins["pnl_usd"].sum()) if "pnl_usd" in trades.columns else float(wins["pnl_pct"].sum())
        gross_loss = abs(float(losses["pnl_usd"].sum())) if "pnl_usd" in trades.columns else abs(float(losses["pnl_pct"].sum()))
        profit_factor = float(gross_win / gross_loss) if gross_loss > 1e-12 else 99.0
        avg_hold_hours = float(trades["hold_bars"].mean() * (24.0 / (bars_per_year / 365.0))) if "hold_bars" in trades.columns else 0.0
        long_trades = int((trades["side"] == "LONG").sum()) if "side" in trades.columns else 0
        short_trades = int((trades["side"] == "SHORT").sum()) if "side" in trades.columns else 0
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_hold_hours = 0.0
        long_trades = 0
        short_trades = 0

    total_funding_usd = float(funding_pnl_series.sum()) if funding_pnl_series is not None else 0.0
    total_fees_usd = float(fee_series.sum()) if fee_series is not None else 0.0

    # Yearly returns breakdown
    yearly = {}
    for yr, grp in daily_eq.groupby(daily_eq.index.year):
        if len(grp) >= 2:
            yearly[str(yr)] = round(float(grp.iloc[-1] / grp.iloc[0] - 1.0) * 100.0, 2)

    return {
        "total_return_pct": round(total_ret * 100.0, 2),
        "cagr_pct": round(cagr * 100.0, 2),
        "ann_vol_pct": round(vol_ann * 100.0, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "n_trades": n_trades,
        "long_trades": long_trades,
        "short_trades": short_trades,
        "win_rate_pct": round(win_rate * 100.0, 2),
        "profit_factor": round(profit_factor, 3),
        "avg_hold_hours": round(avg_hold_hours, 1),
        "total_funding_usd": round(total_funding_usd, 2),
        "total_fees_usd": round(total_fees_usd, 2),
        "liquidations": int(liquidations),
        "final_equity": round(float(eq.iloc[-1]), 2),
        "yearly_returns_pct": yearly,
    }
