"""Performance analytics (all 16 required metrics) + Paired Circular Block Bootstrap + BH-FDR."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mu_quant.config import TRADING_DAYS


def compute_performance_metrics(
    equity: pd.Series,
    weights: pd.Series,
    trades: list[dict[str, Any]],
    total_cost: float,
    initial: float = 100_000.0,
) -> dict[str, Any]:
    eq = equity.dropna()
    if len(eq) < 2:
        return {}
    ret = eq.pct_change().dropna()
    years = max(len(eq) / TRADING_DAYS, 1e-9)
    total_ret = float(eq.iloc[-1] / initial - 1.0)
    cagr = float((eq.iloc[-1] / initial) ** (1.0 / years) - 1.0) if eq.iloc[-1] > 0 else -1.0
    ann_vol = float(ret.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(ret) > 1 else 0.0

    # Standard Sortino: downside deviation relative to MAR = 0
    downside_sq = np.minimum(ret.to_numpy(float), 0.0) ** 2
    downside_dev = float(np.sqrt(np.mean(downside_sq)) * np.sqrt(TRADING_DAYS)) if len(ret) else np.nan
    dd = eq / eq.cummax() - 1.0
    max_dd = float(dd.min())
    sharpe = float(ret.mean() / ret.std(ddof=1) * np.sqrt(TRADING_DAYS)) if ret.std(ddof=1) > 0 else 0.0
    sortino = float(ret.mean() * TRADING_DAYS / downside_dev) if (downside_dev and downside_dev > 0) else 0.0
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else np.nan

    n_trades = len(trades)
    if n_trades > 0:
        tdf = pd.DataFrame(trades)
        wins = tdf[tdf["pnl"] > 0]
        losses = tdf[tdf["pnl"] <= 0]
        win_rate = float(len(wins) / n_trades)
        gross_win = float(wins["pnl"].sum())
        gross_loss = float(-losses["pnl"].sum())
        pf = float(gross_win / gross_loss) if gross_loss > 0 else np.nan
        payoff = (
            float(wins["pnl"].mean() / abs(losses["pnl"].mean()))
            if (len(wins) and len(losses) and losses["pnl"].mean() != 0)
            else np.nan
        )
        avg_hold = float(tdf["hold_days"].mean())
        trades_per_yr = float(n_trades / years)
        long_pnl = float(tdf.loc[tdf["side"] == "LONG", "pnl"].sum())
        short_pnl = float(tdf.loc[tdf["side"] == "SHORT", "pnl"].sum())
    else:
        win_rate = pf = payoff = avg_hold = np.nan
        trades_per_yr = 0.0
        long_pnl = short_pnl = 0.0

    w = weights.reindex(eq.index).fillna(0.0)
    dw = w.diff().abs()
    dw.iloc[0] = abs(w.iloc[0])
    ann_turnover = float(dw.sum() / years)
    avg_abs_pos = float(w.abs().mean())

    return {
        "cagr": cagr,
        "total_ret": total_ret,
        "max_dd": max_dd,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "win_rate": win_rate,
        "payoff": payoff,
        "profit_factor": pf,
        "avg_hold_days": avg_hold,
        "n_trades": n_trades,
        "trades_per_yr": trades_per_yr,
        "ann_turnover": ann_turnover,
        "avg_abs_pos": avg_abs_pos,
        "total_cost": float(total_cost),
        "long_pnl": long_pnl,
        "short_pnl": short_pnl,
    }


def paired_circular_block_bootstrap(
    r_a: np.ndarray,
    r_b: np.ndarray,
    block: int = 20,
    n_sims: int = 1000,
    seed: int = 20261126,
) -> dict[str, dict[str, Any]]:
    """Paired Circular Block Bootstrap returning TRUE sample point estimate, 95% CI, and p-value."""
    n = min(len(r_a), len(r_b))
    ra, rb = np.asarray(r_a[-n:], float), np.asarray(r_b[-n:], float)

    def _batch(r: np.ndarray) -> dict[str, np.ndarray]:
        eq = np.cumprod(1.0 + r, axis=1)
        cagr = np.where(eq[:, -1] > 0, eq[:, -1] ** (TRADING_DAYS / n) - 1.0, -1.0)
        peak = np.maximum.accumulate(eq, axis=1)
        mdd = (eq / np.maximum(peak, 1e-12) - 1.0).min(axis=1)
        mu = r.mean(axis=1)
        sd = r.std(axis=1, ddof=1)
        sharpe = np.where(sd > 0, mu / sd * np.sqrt(TRADING_DAYS), 0.0)
        calmar = np.where(mdd < 0, cagr / np.abs(mdd), np.nan)
        return {"cagr": cagr, "mdd": mdd, "calmar": calmar, "sharpe": sharpe}

    pt_a, pt_b = _batch(ra[None, :]), _batch(rb[None, :])
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_sims, nb))
    idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(n_sims, -1)[:, :n]
    sim_a, sim_b = _batch(ra[idx]), _batch(rb[idx])

    out: dict[str, dict[str, Any]] = {}
    for k in ("cagr", "mdd", "calmar", "sharpe"):
        pt_diff = float(pt_a[k][0] - pt_b[k][0])
        diff = sim_a[k] - sim_b[k]
        diff = diff[np.isfinite(diff)]
        lo, hi = float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))
        p_raw = float(min(1.0, 2.0 * min((diff <= 0).mean(), (diff >= 0).mean())))
        out[k] = {"point_diff": pt_diff, "ci_low": lo, "ci_high": hi, "p_raw": p_raw, "contains_zero": bool(lo <= 0 <= hi)}
    return out


def bh_fdr(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    adj = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(adj, 0.0, 1.0)
    return [float(x) for x in out]
