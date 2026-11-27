"""Event-Driven Perpetual Futures Backtest Engine (BTC-USDT-SWAP).

REALISTIC PERPETUAL CONTRACT MECHANICS:
1. Zero Look-Ahead: Signal generated on bar T close -> executed at bar T+1 open.
2. Transaction Costs: Configurable taker fee (default 5 bps) + slippage (default 3 bps) per side.
3. Real 8-Hour Funding Cash Flows: At every bar carrying a funding settlement,
   `funding_cashflow = -position_qty * mark_price * funding_rate`
   (longs pay when funding > 0, shorts receive when funding > 0).
4. Maintenance Margin & Liquidation Engine: Checked against intrabar worst price
   (low for LONG, high for SHORT); liquidation triggers forced close + penalty.
5. Intrabar Hard Stop-Loss & Trailing ATR Stop: Checked on intrabar low/high with stop slippage.
6. Drawdown Circuit Breaker: Automatically scales down leverage when drawdown exceeds threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from btc_quant.performance.metrics import compute_perp_metrics


@dataclass
class PerpBacktestConfig:
    initial_capital: float = 100_000.0
    taker_fee_bps: float = 5.0          # OKX standard taker ~5 bps
    slippage_bps: float = 3.0           # Conservative BTC-USDT-SWAP slippage
    stop_slippage_bps: float = 8.0      # Extra slippage on intrabar stop-loss fills
    maintenance_margin_rate: float = 0.004  # OKX Tier 1 MMR = 0.4%
    liquidation_penalty_pct: float = 0.015  # 1.5% liquidation/insurance fee
    max_long_leverage: float = 2.0
    max_short_leverage: float = 1.5
    dd_brake_start: float = 0.12        # Start reducing leverage at 12% drawdown
    dd_brake_full: float = 0.25         # Max reduction at 25% drawdown
    dd_min_scale: float = 0.35          # Floor scale during deep drawdown
    rebalance_tol: float = 0.15         # Avoid micro-rebalance churn if target weight delta < 15%
    use_funding: bool = True


def run_perp_backtest(
    df: pd.DataFrame,
    target_leverage: pd.Series,
    stop_distance_pct: pd.Series | None = None,
    trail_distance_pct: pd.Series | None = None,
    cfg: PerpBacktestConfig | None = None,
) -> dict[str, Any]:
    """Run bar-by-bar perpetual futures backtest.

    Parameters
    ----------
    df : DataFrame with columns open, high, low, close, funding_rate
    target_leverage : Signed target notional / equity decided at bar T CLOSE
                      (+1.5 = 1.5x long, -1.0 = 1.0x short, 0.0 = flat).
                      Executed at bar T+1 OPEN!
    stop_distance_pct : Optional initial hard stop distance (e.g. 0.06 = 6% from entry).
    trail_distance_pct : Optional trailing stop distance from favorable extreme.
    """
    c = cfg or PerpBacktestConfig()
    idx = df.index
    n = len(df)
    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    fundings = df["funding_rate"].to_numpy(dtype=float) if ("funding_rate" in df.columns and c.use_funding) else np.zeros(n)

    # Shift target_leverage by 1 bar: decision made at T close -> fill at T+1 open!
    exec_target = target_leverage.reindex(idx).fillna(0.0).shift(1).fillna(0.0).to_numpy(dtype=float)
    stop_arr = stop_distance_pct.reindex(idx).shift(1).to_numpy(dtype=float) if stop_distance_pct is not None else np.full(n, np.nan)
    trail_arr = trail_distance_pct.reindex(idx).shift(1).to_numpy(dtype=float) if trail_distance_pct is not None else np.full(n, np.nan)

    fee_rate = (c.taker_fee_bps + c.slippage_bps) / 1e4
    stop_fee_rate = (c.taker_fee_bps + c.stop_slippage_bps) / 1e4

    cash = c.initial_capital          # Realized USDT balance
    pos_qty = 0.0                     # Signed BTC quantity (+ long, - short)
    entry_px = 0.0
    entry_bar = -1
    entry_side = ""
    best_px = 0.0                     # Highest high since long entry / lowest low since short entry
    stop_px = np.nan
    trade_funding = 0.0
    trade_fees = 0.0
    peak_equity = c.initial_capital
    liquidations = 0
    stopped_this_dir = 0              # Prevent immediate re-entry on same bar after stop

    equity_curve = np.zeros(n)
    leverage_curve = np.zeros(n)
    funding_curve = np.zeros(n)
    fee_curve = np.zeros(n)
    dd_scale_curve = np.ones(n)
    trades: list[dict[str, Any]] = []

    for i in range(n):
        o, h, l, cl, fr = opens[i], highs[i], lows[i], closes[i], fundings[i]

        # 1. Funding settlement at bar open (00:00 / 08:00 / 16:00 UTC) for carried position
        if pos_qty != 0.0 and fr != 0.0:
            f_cf = -pos_qty * o * fr
            cash += f_cf
            trade_funding += f_cf
            funding_curve[i] += f_cf

        # Mark-to-market equity at bar open
        eq_open = cash + pos_qty * (o - entry_px)
        if eq_open <= 0.0:
            eq_open = 0.0
            pos_qty = 0.0
        peak_equity = max(peak_equity, eq_open)
        cur_dd = 1.0 - (eq_open / peak_equity) if peak_equity > 0 else 0.0

        # Drawdown circuit breaker multiplier
        if cur_dd <= c.dd_brake_start:
            dd_scale = 1.0
        elif cur_dd >= c.dd_brake_full:
            dd_scale = c.dd_min_scale
        else:
            frac = (cur_dd - c.dd_brake_start) / max(c.dd_brake_full - c.dd_brake_start, 1e-9)
            dd_scale = 1.0 - frac * (1.0 - c.dd_min_scale)
        dd_scale_curve[i] = dd_scale

        # 2. Signal execution at bar OPEN
        raw_tgt = exec_target[i]
        # Reset stop lockout once signal turns flat or flips
        if stopped_this_dir != 0 and (raw_tgt == 0.0 or np.sign(raw_tgt) != stopped_this_dir):
            stopped_this_dir = 0
        if stopped_this_dir != 0 and np.sign(raw_tgt) == stopped_this_dir:
            raw_tgt = 0.0

        # Clamp leverage by hard risk caps & drawdown brake
        if raw_tgt > 0:
            adj_tgt = min(raw_tgt, c.max_long_leverage) * dd_scale
        elif raw_tgt < 0:
            adj_tgt = max(raw_tgt, -c.max_short_leverage) * dd_scale
        else:
            adj_tgt = 0.0

        cur_lev = (pos_qty * o / eq_open) if eq_open > 0 else 0.0
        flip_or_exit = (pos_qty != 0.0) and (adj_tgt == 0.0 or np.sign(adj_tgt) != np.sign(pos_qty))
        need_rebal = flip_or_exit or (pos_qty == 0.0 and abs(adj_tgt) > 1e-6) or (
            pos_qty != 0.0 and abs(adj_tgt - cur_lev) >= c.rebalance_tol
        )

        if need_rebal and eq_open > 1.0:
            # Close existing position if exiting or flipping
            if flip_or_exit:
                realized = pos_qty * (o - entry_px)
                fee = abs(pos_qty) * o * fee_rate
                cash += realized - fee
                fee_curve[i] += fee
                trade_fees += fee
                net_pnl = realized + trade_funding - trade_fees
                notional_entry = abs(pos_qty) * entry_px
                trades.append(
                    {
                        "entry_time": str(idx[entry_bar]),
                        "exit_time": str(idx[i]),
                        "side": entry_side,
                        "entry_px": round(entry_px, 2),
                        "exit_px": round(o, 2),
                        "qty_btc": round(abs(pos_qty), 5),
                        "pnl_usd": round(net_pnl, 2),
                        "pnl_pct": round(net_pnl / max(notional_entry, 1.0), 5),
                        "funding_usd": round(trade_funding, 2),
                        "fees_usd": round(trade_fees, 2),
                        "hold_bars": i - entry_bar,
                        "exit_reason": "SIGNAL_FLIP" if adj_tgt != 0.0 else "SIGNAL_EXIT",
                    }
                )
                pos_qty = 0.0
                entry_px = 0.0
                stop_px = np.nan
                trade_funding = 0.0
                trade_fees = 0.0
                eq_open = cash

            # Open new or resize existing position
            if abs(adj_tgt) > 1e-6 and eq_open > 1.0:
                target_qty = (adj_tgt * eq_open) / o
                delta_qty = target_qty - pos_qty
                if abs(delta_qty) > 1e-9:
                    fee = abs(delta_qty) * o * fee_rate
                    if pos_qty == 0.0:
                        cash -= fee
                        pos_qty = target_qty
                        entry_px = o
                        entry_bar = i
                        entry_side = "LONG" if pos_qty > 0 else "SHORT"
                        best_px = o
                        trade_funding = 0.0
                        trade_fees = fee
                        sd = stop_arr[i]
                        if np.isfinite(sd) and sd > 0:
                            stop_px = o * (1.0 - sd) if pos_qty > 0 else o * (1.0 + sd)
                        else:
                            stop_px = np.nan
                    else:
                        # Same-direction resize: realize PnL on reduction or blend avg entry on add
                        if np.sign(delta_qty) == np.sign(pos_qty):
                            new_qty = pos_qty + delta_qty
                            entry_px = (abs(pos_qty) * entry_px + abs(delta_qty) * o) / abs(new_qty)
                            pos_qty = new_qty
                            cash -= fee
                        else:
                            closed_qty = -delta_qty  # signed same as pos_qty
                            realized = closed_qty * (o - entry_px)
                            cash += realized - fee
                            pos_qty = target_qty
                        trade_fees += fee
                    fee_curve[i] += fee

        # 3. Intrabar Maintenance Margin / Liquidation Check
        if pos_qty != 0.0:
            worst_px = l if pos_qty > 0 else h
            worst_notional = abs(pos_qty) * worst_px
            worst_eq = cash + pos_qty * (worst_px - entry_px)
            maint_margin = worst_notional * c.maintenance_margin_rate
            if worst_eq <= maint_margin:
                # Liquidated!
                liquidations += 1
                realized = pos_qty * (worst_px - entry_px)
                liq_fee = worst_notional * c.liquidation_penalty_pct
                cash = max(0.0, cash + realized - liq_fee)
                fee_curve[i] += liq_fee
                trade_fees += liq_fee
                net_pnl = realized + trade_funding - trade_fees
                trades.append(
                    {
                        "entry_time": str(idx[entry_bar]),
                        "exit_time": str(idx[i]),
                        "side": entry_side,
                        "entry_px": round(entry_px, 2),
                        "exit_px": round(worst_px, 2),
                        "qty_btc": round(abs(pos_qty), 5),
                        "pnl_usd": round(net_pnl, 2),
                        "pnl_pct": round(net_pnl / max(abs(pos_qty) * entry_px, 1.0), 5),
                        "funding_usd": round(trade_funding, 2),
                        "fees_usd": round(trade_fees, 2),
                        "hold_bars": i - entry_bar,
                        "exit_reason": "LIQUIDATION",
                    }
                )
                stopped_this_dir = int(np.sign(pos_qty))
                pos_qty = 0.0
                entry_px = 0.0
                stop_px = np.nan
                trade_funding = 0.0
                trade_fees = 0.0

        # 4. Intrabar Hard Stop-Loss / Trailing Stop Check
        if pos_qty != 0.0 and np.isfinite(stop_px):
            hit_stop = (pos_qty > 0 and l <= stop_px) or (pos_qty < 0 and h >= stop_px)
            if hit_stop:
                fill_px = min(o, stop_px) if pos_qty > 0 else max(o, stop_px)
                realized = pos_qty * (fill_px - entry_px)
                fee = abs(pos_qty) * fill_px * stop_fee_rate
                cash += realized - fee
                fee_curve[i] += fee
                trade_fees += fee
                net_pnl = realized + trade_funding - trade_fees
                trades.append(
                    {
                        "entry_time": str(idx[entry_bar]),
                        "exit_time": str(idx[i]),
                        "side": entry_side,
                        "entry_px": round(entry_px, 2),
                        "exit_px": round(fill_px, 2),
                        "qty_btc": round(abs(pos_qty), 5),
                        "pnl_usd": round(net_pnl, 2),
                        "pnl_pct": round(net_pnl / max(abs(pos_qty) * entry_px, 1.0), 5),
                        "funding_usd": round(trade_funding, 2),
                        "fees_usd": round(trade_fees, 2),
                        "hold_bars": i - entry_bar,
                        "exit_reason": "STOP_LOSS",
                    }
                )
                stopped_this_dir = int(np.sign(pos_qty))
                pos_qty = 0.0
                entry_px = 0.0
                stop_px = np.nan
                trade_funding = 0.0
                trade_fees = 0.0
            else:
                # Update trailing stop after bar closes safely
                td = trail_arr[i]
                if np.isfinite(td) and td > 0:
                    if pos_qty > 0:
                        best_px = max(best_px, h)
                        new_stop = best_px * (1.0 - td)
                        stop_px = max(stop_px, new_stop) if np.isfinite(stop_px) else new_stop
                    else:
                        best_px = min(best_px, l)
                        new_stop = best_px * (1.0 + td)
                        stop_px = min(stop_px, new_stop) if np.isfinite(stop_px) else new_stop

        # 5. End-of-bar mark-to-market
        eq_close = max(0.0, cash + pos_qty * (cl - entry_px))
        equity_curve[i] = eq_close
        leverage_curve[i] = (pos_qty * cl / eq_close) if eq_close > 0 else 0.0
        peak_equity = max(peak_equity, eq_close)

    eq_s = pd.Series(equity_curve, index=idx, name="equity")
    lev_s = pd.Series(leverage_curve, index=idx, name="leverage")
    fund_s = pd.Series(funding_curve, index=idx, name="funding_usd")
    fee_s = pd.Series(fee_curve, index=idx, name="fee_usd")
    trades_df = pd.DataFrame(trades)
    metrics = compute_perp_metrics(
        eq_s, trades_df, funding_pnl_series=fund_s, fee_series=fee_s, liquidations=liquidations
    )
    return {
        "equity": eq_s,
        "leverage": lev_s,
        "funding_usd": fund_s,
        "fee_usd": fee_s,
        "dd_scale": pd.Series(dd_scale_curve, index=idx, name="dd_scale"),
        "trades": trades_df,
        "metrics": metrics,
    }
