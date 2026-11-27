"""Strict Bidirectional Backtest Engine.

Hard guarantees:
1. T-day close signal -> T+1 Open execution (never fills at T-day close).
2. Unadjusted split-adjusted OHLC execution + explicit cash dividends (long receives +, short pays -).
3. Short borrow fee accrued daily on open short notional.
4. Gap past stop fills at Open[t] (worse than stop); same-day stop on entry day enforced.
5. Trailing stop updated AFTER day t close, effective day t+1 (only moves in risk-reducing direction).
6. Zero leverage hard cap (|weight| <= 1.0).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mu_quant.config import TRADING_DAYS
from mu_quant.performance.metrics import compute_performance_metrics


@dataclass
class BacktestCosts:
    commission_bps: float = 10.0
    slippage_bps: float = 5.0
    borrow_annual: float = 0.015


@dataclass
class ShortSwingParams:
    stop_atr_mult: float = 2.0
    trail_atr_mult: float | None = 2.5
    target_rr: float | None = 2.0
    max_hold_days: int | None = 40
    target_vol: float = 0.35
    max_pos_pct: float = 0.80
    earnings_blackout_before: int = 3
    earnings_blackout_after: int = 1
    exit_before_earnings: bool = True


def run_weight_backtest(
    f: pd.DataFrame,
    target_w: pd.Series,
    *,
    start: str,
    end: str = "2026-12-31",
    costs: BacktestCosts | None = None,
    rebalance_tol: float = 0.0,
    initial: float = 100_000.0,
) -> dict[str, Any]:
    costs = costs or BacktestCosts()
    sl = f.loc[pd.Timestamp(start):pd.Timestamp(end)]
    w_sig = target_w.reindex(f.index).fillna(0.0).shift(1).reindex(sl.index).fillna(0.0).to_numpy(float)
    o = sl["raw_open"].to_numpy(float)
    c = sl["raw_close"].to_numpy(float)
    div = sl["div"].to_numpy(float)
    idx = sl.index

    cash = initial
    shares = 0.0
    total_cost = 0.0
    equity = np.empty(len(sl))
    eff_w = np.empty(len(sl))
    trades: list[dict[str, Any]] = []
    cur_trade: dict[str, Any] | None = None
    slip = costs.slippage_bps / 1e4
    comm = costs.commission_bps / 1e4

    for i in range(len(sl)):
        if shares != 0.0 and div[i] > 0:
            cash += shares * div[i]

        eq_open = cash + shares * o[i]
        if eq_open <= 0:
            equity[i:] = 0.0
            eff_w[i:] = 0.0
            break

        cur_w = (shares * o[i]) / eq_open
        tw = float(np.clip(w_sig[i], -1.0, 1.0))
        need_rebal = (
            abs(tw - cur_w) > rebalance_tol
            or (tw == 0.0 and shares != 0.0)
            or (np.sign(tw) != np.sign(shares) and tw != 0.0)
        )
        if need_rebal:
            target_shares = (tw * eq_open) / o[i]
            delta = target_shares - shares
            if abs(delta) > 1e-8:
                fill = o[i] * (1.0 + slip) if delta > 0 else o[i] * (1.0 - slip)
                if delta > 0 and shares >= 0:
                    max_affordable = max(0.0, cash) / (fill * (1.0 + comm))
                    if delta > max_affordable:
                        delta = max_affordable
                        target_shares = shares + delta
                if abs(delta) <= 1e-8:
                    delta = 0.0
            if abs(delta) > 1e-8:
                fill = o[i] * (1.0 + slip) if delta > 0 else o[i] * (1.0 - slip)
                fee = abs(delta) * fill * comm
                slip_c = abs(delta) * o[i] * slip
                cash -= delta * fill + fee
                total_cost += fee + slip_c

                old_side = 1 if shares > 1e-8 else (-1 if shares < -1e-8 else 0)
                new_side = 1 if target_shares > 1e-8 else (-1 if target_shares < -1e-8 else 0)
                if old_side != 0 and new_side != old_side and cur_trade is not None:
                    cur_trade["exit_date"] = str(idx[i].date())
                    cur_trade["exit_px"] = float(fill)
                    cur_trade["hold_days"] = int(i - cur_trade["entry_i"])
                    cur_trade["exit_reason"] = "signal_exit"
                    if cur_trade["side"] == "LONG":
                        cur_trade["pnl"] = float((fill - cur_trade["entry_px"]) * cur_trade["shares"] - cur_trade["fees"] - fee)
                    else:
                        cur_trade["pnl"] = float((cur_trade["entry_px"] - fill) * abs(cur_trade["shares"]) - cur_trade["fees"] - fee)
                    trades.append(cur_trade)
                    cur_trade = None
                if new_side != 0 and new_side != old_side:
                    cur_trade = {
                        "side": "LONG" if new_side > 0 else "SHORT",
                        "entry_date": str(idx[i].date()),
                        "entry_i": i,
                        "entry_px": float(fill),
                        "shares": float(target_shares),
                        "fees": float(fee),
                    }
                shares = target_shares

        if shares < 0:
            bfee = abs(shares) * c[i] * (costs.borrow_annual / TRADING_DAYS)
            cash -= bfee
            total_cost += bfee

        eq_close = cash + shares * c[i]
        equity[i] = max(eq_close, 0.0)
        eff_w[i] = (shares * c[i] / eq_close) if eq_close > 0 else 0.0

    if cur_trade is not None and shares != 0:
        fill = c[-1]
        cur_trade["exit_date"] = str(idx[-1].date())
        cur_trade["exit_px"] = float(fill)
        cur_trade["hold_days"] = int(len(sl) - 1 - cur_trade["entry_i"])
        cur_trade["exit_reason"] = "eod_open"
        cur_trade["pnl"] = float((fill - cur_trade["entry_px"]) * cur_trade["shares"] - cur_trade["fees"])
        trades.append(cur_trade)

    eq_s = pd.Series(equity, index=idx)
    w_s = pd.Series(eff_w, index=idx)
    return {
        "equity": eq_s,
        "weights": w_s,
        "trades": trades,
        "metrics": compute_performance_metrics(eq_s, w_s, trades, total_cost, initial),
    }


def run_short_swing_backtest(
    f: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series,
    *,
    start: str,
    end: str = "2026-12-31",
    params: ShortSwingParams | None = None,
    costs: BacktestCosts | None = None,
    earnings_dates: pd.Series | None = None,
    initial: float = 100_000.0,
) -> dict[str, Any]:
    params = params or ShortSwingParams()
    costs = costs or BacktestCosts()
    sl = f.loc[pd.Timestamp(start):pd.Timestamp(end)]
    idx = sl.index
    o = sl["raw_open"].to_numpy(float)
    h = sl["raw_high"].to_numpy(float)
    l = sl["raw_low"].to_numpy(float)
    c = sl["raw_close"].to_numpy(float)
    atr = sl["raw_atr14"].to_numpy(float)
    rvol = sl["rvol21"].to_numpy(float)
    div = sl["div"].to_numpy(float)

    sig_in = entries.reindex(f.index).fillna(False).reindex(idx).fillna(False).to_numpy(bool)
    sig_out = exits.reindex(f.index).fillna(False).reindex(idx).fillna(False).to_numpy(bool)

    block = np.zeros(len(sl), dtype=bool)
    pre_earn = np.zeros(len(sl), dtype=bool)
    if earnings_dates is not None and len(earnings_dates) > 0:
        pos = idx.get_indexer(pd.DatetimeIndex(earnings_dates))
        for p in pos:
            if p < 0:
                continue
            block[max(0, p - params.earnings_blackout_before):min(len(sl), p + params.earnings_blackout_after + 1)] = True
            if p - 1 >= 0:
                pre_earn[max(0, p - 2):p] = True

    slip = costs.slippage_bps / 1e4
    comm = costs.commission_bps / 1e4
    cash = initial
    shares = 0.0
    entry_i, entry_px, stop_px, target_px, init_risk = -1, 0.0, np.nan, np.nan, np.nan
    pending_exit, pending_reason = False, ""
    cur_trade: dict[str, Any] | None = None
    trades: list[dict[str, Any]] = []
    total_cost = 0.0
    equity = np.empty(len(sl))
    eff_w = np.empty(len(sl))

    for i in range(len(sl)):
        if shares < 0 and div[i] > 0:
            cash += shares * div[i]  # short pays dividend in lieu

        if shares < 0:
            exit_px, reason = None, ""
            if pending_exit:
                exit_px, reason = o[i] * (1.0 + slip), pending_reason
                pending_exit = False
            elif o[i] >= stop_px:
                exit_px, reason = o[i] * (1.0 + slip), "gap_stop"
            elif h[i] >= stop_px:
                exit_px, reason = max(o[i], stop_px) * (1.0 + slip), "stop"
            elif np.isfinite(target_px) and l[i] <= target_px:
                exit_px, reason = min(o[i], target_px) * (1.0 + slip), "target"

            if exit_px is not None:
                qty = abs(shares)
                fee = qty * exit_px * comm
                slip_c = qty * abs(exit_px - exit_px / (1.0 + slip))
                cash -= qty * exit_px + fee
                total_cost += fee + slip_c
                pnl = (entry_px - exit_px) * qty - cur_trade["fees"] - fee
                cur_trade.update({
                    "exit_date": str(idx[i].date()), "exit_px": float(exit_px),
                    "exit_reason": reason, "pnl": float(pnl), "hold_days": int(i - entry_i),
                })
                trades.append(cur_trade)
                shares, cur_trade = 0.0, None
                equity[i], eff_w[i] = max(cash, 0.0), 0.0
                continue

            bfee = abs(shares) * c[i] * (costs.borrow_annual / TRADING_DAYS)
            cash -= bfee
            total_cost += bfee
            cur_trade["fees"] += bfee

            # Trailing stop updated AFTER close (moves downward only)
            if params.trail_atr_mult is not None and np.isfinite(atr[i]) and atr[i] > 0:
                stop_px = min(stop_px, l[i] + params.trail_atr_mult * atr[i])

            if sig_out[i]:
                pending_exit, pending_reason = True, "signal_exit"
            elif params.max_hold_days is not None and (i - entry_i) >= params.max_hold_days:
                pending_exit, pending_reason = True, "time_stop"
            elif params.exit_before_earnings and pre_earn[i]:
                pending_exit, pending_reason = True, "pre_earnings"

            eq_c = cash + shares * c[i]
            equity[i] = max(eq_c, 0.0)
            eff_w[i] = (shares * c[i] / eq_c) if eq_c > 0 else 0.0
            continue

        equity[i], eff_w[i] = max(cash, 0.0), 0.0
        if i == 0 or not sig_in[i - 1] or block[i] or cash <= 0:
            continue
        a_prev, rv_prev = atr[i - 1], rvol[i - 1]
        if not (np.isfinite(a_prev) and a_prev > 0 and np.isfinite(rv_prev) and rv_prev > 0 and np.isfinite(o[i]) and o[i] > 0):
            continue

        fill = o[i] * (1.0 - slip)
        st = c[i - 1] + params.stop_atr_mult * a_prev
        if fill >= st:
            continue
        risk_ps = st - fill
        tgt = (fill - params.target_rr * risk_ps) if params.target_rr is not None else np.nan
        w_cap = min(params.max_pos_pct, params.target_vol / rv_prev)
        notional = min(cash * w_cap, cash * params.max_pos_pct / (1.0 + comm))
        qty = notional / fill
        if qty <= 0:
            continue

        fee = notional * comm
        total_cost += fee + qty * o[i] * slip
        cash += notional - fee
        shares = -qty
        entry_i, entry_px, stop_px, target_px, init_risk = i, fill, st, tgt, risk_ps
        cur_trade = {
            "side": "SHORT", "entry_date": str(idx[i].date()),
            "entry_px": float(fill), "shares": float(-qty), "stop_px": float(st), "fees": float(fee),
        }

        if h[i] >= stop_px:
            ex_px = max(o[i], stop_px) * (1.0 + slip)
            fee2 = qty * ex_px * comm
            total_cost += fee2 + qty * stop_px * slip
            cash -= qty * ex_px + fee2
            pnl = (entry_px - ex_px) * qty - fee - fee2
            cur_trade.update({"exit_date": str(idx[i].date()), "exit_px": float(ex_px), "exit_reason": "same_day_stop", "pnl": float(pnl), "hold_days": 0})
            trades.append(cur_trade)
            shares, cur_trade = 0.0, None
            equity[i], eff_w[i] = max(cash, 0.0), 0.0
            continue

        bfee = qty * c[i] * (costs.borrow_annual / TRADING_DAYS)
        cash -= bfee
        total_cost += bfee
        cur_trade["fees"] += bfee
        eq_c = cash + shares * c[i]
        equity[i] = max(eq_c, 0.0)
        eff_w[i] = (shares * c[i] / eq_c) if eq_c > 0 else 0.0

    if shares < 0 and cur_trade is not None:
        qty = abs(shares)
        ex_px = c[-1] * (1.0 + slip)
        fee = qty * ex_px * comm
        total_cost += fee
        cash -= qty * ex_px + fee
        pnl = (entry_px - ex_px) * qty - cur_trade["fees"] - fee
        cur_trade.update({"exit_date": str(idx[-1].date()), "exit_px": float(ex_px), "exit_reason": "eod_open", "pnl": float(pnl), "hold_days": int(len(sl) - 1 - entry_i)})
        trades.append(cur_trade)
        equity[-1] = max(cash, 0.0)

    eq_s = pd.Series(equity, index=idx)
    w_s = pd.Series(eff_w, index=idx)
    return {
        "equity": eq_s,
        "weights": w_s,
        "trades": trades,
        "metrics": compute_performance_metrics(eq_s, w_s, trades, total_cost, initial),
    }
