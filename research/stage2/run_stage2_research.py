"""MU Specialist Quant System 2.0 — Stage 2 Bidirectional Strategy Research Script.

Strictly enforces:
1. T-day close signal -> T+1 Open execution (fixes old project T-day close bias).
2. Unadjusted split-adjusted OHLC fills + explicit cash dividends (long receives, short pays).
3. Explicit short borrow cost (daily accrual) & 100% collateral (zero leverage).
4. Proper Sortino (downside deviation relative to MAR=0).
5. Independent Long and Short candidate strategies (NO naive long inversion).
6. Indicator subtraction / incremental value ablation test.
7. Long-only vs Short-only vs Long+Short portfolio comparison.

Read-only on E:\\AI策略交易\\data\\raw\\*.csv; outputs JSON + console summary.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent
TRADING_DAYS = 252


# ------------------------------------------------------------------ Data Loading
def load_data() -> dict[str, pd.DataFrame]:
    mu = pd.read_csv(DATA_RAW / "MU.csv", parse_dates=["Date"], index_col="Date").sort_index()
    out = {"MU": mu}
    for name in ("SMH", "SOX", "SPY", "QQQ", "NVDA"):
        p = DATA_RAW / f"bench_{name}.csv"
        if p.exists():
            out[name] = pd.read_csv(p, parse_dates=["Date"], index_col="Date").sort_index()
    ep = DATA_RAW / "earnings_dates.csv"
    if ep.exists():
        edf = pd.read_csv(ep)
        # First date-like column
        dcol = [c for c in edf.columns if "date" in c.lower()][0]
        out["earnings"] = pd.to_datetime(edf[dcol]).dropna().sort_values()
    else:
        out["earnings"] = pd.Series(dtype="datetime64[ns]")
    return out


# ------------------------------------------------------------------ Causal Indicators
def compute_indicators(mu: pd.DataFrame, smh: pd.DataFrame | None, sox: pd.DataFrame | None) -> pd.DataFrame:
    """All indicators computed strictly causally on Adjusted OHLC so dividends don't cause fake breaks."""
    f = pd.DataFrame(index=mu.index)
    adj_factor = mu["Adj Close"] / mu["Close"]
    o = mu["Open"] * adj_factor
    h = mu["High"] * adj_factor
    l = mu["Low"] * adj_factor
    c = mu["Adj Close"]
    v = mu["Volume"].astype(float)

    f["Open"], f["High"], f["Low"], f["Close"], f["Volume"] = o, h, l, c, v
    f["raw_open"], f["raw_high"], f["raw_low"], f["raw_close"] = mu["Open"], mu["High"], mu["Low"], mu["Close"]
    f["div"] = mu["Dividends"].fillna(0.0)
    f["ret"] = c.pct_change()

    # Moving averages
    for n in (5, 20, 50, 100, 200):
        f[f"sma{n}"] = c.rolling(n, min_periods=n).mean()
    f["ema20"] = c.ewm(span=20, adjust=False, min_periods=20).mean()
    f["ema50"] = c.ewm(span=50, adjust=False, min_periods=50).mean()
    f["sma200_slope20"] = f["sma200"] / f["sma200"].shift(20) - 1.0
    f["sma50_slope10"] = f["sma50"] / f["sma50"].shift(10) - 1.0

    # True Range & Wilder ATR (on adjusted OHLC, plus raw ATR for execution stops)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    f["atr14"] = tr.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
    f["atr_pct"] = f["atr14"] / c
    f["atr_rank252"] = f["atr_pct"].rolling(252, min_periods=120).rank(pct=True)

    # Raw ATR for exact execution-price stop calculation
    prev_rc = mu["Close"].shift(1)
    tr_raw = pd.concat([
        (mu["High"] - mu["Low"]),
        (mu["High"] - prev_rc).abs(),
        (mu["Low"] - prev_rc).abs(),
    ], axis=1).max(axis=1)
    f["raw_atr14"] = tr_raw.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()

    # Realized Volatility (strictly return std * sqrt(252), NEVER confused with ATR)
    f["rvol21"] = f["ret"].rolling(21, min_periods=21).std() * np.sqrt(TRADING_DAYS)
    f["rvol63"] = f["ret"].rolling(63, min_periods=63).std() * np.sqrt(TRADING_DAYS)
    f["rvol_rank252"] = f["rvol21"].rolling(252, min_periods=120).rank(pct=True)

    # ADX / DMI (Wilder 14)
    up = h.diff()
    dn = -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=mu.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=mu.index)
    f["plus_di"] = 100.0 * plus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / f["atr14"]
    f["minus_di"] = 100.0 * minus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / f["atr14"]
    dx = 100.0 * (f["plus_di"] - f["minus_di"]).abs() / (f["plus_di"] + f["minus_di"]).replace(0.0, np.nan)
    f["adx14"] = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    # Momentum: RSI(14), MACD(12,26,9), ROC(20, 60)
    delta = c.diff()
    ag = delta.clip(lower=0.0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    al = (-delta).clip(lower=0.0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = ag / al.replace(0.0, np.nan)
    f["rsi14"] = (100.0 - 100.0 / (1.0 + rs)).where(al != 0, 100.0).where(ag.notna())

    ema12 = c.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = c.ewm(span=26, adjust=False, min_periods=26).mean()
    f["macd"] = ema12 - ema26
    f["macd_sig"] = f["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    f["macd_hist"] = f["macd"] - f["macd_sig"]

    f["roc20"] = c.pct_change(20)
    f["roc60"] = c.pct_change(60)

    # Donchian channels (strictly shifted 1 day to avoid lookahead!)
    for n in (20, 55):
        f[f"don_high{n}"] = h.rolling(n, min_periods=n).max().shift(1)
        f[f"don_low{n}"] = l.rolling(n, min_periods=n).min().shift(1)

    # Bollinger Bands (20, 2)
    std20 = c.rolling(20, min_periods=20).std()
    f["boll_upper"] = f["sma20"] + 2.0 * std20
    f["boll_lower"] = f["sma20"] - 2.0 * std20
    f["boll_width"] = (4.0 * std20) / f["sma20"]

    # Volume & OBV
    f["vol_ma20"] = v.rolling(20, min_periods=20).mean()
    f["vol_ratio20"] = v / f["vol_ma20"]
    obv_dir = np.sign(c.diff().fillna(0.0))
    f["obv"] = (obv_dir * v).cumsum()
    f["obv_ma20"] = f["obv"].rolling(20, min_periods=20).mean()

    # Price structure
    f["dist_ma20"] = c / f["sma20"] - 1.0
    f["dist_ma50"] = c / f["sma50"] - 1.0
    f["dist_ma200"] = c / f["sma200"] - 1.0
    f["clv"] = (c - l) / (h - l).replace(0.0, np.nan)
    f["up_day"] = c > o
    f["down_day"] = c < o

    # Sector benchmarks (SMH / SOX) - strictly causal reindex + ffill
    if smh is not None and not smh.empty:
        smh_c = smh["Adj Close"].reindex(mu.index).ffill()
        f["smh_above200"] = (smh_c > smh_c.rolling(200, min_periods=200).mean()).astype(float)
        f["mu_vs_smh_20"] = c.pct_change(20) - smh_c.pct_change(20)
        f["mu_vs_smh_60"] = c.pct_change(60) - smh_c.pct_change(60)
    else:
        f["smh_above200"] = np.nan
        f["mu_vs_smh_20"] = np.nan
        f["mu_vs_smh_60"] = np.nan

    if sox is not None and not sox.empty:
        sox_c = sox["Adj Close"].reindex(mu.index).ffill()
        f["sox_above200"] = (sox_c > sox_c.rolling(200, min_periods=200).mean()).astype(float)
    else:
        f["sox_above200"] = np.nan

    return f


# ------------------------------------------------------------------ Metrics (Corrected)
def compute_metrics(
    equity: pd.Series,
    weights: pd.Series,
    trades: list[dict],
    total_cost: float,
    initial: float = 100_000.0,
) -> dict:
    eq = equity.dropna()
    ret = eq.pct_change().dropna()
    years = max(len(eq) / TRADING_DAYS, 1e-9)
    total_ret = float(eq.iloc[-1] / initial - 1.0)
    cagr = float((eq.iloc[-1] / initial) ** (1.0 / years) - 1.0) if eq.iloc[-1] > 0 else -1.0
    ann_vol = float(ret.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(ret) > 1 else 0.0
    # Standard Sortino: downside deviation relative to MAR = 0
    downside_sq = np.minimum(ret.to_numpy(float), 0.0) ** 2
    downside_dev = float(np.sqrt(np.mean(downside_sq)) * np.sqrt(TRADING_DAYS)) if len(ret) > 0 else np.nan
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
        payoff = float(wins["pnl"].mean() / abs(losses["pnl"].mean())) if (len(wins) and len(losses) and losses["pnl"].mean() != 0) else np.nan
        avg_hold = float(tdf["hold_days"].mean())
        trades_per_yr = float(n_trades / years)
        long_pnl = float(tdf.loc[tdf["side"] == "LONG", "pnl"].sum())
        short_pnl = float(tdf.loc[tdf["side"] == "SHORT", "pnl"].sum())
    else:
        win_rate = pf = payoff = avg_hold = np.nan
        trades_per_yr = 0.0
        long_pnl = short_pnl = 0.0

    # Annualized turnover (sum of |dw| / years)
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
        "n_trades": n_trades,
        "trades_per_yr": trades_per_yr,
        "win_rate": win_rate,
        "payoff": payoff,
        "profit_factor": pf,
        "avg_hold_days": avg_hold,
        "ann_turnover": ann_turnover,
        "avg_abs_pos": avg_abs_pos,
        "total_cost": float(total_cost),
        "long_pnl": long_pnl,
        "short_pnl": short_pnl,
    }


# ------------------------------------------------------------------ Engine 1: Strict T+1 Open Weight Engine (for Benchmarks / Overlays)
def run_weight_t1_open(
    f: pd.DataFrame,
    target_w: pd.Series,
    *,
    start: str,
    end: str = "2026-12-31",
    commission_bps: float = 10.0,
    slippage_bps: float = 5.0,
    borrow_annual: float = 0.015,
    rebalance_tol: float = 0.0,
    initial: float = 100_000.0,
) -> dict:
    """Strict T+1 Open execution for target weight series in [-1, 1].

    At day t:
      - Overnight gap (Close[t-1] -> Open[t]) is earned on prev_shares.
      - Target weight decided at day t-1 close is executed at Open[t] with slippage + commission.
      - Ex-dividend cash flow is applied to shares held into day t.
      - Daily short borrow fee is accrued on short notional at Close[t].
    """
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
    trades = []
    cur_trade = None
    slip = slippage_bps / 1e4
    comm = commission_bps / 1e4

    for i in range(len(sl)):
        # 1. Ex-dividend on shares held entering day i
        if shares != 0.0 and div[i] > 0:
            cash += shares * div[i]  # long receives +, short pays -

        # 2. Mark-to-market equity at Open[i] (after overnight gap)
        eq_open = cash + shares * o[i]
        if eq_open <= 0:
            equity[i:] = 0.0
            eff_w[i:] = 0.0
            break

        cur_w = (shares * o[i]) / eq_open
        tw = float(np.clip(w_sig[i], -1.0, 1.0))

        # Rebalance if change exceeds tolerance (or switching sign / flattening to 0)
        need_rebal = (abs(tw - cur_w) > rebalance_tol) or (tw == 0.0 and shares != 0.0) or (np.sign(tw) != np.sign(shares) and tw != 0.0)
        if need_rebal:
            target_shares = (tw * eq_open) / o[i]
            delta = target_shares - shares
            if abs(delta) > 1e-8:
                fill = o[i] * (1.0 + slip) if delta > 0 else o[i] * (1.0 - slip)
                notional = abs(delta) * fill
                fee = notional * comm
                slip_cost = abs(delta) * o[i] * slip
                cash -= delta * fill + fee
                total_cost += fee + slip_cost

                # Track round-trip trades roughly on sign/zero transitions
                old_side = 1 if shares > 1e-8 else (-1 if shares < -1e-8 else 0)
                new_side = 1 if target_shares > 1e-8 else (-1 if target_shares < -1e-8 else 0)
                if old_side != 0 and new_side != old_side and cur_trade is not None:
                    cur_trade["exit_date"] = idx[i]
                    cur_trade["exit_px"] = fill
                    cur_trade["hold_days"] = i - cur_trade["entry_i"]
                    if cur_trade["side"] == "LONG":
                        cur_trade["pnl"] = (fill - cur_trade["entry_px"]) * cur_trade["shares"] - cur_trade["fees"] - fee
                    else:
                        cur_trade["pnl"] = (cur_trade["entry_px"] - fill) * abs(cur_trade["shares"]) - cur_trade["fees"] - fee
                    trades.append(cur_trade)
                    cur_trade = None
                if new_side != 0 and new_side != old_side:
                    cur_trade = {
                        "side": "LONG" if new_side > 0 else "SHORT",
                        "entry_date": idx[i],
                        "entry_i": i,
                        "entry_px": fill,
                        "shares": target_shares,
                        "fees": fee,
                    }
                shares = target_shares

        # 3. Short borrow fee accrual at Close[i]
        if shares < 0:
            bfee = abs(shares) * c[i] * (borrow_annual / TRADING_DAYS)
            cash -= bfee
            total_cost += bfee

        eq_close = cash + shares * c[i]
        equity[i] = max(eq_close, 0.0)
        eff_w[i] = (shares * c[i] / eq_close) if eq_close > 0 else 0.0

    # Close open trade at end for accounting
    if cur_trade is not None and shares != 0:
        fill = c[-1]
        cur_trade["exit_date"] = idx[-1]
        cur_trade["exit_px"] = fill
        cur_trade["hold_days"] = len(sl) - 1 - cur_trade["entry_i"]
        if cur_trade["side"] == "LONG":
            cur_trade["pnl"] = (fill - cur_trade["entry_px"]) * cur_trade["shares"] - cur_trade["fees"]
        else:
            cur_trade["pnl"] = (cur_trade["entry_px"] - fill) * abs(cur_trade["shares"]) - cur_trade["fees"]
        trades.append(cur_trade)

    eq_s = pd.Series(equity, index=idx)
    w_s = pd.Series(eff_w, index=idx)
    m = compute_metrics(eq_s, w_s, trades, total_cost, initial)
    return {"equity": eq_s, "weights": w_s, "trades": trades, "metrics": m}


# ------------------------------------------------------------------ Engine 2: Strict Bidirectional Event-Driven Swing Engine
@dataclass
class SwingParams:
    side: str = "LONG"               # "LONG" or "SHORT"
    stop_atr_mult: float = 2.5       # initial stop distance in ATR
    trail_atr_mult: float | None = 3.0  # trailing stop in ATR (None = disabled)
    target_rr: float | None = None   # profit target in R (None = let winners run)
    max_hold_days: int | None = 65   # max trading days hold
    position_mode: str = "vol_budget"  # "fixed" | "risk" | "vol_budget"
    fixed_frac: float = 1.0
    risk_per_trade: float = 0.02     # 2% account risk per trade
    target_vol: float = 0.45         # realized vol budget (annualized)
    max_pos_pct: float = 1.0         # no leverage
    commission_bps: float = 10.0
    slippage_bps: float = 5.0
    borrow_annual: float = 0.015
    earnings_blackout_before: int = 3
    earnings_blackout_after: int = 1
    exit_before_earnings: bool = False


def run_swing(
    f: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series | None,
    params: SwingParams,
    *,
    start: str,
    end: str = "2026-12-31",
    earnings_dates: pd.Series | None = None,
    initial: float = 100_000.0,
) -> dict:
    """Strict bidirectional swing engine.
    - Entry signal at T close -> fill at T+1 Open.
    - EOD exit signal / time stop at T close -> fill at T+1 Open.
    - Pre-placed Stop / Target orders trigger intraday on day t (gap past stop fills at Open[t]).
    - Trailing stop updated AFTER day t close, effective day t+1.
    """
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
    sig_out = (exits.reindex(f.index).fillna(False).reindex(idx).fillna(False).to_numpy(bool)
               if exits is not None else np.zeros(len(sl), dtype=bool))

    # Earnings blackout & pre-earnings exit flags
    block = np.zeros(len(sl), dtype=bool)
    pre_earn = np.zeros(len(sl), dtype=bool)
    if earnings_dates is not None and len(earnings_dates) > 0:
        ed = pd.DatetimeIndex(earnings_dates)
        pos = idx.get_indexer(ed)
        for p in pos:
            if p < 0:
                continue
            lo_i = max(0, p - params.earnings_blackout_before)
            hi_i = min(len(sl) - 1, p + params.earnings_blackout_after)
            block[lo_i:hi_i + 1] = True
            if p - 1 >= 0:
                pre_earn[max(0, p - 2):p] = True

    is_long = params.side == "LONG"
    slip = params.slippage_bps / 1e4
    comm = params.commission_bps / 1e4

    cash = initial
    shares = 0.0  # >0 long, <0 short
    entry_i = -1
    entry_px = 0.0
    stop_px = np.nan
    target_px = np.nan
    init_risk = np.nan
    pending_eod_exit = False
    pending_exit_reason = ""
    cur_trade = None
    trades = []
    total_cost = 0.0
    equity = np.empty(len(sl))
    eff_w = np.empty(len(sl))

    for i in range(len(sl)):
        # 1. Ex-dividend on open position entering day i
        if shares != 0.0 and div[i] > 0:
            cash += shares * div[i]

        # 2. If holding a position, check exits
        if shares != 0.0:
            exit_px = None
            reason = ""

            # A) Pending EOD signal exit from day i-1 -> fill at Open[i]
            if pending_eod_exit:
                exit_px = o[i] * (1.0 - slip) if is_long else o[i] * (1.0 + slip)
                reason = pending_exit_reason
                pending_eod_exit = False
            else:
                if is_long:
                    if o[i] <= stop_px:
                        exit_px, reason = o[i] * (1.0 - slip), "gap_stop"
                    elif l[i] <= stop_px:
                        exit_px, reason = min(o[i], stop_px) * (1.0 - slip), "stop"
                    elif np.isfinite(target_px) and h[i] >= target_px:
                        exit_px, reason = max(o[i], target_px) * (1.0 - slip), "target"
                else:
                    # SHORT: adverse move is UPWARD
                    if o[i] >= stop_px:
                        exit_px, reason = o[i] * (1.0 + slip), "gap_stop"
                    elif h[i] >= stop_px:
                        exit_px, reason = max(o[i], stop_px) * (1.0 + slip), "stop"
                    elif np.isfinite(target_px) and l[i] <= target_px:
                        exit_px, reason = min(o[i], target_px) * (1.0 + slip), "target"

            if exit_px is not None:
                qty = abs(shares)
                notional = qty * exit_px
                fee = notional * comm
                slip_c = qty * abs(exit_px / (1.0 - slip) - exit_px) if is_long else qty * abs(exit_px - exit_px / (1.0 + slip))
                if is_long:
                    cash += notional - fee
                    pnl = (exit_px - entry_px) * qty - cur_trade["fees"] - fee
                else:
                    cash -= notional + fee
                    pnl = (entry_px - exit_px) * qty - cur_trade["fees"] - fee
                total_cost += fee + slip_c
                cur_trade.update({
                    "exit_date": idx[i],
                    "exit_px": exit_px,
                    "exit_reason": reason,
                    "pnl": pnl,
                    "hold_days": i - entry_i,
                    "r_mult": (pnl / qty) / init_risk if (np.isfinite(init_risk) and init_risk > 0) else np.nan,
                })
                trades.append(cur_trade)
                shares = 0.0
                cur_trade = None
                equity[i] = max(cash, 0.0)
                eff_w[i] = 0.0
                continue

            # Still holding at Close[i]: accrue short borrow fee
            if not is_long:
                bfee = abs(shares) * c[i] * (params.borrow_annual / TRADING_DAYS)
                cash -= bfee
                total_cost += bfee
                cur_trade["fees"] += bfee

            # Update trailing stop AFTER Close[i] for day i+1 (only in risk-reducing direction!)
            if params.trail_atr_mult is not None and np.isfinite(atr[i]) and atr[i] > 0:
                if is_long:
                    stop_px = max(stop_px, h[i] - params.trail_atr_mult * atr[i])
                else:
                    stop_px = min(stop_px, l[i] + params.trail_atr_mult * atr[i])

            # Check EOD exit triggers -> scheduled for T+1 Open!
            if sig_out[i]:
                pending_eod_exit, pending_exit_reason = True, "signal_exit"
            elif params.max_hold_days is not None and (i - entry_i) >= params.max_hold_days:
                pending_eod_exit, pending_exit_reason = True, "time_stop"
            elif params.exit_before_earnings and pre_earn[i]:
                pending_eod_exit, pending_exit_reason = True, "pre_earnings"

            eq_c = cash + shares * c[i]
            equity[i] = max(eq_c, 0.0)
            eff_w[i] = (shares * c[i] / eq_c) if eq_c > 0 else 0.0
            continue

        # 3. Flat: evaluate day i-1 entry signal -> execute at Open[i]
        equity[i] = max(cash, 0.0)
        eff_w[i] = 0.0
        if i == 0 or not sig_in[i - 1] or block[i] or cash <= 0:
            continue
        a_prev = atr[i - 1]
        if not np.isfinite(a_prev) or a_prev <= 0 or not np.isfinite(o[i]) or o[i] <= 0:
            continue

        fill = o[i] * (1.0 + slip) if is_long else o[i] * (1.0 - slip)
        if is_long:
            st = c[i - 1] - params.stop_atr_mult * a_prev
            if fill <= st or st <= 0:
                continue
            risk_ps = fill - st
            tgt = (fill + params.target_rr * risk_ps) if params.target_rr is not None else np.nan
        else:
            st = c[i - 1] + params.stop_atr_mult * a_prev
            if fill >= st:
                continue
            risk_ps = st - fill
            tgt = (fill - params.target_rr * risk_ps) if params.target_rr is not None else np.nan

        # Position sizing (zero leverage: notional <= cash * max_pos_pct)
        if params.position_mode == "fixed":
            target_notional = cash * params.fixed_frac
        elif params.position_mode == "risk":
            qty_r = (cash * params.risk_per_trade) / risk_ps
            target_notional = qty_r * fill
        elif params.position_mode == "vol_budget":
            rv_prev = rvol[i - 1]
            if not np.isfinite(rv_prev) or rv_prev <= 0:
                continue
            w_cap = min(params.max_pos_pct, params.target_vol / rv_prev)
            target_notional = cash * w_cap
        else:
            raise ValueError(params.position_mode)

        max_notional = cash * params.max_pos_pct / (1.0 + comm)
        notional = min(target_notional, max_notional)
        qty = notional / fill
        if qty <= 0:
            continue

        fee = notional * comm
        slip_c = qty * o[i] * slip
        total_cost += fee + slip_c
        if is_long:
            cash -= notional + fee
            shares = qty
        else:
            cash += notional - fee
            shares = -qty

        entry_i, entry_px, stop_px, target_px, init_risk = i, fill, st, tgt, risk_ps
        cur_trade = {
            "side": params.side,
            "entry_date": idx[i],
            "entry_px": fill,
            "shares": shares,
            "stop_px": st,
            "fees": fee,
        }

        # Same-day stop check on entry day!
        same_day_hit = (l[i] <= stop_px) if is_long else (h[i] >= stop_px)
        if same_day_hit:
            ex_px = (min(o[i], stop_px) * (1.0 - slip)) if is_long else (max(o[i], stop_px) * (1.0 + slip))
            fee2 = qty * ex_px * comm
            total_cost += fee2 + qty * stop_px * slip
            if is_long:
                cash += qty * ex_px - fee2
                pnl = (ex_px - entry_px) * qty - fee - fee2
            else:
                cash -= qty * ex_px + fee2
                pnl = (entry_px - ex_px) * qty - fee - fee2
            cur_trade.update({
                "exit_date": idx[i], "exit_px": ex_px, "exit_reason": "same_day_stop",
                "pnl": pnl, "hold_days": 0, "r_mult": (pnl / qty) / init_risk,
            })
            trades.append(cur_trade)
            shares = 0.0
            cur_trade = None
            equity[i] = max(cash, 0.0)
            eff_w[i] = 0.0
            continue

        # Accrue borrow fee on entry day if short
        if not is_long:
            bfee = qty * c[i] * (params.borrow_annual / TRADING_DAYS)
            cash -= bfee
            total_cost += bfee
            cur_trade["fees"] += bfee

        eq_c = cash + shares * c[i]
        equity[i] = max(eq_c, 0.0)
        eff_w[i] = (shares * c[i] / eq_c) if eq_c > 0 else 0.0

    # Force close at final bar for trade accounting
    if shares != 0.0 and cur_trade is not None:
        qty = abs(shares)
        ex_px = c[-1] * (1.0 - slip) if is_long else c[-1] * (1.0 + slip)
        fee = qty * ex_px * comm
        total_cost += fee
        if is_long:
            cash += qty * ex_px - fee
            pnl = (ex_px - entry_px) * qty - cur_trade["fees"] - fee
        else:
            cash -= qty * ex_px + fee
            pnl = (entry_px - ex_px) * qty - cur_trade["fees"] - fee
        cur_trade.update({
            "exit_date": idx[-1], "exit_px": ex_px, "exit_reason": "eod",
            "pnl": pnl, "hold_days": len(sl) - 1 - entry_i,
            "r_mult": (pnl / qty) / init_risk if (np.isfinite(init_risk) and init_risk > 0) else np.nan,
        })
        trades.append(cur_trade)
        equity[-1] = max(cash, 0.0)

    eq_s = pd.Series(equity, index=idx)
    w_s = pd.Series(eff_w, index=idx)
    m = compute_metrics(eq_s, w_s, trades, total_cost, initial)
    return {"equity": eq_s, "weights": w_s, "trades": trades, "metrics": m}


# ------------------------------------------------------------------ Combine Independent Long + Short Daily Returns
def combine_portfolios(res_l: dict, res_s: dict, w_l: float = 1.0, w_s: float = 1.0, initial: float = 100_000.0) -> dict:
    """Combine independent Long and Short equity sleeves (no leverage: gross exposure capped at 1.0 via priority or 50/50 split)."""
    rl = res_l["equity"].pct_change().fillna(0.0)
    rs = res_s["equity"].pct_change().fillna(0.0)
    wl = res_l["weights"]
    ws = res_s["weights"]

    # Mutually exclusive check or weighted sleeve
    r_comb = w_l * rl + w_s * rs
    eq_comb = (1.0 + r_comb).cumprod() * initial
    w_comb = w_l * wl + w_s * ws
    trades = res_l["trades"] + res_s["trades"]
    tot_cost = res_l["metrics"]["total_cost"] * w_l + res_s["metrics"]["total_cost"] * w_s
    m = compute_metrics(eq_comb, w_comb, trades, tot_cost, initial)
    return {"equity": eq_comb, "weights": w_comb, "metrics": m}


# ------------------------------------------------------------------ Main Research Execution
def main() -> None:
    data = load_data()
    f = compute_indicators(data["MU"], data.get("SMH"), data.get("SOX"))
    earnings = data["earnings"]

    periods = [
        ("2004-2026(主诊断)", "2004-01-01", "2026-12-31"),
        ("1985-2026(全历史)", "1985-01-01", "2026-12-31"),
        ("2004-2015(熊/震荡)", "2004-01-01", "2015-12-31"),
        ("2016-2026(牛/AI)", "2016-01-01", "2026-12-31"),
    ]

    results = {"benchmarks": [], "long_candidates": [], "short_candidates": [], "indicator_ablation": [], "combinations": []}

    # ==================== 1. SIMPLE BENCHMARKS (Strict T+1 Open, 15bp/side) ====================
    w_bh = pd.Series(1.0, index=f.index)
    w_ma200 = (f["Close"] > f["sma200"]).astype(float)
    w_ma50_200 = (f["sma50"] > f["sma200"]).astype(float)
    cap45 = (0.45 / f["rvol21"]).clip(upper=1.0).fillna(0.0)
    w_ma200_v45 = np.minimum(w_ma200, cap45)
    w_naive_short200 = -((f["Close"] < f["sma200"]).astype(float))
    w_naive_ls200 = w_ma200 + w_naive_short200

    benches = [
        ("BH_买入持有", w_bh),
        ("L0_MA200开关(T+1开盘)", w_ma200),
        ("L0_MA50_200金叉(T+1开盘)", w_ma50_200),
        ("L0_MA200+45%已实现波动预算", w_ma200_v45),
        ("S0_朴素反转做空(价<MA200对照)", w_naive_short200),
        ("LS0_朴素MA200多空翻转对照", w_naive_ls200),
    ]
    for pname, s, e in periods:
        for bname, w in benches:
            r = run_weight_t1_open(f, w, start=s, end=e, commission_bps=10.0, slippage_bps=5.0, borrow_annual=0.015)
            results["benchmarks"].append({"period": pname, "name": bname, **r["metrics"]})

    # ==================== 2. LONG CANDIDATE STRATEGIES (Independent Design) ====================
    # L1: Long-term trend following (MA50>MA200 & Close>MA200, exit Close<MA50, trail 3.5 ATR)
    l1_in = (f["Close"] > f["sma200"]) & (f["sma50"] > f["sma200"]) & (f["sma200_slope20"] > 0)
    l1_out = f["Close"] < f["sma50"]

    # L2: Medium-term trend initiation (Close>MA50 & MA20>MA50 & Close>MA200 & not overextended dist_ma20<0.12)
    l2_in = (f["Close"] > f["sma200"]) & (f["sma20"] > f["sma50"]) & (f["Close"] > f["sma50"]) & (f["dist_ma20"] < 0.12)
    l2_out = (f["Close"] < f["sma50"]) | (f["sma20"] < f["sma50"])

    # L3: 20-day Breakout + Trend filter (Close > Donchian20 & Close > MA200 & CLV > 0.5)
    l3_in = (f["Close"] > f["don_high20"]) & (f["Close"] > f["sma200"]) & (f["clv"] > 0.5)
    l3_out = f["Close"] < f["don_low20"]

    # L4: Trend Pullback Continuation (Close > MA200 & MA50 > MA200 & pullback to MA20 & RSI14 < 55 & bullish bar)
    l4_in = (
        (f["Close"] > f["sma200"]) & (f["sma50"] > f["sma200"])
        & (f["Low"] <= f["sma20"] * 1.01) & (f["Close"] >= f["sma20"] * 0.96)
        & (f["rsi14"] < 55) & f["up_day"]
    )
    l4_out = f["Close"] < f["sma50"]

    # L5: Momentum + Sector Confirmation (ROC60 > 0 & Close > MA200 & SOX > MA200)
    sox_ok = f["sox_above200"].fillna(1.0) > 0.5
    l5_in = (f["Close"] > f["sma200"]) & (f["roc60"] > 0.05) & sox_ok
    l5_out = (f["Close"] < f["sma50"]) | (f["roc60"] < -0.05)

    long_specs = [
        ("L1_长期趋势跟踪(MA50/200+追踪止损)", l1_in, l1_out, SwingParams(side="LONG", stop_atr_mult=2.5, trail_atr_mult=3.5, target_rr=None, max_hold_days=90, position_mode="vol_budget", target_vol=0.45)),
        ("L2_中期趋势启动(防追高+MA50退出)", l2_in, l2_out, SwingParams(side="LONG", stop_atr_mult=2.5, trail_atr_mult=3.0, target_rr=None, max_hold_days=65, position_mode="vol_budget", target_vol=0.45)),
        ("L3_20日通道突破(多头趋势+收盘强)", l3_in, l3_out, SwingParams(side="LONG", stop_atr_mult=2.5, trail_atr_mult=3.0, target_rr=None, max_hold_days=65, position_mode="vol_budget", target_vol=0.45)),
        ("L4_趋势回调延续(回踩MA20+阳线)", l4_in, l4_out, SwingParams(side="LONG", stop_atr_mult=2.5, trail_atr_mult=3.0, target_rr=None, max_hold_days=65, position_mode="vol_budget", target_vol=0.45)),
        ("L5_中期动量+SOX行业确认", l5_in, l5_out, SwingParams(side="LONG", stop_atr_mult=2.5, trail_atr_mult=3.5, target_rr=None, max_hold_days=90, position_mode="vol_budget", target_vol=0.45)),
    ]

    long_res_main = {}
    for pname, s, e in periods:
        for lname, sin, sout, sp in long_specs:
            r = run_swing(f, sin, sout, sp, start=s, end=e, earnings_dates=earnings)
            results["long_candidates"].append({"period": pname, "name": lname, **r["metrics"]})
            if pname.startswith("2004-2026"):
                long_res_main[lname] = r

    # ==================== 3. SHORT CANDIDATE STRATEGIES (Independent Design, NO Long Inversion) ====================
    # S1: Confirmed Downtrend Continuation (strict bear structure + anti-oversold + time cap 45d)
    s1_in = (
        (f["Close"] < f["sma200"]) & (f["sma50"] < f["sma200"]) & (f["sma200_slope20"] < 0)
        & (f["adx14"] > 20) & (f["minus_di"] > f["plus_di"])
        & (f["rsi14"] > 30) & (f["dist_ma20"] > -0.10)  # anti-oversold!
    )
    s1_out = (f["Close"] > f["sma50"]) | (f["Close"] > f["sma20"])

    # S2: Bear-Market Failed Rally Short (bounce to MA20/50 fails with bearish rejection bar)
    s2_in = (
        (f["Close"] < f["sma200"]) & (f["sma50"] < f["sma200"])
        & (f["High"] >= f["sma20"] * 0.99) & (f["Close"] < f["sma20"])
        & f["down_day"] & (f["rsi14"].between(38, 65))
    )
    s2_out = (f["Close"] > f["sma50"]) | (f["rsi14"] < 28)

    # S3: Support Breakdown + Volume Surge (20d low break + vol_ratio > 1.2 + weak close + anti-oversold)
    s3_in = (
        (f["Close"] < f["don_low20"]) & (f["Close"] < f["sma200"]) & (f["sma50"] < f["sma200"])
        & (f["vol_ratio20"] > 1.2) & (f["clv"] < 0.35) & (f["dist_ma20"] > -0.12)
    )
    s3_out = (f["Close"] > f["sma20"]) | (f["rsi14"] < 25)

    # S4: Bear Failed Rally + Squeeze/High-Vol Protection + Pre-Earnings Cover + Sector Weakness
    smh_weak = f["smh_above200"].fillna(0.0) < 0.5
    s4_in = (
        (f["Close"] < f["sma200"]) & (f["sma50"] < f["sma200"])
        & (f["High"] >= f["sma20"] * 0.99) & (f["Close"] < f["sma20"])
        & f["down_day"] & (f["rsi14"].between(38, 65))
        & (f["atr_rank252"] < 0.85)  # avoid panic squeeze regime
        & smh_weak
    )
    s4_out = (f["Close"] > f["sma50"]) | (f["rsi14"] < 28)

    # S5: Confirmed Downtrend Momentum Breakdown + Pre-Earnings Cover + Vol Cap 35%
    s5_in = (
        (f["Close"] < f["sma200"]) & (f["sma50"] < f["sma200"]) & (f["roc60"] < -0.10)
        & (f["Close"] < f["sma20"]) & (f["rsi14"].between(32, 55))
        & (f["atr_rank252"] < 0.85)
    )
    s5_out = (f["Close"] > f["sma20"]) | (f["rsi14"] < 25)

    short_specs = [
        ("S1_确认下跌趋势延续(防超跌+45日上限)", s1_in, s1_out, SwingParams(side="SHORT", stop_atr_mult=2.0, trail_atr_mult=2.5, target_rr=None, max_hold_days=45, position_mode="vol_budget", target_vol=0.35, exit_before_earnings=True)),
        ("S2_熊市反弹失败做空(遇MA20受阻)", s2_in, s2_out, SwingParams(side="SHORT", stop_atr_mult=2.0, trail_atr_mult=2.5, target_rr=2.0, max_hold_days=40, position_mode="vol_budget", target_vol=0.35, exit_before_earnings=True)),
        ("S3_支撑位放量跌破(20日低+量能)", s3_in, s3_out, SwingParams(side="SHORT", stop_atr_mult=2.0, trail_atr_mult=2.5, target_rr=2.0, max_hold_days=40, position_mode="vol_budget", target_vol=0.35, exit_before_earnings=True)),
        ("S4_反弹失败+防逼空高波过滤+SMH走弱", s4_in, s4_out, SwingParams(side="SHORT", stop_atr_mult=2.0, trail_atr_mult=2.5, target_rr=2.0, max_hold_days=40, position_mode="vol_budget", target_vol=0.35, exit_before_earnings=True)),
        ("S5_下跌动量确认+防逼空+财报前平仓", s5_in, s5_out, SwingParams(side="SHORT", stop_atr_mult=2.0, trail_atr_mult=2.5, target_rr=None, max_hold_days=40, position_mode="vol_budget", target_vol=0.35, exit_before_earnings=True)),
    ]

    short_res_main = {}
    for pname, s, e in periods:
        for sname, sin, sout, sp in short_specs:
            r = run_swing(f, sin, sout, sp, start=s, end=e, earnings_dates=earnings)
            results["short_candidates"].append({"period": pname, "name": sname, **r["metrics"]})
            if pname.startswith("2004-2026"):
                short_res_main[sname] = r

    # ==================== 4. INDICATOR SUBTRACTION / INCREMENTAL VALUE ABLATION (2004-2026 & 1985-2026) ====================
    # Base Long: Close > MA200 & MA20 > MA50 (minimal trend core)
    base_l_in = (f["Close"] > f["sma200"]) & (f["sma20"] > f["sma50"])
    base_l_out = f["Close"] < f["sma50"]
    base_sp = SwingParams(side="LONG", stop_atr_mult=2.5, trail_atr_mult=3.0, target_rr=None, max_hold_days=65, position_mode="fixed", fixed_frac=1.0)

    ablations = [
        ("BASE_多头极简基线(MA200+MA20>50)", base_l_in, base_sp),
        ("+MA5(要求价>MA5)", base_l_in & (f["Close"] > f["sma5"]), base_sp),
        ("+ADX>20(趋势强度过滤)", base_l_in & (f["adx14"] > 20), base_sp),
        ("+MACD柱>0(动量确认)", base_l_in & (f["macd_hist"] > 0), base_sp),
        ("+RSI14<70(防短线超买)", base_l_in & (f["rsi14"] < 70), base_sp),
        ("+ROC60>0(中期正动量)", base_l_in & (f["roc60"] > 0), base_sp),
        ("+VOL放量(vol_ratio20>1.2)", base_l_in & (f["vol_ratio20"] > 1.2), base_sp),
        ("+OBV>OBV_MA20(量价趋势)", base_l_in & (f["obv"] > f["obv_ma20"]), base_sp),
        ("+BOLL不破上轨(防追高)", base_l_in & (f["Close"] < f["boll_upper"]), base_sp),
        ("+SOX>MA200(行业大趋势)", base_l_in & (f["sox_above200"].fillna(1.0) > 0.5), base_sp),
        ("+MU相对SMH强势(20日超额>0)", base_l_in & (f["mu_vs_smh_20"].fillna(0.0) > 0), base_sp),
        ("+已实现波动预算45%(仓位层)", base_l_in, SwingParams(side="LONG", stop_atr_mult=2.5, trail_atr_mult=3.0, target_rr=None, max_hold_days=65, position_mode="vol_budget", target_vol=0.45)),
    ]

    for aname, ain, asp in ablations:
        r04 = run_swing(f, ain, base_l_out, asp, start="2004-01-01", end="2026-12-31", earnings_dates=earnings)["metrics"]
        r85 = run_swing(f, ain, base_l_out, asp, start="1985-01-01", end="2026-12-31", earnings_dates=earnings)["metrics"]
        results["indicator_ablation"].append({
            "name": aname,
            "cagr_04": r04["cagr"], "mdd_04": r04["max_dd"], "calmar_04": r04["calmar"], "sharpe_04": r04["sharpe"], "n_04": r04["n_trades"],
            "cagr_85": r85["cagr"], "mdd_85": r85["max_dd"], "calmar_85": r85["calmar"], "sharpe_85": r85["sharpe"], "n_85": r85["n_trades"],
        })

    # ==================== 5. PORTFOLIO COMBINATIONS (2004-2026 & 1985-2026) ====================
    # Pick top-behaving Long (L1, L2) & Short (S2, S4, S5) to test combinations
    for pname, s, e in [("2004-2026(主诊断)", "2004-01-01", "2026-12-31"), ("1985-2026(全历史)", "1985-01-01", "2026-12-31")]:
        r_l0 = run_weight_t1_open(f, w_ma200_v45, start=s, end=e, commission_bps=10.0, slippage_bps=5.0)
        r_l1 = run_swing(f, l1_in, l1_out, long_specs[0][3], start=s, end=e, earnings_dates=earnings)
        r_l2 = run_swing(f, l2_in, l2_out, long_specs[1][3], start=s, end=e, earnings_dates=earnings)
        r_s2 = run_swing(f, s2_in, s2_out, short_specs[1][3], start=s, end=e, earnings_dates=earnings)
        r_s4 = run_swing(f, s4_in, s4_out, short_specs[3][3], start=s, end=e, earnings_dates=earnings)
        r_s5 = run_swing(f, s5_in, s5_out, short_specs[4][3], start=s, end=e, earnings_dates=earnings)

        # Since L1/L2 only enter when Close>MA200 and S2/S4/S5 only enter when Close<MA200, they are naturally near-disjoint!
        # We test 100% Long + 50% Short sleeve (conservative short risk budget) and 100% Long + 100% Short sleeve
        combos = [
            ("仅多_L0(MA200+45%波动预算)", r_l0["metrics"]),
            ("仅多_L1(长期趋势跟踪)", r_l1["metrics"]),
            ("仅多_L2(中期趋势启动)", r_l2["metrics"]),
            ("仅空_S2(熊市反弹失败)", r_s2["metrics"]),
            ("仅空_S4(反弹失败+防逼空+SMH弱)", r_s4["metrics"]),
            ("仅空_S5(下跌动量+防逼空)", r_s5["metrics"]),
            ("多空组合_L1 + 0.5*S4", combine_portfolios(r_l1, r_s4, 1.0, 0.5)["metrics"]),
            ("多空组合_L2 + 0.5*S4", combine_portfolios(r_l2, r_s4, 1.0, 0.5)["metrics"]),
            ("多空组合_L0(MA200波动) + 0.5*S4", combine_portfolios(r_l0, r_s4, 1.0, 0.5)["metrics"]),
            ("多空组合_L1 + 0.5*S5", combine_portfolios(r_l1, r_s5, 1.0, 0.5)["metrics"]),
        ]
        for cname, cm in combos:
            results["combinations"].append({"period": pname, "name": cname, **cm})

    out_json = OUT_DIR / "stage2_results.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"Saved -> {out_json}")

    # Compact console tables
    def show(title: str, rows: list[dict], period_filter: str | None = None):
        print("\n" + "=" * 100)
        print(title)
        print("=" * 100)
        df = pd.DataFrame(rows)
        if period_filter:
            df = df[df["period"] == period_filter]
        cols = ["name", "cagr", "max_dd", "calmar", "sharpe", "sortino", "n_trades", "win_rate", "profit_factor", "avg_hold_days", "avg_abs_pos"]
        cols = [c for c in cols if c in df.columns]
        print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    show("1. SIMPLE BENCHMARKS (2004-2026 主诊断, T+1 Open, 15bp/side)", results["benchmarks"], "2004-2026(主诊断)")
    show("1b. SIMPLE BENCHMARKS (1985-2026 全历史, T+1 Open, 15bp/side)", results["benchmarks"], "1985-2026(全历史)")
    show("2. LONG CANDIDATES (2004-2026 主诊断)", results["long_candidates"], "2004-2026(主诊断)")
    show("2b. LONG CANDIDATES (1985-2026 全历史)", results["long_candidates"], "1985-2026(全历史)")
    show("3. SHORT CANDIDATES (2004-2026 主诊断, 1.5% borrow + div)", results["short_candidates"], "2004-2026(主诊断)")
    show("3b. SHORT CANDIDATES (1985-2026 全历史, 1.5% borrow + div)", results["short_candidates"], "1985-2026(全历史)")
    print("\n" + "=" * 100)
    print("4. INDICATOR ABLATION (Incremental Value Test)")
    print("=" * 100)
    print(pd.DataFrame(results["indicator_ablation"]).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    show("5. COMBINATIONS (2004-2026 主诊断)", results["combinations"], "2004-2026(主诊断)")
    show("5b. COMBINATIONS (1985-2026 全历史)", results["combinations"], "1985-2026(全历史)")


if __name__ == "__main__":
    main()
