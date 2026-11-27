"""Strictly causal technical features (subtraction-screened in Stage 2).

Every column depends ONLY on bar t and earlier bars.
Donchian channels are explicitly shifted by 1 bar to prevent breakout lookahead.
ATR (price range) and Realized/Downside Volatility (return std) are strictly separated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mu_quant.config import TRADING_DAYS


def build_features(
    mu: pd.DataFrame,
    smh: pd.DataFrame | None = None,
    sox: pd.DataFrame | None = None,
) -> pd.DataFrame:
    f = pd.DataFrame(index=mu.index)
    adj_factor = mu["Adj Close"] / mu["Close"]
    o = mu["Open"] * adj_factor
    h = mu["High"] * adj_factor
    l = mu["Low"] * adj_factor
    c = mu["Adj Close"]
    v = mu["Volume"].astype(float)

    f["Open"], f["High"], f["Low"], f["Close"], f["Volume"] = o, h, l, c, v
    f["raw_open"], f["raw_high"], f["raw_low"], f["raw_close"] = mu["Open"], mu["High"], mu["Low"], mu["Close"]
    f["div"] = mu.get("Dividends", pd.Series(0.0, index=mu.index)).fillna(0.0)
    f["ret"] = c.pct_change()

    # Moving averages (MA5 retained for dashboard visibility; MA20/50/200 are core trend skeleton)
    for n in (5, 20, 50, 200):
        f[f"sma{n}"] = c.rolling(n, min_periods=n).mean()
    f["sma200_slope20"] = f["sma200"] / f["sma200"].shift(20) - 1.0

    # Wilder ATR(14) — used ONLY for stop-loss distance & regime range monitoring
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    f["atr14"] = tr.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
    f["atr_pct"] = f["atr14"] / c
    f["atr_rank252"] = f["atr_pct"].rolling(252, min_periods=120).rank(pct=True)

    prev_rc = mu["Close"].shift(1)
    tr_raw = pd.concat([
        (mu["High"] - mu["Low"]),
        (mu["High"] - prev_rc).abs(),
        (mu["Low"] - prev_rc).abs(),
    ], axis=1).max(axis=1)
    f["raw_atr14"] = tr_raw.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()

    # Realized Volatility & Downside-Only Volatility — used ONLY for volatility position budgeting
    f["rvol21"] = f["ret"].rolling(21, min_periods=21).std() * np.sqrt(TRADING_DAYS)
    neg_ret = f["ret"].clip(upper=0.0)
    f["dvol21"] = np.sqrt((neg_ret ** 2).rolling(21, min_periods=21).mean() * 2.0 * TRADING_DAYS)

    # Wilder ADX(14) — used ONLY for market regime monitoring (removed from signal rules)
    up = h.diff()
    dn = -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=mu.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=mu.index)
    plus_di = 100.0 * plus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / f["atr14"]
    minus_di = 100.0 * minus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / f["atr14"]
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    f["adx14"] = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    # RSI(14) — anti-extreme guard
    delta = c.diff()
    ag = delta.clip(lower=0.0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    al = (-delta).clip(lower=0.0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = ag / al.replace(0.0, np.nan)
    f["rsi14"] = (100.0 - 100.0 / (1.0 + rs)).where(al != 0, 100.0).where(ag.notna())

    # MACD(12,26,9) — retained for dashboard display only (removed from signal rules in Stage 2)
    ema12 = c.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = c.ewm(span=26, adjust=False, min_periods=26).mean()
    f["macd"] = ema12 - ema26
    f["macd_sig"] = f["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    f["macd_hist"] = f["macd"] - f["macd_sig"]

    # Donchian(20) — strictly shifted by 1 day (no breakout lookahead)
    f["don_high20"] = h.rolling(20, min_periods=20).max().shift(1)
    f["don_low20"] = l.rolling(20, min_periods=20).min().shift(1)

    # Volume
    f["vol_ma20"] = v.rolling(20, min_periods=20).mean()
    f["vol_ratio20"] = v / f["vol_ma20"]

    # Price structure
    f["dist_ma20"] = c / f["sma20"] - 1.0
    f["dist_ma200"] = c / f["sma200"] - 1.0
    f["clv"] = (c - l) / (h - l).replace(0.0, np.nan)

    # Sector benchmarks (causal reindex + ffill)
    if smh is not None and not smh.empty:
        smh_c = smh["Adj Close"].reindex(mu.index).ffill()
        f["smh_above200"] = (smh_c > smh_c.rolling(200, min_periods=200).mean()).astype(float)
    else:
        f["smh_above200"] = np.nan

    if sox is not None and not sox.empty:
        sox_c = sox["Adj Close"].reindex(mu.index).ffill()
        f["sox_above200"] = (sox_c > sox_c.rolling(200, min_periods=200).mean()).astype(float)
    else:
        f["sox_above200"] = np.nan

    return f
