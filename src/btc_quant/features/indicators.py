"""Feature engineering for BTC-USDT-SWAP (4H execution bars + strictly lagged 1D context).

ZERO LOOK-AHEAD GUARANTEE:
- Every 1D indicator is indexed at its *bar close* timestamp (day_open + 24h) before
  backward merge_asof onto 4H bars, so a 4H bar ONLY ever sees completed prior 1D bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BARS_PER_YEAR_4H = 365 * 6  # 2190 4H bars per year (24/7 crypto market)
BARS_PER_YEAR_1D = 365


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = _atr(df, window).replace(0.0, np.nan)
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / window, adjust=False).mean() / atr
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / window, adjust=False).mean() / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def build_1d_features(df_1d: pd.DataFrame) -> pd.DataFrame:
    d = df_1d.copy()
    d["ret_1d"] = d["close"].pct_change()
    d["ema_20d"] = _ema(d["close"], 20)
    d["ema_50d"] = _ema(d["close"], 50)
    d["sma_50d"] = d["close"].rolling(50).mean()
    d["sma_200d"] = d["close"].rolling(200).mean()
    d["atr_14d"] = _atr(d, 14)
    d["atr_pct_14d"] = d["atr_14d"] / d["close"]
    d["adx_14d"] = _adx(d, 14)
    d["rsi_14d"] = _rsi(d["close"], 14)
    d["rvol_20d"] = d["ret_1d"].rolling(20).std() * np.sqrt(BARS_PER_YEAR_1D)
    neg_ret = d["ret_1d"].clip(upper=0.0)
    d["dvol_20d"] = np.sqrt((neg_ret ** 2).rolling(20).mean()) * np.sqrt(BARS_PER_YEAR_1D)
    d["donchian_hi_20d"] = d["high"].rolling(20).max()
    d["donchian_lo_20d"] = d["low"].rolling(20).min()
    d["donchian_hi_55d"] = d["high"].rolling(55).max()
    d["donchian_lo_55d"] = d["low"].rolling(55).min()
    if "funding_rate_daily" not in d.columns:
        d["funding_rate_daily"] = 0.0
    d["funding_7d_ann"] = d["funding_rate_daily"].rolling(7, min_periods=1).mean() * 365.0
    d["funding_30d_ann"] = d["funding_rate_daily"].rolling(30, min_periods=1).mean() * 365.0
    if "basis_pct" not in d.columns:
        d["basis_pct"] = 0.0
    d["basis_7d_avg"] = d["basis_pct"].rolling(7, min_periods=1).mean()
    return d


def build_4h_features(df_4h: pd.DataFrame, df_1d_feat: pd.DataFrame | None = None) -> pd.DataFrame:
    f = df_4h.copy()
    f["ret_4h"] = f["close"].pct_change()
    f["ema_21"] = _ema(f["close"], 21)
    f["ema_55"] = _ema(f["close"], 55)
    f["ema_120"] = _ema(f["close"], 120)
    f["ema_200"] = _ema(f["close"], 200)
    f["atr_14"] = _atr(f, 14)
    f["atr_pct"] = f["atr_14"] / f["close"]
    f["adx_14"] = _adx(f, 14)
    f["rsi_14"] = _rsi(f["close"], 14)

    # Realized & downside annualized volatility (rolling 7 days = 42 bars, 14 days = 84 bars)
    f["rvol_7d"] = f["ret_4h"].rolling(42).std() * np.sqrt(BARS_PER_YEAR_4H)
    f["rvol_14d"] = f["ret_4h"].rolling(84).std() * np.sqrt(BARS_PER_YEAR_4H)
    neg_ret = f["ret_4h"].clip(upper=0.0)
    f["dvol_14d"] = np.sqrt((neg_ret ** 2).rolling(84).mean()) * np.sqrt(BARS_PER_YEAR_4H)

    # Donchian breakout channels (strictly prior bars: shift(1) so current close breaks PRIOR N-bar high/low)
    for w in (20, 30, 42, 60, 84):
        f[f"dc_hi_{w}"] = f["high"].rolling(w).max().shift(1)
        f[f"dc_lo_{w}"] = f["low"].rolling(w).min().shift(1)

    # Volume surge ratio
    f["vol_ma_42"] = f["vol_quote"].rolling(42).mean()
    f["vol_ratio"] = f["vol_quote"] / f["vol_ma_42"].replace(0.0, np.nan)

    # Funding rate rolling features (3 settlements per day -> 21 settlements = 7 days)
    if "funding_rate" not in f.columns:
        f["funding_rate"] = 0.0
    # Rolling 7-day (42 4H bars) sum of funding rates, annualized
    f["funding_7d_sum"] = f["funding_rate"].rolling(42, min_periods=1).sum()
    f["funding_7d_ann"] = f["funding_7d_sum"] * (365.0 / 7.0)
    f["funding_30d_ann"] = f["funding_rate"].rolling(180, min_periods=1).sum() * (365.0 / 30.0)

    # Merge strictly COMPLETED 1D features onto 4H bars (zero look-ahead!)
    if df_1d_feat is not None and not df_1d_feat.empty:
        d_cols = [
            "close",
            "ema_20d",
            "ema_50d",
            "sma_50d",
            "sma_200d",
            "adx_14d",
            "rsi_14d",
            "rvol_20d",
            "dvol_20d",
            "basis_pct",
            "basis_7d_avg",
        ]
        d_sub = df_1d_feat[[c for c in d_cols if c in df_1d_feat.columns]].copy()
        d_sub = d_sub.rename(columns={"close": "d1_close"})
        # A 1D bar opened at T_open is only COMPLETED at T_open + 1 day!
        d_sub["available_at"] = d_sub.index + pd.Timedelta(days=1)
        d_sub = d_sub.reset_index(drop=True).sort_values("available_at")

        f_reset = f.reset_index().rename(columns={f.index.name or "index": "dt"})
        # A 4H bar closes at dt + 4h, so at decision time (4H bar close), any 1D bar with available_at <= dt + 4h is completed
        f_reset["bar_close_dt"] = f_reset["dt"] + pd.Timedelta(hours=4)
        merged = pd.merge_asof(
            f_reset.sort_values("bar_close_dt"),
            d_sub,
            left_on="bar_close_dt",
            right_on="available_at",
            direction="backward",
        )
        merged.drop(columns=["bar_close_dt", "available_at"], inplace=True)
        merged.set_index("dt", inplace=True)
        f = merged

    return f
