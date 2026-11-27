"""Production BTC-USDT-SWAP Perpetual Futures Strategies.

All strategies output `(target_leverage, stop_dist_pct, trail_dist_pct)` strictly
from information available at bar T close (zero look-ahead). The backtest engine
and execution engine apply bar T close signals at bar T+1 open.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from btc_quant.market_regime.regime import classify_regime_series


def _donchian_state(df: pd.DataFrame, entry_w: int = 42, exit_w: int = 20) -> np.ndarray:
    hi_in = df[f"dc_hi_{entry_w}"].to_numpy()
    lo_in = df[f"dc_lo_{entry_w}"].to_numpy()
    hi_out = df[f"dc_hi_{exit_w}"].to_numpy()
    lo_out = df[f"dc_lo_{exit_w}"].to_numpy()
    close = df["close"].to_numpy()
    d1_cl = df["d1_close"].fillna(df["close"]).to_numpy()
    ema50d = df["ema_50d"].fillna(df["ema_200"]).to_numpy()
    sma200d = df["sma_200d"].fillna(df["ema_200"]).to_numpy()
    fr_ann = df["funding_7d_ann"].fillna(0.0).to_numpy()
    adx_4h = df["adx_14"].fillna(20.0).to_numpy()

    n = len(df)
    state = np.zeros(n)
    cur = 0
    for i in range(n):
        if not np.isfinite(hi_in[i]) or not np.isfinite(lo_in[i]):
            state[i] = 0
            continue
        bull_ok = (d1_cl[i] > ema50d[i]) or (d1_cl[i] > sma200d[i])
        bear_ok = (d1_cl[i] < ema50d[i]) and (d1_cl[i] < sma200d[i] * 1.03)
        if cur == 0:
            if close[i] > hi_in[i] and bull_ok and adx_4h[i] >= 16.0:
                cur = 1
            elif close[i] < lo_in[i] and bear_ok and adx_4h[i] >= 18.0 and fr_ann[i] > -0.35:
                cur = -1
        elif cur == 1:
            if close[i] < lo_out[i]:
                cur = 0
        elif cur == -1:
            if close[i] > hi_out[i]:
                cur = 0
        state[i] = cur
    return state


def generate_strategy_signals(
    df: pd.DataFrame,
    mode: str = "dual_horizon_ensemble",
    params: dict[str, Any] | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Dispatch strategy signal generation by mode."""
    p = params or {}
    vol_target = float(p.get("vol_target", 0.60))
    max_long = float(p.get("max_long_leverage", 2.0))
    max_short = float(p.get("max_short_leverage", 1.2))
    entry_w = int(p.get("donchian_entry_window", 42))
    exit_w = int(p.get("donchian_exit_window", 20))
    stop_atr = float(p.get("stop_atr_mult", 3.5))
    trail_atr = float(p.get("trail_atr_mult", 4.5))
    fr_long_cap = float(p.get("funding_crowding_long_ann", 0.40))
    fr_short_cap = float(p.get("funding_crowding_short_ann", -0.25))

    if mode == "buy_and_hold":
        nan_s = pd.Series(np.nan, index=df.index)
        return pd.Series(1.0, index=df.index), nan_s, nan_s

    if mode == "cta_trend_ensemble":
        t1, s1, tr1 = generate_strategy_signals(df, mode="donchian_cta", params=params)
        t2, s2, tr2 = generate_strategy_signals(df, mode="dual_horizon_ensemble", params=params)
        tgt_comb = 0.50 * t1 + 0.50 * t2
        return tgt_comb.rename("target_leverage"), 0.5 * (s1 + s2), 0.5 * (tr1 + tr2)

    rvol = df["rvol_14d"].clip(lower=0.25).fillna(0.60).to_numpy()
    dvol = df["dvol_14d"].clip(lower=0.22).fillna(0.50).to_numpy()
    fr_ann = df["funding_7d_ann"].fillna(0.0).to_numpy()

    if mode == "donchian_cta":
        state = _donchian_state(df, entry_w=entry_w, exit_w=exit_w)
        size_long = np.minimum(vol_target / rvol, max_long) * np.where(fr_ann > fr_long_cap, 0.65, 1.0)
        size_short = np.minimum((vol_target * 0.80) / rvol, max_short) * np.where(fr_ann < fr_short_cap, 0.60, 1.0)
        tgt = np.where(state > 0, size_long, np.where(state < 0, -size_short, 0.0))
        tgt = np.where(rvol > 1.05, np.clip(tgt, -0.5, 0.5), tgt)
    elif mode == "mtf_regime_trend":
        reg = classify_regime_series(df)
        base_size = np.minimum(vol_target / rvol, max_long)
        long_cond = (df["ema_21"] > df["ema_55"]) & (df["close"] > df["ema_55"]) & reg.isin(["STRONG_BULL", "BULL_PULLBACK"])
        short_cond = (df["ema_21"] < df["ema_55"]) & (df["close"] < df["ema_55"]) & (reg == "BEAR_TREND")
        long_mult = np.where(fr_ann > fr_long_cap, 0.65, 1.0)
        short_mult = np.where(fr_ann < fr_short_cap, 0.50, 1.0)
        tgt = np.where(long_cond, base_size * long_mult, np.where(short_cond, -np.minimum(base_size * 0.75, max_short) * short_mult, 0.0))
    else:
        # Flagship: dual_horizon_ensemble (or long_only_control when max_short == 0)
        d1_cl = df["d1_close"].fillna(df["close"])
        ema50d = df["ema_50d"].fillna(df["ema_120"])
        sma200d = df["sma_200d"].fillna(df["ema_200"])

        macro_long = ((d1_cl > ema50d).astype(float) * 0.5 + (d1_cl > sma200d).astype(float) * 0.5).to_numpy()
        macro_bear = ((d1_cl < ema50d) & (d1_cl < sma200d)).astype(float).to_numpy()

        dc_state = _donchian_state(df, entry_w=entry_w, exit_w=exit_w)
        ema_dir = np.where(
            (df["ema_21"] > df["ema_55"]) & (df["close"] > df["ema_55"]),
            1.0,
            np.where((df["ema_21"] < df["ema_55"]) & (df["close"] < df["ema_55"]), -1.0, 0.0),
        )

        long_score = np.where(
            macro_long >= 0.5,
            0.45 * macro_long + 0.30 * np.maximum(ema_dir, 0.0) + 0.25 * np.maximum(dc_state, 0.0),
            0.0,
        )
        long_score = np.where(long_score >= 0.45, long_score, 0.0)

        short_score = np.where(
            macro_bear > 0.5,
            0.50 * np.maximum(-ema_dir, 0.0) + 0.50 * np.maximum(-dc_state, 0.0),
            0.0,
        )
        short_score = np.where(short_score >= 0.50, short_score, 0.0)

        long_score = long_score * np.where(fr_ann > fr_long_cap, 0.65, np.where(fr_ann < -0.05, 1.10, 1.0))
        short_score = short_score * np.where(fr_ann < fr_short_cap, 0.50, np.where(fr_ann > 0.05, 1.10, 1.0))

        eff_vol = 0.6 * dvol + 0.4 * rvol
        raw_lev = vol_target / eff_vol

        if mode == "long_only_control":
            max_short_eff = 0.0
        else:
            max_short_eff = max_short

        tgt = np.where(
            long_score > 0,
            np.minimum(long_score * raw_lev, max_long),
            np.where((short_score > 0) & (max_short_eff > 0), -np.minimum(short_score * raw_lev * 0.75, max_short_eff), 0.0),
        )
        tgt = np.where(rvol > 1.05, np.clip(tgt, -0.5, 0.5), tgt)

    stop_dist = (df["atr_pct"] * stop_atr).clip(0.035, 0.14)
    trail_dist = (df["atr_pct"] * trail_atr).clip(0.045, 0.18)
    return pd.Series(tgt, index=df.index, name="target_leverage"), stop_dist, trail_dist


def latest_signal_snapshot(
    df: pd.DataFrame,
    mode: str = "dual_horizon_ensemble",
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute the latest actionable signal at the most recent closed 4H bar."""
    tgt_s, stop_s, trail_s = generate_strategy_signals(df, mode=mode, params=params)
    last = df.iloc[-1]
    tgt_lev = float(tgt_s.iloc[-1])
    close_px = float(last["close"])
    stop_d = float(stop_s.iloc[-1]) if pd.notna(stop_s.iloc[-1]) else 0.06
    trail_d = float(trail_s.iloc[-1]) if pd.notna(trail_s.iloc[-1]) else 0.08

    if tgt_lev > 0.05:
        direction = "LONG"
        stop_px = round(close_px * (1.0 - stop_d), 2)
        trail_px = round(close_px * (1.0 - trail_d), 2)
    elif tgt_lev < -0.05:
        direction = "SHORT"
        stop_px = round(close_px * (1.0 + stop_d), 2)
        trail_px = round(close_px * (1.0 + trail_d), 2)
    else:
        direction = "FLAT"
        stop_px = None
        trail_px = None

    return {
        "bar_time_utc": str(df.index[-1]),
        "mode": mode,
        "direction": direction,
        "target_leverage": round(tgt_lev, 3),
        "close_px": round(close_px, 2),
        "atr_14": round(float(last["atr_14"]), 2) if pd.notna(last.get("atr_14")) else None,
        "atr_pct": round(float(last["atr_pct"]) * 100, 2) if pd.notna(last.get("atr_pct")) else None,
        "stop_distance_pct": round(stop_d * 100, 2),
        "trail_distance_pct": round(trail_d * 100, 2),
        "suggested_stop_px": stop_px,
        "suggested_trail_px": trail_px,
        "ema_21": round(float(last["ema_21"]), 2) if pd.notna(last.get("ema_21")) else None,
        "ema_55": round(float(last["ema_55"]), 2) if pd.notna(last.get("ema_55")) else None,
        "rsi_14": round(float(last["rsi_14"]), 2) if pd.notna(last.get("rsi_14")) else None,
        "adx_14": round(float(last["adx_14"]), 2) if pd.notna(last.get("adx_14")) else None,
        "funding_7d_ann_pct": round(float(last["funding_7d_ann"]) * 100, 2) if pd.notna(last.get("funding_7d_ann")) else 0.0,
    }
