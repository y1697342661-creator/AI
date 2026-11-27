"""5-State Crypto Market Regime Classifier for BTC-USDT-SWAP.

Uses strictly completed 1D trend structure + 4H volatility/funding state:
- STRONG_BULL: d1_close > ema_50d > sma_200d (or d1_close > sma_200d & ema_20d > ema_50d) & normal vol
- BULL_PULLBACK: d1_close > sma_200d but below ema_50d (uptrend digestion / pullback)
- BEAR_TREND: d1_close < ema_50d and d1_close < sma_200d (macro downtrend)
- VOL_CRISIS: annualized 14d realized vol > 105% (liquidation cascade / extreme panic)
- SIDEWAYS_CHOP: transitional / low-ADX range where breakout whipsaws dominate
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def classify_regime_series(df: pd.DataFrame) -> pd.Series:
    rvol = df.get("rvol_14d", pd.Series(0.5, index=df.index)).fillna(0.5)
    d1_close = df.get("d1_close", df["close"]).fillna(df["close"])
    ema50 = df.get("ema_50d", df["ema_120"]).fillna(df["ema_120"])
    sma200 = df.get("sma_200d", df["ema_200"]).fillna(df["ema_200"])
    adx_d = df.get("adx_14d", df["adx_14"]).fillna(20.0)

    regime = pd.Series("SIDEWAYS_CHOP", index=df.index, dtype="object")

    bull_macro = (d1_close > sma200) & (d1_close > ema50) & (ema50 >= sma200 * 0.98)
    bull_pullback = (d1_close > sma200) & (d1_close <= ema50)
    bear_macro = (d1_close < sma200) & (d1_close < ema50)
    chop = adx_d < 16.0

    regime = np.where(bull_macro, "STRONG_BULL", regime)
    regime = np.where(bull_pullback, "BULL_PULLBACK", regime)
    regime = np.where(bear_macro, "BEAR_TREND", regime)
    regime = np.where(chop & (~bear_macro) & (~bull_macro), "SIDEWAYS_CHOP", regime)
    regime = np.where(rvol > 1.05, "VOL_CRISIS", regime)

    return pd.Series(regime, index=df.index, name="regime")


def latest_regime_summary(df: pd.DataFrame) -> dict[str, Any]:
    reg = classify_regime_series(df)
    last = df.iloc[-1]
    cur = str(reg.iloc[-1])
    labels = {
        "STRONG_BULL": "强势多头趋势（顺势做多优先）",
        "BULL_PULLBACK": "牛市回踩整固（严控多头仓位）",
        "SIDEWAYS_CHOP": "震荡洗盘区间（降低杠杆/过滤假突破）",
        "BEAR_TREND": "空头下跌趋势（允许顺势做空/禁逆势抄底）",
        "VOL_CRISIS": "极端波动危机（强制降杠杆防守）",
    }
    return {
        "regime": cur,
        "regime_zh": labels.get(cur, cur),
        "close": float(last["close"]),
        "d1_ema_50": float(last["ema_50d"]) if pd.notna(last.get("ema_50d")) else None,
        "d1_sma_200": float(last["sma_200d"]) if pd.notna(last.get("sma_200d")) else None,
        "rvol_14d_pct": round(float(last["rvol_14d"]) * 100, 2) if pd.notna(last.get("rvol_14d")) else None,
        "adx_14d": round(float(last["adx_14d"]), 2) if pd.notna(last.get("adx_14d")) else None,
        "funding_7d_ann_pct": round(float(last["funding_7d_ann"]) * 100, 2) if pd.notna(last.get("funding_7d_ann")) else 0.0,
        "basis_pct": round(float(last["basis_pct"]) * 100, 3) if pd.notna(last.get("basis_pct")) else 0.0,
    }
