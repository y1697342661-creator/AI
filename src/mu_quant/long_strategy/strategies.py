"""Independent Long Strategy Module.

Supported modes (all strictly causal, zero leverage):
1. dual_confirm_dvol55 (Default v2.0.0):
   - 50% weight if Close > MA200 + 50% weight if MA50 > MA200 (100% / 50% / 0%)
   - Capped by 55% Downside-Only Realized Volatility budget (does NOT punish upside surges!)
2. ma200_rvol45 (Defensive):
   - 100% if Close > MA200 else 0%, capped by 45% Total Realized Volatility budget (BH-FDR significant MDD reduction)
3. ma50_200 (Simple Dual-MA Reference):
   - 100% if MA50 > MA200 else 0%
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_long_weights(f: pd.DataFrame, cfg: dict[str, Any], mode: str | None = None) -> pd.Series:
    lc = cfg.get("long_strategy", {})
    m = mode or lc.get("default_mode", "dual_confirm_dvol55")
    ma_fast = int(lc.get("ma_fast", 50))
    ma_slow = int(lc.get("ma_slow", 200))

    c = f["Close"]
    sma_f = f[f"sma{ma_fast}"] if f"sma{ma_fast}" in f else c.rolling(ma_fast, min_periods=ma_fast).mean()
    sma_s = f[f"sma{ma_slow}"] if f"sma{ma_slow}" in f else c.rolling(ma_slow, min_periods=ma_slow).mean()

    above_slow = (c > sma_s).astype(float)
    above_slow[sma_s.isna()] = 0.0
    fast_above_slow = (sma_f > sma_s).astype(float)
    fast_above_slow[sma_f.isna() | sma_s.isna()] = 0.0

    if m == "dual_confirm_dvol55":
        base = 0.5 * above_slow + 0.5 * fast_above_slow
        tv = float(lc.get("dvol_target", 0.55))
        cap = (tv / f["dvol21"]).clip(upper=1.0).fillna(0.0)
        return np.minimum(base, cap).clip(0.0, 1.0)

    if m == "ma200_rvol45":
        tv = float(lc.get("defensive_rvol_target", 0.45))
        cap = (tv / f["rvol21"]).clip(upper=1.0).fillna(0.0)
        return np.minimum(above_slow, cap).clip(0.0, 1.0)

    if m == "ma50_200":
        return fast_above_slow.clip(0.0, 1.0)

    raise ValueError(f"Unknown long strategy mode: {m}")


def latest_long_signal(f: pd.DataFrame, cfg: dict[str, Any], mode: str | None = None) -> dict[str, Any]:
    w = compute_long_weights(f, cfg, mode=mode)
    tw = float(w.iloc[-1])
    row = f.iloc[-1]
    c = float(row["Close"])
    sma50 = float(row["sma50"]) if pd.notna(row["sma50"]) else np.nan
    sma200 = float(row["sma200"]) if pd.notna(row["sma200"]) else np.nan
    reasons = []
    if np.isfinite(sma200):
        reasons.append("收盘 > MA200" if c > sma200 else "收盘 < MA200")
    if np.isfinite(sma50) and np.isfinite(sma200):
        reasons.append("MA50 > MA200" if sma50 > sma200 else "MA50 < MA200")
    if pd.notna(row["dvol21"]):
        reasons.append(f"下行波动={row['dvol21']:.1%}")

    return {
        "mode": mode or cfg.get("long_strategy", {}).get("default_mode", "dual_confirm_dvol55"),
        "signal": "LONG" if tw > 0.01 else "FLAT",
        "target_weight": tw,
        "reason": " | ".join(reasons),
    }
