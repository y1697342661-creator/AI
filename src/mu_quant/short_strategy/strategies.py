"""Independent Short Strategy Module (NEVER an inversion of Long rules).

Implements S3_Breakdown_Vol (the sole short candidate passing Stage 2 & Stage 3 random-short tests):
- Established bear structure: Close < MA200 AND MA50 < MA200
- Valid support breakdown: Close < DonchianLow(20) [strictly shifted by 1 bar]
- Volume confirmation: vol_ratio20 > 1.2
- Weak close: CLV < 0.35
- Anti-oversold protection: dist_ma20 > -12% (prevents chasing panic bottoms)
- Exit: Close > MA20 OR RSI14 < 25 (take-profit on oversold flush) OR 2.0*ATR stop / 2.5*ATR trail / 2R target / 40d time cap / mandatory pre-earnings exit.

Tagged as EXPERIMENTAL pending forward paper-trading validation (small historical sample size: 19 trades in 22 years).
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def compute_short_signals(f: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.Series, pd.Series]:
    sc = cfg.get("short_strategy", {})
    vol_min = float(sc.get("vol_ratio_min", 1.2))
    clv_max = float(sc.get("clv_max", 0.35))
    dist_min = float(sc.get("dist_ma20_min", -0.12))
    exit_rsi = float(sc.get("exit_rsi_min", 25.0))

    entries = (
        (f["Close"] < f["don_low20"])
        & (f["Close"] < f["sma200"])
        & (f["sma50"] < f["sma200"])
        & (f["vol_ratio20"] > vol_min)
        & (f["clv"] < clv_max)
        & (f["dist_ma20"] > dist_min)
    ).fillna(False)

    exits = ((f["Close"] > f["sma20"]) | (f["rsi14"] < exit_rsi)).fillna(False)
    return entries, exits


def latest_short_signal(f: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    entries, exits = compute_short_signals(f, cfg)
    sc = cfg.get("short_strategy", {})
    row = f.iloc[-1]
    trig = bool(entries.iloc[-1])
    ex = bool(exits.iloc[-1])

    conds = []
    conds.append("熊市排列" if (row["Close"] < row["sma200"] and row["sma50"] < row["sma200"]) else "非熊市排列")
    if pd.notna(row["don_low20"]):
        conds.append("跌破20日低" if row["Close"] < row["don_low20"] else "未破20日低")
    if pd.notna(row["vol_ratio20"]):
        conds.append(f"量比={row['vol_ratio20']:.2f}")

    w_short = 0.0
    if trig and pd.notna(row["rvol21"]) and row["rvol21"] > 0:
        w_cap = min(1.0, float(sc.get("vol_target", 0.35)) / float(row["rvol21"]))
        w_short = w_cap * float(sc.get("portfolio_weight", 0.5))

    return {
        "name": sc.get("name", "S3_Breakdown_Vol"),
        "experimental": True,
        "signal": "SHORT" if trig else ("COVER_SIGNAL" if ex else "FLAT"),
        "target_weight": -float(w_short) if trig else 0.0,
        "reason": " | ".join(conds),
    }
