"""Market regime classifier (risk monitoring tool ONLY — NOT a predictive trading signal).

Stage 1/3 audit confirmed that 4-state regimes distinguish volatility/trend characteristics
but show 0/20 statistically significant return predictability after BH-FDR correction.
Therefore this module is strictly used for dashboard risk monitoring and context reporting.
"""
from __future__ import annotations

import pandas as pd

STATE_LABELS = {
    "strong_up": "强趋势上涨",
    "range": "震荡整理",
    "down": "下跌趋势",
    "extreme_vol": "极端波动（高风控警惕）",
    "warmup": "预热期",
}


def classify_regime(f: pd.DataFrame) -> pd.Series:
    warm = f[["sma200", "sma50", "adx14", "atr_rank252"]].notna().all(axis=1)
    extreme_vol = f["atr_rank252"] >= 0.90
    strong_up = (f["Close"] > f["sma200"]) & (f["sma50"] > f["sma200"]) & (f["sma200_slope20"] > 0) & (f["adx14"] >= 25.0)
    down = (f["Close"] < f["sma200"]) & (f["sma50"] < f["sma200"])

    state = pd.Series("range", index=f.index, dtype="object")
    state[down] = "down"
    state[strong_up] = "strong_up"
    state[extreme_vol] = "extreme_vol"
    state[~warm] = "warmup"
    return state


def latest_regime_summary(f: pd.DataFrame) -> dict:
    s = classify_regime(f)
    code = str(s.iloc[-1])
    row = f.iloc[-1]
    return {
        "code": code,
        "label": STATE_LABELS.get(code, code),
        "atr_pct": float(row["atr_pct"]) if pd.notna(row["atr_pct"]) else None,
        "atr_rank252": float(row["atr_rank252"]) if pd.notna(row["atr_rank252"]) else None,
        "rvol21": float(row["rvol21"]) if pd.notna(row["rvol21"]) else None,
        "dvol21": float(row["dvol21"]) if pd.notna(row["dvol21"]) else None,
        "adx14": float(row["adx14"]) if pd.notna(row["adx14"]) else None,
        "sox_above200": bool(row["sox_above200"] > 0.5) if pd.notna(row["sox_above200"]) else None,
        "smh_above200": bool(row["smh_above200"] > 0.5) if pd.notna(row["smh_above200"]) else None,
        "note": "仅作风险监控与环境描述工具，不直接作为择时预测信号。",
    }
