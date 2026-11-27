"""Risk management module.

Strictly separates:
1. Account equity risk & drawdown brake/halt-resume.
2. Single-trade ATR stop-loss distance & potential loss calculation.
3. Portfolio volatility budget (realized vol / downside-only vol, NEVER ATR).
4. Hard constraints: zero leverage, no averaging down, short borrow & recall warnings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class RiskConfig:
    max_gross_exposure: float = 1.0          # Zero leverage hard cap
    single_trade_risk_budget: float = 0.02   # 2% reference single-trade risk budget
    dd_brake_enabled: bool = False
    dd_brake_start: float = 0.15
    dd_brake_full: float = 0.35
    dd_min_scale: float = 0.25               # Never 0 while active (prevents deadlock)
    dd_halt_threshold: float = 0.40
    dd_resume_days: int = 120
    max_consec_losses: int = 4
    cooldown_days: int = 10
    earnings_blackout_before: int = 3
    earnings_blackout_after: int = 1


@dataclass
class TradeRiskPlan:
    side: str
    entry_ref_price: float
    stop_price: float
    stop_distance_pct: float
    target_weight: float
    shares: float
    notional: float
    potential_loss_usd: float
    potential_loss_pct: float
    gap_2x_loss_pct: float
    warnings: list[str] = field(default_factory=list)


def compute_trade_risk_plan(
    *,
    equity: float,
    price: float,
    atr: float,
    side: str,
    target_weight: float,
    stop_atr_mult: float,
    is_earnings_blackout: bool = False,
    borrow_annual: float = 0.015,
) -> TradeRiskPlan:
    warnings: list[str] = []
    w = float(np.clip(abs(target_weight), 0.0, 1.0))
    if side == "FLAT" or w <= 1e-6 or price <= 0 or equity <= 0:
        return TradeRiskPlan(
            side="FLAT", entry_ref_price=price, stop_price=0.0, stop_distance_pct=0.0,
            target_weight=0.0, shares=0.0, notional=0.0, potential_loss_usd=0.0,
            potential_loss_pct=0.0, gap_2x_loss_pct=0.0,
            warnings=["当前为空仓（合法的风险决策）"],
        )

    if side == "LONG":
        stop_px = max(0.01, price - stop_atr_mult * atr)
        stop_dist = (price - stop_px) / price
    else:
        stop_px = price + stop_atr_mult * atr
        stop_dist = (stop_px - price) / price
        warnings.append(f"做空需承担融券费（默认估算 {borrow_annual:.1%}/年）、借券可用性及召回/逼空跳空风险。")

    if is_earnings_blackout:
        warnings.append("处于财报窗口（前3/后1交易日），建议禁止新开仓或空头强制提前平仓。")

    notional = equity * w
    shares = notional / price
    pot_loss_usd = notional * stop_dist
    pot_loss_pct = pot_loss_usd / equity
    return TradeRiskPlan(
        side=side,
        entry_ref_price=float(price),
        stop_price=float(stop_px),
        stop_distance_pct=float(stop_dist),
        target_weight=float(w if side == "LONG" else -w),
        shares=float(shares),
        notional=float(notional),
        potential_loss_usd=float(pot_loss_usd),
        potential_loss_pct=float(pot_loss_pct),
        gap_2x_loss_pct=float(pot_loss_pct * 2.0),
        warnings=warnings,
    )


def evaluate_account_risk_status(equity_series: pd.Series, cfg: RiskConfig) -> dict[str, Any]:
    eq = equity_series.dropna()
    if eq.empty:
        return {"current_dd": 0.0, "max_dd": 0.0, "status": "NORMAL", "scale": 1.0}
    peak = eq.cummax()
    dd_s = 1.0 - eq / peak
    cur_dd = float(dd_s.iloc[-1])
    max_dd = float(dd_s.max())
    status = "NORMAL"
    scale = 1.0
    if cur_dd >= cfg.dd_halt_threshold:
        status = "HALT_ALERT"
        scale = 0.0 if cfg.dd_brake_enabled else 1.0
    elif cur_dd >= cfg.dd_brake_start:
        status = "CAUTION_DD"
        if cfg.dd_brake_enabled:
            span = max(cfg.dd_brake_full - cfg.dd_brake_start, 1e-9)
            scale = float(np.clip(1.0 - (cur_dd - cfg.dd_brake_start) / span, cfg.dd_min_scale, 1.0))
    return {
        "current_dd": -cur_dd,
        "max_dd": -max_dd,
        "status": status,
        "dd_brake_enabled": cfg.dd_brake_enabled,
        "risk_scale": scale,
        "leverage_allowed": False,
        "averaging_down_allowed": False,
    }
