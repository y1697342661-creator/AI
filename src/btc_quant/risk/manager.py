"""Perpetual Futures Risk Manager for BTC-USDT-SWAP.

Computes isolated margin sizing, OKX contract quantity (ctVal = 0.01 BTC),
estimated liquidation price, stop-loss risk budget, funding carry alerts,
and drawdown circuit breaker status.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

OKX_BTC_SWAP_CT_VAL = 0.01  # 1 contract = 0.01 BTC
OKX_BTC_SWAP_LOT_SZ = 0.01  # Minimum step = 0.01 contracts (= 0.0001 BTC)
OKX_MMR_TIER1 = 0.004       # 0.4% maintenance margin rate


@dataclass
class BtcRiskConfig:
    max_long_leverage: float = 2.0
    max_short_leverage: float = 1.2
    hard_cap_leverage: float = 3.0       # Absolute safety ceiling
    single_trade_risk_pct: float = 0.025 # Max 2.5% equity risk to initial stop
    dd_brake_start: float = 0.12
    dd_brake_full: float = 0.25
    dd_min_scale: float = 0.35
    dd_halt_threshold: float = 0.32      # Hard kill-switch at 32% drawdown
    min_liq_buffer_pct: float = 0.20     # Require >=20% distance to liquidation


def compute_contract_risk_plan(
    equity_usdt: float,
    price: float,
    target_leverage: float,
    stop_distance_pct: float,
    current_drawdown: float = 0.0,
    funding_7d_ann_pct: float = 0.0,
    cfg: BtcRiskConfig | None = None,
) -> dict[str, Any]:
    c = cfg or BtcRiskConfig()

    # 1. Drawdown circuit breaker
    if current_drawdown >= c.dd_halt_threshold:
        dd_scale = 0.0
        risk_state = "HALTED_MAX_DD"
    elif current_drawdown >= c.dd_brake_full:
        dd_scale = c.dd_min_scale
        risk_state = "DEEP_DD_BRAKE"
    elif current_drawdown >= c.dd_brake_start:
        frac = (current_drawdown - c.dd_brake_start) / max(c.dd_brake_full - c.dd_brake_start, 1e-9)
        dd_scale = 1.0 - frac * (1.0 - c.dd_min_scale)
        risk_state = "DD_BRAKE_ACTIVE"
    else:
        dd_scale = 1.0
        risk_state = "NORMAL"

    # 2. Clamp by directional cap & hard safety ceiling
    if target_leverage > 0:
        capped_lev = min(target_leverage, c.max_long_leverage, c.hard_cap_leverage) * dd_scale
        side = "LONG"
    elif target_leverage < 0:
        capped_lev = -min(abs(target_leverage), c.max_short_leverage, c.hard_cap_leverage) * dd_scale
        side = "SHORT"
    else:
        capped_lev = 0.0
        side = "FLAT"

    # 3. Single-trade stop-loss risk budget check
    if abs(capped_lev) > 0 and stop_distance_pct > 0:
        max_lev_by_stop = c.single_trade_risk_pct / stop_distance_pct
        if abs(capped_lev) > max_lev_by_stop:
            capped_lev = (1.0 if capped_lev > 0 else -1.0) * max_lev_by_stop

    notional_usdt = abs(capped_lev) * equity_usdt
    btc_qty = notional_usdt / price if price > 0 else 0.0
    # Convert to OKX BTC-USDT-SWAP contracts (1 contract = 0.01 BTC, lotSz = 0.01)
    contracts = round(btc_qty / OKX_BTC_SWAP_CT_VAL, 2)
    actual_btc = contracts * OKX_BTC_SWAP_CT_VAL
    actual_notional = actual_btc * price

    # Isolated margin leverage setting (use ceil(abs_lev) clamped to [1, 3] for healthy margin buffer)
    iso_lev = max(1, min(int(c.hard_cap_leverage), int(round(max(abs(capped_lev), 1.0) + 0.49))))
    init_margin_usdt = actual_notional / iso_lev if iso_lev > 0 else 0.0

    # Estimated liquidation price (isolated margin approximation)
    if side == "LONG" and actual_notional > 0:
        est_liq_px = round(price * (1.0 - (1.0 / iso_lev) + OKX_MMR_TIER1), 2)
        liq_dist_pct = round((price - est_liq_px) / price * 100.0, 2)
        stop_px = round(price * (1.0 - stop_distance_pct), 2)
    elif side == "SHORT" and actual_notional > 0:
        est_liq_px = round(price * (1.0 + (1.0 / iso_lev) - OKX_MMR_TIER1), 2)
        liq_dist_pct = round((est_liq_px - price) / price * 100.0, 2)
        stop_px = round(price * (1.0 + stop_distance_pct), 2)
    else:
        est_liq_px = None
        liq_dist_pct = None
        stop_px = None

    est_stop_loss_usdt = round(actual_notional * stop_distance_pct, 2)
    est_stop_loss_pct_eq = round((est_stop_loss_usdt / equity_usdt * 100.0), 2) if equity_usdt > 0 else 0.0

    warnings: list[str] = []
    if risk_state != "NORMAL":
        warnings.append(f"回撤熔断生效 ({risk_state})，仓位缩放至 {dd_scale:.0%}")
    if side == "LONG" and funding_7d_ann_pct > 35.0:
        warnings.append(f"多头资金费率偏高 (7日年化 {funding_7d_ann_pct:.1f}%)，已自动降杠杆")
    if side == "SHORT" and funding_7d_ann_pct < -20.0:
        warnings.append(f"空头拥挤 (7日年化 {funding_7d_ann_pct:.1f}%)，注意轧空风险")
    if liq_dist_pct is not None and liq_dist_pct < c.min_liq_buffer_pct * 100.0:
        warnings.append(f"强平距离偏近 ({liq_dist_pct:.1f}%)，建议保持低倍逐仓")

    return {
        "risk_state": risk_state,
        "dd_scale": round(dd_scale, 3),
        "side": side,
        "approved_leverage": round(capped_lev, 3),
        "isolated_margin_leverage": iso_lev,
        "notional_usdt": round(actual_notional, 2),
        "btc_qty": round(actual_btc, 5),
        "okx_contracts": contracts,
        "required_margin_usdt": round(init_margin_usdt, 2),
        "stop_px": stop_px,
        "est_liq_px": est_liq_px,
        "liq_distance_pct": liq_dist_pct,
        "est_stop_loss_usdt": est_stop_loss_usdt,
        "est_stop_loss_pct_eq": est_stop_loss_pct_eq,
        "warnings": warnings,
    }
