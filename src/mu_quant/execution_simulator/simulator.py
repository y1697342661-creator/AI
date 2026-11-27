"""Simulated order execution module.

Hard safety invariant:
- NEVER connects to any live broker or exchange API.
- ONLY produces auditable simulated execution tickets with slippage, commission, and locate checks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from mu_quant.config import LIVE_AUTO_ORDER_FORBIDDEN
from mu_quant.logs.logger import get_logger

log = get_logger("mu_quant.execution_simulator")


@dataclass
class SimulatedOrderTicket:
    date: str
    ticker: str
    action: str           # BUY | SELL | SHORT | COVER | HOLD
    shares: float
    ref_price: float
    est_fill_price: float
    est_commission: float
    est_slippage_cost: float
    target_weight: float
    stop_price: float
    strategy_version: str
    is_live_order: bool = False
    note: str = ""


def simulate_rebalance_order(
    *,
    date: str,
    ticker: str,
    current_shares: float,
    equity: float,
    ref_price: float,
    target_weight: float,
    stop_price: float,
    strategy_version: str,
    commission_bps: float = 10.0,
    slippage_bps: float = 5.0,
    rebalance_tol: float = 0.05,
) -> dict[str, Any]:
    if not LIVE_AUTO_ORDER_FORBIDDEN:
        raise RuntimeError("Safety violation: live auto-ordering is strictly forbidden.")

    cur_w = (current_shares * ref_price / equity) if equity > 0 else 0.0
    if abs(target_weight - cur_w) < rebalance_tol and not (target_weight == 0.0 and abs(current_shares) > 1e-6):
        ticket = SimulatedOrderTicket(
            date=date, ticker=ticker, action="HOLD", shares=0.0, ref_price=ref_price,
            est_fill_price=ref_price, est_commission=0.0, est_slippage_cost=0.0,
            target_weight=target_weight, stop_price=stop_price, strategy_version=strategy_version,
            note=f"目标仓位({target_weight:.1%})与当前仓位({cur_w:.1%})差异小于容忍度({rebalance_tol:.0%})，保持观望。",
        )
        return asdict(ticket)

    target_shares = (target_weight * equity / ref_price) if ref_price > 0 else 0.0
    delta = target_shares - current_shares
    slip = slippage_bps / 1e4
    comm = commission_bps / 1e4

    if delta > 0:
        action = "COVER" if current_shares < 0 and target_shares <= 0 else "BUY"
        fill_px = ref_price * (1.0 + slip)
    else:
        action = "SHORT" if target_shares < 0 else "SELL"
        fill_px = ref_price * (1.0 - slip)

    qty = abs(delta)
    est_comm = qty * fill_px * comm
    est_slip = qty * ref_price * slip
    note = "仅模拟成交估算，严禁自动实盘下单。"
    if action == "SHORT":
        note += " 做空需人工核验券商融券可用性与实时费率。"

    ticket = SimulatedOrderTicket(
        date=date, ticker=ticker, action=action, shares=round(qty, 4),
        ref_price=round(ref_price, 4), est_fill_price=round(fill_px, 4),
        est_commission=round(est_comm, 2), est_slippage_cost=round(est_slip, 2),
        target_weight=round(target_weight, 4), stop_price=round(stop_price, 4),
        strategy_version=strategy_version, is_live_order=False, note=note,
    )
    log.info("Simulated ticket: %s %s %.4f @ %.2f (ver=%s)", action, ticker, qty, fill_px, strategy_version)
    return asdict(ticket)
