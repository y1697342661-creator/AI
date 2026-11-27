"""Append-Only Forward Paper Trading Engine for BTC-USDT-SWAP.

Tracks forward equity, position size (BTC & OKX contracts), realized/unrealized PnL,
cumulative funding & fees, peak equity, and drawdown circuit breaker state.
"""
from __future__ import annotations

import datetime as dt
import json
from typing import Any

import pandas as pd

from btc_quant.config import PAPER_DIR
from btc_quant.logs.logger import get_logger
from btc_quant.risk.manager import BtcRiskConfig, OKX_BTC_SWAP_CT_VAL, compute_contract_risk_plan

log = get_logger("btc_quant.paper")

SIGNALS_LOG = PAPER_DIR / "btc_signals_log.csv"
NAV_LOG = PAPER_DIR / "btc_forward_nav.csv"
STATE_FILE = PAPER_DIR / "btc_forward_state.json"


def _default_state(initial_capital: float = 100_000.0) -> dict[str, Any]:
    return {
        "initialized_at": dt.datetime.now().isoformat(timespec="seconds"),
        "last_bar_time": None,
        "initial_capital": initial_capital,
        "cash_usdt": initial_capital,
        "equity_usdt": initial_capital,
        "peak_equity_usdt": initial_capital,
        "drawdown_pct": 0.0,
        "pos_btc": 0.0,
        "pos_contracts": 0.0,
        "entry_px": 0.0,
        "stop_px": None,
        "cum_fees_usdt": 0.0,
        "cum_funding_usdt": 0.0,
        "trades_count": 0,
    }


def load_paper_state(initial_capital: float = 100_000.0) -> dict[str, Any]:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8-sig"))
    st = _default_state(initial_capital)
    STATE_FILE.write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8")
    return st


def save_paper_state(state: dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def record_paper_step(
    signal: dict[str, Any],
    regime: dict[str, Any],
    risk_cfg: BtcRiskConfig | None = None,
    fee_bps: float = 8.0,
    funding_rate_8h: float = 0.0,
) -> dict[str, Any]:
    """Evaluate & record one 4H forward paper trading step (idempotent per bar_time_utc)."""
    st = load_paper_state()
    bar_time = signal["bar_time_utc"]
    px = float(signal["close_px"])
    already_processed = st.get("last_bar_time") == bar_time

    # Mark-to-market existing position
    upl = st["pos_btc"] * (px - st["entry_px"]) if abs(st["pos_btc"]) > 1e-9 else 0.0
    if not already_processed and abs(st["pos_btc"]) > 1e-9 and funding_rate_8h != 0.0:
        f_cf = -st["pos_btc"] * px * funding_rate_8h
        st["cash_usdt"] += f_cf
        st["cum_funding_usdt"] = round(st["cum_funding_usdt"] + f_cf, 4)

    eq = max(0.0, st["cash_usdt"] + upl)
    st["peak_equity_usdt"] = max(st["peak_equity_usdt"], eq)
    dd = (1.0 - eq / st["peak_equity_usdt"]) if st["peak_equity_usdt"] > 0 else 0.0

    plan = compute_contract_risk_plan(
        equity_usdt=eq,
        price=px,
        target_leverage=float(signal["target_leverage"]),
        stop_distance_pct=float(signal["stop_distance_pct"]) / 100.0,
        current_drawdown=dd,
        funding_7d_ann_pct=float(signal.get("funding_7d_ann_pct", 0.0)),
        cfg=risk_cfg,
    )

    action = "HOLD"
    if not already_processed:
        tgt_btc = plan["btc_qty"] if plan["side"] == "LONG" else (-plan["btc_qty"] if plan["side"] == "SHORT" else 0.0)
        cur_btc = float(st["pos_btc"])
        cur_lev = (cur_btc * px / eq) if eq > 0 else 0.0
        tgt_lev = float(plan["approved_leverage"])

        if abs(tgt_lev - cur_lev) >= 0.15 or (cur_btc != 0.0 and tgt_btc == 0.0) or (cur_btc * tgt_btc < 0):
            # Realize existing PnL if closing or flipping
            if abs(cur_btc) > 1e-9 and (tgt_btc == 0.0 or cur_btc * tgt_btc < 0):
                realized = cur_btc * (px - st["entry_px"])
                close_fee = abs(cur_btc) * px * (fee_bps / 1e4)
                st["cash_usdt"] += realized - close_fee
                st["cum_fees_usdt"] = round(st["cum_fees_usdt"] + close_fee, 4)
                st["pos_btc"] = 0.0
                st["entry_px"] = 0.0
                st["trades_count"] += 1
                action = "CLOSE"

            delta = tgt_btc - st["pos_btc"]
            if abs(delta) > 1e-6:
                fee = abs(delta) * px * (fee_bps / 1e4)
                st["cash_usdt"] -= fee
                st["cum_fees_usdt"] = round(st["cum_fees_usdt"] + fee, 4)
                if abs(st["pos_btc"]) < 1e-9:
                    st["entry_px"] = px
                    st["pos_btc"] = tgt_btc
                    action = f"OPEN_{plan['side']}"
                else:
                    if delta * st["pos_btc"] > 0:
                        new_qty = st["pos_btc"] + delta
                        st["entry_px"] = (abs(st["pos_btc"]) * st["entry_px"] + abs(delta) * px) / abs(new_qty)
                        st["pos_btc"] = new_qty
                        action = "ADD"
                    else:
                        realized = (-delta) * (px - st["entry_px"])
                        st["cash_usdt"] += realized
                        st["pos_btc"] = tgt_btc
                        action = "REDUCE"

        st["pos_contracts"] = round(st["pos_btc"] / OKX_BTC_SWAP_CT_VAL, 2)
        st["stop_px"] = plan["stop_px"]
        st["equity_usdt"] = round(max(0.0, st["cash_usdt"] + st["pos_btc"] * (px - st["entry_px"])), 2)
        st["peak_equity_usdt"] = round(max(st["peak_equity_usdt"], st["equity_usdt"]), 2)
        st["drawdown_pct"] = round((1.0 - st["equity_usdt"] / st["peak_equity_usdt"]) * 100.0, 2) if st["peak_equity_usdt"] > 0 else 0.0
        st["last_bar_time"] = bar_time
        save_paper_state(st)

        # Append signal & NAV rows
        sig_row = {
            "recorded_at": dt.datetime.now().isoformat(timespec="seconds"),
            "bar_time_utc": bar_time,
            "close_px": px,
            "regime": regime["regime"],
            "direction": plan["side"],
            "approved_leverage": plan["approved_leverage"],
            "okx_contracts": plan["okx_contracts"],
            "stop_px": plan["stop_px"],
            "est_liq_px": plan["est_liq_px"],
            "action": action,
            "equity_usdt": st["equity_usdt"],
            "drawdown_pct": st["drawdown_pct"],
        }
        s_df = pd.read_csv(SIGNALS_LOG, encoding="utf-8-sig") if SIGNALS_LOG.exists() else pd.DataFrame()
        pd.concat([s_df, pd.DataFrame([sig_row])], ignore_index=True).to_csv(SIGNALS_LOG, index=False, encoding="utf-8-sig")

        nav_row = {
            "bar_time_utc": bar_time,
            "close_px": px,
            "equity_usdt": st["equity_usdt"],
            "pos_btc": round(st["pos_btc"], 5),
            "leverage": round((st["pos_btc"] * px / st["equity_usdt"]), 3) if st["equity_usdt"] > 0 else 0.0,
            "drawdown_pct": st["drawdown_pct"],
            "cum_funding_usdt": st["cum_funding_usdt"],
            "cum_fees_usdt": st["cum_fees_usdt"],
        }
        n_df = pd.read_csv(NAV_LOG, encoding="utf-8-sig") if NAV_LOG.exists() else pd.DataFrame()
        pd.concat([n_df, pd.DataFrame([nav_row])], ignore_index=True).to_csv(NAV_LOG, index=False, encoding="utf-8-sig")
        log.info("Recorded BTC paper step %s | %s | eq=$%.2f | pos=%.4f BTC", bar_time, action, st["equity_usdt"], st["pos_btc"])

    return {"state": st, "risk_plan": plan, "action": action, "already_processed": already_processed}


def get_paper_summary() -> dict[str, Any]:
    st = load_paper_state()
    recent_signals = []
    if SIGNALS_LOG.exists():
        df = pd.read_csv(SIGNALS_LOG, encoding="utf-8-sig")
        recent_signals = df.tail(15).fillna("").to_dict(orient="records")
    return {"state": st, "recent_signals": recent_signals}
