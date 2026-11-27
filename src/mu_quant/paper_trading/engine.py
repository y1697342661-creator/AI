"""Forward Paper Trading Engine (Append-Only Audit Trail).

Hard rules:
- All strategy versions are recorded on every signal & fill row.
- NEVER silently overwrites historical trade or signal records (strictly append-only).
- Tracks simulated cash, shares, unrealized/realized PnL, and compares against frozen strategy signals.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd

from mu_quant.config import DATA_PAPER, load_active_config
from mu_quant.logs.logger import get_logger

log = get_logger("mu_quant.paper_trading")

SIGNALS_FILE: Path = DATA_PAPER / "signals_log.csv"
LEDGER_FILE: Path = DATA_PAPER / "trades_ledger.csv"

SIGNAL_COLS = [
    "recorded_at", "signal_date", "strategy_version", "portfolio_mode",
    "close_price", "regime", "long_signal", "short_signal",
    "target_weight", "stop_price", "note",
]
LEDGER_COLS = [
    "recorded_at", "trade_date", "strategy_version", "action",
    "shares", "fill_price", "commission", "borrow_fee", "note",
]


def _load_csv_safe(path: Path, cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=cols)
    return pd.read_csv(path, encoding="utf-8-sig")


def record_daily_signal(row: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    """Append a daily signal snapshot without ever overwriting existing rows."""
    p = path or SIGNALS_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    df = _load_csv_safe(p, SIGNAL_COLS)
    rec = {c: row.get(c, "") for c in SIGNAL_COLS}
    rec["recorded_at"] = dt.datetime.now().isoformat(timespec="seconds")
    out = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
    out.to_csv(p, index=False, encoding="utf-8-sig")
    log.info("Appended paper signal: date=%s ver=%s w=%.3f", rec["signal_date"], rec["strategy_version"], float(rec["target_weight"] or 0))
    return rec


def record_paper_trade(row: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    """Append a simulated fill record without ever overwriting existing ledger rows."""
    p = path or LEDGER_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    df = _load_csv_safe(p, LEDGER_COLS)
    rec = {c: row.get(c, "") for c in LEDGER_COLS}
    rec["recorded_at"] = dt.datetime.now().isoformat(timespec="seconds")
    out = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
    out.to_csv(p, index=False, encoding="utf-8-sig")
    log.info("Appended paper trade: date=%s action=%s shares=%s px=%s", rec["trade_date"], rec["action"], rec["shares"], rec["fill_price"])
    return rec


def get_paper_status(latest_price: float, ledger_path: Path | None = None, signals_path: Path | None = None) -> dict[str, Any]:
    cfg = load_active_config()
    initial_cash = float(cfg.get("initial_cash", 100_000.0))
    ledger = _load_csv_safe(ledger_path or LEDGER_FILE, LEDGER_COLS)
    signals = _load_csv_safe(signals_path or SIGNALS_FILE, SIGNAL_COLS)

    cash = initial_cash
    shares = 0.0
    total_fees = 0.0
    for _, r in ledger.iterrows():
        act = str(r.get("action", "")).upper()
        q = float(r.get("shares", 0.0) or 0.0)
        px = float(r.get("fill_price", 0.0) or 0.0)
        fee = float(r.get("commission", 0.0) or 0.0) + float(r.get("borrow_fee", 0.0) or 0.0)
        total_fees += fee
        if act in ("BUY", "COVER"):
            cash -= q * px + fee
            shares += q
        elif act in ("SELL", "SHORT"):
            cash += q * px - fee
            shares -= q

    market_val = shares * latest_price
    equity = cash + market_val
    cur_weight = (market_val / equity) if equity > 0 else 0.0
    return {
        "active_version": cfg.get("active_version", "v2.0.0"),
        "portfolio_mode": cfg.get("portfolio_mode", "long_only"),
        "initial_cash": initial_cash,
        "cash": round(cash, 2),
        "shares": round(shares, 4),
        "market_value": round(market_val, 2),
        "equity": round(equity, 2),
        "total_return_pct": round((equity / initial_cash - 1.0) * 100, 2) if initial_cash > 0 else 0.0,
        "current_weight": round(cur_weight, 4),
        "total_fees": round(total_fees, 2),
        "n_signals_logged": int(len(signals)),
        "n_trades_logged": int(len(ledger)),
        "recent_signals": signals.tail(10).to_dict(orient="records"),
        "recent_trades": ledger.tail(15).to_dict(orient="records"),
        "auto_live_order_disabled": True,
    }
