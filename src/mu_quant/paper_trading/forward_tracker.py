"""Standalone Forward Paper-Trading Scorecard (100% separate from historical backtests).

Tracks real forward performance from Day 1 of frozen v2.0.0 deployment:
- `data/paper_trading/forward_nav.csv` (append-only daily NAV & benchmark comparison)
- `data/paper_trading/forward_state.json` (persistent simulated portfolio state: cash, shares, peak NAV)

Compares side-by-side:
1. Frozen v2.0.0 Strategy Forward NAV (strict T-close signal -> T+1 Open fill, 15bp/side real cost, rebalance_tol=5%)
2. Same-Start Buy & Hold MU NAV (100% long from Day 1)
3. Forward Alpha (Strategy Return - Buy & Hold Return), Forward MaxDD, and live OKX XMU-USDT spread/tracking error.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mu_quant.config import DATA_PAPER, load_strategy_version
from mu_quant.logs.logger import get_logger

log = get_logger("mu_quant.forward_tracker")

FORWARD_NAV_FILE: Path = DATA_PAPER / "forward_nav.csv"
FORWARD_STATE_FILE: Path = DATA_PAPER / "forward_state.json"

NAV_COLS = [
    "date", "recorded_at", "strategy_version", "mu_open", "mu_close",
    "target_weight", "actual_weight", "action", "trade_cost_usd",
    "strategy_nav", "strategy_ret_pct", "strategy_dd_pct",
    "bh_nav", "bh_ret_pct", "bh_dd_pct", "alpha_pct",
    "xmu_px", "xmu_spread_bps", "note",
]


def _load_nav_df(path: Path | None = None) -> pd.DataFrame:
    p = path or FORWARD_NAV_FILE
    if not p.exists():
        return pd.DataFrame(columns=NAV_COLS)
    return pd.read_csv(p, encoding="utf-8-sig")


def update_forward_scorecard(
    *,
    market_date: str,
    mu_open: float,
    mu_close: float,
    target_weight: float,
    strategy_version: str = "v2.0.0",
    initial_capital: float = 100_000.0,
    xmu_px: float | None = None,
    xmu_spread_bps: float | None = None,
    nav_path: Path | None = None,
    state_path: Path | None = None,
) -> dict[str, Any]:
    """Update standalone forward paper-trading NAV for `market_date` (idempotent per market_date, append-only)."""
    np_file = nav_path or FORWARD_NAV_FILE
    st_file = state_path or FORWARD_STATE_FILE
    np_file.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_strategy_version(strategy_version)
    comm = float(cfg["execution"].get("commission_bps", 10.0)) / 1e4
    slip = float(cfg["execution"].get("slippage_bps", 5.0)) / 1e4
    rebal_tol = float(cfg["long_strategy"].get("rebalance_tolerance", 0.05))

    df = _load_nav_df(np_file)
    if not df.empty and market_date in set(df["date"].astype(str)):
        # Already recorded for this trading date; return latest summary without duplicating or overwriting history
        return get_forward_scorecard(nav_path=np_file, state_path=st_file)

    tw = float(np.clip(target_weight, 0.0, 1.0))
    now_str = dt.datetime.now().isoformat(timespec="seconds")

    if not st_file.exists() or df.empty:
        # Day 1 initialization: start clean at initial_capital on market_date close, schedule pending target_weight for next open
        # Also record initial Day-1 baseline NAV = initial_capital for BOTH strategy and buy-and-hold!
        state = {
            "start_date": market_date,
            "initial_capital": initial_capital,
            "strategy_version": strategy_version,
            "cash": initial_capital,
            "shares": 0.0,
            "pending_target_weight": tw,
            "peak_strategy_nav": initial_capital,
            "bh_shares": initial_capital / (mu_close * (1.0 + slip + comm)),
            "peak_bh_nav": initial_capital,
            "total_cost_usd": 0.0,
            "n_rebalances": 0,
        }
        row = {
            "date": market_date, "recorded_at": now_str, "strategy_version": strategy_version,
            "mu_open": round(mu_open, 4), "mu_close": round(mu_close, 4),
            "target_weight": round(tw, 4), "actual_weight": 0.0,
            "action": "INIT_DAY1", "trade_cost_usd": 0.0,
            "strategy_nav": round(initial_capital, 2), "strategy_ret_pct": 0.0, "strategy_dd_pct": 0.0,
            "bh_nav": round(initial_capital, 2), "bh_ret_pct": 0.0, "bh_dd_pct": 0.0,
            "alpha_pct": 0.0,
            "xmu_px": round(xmu_px, 4) if xmu_px else "",
            "xmu_spread_bps": round(xmu_spread_bps, 2) if xmu_spread_bps is not None else "",
            "note": f"前瞻模拟盘起跑日（待下一美股开盘建仓目标 {tw*100:.1f}%）",
        }
    else:
        state = json.loads(st_file.read_text(encoding="utf-8-sig"))
        cash = float(state["cash"])
        shares = float(state["shares"])
        exec_tw = float(state.get("pending_target_weight", 0.0))

        # Execute pending T-1 close signal at T Open (strict T+1 Open!)
        eq_open = cash + shares * mu_open
        cur_w = (shares * mu_open / eq_open) if eq_open > 0 else 0.0
        action = "HOLD"
        trade_cost = 0.0

        need_rebal = (abs(exec_tw - cur_w) > rebal_tol) or (exec_tw == 0.0 and shares > 1e-6)
        if need_rebal and eq_open > 0:
            target_shares = (exec_tw * eq_open) / mu_open
            delta = target_shares - shares
            if delta > 1e-6:
                fill = mu_open * (1.0 + slip)
                max_aff = max(0.0, cash) / (fill * (1.0 + comm))
                delta = min(delta, max_aff)
                fee = delta * fill * comm
                slip_c = delta * mu_open * slip
                cash -= delta * fill + fee
                shares += delta
                trade_cost = fee + slip_c
                action = "BUY"
                state["n_rebalances"] = int(state.get("n_rebalances", 0)) + 1
            elif delta < -1e-6:
                sell_q = min(shares, abs(delta))
                fill = mu_open * (1.0 - slip)
                fee = sell_q * fill * comm
                slip_c = sell_q * mu_open * slip
                cash += sell_q * fill - fee
                shares -= sell_q
                trade_cost = fee + slip_c
                action = "SELL"
                state["n_rebalances"] = int(state.get("n_rebalances", 0)) + 1

        strat_nav = max(0.0, cash + shares * mu_close)
        actual_w = (shares * mu_close / strat_nav) if strat_nav > 0 else 0.0
        bh_nav = float(state["bh_shares"]) * mu_close

        init_cap = float(state["initial_capital"])
        peak_s = max(float(state["peak_strategy_nav"]), strat_nav)
        peak_b = max(float(state["peak_bh_nav"]), bh_nav)
        s_ret = (strat_nav / init_cap - 1.0) * 100.0
        b_ret = (bh_nav / init_cap - 1.0) * 100.0
        s_dd = (strat_nav / peak_s - 1.0) * 100.0 if peak_s > 0 else 0.0
        b_dd = (bh_nav / peak_b - 1.0) * 100.0 if peak_b > 0 else 0.0

        state.update({
            "cash": cash, "shares": shares,
            "pending_target_weight": tw,
            "peak_strategy_nav": peak_s, "peak_bh_nav": peak_b,
            "total_cost_usd": float(state.get("total_cost_usd", 0.0)) + trade_cost,
        })
        row = {
            "date": market_date, "recorded_at": now_str, "strategy_version": strategy_version,
            "mu_open": round(mu_open, 4), "mu_close": round(mu_close, 4),
            "target_weight": round(tw, 4), "actual_weight": round(actual_w, 4),
            "action": action, "trade_cost_usd": round(trade_cost, 2),
            "strategy_nav": round(strat_nav, 2), "strategy_ret_pct": round(s_ret, 2), "strategy_dd_pct": round(s_dd, 2),
            "bh_nav": round(bh_nav, 2), "bh_ret_pct": round(b_ret, 2), "bh_dd_pct": round(b_dd, 2),
            "alpha_pct": round(s_ret - b_ret, 2),
            "xmu_px": round(xmu_px, 4) if xmu_px else "",
            "xmu_spread_bps": round(xmu_spread_bps, 2) if xmu_spread_bps is not None else "",
            "note": f"按T+1开盘执行前日目标{exec_tw*100:.1f}%，明日新目标{tw*100:.1f}%",
        }

    out_df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    out_df.to_csv(np_file, index=False, encoding="utf-8-sig")
    st_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Updated forward scorecard %s: strat_nav=$%.2f bh_nav=$%.2f action=%s", market_date, row["strategy_nav"], row["bh_nav"], row["action"])
    return get_forward_scorecard(nav_path=np_file, state_path=st_file)


def get_forward_scorecard(nav_path: Path | None = None, state_path: Path | None = None) -> dict[str, Any]:
    np_file = nav_path or FORWARD_NAV_FILE
    st_file = state_path or FORWARD_STATE_FILE
    df = _load_nav_df(np_file)
    if df.empty or not st_file.exists():
        return {
            "initialized": False,
            "days_tracked": 0,
            "start_date": None,
            "rows": [],
        }
    state = json.loads(st_file.read_text(encoding="utf-8-sig"))
    last = df.iloc[-1]
    max_s_dd = float(df["strategy_dd_pct"].min())
    max_b_dd = float(df["bh_dd_pct"].min())
    return {
        "initialized": True,
        "start_date": state["start_date"],
        "last_date": str(last["date"]),
        "days_tracked": int(len(df)),
        "strategy_version": state.get("strategy_version", "v2.0.0"),
        "initial_capital": float(state["initial_capital"]),
        "strategy_nav": float(last["strategy_nav"]),
        "strategy_ret_pct": float(last["strategy_ret_pct"]),
        "strategy_max_dd_pct": round(max_s_dd, 2),
        "bh_nav": float(last["bh_nav"]),
        "bh_ret_pct": float(last["bh_ret_pct"]),
        "bh_max_dd_pct": round(max_b_dd, 2),
        "alpha_pct": float(last["alpha_pct"]),
        "actual_weight": float(last["actual_weight"]),
        "pending_next_weight": float(state.get("pending_target_weight", 0.0)),
        "total_cost_usd": round(float(state.get("total_cost_usd", 0.0)), 2),
        "n_rebalances": int(state.get("n_rebalances", 0)),
        "rows": df.tail(30).to_dict(orient="records"),
    }
