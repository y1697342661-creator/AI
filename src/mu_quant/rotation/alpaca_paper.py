"""Alpaca Paper Trading (SIMULATED ONLY) Auto-Sync for US Stock Full-Market Screener.

HARD SAFETY INVARIANTS (cannot be bypassed):
1. HARDCODED endpoint `https://paper-api.alpaca.markets` (live `api.alpaca.markets` is unreachable by design).
2. Clean US stock ticker validation (`^[A-Z]{1,5}$`), single-stock hard cap `<= 40%`, total weight `<= 100%`.
3. Zero leverage: sizes strictly off `equity` (NEVER `buying_power`), sells exited positions first, buys with cash only.
4. Supports GitHub Secret `ALPACA_PAPER` / `AMEREICA` (JSON `{"apiKey":"...","secretKey":"..."}`), env vars, or local gitignored
   `configs/alpaca_paper_secrets.local.json`.
5. Pure Python stdlib (zero extra dependencies).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mu_quant.config import CONFIGS_DIR, ROOT
from mu_quant.logs.logger import get_logger

log = get_logger("mu_quant.alpaca_paper")

# HARDCODED PAPER TRADING ENDPOINT ONLY — NEVER LIVE
ALPACA_PAPER_BASE = "https://paper-api.alpaca.markets"
SECRETS_FILE = CONFIGS_DIR / "alpaca_paper_secrets.local.json"
ALPACA_DATA_DIR = ROOT / "data" / "rotation"
ALPACA_ORDERS_LOG = ALPACA_DATA_DIR / "alpaca_paper_orders.csv"
ALPACA_SNAPSHOT_FILE = ALPACA_DATA_DIR / "alpaca_account_snapshot.json"
ALPACA_EQUITY_CURVE_FILE = ALPACA_DATA_DIR / "alpaca_equity_curve.csv"
ALPACA_RISK_STATE_FILE = ALPACA_DATA_DIR / "alpaca_risk_state.json"
SCREENER_ALLOC_FILE = ROOT / "data" / "screener" / "latest_allocation.json"
ALPACA_DATA_DIR.mkdir(parents=True, exist_ok=True)
INITIAL_PAPER_EQUITY = 100_000.0
_US_TICKER_RE = re.compile(r"^[A-Z]{1,5}$")


def _load_alpaca_paper_secrets() -> dict[str, Any]:
    # 1) Single JSON secret (GitHub Secret `ALPACA_PAPER` or `AMEREICA`)
    raw_json = (os.environ.get("ALPACA_PAPER_JSON") or "").strip()
    if raw_json:
        try:
            obj = json.loads(raw_json)
            k = str(obj.get("apiKey") or obj.get("api_key") or obj.get("APCA_API_KEY_ID") or "").strip()
            s = str(obj.get("secretKey") or obj.get("secret_key") or obj.get("APCA_API_SECRET_KEY") or "").strip()
            if k and s:
                return {"apiKey": k, "secretKey": s, "paper_only": True, "source": "env_json"}
        except Exception:
            pass

    # 2) Separate env vars
    env_k = (os.environ.get("ALPACA_PAPER_API_KEY") or os.environ.get("APCA_API_KEY_ID") or "").strip()
    env_s = (os.environ.get("ALPACA_PAPER_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY") or "").strip()
    if env_k and env_s:
        return {"apiKey": env_k, "secretKey": env_s, "paper_only": True, "source": "env"}

    # 3) Local gitignored file
    if not SECRETS_FILE.exists():
        raise FileNotFoundError(f"Missing Alpaca paper secrets: {SECRETS_FILE} (and no ALPACA_PAPER env vars set)")
    sec = json.loads(SECRETS_FILE.read_text(encoding="utf-8-sig"))
    if sec.get("paper_only") is not True:
        raise RuntimeError("Safety block: alpaca_paper_secrets.local.json MUST set paper_only=true.")
    for field in ("apiKey", "secretKey"):
        if not sec.get(field):
            raise ValueError(f"Missing required field '{field}' in {SECRETS_FILE}")
    return sec


def _alpaca_request(method: str, path: str, payload: dict[str, Any] | None = None) -> tuple[int, Any]:
    sec = _load_alpaca_paper_secrets()
    assert ALPACA_PAPER_BASE.startswith("https://paper-api.alpaca.markets"), "Safety block: Paper API ONLY"
    url = ALPACA_PAPER_BASE + path
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {
        "APCA-API-KEY-ID": sec["apiKey"],
        "APCA-API-SECRET-KEY": sec["secretKey"],
        "Accept": "application/json",
        "User-Agent": "us-screener-alpaca-paper/2.0",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, (json.loads(raw) if raw.strip() else {})
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"message": raw}
        return e.code, parsed


def get_alpaca_paper_account() -> dict[str, Any]:
    code, acc = _alpaca_request("GET", "/v2/account")
    if code != 200:
        return {"ok": False, "status": code, "error": acc}
    code_p, pos = _alpaca_request("GET", "/v2/positions")
    positions = pos if code_p == 200 and isinstance(pos, list) else []

    equity = float(acc.get("equity") or 0.0)
    raw_last_eq = float(acc.get("last_equity") or 0.0)
    last_equity = raw_last_eq if raw_last_eq > 0 else equity
    cash = float(acc.get("cash") or 0.0)
    daily_pl = equity - last_equity
    daily_plpc = (daily_pl / last_equity) if last_equity > 0 else 0.0
    total_return_usd = equity - INITIAL_PAPER_EQUITY
    total_return_pct = (total_return_usd / INITIAL_PAPER_EQUITY) if INITIAL_PAPER_EQUITY > 0 else 0.0

    pos_list: list[dict[str, Any]] = []
    for p in positions:
        mv = float(p.get("market_value") or 0.0)
        pos_list.append({
            "symbol": str(p.get("symbol") or "").upper(),
            "qty": float(p.get("qty") or 0.0),
            "avg_entry_price": float(p.get("avg_entry_price") or 0.0),
            "current_price": float(p.get("current_price") or 0.0),
            "cost_basis": float(p.get("cost_basis") or 0.0),
            "market_value": mv,
            "weight": round(mv / equity, 4) if equity > 0 else 0.0,
            "unrealized_pl": float(p.get("unrealized_pl") or 0.0),
            "unrealized_plpc": float(p.get("unrealized_plpc") or 0.0),
            "change_today": float(p.get("change_today") or 0.0),
        })
    pos_list.sort(key=lambda x: x["market_value"], reverse=True)

    return {
        "ok": True,
        "paper_only": True,
        "synced_at": dt.datetime.now().isoformat(timespec="seconds"),
        "account_number": acc.get("account_number"),
        "status": acc.get("status"),
        "initial_equity": INITIAL_PAPER_EQUITY,
        "equity": round(equity, 2),
        "last_equity": round(last_equity, 2),
        "cash": round(cash, 2),
        "invested_value": round(sum(x["market_value"] for x in pos_list), 2),
        "daily_pl": round(daily_pl, 2),
        "daily_plpc": round(daily_plpc, 6),
        "total_return_usd": round(total_return_usd, 2),
        "total_return_pct": round(total_return_pct, 6),
        "positions": pos_list,
    }


def get_alpaca_open_orders() -> list[dict[str, Any]]:
    code, rows = _alpaca_request("GET", "/v2/orders?status=open&limit=50")
    if code != 200 or not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for o in rows:
        out.append({
            "id": str(o.get("id") or ""),
            "symbol": str(o.get("symbol") or "").upper(),
            "side": str(o.get("side") or "").upper(),
            "notional": float(o["notional"]) if o.get("notional") not in (None, "") else None,
            "qty": float(o["qty"]) if o.get("qty") not in (None, "") else None,
            "status": str(o.get("status") or ""),
            "submitted_at": str(o.get("submitted_at") or ""),
        })
    return out


def save_alpaca_account_snapshot(
    acct: dict[str, Any],
    open_orders: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Persist latest Alpaca Paper account snapshot & update daily equity curve CSV."""
    snap = dict(acct)
    snap["open_orders"] = open_orders if open_orders is not None else []
    ALPACA_SNAPSHOT_FILE.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")

    today_str = dt.date.today().isoformat()
    row = {
        "date": today_str,
        "synced_at": snap.get("synced_at", dt.datetime.now().isoformat(timespec="seconds")),
        "equity": snap.get("equity", INITIAL_PAPER_EQUITY),
        "cash": snap.get("cash", INITIAL_PAPER_EQUITY),
        "invested_value": snap.get("invested_value", 0.0),
        "total_return_pct": snap.get("total_return_pct", 0.0),
        "daily_pl": snap.get("daily_pl", 0.0),
        "positions_count": len(snap.get("positions", [])),
    }
    if ALPACA_EQUITY_CURVE_FILE.exists():
        df = pd.read_csv(ALPACA_EQUITY_CURVE_FILE, encoding="utf-8-sig")
        df = df[df["date"].astype(str) != today_str]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.sort_values("date").to_csv(ALPACA_EQUITY_CURVE_FILE, index=False, encoding="utf-8-sig")
    return snap


def load_alpaca_portfolio_view(try_live: bool = False) -> dict[str, Any]:
    """Unified dashboard read model: live Alpaca Paper API (if keys exist & requested) or latest cloud snapshot + ledger."""
    live_error: str | None = None
    source = "cloud_snapshot"
    snapshot: dict[str, Any] | None = None

    if try_live:
        try:
            acct = get_alpaca_paper_account()
            if acct.get("ok"):
                open_orders = get_alpaca_open_orders()
                snapshot = save_alpaca_account_snapshot(acct, open_orders)
                source = "live_alpaca_paper"
            else:
                live_error = str(acct.get("error") or "Alpaca API error")
        except Exception as exc:  # noqa: BLE001
            live_error = str(exc)

    if snapshot is None and ALPACA_SNAPSHOT_FILE.exists():
        try:
            snapshot = json.loads(ALPACA_SNAPSHOT_FILE.read_text(encoding="utf-8-sig"))
        except Exception:
            snapshot = None

    equity_curve: list[dict[str, Any]] = []
    if ALPACA_EQUITY_CURVE_FILE.exists():
        try:
            equity_curve = pd.read_csv(ALPACA_EQUITY_CURVE_FILE, encoding="utf-8-sig").to_dict(orient="records")
        except Exception:
            equity_curve = []

    orders: list[dict[str, Any]] = []
    if ALPACA_ORDERS_LOG.exists():
        try:
            orders = pd.read_csv(ALPACA_ORDERS_LOG, encoding="utf-8-sig").tail(60).iloc[::-1].to_dict(orient="records")
        except Exception:
            orders = []

    if snapshot is None:
        source = "orders_ledger_only"

    latest_alloc: dict[str, Any] | None = None
    try:
        latest_alloc = load_screener_allocation()
    except Exception:
        latest_alloc = None

    return {
        "ok": True,
        "paper_only": True,
        "source": source,
        "live_error": live_error,
        "initial_equity": INITIAL_PAPER_EQUITY,
        "snapshot": snapshot,
        "equity_curve": equity_curve,
        "orders": orders,
        "latest_allocation": latest_alloc,
    }


def load_screener_allocation(path: Path = SCREENER_ALLOC_FILE) -> dict[str, Any]:
    """Load latest offense+defense dynamic allocation produced by the US Stock Screener."""
    if not path.exists():
        raise FileNotFoundError(f"Missing screener allocation file: {path}")
    alloc = json.loads(path.read_text(encoding="utf-8-sig"))
    raw_tw = alloc.get("target_weights") or {}
    clean_tw: dict[str, float] = {}
    for sym, w in raw_tw.items():
        s = str(sym).strip().upper()
        wf = float(w or 0.0)
        if not _US_TICKER_RE.match(s) or wf <= 0.0:
            continue
        # Hard single-stock safety cap 40%
        clean_tw[s] = min(wf, 0.40)
    total_w = sum(clean_tw.values())
    if total_w > 1.0:
        clean_tw = {s: round(w / total_w, 4) for s, w in clean_tw.items()}
    alloc["target_weights"] = clean_tw
    return alloc


def _load_risk_state() -> dict[str, Any]:
    if not ALPACA_RISK_STATE_FILE.exists():
        return {"peak_equity": INITIAL_PAPER_EQUITY, "trailing_stops": {}}
    try:
        data = json.loads(ALPACA_RISK_STATE_FILE.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            data.setdefault("peak_equity", INITIAL_PAPER_EQUITY)
            data.setdefault("trailing_stops", {})
            return data
    except Exception:
        pass
    return {"peak_equity": INITIAL_PAPER_EQUITY, "trailing_stops": {}}


def _save_risk_state(state: dict[str, Any]) -> None:
    try:
        ALPACA_RISK_STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        log.exception("Failed to save alpaca_risk_state.json")


def apply_execution_risk_guard(
    target_weights: dict[str, float],
    pos_map: dict[str, dict[str, Any]],
    equity: float,
    alloc: dict[str, Any],
    risk_state: dict[str, Any] | None = None,
    *,
    dd_start: float = 0.15,
    dd_full: float = 0.25,
    dd_min_scale: float = 0.40,
    fallback_stop_pct: float = 0.08,
) -> tuple[dict[str, float], dict[str, Any], dict[str, Any]]:
    """Execution-layer risk protection (independent of Screener brain):
    1. Account-level smooth linear drawdown brake (-15% start -> -25% full 0.40x scale, auto-unlocks on recovery).
    2. Position-level ratchet-up Trailing ATR Stop (stop only moves UP as price makes new highs; instant exit on breach).
    """
    state = dict(risk_state) if risk_state is not None else _load_risk_state()
    peak_eq = max(float(state.get("peak_equity") or INITIAL_PAPER_EQUITY), float(equity), INITIAL_PAPER_EQUITY)
    state["peak_equity"] = round(peak_eq, 2)
    cur_dd = max(0.0, 1.0 - (equity / peak_eq)) if peak_eq > 0 else 0.0

    # 1) Smooth linear drawdown brake
    if cur_dd <= dd_start:
        brake_scale = 1.0
        brake_status = "NORMAL"
    else:
        frac = min(1.0, (cur_dd - dd_start) / max(1e-6, dd_full - dd_start))
        brake_scale = round(1.0 - frac * (1.0 - dd_min_scale), 4)
        brake_status = "FULL_BRAKE" if frac >= 1.0 else "SOFT_BRAKE"

    # Build reference stop distance from screener allocation (2.5x ATR)
    alloc_stop_map: dict[str, float] = {}
    for p in alloc.get("positions") or []:
        sym = str(p.get("symbol") or "").upper()
        sp = p.get("stop_price")
        if sym and sp is not None and float(sp) > 0:
            alloc_stop_map[sym] = float(sp)

    # 2) Ratchet-up trailing stop per currently held position
    prev_stops: dict[str, Any] = dict(state.get("trailing_stops") or {})
    new_stops: dict[str, Any] = {}
    stopped_symbols: list[dict[str, Any]] = []
    adjusted_tw = {k: float(v) for k, v in target_weights.items()}

    for sym, pos in pos_map.items():
        cur_px = float(pos.get("current_price") or 0.0)
        entry_px = float(pos.get("avg_entry_price") or cur_px)
        if cur_px <= 0:
            continue
        rec = prev_stops.get(sym) or {}
        high_px = max(float(rec.get("high_price") or entry_px), entry_px, cur_px)
        prev_stop = float(rec.get("stop_price") or 0.0)

        # Estimate stop distance: prefer screener's 2.5x ATR distance, fallback to 8%
        ref_stop = alloc_stop_map.get(sym)
        if ref_stop and cur_px > ref_stop:
            atr_dist = min(cur_px * 0.18, max(cur_px * 0.03, cur_px - ref_stop))
        else:
            atr_dist = high_px * fallback_stop_pct

        # Ratchet UP only — never lower a trailing stop
        cand_stop = max(prev_stop, high_px - atr_dist)

        if cur_px < cand_stop:
            stopped_symbols.append({
                "symbol": sym,
                "current_price": round(cur_px, 2),
                "trailing_stop": round(cand_stop, 2),
                "high_price": round(high_px, 2),
            })
            adjusted_tw.pop(sym, None)
        else:
            new_stops[sym] = {
                "high_price": round(high_px, 3),
                "stop_price": round(cand_stop, 3),
                "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
            }

    state["trailing_stops"] = new_stops

    # Apply drawdown brake scaling to remaining target weights
    if brake_scale < 1.0:
        adjusted_tw = {s: round(w * brake_scale, 4) for s, w in adjusted_tw.items() if w * brake_scale > 0.005}

    guard_summary = {
        "peak_equity": round(peak_eq, 2),
        "current_drawdown_pct": round(cur_dd * 100.0, 2),
        "brake_status": brake_status,
        "brake_scale": brake_scale,
        "dd_start_pct": round(dd_start * 100.0, 1),
        "dd_full_pct": round(dd_full * 100.0, 1),
        "stopped_symbols": stopped_symbols,
        "active_trailing_stops": new_stops,
    }
    return adjusted_tw, guard_summary, state


def sync_screener_alpaca_paper(
    alloc: dict[str, Any] | None = None,
    *,
    rebalance_tol: float = 0.02,
    min_order_usd: float = 15.0,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Sync Alpaca Paper Trading (`paper-api.alpaca.markets` ONLY) directly to the Screener's `latest_allocation.json`.

    - Exits any currently held stock that is no longer in `target_weights` (full liquidation first).
    - Trims overweight positions, then buys underweight / newly screened positions with cash only (zero leverage).
    - Cancels stale pending orders before submitting fresh rebalance orders.
    - Persists post-sync account snapshot & equity curve for dashboard visibility.
    """
    if alloc is None:
        alloc = load_screener_allocation()
    target_weights: dict[str, float] = dict(alloc.get("target_weights") or {})
    ver = str(alloc.get("version") or "screener-v2.0.0")

    acct = get_alpaca_paper_account()
    if not acct.get("ok"):
        return {"ok": False, "paper_only": True, "error": acct.get("error")}

    equity = float(acct["equity"])
    cash = float(acct["cash"])
    pos_map = {
        str(p["symbol"]).upper(): p
        for p in acct["positions"]
        if _US_TICKER_RE.match(str(p.get("symbol", "")).upper())
    }

    # Apply execution-layer risk guard (Mid_-15/-25 smooth drawdown brake + ratchet-up Trailing ATR Stop)
    target_weights, risk_guard, new_risk_state = apply_execution_risk_guard(
        target_weights, pos_map, equity, alloc
    )
    if not dry_run:
        _save_risk_state(new_risk_state)

    all_syms = sorted(set(pos_map.keys()) | set(target_weights.keys()))
    cur_weights = {s: (pos_map[s]["market_value"] / equity if (s in pos_map and equity > 0) else 0.0) for s in all_syms}

    if not dry_run:
        _alpaca_request("DELETE", "/v2/orders")

    actions_planned: list[dict[str, Any]] = []
    orders_executed: list[dict[str, Any]] = []

    # 1) SELLS & FULL EXITS first
    for sym in all_syms:
        tw = float(target_weights.get(sym, 0.0))
        cw = float(cur_weights.get(sym, 0.0))
        cur_val = pos_map[sym]["market_value"] if sym in pos_map else 0.0
        if cur_val < min_order_usd:
            continue
        is_full_exit = (tw == 0.0)
        is_trim = (cw - tw >= rebalance_tol and (cw - tw) * equity >= min_order_usd)
        if is_full_exit or is_trim:
            sell_usd = cur_val if is_full_exit else (cw - tw) * equity
            actions_planned.append({
                "symbol": sym,
                "side": "sell",
                "prev_w": round(cw, 4),
                "target_w": round(tw, 4),
                "est_usd": round(sell_usd, 2),
                "full_exit": is_full_exit,
            })
            if not dry_run:
                if is_full_exit:
                    code, resp = _alpaca_request("DELETE", f"/v2/positions/{sym}")
                else:
                    code, resp = _alpaca_request("POST", "/v2/orders", {
                        "symbol": sym,
                        "notional": f"{sell_usd:.2f}",
                        "side": "sell",
                        "type": "market",
                        "time_in_force": "day",
                    })
                ok = code in (200, 201)
                ord_id = str(resp.get("id", "")) if isinstance(resp, dict) else ""
                status_txt = str(resp.get("status", resp.get("message", ""))) if isinstance(resp, dict) else str(resp)
                if ok:
                    cash += sell_usd * 0.998
                orders_executed.append({
                    "recorded_at": dt.datetime.now().isoformat(timespec="seconds"),
                    "paper_only": True,
                    "version": ver,
                    "symbol": sym,
                    "side": "SELL",
                    "prev_weight": round(cw, 4),
                    "target_weight": round(tw, 4),
                    "notional_usd": round(sell_usd, 2),
                    "ok": ok,
                    "ord_id": ord_id,
                    "status": status_txt,
                })

    # 2) BUYS second (strictly cash only, zero leverage; larger target weights first)
    buy_candidates = sorted(target_weights.keys(), key=lambda s: target_weights[s], reverse=True)
    for sym in buy_candidates:
        tw = float(target_weights[sym])
        cw = float(cur_weights.get(sym, 0.0))
        if tw - cw >= rebalance_tol:
            want_usd = (tw - cw) * equity
            spend_usd = min(want_usd, max(0.0, cash * 0.99))
            if spend_usd >= min_order_usd:
                actions_planned.append({
                    "symbol": sym,
                    "side": "buy",
                    "prev_w": round(cw, 4),
                    "target_w": round(tw, 4),
                    "est_usd": round(spend_usd, 2),
                    "full_exit": False,
                })
                if not dry_run:
                    code, resp = _alpaca_request("POST", "/v2/orders", {
                        "symbol": sym,
                        "notional": f"{spend_usd:.2f}",
                        "side": "buy",
                        "type": "market",
                        "time_in_force": "day",
                    })
                    ok = code in (200, 201)
                    ord_id = str(resp.get("id", "")) if isinstance(resp, dict) else ""
                    status_txt = str(resp.get("status", resp.get("message", ""))) if isinstance(resp, dict) else str(resp)
                    if ok:
                        cash = max(0.0, cash - spend_usd)
                    orders_executed.append({
                        "recorded_at": dt.datetime.now().isoformat(timespec="seconds"),
                        "paper_only": True,
                        "version": ver,
                        "symbol": sym,
                        "side": "BUY",
                        "prev_weight": round(cw, 4),
                        "target_weight": round(tw, 4),
                        "notional_usd": round(spend_usd, 2),
                        "ok": ok,
                        "ord_id": ord_id,
                        "status": status_txt,
                    })

    if orders_executed:
        df_ord = pd.read_csv(ALPACA_ORDERS_LOG, encoding="utf-8-sig") if ALPACA_ORDERS_LOG.exists() else pd.DataFrame()
        pd.concat([df_ord, pd.DataFrame(orders_executed)], ignore_index=True).to_csv(
            ALPACA_ORDERS_LOG, index=False, encoding="utf-8-sig"
        )

    # 3) Always refresh & persist latest account snapshot + equity curve for local dashboard
    try:
        latest_acct = get_alpaca_paper_account()
        if latest_acct.get("ok"):
            save_alpaca_account_snapshot(latest_acct, get_alpaca_open_orders())
    except Exception:  # noqa: BLE001
        log.exception("Failed to persist Alpaca account snapshot")

    all_ok = all(o["ok"] for o in orders_executed) if orders_executed else True
    return {
        "ok": all_ok,
        "paper_only": True,
        "source": "screener_latest_allocation",
        "regime": alloc.get("regime"),
        "signal_date": alloc.get("signal_date"),
        "dry_run": dry_run,
        "account_number": acct.get("account_number"),
        "equity_usd": round(equity, 2),
        "cash_usd": round(cash, 2),
        "cur_weights": {k: round(v, 4) for k, v in cur_weights.items()},
        "target_weights": {k: round(v, 4) for k, v in target_weights.items()},
        "risk_guard": risk_guard,
        "actions_planned": actions_planned,
        "orders_executed": orders_executed,
    }
