"""6-Asset Cross-Sector High-Elasticity Top-2 Trend Rotation Engine.

HARD SAFETY INVARIANTS:
1. Completely isolated from btc_quant (NEVER touches BTC, BTC-USDT-SWAP, or data/btc/).
2. Strictly whitelisted to the 6 US Stock Token SPOT pairs (`tdMode="cash"`, zero leverage):
   XMU-USDT, XAVGO-USDT, XPLTR-USDT, XHOOD-USDT, XRKLB-USDT, XQQQ-USDT.
3. Hardcoded OKX Simulated Trading header (`x-simulated-trading: 1`) via okx_demo_channel._demo_private_request.
4. Sleeve equity is computed ONLY from free USDT + the 6 whitelisted stock tokens, so BTC margin is never disturbed.
5. Append-only forward logs under data/rotation/.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mu_quant.config import CONFIGS_DIR, DATA_RAW, ROOT
from mu_quant.execution_simulator.okx_demo_channel import (
    ALLOWED_TD_MODE,
    _demo_private_request,
    _http_request_json,
)
from mu_quant.logs.logger import get_logger

log = get_logger("mu_quant.rotation")

ROTATION_CFG_PATH = CONFIGS_DIR / "rotation_v1.0.0.json"
ROTATION_DIR = ROOT / "data" / "rotation"
ROTATION_RAW_DIR = ROTATION_DIR / "raw"
ROTATION_SIGNALS_LOG = ROTATION_DIR / "signals_log.csv"
ROTATION_ORDERS_LOG = ROTATION_DIR / "demo_orders_ledger.csv"
ROTATION_FORWARD_NAV = ROTATION_DIR / "forward_nav.csv"
ROTATION_STATE_FILE = ROTATION_DIR / "forward_state.json"

for _d in (ROTATION_DIR, ROTATION_RAW_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def load_rotation_config() -> dict[str, Any]:
    cfg = json.loads(ROTATION_CFG_PATH.read_text(encoding="utf-8-sig"))
    cfg.setdefault("execution", {})["allow_live_auto_order"] = False
    cfg["execution"]["simulated_only"] = True
    cfg["execution"]["td_mode"] = ALLOWED_TD_MODE
    return cfg


def _fetch_eastmoney_kline(secid: str, beg: str = "20200101") -> pd.DataFrame:
    url = (
        f"https://push2his.eastmoney.com/api/qt/stock/kline/get?"
        f"secid={secid}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56"
        f"&klt=101&fqt=1&beg={beg}&end=20500101"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read().decode("utf-8"))
    klines = (data.get("data") or {}).get("klines") or []
    rows = []
    for line in klines:
        d, o, c, h, l, v = line.split(",")
        rows.append({
            "Date": d,
            "Open": float(o),
            "High": float(h),
            "Low": float(l),
            "Close": float(c),
            "Volume": float(v),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").sort_index()


def load_or_refresh_asset(item: dict[str, Any], refresh: bool = True) -> pd.DataFrame:
    sym = item["symbol"]
    cache_path = ROTATION_RAW_DIR / f"{sym}.csv"
    # Bootstrap from existing MU / QQQ raw files if cache absent
    if not cache_path.exists():
        if sym == "MU" and (DATA_RAW / "MU.csv").exists():
            df0 = pd.read_csv(DATA_RAW / "MU.csv", parse_dates=["Date"]).set_index("Date").sort_index()
            df0[["Open", "High", "Low", "Close", "Volume"]].to_csv(cache_path)
        elif sym == "QQQ" and (DATA_RAW / "bench_QQQ.csv").exists():
            df0 = pd.read_csv(DATA_RAW / "bench_QQQ.csv", parse_dates=["Date"]).set_index("Date").sort_index()
            df0[["Open", "High", "Low", "Close", "Volume"]].to_csv(cache_path)

    existing = (
        pd.read_csv(cache_path, parse_dates=["Date"]).set_index("Date").sort_index()
        if cache_path.exists()
        else pd.DataFrame()
    )

    if refresh:
        try:
            fresh = _fetch_eastmoney_kline(item["eastmoney_secid"], beg="20200101")
            if not fresh.empty and len(fresh) >= 220:
                if existing.empty or len(fresh) >= len(existing) * 0.8:
                    fresh.to_csv(cache_path)
                    existing = fresh
                else:
                    # Safe incremental append without shrinking history
                    combined = pd.concat([existing, fresh])
                    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
                    combined.to_csv(cache_path)
                    existing = combined
        except Exception as e:
            log.warning("Refresh failed for %s (%s); using cached data (%d rows)", sym, e, len(existing))

    if existing.empty:
        raise RuntimeError(f"No historical data available for rotation asset {sym}")
    for col in ("Open", "High", "Low", "Close", "Volume"):
        existing[col] = pd.to_numeric(existing[col], errors="coerce")
    return existing.dropna(subset=["Open", "High", "Low", "Close"])


def compute_rotation_signal(cfg: dict[str, Any], refresh: bool = True) -> dict[str, Any]:
    """Strictly causal T-day close Top-2 rotation scoring and target weights."""
    rules = cfg["rules"]
    ma_n = int(rules["trend_ma_days"])
    mom_n = int(rules["momentum_days"])
    top_k = int(rules["top_k"])
    max_w = float(rules["max_weight_per_asset"])
    target_dvol = float(rules["target_downside_vol"])
    dvol_n = int(rules["dvol_window"])

    scores = []
    latest_dates = []
    for item in cfg["universe"]:
        sym = item["symbol"]
        df = load_or_refresh_asset(item, refresh=refresh)
        close = df["Close"]
        ret = close.pct_change()
        sma200 = close.rolling(ma_n, min_periods=ma_n).mean()
        mom60 = close / close.shift(mom_n) - 1.0
        neg_sq = np.where(ret < 0.0, ret ** 2, 0.0)
        dvol21 = np.sqrt(pd.Series(neg_sq, index=df.index).rolling(dvol_n, min_periods=dvol_n).mean() * 252.0)

        last_dt = df.index[-1]
        latest_dates.append(last_dt)
        c_val = float(close.iloc[-1])
        ma_val = float(sma200.iloc[-1]) if pd.notna(sma200.iloc[-1]) else float("nan")
        m_val = float(mom60.iloc[-1]) if pd.notna(mom60.iloc[-1]) else float("nan")
        dv_val = float(dvol21.iloc[-1]) if pd.notna(dvol21.iloc[-1]) else float("nan")

        above200 = bool(pd.notna(ma_val) and c_val > ma_val)
        dvol_cap = float(np.clip(target_dvol / max(dv_val, 0.10), 0.0, 1.0)) if pd.notna(dv_val) else 0.0

        scores.append({
            "symbol": sym,
            "okx_inst_id": item["okx_inst_id"],
            "theme": item["theme"],
            "date": str(last_dt.date()),
            "open": round(float(df["Open"].iloc[-1]), 4),
            "close": round(c_val, 4),
            "sma200": round(ma_val, 4) if pd.notna(ma_val) else None,
            "above200": above200,
            "mom60": round(m_val, 6) if pd.notna(m_val) else None,
            "dvol21": round(dv_val, 6) if pd.notna(dv_val) else None,
            "dvol_cap": round(dvol_cap, 4),
            "target_weight": 0.0,
        })

    # Rank eligible (above MA200) by 60d momentum descending
    eligible = [s for s in scores if s["above200"] and s["mom60"] is not None]
    eligible.sort(key=lambda x: x["mom60"], reverse=True)
    selected = eligible[:top_k]

    total_w = 0.0
    for s in selected:
        w = min(max_w, max_w * s["dvol_cap"])
        s["target_weight"] = round(w, 4)
        total_w += s["target_weight"]

    # Hard zero-leverage invariant
    if total_w > 1.0:
        for s in selected:
            s["target_weight"] = round(s["target_weight"] / total_w, 4)
        total_w = sum(s["target_weight"] for s in selected)

    scores.sort(key=lambda x: (x["above200"], x["mom60"] if x["mom60"] is not None else -999.0), reverse=True)
    signal_date = str(max(latest_dates).date())
    target_weights = {s["symbol"]: s["target_weight"] for s in scores}

    return {
        "version": cfg["version"],
        "signal_date": signal_date,
        "computed_at": dt.datetime.now().isoformat(timespec="seconds"),
        "scores": scores,
        "selected_symbols": [s["symbol"] for s in selected],
        "target_weights": target_weights,
        "total_invested_weight": round(total_w, 4),
        "cash_weight": round(max(0.0, 1.0 - total_w), 4),
    }


def record_rotation_signal(sig: dict[str, Any]) -> dict[str, Any]:
    """Append-only signal log (never overwrites)."""
    rec = {
        "recorded_at": sig["computed_at"],
        "signal_date": sig["signal_date"],
        "version": sig["version"],
        "selected": ",".join(sig["selected_symbols"]) if sig["selected_symbols"] else "CASH",
        "total_weight": sig["total_invested_weight"],
        "cash_weight": sig["cash_weight"],
        "weights_json": json.dumps(sig["target_weights"], ensure_ascii=False),
    }
    df = pd.read_csv(ROTATION_SIGNALS_LOG, encoding="utf-8-sig") if ROTATION_SIGNALS_LOG.exists() else pd.DataFrame()
    pd.concat([df, pd.DataFrame([rec])], ignore_index=True).to_csv(ROTATION_SIGNALS_LOG, index=False, encoding="utf-8-sig")
    return rec


def update_rotation_forward_scorecard(sig: dict[str, Any], initial_capital: float = 100_000.0) -> dict[str, Any]:
    """Standalone forward paper scorecard for 6-Asset Rotation (strict T+1 Open fill, 15bp/side)."""
    mkt_date = sig["signal_date"]
    opens = {s["symbol"]: float(s["open"]) for s in sig["scores"]}
    closes = {s["symbol"]: float(s["close"]) for s in sig["scores"]}
    tw = sig["target_weights"]
    cost_rate = 0.0015

    if not ROTATION_STATE_FILE.exists():
        state = {
            "start_date": mkt_date,
            "last_date": mkt_date,
            "initial_capital": initial_capital,
            "cash": initial_capital,
            "shares": {sym: 0.0 for sym in closes},
            "pending_weights": tw,
            "nav": initial_capital,
            "peak_nav": initial_capital,
            "days": 1,
        }
        ROTATION_STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        row = {
            "date": mkt_date,
            "nav": round(initial_capital, 2),
            "cum_ret_pct": 0.0,
            "dd_pct": 0.0,
            "invested_pct": 0.0,
            "selected": ",".join(sig["selected_symbols"]) if sig["selected_symbols"] else "CASH",
        }
        pd.DataFrame([row]).to_csv(ROTATION_FORWARD_NAV, index=False, encoding="utf-8-sig")
        return state

    state = json.loads(ROTATION_STATE_FILE.read_text(encoding="utf-8-sig"))
    if mkt_date <= state["last_date"]:
        # Same trading day: refresh pending target weights only
        state["pending_weights"] = tw
        ROTATION_STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        return state

    # New trading day: execute previous day's pending_weights at today's Open
    cash = float(state["cash"])
    shares = {sym: float(state["shares"].get(sym, 0.0)) for sym in closes}
    pw = state.get("pending_weights", {})
    eq_open = cash + sum(shares[s] * opens[s] for s in opens)

    # Sells first
    for s in opens:
        tgt_w = float(pw.get(s, 0.0))
        cur_val = shares[s] * opens[s]
        tgt_val = eq_open * tgt_w
        if tgt_val < cur_val - 1.0:
            sell_sh = min(shares[s], (cur_val - tgt_val) / opens[s])
            shares[s] -= sell_sh
            cash += sell_sh * opens[s] * (1.0 - cost_rate)
    # Buys second (strict zero leverage)
    for s in opens:
        tgt_w = float(pw.get(s, 0.0))
        cur_val = shares[s] * opens[s]
        tgt_val = eq_open * tgt_w
        if tgt_val > cur_val + 1.0:
            want = tgt_val - cur_val
            spend = min(want, cash / (1.0 + cost_rate))
            if spend > 1.0:
                shares[s] += spend / opens[s]
                cash -= spend * (1.0 + cost_rate)

    nav = cash + sum(shares[s] * closes[s] for s in closes)
    peak = max(float(state.get("peak_nav", nav)), nav)
    dd_pct = (nav / peak - 1.0) * 100.0 if peak > 0 else 0.0
    cum_ret = (nav / float(state["initial_capital"]) - 1.0) * 100.0
    inv_pct = (sum(shares[s] * closes[s] for s in closes) / nav * 100.0) if nav > 0 else 0.0

    state.update({
        "last_date": mkt_date,
        "cash": round(cash, 4),
        "shares": {k: round(v, 6) for k, v in shares.items()},
        "pending_weights": tw,
        "nav": round(nav, 2),
        "peak_nav": round(peak, 2),
        "days": int(state.get("days", 1)) + 1,
    })
    ROTATION_STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    row = {
        "date": mkt_date,
        "nav": round(nav, 2),
        "cum_ret_pct": round(cum_ret, 3),
        "dd_pct": round(dd_pct, 3),
        "invested_pct": round(inv_pct, 1),
        "selected": ",".join(sig["selected_symbols"]) if sig["selected_symbols"] else "CASH",
    }
    df_nav = pd.read_csv(ROTATION_FORWARD_NAV, encoding="utf-8-sig") if ROTATION_FORWARD_NAV.exists() else pd.DataFrame()
    pd.concat([df_nav, pd.DataFrame([row])], ignore_index=True).to_csv(ROTATION_FORWARD_NAV, index=False, encoding="utf-8-sig")
    return state


def sync_rotation_okx_demo(sig: dict[str, Any], cfg: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    """Sync the 6 whitelisted X*-USDT spot positions on OKX Simulated Trading (`x-simulated-trading: 1`).

    STRICT ISOLATION:
    - ONLY touches the 6 whitelisted stock tokens in cfg['universe'].
    - NEVER touches BTC, BTC-USDT-SWAP, or any non-whitelisted asset.
    - Sells exited stock tokens first to free USDT, then buys newly selected Top-2 tokens with cash only.
    """
    rules = cfg["rules"]
    tol = float(rules.get("rebalance_tolerance", 0.05))
    min_order_usdt = float(rules.get("min_order_usdt", 15.0))
    universe = cfg["universe"]
    allowed_insts = {u["okx_inst_id"] for u in universe}
    sym_to_inst = {u["symbol"]: u["okx_inst_id"] for u in universe}
    sym_to_ccy = {u["symbol"]: u["okx_inst_id"].split("-")[0] for u in universe}

    # 1) Read OKX Demo balances (HARDCODED x-simulated-trading: 1)
    bal_resp = _demo_private_request("GET", "/api/v5/account/balance")
    if bal_resp.get("code") != "0" or not bal_resp.get("data"):
        return {"ok": False, "error": f"Balance query failed: {bal_resp}"}

    details = bal_resp["data"][0].get("details", [])
    avail_usdt = 0.0
    held_qty = {u["symbol"]: 0.0 for u in universe}
    ccy_to_sym = {v: k for k, v in sym_to_ccy.items()}
    for d in details:
        ccy = d.get("ccy", "")
        if ccy == "USDT":
            avail_usdt = float(d.get("availBal") or d.get("cashBal") or 0.0)
        elif ccy in ccy_to_sym:
            held_qty[ccy_to_sym[ccy]] = float(d.get("availBal") or d.get("eq") or 0.0)

    # 2) Fetch live OKX spot prices for all 6 whitelisted tokens
    prices: dict[str, float] = {}
    for sym, inst_id in sym_to_inst.items():
        tk = _http_request_json("GET", f"/api/v5/market/ticker?instId={inst_id}")
        if tk.get("code") == "0" and tk.get("data"):
            prices[sym] = float(tk["data"][0]["last"])
        else:
            prices[sym] = 0.0

    # Sleeve equity = free USDT + value of ONLY the 6 whitelisted stock tokens (isolates BTC margin!)
    pos_vals = {sym: held_qty[sym] * prices[sym] for sym in held_qty}
    sleeve_eq = avail_usdt + sum(pos_vals.values())
    cur_weights = {sym: (pos_vals[sym] / sleeve_eq if sleeve_eq > 0 else 0.0) for sym in held_qty}
    target_weights = sig["target_weights"]

    orders_executed: list[dict[str, Any]] = []
    actions_planned: list[dict[str, Any]] = []

    # 3) Plan SELLS first (including full exit when target_weight == 0)
    for sym in [u["symbol"] for u in universe]:
        inst_id = sym_to_inst[sym]
        assert inst_id in allowed_insts, f"Safety block: {inst_id} not in whitelist"
        tw = float(target_weights.get(sym, 0.0))
        cw = float(cur_weights.get(sym, 0.0))
        px = prices[sym]
        if px <= 0 or held_qty[sym] <= 0:
            continue
        is_full_exit = (tw == 0.0 and pos_vals[sym] >= min_order_usdt)
        is_trim = (cw - tw >= tol and (cw - tw) * sleeve_eq >= min_order_usdt)
        if is_full_exit or is_trim:
            sell_qty = held_qty[sym] if is_full_exit else min(held_qty[sym], ((cw - tw) * sleeve_eq) / px)
            sell_qty_str = f"{sell_qty:.6f}".rstrip("0").rstrip(".")
            act = {
                "symbol": sym,
                "inst_id": inst_id,
                "side": "sell",
                "prev_w": round(cw, 4),
                "target_w": round(tw, 4),
                "sz": sell_qty_str,
                "est_usdt": round(sell_qty * px, 2),
                "ref_px": px,
            }
            actions_planned.append(act)
            if not dry_run:
                payload = {
                    "instId": inst_id,
                    "tdMode": ALLOWED_TD_MODE,  # cash spot only
                    "side": "sell",
                    "ordType": "market",
                    "sz": sell_qty_str,
                }
                resp = _demo_private_request("POST", "/api/v5/trade/order", payload)
                ok = resp.get("code") == "0"
                ord_id = resp["data"][0].get("ordId", "") if ok and resp.get("data") else ""
                msg = resp["data"][0].get("sMsg", "Order placed") if ok and resp.get("data") else resp.get("msg", str(resp))
                if ok:
                    avail_usdt += sell_qty * px * 0.998
                rec = {
                    "recorded_at": dt.datetime.now().isoformat(timespec="seconds"),
                    "simulated_only": True,
                    "version": cfg["version"],
                    "inst_id": inst_id,
                    "side": "SELL",
                    "prev_weight": round(cw, 4),
                    "target_weight": round(tw, 4),
                    "order_sz": sell_qty_str,
                    "est_usdt": round(sell_qty * px, 2),
                    "ref_px": px,
                    "ok": ok,
                    "ord_id": ord_id,
                    "msg": msg,
                }
                orders_executed.append(rec)

    # 4) Plan BUYS second (strict cash only, zero leverage)
    for sym in [u["symbol"] for u in universe]:
        inst_id = sym_to_inst[sym]
        assert inst_id in allowed_insts, f"Safety block: {inst_id} not in whitelist"
        tw = float(target_weights.get(sym, 0.0))
        cw = float(cur_weights.get(sym, 0.0))
        px = prices[sym]
        if px <= 0 or tw <= 0:
            continue
        if tw - cw >= tol:
            want_usdt = (tw - cw) * sleeve_eq
            spend_usdt = min(want_usdt, max(0.0, avail_usdt * 0.99))
            if spend_usdt >= min_order_usdt:
                sz_str = f"{spend_usdt:.2f}"
                act = {
                    "symbol": sym,
                    "inst_id": inst_id,
                    "side": "buy",
                    "prev_w": round(cw, 4),
                    "target_w": round(tw, 4),
                    "sz": sz_str,
                    "est_usdt": round(spend_usdt, 2),
                    "ref_px": px,
                }
                actions_planned.append(act)
                if not dry_run:
                    payload = {
                        "instId": inst_id,
                        "tdMode": ALLOWED_TD_MODE,  # cash spot only
                        "side": "buy",
                        "ordType": "market",
                        "sz": sz_str,
                        "tgtCcy": "quote_ccy",
                    }
                    resp = _demo_private_request("POST", "/api/v5/trade/order", payload)
                    ok = resp.get("code") == "0"
                    ord_id = resp["data"][0].get("ordId", "") if ok and resp.get("data") else ""
                    msg = resp["data"][0].get("sMsg", "Order placed") if ok and resp.get("data") else resp.get("msg", str(resp))
                    if ok:
                        avail_usdt = max(0.0, avail_usdt - spend_usdt)
                    rec = {
                        "recorded_at": dt.datetime.now().isoformat(timespec="seconds"),
                        "simulated_only": True,
                        "version": cfg["version"],
                        "inst_id": inst_id,
                        "side": "BUY",
                        "prev_weight": round(cw, 4),
                        "target_weight": round(tw, 4),
                        "order_sz": sz_str,
                        "est_usdt": round(spend_usdt, 2),
                        "ref_px": px,
                        "ok": ok,
                        "ord_id": ord_id,
                        "msg": msg,
                    }
                    orders_executed.append(rec)

    if orders_executed:
        df_ord = pd.read_csv(ROTATION_ORDERS_LOG, encoding="utf-8-sig") if ROTATION_ORDERS_LOG.exists() else pd.DataFrame()
        pd.concat([df_ord, pd.DataFrame(orders_executed)], ignore_index=True).to_csv(
            ROTATION_ORDERS_LOG, index=False, encoding="utf-8-sig"
        )

    all_ok = all(o["ok"] for o in orders_executed) if orders_executed else True
    return {
        "ok": all_ok,
        "simulated_only": True,
        "dry_run": dry_run,
        "sleeve_equity_usdt": round(sleeve_eq, 2),
        "avail_usdt": round(avail_usdt, 2),
        "cur_weights": {k: round(v, 4) for k, v in cur_weights.items()},
        "target_weights": target_weights,
        "actions_planned": actions_planned,
        "orders_executed": orders_executed,
    }
