"""Isolated OKX Stock Token (XMU-USDT) Channel Monitor & SIMULATED-ONLY Spot Sync.

HARD SAFETY INVARIANTS (cannot be bypassed):
1. Private API calls HARDCODE header `x-simulated-trading: 1` (demo/simulated account ONLY).
2. Requires `"simulated_only": true` in local gitignored `configs/okx_demo_secrets.local.json`.
3. Strictly SPOT `XMU-USDT` with `tdMode="cash"` ONLY (zero leverage; SWAP/FUTURES/MARGIN blocked).
4. Completely isolated from NASDAQ: MU research & backtest CSVs (writes ONLY to data/okx_channel/).
"""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import json
import os
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

log = get_logger("mu_quant.okx_demo")

OKX_BASE = "https://www.okx.com"
ALLOWED_INST_ID = "XMU-USDT"
ALLOWED_TD_MODE = "cash"  # Spot cash only (zero leverage, no margin, no swap)
CHANNEL_DIR = ROOT / "data" / "okx_channel"
CHANNEL_DIR.mkdir(parents=True, exist_ok=True)
SPREAD_LOG = CHANNEL_DIR / "xmu_spread_log.csv"
DEMO_ORDERS_LOG = CHANNEL_DIR / "demo_orders_ledger.csv"
SECRETS_FILE = CONFIGS_DIR / "okx_demo_secrets.local.json"


def _http_request_json(method: str, path: str, headers: dict[str, str] | None = None, body: str = "") -> dict[str, Any]:
    hdr = {"User-Agent": "Mozilla/5.0"}
    if headers:
        hdr.update(headers)
    data = body.encode("utf-8") if body else None
    req = urllib.request.Request(OKX_BASE + path, data=data, headers=hdr, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        return {"code": str(e.code), "msg": err_body, "data": []}


def _load_demo_secrets() -> dict[str, Any]:
    # 1) Cloud / GitHub Actions Secrets via environment variables (still hardcoded x-simulated-trading: 1)
    env_key = os.environ.get("MU_OKX_DEMO_API_KEY") or os.environ.get("OKX_DEMO_API_KEY") or ""
    env_sec = os.environ.get("MU_OKX_DEMO_SECRET_KEY") or os.environ.get("OKX_DEMO_SECRET_KEY") or ""
    env_pass = os.environ.get("MU_OKX_DEMO_PASSPHRASE") or os.environ.get("OKX_DEMO_PASSPHRASE") or ""
    if env_key.strip() and env_sec.strip() and env_pass.strip():
        return {
            "apiKey": env_key.strip(),
            "secretKey": env_sec.strip(),
            "passphrase": env_pass.strip(),
            "simulated_only": True,
            "source": "env",
        }

    # 2) Local gitignored secrets file
    if not SECRETS_FILE.exists():
        raise FileNotFoundError(f"Missing demo secrets file: {SECRETS_FILE} (and no OKX_DEMO_* env vars set)")
    sec = json.loads(SECRETS_FILE.read_text(encoding="utf-8-sig"))
    if sec.get("simulated_only") is not True:
        raise RuntimeError("Safety block: okx_demo_secrets.local.json MUST set simulated_only=true.")
    for k in ("apiKey", "secretKey", "passphrase"):
        if not sec.get(k):
            raise ValueError(f"Missing required field '{k}' in {SECRETS_FILE}")
    return sec


def _sign_okx(secret: str, ts: str, method: str, path: str, body: str = "") -> str:
    msg = f"{ts}{method.upper()}{path}{body}".encode("utf-8")
    mac = hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).digest()
    return base64.b64encode(mac).decode("utf-8")


def _demo_private_request(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Authenticated request HARDCODED to OKX Simulated Trading (`x-simulated-trading: 1`)."""
    sec = _load_demo_secrets()
    body = json.dumps(payload, separators=(",", ":")) if payload is not None else ""
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    headers = {
        "OK-ACCESS-KEY": sec["apiKey"],
        "OK-ACCESS-SIGN": _sign_okx(sec["secretKey"], ts, method, path, body),
        "OK-ACCESS-TIMESTAMP": ts,
        "OK-ACCESS-PASSPHRASE": sec["passphrase"],
        "x-simulated-trading": "1",  # HARDCODED SIMULATED TRADING ONLY
        "Content-Type": "application/json",
    }
    return _http_request_json(method, path, headers=headers, body=body)


def probe_xmu_market(mu_ref_close: float | None = None) -> dict[str, Any]:
    """Read-only public probe of XMU-USDT spot ticker, spread, top-5 depth, and tracking error vs MU."""
    tk = _http_request_json("GET", f"/api/v5/market/ticker?instId={ALLOWED_INST_ID}")["data"][0]
    bk = _http_request_json("GET", f"/api/v5/market/books?instId={ALLOWED_INST_ID}&sz=5")["data"][0]
    last_px = float(tk["last"])
    bid = float(tk["bidPx"])
    ask = float(tk["askPx"])
    mid = (bid + ask) / 2.0
    spread_bps = (ask - bid) / mid * 1e4 if mid > 0 else 0.0
    bid5_usd = sum(float(p) * float(q) for p, q, *_ in bk["bids"])
    ask5_usd = sum(float(p) * float(q) for p, q, *_ in bk["asks"])
    track_err_pct = ((mid / mu_ref_close) - 1.0) * 100.0 if (mu_ref_close and mu_ref_close > 0) else None

    rec = {
        "recorded_at": dt.datetime.now().isoformat(timespec="seconds"),
        "inst_id": ALLOWED_INST_ID,
        "last_px": last_px,
        "bid_px": bid,
        "ask_px": ask,
        "spread_bps": round(spread_bps, 2),
        "vol_24h_usdt": round(float(tk["volCcy24h"]), 2),
        "top5_bid_depth_usd": round(bid5_usd, 2),
        "top5_ask_depth_usd": round(ask5_usd, 2),
        "mu_ref_close": mu_ref_close,
        "tracking_err_pct": round(track_err_pct, 4) if track_err_pct is not None else "",
    }
    df = pd.read_csv(SPREAD_LOG, encoding="utf-8-sig") if SPREAD_LOG.exists() else pd.DataFrame()
    pd.concat([df, pd.DataFrame([rec])], ignore_index=True).to_csv(SPREAD_LOG, index=False, encoding="utf-8-sig")
    log.info("Probed %s: last=%.2f spread=%.2fbp depth5=$%.0f/$%.0f", ALLOWED_INST_ID, last_px, spread_bps, bid5_usd, ask5_usd)
    return rec


def get_demo_account_balance() -> dict[str, Any]:
    return _demo_private_request("GET", "/api/v5/account/balance")


def get_demo_position_summary() -> dict[str, Any]:
    bal = get_demo_account_balance()
    if bal.get("code") != "0" or not bal.get("data"):
        return {"ok": False, "error": bal.get("msg", "Failed to fetch demo balance"), "raw": bal}
    d0 = bal["data"][0]
    total_eq = float(d0.get("totalEq") or 0.0)
    usdt_avail = 0.0
    xmu_avail = 0.0
    xmu_eq_usd = 0.0
    for item in d0.get("details", []):
        ccy = str(item.get("ccy", "")).upper()
        if ccy == "USDT":
            usdt_avail = float(item.get("availBal") or 0.0)
        elif ccy == "XMU":
            xmu_avail = float(item.get("availBal") or 0.0)
            xmu_eq_usd = float(item.get("eqUsd") or 0.0)
    cur_weight = (xmu_eq_usd / total_eq) if total_eq > 0 else 0.0
    return {
        "ok": True,
        "simulated_only": True,
        "total_eq_usd": round(total_eq, 2),
        "usdt_avail": round(usdt_avail, 2),
        "xmu_shares": round(xmu_avail, 6),
        "xmu_eq_usd": round(xmu_eq_usd, 2),
        "current_weight": round(cur_weight, 4),
    }


def _append_demo_order_log(row: dict[str, Any]) -> None:
    df = pd.read_csv(DEMO_ORDERS_LOG, encoding="utf-8-sig") if DEMO_ORDERS_LOG.exists() else pd.DataFrame()
    pd.concat([df, pd.DataFrame([row])], ignore_index=True).to_csv(DEMO_ORDERS_LOG, index=False, encoding="utf-8-sig")


def sync_demo_spot_position(
    target_weight: float,
    *,
    strategy_version: str = "v2.0.0",
    rebalance_tol: float = 0.05,
    mu_ref_close: float | None = None,
) -> dict[str, Any]:
    """Sync OKX SIMULATED SPOT XMU-USDT position to target_weight in [0.0, 1.0].

    Strictly refuses negative weights (shorting requires leverage/swap on OKX, which is forbidden).
    """
    tw = max(0.0, min(1.0, float(target_weight)))
    mkt = probe_xmu_market(mu_ref_close=mu_ref_close)
    pos = get_demo_position_summary()
    if not pos["ok"]:
        return {"ok": False, "error": pos["error"]}

    cur_w = pos["current_weight"]
    total_eq = pos["total_eq_usd"]
    last_px = mkt["last_px"]

    if abs(tw - cur_w) < rebalance_tol and not (tw == 0.0 and pos["xmu_shares"] > 1e-4):
        return {
            "ok": True, "action": "HOLD", "simulated_only": True,
            "reason": f"目标仓位 {tw:.1%} 与当前模拟盘仓位 {cur_w:.1%} 差异小于 {rebalance_tol:.0%}，保持观望。",
            "position": pos, "market": mkt,
        }

    delta_usd = (tw - cur_w) * total_eq
    if delta_usd > 20.0:
        buy_usdt = min(delta_usd, pos["usdt_avail"] * 0.995)
        if buy_usdt < 10.0:
            return {"ok": False, "action": "SKIP", "reason": "可用模拟 USDT 不足 $10", "position": pos}
        order_payload = {
            "instId": ALLOWED_INST_ID,
            "tdMode": ALLOWED_TD_MODE,
            "side": "buy",
            "ordType": "market",
            "tgtCcy": "quote_ccy",
            "sz": f"{buy_usdt:.2f}",
        }
        action_label = "BUY"
        est_shares = buy_usdt / last_px
    elif delta_usd < -20.0 or (tw == 0.0 and pos["xmu_shares"] > 1e-4):
        sell_shares = pos["xmu_shares"] if tw == 0.0 else min(pos["xmu_shares"], abs(delta_usd) / last_px)
        if sell_shares * last_px < 10.0:
            return {"ok": False, "action": "SKIP", "reason": "卖出金额不足 $10", "position": pos}
        order_payload = {
            "instId": ALLOWED_INST_ID,
            "tdMode": ALLOWED_TD_MODE,
            "side": "sell",
            "ordType": "market",
            "tgtCcy": "base_ccy",
            "sz": f"{sell_shares:.4f}",
        }
        action_label = "SELL"
        est_shares = sell_shares
    else:
        return {"ok": True, "action": "HOLD", "reason": "调仓金额小于 $20 阈值", "position": pos}

    # Final hard assertion before sending to OKX SIMULATED endpoint
    assert order_payload["instId"] == ALLOWED_INST_ID and order_payload["tdMode"] == ALLOWED_TD_MODE
    resp = _demo_private_request("POST", "/api/v5/trade/order", payload=order_payload)
    ord_ok = resp.get("code") == "0"
    ord_id = ((resp.get("data") or [{}])[0]).get("ordId", "")

    rec = {
        "recorded_at": dt.datetime.now().isoformat(timespec="seconds"),
        "simulated_only": True,
        "strategy_version": strategy_version,
        "inst_id": ALLOWED_INST_ID,
        "action": action_label,
        "target_weight": round(tw, 4),
        "prev_weight": round(cur_w, 4),
        "order_sz": order_payload["sz"],
        "est_shares": round(est_shares, 4),
        "ref_xmu_px": last_px,
        "spread_bps": mkt["spread_bps"],
        "ok": ord_ok,
        "ord_id": ord_id,
        "msg": resp.get("msg", "") or ((resp.get("data") or [{}])[0]).get("sMsg", ""),
    }
    _append_demo_order_log(rec)
    log.info("OKX DEMO %s %s sz=%s ok=%s ordId=%s", action_label, ALLOWED_INST_ID, order_payload["sz"], ord_ok, ord_id)

    after_pos = get_demo_position_summary()
    return {"ok": ord_ok, "action": action_label, "order": rec, "before": pos, "after": after_pos, "raw_resp": resp}
