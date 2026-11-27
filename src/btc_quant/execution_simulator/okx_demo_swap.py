"""Isolated OKX Simulated Trading (`x-simulated-trading: 1`) Channel for BTC-USDT-SWAP.

HARD SAFETY INVARIANTS (CANNOT BE BYPASSED):
1. Every authenticated API call HARDCODES header `x-simulated-trading: 1` (Demo account ONLY).
2. Requires `"simulated_only": true` in `configs/okx_demo_secrets.local.json`.
3. Whitelists ONLY `BTC-USDT-SWAP` in `tdMode="isolated"` (NEVER cross margin, NEVER live).
4. Enforces hard leverage ceiling `<= 3x` on simulated account before placing any demo order.
5. Writes exclusively to `data/btc/okx_demo/` (100% isolated from MU system).
"""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import json
import os
import urllib.error
import urllib.request
from typing import Any

import pandas as pd

from btc_quant.config import OKX_DEMO_DIR, OKX_DEMO_SECRETS_FILE
from btc_quant.logs.logger import get_logger
from btc_quant.risk.manager import OKX_BTC_SWAP_CT_VAL

log = get_logger("btc_quant.okx_demo_swap")

OKX_BASE = "https://www.okx.com"
ALLOWED_INST_ID = "BTC-USDT-SWAP"
ALLOWED_TD_MODE = "isolated"
MAX_ALLOWED_LEVERAGE = 3

DEMO_ORDERS_LOG = OKX_DEMO_DIR / "btc_swap_demo_orders.csv"


def _http_json(method: str, path: str, headers: dict[str, str] | None = None, body: str = "") -> dict[str, Any]:
    hdr = {"User-Agent": "Mozilla/5.0"}
    if headers:
        hdr.update(headers)
    data = body.encode("utf-8") if body else None
    req = urllib.request.Request(OKX_BASE + path, data=data, headers=hdr, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"code": str(e.code), "msg": e.read().decode("utf-8", errors="replace"), "data": []}


def _load_demo_secrets() -> dict[str, Any]:
    """Load OKX Simulated Trading secrets from GitHub Actions env vars OR local gitignored JSON."""
    env_key = os.environ.get("OKX_DEMO_API_KEY", "").strip()
    env_sec = os.environ.get("OKX_DEMO_SECRET_KEY", "").strip()
    env_pwd = os.environ.get("OKX_DEMO_PASSPHRASE", "").strip()
    if env_key and env_sec and env_pwd:
        return {
            "apiKey": env_key,
            "secretKey": env_sec,
            "passphrase": env_pwd,
            "simulated_only": True,  # Header x-simulated-trading: 1 is still hardcoded below
        }
    if not OKX_DEMO_SECRETS_FILE.exists():
        raise FileNotFoundError(
            f"Missing OKX demo secrets (neither env OKX_DEMO_* nor {OKX_DEMO_SECRETS_FILE} found)"
        )
    sec = json.loads(OKX_DEMO_SECRETS_FILE.read_text(encoding="utf-8-sig"))
    if sec.get("simulated_only") is not True:
        raise RuntimeError("SAFETY BLOCK: okx_demo_secrets.local.json MUST set simulated_only=true.")
    for k in ("apiKey", "secretKey", "passphrase"):
        if not sec.get(k):
            raise ValueError(f"Missing field '{k}' in {OKX_DEMO_SECRETS_FILE}")
    return sec


def _sign_okx(secret: str, ts: str, method: str, path: str, body: str = "") -> str:
    msg = f"{ts}{method.upper()}{path}{body}".encode("utf-8")
    mac = hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).digest()
    return base64.b64encode(mac).decode("utf-8")


def _demo_private_request(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
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
    return _http_json(method, path, headers=headers, body=body)


def get_demo_swap_account_state() -> dict[str, Any]:
    """Read USDT equity and current BTC-USDT-SWAP position on OKX Simulated Trading account."""
    has_env = bool(os.environ.get("OKX_DEMO_API_KEY") and os.environ.get("OKX_DEMO_SECRET_KEY"))
    if not has_env and not OKX_DEMO_SECRETS_FILE.exists():
        return {"configured": False, "error": "OKX demo secrets not configured"}
    bal = _demo_private_request("GET", "/api/v5/account/balance?ccy=USDT")
    pos = _demo_private_request("GET", f"/api/v5/account/positions?instType=SWAP&instId={ALLOWED_INST_ID}")
    if bal.get("code") != "0":
        return {"configured": True, "ok": False, "error": bal.get("msg", "OKX auth error")}

    usdt_eq = 0.0
    usdt_avail = 0.0
    for item in bal.get("data", []):
        for det in item.get("details", []):
            if det.get("ccy") == "USDT":
                usdt_eq = float(det.get("eq") or 0.0)
                usdt_avail = float(det.get("availBal") or 0.0)

    net_contracts = 0.0
    avg_px = 0.0
    upl = 0.0
    liq_px = None
    cur_lev = None
    for p in pos.get("data", []):
        sz = float(p.get("pos") or 0.0)
        if abs(sz) > 0:
            net_contracts += sz
            avg_px = float(p.get("avgPx") or 0.0)
            upl += float(p.get("upl") or 0.0)
            liq_px = float(p["liqPx"]) if p.get("liqPx") else None
            cur_lev = float(p["lever"]) if p.get("lever") else None

    return {
        "configured": True,
        "ok": True,
        "simulated_only": True,
        "inst_id": ALLOWED_INST_ID,
        "usdt_equity": round(usdt_eq, 2),
        "usdt_available": round(usdt_avail, 2),
        "net_contracts": round(net_contracts, 2),
        "net_btc": round(net_contracts * OKX_BTC_SWAP_CT_VAL, 5),
        "avg_entry_px": round(avg_px, 2),
        "unrealized_pnl_usdt": round(upl, 2),
        "liq_px": liq_px,
        "lever": cur_lev,
    }


def sync_demo_swap_target(
    target_leverage: float,
    ref_price: float,
    isolated_leverage: int = 2,
    allocation_usdt: float | None = None,
    stop_loss_px: float | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Rebalance OKX Simulated BTC-USDT-SWAP isolated position to match target_leverage.

    Defaults to `dry_run=True` for safety; set `dry_run=False` to actually submit
    to OKX Simulated Trading (`x-simulated-trading: 1`). When opening/increasing a
    position and `stop_loss_px` is provided, attaches an exchange-native OKX stop-loss
    (`attachAlgoOrds`) so protection stays active 24/7 on OKX's matching engine.
    """
    iso_lev = max(1, min(MAX_ALLOWED_LEVERAGE, int(isolated_leverage)))
    state = get_demo_swap_account_state()
    if not state.get("ok"):
        return {"ok": False, "error": state.get("error", "Account state unavailable")}

    cap_usdt = float(allocation_usdt) if (allocation_usdt and allocation_usdt > 0) else float(state["usdt_equity"])
    capped_tgt = max(-float(MAX_ALLOWED_LEVERAGE), min(float(MAX_ALLOWED_LEVERAGE), float(target_leverage)))
    target_notional = capped_tgt * cap_usdt
    target_contracts = round((target_notional / ref_price) / OKX_BTC_SWAP_CT_VAL, 2) if ref_price > 0 else 0.0
    cur_contracts = float(state["net_contracts"])
    delta_contracts = round(target_contracts - cur_contracts, 2)

    plan = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "simulated_only": True,
        "dry_run": dry_run,
        "inst_id": ALLOWED_INST_ID,
        "td_mode": ALLOWED_TD_MODE,
        "isolated_leverage": iso_lev,
        "capital_base_usdt": round(cap_usdt, 2),
        "ref_price": round(ref_price, 2),
        "target_leverage": round(capped_tgt, 3),
        "current_contracts": cur_contracts,
        "target_contracts": target_contracts,
        "delta_contracts": delta_contracts,
        "stop_loss_px": stop_loss_px,
        "action": "HOLD",
        "order_response": None,
    }

    if abs(delta_contracts) < 0.01:
        return {"ok": True, **plan}

    side = "buy" if delta_contracts > 0 else "sell"
    sz_str = f"{abs(delta_contracts):.2f}"
    plan["action"] = f"{side.upper()} {sz_str} contracts ({abs(delta_contracts) * OKX_BTC_SWAP_CT_VAL:.4f} BTC)"

    if not dry_run:
        # 1. Enforce isolated leverage cap on OKX demo account
        _demo_private_request(
            "POST",
            "/api/v5/account/set-leverage",
            {"instId": ALLOWED_INST_ID, "lever": str(iso_lev), "mgnMode": ALLOWED_TD_MODE},
        )
        # 2. Build simulated market order payload (with optional exchange-native stop-loss)
        order_payload: dict[str, Any] = {
            "instId": ALLOWED_INST_ID,
            "tdMode": ALLOWED_TD_MODE,
            "side": side,
            "ordType": "market",
            "sz": sz_str,
        }
        is_opening_or_adding = (target_contracts != 0.0) and (
            cur_contracts == 0.0 or (target_contracts * delta_contracts > 0)
        )
        if is_opening_or_adding and stop_loss_px and stop_loss_px > 0:
            order_payload["attachAlgoOrds"] = [
                {
                    "slTriggerPx": f"{float(stop_loss_px):.1f}",
                    "slOrdPx": "-1",  # Market stop-loss on OKX matching engine
                    "slTriggerPxType": "mark",
                }
            ]

        resp = _demo_private_request("POST", "/api/v5/trade/order", order_payload)
        plan["order_response"] = resp
        log.info("OKX Demo SWAP order (%s %s): code=%s msg=%s", side, sz_str, resp.get("code"), resp.get("msg"))

        # Append to isolated demo orders ledger
        rec = {
            "timestamp": plan["timestamp"],
            "inst_id": ALLOWED_INST_ID,
            "side": side,
            "sz_contracts": sz_str,
            "ref_price": ref_price,
            "target_leverage": capped_tgt,
            "stop_loss_px": stop_loss_px or "",
            "okx_code": resp.get("code"),
            "okx_ord_id": (resp.get("data") or [{}])[0].get("ordId", ""),
            "okx_msg": resp.get("msg", ""),
        }
        df = pd.read_csv(DEMO_ORDERS_LOG, encoding="utf-8-sig") if DEMO_ORDERS_LOG.exists() else pd.DataFrame()
        pd.concat([df, pd.DataFrame([rec])], ignore_index=True).to_csv(DEMO_ORDERS_LOG, index=False, encoding="utf-8-sig")

    return {"ok": True, **plan}
