"""Local Web Dashboard Server for BTC Perpetual Futures Quant System (Port 8502).

Pure Python stdlib ThreadingHTTPServer — zero external web framework required.
Run:
    python src/btc_quant/dashboard/server.py --port 8502
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from btc_quant.backtest.engine import PerpBacktestConfig, run_perp_backtest
from btc_quant.config import load_active_config
from btc_quant.data.loader import (
    fetch_live_snapshot,
    load_btc_datasets,
    refresh_btc_market_data,
    validate_btc_data_quality,
)
from btc_quant.execution_simulator.okx_demo_swap import get_demo_swap_account_state, sync_demo_swap_target
from btc_quant.features.indicators import build_1d_features, build_4h_features
from btc_quant.logs.logger import get_logger
from btc_quant.market_regime.regime import latest_regime_summary
from btc_quant.paper_trading.engine import get_paper_summary, record_paper_step
from btc_quant.risk.manager import BtcRiskConfig, compute_contract_risk_plan
from btc_quant.strategies.perp_strategies import generate_strategy_signals, latest_signal_snapshot

log = get_logger("btc_quant.dashboard")
STATIC_DIR = Path(__file__).resolve().parent / "static"


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if (not np.isfinite(obj)) else round(float(obj), 6)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def build_dashboard_payload(
    mode: str = "dual_horizon_ensemble",
    start_date: str = "2021-02-01",
    fee_bps: float = 5.0,
    slippage_bps: float = 3.0,
) -> dict[str, Any]:
    cfg = load_active_config()
    ds = load_btc_datasets()
    qa = validate_btc_data_quality(ds)
    df_1d_f = build_1d_features(ds["df_1d"])
    df_4h_f = build_4h_features(ds["df_4h"], df_1d_f)

    df_bt = df_4h_f.loc[start_date:].copy()
    if len(df_bt) < 50:
        df_bt = df_4h_f.iloc[-1000:].copy()

    strat_cfg = cfg.get("strategy", {})
    risk_raw = cfg.get("risk", {})
    risk_cfg = BtcRiskConfig(
        max_long_leverage=float(strat_cfg.get("max_long_leverage", 2.0)),
        max_short_leverage=float(strat_cfg.get("max_short_leverage", 1.2)),
        hard_cap_leverage=float(risk_raw.get("hard_cap_leverage", 3.0)),
        single_trade_risk_pct=float(risk_raw.get("single_trade_risk_pct", 0.025)),
        dd_brake_start=float(risk_raw.get("dd_brake_start", 0.12)),
        dd_brake_full=float(risk_raw.get("dd_brake_full", 0.25)),
        dd_min_scale=float(risk_raw.get("dd_min_scale", 0.35)),
        dd_halt_threshold=float(risk_raw.get("dd_halt_threshold", 0.32)),
    )

    bt_cfg = PerpBacktestConfig(
        taker_fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        max_long_leverage=risk_cfg.max_long_leverage,
        max_short_leverage=risk_cfg.max_short_leverage,
        rebalance_tol=float(cfg.get("execution", {}).get("rebalance_tol", 0.20)),
    )

    compare_modes = [
        ("dual_horizon_ensemble", "双周期多空主力 (Flagship)"),
        ("cta_trend_ensemble", "CTA突破+趋势组合"),
        ("donchian_cta", "唐奇安通道CTA"),
        ("long_only_control", "纯多头对照 (零做空)"),
        ("buy_and_hold", "BTC永续1x买入持有"),
    ]

    comparison = {}
    curves_daily = {}
    selected_trades = []
    for m_key, m_label in compare_modes:
        tgt, sd, td = generate_strategy_signals(df_bt, mode=m_key, params=strat_cfg)
        res = run_perp_backtest(df_bt, tgt, sd, td, bt_cfg)
        comparison[m_key] = {"label": m_label, **res["metrics"]}
        # Downsample equity curve to 1D for crisp fast chart rendering
        eq_d = res["equity"].resample("1D").last().ffill().dropna()
        curves_daily[m_key] = [
            {"date": str(ts.date()), "equity": round(float(v), 2)} for ts, v in eq_d.items()
        ]
        if m_key == mode and res["trades"] is not None and not res["trades"].empty:
            selected_trades = res["trades"].tail(20).iloc[::-1].to_dict(orient="records")

    regime = latest_regime_summary(df_4h_f)
    signal = latest_signal_snapshot(df_4h_f, mode=mode, params=strat_cfg)
    paper = get_paper_summary()
    st = paper["state"]
    risk_plan = compute_contract_risk_plan(
        equity_usdt=float(st.get("equity_usdt", 100_000.0)),
        price=float(signal["close_px"]),
        target_leverage=float(signal["target_leverage"]),
        stop_distance_pct=float(signal["stop_distance_pct"]) / 100.0,
        current_drawdown=float(st.get("drawdown_pct", 0.0)) / 100.0,
        funding_7d_ann_pct=float(signal.get("funding_7d_ann_pct", 0.0)),
        cfg=risk_cfg,
    )

    try:
        live = fetch_live_snapshot()
    except Exception as exc:
        live = {"error": str(exc)}

    try:
        okx_demo = get_demo_swap_account_state()
    except Exception as exc:
        okx_demo = {"configured": False, "error": str(exc)}

    return _clean(
        {
            "version": cfg.get("version", "v1.0.0"),
            "selected_mode": mode,
            "data_audit": qa,
            "live_snapshot": live,
            "regime": regime,
            "signal": signal,
            "risk_plan": risk_plan,
            "comparison": comparison,
            "curves_daily": curves_daily,
            "recent_trades": selected_trades,
            "paper": paper,
            "okx_demo": okx_demo,
        }
    )


class BtcDashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        log.info("HTTP %s - %s", self.address_string(), fmt % args)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(_clean(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_file(self, path: Path, ctype: str) -> None:
        raw = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        try:
            if parsed.path in ("/", "/index.html"):
                self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            elif parsed.path == "/api/dashboard":
                mode = qs.get("mode", ["dual_horizon_ensemble"])[0]
                start = qs.get("start", ["2021-02-01"])[0]
                fee = float(qs.get("fee_bps", ["5.0"])[0])
                slip = float(qs.get("slip_bps", ["3.0"])[0])
                self._send_json(build_dashboard_payload(mode=mode, start_date=start, fee_bps=fee, slippage_bps=slip))
            else:
                self._send_json({"error": "Not found"}, status=404)
        except Exception as exc:
            log.exception("GET %s failed", parsed.path)
            self._send_json({"error": str(exc)}, status=500)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
        try:
            if parsed.path == "/api/refresh":
                meta = refresh_btc_market_data()
                self._send_json({"ok": True, "meta": meta})
            elif parsed.path == "/api/paper_step":
                cfg = load_active_config()
                ds = load_btc_datasets()
                df_4h_f = build_4h_features(ds["df_4h"], build_1d_features(ds["df_1d"]))
                strat_cfg = cfg.get("strategy", {})
                regime = latest_regime_summary(df_4h_f)
                signal = latest_signal_snapshot(df_4h_f, mode=strat_cfg.get("default_mode", "dual_horizon_ensemble"), params=strat_cfg)
                res = record_paper_step(signal, regime, funding_rate_8h=float(df_4h_f["funding_rate"].iloc[-1]))
                self._send_json({"ok": True, "result": res})
            elif parsed.path == "/api/okx_demo_sync":
                dry_run = bool(body.get("dry_run", True))
                alloc = body.get("allocation_usdt")
                cfg = load_active_config()
                ds = load_btc_datasets()
                df_4h_f = build_4h_features(ds["df_4h"], build_1d_features(ds["df_1d"]))
                strat_cfg = cfg.get("strategy", {})
                signal = latest_signal_snapshot(df_4h_f, mode=strat_cfg.get("default_mode", "dual_horizon_ensemble"), params=strat_cfg)
                live = fetch_live_snapshot()
                plan = compute_contract_risk_plan(
                    equity_usdt=float(alloc or 100_000.0),
                    price=float(live["last"]),
                    target_leverage=float(signal["target_leverage"]),
                    stop_distance_pct=float(signal["stop_distance_pct"]) / 100.0,
                    funding_7d_ann_pct=float(signal.get("funding_7d_ann_pct", 0.0)),
                )
                sync_res = sync_demo_swap_target(
                    target_leverage=plan["approved_leverage"],
                    ref_price=float(live["last"]),
                    isolated_leverage=plan["isolated_margin_leverage"],
                    allocation_usdt=float(alloc) if alloc else None,
                    dry_run=dry_run,
                )
                self._send_json(sync_res)
            else:
                self._send_json({"error": "Not found"}, status=404)
        except Exception as exc:
            log.exception("POST %s failed", parsed.path)
            self._send_json({"error": str(exc)}, status=500)


def main() -> None:
    ap = argparse.ArgumentParser(description="BTC Perpetual Quant Dashboard")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8502)
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), BtcDashboardHandler)
    log.info("BTC Perpetual Quant Dashboard running at http://%s:%d", args.host, args.port)
    print(f"BTC Perpetual Quant Dashboard running at http://{args.host}:{args.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.server_close()


if __name__ == "__main__":
    main()
