"""Local Web Dashboard HTTP Server (Pure Python stdlib — zero external web framework needed).

Run:
    python -m mu_quant.dashboard.server --port 8501
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Allow running directly as script
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mu_quant.config import list_strategy_versions, load_active_config, load_strategy_version
from mu_quant.data.loader import fetch_live_mu_quote, load_mu_and_benchmarks, refresh_all_market_data, validate_mu_quality
from mu_quant.execution_simulator.okx_demo_channel import get_demo_account_balance, probe_xmu_market
from mu_quant.execution_simulator.simulator import simulate_rebalance_order
from mu_quant.features.indicators import build_features
from mu_quant.logs.logger import get_logger
from mu_quant.long_strategy.strategies import latest_long_signal
from mu_quant.market_regime.regime import latest_regime_summary
from mu_quant.paper_trading.engine import get_paper_status, record_daily_signal, record_paper_trade
from mu_quant.paper_trading.forward_tracker import get_forward_scorecard
from mu_quant.portfolio.combiner import run_all_portfolio_views
from mu_quant.risk.manager import RiskConfig, compute_trade_risk_plan, evaluate_account_risk_status
from mu_quant.short_strategy.strategies import latest_short_signal

log = get_logger("mu_quant.dashboard")
STATIC_DIR = Path(__file__).resolve().parent / "static"


def _clean_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if (not np.isfinite(obj)) else round(float(obj), 6)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def build_dashboard_payload(
    long_mode: str = "dual_confirm_dvol55",
    portfolio_mode: str = "long_only",
    start_date: str = "2004-01-01",
) -> dict[str, Any]:
    cfg = load_strategy_version()
    active_cfg = load_active_config()
    data = load_mu_and_benchmarks()
    mu = data["MU"]
    earnings = data.get("earnings")
    f = build_features(mu, data.get("SMH"), data.get("SOX"))
    quality = validate_mu_quality(mu, data.get("SPY"))

    last_row = f.iloc[-1]
    prev_row = f.iloc[-2]
    last_date = str(f.index[-1].date())
    close_px = float(last_row["raw_close"])
    chg_pct = float(last_row["Close"] / prev_row["Close"] - 1.0)
    atr_val = float(last_row["raw_atr14"]) if pd.notna(last_row["raw_atr14"]) else 0.0

    regime = latest_regime_summary(f)
    long_sig = latest_long_signal(f, cfg, mode=long_mode)
    short_sig = latest_short_signal(f, cfg)

    # Combine target weight according to selected portfolio_mode
    if portfolio_mode == "long_only":
        target_w = long_sig["target_weight"]
        active_side = long_sig["signal"]
        stop_mult = float(cfg["long_strategy"].get("stop_atr_mult", 2.5))
    elif portfolio_mode == "short_only":
        target_w = short_sig["target_weight"]
        active_side = "SHORT" if target_w < -1e-4 else "FLAT"
        stop_mult = float(cfg["short_strategy"].get("stop_atr_mult", 2.0))
    else:  # long_short_combo
        if long_sig["target_weight"] > 1e-4:
            target_w = long_sig["target_weight"]
            active_side = "LONG"
            stop_mult = float(cfg["long_strategy"].get("stop_atr_mult", 2.5))
        elif short_sig["target_weight"] < -1e-4:
            target_w = short_sig["target_weight"]
            active_side = "SHORT"
            stop_mult = float(cfg["short_strategy"].get("stop_atr_mult", 2.0))
        else:
            target_w = 0.0
            active_side = "FLAT"
            stop_mult = 2.5

    paper = get_paper_status(close_px)
    risk_plan = compute_trade_risk_plan(
        equity=paper["equity"],
        price=close_px,
        atr=atr_val,
        side=active_side,
        target_weight=target_w,
        stop_atr_mult=stop_mult,
        borrow_annual=float(cfg["execution"].get("short_borrow_annual", 0.015)),
    )
    sim_ticket = simulate_rebalance_order(
        date=last_date,
        ticker="MU",
        current_shares=paper["shares"],
        equity=paper["equity"],
        ref_price=close_px,
        target_weight=target_w,
        stop_price=risk_plan.stop_price,
        strategy_version=cfg["version"],
        commission_bps=float(cfg["execution"].get("commission_bps", 10.0)),
        slippage_bps=float(cfg["execution"].get("slippage_bps", 5.0)),
        rebalance_tol=float(cfg["long_strategy"].get("rebalance_tolerance", 0.05)),
    )

    views = run_all_portfolio_views(f, cfg, earnings, start=start_date, long_mode=long_mode, initial=float(active_cfg.get("initial_cash", 100_000.0)))
    risk_cfg = RiskConfig(
        dd_brake_enabled=bool(cfg["risk"].get("dd_brake_enabled", False)),
        dd_brake_start=float(cfg["risk"].get("dd_brake_start", 0.15)),
        dd_brake_full=float(cfg["risk"].get("dd_brake_full", 0.35)),
        dd_min_scale=float(cfg["risk"].get("dd_min_scale", 0.25)),
        dd_halt_threshold=float(cfg["risk"].get("dd_halt_threshold", 0.40)),
    )
    acc_risk = evaluate_account_risk_status(views["long_primary"]["equity"], risk_cfg)

    # Downsample equity curves to ~350 points for fast crisp SVG charting
    def _sample_curve(s: pd.Series) -> list[list[Any]]:
        step = max(1, len(s) // 350)
        sub = s.iloc[::step]
        if sub.index[-1] != s.index[-1]:
            sub = pd.concat([sub, s.iloc[[-1]]])
        return [[str(d.date()), round(float(v), 2)] for d, v in sub.items()]

    # Recent 120-day price + MA5/20/50/200 for chart
    tail_f = f.tail(120)
    price_chart = {
        "dates": [str(d.date()) for d in tail_f.index],
        "close": [float(x) for x in tail_f["Close"]],
        "ma5": [float(x) if pd.notna(x) else None for x in tail_f["sma5"]],
        "ma20": [float(x) if pd.notna(x) else None for x in tail_f["sma20"]],
        "ma50": [float(x) if pd.notna(x) else None for x in tail_f["sma50"]],
        "ma200": [float(x) if pd.notna(x) else None for x in tail_f["sma200"]],
    }

    # Build plain-Chinese actionable Trading Playbook (Entry / Exit / Stop-Loss / Take-Profit checklist)
    sma20 = float(last_row["sma20"]) if pd.notna(last_row["sma20"]) else np.nan
    sma50 = float(last_row["sma50"]) if pd.notna(last_row["sma50"]) else np.nan
    sma200 = float(last_row["sma200"]) if pd.notna(last_row["sma200"]) else np.nan
    don_low20 = float(last_row["don_low20"]) if pd.notna(last_row["don_low20"]) else np.nan
    vol_ratio = float(last_row["vol_ratio20"]) if pd.notna(last_row["vol_ratio20"]) else np.nan
    clv = float(last_row["clv"]) if pd.notna(last_row["clv"]) else np.nan
    dist_ma20 = float(last_row["dist_ma20"]) if pd.notna(last_row["dist_ma20"]) else np.nan
    dvol21 = float(last_row["dvol21"]) if pd.notna(last_row["dvol21"]) else np.nan
    rsi14 = float(last_row["rsi14"]) if pd.notna(last_row["rsi14"]) else np.nan

    long_stop = max(0.01, close_px - 2.5 * atr_val)
    long_r = max(0.01, close_px - long_stop)
    short_stop = close_px + 2.0 * atr_val
    short_r = max(0.01, short_stop - close_px)

    if active_side == "LONG":
        headline = f"建议持有多头 {target_w*100:.1f}% 仓位（其余 {(1-target_w)*100:.1f}% 保留现金防守）"
        action_detail = (
            f"按 T+1 美股开盘价执行调仓。当前参考止损价 ${long_stop:.2f}（距现价 -{long_r/close_px*100:.1f}%）；"
            f"若收盘价同时跌破 MA200(${sma200:.2f}) 且 MA50(${sma50:.2f}) 下穿 MA200，则次日开盘彻底清仓离场。"
        )
    elif active_side == "SHORT":
        headline = f"触发 S3 实验性做空信号：建议保守空仓 {abs(target_w)*100:.1f}%（需人工核验融券）"
        action_detail = (
            f"空头初始止损价 ${short_stop:.2f}（+{short_r/close_px*100:.1f}%），2R 止盈参考价 ${max(0.01, close_px - 2*short_r):.2f}；"
            f"站回 MA20(${sma20:.2f})、RSI<25、满 40 交易日或财报前 2 日必须平仓。"
        )
    else:
        headline = "建议空仓观望（100% 现金保护本金）"
        action_detail = (
            f"当前不满足做多或做空条件，空仓是合法且重要的风控决策。等待股价重新站稳 MA200(${sma200:.2f}) 或满足下方入场清单后再行动。"
        )

    playbook = {
        "headline": headline,
        "action_detail": action_detail,
        "long_rules": {
            "entry_checklist": [
                {"rule": "中长期多头排列 (MA50 > MA200)", "met": bool(sma50 > sma200), "detail": f"MA50=${sma50:.2f} vs MA200=${sma200:.2f}（满足给 50% 基础仓）"},
                {"rule": "股价站稳年线 (收盘价 > MA200)", "met": bool(close_px > sma200), "detail": f"现价=${close_px:.2f} vs MA200=${sma200:.2f}（满足再给 50% 基础仓）"},
                {"rule": "仅下行波动率控仓 (55% ÷ dvol21)", "met": bool(np.isfinite(dvol21) and dvol21 <= 0.55), "detail": f"当前下行波动={dvol21*100:.1f}% → 波动上限={min(100.0, 55.0/max(dvol21*100,1e-3)):.1f}%（暴跌期自动降仓，上涨暴涨不砍仓）"},
            ],
            "stop_loss": f"参考硬止损：${long_stop:.2f}（现价 - 2.5×ATR14，幅度 -{long_r/close_px*100:.1f}%）；止损只允许随盈利上移，严禁向下挪动或亏损补仓。",
            "take_profit": f"趋势主线不设硬上限以吃足主升浪；波段参考止盈位：2R=${close_px + 2*long_r:.2f} / 3R=${close_px + 3*long_r:.2f}；当跌破 MA200(${sma200:.2f}) 时先减半仓锁利，下行波动超 55% 时自动缩仓锁利。",
            "exit_all": f"彻底清仓离场：当 收盘价 < MA200(${sma200:.2f}) 且 MA50(${sma50:.2f}) < MA200 时，次日开盘全清转为空仓。",
        },
        "short_rules": {
            "entry_checklist": [
                {"rule": "熊市趋势确认 (收盘<MA200 且 MA50<MA200)", "met": bool(close_px < sma200 and sma50 < sma200), "detail": f"现价=${close_px:.2f}, MA50=${sma50:.2f}, MA200=${sma200:.2f}"},
                {"rule": "支撑位有效跌破 (收盘 < 20日低点)", "met": bool(close_px < don_low20), "detail": f"现价=${close_px:.2f} vs 20日低=${don_low20:.2f}"},
                {"rule": "放量下跌确认 (量比 > 1.2)", "met": bool(vol_ratio > 1.2), "detail": f"当前20日量比={vol_ratio:.2f}"},
                {"rule": "弱势收盘 (CLV < 0.35)", "met": bool(clv < 0.35), "detail": f"当前CLV={clv:.2f}（越接近0说明越收在全天最低附近）"},
                {"rule": "防超跌追空 (距MA20偏离 > -12%)", "met": bool(dist_ma20 > -0.12), "detail": f"当前距MA20={dist_ma20*100:+.1f}%（跌太猛后禁止追空防暴力反弹）"},
            ],
            "stop_loss": f"初始止损：${short_stop:.2f}（现价 + 2.0×ATR14，幅度 +{short_r/close_px*100:.1f}%）；移动止损 2.5×ATR 只向下移动，跳空穿损按开盘价止损。",
            "take_profit": f"明确止盈位：2R 目标价=${max(0.01, close_px - 2*short_r):.2f}，或 RSI(14) 跌破 25（当前 {rsi14:.1f}）超跌落袋平仓。",
            "exit_all": f"强制离场：收盘站回 MA20(${sma20:.2f}) 上方 / 持仓满 40 交易日 / 财报前 2 日强制平仓防跳空逼空。",
        },
    }

    payload = {
        "meta": {
            "version": cfg["version"],
            "available_versions": list_strategy_versions(),
            "long_mode": long_mode,
            "portfolio_mode": portfolio_mode,
            "start_date": start_date,
            "last_date": last_date,
            "live_auto_order_forbidden": True,
        },
        "playbook": playbook,
        "market": {
            "date": last_date,
            "open": float(last_row["raw_open"]),
            "high": float(last_row["raw_high"]),
            "low": float(last_row["raw_low"]),
            "close": close_px,
            "chg_pct": chg_pct,
            "volume": float(last_row["Volume"]),
            "vol_ratio20": float(last_row["vol_ratio20"]) if pd.notna(last_row["vol_ratio20"]) else None,
            "ma5": float(last_row["sma5"]) if pd.notna(last_row["sma5"]) else None,
            "ma20": float(last_row["sma20"]) if pd.notna(last_row["sma20"]) else None,
            "ma50": float(last_row["sma50"]) if pd.notna(last_row["sma50"]) else None,
            "ma200": float(last_row["sma200"]) if pd.notna(last_row["sma200"]) else None,
            "atr14": atr_val,
            "atr_pct": float(last_row["atr_pct"]) if pd.notna(last_row["atr_pct"]) else None,
            "rvol21": float(last_row["rvol21"]) if pd.notna(last_row["rvol21"]) else None,
            "dvol21": float(last_row["dvol21"]) if pd.notna(last_row["dvol21"]) else None,
            "rsi14": float(last_row["rsi14"]) if pd.notna(last_row["rsi14"]) else None,
            "macd": float(last_row["macd"]) if pd.notna(last_row["macd"]) else None,
            "macd_sig": float(last_row["macd_sig"]) if pd.notna(last_row["macd_sig"]) else None,
            "macd_hist": float(last_row["macd_hist"]) if pd.notna(last_row["macd_hist"]) else None,
        },
        "quality": quality,
        "regime": regime,
        "signals": {
            "long": long_sig,
            "short": short_sig,
            "active_side": active_side,
            "target_weight": target_w,
        },
        "risk_plan": asdict(risk_plan),
        "account_risk": acc_risk,
        "simulated_ticket": sim_ticket,
        "paper_trading": paper,
        "forward_scorecard": get_forward_scorecard(),
        "price_chart": price_chart,
        "backtest_comparison": {
            k: {
                "metrics": v["metrics"],
                "curve": _sample_curve(v["equity"]),
            }
            for k, v in views.items()
        },
        "recent_backtest_trades": {
            "long": views["long_primary"]["trades"][-20:][::-1],
            "short": views["short_experimental"]["trades"][-20:][::-1],
        },
    }
    return _clean_json(payload)


class DashboardHandler(BaseHTTPRequestHandler):
    def _send_json(self, data: Any, status: int = 200) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            html = (STATIC_DIR / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return
        if parsed.path in ("/screener", "/screener.html"):
            html = (STATIC_DIR / "screener.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return
        if parsed.path == "/api/screener":
            qs = urllib.parse.parse_qs(parsed.query)
            preset = qs.get("preset", ["core_momentum"])[0]
            sector = qs.get("sector", ["ALL"])[0]
            scan_mode = qs.get("scan_mode", ["curated"])[0]
            do_refresh = qs.get("refresh", ["0"])[0] in ("1", "true", "yes")
            try:
                from mu_quant.screener.engine import run_screener
                res = run_screener(
                    preset=preset,
                    sector=sector,
                    refresh=do_refresh,
                    scan_mode=scan_mode,
                    record_history=do_refresh,
                )
                self._send_json(_clean_json(res))
            except Exception as exc:  # noqa: BLE001
                log.exception("Screener API error")
                self._send_json({"ok": False, "error": str(exc)}, status=500)
            return
        if parsed.path == "/api/dashboard":
            qs = urllib.parse.parse_qs(parsed.query)
            long_mode = qs.get("long_mode", ["dual_confirm_dvol55"])[0]
            portfolio_mode = qs.get("portfolio_mode", ["long_only"])[0]
            start_date = qs.get("start_date", ["2004-01-01"])[0]
            try:
                self._send_json(build_dashboard_payload(long_mode, portfolio_mode, start_date))
            except Exception as exc:  # noqa: BLE001
                log.exception("Dashboard API error")
                self._send_json({"error": str(exc)}, status=500)
            return
        if parsed.path == "/api/live":
            mu_live = fetch_live_mu_quote()
            ref_px = mu_live.get("live_price") or mu_live.get("regular_price")
            try:
                xmu = probe_xmu_market(mu_ref_close=ref_px)
            except Exception as exc:  # noqa: BLE001
                xmu = {"ok": False, "error": f"OKX暂不可达({exc})"}
            self._send_json(_clean_json({
                "ok": bool(mu_live.get("ok") or xmu.get("last_px")),
                "mu_live": mu_live,
                "xmu_live": xmu,
            }))
            return
        if parsed.path == "/api/okx/demo":
            try:
                mu = load_mu_and_benchmarks()["MU"]
                mu_close = float(mu["Close"].iloc[-1])
                mkt = probe_xmu_market(mu_ref_close=mu_close)
                bal = get_demo_account_balance()
                total_eq = None
                usdt_avail = None
                if bal.get("code") == "0" and bal.get("data"):
                    d0 = bal["data"][0]
                    total_eq = float(d0.get("totalEq") or 0.0)
                    for item in d0.get("details", []):
                        if item.get("ccy") == "USDT":
                            usdt_avail = float(item.get("availBal") or 0.0)
                self._send_json(_clean_json({
                    "ok": True,
                    "simulated_only": True,
                    "market": mkt,
                    "account": {"ok": bal.get("code") == "0", "total_eq_usd": total_eq, "usdt_avail": usdt_avail, "msg": bal.get("msg", "")},
                }))
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=500)
            return
        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
        try:
            if parsed.path == "/api/paper/signal":
                rec = record_daily_signal(body)
                self._send_json({"ok": True, "record": rec})
                return
            if parsed.path == "/api/paper/trade":
                rec = record_paper_trade(body)
                self._send_json({"ok": True, "record": rec})
                return
            if parsed.path == "/api/data/refresh":
                res = refresh_all_market_data(force=True)
                self._send_json(res)
                return
            if parsed.path == "/api/screener/symbol":
                from mu_quant.screener.engine import add_custom_symbol, remove_custom_symbol
                action = str(body.get("action", "add")).lower()
                symbol = str(body.get("symbol", "")).strip().upper()
                if action == "remove":
                    res = remove_custom_symbol(symbol)
                else:
                    res = add_custom_symbol(
                        symbol=symbol,
                        sector=str(body.get("sector", "自选观察")),
                        theme=str(body.get("theme", "自定义美股标的")),
                    )
                self._send_json(_clean_json(res))
                return
            self._send_json({"error": "Not found"}, status=404)
        except Exception as exc:  # noqa: BLE001
            log.exception("POST error on %s", parsed.path)
            self._send_json({"ok": False, "error": str(exc)}, status=500)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return


def _background_auto_refresh_loop(interval_hours: float = 4.0) -> None:
    import time
    while True:
        try:
            refresh_all_market_data(force=False, min_interval_hours=interval_hours)
        except Exception:  # noqa: BLE001
            log.exception("Background market data auto-refresh error")
        time.sleep(interval_hours * 3600)


def main() -> None:
    import threading
    parser = argparse.ArgumentParser(description="MU Specialist Quant System 2.0 Web Dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--no-auto-refresh", action="store_true", help="Disable background auto-refresh")
    args = parser.parse_args()
    if not args.no_auto_refresh:
        threading.Thread(target=_background_auto_refresh_loop, args=(4.0,), daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"MU Specialist Quant System 2.0 Dashboard running at http://{args.host}:{args.port}")
    log.info("Dashboard server started on http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
