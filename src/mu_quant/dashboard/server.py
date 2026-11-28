"""Local Web Dashboard HTTP Server for US Stock Offense+Defense Screener (Pure Python stdlib).

Run:
    python -m mu_quant.dashboard.server --port 8501
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

from mu_quant.logs.logger import get_logger
from mu_quant.rotation.alpaca_paper import ALPACA_ORDERS_LOG, load_alpaca_portfolio_view

log = get_logger("us_screener.dashboard")
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
        if parsed.path in ("/", "/index.html", "/screener", "/screener.html"):
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

        if parsed.path == "/api/alpaca/portfolio":
            qs = urllib.parse.parse_qs(parsed.query)
            try_live = qs.get("live", ["0"])[0] in ("1", "true", "yes")
            try:
                self._send_json(_clean_json(load_alpaca_portfolio_view(try_live=try_live)))
            except Exception as exc:  # noqa: BLE001
                log.exception("Alpaca portfolio API error")
                self._send_json({"ok": False, "error": str(exc)}, status=500)
            return

        if parsed.path == "/api/alpaca/orders":
            try:
                rows = (
                    pd.read_csv(ALPACA_ORDERS_LOG, encoding="utf-8-sig").tail(50).to_dict(orient="records")
                    if ALPACA_ORDERS_LOG.exists()
                    else []
                )
                self._send_json(_clean_json({"ok": True, "paper_only": True, "orders": rows}))
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=500)
            return

        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
        try:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="US Stock Offense+Defense Screener Web Dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"US Stock Screener Dashboard running at http://{args.host}:{args.port}/screener")
    log.info("Dashboard server started on http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
