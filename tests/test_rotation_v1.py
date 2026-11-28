"""Unit & Safety Tests for Screener-Linked Alpaca Paper Trading Executor (Zero Leverage, Paper API Only)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mu_quant.rotation.alpaca_paper import ALPACA_PAPER_BASE, load_screener_allocation


class TestScreenerAlpacaExecutor(unittest.TestCase):
    def test_01_alpaca_paper_hard_safety_lock(self) -> None:
        self.assertEqual(ALPACA_PAPER_BASE, "https://paper-api.alpaca.markets")
        self.assertNotIn("https://api.alpaca.markets", ALPACA_PAPER_BASE)

    def test_02_live_screener_allocation_zero_leverage_and_cap(self) -> None:
        alloc = load_screener_allocation()
        tw = alloc["target_weights"]
        self.assertGreater(len(tw), 0)
        self.assertLessEqual(sum(tw.values()), 1.0 + 1e-9)
        for sym, w in tw.items():
            self.assertRegex(sym, r"^[A-Z]{1,5}$")
            self.assertGreater(w, 0.0)
            self.assertLessEqual(w, 0.40 + 1e-9, f"{sym} exceeded 40% single-stock cap")

    def test_03_enforces_single_cap_normalization_and_ticker_sanitization(self) -> None:
        fake = {
            "version": "test-v1",
            "target_weights": {
                "COST": 0.60,
                "WMT": 0.60,
                "AAPL": 0.50,
                "BTC-USDT": 0.30,
                "BAD.TICKER": 0.20,
                "ZERO": 0.0,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "alloc.json"
            p.write_text(json.dumps(fake), encoding="utf-8")
            alloc = load_screener_allocation(p)
        tw = alloc["target_weights"]
        self.assertEqual(set(tw.keys()), {"COST", "WMT", "AAPL"})
        self.assertLessEqual(sum(tw.values()), 1.0 + 1e-4)
        for w in tw.values():
            self.assertLessEqual(w, 0.40 + 1e-9)

    def test_04_portfolio_view_returns_paper_only_ledger(self) -> None:
        from mu_quant.rotation.alpaca_paper import load_alpaca_portfolio_view
        view = load_alpaca_portfolio_view(try_live=False)
        self.assertTrue(view["ok"])
        self.assertTrue(view["paper_only"])
        self.assertIsInstance(view["orders"], list)
        self.assertIsInstance(view["equity_curve"], list)

    def test_05_execution_risk_guard_drawdown_brake_and_trailing_stop(self) -> None:
        from mu_quant.rotation.alpaca_paper import apply_execution_risk_guard

        tw = {"COST": 0.35, "AAPL": 0.25}
        alloc = {"positions": [{"symbol": "COST", "stop_price": 920.0}, {"symbol": "AAPL", "stop_price": 330.0}]}
        pos_map = {
            "COST": {"current_price": 1000.0, "avg_entry_price": 980.0},
            "AAPL": {"current_price": 350.0, "avg_entry_price": 340.0},
        }
        # 1) Normal equity (DD = 5% < 15%) -> scale 1.0, no stop hit
        tw1, g1, st1 = apply_execution_risk_guard(tw, pos_map, 95_000.0, alloc, {"peak_equity": 100_000.0, "trailing_stops": {}})
        self.assertEqual(g1["brake_status"], "NORMAL")
        self.assertEqual(g1["brake_scale"], 1.0)
        self.assertEqual(tw1, tw)
        self.assertGreater(st1["trailing_stops"]["COST"]["stop_price"], 900.0)

        # 2) Account DD = -20% (halfway between -15% and -25%) -> smooth 0.70x scale
        tw2, g2, _ = apply_execution_risk_guard(tw, pos_map, 80_000.0, alloc, st1)
        self.assertEqual(g2["brake_status"], "SOFT_BRAKE")
        self.assertAlmostEqual(g2["brake_scale"], 0.70, places=2)
        self.assertAlmostEqual(tw2["COST"], 0.35 * 0.70, places=3)

        # 3) AAPL drops below its ratcheted trailing stop -> instant full exit of AAPL!
        pos_drop = {
            "COST": {"current_price": 1000.0, "avg_entry_price": 980.0},
            "AAPL": {"current_price": 310.0, "avg_entry_price": 340.0},
        }
        tw3, g3, _ = apply_execution_risk_guard(tw, pos_drop, 95_000.0, alloc, st1)
        self.assertNotIn("AAPL", tw3)
        self.assertIn("COST", tw3)
        self.assertEqual([x["symbol"] for x in g3["stopped_symbols"]], ["AAPL"])


if __name__ == "__main__":
    unittest.main()
