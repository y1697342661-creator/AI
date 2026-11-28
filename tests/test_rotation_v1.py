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


if __name__ == "__main__":
    unittest.main()
