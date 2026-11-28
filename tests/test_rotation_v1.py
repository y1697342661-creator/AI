"""Unit & Safety Tests for 6-Asset Cross-Sector Top-2 Rotation (Zero Leverage, BTC-Isolated)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mu_quant.rotation.engine import compute_rotation_signal, load_rotation_config


class TestRotationV1(unittest.TestCase):
    def test_01_safety_locks_and_btc_isolation(self) -> None:
        cfg = load_rotation_config()
        self.assertTrue(cfg["execution"]["simulated_only"])
        self.assertFalse(cfg["execution"]["allow_live_auto_order"])
        self.assertEqual(cfg["execution"]["td_mode"], "cash")
        syms = {u["symbol"] for u in cfg["universe"]}
        insts = {u["okx_inst_id"] for u in cfg["universe"]}
        self.assertEqual(syms, {"MU", "PLTR", "HOOD", "GEV", "CRWD", "TSM", "AMZN", "LLY", "WMT", "KO"})
        # Must NEVER touch BTC or SWAP
        for inst in insts:
            self.assertNotIn("BTC", inst)
            self.assertNotIn("SWAP", inst)
            self.assertTrue(inst.endswith("-USDT"))

    def test_02_signal_zero_leverage_and_top_k(self) -> None:
        cfg = load_rotation_config()
        sig = compute_rotation_signal(cfg, refresh=True)
        self.assertLessEqual(len(sig["selected_symbols"]), cfg["rules"]["top_k"])
        self.assertLessEqual(sig["total_invested_weight"], 1.0 + 1e-9)
        self.assertGreaterEqual(sig["cash_weight"], 0.0)
        for sym, w in sig["target_weights"].items():
            self.assertGreaterEqual(w, 0.0)
            self.assertLessEqual(w, cfg["rules"]["max_weight_per_asset"] + 1e-9, f"{sym} weight exceeded cap")

    def test_03_alpaca_paper_hard_safety_lock(self) -> None:
        from mu_quant.rotation.alpaca_paper import ALPACA_PAPER_BASE
        self.assertEqual(ALPACA_PAPER_BASE, "https://paper-api.alpaca.markets")


if __name__ == "__main__":
    unittest.main()
