"""Automated unit, causality (zero-lookahead), execution-order, and dashboard smoke tests."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mu_quant.backtest.engine import BacktestCosts, ShortSwingParams, run_short_swing_backtest, run_weight_backtest
from mu_quant.config import load_active_config, load_strategy_version
from mu_quant.dashboard.server import build_dashboard_payload
from mu_quant.data.loader import load_mu_and_benchmarks, validate_mu_quality
from mu_quant.execution_simulator.simulator import simulate_rebalance_order
from mu_quant.features.indicators import build_features
from mu_quant.long_strategy.strategies import compute_long_weights
from mu_quant.market_regime.regime import classify_regime
from mu_quant.paper_trading.engine import get_paper_status, record_daily_signal, record_paper_trade
from mu_quant.short_strategy.strategies import compute_short_signals


class TestMUSystemV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_strategy_version("v2.0.0")
        cls.data = load_mu_and_benchmarks()
        cls.mu = cls.data["MU"]
        cls.f = build_features(cls.mu, cls.data.get("SMH"), cls.data.get("SOX"))

    def test_01_data_quality_pass(self) -> None:
        q = validate_mu_quality(self.mu, self.data.get("SPY"))
        self.assertEqual(q["overall"], "PASS")
        self.assertGreater(q["rows"], 10000)

    def test_02_zero_lookahead_causal_replay(self) -> None:
        """Truncating data at historical dates MUST produce identical features, weights, signals, and regimes."""
        w_full = compute_long_weights(self.f, self.cfg)
        s_in_full, s_out_full = compute_short_signals(self.f, self.cfg)
        reg_full = classify_regime(self.f)

        rng = np.random.default_rng(42)
        sample_dates = sorted(rng.choice(self.mu.index[300:], size=8, replace=False))
        for ts in sample_dates:
            ts = pd.Timestamp(ts)
            f_sub = build_features(
                self.mu.loc[:ts],
                self.data["SMH"].loc[:ts] if "SMH" in self.data else None,
                self.data["SOX"].loc[:ts] if "SOX" in self.data else None,
            )
            w_sub = compute_long_weights(f_sub, self.cfg)
            s_in_sub, s_out_sub = compute_short_signals(f_sub, self.cfg)
            reg_sub = classify_regime(f_sub)

            self.assertAlmostEqual(float(w_sub.iloc[-1]), float(w_full.loc[ts]), places=10)
            self.assertEqual(bool(s_in_sub.iloc[-1]), bool(s_in_full.loc[ts]))
            self.assertEqual(bool(s_out_sub.iloc[-1]), bool(s_out_full.loc[ts]))
            self.assertEqual(str(reg_sub.iloc[-1]), str(reg_full.loc[ts]))

    def test_03_strict_t1_open_no_overnight_leak(self) -> None:
        """Signal generated on day 0 close MUST fill at day 1 Open, NOT capture day 0 Close -> day 1 Open gap."""
        dates = pd.bdate_range("2025-01-02", periods=5)
        # Huge +50% overnight gap from day 0 close (100) to day 1 open (150), then flat intraday (150 -> 150)
        df = pd.DataFrame({
            "raw_open": [100.0, 150.0, 150.0, 150.0, 150.0],
            "raw_high": [100.0, 150.0, 150.0, 150.0, 150.0],
            "raw_low": [100.0, 150.0, 150.0, 150.0, 150.0],
            "raw_close": [100.0, 150.0, 150.0, 150.0, 150.0],
            "div": [0.0, 0.0, 0.0, 0.0, 0.0],
        }, index=dates)
        w = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=dates)
        res = run_weight_backtest(df, w, start="2025-01-02", costs=BacktestCosts(0.0, 0.0, 0.0), initial=100_000.0)
        # Since signal at day 0 fills at day 1 open (150) and stays at 150, final equity MUST stay 100,000 (not 150,000!)
        self.assertAlmostEqual(float(res["equity"].iloc[-1]), 100_000.0, places=4)

    def test_04_short_gap_up_stop_and_dividend_in_lieu(self) -> None:
        """Short position must pay ex-dividend cash and fill gap-up stop at worse Open price."""
        dates = pd.bdate_range("2025-02-03", periods=5)
        df = pd.DataFrame({
            "raw_open": [100.0, 100.0, 100.0, 120.0, 120.0],
            "raw_high": [101.0, 101.0, 101.0, 121.0, 121.0],
            "raw_low": [99.0, 99.0, 99.0, 119.0, 119.0],
            "raw_close": [100.0, 100.0, 100.0, 120.0, 120.0],
            "raw_atr14": [2.0, 2.0, 2.0, 2.0, 2.0],
            "rvol21": [0.35, 0.35, 0.35, 0.35, 0.35],
            "div": [0.0, 0.0, 1.0, 0.0, 0.0],  # $1 dividend on day 2 while short
        }, index=dates)
        entries = pd.Series([True, False, False, False, False], index=dates)
        exits = pd.Series([False] * 5, index=dates)
        res = run_short_swing_backtest(
            df, entries, exits, start="2025-02-03",
            params=ShortSwingParams(stop_atr_mult=2.0, trail_atr_mult=None, target_rr=None, target_vol=0.35, max_pos_pct=1.0),
            costs=BacktestCosts(0.0, 0.0, 0.0), initial=100_000.0,
        )
        # Entry day 1 @ 100 (1000 shares short), stop = 104. Day 2 pays $1000 div. Day 3 opens @ 120 -> gap_stop @ 120 (-20k price loss - 1k div = 79k)
        self.assertEqual(len(res["trades"]), 1)
        self.assertEqual(res["trades"][0]["exit_reason"], "gap_stop")
        self.assertAlmostEqual(float(res["trades"][0]["exit_px"]), 120.0, places=4)
        self.assertAlmostEqual(float(res["equity"].iloc[-1]), 79_000.0, places=2)

    def test_05_safety_locks_and_append_only_paper_trading(self) -> None:
        ac = load_active_config()
        self.assertFalse(ac["allow_live_auto_order"])
        tk = simulate_rebalance_order(
            date="2026-11-23", ticker="MU", current_shares=0.0, equity=100_000.0,
            ref_price=700.0, target_weight=0.5, stop_price=650.0, strategy_version="v2.0.0",
        )
        self.assertFalse(tk["is_live_order"])

        with tempfile.TemporaryDirectory() as tmp:
            sp = Path(tmp) / "signals.csv"
            lp = Path(tmp) / "ledger.csv"
            record_daily_signal({"signal_date": "2026-11-22", "strategy_version": "v2.0.0", "target_weight": 0.5}, path=sp)
            record_daily_signal({"signal_date": "2026-11-23", "strategy_version": "v2.0.0", "target_weight": 0.0}, path=sp)
            record_paper_trade({"trade_date": "2026-11-23", "strategy_version": "v2.0.0", "action": "BUY", "shares": 10, "fill_price": 700, "commission": 7, "borrow_fee": 0}, path=lp)
            st = get_paper_status(707.7, ledger_path=lp, signals_path=sp)
            self.assertEqual(st["n_signals_logged"], 2)
            self.assertEqual(st["n_trades_logged"], 1)
            self.assertAlmostEqual(st["shares"], 10.0, places=4)

    def test_06_dashboard_payload_end_to_end(self) -> None:
        payload = build_dashboard_payload("dual_confirm_dvol55", "long_only", "2004-01-01")
        # Ensure strict JSON serializability & required keys
        raw = json.dumps(payload, ensure_ascii=False)
        self.assertGreater(len(raw), 5000)
        for key in ("meta", "market", "quality", "regime", "signals", "risk_plan", "account_risk",
                    "simulated_ticket", "paper_trading", "price_chart", "backtest_comparison", "recent_backtest_trades"):
            self.assertIn(key, payload)
        self.assertTrue(payload["meta"]["live_auto_order_forbidden"])


if __name__ == "__main__":
    unittest.main()
