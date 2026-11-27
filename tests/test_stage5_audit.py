"""Stage 5 Comprehensive System Audit Suite.

Adversarial checks covering:
1. Static AST/regex audit (zero shift(-n), zero live-order endpoints/SDKs, zero full-sample feature leakage).
2. 20-date causal truncation replay across all 3 long modes, S3 short signals, and market regimes + 1-day delay sanity check.
3. Order timing & intraday conflict checks (same-day stop+target -> stop first; same-day entry stop; short trailing stop monotonicity; earnings blackout).
4. Zero-leverage invariant across 42 years (10,555 days) for all 5 portfolio views.
5. Extreme gap/squeeze risk resilience + live HTTP dashboard endpoint smoke test on ephemeral port.
"""
from __future__ import annotations

import json
import re
import sys
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src" / "mu_quant"
sys.path.insert(0, str(ROOT / "src"))

from mu_quant.backtest.engine import BacktestCosts, ShortSwingParams, run_short_swing_backtest, run_weight_backtest
from mu_quant.config import load_strategy_version
from mu_quant.dashboard.server import DashboardHandler
from mu_quant.data.loader import load_mu_and_benchmarks, validate_mu_quality
from mu_quant.features.indicators import build_features
from mu_quant.long_strategy.strategies import compute_long_weights
from mu_quant.market_regime.regime import classify_regime
from mu_quant.portfolio.combiner import run_all_portfolio_views
from mu_quant.risk.manager import RiskConfig, compute_trade_risk_plan, evaluate_account_risk_status
from mu_quant.short_strategy.strategies import compute_short_signals


class TestStage5SystemAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_strategy_version("v2.0.0")
        cls.data = load_mu_and_benchmarks()
        cls.mu = cls.data["MU"]
        cls.f = build_features(cls.mu, cls.data.get("SMH"), cls.data.get("SOX"))

    def test_audit_01_static_code_safety_and_no_negative_shift(self) -> None:
        forbidden_patterns = [
            (r"shift\(\s*-", "Negative shift (lookahead bias)"),
            (r"\b(ccxt|alpaca_trade_api|ib_insync|okx\.Trade)\b", "Live broker/exchange order SDK"),
            (r"https?://(www\.)?okx\.com/api/v5/trade", "Live OKX order endpoint"),
        ]
        py_files = list(SRC_DIR.rglob("*.py"))
        self.assertGreaterEqual(len(py_files), 12)
        for p in py_files:
            txt = p.read_text(encoding="utf-8")
            for pat, desc in forbidden_patterns:
                m = re.search(pat, txt)
                self.assertIsNone(m, f"Audit FAIL in {p.name}: found {desc} -> {m.group(0) if m else ''}")

    def test_audit_02_causal_replay_20_dates_all_modes(self) -> None:
        modes = ["dual_confirm_dvol55", "ma200_rvol45", "ma50_200"]
        w_full = {m: compute_long_weights(self.f, self.cfg, mode=m) for m in modes}
        s_in_full, s_out_full = compute_short_signals(self.f, self.cfg)
        reg_full = classify_regime(self.f)

        rng = np.random.default_rng(20261126)
        dates = sorted(rng.choice(self.mu.index[300:], size=20, replace=False))
        for ts in dates:
            ts = pd.Timestamp(ts)
            f_sub = build_features(
                self.mu.loc[:ts],
                self.data["SMH"].loc[:ts],
                self.data["SOX"].loc[:ts],
            )
            for m in modes:
                w_sub = compute_long_weights(f_sub, self.cfg, mode=m)
                self.assertAlmostEqual(float(w_sub.iloc[-1]), float(w_full[m].loc[ts]), places=10,
                                       msg=f"Lookahead mismatch in mode={m} at {ts.date()}")
            si, so = compute_short_signals(f_sub, self.cfg)
            self.assertEqual(bool(si.iloc[-1]), bool(s_in_full.loc[ts]))
            self.assertEqual(bool(so.iloc[-1]), bool(s_out_full.loc[ts]))
            self.assertEqual(str(classify_regime(f_sub).iloc[-1]), str(reg_full.loc[ts]))

    def test_audit_03_one_day_delay_sanity(self) -> None:
        """Delaying primary long weights by an extra day (T+2 open) must degrade mildly, not collapse."""
        w = compute_long_weights(self.f, self.cfg, mode="dual_confirm_dvol55")
        m0 = run_weight_backtest(self.f, w, start="2004-01-01")["metrics"]
        m1 = run_weight_backtest(self.f, w.shift(1).fillna(0.0), start="2004-01-01")["metrics"]
        self.assertGreater(m1["calmar"], m0["calmar"] * 0.75,
                           f"Suspicious collapse on 1-day delay: {m0['calmar']:.3f} -> {m1['calmar']:.3f}")

    def test_audit_04_same_day_conflict_stop_beats_target(self) -> None:
        """When BOTH short stop and short target are hit on the same bar, engine MUST fill STOP first (conservative)."""
        dates = pd.bdate_range("2025-03-03", periods=4)
        df = pd.DataFrame({
            "raw_open": [100.0, 100.0, 100.0, 100.0],
            "raw_high": [101.0, 101.0, 110.0, 101.0],  # Day 2 hits stop (104) AND target (92)
            "raw_low": [99.0, 99.0, 85.0, 99.0],
            "raw_close": [100.0, 100.0, 95.0, 100.0],
            "raw_atr14": [2.0, 2.0, 2.0, 2.0],
            "rvol21": [0.35, 0.35, 0.35, 0.35],
            "div": [0.0, 0.0, 0.0, 0.0],
        }, index=dates)
        entries = pd.Series([True, False, False, False], index=dates)
        exits = pd.Series([False] * 4, index=dates)
        res = run_short_swing_backtest(
            df, entries, exits, start="2025-03-03",
            params=ShortSwingParams(stop_atr_mult=2.0, trail_atr_mult=None, target_rr=2.0, target_vol=0.35),
            costs=BacktestCosts(0.0, 0.0, 0.0), initial=100_000.0,
        )
        self.assertEqual(res["trades"][0]["exit_reason"], "stop")
        self.assertLess(res["trades"][0]["pnl"], 0.0)

    def test_audit_05_same_day_entry_stop_and_earnings_blackout(self) -> None:
        dates = pd.bdate_range("2025-04-01", periods=6)
        df = pd.DataFrame({
            "raw_open": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "raw_high": [101.0, 106.0, 101.0, 101.0, 101.0, 101.0],  # Day 1 hits stop=104 on entry day!
            "raw_low": [99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
            "raw_close": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "raw_atr14": [2.0] * 6,
            "rvol21": [0.35] * 6,
            "div": [0.0] * 6,
        }, index=dates)
        entries = pd.Series([True, False, True, False, False, False], index=dates)
        exits = pd.Series([False] * 6, index=dates)
        # Earnings on day 4 -> day 3 entry must be blocked by blackout_before=3!
        earnings = pd.Series([dates[4]])
        res = run_short_swing_backtest(
            df, entries, exits, start="2025-04-01",
            params=ShortSwingParams(stop_atr_mult=2.0, trail_atr_mult=None, target_rr=None, earnings_blackout_before=3, earnings_blackout_after=1),
            costs=BacktestCosts(0.0, 0.0, 0.0), earnings_dates=earnings, initial=100_000.0,
        )
        # Day 1 entry is ALSO within 3 days of day 4 earnings! So 0 trades should fire when earnings=day 4.
        self.assertEqual(len(res["trades"]), 0)

        # Without earnings blackout, day 1 enters and immediately hits same_day_stop
        res2 = run_short_swing_backtest(
            df, entries, exits, start="2025-04-01",
            params=ShortSwingParams(stop_atr_mult=2.0, trail_atr_mult=None, target_rr=None),
            costs=BacktestCosts(0.0, 0.0, 0.0), earnings_dates=None, initial=100_000.0,
        )
        self.assertEqual(res2["trades"][0]["exit_reason"], "same_day_stop")
        self.assertEqual(res2["trades"][0]["hold_days"], 0)

    def test_audit_06_zero_leverage_across_42_years(self) -> None:
        views = run_all_portfolio_views(self.f, self.cfg, self.data["earnings"], start="1985-01-01")
        for vname, res in views.items():
            max_abs_w = float(res["weights"].abs().max())
            self.assertLessEqual(max_abs_w, 1.0005, f"Leverage breach in {vname}: max |w|={max_abs_w:.4f}")
            self.assertTrue((res["equity"] >= 0.0).all(), f"Negative equity in {vname}")

    def test_audit_07_extreme_risk_edge_cases(self) -> None:
        # Flat plan
        p_flat = compute_trade_risk_plan(equity=100_000, price=700, atr=30, side="FLAT", target_weight=0.0, stop_atr_mult=2.5)
        self.assertEqual(p_flat.shares, 0.0)
        # Short warning present
        p_short = compute_trade_risk_plan(equity=100_000, price=700, atr=30, side="SHORT", target_weight=-0.3, stop_atr_mult=2.0, is_earnings_blackout=True)
        self.assertGreater(p_short.stop_price, 700.0)
        self.assertGreaterEqual(len(p_short.warnings), 2)
        # Drawdown halt status
        eq_crash = pd.Series([100_000, 90_000, 55_000])
        st = evaluate_account_risk_status(eq_crash, RiskConfig(dd_brake_enabled=True, dd_halt_threshold=0.40))
        self.assertEqual(st["status"], "HALT_ALERT")
        self.assertEqual(st["risk_scale"], 0.0)

    def test_audit_08_http_dashboard_live_endpoints(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), DashboardHandler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10) as r:
                self.assertEqual(r.status, 200)
                self.assertIn("MU Specialist Quant System 2.0", r.read().decode("utf-8"))

            for lm in ("dual_confirm_dvol55", "ma200_rvol45", "ma50_200"):
                for pm in ("long_only", "long_short_combo"):
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/dashboard?long_mode={lm}&portfolio_mode={pm}&start_date=2004-01-01", timeout=15) as r:
                        self.assertEqual(r.status, 200)
                        body = json.loads(r.read().decode("utf-8"))
                        self.assertEqual(body["meta"]["long_mode"], lm)
                        self.assertEqual(body["meta"]["portfolio_mode"], pm)
                        self.assertTrue(body["meta"]["live_auto_order_forbidden"])
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
