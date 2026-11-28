"""Automated Unit Tests & Quantitative Audit Suite for BTC Perpetual Futures System (btc_quant)."""
from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from btc_quant.backtest.engine import PerpBacktestConfig, run_perp_backtest
from btc_quant.data.loader import load_btc_datasets, validate_btc_data_quality
from btc_quant.execution_simulator import okx_demo_swap
from btc_quant.features.indicators import build_1d_features, build_4h_features
from btc_quant.risk.manager import BtcRiskConfig, compute_contract_risk_plan
from btc_quant.strategies.perp_strategies import generate_strategy_signals


def _make_synthetic_4h(n: int = 30, start_px: float = 50_000.0, funding: float = 0.0) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
    px = np.full(n, start_px, dtype=float)
    return pd.DataFrame(
        {
            "open": px,
            "high": px * 1.002,
            "low": px * 0.998,
            "close": px,
            "vol": 100.0,
            "vol_ccy": 100.0,
            "vol_quote": px * 100.0,
            "funding_rate": np.full(n, funding, dtype=float),
        },
        index=idx,
    )


class TestBtcPerpQuantSystem(unittest.TestCase):
    def test_01_data_quality_and_zero_gaps(self) -> None:
        ds = load_btc_datasets()
        qa = validate_btc_data_quality(ds)
        self.assertTrue(qa["ok"])
        self.assertGreater(qa["bars_4h"], 10_000)
        self.assertGreater(qa["bars_1d"], 2_000)
        self.assertEqual(qa["gaps_4h"], 0)
        self.assertEqual(qa["gaps_1d"], 0)

    def test_02_zero_lookahead_prefix_invariance(self) -> None:
        """Signals at bar T must be identical whether future bars T+1..N exist or not."""
        ds = load_btc_datasets()
        df_1d_full = build_1d_features(ds["df_1d"])
        df_4h_full = build_4h_features(ds["df_4h"], df_1d_full)
        tgt_full, _, _ = generate_strategy_signals(df_4h_full, mode="dual_horizon_ensemble")

        cut_idx = len(ds["df_4h"]) - 250
        cut_ts = ds["df_4h"].index[cut_idx]
        # Keep only 1D bars whose close (index + 1D) <= cut_ts + 4H
        df_1d_cut = ds["df_1d"].loc[ds["df_1d"].index + pd.Timedelta(days=1) <= cut_ts + pd.Timedelta(hours=4)].copy()
        df_4h_cut = ds["df_4h"].iloc[: cut_idx + 1].copy()

        df_4h_cut_f = build_4h_features(df_4h_cut, build_1d_features(df_1d_cut))
        tgt_cut, _, _ = generate_strategy_signals(df_4h_cut_f, mode="dual_horizon_ensemble")

        self.assertAlmostEqual(float(tgt_full.loc[cut_ts]), float(tgt_cut.iloc[-1]), places=10)
        self.assertAlmostEqual(float(df_4h_full.loc[cut_ts, "d1_close"]), float(df_4h_cut_f.iloc[-1]["d1_close"]), places=8)

    def test_03_funding_rate_cashflow_sign(self) -> None:
        """Positive funding rate must debit LONG and credit SHORT."""
        df = _make_synthetic_4h(n=20, start_px=50_000.0, funding=0.001)  # 0.1% per bar
        cfg = PerpBacktestConfig(initial_capital=100_000.0, taker_fee_bps=0.0, slippage_bps=0.0, use_funding=True)
        res_long = run_perp_backtest(df, pd.Series(1.0, index=df.index), cfg=cfg)
        res_short = run_perp_backtest(df, pd.Series(-1.0, index=df.index), cfg=cfg)

        self.assertLess(res_long["metrics"]["total_funding_usd"], -1000.0)
        self.assertGreater(res_short["metrics"]["total_funding_usd"], 1000.0)
        self.assertLess(res_long["equity"].iloc[-1], 100_000.0)
        self.assertGreater(res_short["equity"].iloc[-1], 100_000.0)

    def test_04_liquidation_engine_triggers_on_crash(self) -> None:
        """A 5x long hit by a -25% intrabar crash must trigger LIQUIDATION with non-negative equity."""
        df = _make_synthetic_4h(n=10, start_px=50_000.0)
        df.iloc[4, df.columns.get_loc("low")] = 37_000.0  # -26% crash
        df.iloc[4, df.columns.get_loc("close")] = 38_000.0
        cfg = PerpBacktestConfig(
            initial_capital=100_000.0,
            max_long_leverage=5.0,
            taker_fee_bps=0.0,
            slippage_bps=0.0,
            use_funding=False,
        )
        res = run_perp_backtest(df, pd.Series(5.0, index=df.index), cfg=cfg)
        self.assertEqual(res["metrics"]["liquidations"], 1)
        self.assertGreaterEqual(float(res["equity"].min()), 0.0)
        self.assertEqual(res["trades"].iloc[0]["exit_reason"], "LIQUIDATION")

    def test_05_intrabar_stop_loss_execution(self) -> None:
        """Stop loss must trigger intrabar on low breach."""
        df = _make_synthetic_4h(n=10, start_px=50_000.0)
        df.iloc[5, df.columns.get_loc("low")] = 46_000.0  # -8% wick
        cfg = PerpBacktestConfig(initial_capital=100_000.0, taker_fee_bps=5.0, slippage_bps=3.0, use_funding=False)
        res = run_perp_backtest(
            df,
            pd.Series(1.0, index=df.index),
            stop_distance_pct=pd.Series(0.05, index=df.index),
            cfg=cfg,
        )
        self.assertGreaterEqual(len(res["trades"]), 1)
        self.assertEqual(res["trades"].iloc[0]["exit_reason"], "STOP_LOSS")
        self.assertAlmostEqual(res["trades"].iloc[0]["exit_px"], 47_500.0, places=1)

    def test_06_risk_manager_caps_and_circuit_breaker(self) -> None:
        cfg = BtcRiskConfig(max_long_leverage=2.0, max_short_leverage=1.2, hard_cap_leverage=3.0, dd_halt_threshold=0.32)
        # Over-leverage request gets clamped
        plan = compute_contract_risk_plan(100_000.0, 60_000.0, target_leverage=10.0, stop_distance_pct=0.01, cfg=cfg)
        self.assertLessEqual(plan["approved_leverage"], 2.0)
        self.assertLessEqual(plan["isolated_margin_leverage"], 3)
        # Deep drawdown triggers full halt
        halted = compute_contract_risk_plan(100_000.0, 60_000.0, target_leverage=1.5, stop_distance_pct=0.05, current_drawdown=0.35, cfg=cfg)
        self.assertEqual(halted["risk_state"], "HALTED_MAX_DD")
        self.assertEqual(halted["approved_leverage"], 0.0)
        self.assertEqual(halted["okx_contracts"], 0.0)

    def test_07_okx_demo_hard_safety_invariants(self) -> None:
        """OKX swap channel MUST hardcode simulated trading, whitelisted swaps, isolated mode, <=3x cap."""
        self.assertEqual(okx_demo_swap.ALLOWED_INST_ID, "BTC-USDT-SWAP")
        self.assertEqual(set(okx_demo_swap.ALLOWED_INST_IDS), {"BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"})
        self.assertEqual(okx_demo_swap.ALLOWED_TD_MODE, "isolated")
        self.assertLessEqual(okx_demo_swap.MAX_ALLOWED_LEVERAGE, 4)
        src = inspect.getsource(okx_demo_swap._demo_private_request)
        self.assertIn('"x-simulated-trading": "1"', src)
        with self.assertRaises(ValueError):
            okx_demo_swap.get_demo_swap_account_state("DOGE-USDT-SWAP")


if __name__ == "__main__":
    unittest.main()
