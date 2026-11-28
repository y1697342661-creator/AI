"""Unit tests for the US Stock Offense+Defense Full-Market Screener (v2.0.0)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mu_quant.screener.engine import (
    _score_and_tag,
    apply_preset_filter,
    build_offense_defense_allocation,
    compute_stock_factors,
    evaluate_market_regime,
    load_screener_config,
)


def _make_synthetic_ohlcv(n: int = 260, drift: float = 0.002, vol: float = 0.015, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=n)
    rets = drift + vol * rng.standard_normal(n)
    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = close * (1.0 + 0.001 * rng.standard_normal(n))
    high = np.maximum(open_, close) * (1.0 + np.abs(0.005 * rng.standard_normal(n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(0.005 * rng.standard_normal(n)))
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


class TestUsStockScreenerV2(unittest.TestCase):
    def test_config_has_offense_defense_and_benchmarks(self) -> None:
        cfg = load_screener_config()
        self.assertEqual(cfg["benchmark"], "QQQ")
        roles = {x["role"] for x in cfg["universe"]}
        self.assertTrue({"offense", "defense", "benchmark"}.issubset(roles))
        def_syms = {x["symbol"] for x in cfg["universe"] if x["role"] == "defense"}
        self.assertTrue({"USMV", "XLV", "WMT", "COST"}.issubset(def_syms))
        self.assertTrue(def_syms.isdisjoint({"GLD", "TLT", "IEF", "SGOV"}))

    def test_ma200_hard_penalty_and_ranking(self) -> None:
        cfg = load_screener_config()
        rules = cfg["rules"]
        df_bull = _make_synthetic_ohlcv(drift=0.0035, vol=0.012, seed=1)
        df_qqq = _make_synthetic_ohlcv(drift=0.0008, vol=0.010, seed=2)
        df_bear = _make_synthetic_ohlcv(drift=-0.0030, vol=0.020, seed=3)

        r_bull = compute_stock_factors(df_bull, {"symbol": "NVDA", "role": "offense", "sector": "AI芯片", "theme": "GPU"}, rules)
        r_qqq = compute_stock_factors(df_qqq, {"symbol": "QQQ", "role": "benchmark", "sector": "ETF", "theme": "纳指"}, rules)
        r_bear = compute_stock_factors(df_bear, {"symbol": "BEAR", "role": "offense", "sector": "测试", "theme": "下跌"}, rules)

        scored = _score_and_tag([r_bull, r_qqq, r_bear], rules, benchmark="QQQ")
        by_sym = {x["symbol"]: x for x in scored}

        self.assertTrue(by_sym["NVDA"]["above_ma200"])
        self.assertFalse(by_sym["BEAR"]["above_ma200"])
        self.assertLessEqual(by_sym["BEAR"]["composite_score"], 45.0)
        self.assertEqual(by_sym["BEAR"]["grade"], "D")

        core = apply_preset_filter(scored, preset="core_momentum")
        self.assertIn("NVDA", [x["symbol"] for x in core])
        self.assertNotIn("BEAR", [x["symbol"] for x in core])

    def test_bear_regime_shifts_budget_to_defense_and_zero_leverage(self) -> None:
        cfg = load_screener_config()
        rules = cfg["rules"]
        records = [
            compute_stock_factors(_make_synthetic_ohlcv(drift=-0.002, vol=0.015, seed=10), {"symbol": "SPY", "role": "benchmark", "sector": "ETF", "theme": "标普"}, rules),
            compute_stock_factors(_make_synthetic_ohlcv(drift=-0.0025, vol=0.018, seed=11), {"symbol": "QQQ", "role": "benchmark", "sector": "ETF", "theme": "纳指"}, rules),
            compute_stock_factors(_make_synthetic_ohlcv(drift=-0.003, vol=0.025, seed=12), {"symbol": "NVDA", "role": "offense", "sector": "AI芯片", "theme": "GPU"}, rules),
            compute_stock_factors(_make_synthetic_ohlcv(drift=0.0012, vol=0.006, seed=13), {"symbol": "COST", "role": "defense", "sector": "防御蓝筹", "theme": "开市客"}, rules),
            compute_stock_factors(_make_synthetic_ohlcv(drift=0.0005, vol=0.004, seed=14), {"symbol": "USMV", "role": "defense", "sector": "防御低波", "theme": "低波ETF"}, rules),
        ]
        scored = _score_and_tag(records, rules, benchmark="QQQ")
        regime = evaluate_market_regime(scored, cfg["allocation"])
        self.assertEqual(regime["regime"], "BEAR_DEFENSE")

        plan = build_offense_defense_allocation(scored, regime, cfg["allocation"], "2026-11-27", cfg["version"])
        self.assertLessEqual(plan["total_invested_weight"], 1.0 - regime["cash_floor"] + 1e-6)
        self.assertGreater(plan["defense_weight"], plan["offense_weight"])
        self.assertGreaterEqual(plan["cash_weight"], 0.20 - 1e-4)

    def test_causality_future_rows_do_not_change_past_factors(self) -> None:
        cfg = load_screener_config()
        df = _make_synthetic_ohlcv(n=280, drift=0.002, vol=0.015, seed=7)
        item = {"symbol": "MU", "role": "offense", "sector": "AI芯片", "theme": "存储"}
        f_t = compute_stock_factors(df.iloc[:250], item, cfg["rules"])
        df_mut = df.copy()
        df_mut.iloc[250:] = df_mut.iloc[250:] * 5.0
        f_t2 = compute_stock_factors(df_mut.iloc[:250], item, cfg["rules"])
        self.assertEqual(f_t["mom60"], f_t2["mom60"])
        self.assertEqual(f_t["dvol21"], f_t2["dvol21"])
        self.assertEqual(f_t["sma200"], f_t2["sma200"])


if __name__ == "__main__":
    unittest.main()
