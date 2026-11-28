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

    def test_incumbent_buffer_prevents_small_score_churn(self) -> None:
        cfg = load_screener_config()
        regime = {
            "regime": "BULL_OFFENSE",
            "label": "牛市进攻",
            "offense_budget": 0.80,
            "defense_budget": 0.20,
            "cash_floor": 0.0,
        }
        def _mk(sym: str, score: float, above200: bool = True, sec: str = "S1") -> dict:
            return {
                "symbol": sym, "role": "offense", "sector": sec, "theme": sym,
                "above_ma200": above200, "mom60": 0.15, "rs60_vs_qqq": 0.10,
                "composite_score": score, "dvol21": 0.25, "dvol_cap": 1.0,
                "stop_price": 90.0, "discovered": False,
            }

        # ABBV is held at 88.4; challenger NEW1 at 88.5 (+0.1) MUST NOT kick ABBV out
        scored = [
            _mk("AAA", 95.0, sec="S1"),
            _mk("BBB", 92.0, sec="S2"),
            _mk("NEW1", 88.5, sec="S3"),
            _mk("ABBV", 88.4, sec="S4"),
        ]
        plan = build_offense_defense_allocation(
            scored, regime, cfg["allocation"], "2026-11-28", cfg["version"],
            prev_offense={"AAA", "BBB", "ABBV"}, prev_defense=set(),
        )
        self.assertIn("ABBV", plan["selected_symbols"])
        self.assertNotIn("NEW1", plan["selected_symbols"])

        # If challenger NEW2 beats ABBV by > 5.0 pts (94.0 vs 88.4), ABBV gets replaced
        scored2 = [
            _mk("AAA", 95.0, sec="S1"),
            _mk("BBB", 92.0, sec="S2"),
            _mk("NEW2", 94.0, sec="S3"),
            _mk("ABBV", 88.4, sec="S4"),
        ]
        plan2 = build_offense_defense_allocation(
            scored2, regime, cfg["allocation"], "2026-11-28", cfg["version"],
            prev_offense={"AAA", "BBB", "ABBV"}, prev_defense=set(),
        )
        self.assertIn("NEW2", plan2["selected_symbols"])
        self.assertNotIn("ABBV", plan2["selected_symbols"])

        # If ABBV breaks below MA200, protection is immediately revoked
        scored3 = [
            _mk("AAA", 95.0, sec="S1"),
            _mk("BBB", 92.0, sec="S2"),
            _mk("NEW1", 88.5, sec="S3"),
            _mk("ABBV", 88.4, above200=False, sec="S4"),
        ]
        plan3 = build_offense_defense_allocation(
            scored3, regime, cfg["allocation"], "2026-11-28", cfg["version"],
            prev_offense={"AAA", "BBB", "ABBV"}, prev_defense=set(),
        )
        self.assertNotIn("ABBV", plan3["selected_symbols"])
        self.assertIn("NEW1", plan3["selected_symbols"])

    def test_biweekly_hold_and_daily_emergency_guard(self) -> None:
        cfg = load_screener_config()
        bull_reg = {"regime": "BULL_OFFENSE", "label": "牛市进攻", "offense_budget": 0.80, "defense_budget": 0.20, "cash_floor": 0.0}
        bear_reg = {"regime": "BEAR_DEFENSE", "label": "熊市防守", "offense_budget": 0.15, "defense_budget": 0.65, "cash_floor": 0.20}

        def _mk(sym: str, score: float, above200: bool = True, role: str = "offense", sec: str = "S1") -> dict:
            return {
                "symbol": sym, "role": role, "sector": sec, "theme": sym,
                "above_ma200": above200, "mom60": 0.15, "rs60_vs_qqq": 0.10,
                "composite_score": score, "dvol21": 0.25, "dvol_cap": 1.0,
                "stop_price": 90.0, "discovered": False,
            }

        prev_alloc = {
            "signal_date": "2026-11-20",
            "last_rebalance_date": "2026-11-20",
            "regime": "BULL_OFFENSE",
            "positions": [
                {"symbol": "AAA", "sleeve": "OFFENSE", "weight": 0.40, "score": 85.0},
                {"symbol": "COST", "sleeve": "DEFENSE", "weight": 0.20, "score": 90.0},
            ],
        }

        # 1) 5 days later (<14d), all healthy & same regime -> HOLD even if ZZZ has 99 pts
        scored_hold = [_mk("AAA", 82.0, True), _mk("ZZZ", 99.0, True, sec="S2"), _mk("COST", 90.0, True, role="defense", sec="D1")]
        p_hold = build_offense_defense_allocation(scored_hold, bull_reg, cfg["allocation"], "2026-11-25", cfg["version"], prev_allocation=prev_alloc)
        self.assertEqual(p_hold["rebalance_action"], "HOLD")
        self.assertEqual(p_hold["selected_symbols"], ["AAA", "COST"])
        self.assertEqual(p_hold["last_rebalance_date"], "2026-11-20")

        # 2) 5 days later, AAA breaks below MA200 -> EMERGENCY_MA200_STOP triggers immediately!
        scored_brk = [_mk("AAA", 40.0, False), _mk("ZZZ", 99.0, True, sec="S2"), _mk("COST", 90.0, True, role="defense", sec="D1")]
        p_brk = build_offense_defense_allocation(scored_brk, bull_reg, cfg["allocation"], "2026-11-25", cfg["version"], prev_allocation=prev_alloc)
        self.assertEqual(p_brk["rebalance_action"], "REBALANCE")
        self.assertIn("EMERGENCY_MA200_STOP", p_brk["rebalance_reason"])
        self.assertNotIn("AAA", p_brk["selected_symbols"])

        # 3) 5 days later, regime downgrades BULL -> BEAR -> EMERGENCY_REGIME_DOWNGRADE triggers!
        p_reg = build_offense_defense_allocation(scored_hold, bear_reg, cfg["allocation"], "2026-11-25", cfg["version"], prev_allocation=prev_alloc)
        self.assertEqual(p_reg["rebalance_action"], "REBALANCE")
        self.assertIn("EMERGENCY_REGIME_DOWNGRADE", p_reg["rebalance_reason"])
        self.assertGreaterEqual(p_reg["cash_weight"], 0.20 - 1e-4)

        # 4) 15 days later (>=14d) -> normal BIWEEKLY_SCHEDULED rebalance!
        p_due = build_offense_defense_allocation(scored_hold, bull_reg, cfg["allocation"], "2026-12-05", cfg["version"], prev_allocation=prev_alloc)
        self.assertEqual(p_due["rebalance_action"], "REBALANCE")
        self.assertIn("BIWEEKLY_SCHEDULED", p_due["rebalance_reason"])
        self.assertIn("ZZZ", p_due["selected_symbols"])


if __name__ == "__main__":
    unittest.main()
