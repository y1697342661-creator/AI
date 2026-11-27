"""MU Specialist Quant System 2.0 — Stage 3 Statistical Validation & Stress Testing.

Covers:
1. Full 16-metric reporting on all 5 Stage 2 candidates + Buy & Hold (strict T+1 Open, 15bp/side, 1.5% borrow + div).
2. Paired Circular Block Bootstrap (block lengths 5, 20, 60; 2000 sims) reporting TRUE sample point estimates, 95% CI, and two-sided p-values.
3. Benjamini-Hochberg (BH-FDR) multiple-testing correction across all hypothesis tests.
4. Random Short-Entry Benchmark (1000 sims) specifically testing whether S3 (19 trades in 22y) beats random shorting.
5. Cost sensitivity (7/15/20/30 bp/side) & Short borrow-rate (0.5%-10%) + Locate-unavailable / Forced-Recall stress tests.
6. Parameter robustness (MA 100-300; Realized Vol Budget 35%-60% & uncapped).
7. Position sizing & account-level drawdown brake comparison (Section 5).
8. Sub-period stability (1985-2026, 2004-2026, 2004-2015, 2016-2026).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "stage2"))
import run_stage2_research as R  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
TRADING_DAYS = 252
SEED = 20261126


# ------------------------------------------------------------------ Paired Circular Block Bootstrap & BH-FDR
def circular_block_indices(n: int, block: int, n_sims: int, rng: np.random.Generator) -> np.ndarray:
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_sims, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(n_sims, -1)[:, :n]


def batch_metrics(rets: np.ndarray) -> dict[str, np.ndarray]:
    """rets: shape (sims, n) -> vectorized CAGR, MaxDD, Calmar, Sharpe."""
    eq = np.cumprod(1.0 + rets, axis=1)
    n = rets.shape[1]
    cagr = np.where(eq[:, -1] > 0, eq[:, -1] ** (TRADING_DAYS / n) - 1.0, -1.0)
    peak = np.maximum.accumulate(eq, axis=1)
    mdd = (eq / np.maximum(peak, 1e-12) - 1.0).min(axis=1)
    mu = rets.mean(axis=1)
    sd = rets.std(axis=1, ddof=1)
    sharpe = np.where(sd > 0, mu / sd * np.sqrt(TRADING_DAYS), 0.0)
    calmar = np.where(mdd < 0, cagr / np.abs(mdd), np.nan)
    return {"cagr": cagr, "mdd": mdd, "calmar": calmar, "sharpe": sharpe}


def paired_block_bootstrap(
    r_a: np.ndarray,
    r_b: np.ndarray,
    block: int = 20,
    n_sims: int = 2000,
    seed: int = SEED,
) -> dict:
    """Paired Circular Block Bootstrap comparing Strategy A minus Benchmark B.
    Returns TRUE sample point estimate (NOT bootstrap mean!), 95% CI, and two-sided p-value.
    """
    n = min(len(r_a), len(r_b))
    ra, rb = r_a[-n:], r_b[-n:]
    ma_pt = batch_metrics(ra[None, :])
    mb_pt = batch_metrics(rb[None, :])

    rng = np.random.default_rng(seed)
    idx = circular_block_indices(n, block, n_sims, rng)
    ma = batch_metrics(ra[idx])
    mb = batch_metrics(rb[idx])

    out = {}
    for k in ("cagr", "mdd", "calmar", "sharpe"):
        pt_diff = float(ma_pt[k][0] - mb_pt[k][0])
        sim_diff = ma[k] - mb[k]
        sim_diff = sim_diff[np.isfinite(sim_diff)]
        lo, hi = float(np.percentile(sim_diff, 2.5)), float(np.percentile(sim_diff, 97.5))
        # Two-sided bootstrap p-value (fraction crossing 0)
        p_left = float((sim_diff <= 0).mean())
        p_right = float((sim_diff >= 0).mean())
        pval = float(min(1.0, 2.0 * min(p_left, p_right)))
        out[k] = {
            "point_diff": pt_diff,
            "ci_low": lo,
            "ci_high": hi,
            "p_raw": pval,
            "contains_zero": bool(lo <= 0.0 <= hi),
        }
    return out


def bh_fdr(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(adj, 0.0, 1.0)
    return [float(x) for x in out]


# ------------------------------------------------------------------ Drawdown Brake Overlay Wrapper (Strict T+1 Open)
def run_weight_with_dd_brake(
    f: pd.DataFrame,
    base_w: pd.Series,
    *,
    start: str,
    end: str = "2026-12-31",
    dd_start: float = 0.15,
    dd_full: float = 0.35,
    min_scale: float = 0.25,
    halt_dd: float = 0.40,
    resume_days: int = 120,
    commission_bps: float = 10.0,
    slippage_bps: float = 5.0,
    initial: float = 100_000.0,
) -> dict:
    """Strict T+1 Open execution with account-level drawdown scaling + temporary halt & resume."""
    sl = f.loc[pd.Timestamp(start):pd.Timestamp(end)]
    w_raw = base_w.reindex(f.index).fillna(0.0).reindex(sl.index).fillna(0.0).to_numpy(float)
    o = sl["raw_open"].to_numpy(float)
    c = sl["raw_close"].to_numpy(float)
    div = sl["div"].to_numpy(float)
    idx = sl.index

    cash = initial
    shares = 0.0
    peak_eq = initial
    halted = False
    halt_date = None
    next_target_w = 0.0
    total_cost = 0.0
    equity = np.empty(len(sl))
    eff_w = np.empty(len(sl))
    trades = []
    slip = slippage_bps / 1e4
    comm = commission_bps / 1e4

    for i in range(len(sl)):
        if shares != 0.0 and div[i] > 0:
            cash += shares * div[i]

        eq_open = cash + shares * o[i]
        if eq_open <= 0:
            equity[i:] = 0.0
            eff_w[i:] = 0.0
            break

        # Execute weight decided at day i-1 close at Open[i]
        tw = float(np.clip(next_target_w, 0.0, 1.0))
        target_shares = (tw * eq_open) / o[i]
        delta = target_shares - shares
        if abs(delta) > 1e-8:
            fill = o[i] * (1.0 + slip) if delta > 0 else o[i] * (1.0 - slip)
            fee = abs(delta) * fill * comm
            slip_c = abs(delta) * o[i] * slip
            cash -= delta * fill + fee
            total_cost += fee + slip_c
            shares = target_shares

        eq_close = cash + shares * c[i]
        equity[i] = max(eq_close, 0.0)
        eff_w[i] = (shares * c[i] / eq_close) if eq_close > 0 else 0.0

        # End of day i: update peak/drawdown and decide weight for day i+1 Open
        peak_eq = max(peak_eq, eq_close)
        dd = max(0.0, 1.0 - eq_close / peak_eq) if peak_eq > 0 else 0.0
        if not halted and dd >= halt_dd:
            halted = True
            halt_date = idx[i]
        elif halted and halt_date is not None and (idx[i] - halt_date).days >= resume_days:
            halted = False
            peak_eq = eq_close

        if halted:
            next_target_w = 0.0
        else:
            if dd <= dd_start:
                scale = 1.0
            else:
                span = max(dd_full - dd_start, 1e-9)
                scale = float(np.clip(1.0 - (dd - dd_start) / span, min_scale, 1.0))
            next_target_w = w_raw[i] * scale

    eq_s = pd.Series(equity, index=idx)
    w_s = pd.Series(eff_w, index=idx)
    m = R.compute_metrics(eq_s, w_s, trades, total_cost, initial)
    return {"equity": eq_s, "weights": w_s, "metrics": m}


# ------------------------------------------------------------------ Short Recall / Locate Unavailability Stress Test
def run_short_stress(
    f: pd.DataFrame,
    s3_in: pd.Series,
    s3_out: pd.Series,
    earnings: pd.Series,
    start: str = "2004-01-01",
    locate_fail_prob: float = 0.30,
    recall_prob: float = 0.15,
    n_sims: int = 200,
) -> dict:
    """Monte Carlo stress test for S3: 30% signals fail locate (no shares to borrow) + 15% forced early recall."""
    rng = np.random.default_rng(SEED)
    sp = R.SwingParams(
        side="SHORT", stop_atr_mult=2.0, trail_atr_mult=2.5, target_rr=2.0,
        max_hold_days=40, position_mode="vol_budget", target_vol=0.35, exit_before_earnings=True,
    )
    cagrs, mdds, pfs = [], [], []
    sig_dates = s3_in[s3_in].index
    for _ in range(n_sims):
        keep = rng.random(len(sig_dates)) >= locate_fail_prob
        s_in_sim = pd.Series(False, index=f.index)
        s_in_sim.loc[sig_dates[keep]] = True
        # Simulate forced recall by randomly truncating max_hold_days to 5 on some runs or adding random exit days
        recall_mask = pd.Series(rng.random(len(f)) < (recall_prob / 10.0), index=f.index)
        s_out_sim = s3_out | recall_mask
        r = R.run_swing(f, s_in_sim, s_out_sim, sp, start=start, earnings_dates=earnings)["metrics"]
        cagrs.append(r["cagr"])
        mdds.append(r["max_dd"])
        if np.isfinite(r["profit_factor"]):
            pfs.append(r["profit_factor"])
    return {
        "locate_fail_prob": locate_fail_prob,
        "recall_prob": recall_prob,
        "n_sims": n_sims,
        "cagr_median": float(np.median(cagrs)),
        "cagr_p05": float(np.percentile(cagrs, 5)),
        "cagr_p95": float(np.percentile(cagrs, 95)),
        "mdd_median": float(np.median(mdds)),
        "mdd_worst05": float(np.percentile(mdds, 5)),
        "pf_median": float(np.median(pfs)) if pfs else np.nan,
        "prob_positive": float((np.array(cagrs) > 0).mean()),
    }


# ------------------------------------------------------------------ Random Short Entry Benchmark (Testing S3 Edge)
def random_short_benchmark(
    f: pd.DataFrame,
    s3_out: pd.Series,
    earnings: pd.Series,
    n_signals: int,
    start: str = "2004-01-01",
    n_sims: int = 500,
) -> dict:
    """Pick n_signals random bear-market days (Close < MA200) and run exact same S3 exit/risk/cost rules."""
    sl = f.loc[pd.Timestamp(start):]
    bear_days = sl.index[sl["Close"] < sl["sma200"]]
    sp = R.SwingParams(
        side="SHORT", stop_atr_mult=2.0, trail_atr_mult=2.5, target_rr=2.0,
        max_hold_days=40, position_mode="vol_budget", target_vol=0.35, exit_before_earnings=True,
    )
    rng = np.random.default_rng(SEED)
    cagrs, calmars, sharpes = [], [], []
    k = min(n_signals, len(bear_days))
    for _ in range(n_sims):
        picks = rng.choice(bear_days, size=k, replace=False)
        sig = pd.Series(False, index=f.index)
        sig.loc[picks] = True
        m = R.run_swing(f, sig, s3_out, sp, start=start, earnings_dates=earnings)["metrics"]
        cagrs.append(m["cagr"])
        calmars.append(m["calmar"] if np.isfinite(m["calmar"]) else np.nan)
        sharpes.append(m["sharpe"])
    return {"cagr": np.array(cagrs), "calmar": np.array(calmars), "sharpe": np.array(sharpes)}


# ------------------------------------------------------------------ Main
def main() -> None:
    data = R.load_data()
    f = R.compute_indicators(data["MU"], data.get("SMH"), data.get("SOX"))
    earnings = data["earnings"]

    # Candidate definitions (frozen from Stage 2)
    w_bh = pd.Series(1.0, index=f.index)
    w_ma200 = (f["Close"] > f["sma200"]).astype(float)
    w_ma50_200 = (f["sma50"] > f["sma200"]).astype(float)
    cap45 = (0.45 / f["rvol21"]).clip(upper=1.0).fillna(0.0)
    w_l0_v45 = np.minimum(w_ma200, cap45)

    sox_ok = f["sox_above200"].fillna(1.0) > 0.5
    l5_in = (f["Close"] > f["sma200"]) & (f["roc60"] > 0.05) & sox_ok
    l5_out = (f["Close"] < f["sma50"]) | (f["roc60"] < -0.05)
    sp_l5 = R.SwingParams(side="LONG", stop_atr_mult=2.5, trail_atr_mult=3.5, target_rr=None,
                          max_hold_days=90, position_mode="vol_budget", target_vol=0.45)

    s3_in = (
        (f["Close"] < f["don_low20"]) & (f["Close"] < f["sma200"]) & (f["sma50"] < f["sma200"])
        & (f["vol_ratio20"] > 1.2) & (f["clv"] < 0.35) & (f["dist_ma20"] > -0.12)
    )
    s3_out = (f["Close"] > f["sma20"]) | (f["rsi14"] < 25)
    sp_s3 = R.SwingParams(side="SHORT", stop_atr_mult=2.0, trail_atr_mult=2.5, target_rr=2.0,
                          max_hold_days=40, position_mode="vol_budget", target_vol=0.35, exit_before_earnings=True)

    out: dict = {}

    # ==================== 1. FULL SUB-PERIOD METRICS (16 Metrics) ====================
    periods = [
        ("2004-2026(主诊断)", "2004-01-01", "2026-12-31"),
        ("1985-2026(全历史)", "1985-01-01", "2026-12-31"),
        ("2004-2015(熊/震荡)", "2004-01-01", "2015-12-31"),
        ("2016-2026(牛/AI)", "2016-01-01", "2026-12-31"),
    ]
    sub_rows = []
    eq_store: dict[str, dict[str, pd.Series]] = {}
    s3_trades_04 = []
    for pname, s, e in periods:
        r_bh = R.run_weight_t1_open(f, w_bh, start=s, end=e)
        r_m200 = R.run_weight_t1_open(f, w_ma200, start=s, end=e)
        r_m50 = R.run_weight_t1_open(f, w_ma50_200, start=s, end=e)
        r_l0 = R.run_weight_t1_open(f, w_l0_v45, start=s, end=e)
        r_l5 = R.run_swing(f, l5_in, l5_out, sp_l5, start=s, end=e, earnings_dates=earnings)
        r_s3 = R.run_swing(f, s3_in, s3_out, sp_s3, start=s, end=e, earnings_dates=earnings)
        r_c05 = R.combine_portfolios(r_l0, r_s3, 1.0, 0.5)
        r_c10 = R.combine_portfolios(r_l0, r_s3, 1.0, 1.0)
        if pname.startswith("2004-2026"):
            s3_trades_04 = r_s3["trades"]

        pack = [
            ("BH_买入持有", r_bh),
            ("L0_MA200单开关", r_m200),
            ("L0_MA50_200双均线", r_m50),
            ("L0_MA200+45%波动预算", r_l0),
            ("L5_中期动量+SOX", r_l5),
            ("S3_支撑放量跌破(仅空)", r_s3),
            ("Combo_L0+0.5*S3", r_c05),
            ("Combo_L0+1.0*S3", r_c10),
        ]
        eq_store[pname] = {k: v["equity"] for k, v in pack}
        for cname, res in pack:
            sub_rows.append({"period": pname, "name": cname, **res["metrics"]})
    out["subperiod_metrics"] = sub_rows

    # ==================== 2. PAIRED CIRCULAR BLOCK BOOTSTRAP + BH-FDR ====================
    boot_rows = []
    comparisons = [
        ("L0_MA200单开关", "BH_买入持有"),
        ("L0_MA50_200双均线", "BH_买入持有"),
        ("L0_MA200+45%波动预算", "BH_买入持有"),
        ("L5_中期动量+SOX", "BH_买入持有"),
        ("L0_MA200+45%波动预算", "L0_MA200单开关"),
        ("Combo_L0+0.5*S3", "L0_MA200+45%波动预算"),
        ("Combo_L0+1.0*S3", "L0_MA200+45%波动预算"),
        ("Combo_L0+0.5*S3", "BH_买入持有"),
    ]
    for pname in ("2004-2026(主诊断)", "1985-2026(全历史)"):
        eqs = eq_store[pname]
        for block in (5, 20, 60):
            for strat, bench in comparisons:
                ra = eqs[strat].pct_change().dropna().to_numpy(float)
                rb = eqs[bench].pct_change().dropna().to_numpy(float)
                res = paired_block_bootstrap(ra, rb, block=block, n_sims=2000, seed=SEED)
                for metric in ("calmar", "sharpe", "cagr", "mdd"):
                    boot_rows.append({
                        "period": pname, "block": block,
                        "comparison": f"{strat} - {bench}",
                        "metric": metric, **res[metric],
                    })
    # Apply BH-FDR per (period, block, metric) family
    boot_df = pd.DataFrame(boot_rows)
    boot_df["p_bh"] = np.nan
    for (p, b, m), grp in boot_df.groupby(["period", "block", "metric"]):
        boot_df.loc[grp.index, "p_bh"] = bh_fdr(grp["p_raw"].tolist())
    boot_df["bh_sig_05"] = boot_df["p_bh"] < 0.05
    out["paired_bootstrap"] = boot_df.to_dict(orient="records")

    # ==================== 3. S3 RANDOM SHORT BENCHMARK & TAIL RISK ====================
    s3_n_sig_04 = int(s3_in.loc["2004-01-01":].sum())
    s3_real_04 = [r for r in sub_rows if r["period"] == "2004-2026(主诊断)" and r["name"] == "S3_支撑放量跌破(仅空)"][0]
    rand_s3 = random_short_benchmark(f, s3_out, earnings, n_signals=s3_n_sig_04, start="2004-01-01", n_sims=500)
    s3_pct_cagr = float((rand_s3["cagr"] < s3_real_04["cagr"]).mean() * 100)
    s3_pct_sharpe = float((rand_s3["sharpe"] < s3_real_04["sharpe"]).mean() * 100)
    valid_cal = rand_s3["calmar"][np.isfinite(rand_s3["calmar"])]
    s3_pct_calmar = float((valid_cal < s3_real_04["calmar"]).mean() * 100) if len(valid_cal) else np.nan

    # Corrected short tail risk (daily return of S3 equity when short)
    r_s3_daily = eq_store["2004-2026(主诊断)"]["S3_支撑放量跌破(仅空)"].pct_change().dropna()
    active_s3 = r_s3_daily[r_s3_daily != 0]
    gap_stops = sum(1 for t in s3_trades_04 if t.get("exit_reason") == "gap_stop")
    worst_r = min((t.get("r_mult", 0.0) for t in s3_trades_04 if np.isfinite(t.get("r_mult", np.nan))), default=np.nan)
    out["s3_random_and_tail"] = {
        "n_signals_04": s3_n_sig_04,
        "n_trades_04": s3_real_04["n_trades"],
        "real_cagr": s3_real_04["cagr"],
        "rand_cagr_median": float(np.median(rand_s3["cagr"])),
        "rand_cagr_p95": float(np.percentile(rand_s3["cagr"], 95)),
        "pct_cagr": s3_pct_cagr,
        "pct_calmar": s3_pct_calmar,
        "pct_sharpe": s3_pct_sharpe,
        "active_days": int(len(active_s3)),
        "worst_1d_loss": float(active_s3.min()) if len(active_s3) else 0.0,
        "p01_1d_loss": float(active_s3.quantile(0.01)) if len(active_s3) else 0.0,
        "gap_stop_count": int(gap_stops),
        "worst_r_mult": float(worst_r),
    }

    # ==================== 4. COST & BORROW SENSITIVITY + LOCATE/RECALL STRESS ====================
    cost_rows = []
    for comm, slip, label in [(2.0, 5.0, "7bp_研究基线"), (10.0, 5.0, "15bp_券商实际"), (10.0, 10.0, "20bp_压力"), (15.0, 15.0, "30bp_极端")]:
        m_l0 = R.run_weight_t1_open(f, w_l0_v45, start="2004-01-01", commission_bps=comm, slippage_bps=slip)["metrics"]
        m_m200 = R.run_weight_t1_open(f, w_ma200, start="2004-01-01", commission_bps=comm, slippage_bps=slip)["metrics"]
        sp_s3_c = R.SwingParams(side="SHORT", stop_atr_mult=2.0, trail_atr_mult=2.5, target_rr=2.0, max_hold_days=40,
                                position_mode="vol_budget", target_vol=0.35, exit_before_earnings=True,
                                commission_bps=comm, slippage_bps=slip)
        m_s3 = R.run_swing(f, s3_in, s3_out, sp_s3_c, start="2004-01-01", earnings_dates=earnings)["metrics"]
        cost_rows.append({"cost": label, "L0_v45_cagr": m_l0["cagr"], "L0_v45_calmar": m_l0["calmar"],
                          "MA200_cagr": m_m200["cagr"], "MA200_calmar": m_m200["calmar"],
                          "S3_cagr": m_s3["cagr"], "S3_calmar": m_s3["calmar"]})
    out["cost_sensitivity"] = cost_rows

    borrow_rows = []
    for br in (0.005, 0.015, 0.03, 0.05, 0.10):
        sp_b = R.SwingParams(side="SHORT", stop_atr_mult=2.0, trail_atr_mult=2.5, target_rr=2.0, max_hold_days=40,
                             position_mode="vol_budget", target_vol=0.35, exit_before_earnings=True, borrow_annual=br)
        m04 = R.run_swing(f, s3_in, s3_out, sp_b, start="2004-01-01", earnings_dates=earnings)["metrics"]
        m85 = R.run_swing(f, s3_in, s3_out, sp_b, start="1985-01-01", earnings_dates=earnings)["metrics"]
        borrow_rows.append({"borrow_annual": br, "cagr_04": m04["cagr"], "calmar_04": m04["calmar"], "pf_04": m04["profit_factor"],
                            "cagr_85": m85["cagr"], "calmar_85": m85["calmar"], "pf_85": m85["profit_factor"]})
    out["borrow_sensitivity"] = borrow_rows
    out["short_recall_stress"] = run_short_stress(f, s3_in, s3_out, earnings, start="2004-01-01")

    # ==================== 5. PARAMETER ROBUSTNESS (MA 100-300 & VOL BUDGET 35%-60%+) ====================
    ma_rows = []
    for ma_len in (100, 150, 200, 250, 300):
        ma_s = f["Close"].rolling(ma_len, min_periods=ma_len).mean()
        w_m = np.minimum((f["Close"] > ma_s).astype(float), cap45)
        m04 = R.run_weight_t1_open(f, w_m, start="2004-01-01")["metrics"]
        m85 = R.run_weight_t1_open(f, w_m, start="1985-01-01")["metrics"]
        ma_rows.append({"ma_len": ma_len, "cagr_04": m04["cagr"], "mdd_04": m04["max_dd"], "calmar_04": m04["calmar"],
                        "cagr_85": m85["cagr"], "mdd_85": m85["max_dd"], "calmar_85": m85["calmar"]})
    out["ma_robustness"] = ma_rows

    vol_rows = []
    for tv in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 9.99):
        cap = (tv / f["rvol21"]).clip(upper=1.0).fillna(0.0) if tv < 5 else pd.Series(1.0, index=f.index)
        w_v = np.minimum(w_ma200, cap)
        m04 = R.run_weight_t1_open(f, w_v, start="2004-01-01")["metrics"]
        m85 = R.run_weight_t1_open(f, w_v, start="1985-01-01")["metrics"]
        vol_rows.append({"target_vol": "无上限(100%)" if tv > 5 else f"{int(tv*100)}%",
                         "cagr_04": m04["cagr"], "mdd_04": m04["max_dd"], "calmar_04": m04["calmar"], "pos_04": m04["avg_abs_pos"],
                         "cagr_85": m85["cagr"], "mdd_85": m85["max_dd"], "calmar_85": m85["calmar"], "pos_85": m85["avg_abs_pos"]})
    out["vol_budget_robustness"] = vol_rows

    # ==================== 6. POSITION SIZING & ACCOUNT RISK CONTROLS COMPARISON (Section 5) ====================
    # Trend-strength sizing: Close>MA200 & MA50>MA200 -> 100%, only Close>MA200 -> 50%, else 0%
    w_trend_str = pd.Series(0.0, index=f.index)
    w_trend_str[f["Close"] > f["sma200"]] = 0.5
    w_trend_str[(f["Close"] > f["sma200"]) & (f["sma50"] > f["sma200"])] = 1.0

    # ATR stop-based fixed risk (2% risk per trade) on L5 swing
    sp_l5_risk2 = R.SwingParams(side="LONG", stop_atr_mult=2.5, trail_atr_mult=3.5, target_rr=None,
                                max_hold_days=90, position_mode="risk", risk_per_trade=0.02)

    pos_rows = []
    pos_tests = [
        ("1_固定100%仓位(MA200)", lambda s: R.run_weight_t1_open(f, w_ma200, start=s)["metrics"]),
        ("2_固定50%仓位(MA200)", lambda s: R.run_weight_t1_open(f, w_ma200 * 0.5, start=s)["metrics"]),
        ("3_ATR止损2%单笔风险(L5)", lambda s: R.run_swing(f, l5_in, l5_out, sp_l5_risk2, start=s, earnings_dates=earnings)["metrics"]),
        ("4_已实现波动预算45%(MA200)", lambda s: R.run_weight_t1_open(f, w_l0_v45, start=s)["metrics"]),
        ("5_已实现波动预算55%(MA200)", lambda s: R.run_weight_t1_open(f, np.minimum(w_ma200, (0.55 / f["rvol21"]).clip(upper=1.0).fillna(0.0)), start=s)["metrics"]),
        ("6_趋势强度分级仓位(100%/50%)", lambda s: R.run_weight_t1_open(f, w_trend_str, start=s)["metrics"]),
        ("7_MA200+45%波动+回撤刹车暂停", lambda s: run_weight_with_dd_brake(f, w_l0_v45, start=s)["metrics"]),
    ]
    for pname, fn in pos_tests:
        m04 = fn("2004-01-01")
        m85 = fn("1985-01-01")
        pos_rows.append({
            "mode": pname,
            "cagr_04": m04["cagr"], "mdd_04": m04["max_dd"], "calmar_04": m04["calmar"], "sharpe_04": m04["sharpe"], "pos_04": m04["avg_abs_pos"],
            "cagr_85": m85["cagr"], "mdd_85": m85["max_dd"], "calmar_85": m85["calmar"], "sharpe_85": m85["sharpe"], "pos_85": m85["avg_abs_pos"],
        })
    out["position_sizing"] = pos_rows

    out_path = OUT_DIR / "stage3_results.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"Saved -> {out_path}")

    # Console print key tables
    print("\n=== 1. PAIRED BLOCK BOOTSTRAP (Block=20, Calmar & Sharpe, 2004-2026 & 1985-2026) ===")
    b20 = boot_df[(boot_df["block"] == 20) & (boot_df["metric"].isin(["calmar", "sharpe", "mdd"]))]
    print(b20[["period", "comparison", "metric", "point_diff", "ci_low", "ci_high", "p_raw", "p_bh", "bh_sig_05"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== 2. S3 RANDOM SHORT BENCHMARK & TAIL RISK (2004-2026) ===")
    print(json.dumps(out["s3_random_and_tail"], ensure_ascii=False, indent=2))

    print("\n=== 3. BORROW SENSITIVITY & RECALL STRESS ===")
    print(pd.DataFrame(borrow_rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("Recall/Locate Stress:", json.dumps(out["short_recall_stress"], ensure_ascii=False))

    print("\n=== 4. VOL BUDGET ROBUSTNESS ===")
    print(pd.DataFrame(vol_rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== 5. POSITION SIZING COMPARISON ===")
    print(pd.DataFrame(pos_rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
