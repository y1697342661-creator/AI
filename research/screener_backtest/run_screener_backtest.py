"""Historical Backtest & Factor Ablation Study for US Stock Offense+Defense Screener (2021-2026).

Strictly causal:
- Signal computed on T-day close using only data <= T.
- Orders filled at T+1 Open with 15bp (0.15%) per-side commission+slippage.
- Zero leverage hard cap (sum(weights) <= 1.0).
- Compares:
  1. Screener v2.0 Offense+Defense (Weekly / Bi-weekly / Monthly rebalance)
  2. Offense-Only (No Defense Shield, cash when below MA200)
  3. Benchmarks: QQQ Buy & Hold, SPY Buy & Hold
  4. Factor Ablation: removing each of the 5 factors one-by-one to see what truly adds value.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mu_quant.screener.engine import (
    SCREENER_RAW_DIR,
    _compute_rsi,
    _fetch_eastmoney_kline,
    load_screener_config,
)

OUT_DIR = Path(__file__).resolve().parent


def fetch_long_history(cfg: dict[str, Any], beg: str = "20200101") -> dict[str, pd.DataFrame]:
    """Ensure all 44 core assets have history back to 2020-01-01 where available."""
    out: dict[str, pd.DataFrame] = {}

    def _one(item: dict[str, Any]) -> tuple[str, pd.DataFrame]:
        sym = item["symbol"].upper()
        p = SCREENER_RAW_DIR / f"{sym}.csv"
        df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index() if p.exists() else pd.DataFrame()
        if df.empty or df.index[0] > pd.Timestamp("2020-06-01"):
            try:
                fresh = _fetch_eastmoney_kline(item["eastmoney_secid"], beg=beg, timeout=15)
                if not fresh.empty and len(fresh) >= len(df):
                    fresh.to_csv(p)
                    df = fresh
            except Exception:
                pass
        for col in ("Open", "High", "Low", "Close", "Volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return sym, df.dropna(subset=["Open", "High", "Low", "Close"])

    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = [pool.submit(_one, item) for item in cfg["universe"]]
        for fut in as_completed(futs):
            sym, df = fut.result()
            if not df.empty:
                out[sym] = df
    return out


def precompute_panel(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Precompute strictly causal daily indicator matrices aligned to SPY trading dates."""
    cal = data["SPY"].index
    syms = sorted(data.keys())

    opens = pd.DataFrame(index=cal, columns=syms, dtype=float)
    closes = pd.DataFrame(index=cal, columns=syms, dtype=float)
    highs = pd.DataFrame(index=cal, columns=syms, dtype=float)
    vols = pd.DataFrame(index=cal, columns=syms, dtype=float)

    for s in syms:
        df = data[s].reindex(cal)
        opens[s] = df["Open"]
        closes[s] = df["Close"]
        highs[s] = df["High"]
        vols[s] = df["Volume"]

    rets = closes.pct_change(fill_method=None)
    sma20 = closes.rolling(20, min_periods=20).mean()
    sma50 = closes.rolling(50, min_periods=50).mean()
    sma200 = closes.rolling(200, min_periods=200).mean()
    mom20 = closes / closes.shift(20) - 1.0
    mom60 = closes / closes.shift(60) - 1.0

    neg_sq = rets.where(rets < 0.0, 0.0) ** 2
    dvol21 = np.sqrt(neg_sq.rolling(21, min_periods=21).mean() * 252.0)

    high52w = highs.rolling(252, min_periods=60).max()
    dd52w = closes / high52w - 1.0
    vol_ratio = vols.rolling(5, min_periods=5).mean() / vols.rolling(20, min_periods=20).mean()

    rsi14 = pd.DataFrame({s: _compute_rsi(closes[s].dropna(), 14).reindex(cal) for s in syms})
    above200 = closes > sma200
    above50 = closes > sma50
    bull_aligned = (closes > sma20) & (sma20 > sma50) & (sma50 > sma200)
    dvol_cap = (0.55 / dvol21.clip(lower=0.10)).clip(0.0, 1.0)
    risk_adj_mom = mom60 / dvol21.clip(lower=0.15)
    rs60 = mom60.sub(mom60["QQQ"], axis=0)

    return {
        "dates": cal,
        "open": opens,
        "close": closes,
        "mom20": mom20,
        "mom60": mom60,
        "rs60": rs60,
        "dvol21": dvol21,
        "dvol_cap": dvol_cap,
        "risk_adj_mom": risk_adj_mom,
        "dd52w": dd52w,
        "vol_ratio": vol_ratio,
        "rsi14": rsi14,
        "above200": above200,
        "above50": above50,
        "bull_aligned": bull_aligned,
    }


def _rank_dict(d: dict[str, float]) -> dict[str, float]:
    if not d:
        return {}
    s = pd.Series(d, dtype=float)
    r = s.rank(pct=True)
    return {k: float(r[k]) for k in d}


def run_backtest(
    panel: dict[str, Any],
    meta: dict[str, dict[str, Any]],
    start_date: str = "2021-01-04",
    rebalance_every: int = 5,
    use_defense: bool = True,
    weights_cfg: dict[str, float] | None = None,
    use_rsi_guard: bool = True,
    incumbent_buffer: float = 0.0,
    cost_rate: float = 0.0015,
    rebalance_tol: float = 0.05,
) -> dict[str, Any]:
    """Run strictly causal T-close -> T+1-open backtest."""
    wcfg = weights_cfg or {
        "ram": 35.0,
        "rs60": 25.0,
        "mom20": 15.0,
        "dd52w": 15.0,
        "trend": 6.0,
        "vol": 4.0,
    }
    dates = panel["dates"]
    start_idx = int(dates.searchsorted(pd.Timestamp(start_date)))
    offense_syms = [s for s, m in meta.items() if m["role"] == "offense"]
    defense_syms = [s for s, m in meta.items() if m["role"] == "defense"]

    nav = 100_000.0
    cash = nav
    shares: dict[str, float] = {}
    pending_tw: dict[str, float] | None = None
    nav_series: list[tuple[pd.Timestamp, float]] = []
    turnover_total = 0.0
    trades = 0
    regime_days = {"BULL_OFFENSE": 0, "NEUTRAL_BALANCE": 0, "BEAR_DEFENSE": 0}

    for i in range(start_idx, len(dates)):
        dt_i = dates[i]

        # 1. Execute T-1 signal at today's Open
        if pending_tw is not None:
            # Mark portfolio at today's Open
            port_open = cash
            for sym, sh in list(shares.items()):
                px = panel["open"].iat[i, panel["open"].columns.get_loc(sym)]
                if np.isnan(px):
                    px = panel["close"].iat[i - 1, panel["close"].columns.get_loc(sym)]
                port_open += sh * px

            # Current weights at open
            all_syms = set(shares.keys()) | set(pending_tw.keys())
            need_rebal = False
            for sym in all_syms:
                px = panel["open"].iat[i, panel["open"].columns.get_loc(sym)]
                if np.isnan(px) or px <= 0:
                    continue
                cur_w = (shares.get(sym, 0.0) * px) / max(port_open, 1e-6)
                tgt_w = pending_tw.get(sym, 0.0)
                if abs(cur_w - tgt_w) >= rebalance_tol or (tgt_w == 0.0 and cur_w > 0.005):
                    need_rebal = True
                    break

            if need_rebal:
                # Sell first, then buy
                for sym in list(all_syms):
                    px = panel["open"].iat[i, panel["open"].columns.get_loc(sym)]
                    if np.isnan(px) or px <= 0:
                        continue
                    cur_sh = shares.get(sym, 0.0)
                    tgt_val = port_open * pending_tw.get(sym, 0.0)
                    diff_val = tgt_val - cur_sh * px
                    if diff_val < -10.0:
                        sell_sh = min(cur_sh, -diff_val / px)
                        proceeds = sell_sh * px
                        cash += proceeds * (1.0 - cost_rate)
                        shares[sym] = cur_sh - sell_sh
                        if shares[sym] < 1e-6:
                            shares.pop(sym, None)
                        turnover_total += proceeds / max(port_open, 1e-6)
                        trades += 1
                for sym in list(all_syms):
                    px = panel["open"].iat[i, panel["open"].columns.get_loc(sym)]
                    if np.isnan(px) or px <= 0:
                        continue
                    cur_sh = shares.get(sym, 0.0)
                    tgt_val = port_open * pending_tw.get(sym, 0.0)
                    diff_val = tgt_val - cur_sh * px
                    if diff_val > 10.0:
                        spend = min(diff_val, cash / (1.0 + cost_rate))
                        if spend > 10.0:
                            buy_sh = spend / px
                            cash -= spend * (1.0 + cost_rate)
                            shares[sym] = cur_sh + buy_sh
                            turnover_total += spend / max(port_open, 1e-6)
                            trades += 1
            pending_tw = None

        # 2. Mark-to-market at Close
        nav = cash
        for sym, sh in shares.items():
            px = panel["close"].iat[i, panel["close"].columns.get_loc(sym)]
            if np.isnan(px):
                px = panel["close"].iloc[: i + 1][sym].dropna().iloc[-1]
            nav += sh * px
        nav_series.append((dt_i, nav))

        # 3. Generate signal at T-day Close on rebalance days
        if (i - start_idx) % rebalance_every != 0:
            continue

        # Cross-sectional factor extraction at index i
        valid_syms = [s for s in meta if pd.notna(panel["close"][s].iat[i]) and pd.notna(panel["mom60"][s].iat[i])]
        if "QQQ" not in valid_syms or "SPY" not in valid_syms:
            continue

        ram_d = {s: float(panel["risk_adj_mom"][s].iat[i]) for s in valid_syms if pd.notna(panel["risk_adj_mom"][s].iat[i])}
        rs_d = {s: float(panel["rs60"][s].iat[i]) for s in valid_syms if pd.notna(panel["rs60"][s].iat[i])}
        m20_d = {s: float(panel["mom20"][s].iat[i]) for s in valid_syms if pd.notna(panel["mom20"][s].iat[i])}
        dd_d = {s: float(panel["dd52w"][s].iat[i]) for s in valid_syms if pd.notna(panel["dd52w"][s].iat[i])}
        vol_d = {s: float(panel["vol_ratio"][s].iat[i]) for s in valid_syms if pd.notna(panel["vol_ratio"][s].iat[i])}
        lowv_d = {s: -float(panel["dvol21"][s].iat[i]) for s in valid_syms if pd.notna(panel["dvol21"][s].iat[i])}

        r_ram, r_rs, r_m20, r_dd, r_vol, r_lv = (
            _rank_dict(ram_d), _rank_dict(rs_d), _rank_dict(m20_d),
            _rank_dict(dd_d), _rank_dict(vol_d), _rank_dict(lowv_d),
        )

        # Regime evaluation
        spy_up = bool(panel["above200"]["SPY"].iat[i])
        qqq_up = bool(panel["above200"]["QQQ"].iat[i])
        off_valid = [s for s in offense_syms if s in valid_syms]
        off_up = sum(1 for s in off_valid if bool(panel["above200"][s].iat[i]))
        breadth = off_up / max(1, len(off_valid))

        if spy_up and qqq_up and breadth >= 0.50:
            reg = "BULL_OFFENSE"
            off_b, def_b, cash_f = (0.80, 0.20, 0.00) if use_defense else (1.00, 0.00, 0.00)
        elif (not spy_up and not qqq_up) or breadth < 0.25:
            reg = "BEAR_DEFENSE"
            off_b, def_b, cash_f = (0.15, 0.65, 0.20) if use_defense else (0.15, 0.00, 0.85)
        else:
            reg = "NEUTRAL_BALANCE"
            off_b, def_b, cash_f = (0.45, 0.45, 0.10) if use_defense else (0.50, 0.00, 0.50)
        regime_days[reg] += rebalance_every

        # Score offense
        off_scored = []
        for s in off_valid:
            if not bool(panel["above200"][s].iat[i]):
                continue
            m60_v = panel["mom60"][s].iat[i]
            rs_v = panel["rs60"][s].iat[i]
            rsi_v = panel["rsi14"][s].iat[i]
            if pd.isna(m60_v) or m60_v <= 0 or pd.isna(rs_v) or rs_v <= 0:
                continue
            if use_rsi_guard and pd.notna(rsi_v) and rsi_v > 82.0:
                continue
            align = 1.0 if bool(panel["bull_aligned"][s].iat[i]) else (0.5 if bool(panel["above50"][s].iat[i]) else 0.0)
            sc = (
                wcfg["ram"] * r_ram.get(s, 0.0)
                + wcfg["rs60"] * r_rs.get(s, 0.0)
                + wcfg["mom20"] * r_m20.get(s, 0.0)
                + wcfg["dd52w"] * r_dd.get(s, 0.0)
                + wcfg["trend"] * align
                + wcfg["vol"] * r_vol.get(s, 0.0)
            )
            if use_rsi_guard and pd.notna(rsi_v) and rsi_v > 80.0:
                sc -= 8.0
            if incumbent_buffer > 0.0 and shares.get(s, 0.0) > 0.0:
                sc += incumbent_buffer
            off_scored.append((sc, s))
        off_scored.sort(reverse=True)

        picked_off: list[str] = []
        sec_cnt: dict[str, int] = {}
        for _, s in off_scored:
            sec = meta[s]["sector"]
            if sec_cnt.get(sec, 0) >= 2:
                continue
            picked_off.append(s)
            sec_cnt[sec] = sec_cnt.get(sec, 0) + 1
            if len(picked_off) >= 3:
                break

        # Score defense
        picked_def: list[str] = []
        if use_defense and def_b > 0:
            def_scored = []
            for s in defense_syms:
                if s not in valid_syms or not bool(panel["above200"][s].iat[i]):
                    continue
                align = 1.0 if bool(panel["bull_aligned"][s].iat[i]) else (0.5 if bool(panel["above50"][s].iat[i]) else 0.0)
                sc = (
                    35.0 * r_ram.get(s, 0.0)
                    + 25.0 * r_lv.get(s, 0.0)
                    + 20.0 * r_rs.get(s, 0.0)
                    + 10.0 * r_dd.get(s, 0.0)
                    + 10.0 * align
                )
                if incumbent_buffer > 0.0 and shares.get(s, 0.0) > 0.0:
                    sc += incumbent_buffer
                def_scored.append((sc, s))
            def_scored.sort(reverse=True)
            picked_def = [s for _, s in def_scored[:2]]

        tw: dict[str, float] = {}
        if picked_off and off_b > 0:
            bw = off_b / len(picked_off)
            for s in picked_off:
                dcap = float(panel["dvol_cap"][s].iat[i]) if pd.notna(panel["dvol_cap"][s].iat[i]) else 0.5
                tw[s] = min(0.35, bw * max(0.35, dcap))
        if picked_def and def_b > 0:
            bw = def_b / len(picked_def)
            for s in picked_def:
                dcap = float(panel["dvol_cap"][s].iat[i]) if pd.notna(panel["dvol_cap"][s].iat[i]) else 0.8
                tw[s] = min(0.35, bw * max(0.60, dcap))

        max_tot = max(0.0, 1.0 - cash_f)
        tot_w = sum(tw.values())
        if tot_w > max_tot and tot_w > 0:
            tw = {k: v * max_tot / tot_w for k, v in tw.items()}
        pending_tw = tw

    eq = pd.Series({d: v for d, v in nav_series})
    return _calc_metrics(eq, trades=trades, turnover=turnover_total, regime_days=regime_days)


def _calc_metrics(eq: pd.Series, trades: int = 0, turnover: float = 0.0, regime_days: dict[str, int] | None = None) -> dict[str, Any]:
    rets = eq.pct_change().dropna()
    years = max(1e-6, (eq.index[-1] - eq.index[0]).days / 365.25)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1.0
    vol = float(rets.std() * np.sqrt(252.0))
    sharpe = float((rets.mean() * 252.0) / max(vol, 1e-6))
    dd = eq / eq.cummax() - 1.0
    max_dd = float(dd.min())
    calmar = cagr / max(abs(max_dd), 1e-6)

    # Yearly returns
    yearly: dict[str, float] = {}
    for yr, grp in eq.groupby(eq.index.year):
        prev = eq.loc[:grp.index[0]].iloc[-2] if len(eq.loc[:grp.index[0]]) >= 2 else grp.iloc[0]
        yearly[str(yr)] = round(float(grp.iloc[-1] / prev - 1.0) * 100.0, 1)

    return {
        "start": str(eq.index[0].date()),
        "end": str(eq.index[-1].date()),
        "total_ret_pct": round(total_ret * 100.0, 1),
        "cagr_pct": round(cagr * 100.0, 1),
        "max_dd_pct": round(max_dd * 100.0, 1),
        "sharpe": round(sharpe, 2),
        "calmar": round(calmar, 2),
        "ann_vol_pct": round(vol * 100.0, 1),
        "trades": trades,
        "ann_turnover": round(turnover / years, 1),
        "yearly_pct": yearly,
        "regime_days": regime_days or {},
    }


def main() -> None:
    cfg = load_screener_config()
    print("Fetching 2020-2026 history for 44 core assets...")
    data = fetch_long_history(cfg, beg="20200101")
    panel = precompute_panel(data)
    meta = {x["symbol"].upper(): x for x in cfg["universe"]}

    start_date = "2021-01-04"
    cal = panel["dates"]
    s_idx = int(cal.searchsorted(pd.Timestamp(start_date)))

    # Benchmarks
    qqq_eq = 100_000.0 * panel["close"]["QQQ"].iloc[s_idx:] / panel["close"]["QQQ"].iloc[s_idx]
    spy_eq = 100_000.0 * panel["close"]["SPY"].iloc[s_idx:] / panel["close"]["SPY"].iloc[s_idx]

    results: dict[str, Any] = {
        "QQQ_BuyHold": _calc_metrics(qqq_eq.dropna()),
        "SPY_BuyHold": _calc_metrics(spy_eq.dropna()),
        "V2_OffDef_Weekly": run_backtest(panel, meta, start_date=start_date, rebalance_every=5, use_defense=True),
        "V2_OffDef_BiWeekly": run_backtest(panel, meta, start_date=start_date, rebalance_every=10, use_defense=True),
        "V2_OffDef_Monthly": run_backtest(panel, meta, start_date=start_date, rebalance_every=21, use_defense=True),
        "V2_OffenseOnly_Weekly": run_backtest(panel, meta, start_date=start_date, rebalance_every=5, use_defense=False),
    }

    # Factor ablation on BiWeekly (cleanest turnover)
    base_w = {"ram": 35.0, "rs60": 25.0, "mom20": 15.0, "dd52w": 15.0, "trend": 6.0, "vol": 4.0}
    ablation = {}
    for drop_k, label in [
        ("ram", "去掉_风险调整动量"),
        ("rs60", "去掉_跑赢QQQ超额"),
        ("mom20", "去掉_20日短线加速"),
        ("dd52w", "去掉_52周新高位置"),
        ("trend", "去掉_均线多头排列"),
        ("vol", "去掉_放量比"),
    ]:
        w2 = dict(base_w)
        w2[drop_k] = 0.0
        ablation[label] = run_backtest(panel, meta, start_date=start_date, rebalance_every=10, use_defense=True, weights_cfg=w2)
    ablation["去掉_RSI过热扣分"] = run_backtest(panel, meta, start_date=start_date, rebalance_every=10, use_defense=True, use_rsi_guard=False)

    out = {"main": results, "ablation_vs_BiWeekly": ablation}
    (OUT_DIR / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 2021-2026 美股攻守选股策略历史回测成绩单 (严格 T+1开盘成交 · 扣双边0.15%成本 · 零杠杆) ===")
    print(f"{'策略名称':<24} {'总收益':>8} {'年化CAGR':>8} {'最大回撤':>8} {'夏普':>6} {'卡玛':>6} {'2022熊市':>8}")
    print("-" * 78)
    for k, m in results.items():
        y22 = f"{m['yearly_pct'].get('2022', 0):+.1f}%"
        print(f"{k:<24} {m['total_ret_pct']:>+7.1f}% {m['cagr_pct']:>+7.1f}% {m['max_dd_pct']:>7.1f}% {m['sharpe']:>6.2f} {m['calmar']:>6.2f} {y22:>8}")

    base_m = results["V2_OffDef_BiWeekly"]
    print(f"\n=== 因子“体检/消融”实验（对比基准 V2双周调仓: 年化 {base_m['cagr_pct']:+.1f}%, 回撤 {base_m['max_dd_pct']:.1f}%, 夏普 {base_m['sharpe']:.2f}）===")
    for k, m in ablation.items():
        dc = m["cagr_pct"] - base_m["cagr_pct"]
        ds = m["sharpe"] - base_m["sharpe"]
        verdict = "去掉后变差(有用!)" if ds < -0.02 else ("去掉后变好(可删/减权)" if ds > 0.02 else "影响中性")
        print(f"  {k:<18} 年化 {m['cagr_pct']:>+6.1f}% ({dc:>+5.1f}%) | 回撤 {m['max_dd_pct']:>6.1f}% | 夏普 {m['sharpe']:.2f} ({ds:>+.2f}) -> {verdict}")


if __name__ == "__main__":
    main()
