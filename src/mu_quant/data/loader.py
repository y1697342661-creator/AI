"""Data loading, automatic multi-source refresh (Yahoo + Stooq fallback), and quality validation.

Strict rules:
- Research & backtesting use NASDAQ: MU common stock ONLY (never perpetuals or OKX stock tokens).
- Preserve unadjusted split-adjusted OHLCV + Adj Close + Dividends + Splits.
- Never forward-fill or impute missing MU prices.
- Fallback merge refuses to overwrite if overlapping prices deviate > 0.5%.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mu_quant.config import DATA_RAW, TRADING_DAYS
from mu_quant.logs.logger import get_logger

log = get_logger("mu_quant.data")

UNIVERSE_FILES: dict[str, tuple[str, str, str | None]] = {
    # key: (yahoo_symbol, local_filename, eastmoney_secid)
    "MU": ("MU", "MU.csv", "105.MU"),
    "SMH": ("SMH", "bench_SMH.csv", "105.SMH"),
    "SOX": ("^SOX", "bench_SOX.csv", "100.SOX"),
    "SPY": ("SPY", "bench_SPY.csv", "107.SPY"),
    "QQQ": ("QQQ", "bench_QQQ.csv", "105.QQQ"),
    "NVDA": ("NVDA", "bench_NVDA.csv", "105.NVDA"),
}
REFRESH_META_PATH: Path = DATA_RAW / "_last_refresh.json"


def load_raw_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Market data file missing: {path}")
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date").sort_index()
    df.index.name = "Date"
    return df


def load_mu_and_benchmarks() -> dict[str, Any]:
    mu = load_raw_csv(DATA_RAW / "MU.csv")
    out: dict[str, Any] = {"MU": mu}
    for sym in ("SMH", "SOX", "SPY", "QQQ", "NVDA"):
        p = DATA_RAW / f"bench_{sym}.csv"
        if p.exists():
            out[sym] = load_raw_csv(p)
    ep = DATA_RAW / "earnings_dates.csv"
    if ep.exists():
        edf = pd.read_csv(ep)
        dcol = [c for c in edf.columns if "date" in c.lower()][0]
        out["earnings"] = pd.to_datetime(edf[dcol]).dropna().sort_values()
    else:
        out["earnings"] = pd.Series(dtype="datetime64[ns]")
    return out


def validate_mu_quality(mu: pd.DataFrame, spy: pd.DataFrame | None = None) -> dict[str, Any]:
    """Run core data integrity checks on MU daily OHLCV."""
    checks = []
    dup = int(mu.index.duplicated().sum())
    mono = bool(mu.index.is_monotonic_increasing)
    checks.append({"item": "索引唯一且递增", "status": "PASS" if (dup == 0 and mono) else "FAIL", "detail": f"{len(mu)}行, 重复={dup}, 递增={mono}"})

    req = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
    nulls = int(mu[req].isna().sum().sum())
    checks.append({"item": "关键字段无缺失", "status": "PASS" if nulls == 0 else "FAIL", "detail": f"缺失数={nulls}"})

    bad_hi = int((mu["High"] < mu[["Open", "Low", "Close"]].max(axis=1) - 1e-6).sum())
    bad_lo = int((mu["Low"] > mu[["Open", "High", "Close"]].min(axis=1) + 1e-6).sum())
    non_pos = int((mu[["Open", "High", "Low", "Close"]] <= 0).any(axis=1).sum())
    checks.append({"item": "OHLC内部一致性", "status": "PASS" if (bad_hi == 0 and bad_lo == 0 and non_pos == 0) else "FAIL",
                   "detail": f"High异常={bad_hi}, Low异常={bad_lo}, 非正价={non_pos}"})

    zvol = int((mu["Volume"] <= 0).sum())
    checks.append({"item": "成交量合理性", "status": "PASS" if zvol == 0 else "WARN", "detail": f"零成交量天数={zvol}"})

    af = (mu["Adj Close"] / mu["Close"]).dropna()
    bad_drop = int((af.diff() < -1e-4).sum())
    checks.append({"item": "复权因子单调性", "status": "PASS" if bad_drop == 0 else "WARN",
                   "detail": f"范围 {af.min():.4f}->{af.max():.4f}, 异常下降={bad_drop}"})

    if spy is not None and not spy.empty:
        overlap_start = max(mu.index.min(), spy.index.min())
        spy_dates = spy.loc[overlap_start:].index
        missing_days = len(spy_dates.difference(mu.index))
        checks.append({"item": "交易日历对齐(SPY)", "status": "PASS" if missing_days == 0 else "WARN",
                       "detail": f"重叠期缺失={missing_days}天"})

    n_div = int((mu.get("Dividends", pd.Series(0, index=mu.index)) > 0).sum())
    n_spl = int((mu.get("Splits", pd.Series(0, index=mu.index)) > 0).sum())
    checks.append({"item": "公司行动记录", "status": "INFO", "detail": f"分红={n_div}次, 拆股={n_spl}次"})

    ret = mu["Adj Close"].pct_change().dropna()
    big_moves = int((ret.abs() > 0.20).sum())
    checks.append({"item": "极端单日波动(>20%)", "status": "INFO", "detail": f"历史共{big_moves}天, 年化波动={ret.std()*np.sqrt(TRADING_DAYS):.1%}"})

    overall = "FAIL" if any(c["status"] == "FAIL" for c in checks) else "PASS"
    return {
        "overall": overall,
        "rows": int(len(mu)),
        "first_date": str(mu.index.min().date()),
        "last_date": str(mu.index.max().date()),
        "checks": checks,
    }


def refresh_yahoo_daily(ticker: str = "MU", dest: Path | None = None) -> dict[str, Any]:
    """Incremental safe refresh from Yahoo Finance chart API v8 (uses fast range=3mo if local CSV exists)."""
    out_path = dest or (DATA_RAW / f"{ticker}.csv")
    use_range = "3mo" if out_path.exists() else "max"
    enc_ticker = urllib.parse.quote(ticker, safe="")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{enc_ticker}?range={use_range}&interval=1d&events=div%2Csplits"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        res = payload["chart"]["result"][0]
        ts = res["timestamp"]
        q = res["indicators"]["quote"][0]
        adj = res["indicators"]["adjclose"][0]["adjclose"]
        dates = pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York").normalize().tz_localize(None)
        df = pd.DataFrame({
            "Open": q["open"], "High": q["high"], "Low": q["low"],
            "Close": q["close"], "Volume": q["volume"], "Adj Close": adj,
        }, index=dates)
        df.index.name = "Date"
        df = df[~df.index.duplicated(keep="last")].dropna(subset=["Open", "High", "Low", "Close", "Adj Close"]).sort_index()
        df["Dividends"] = 0.0
        df["Splits"] = 0.0
        events = res.get("events") or {}
        for _, dv in (events.get("dividends") or {}).items():
            d = pd.to_datetime(dv["date"], unit="s", utc=True).tz_convert("America/New_York").normalize().tz_localize(None)
            if d in df.index:
                df.loc[d, "Dividends"] = float(dv["amount"])
        for _, sp in (events.get("splits") or {}).items():
            d = pd.to_datetime(sp["date"], unit="s", utc=True).tz_convert("America/New_York").normalize().tz_localize(None)
            if d in df.index:
                df.loc[d, "Splits"] = float(sp["numerator"]) / float(sp["denominator"])

        if out_path.exists():
            existing = load_raw_csv(out_path)
            overlap = existing.index.intersection(df.index)[-5:]
            if len(overlap) >= 2:
                rel_diff = (existing.loc[overlap, "Close"] - df.loc[overlap, "Close"]).abs() / existing.loc[overlap, "Close"]
                if float(rel_diff.max()) > 0.005:
                    return {"ok": False, "source": "yahoo", "error": f"Overlap mismatch {rel_diff.max():.2%} > 0.5%, refused merge"}
            newer = df.loc[df.index > existing.index.max()]
            if newer.empty:
                return {"ok": True, "source": "yahoo_noop", "rows": len(existing), "added": 0, "last_date": str(existing.index.max().date())}
            cols = [c for c in existing.columns if c in newer.columns]
            merged = pd.concat([existing, newer[cols]]).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]
            merged.to_csv(out_path, date_format="%Y-%m-%d")
            log.info("Appended %d newer rows via Yahoo to %s (total=%d, last=%s)", len(newer), out_path.name, len(merged), merged.index.max().date())
            return {"ok": True, "source": "yahoo", "rows": len(merged), "added": len(newer), "last_date": str(merged.index.max().date())}

        df.to_csv(out_path, date_format="%Y-%m-%d")
        log.info("Downloaded %s via Yahoo: %d rows (last=%s)", ticker, len(df), df.index.max().date())
        return {"ok": True, "source": "yahoo", "rows": len(df), "added": len(df), "last_date": str(df.index.max().date())}
    except Exception as exc:  # noqa: BLE001
        log.warning("Yahoo refresh failed for %s: %s", ticker, exc)
        return {"ok": False, "source": "yahoo", "error": str(exc)}


def _refresh_eastmoney_append(secid: str, dest: Path) -> dict[str, Any]:
    """China-mainland direct fallback: Eastmoney daily kline safe-append (5-day overlap <= 0.5%)."""
    if not dest.exists():
        return {"ok": False, "source": "eastmoney", "error": "Base CSV missing for safe append"}
    url = (
        f"https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={urllib.parse.quote(secid, safe='')}"
        "&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56&klt=101&fqt=0&lmt=30&end=20500101"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            d = json.loads(resp.read().decode("utf-8"))
        kl = (d.get("data") or {}).get("klines") or []
        if not kl:
            return {"ok": False, "source": "eastmoney", "error": "Empty Eastmoney klines"}
        rows = []
        for line in kl:
            dt_s, o, c, h, l, v = line.split(",")[:6]
            rows.append({"Date": pd.Timestamp(dt_s), "Open": float(o), "High": float(h), "Low": float(l),
                         "Close": float(c), "Volume": float(v), "Adj Close": float(c), "Dividends": 0.0, "Splits": 0.0})
        edf = pd.DataFrame(rows).set_index("Date").sort_index()

        base = load_raw_csv(dest)
        overlap = base.index.intersection(edf.index)[-5:]
        if len(overlap) >= 2:
            rel_diff = (base.loc[overlap, "Close"] - edf.loc[overlap, "Close"]).abs() / base.loc[overlap, "Close"]
            if float(rel_diff.max()) > 0.005:
                return {"ok": False, "source": "eastmoney", "error": f"Overlap mismatch {rel_diff.max():.2%} > 0.5%, refused merge"}

        newer = edf.loc[edf.index > base.index.max()]
        if newer.empty:
            return {"ok": True, "source": "eastmoney_noop", "rows": len(base), "added": 0, "last_date": str(base.index.max().date())}
        cols = [c for c in base.columns if c in newer.columns]
        merged = pd.concat([base, newer[cols]]).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        merged.to_csv(dest, date_format="%Y-%m-%d")
        log.info("Appended %d newer bars via Eastmoney to %s (last=%s)", len(newer), dest.name, merged.index.max().date())
        return {"ok": True, "source": "eastmoney", "rows": len(merged), "added": len(newer), "last_date": str(merged.index.max().date())}
    except Exception as exc:  # noqa: BLE001
        log.warning("Eastmoney fallback failed for %s: %s", secid, exc)
        return {"ok": False, "source": "eastmoney", "error": str(exc)}


def refresh_all_market_data(force: bool = False, min_interval_hours: float = 4.0) -> dict[str, Any]:
    """Refresh MU + all 5 benchmark CSVs (Yahoo 3mo fast incremental -> Eastmoney China-direct fallback)."""
    now = dt.datetime.now()
    if not force and REFRESH_META_PATH.exists():
        try:
            meta = json.loads(REFRESH_META_PATH.read_text(encoding="utf-8"))
            last_ts = dt.datetime.fromisoformat(meta.get("refreshed_at", "2000-01-01T00:00:00"))
            if (now - last_ts).total_seconds() < min_interval_hours * 3600:
                return {"ok": True, "skipped": True, "reason": f"距上次自动更新不足 {min_interval_hours} 小时", "last_meta": meta}
        except Exception:  # noqa: BLE001
            pass

    results: dict[str, Any] = {}
    any_ok = False
    for key, (y_sym, fname, em_secid) in UNIVERSE_FILES.items():
        dest = DATA_RAW / fname
        # Prefer China-mainland direct Eastmoney first if base CSV exists, then Yahoo 3mo fallback
        r: dict[str, Any] = {"ok": False}
        if dest.exists() and em_secid:
            r = _refresh_eastmoney_append(em_secid, dest=dest)
        if not r.get("ok"):
            r = refresh_yahoo_daily(y_sym, dest=dest)
        results[key] = r
        any_ok = any_ok or bool(r.get("ok"))

    summary = {
        "ok": any_ok,
        "skipped": False,
        "refreshed_at": now.isoformat(timespec="seconds"),
        "symbols": results,
    }
    REFRESH_META_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _fetch_tencent_live_mu() -> dict[str, Any]:
    """China-mainland direct real-time US stock quote from Tencent Finance (NO VPN required)."""
    url = "https://qt.gtimg.cn/q=usMU"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=4) as resp:
            txt = resp.read().decode("gbk", errors="replace")
        parts = txt.split("~")
        if len(parts) < 35:
            return {"ok": False, "source": "tencent", "error": "Unexpected Tencent payload"}
        px = float(parts[3])
        prev_c = float(parts[4]) if parts[4] else px
        chg = ((px / prev_c) - 1.0) if prev_c > 0 else 0.0
        quote_time = parts[30] if len(parts) > 30 else ""
        return {
            "ok": True,
            "source": "tencent_direct",
            "regular_price": round(prev_c, 4),
            "live_price": round(px, 4),
            "chg_vs_regular": round(chg, 6),
            "live_time_bj": f"{quote_time} (美东)",
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "source": "tencent", "error": str(exc)}


def fetch_live_mu_quote() -> dict[str, Any]:
    """Fetch live MU quote: tries China-direct Tencent Finance first, then Yahoo pre/post-market if reachable."""
    tc = _fetch_tencent_live_mu()
    # Try Yahoo briefly (3s timeout) only to enrich pre/post-market if network allows
    url = "https://query1.finance.yahoo.com/v8/finance/chart/MU?range=1d&interval=1m&includePrePost=true"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            d = json.loads(resp.read().decode("utf-8"))
        res = d["chart"]["result"][0]
        meta = res.get("meta") or {}
        ts = res.get("timestamp") or []
        closes = (res.get("indicators", {}).get("quote") or [{}])[0].get("close") or []
        valid = [(t, c) for t, c in zip(ts, closes) if c is not None]
        reg_px = float(meta.get("regularMarketPrice") or 0.0)
        if valid:
            last_t, last_c = valid[-1]
            t_bj = dt.datetime.fromtimestamp(last_t, tz=dt.timezone(dt.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
            live_px = float(last_c)
        else:
            t_bj = ""
            live_px = reg_px
        chg_vs_reg = ((live_px / reg_px) - 1.0) if (reg_px > 0 and live_px > 0) else 0.0
        return {
            "ok": True,
            "source": "yahoo_prepost",
            "regular_price": round(reg_px, 4),
            "live_price": round(live_px, 4),
            "chg_vs_regular": round(chg_vs_reg, 6),
            "live_time_bj": t_bj,
            "tencent_direct_px": tc.get("live_price") if tc.get("ok") else None,
        }
    except Exception:  # noqa: BLE001
        if tc.get("ok"):
            return tc
        return {"ok": False, "error": tc.get("error", "All live sources unreachable")}


# Needed by refresh_yahoo_daily & _refresh_stooq_append
import urllib.parse  # noqa: E402
