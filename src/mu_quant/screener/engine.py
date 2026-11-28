"""US Stock Offense+Defense Full-Market Funnel Screener & Dynamic Allocation Engine (v2.0.0).

Core Design:
1. Two-Stage Full US Market Funnel:
   - Stage 1: Scans EastMoney's 13,000+ US market snapshot by daily turnover, filtering out
     leveraged/inverse ETFs, penny stocks (<$5), small illiquid names (<$3B mcap or <$50M/day).
   - Stage 2: Concurrently fetches daily OHLCV bars and computes strictly causal T-day close factors.
2. Offense + Defense Architecture:
   - Offense pool: Cross-sector high-elasticity leaders + full-market discoveries.
   - Defense pool: Gold (GLD), Treasuries (TLT/IEF/SGOV), Low-Vol/Defensive ETFs (USMV/XLV/XLP/XLU),
     and defensive blue-chips (WMT/COST/LLY/JNJ).
3. Market Regime Thermometer (BULL_OFFENSE / NEUTRAL_BALANCE / BEAR_DEFENSE):
   - Evaluates SPY & QQQ MA200 trend + Offense Market Breadth to dynamically adjust Offense vs Defense budgets.
4. Rotation-Compatible Portfolio Plan Output:
   - Produces standardized `target_weights` (zero leverage, sector-diversified, DVol-capped) ready to
     link directly with or replace the existing US rotation execution engine.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mu_quant.config import CONFIGS_DIR, ROOT
from mu_quant.logs.logger import get_logger

log = get_logger("mu_quant.screener")

SCREENER_CFG_PATH = CONFIGS_DIR / "screener_v1.0.0.json"
SCREENER_DIR = ROOT / "data" / "screener"
SCREENER_RAW_DIR = SCREENER_DIR / "raw"
SCREENER_CUSTOM_FILE = SCREENER_DIR / "custom_universe.json"
SCREENER_HISTORY_LOG = SCREENER_DIR / "scan_history.csv"
SCREENER_ALLOCATION_FILE = SCREENER_DIR / "latest_allocation.json"

for _d in (SCREENER_DIR, SCREENER_RAW_DIR):
    _d.mkdir(parents=True, exist_ok=True)

_LEVERAGED_KEYWORDS = (
    "做空", "反向", "两倍", "三倍", "2倍", "3倍", "杠杆", "1.5倍", "2X", "3X",
    "ULTRA", "SHORT", "INVERSE", "BEAR", "DIREXION", "PROSHARES", "VIX", "UVXY", "SQQQ", "TQQQ", "SOXL", "SOXS",
)


def _load_custom_entries() -> list[dict[str, Any]]:
    if not SCREENER_CUSTOM_FILE.exists():
        return []
    try:
        data = json.loads(SCREENER_CUSTOM_FILE.read_text(encoding="utf-8-sig"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_custom_entries(entries: list[dict[str, Any]]) -> None:
    SCREENER_CUSTOM_FILE.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_screener_config() -> dict[str, Any]:
    cfg = json.loads(SCREENER_CFG_PATH.read_text(encoding="utf-8-sig"))
    base_syms = {item["symbol"].upper() for item in cfg.get("universe", [])}
    merged = [dict(item, custom=False, discovered=False) for item in cfg.get("universe", [])]
    for entry in _load_custom_entries():
        sym = str(entry.get("symbol", "")).upper().strip()
        if sym and sym not in base_syms:
            e = dict(entry, symbol=sym, custom=True, discovered=False)
            e.setdefault("role", "offense")
            merged.append(e)
            base_syms.add(sym)
    cfg["universe"] = merged
    return cfg


def _fetch_eastmoney_kline(secid: str, beg: str = "20230101", timeout: int = 12) -> pd.DataFrame:
    url = (
        f"https://push2his.eastmoney.com/api/qt/stock/kline/get?"
        f"secid={secid}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56"
        f"&klt=101&fqt=1&beg={beg}&end=20500101"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    klines = (data.get("data") or {}).get("klines") or []
    rows = []
    for line in klines:
        d, o, c, h, l, v = line.split(",")
        rows.append({
            "Date": d,
            "Open": float(o),
            "High": float(h),
            "Low": float(l),
            "Close": float(c),
            "Volume": float(v),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").sort_index()


def fetch_us_market_funnel(
    existing_symbols: set[str],
    funnel_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stage 1: Scan EastMoney full US market snapshot and extract top liquid real equities."""
    fetch_size = int(funnel_cfg.get("snapshot_fetch_size", 260))
    top_n = int(funnel_cfg.get("funnel_top_n", 60))
    min_px = float(funnel_cfg.get("min_price_usd", 5.0))
    min_mcap = float(funnel_cfg.get("min_market_cap_usd", 3_000_000_000))
    min_amt = float(funnel_cfg.get("min_daily_turnover_usd", 50_000_000))

    url = (
        f"https://push2.eastmoney.com/api/qt/clist/get?"
        f"pn=1&pz={fetch_size}&po=1&np=1&fltt=2&invt=2&fid=f6"
        f"&fs=m:105,m:106,m:107&fields=f12,f13,f14,f2,f3,f6,f20,f100"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read().decode("utf-8"))

    total_market = int((data.get("data") or {}).get("total") or 0)
    diffs = (data.get("data") or {}).get("diff") or []

    discovered: list[dict[str, Any]] = []
    seen = set(existing_symbols)
    for row in diffs:
        sym = str(row.get("f12") or "").upper().strip()
        mkt = str(row.get("f13") or "").strip()
        name = str(row.get("f14") or "").strip()
        px = row.get("f2")
        amt = row.get("f6")
        mcap = row.get("f20")
        industry = str(row.get("f100") or "").strip()

        if not sym or sym in seen or mkt not in ("105", "106", "107"):
            continue
        # Skip non-standard tickers with dots/warrants/units
        if not sym.isalnum() or len(sym) > 5:
            continue
        try:
            px_f = float(px)
            amt_f = float(amt)
            mcap_f = float(mcap)
        except (TypeError, ValueError):
            # ETFs / leveraged funds usually have mcap == '-'
            continue

        if px_f < min_px or amt_f < min_amt or mcap_f < min_mcap:
            continue
        upper_name = f"{name} {sym}".upper()
        if any(kw in upper_name for kw in _LEVERAGED_KEYWORDS) or "ETF" in upper_name:
            continue

        seen.add(sym)
        discovered.append({
            "symbol": sym,
            "eastmoney_secid": f"{mkt}.{sym}",
            "role": "offense",
            "sector": f"全市场·{industry}" if (industry and industry != "-") else "全市场活跃正股",
            "theme": f"{name} · 市值${mcap_f/1e9:.0f}B · 日成交${amt_f/1e6:.0f}M",
            "custom": False,
            "discovered": True,
        })
        if len(discovered) >= top_n:
            break

    meta = {
        "us_market_total": total_market,
        "snapshot_checked": len(diffs),
        "funnel_passed": len(discovered),
    }
    return discovered, meta


def resolve_us_secid(symbol: str) -> tuple[str, pd.DataFrame]:
    """Resolve a US stock/ETF ticker to EastMoney secid (105=NASDAQ, 106=NYSE, 107=AMEX/ETF)."""
    sym = symbol.upper().strip()
    if not sym or not sym.replace(".", "").replace("-", "").isalnum():
        raise ValueError(f"无效的美股代码: {symbol}")
    for prefix in ("105", "106", "107"):
        cand = f"{prefix}.{sym}"
        try:
            df = _fetch_eastmoney_kline(cand, beg="20230101", timeout=10)
            if not df.empty and len(df) >= 20:
                return cand, df
        except Exception:
            continue
    raise ValueError(f"未在东方财富行情源中找到美股代码 {sym}（已尝试 105/106/107）")


def add_custom_symbol(
    symbol: str,
    sector: str = "自选观察",
    theme: str = "自定义美股标的",
    role: str = "offense",
) -> dict[str, Any]:
    sym = symbol.upper().strip()
    cfg = load_screener_config()
    if any(x["symbol"].upper() == sym for x in cfg["universe"]):
        return {"ok": True, "already_exists": True, "symbol": sym, "message": f"{sym} 已在选股池中"}

    secid, df = resolve_us_secid(sym)
    cache_path = SCREENER_RAW_DIR / f"{sym}.csv"
    df.to_csv(cache_path)

    entries = _load_custom_entries()
    entry = {
        "symbol": sym,
        "eastmoney_secid": secid,
        "role": role if role in ("offense", "defense") else "offense",
        "sector": sector.strip() or "自选观察",
        "theme": theme.strip() or "自定义美股标的",
    }
    entries.append(entry)
    _save_custom_entries(entries)
    return {"ok": True, "already_exists": False, "entry": entry, "rows": len(df)}


def remove_custom_symbol(symbol: str) -> dict[str, Any]:
    sym = symbol.upper().strip()
    entries = _load_custom_entries()
    new_entries = [e for e in entries if str(e.get("symbol", "")).upper() != sym]
    if len(new_entries) == len(entries):
        return {"ok": False, "error": f"{sym} 不是自定义标的（或不存在），核心池标的不可删除"}
    _save_custom_entries(new_entries)
    return {"ok": True, "symbol": sym}


def load_or_refresh_stock(item: dict[str, Any], refresh: bool = False) -> pd.DataFrame:
    sym = item["symbol"].upper()
    cache_path = SCREENER_RAW_DIR / f"{sym}.csv"

    existing = (
        pd.read_csv(cache_path, parse_dates=["Date"]).set_index("Date").sort_index()
        if cache_path.exists()
        else pd.DataFrame()
    )

    if refresh or existing.empty:
        try:
            fresh = _fetch_eastmoney_kline(item["eastmoney_secid"], beg="20230101")
            if not fresh.empty and len(fresh) >= 30:
                if existing.empty or len(fresh) >= len(existing) * 0.8:
                    fresh.to_csv(cache_path)
                    existing = fresh
                else:
                    combined = pd.concat([existing, fresh])
                    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
                    combined.to_csv(cache_path)
                    existing = combined
        except Exception as e:
            log.warning("Screener refresh failed for %s (%s); using cache (%d rows)", sym, e, len(existing))

    if existing.empty:
        raise RuntimeError(f"无法获取 {sym} 历史行情")

    for col in ("Open", "High", "Low", "Close", "Volume"):
        existing[col] = pd.to_numeric(existing[col], errors="coerce")
    return existing.dropna(subset=["Open", "High", "Low", "Close"])


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, prev_close = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def compute_stock_factors(df: pd.DataFrame, item: dict[str, Any], rules: dict[str, Any]) -> dict[str, Any]:
    """Compute strictly causal T-day close indicators for a single US stock/ETF."""
    close = df["Close"]
    vol = df["Volume"]
    ret = close.pct_change()

    ma20_n = int(rules.get("short_ma_days", 20))
    ma50_n = int(rules.get("mid_ma_days", 50))
    ma200_n = int(rules.get("trend_ma_days", 200))
    m20_n = int(rules.get("mom_short_days", 20))
    m60_n = int(rules.get("mom_mid_days", 60))
    m120_n = int(rules.get("mom_long_days", 120))
    dvol_n = int(rules.get("dvol_window", 21))
    target_dvol = float(rules.get("target_downside_vol", 0.55))
    atr_n = int(rules.get("atr_window", 14))
    stop_mult = float(rules.get("stop_atr_mult", 2.5))
    rsi_n = int(rules.get("rsi_window", 14))

    sma20 = close.rolling(ma20_n, min_periods=ma20_n).mean()
    sma50 = close.rolling(ma50_n, min_periods=ma50_n).mean()
    # For shorter-history defensive ETFs like SGOV (~194 bars), allow min_periods=120 fallback
    min_ma200 = min(ma200_n, max(120, len(df))) if item.get("role") == "defense" else ma200_n
    sma200 = close.rolling(ma200_n, min_periods=min_ma200).mean()

    mom20 = close / close.shift(m20_n) - 1.0
    mom60 = close / close.shift(m60_n) - 1.0
    mom120 = close / close.shift(m120_n) - 1.0

    neg_sq = np.where(ret < 0.0, ret ** 2, 0.0)
    dvol21 = np.sqrt(pd.Series(neg_sq, index=df.index).rolling(dvol_n, min_periods=dvol_n).mean() * 252.0)

    atr14 = _compute_atr(df, atr_n)
    rsi14 = _compute_rsi(close, rsi_n)
    high_52w = df["High"].rolling(min(252, len(df)), min_periods=20).max()
    vol_ma5 = vol.rolling(5, min_periods=5).mean()
    vol_ma20 = vol.rolling(20, min_periods=20).mean()

    def _val(s: pd.Series) -> float | None:
        v = s.iloc[-1]
        return None if pd.isna(v) or not np.isfinite(v) else float(v)

    c_val = float(close.iloc[-1])
    prev_c = float(close.iloc[-2]) if len(close) >= 2 else c_val
    chg1d = (c_val / prev_c - 1.0) if prev_c > 0 else 0.0

    ma20_v = _val(sma20)
    ma50_v = _val(sma50)
    ma200_v = _val(sma200)
    m20_v = _val(mom20)
    m60_v = _val(mom60)
    m120_v = _val(mom120)
    dv_v = _val(dvol21)
    atr_v = _val(atr14)
    rsi_v = _val(rsi14)
    h52_v = _val(high_52w)
    vm5_v = _val(vol_ma5)
    vm20_v = _val(vol_ma20)

    above_ma200 = bool(ma200_v is not None and c_val > ma200_v)
    above_ma50 = bool(ma50_v is not None and c_val > ma50_v)
    above_ma20 = bool(ma20_v is not None and c_val > ma20_v)
    ma_bull_aligned = bool(
        ma20_v is not None
        and ma50_v is not None
        and ma200_v is not None
        and c_val > ma20_v > ma50_v > ma200_v
    )

    dist_ma200 = (c_val / ma200_v - 1.0) if (ma200_v and ma200_v > 0) else None
    dd_52w = (c_val / h52_v - 1.0) if (h52_v and h52_v > 0) else None
    vol_ratio = (vm5_v / vm20_v) if (vm5_v is not None and vm20_v and vm20_v > 0) else 1.0

    dvol_cap = float(np.clip(target_dvol / max(dv_v, 0.10), 0.0, 1.0)) if dv_v is not None else 0.0
    risk_adj_mom = (m60_v / max(dv_v, 0.15)) if (m60_v is not None and dv_v is not None) else None
    stop_px = max(0.01, c_val - stop_mult * atr_v) if atr_v is not None else None
    stop_dist_pct = ((c_val - stop_px) / c_val) if (stop_px is not None and c_val > 0) else None

    tail = df.tail(60)
    tail_ma20 = sma20.tail(60)
    tail_ma50 = sma50.tail(60)
    spark = {
        "dates": [str(d.date()) for d in tail.index],
        "close": [round(float(x), 3) for x in tail["Close"]],
        "ma20": [None if pd.isna(x) else round(float(x), 3) for x in tail_ma20],
        "ma50": [None if pd.isna(x) else round(float(x), 3) for x in tail_ma50],
    }

    return {
        "symbol": item["symbol"].upper(),
        "role": item.get("role", "offense"),
        "sector": item.get("sector", "其他"),
        "theme": item.get("theme", ""),
        "custom": bool(item.get("custom", False)),
        "discovered": bool(item.get("discovered", False)),
        "date": str(df.index[-1].date()),
        "rows": int(len(df)),
        "close": round(c_val, 3),
        "chg1d": round(chg1d, 6),
        "sma20": round(ma20_v, 3) if ma20_v is not None else None,
        "sma50": round(ma50_v, 3) if ma50_v is not None else None,
        "sma200": round(ma200_v, 3) if ma200_v is not None else None,
        "above_ma200": above_ma200,
        "above_ma50": above_ma50,
        "above_ma20": above_ma20,
        "ma_bull_aligned": ma_bull_aligned,
        "dist_ma200": round(dist_ma200, 6) if dist_ma200 is not None else None,
        "mom20": round(m20_v, 6) if m20_v is not None else None,
        "mom60": round(m60_v, 6) if m60_v is not None else None,
        "mom120": round(m120_v, 6) if m120_v is not None else None,
        "dvol21": round(dv_v, 6) if dv_v is not None else None,
        "dvol_cap": round(dvol_cap, 4),
        "risk_adj_mom": round(risk_adj_mom, 4) if risk_adj_mom is not None else None,
        "high_52w": round(h52_v, 3) if h52_v is not None else None,
        "drawdown_52w": round(dd_52w, 6) if dd_52w is not None else None,
        "vol_ratio": round(vol_ratio, 3),
        "rsi14": round(rsi_v, 2) if rsi_v is not None else None,
        "atr14": round(atr_v, 3) if atr_v is not None else None,
        "stop_price": round(stop_px, 3) if stop_px is not None else None,
        "stop_dist_pct": round(stop_dist_pct, 4) if stop_dist_pct is not None else None,
        "sparkline": spark,
    }


def _pct_ranks(values: list[float | None]) -> list[float]:
    s = pd.Series([np.nan if v is None else float(v) for v in values], dtype="float64")
    r = s.rank(pct=True, na_option="bottom")
    return [0.0 if pd.isna(s.iloc[i]) else float(r.iloc[i]) for i in range(len(values))]


def _score_and_tag(records: list[dict[str, Any]], rules: dict[str, Any], benchmark: str = "QQQ") -> list[dict[str, Any]]:
    rsi_overheat = float(rules.get("rsi_overheat", 80.0))

    bench = next((r for r in records if r["symbol"] == benchmark), None)
    bench_m60 = bench["mom60"] if bench and bench["mom60"] is not None else 0.0
    bench_m20 = bench["mom20"] if bench and bench["mom20"] is not None else 0.0

    for r in records:
        r["rs60_vs_qqq"] = round(r["mom60"] - bench_m60, 6) if r["mom60"] is not None else None
        r["rs20_vs_qqq"] = round(r["mom20"] - bench_m20, 6) if r["mom20"] is not None else None
        r["in_rotation_pool"] = False

    r_ram = _pct_ranks([r["risk_adj_mom"] for r in records])
    r_rs60 = _pct_ranks([r["rs60_vs_qqq"] for r in records])
    r_m20 = _pct_ranks([r["mom20"] for r in records])
    r_52w = _pct_ranks([r["drawdown_52w"] for r in records])
    r_vol = _pct_ranks([r["vol_ratio"] for r in records])
    # Low-vol rank (lower DVol21 = higher score for defense)
    r_lowvol = _pct_ranks([(-r["dvol21"] if r["dvol21"] is not None else None) for r in records])

    for i, r in enumerate(records):
        align_bonus = 1.0 if r["ma_bull_aligned"] else (0.5 if r["above_ma50"] else 0.0)
        if r["role"] == "defense":
            # Defensive scoring rewards low downside volatility + positive trend + resilience vs QQQ
            raw_score = (
                35.0 * r_ram[i]
                + 25.0 * r_lowvol[i]
                + 20.0 * r_rs60[i]
                + 10.0 * r_52w[i]
                + 10.0 * align_bonus
            )
        else:
            raw_score = (
                35.0 * r_ram[i]
                + 25.0 * r_rs60[i]
                + 15.0 * r_m20[i]
                + 15.0 * r_52w[i]
                + 6.0 * align_bonus
                + 4.0 * r_vol[i]
            )

        tags: list[str] = []
        if r["role"] == "defense":
            tags.append("🛡️防守盾")
        if r["discovered"]:
            tags.append("🌐全市场发现")
        if r["in_rotation_pool"]:
            tags.append("⭐轮动池")
        if r["ma_bull_aligned"]:
            tags.append("多头排列")
        elif not r["above_ma200"]:
            tags.append("跌破年线")

        if r["rs60_vs_qqq"] is not None and r["rs60_vs_qqq"] > 0.05:
            tags.append("强于大盘")
        if r["drawdown_52w"] is not None and r["drawdown_52w"] >= -0.05:
            tags.append("逼近新高")
        if r["vol_ratio"] is not None and r["vol_ratio"] >= 1.25:
            tags.append("放量")
        if r["dvol21"] is not None and r["dvol21"] <= 0.30:
            tags.append("极低波动")

        overheat = bool(r["rsi14"] is not None and r["rsi14"] > rsi_overheat)
        if overheat:
            raw_score -= 8.0
            tags.append("短期过热")

        if not r["above_ma200"]:
            final_score = min(45.0, raw_score * 0.48)
        else:
            final_score = float(np.clip(raw_score, 0.0, 100.0))

        if not r["above_ma200"]:
            grade, action = "D", "年线下方·观望"
        elif final_score >= 78.0 and not overheat:
            grade, action = "S", "强势避险·首选" if r["role"] == "defense" else "强势龙头·重点关注"
        elif final_score >= 65.0:
            grade, action = "A", "稳健防御·备选" if r["role"] == "defense" else "趋势优良·备选"
        elif final_score >= 50.0:
            grade, action = "B", "中性跟踪"
        else:
            grade, action = "C", "动量偏弱"

        r["composite_score"] = round(final_score, 1)
        r["grade"] = grade
        r["action_hint"] = action
        r["tags"] = tags
        r["factor_breakdown"] = {
            "risk_adj_mom": round(35.0 * r_ram[i], 1),
            "rs60_vs_qqq": round((20.0 if r["role"] == "defense" else 25.0) * r_rs60[i], 1),
            "mom20": round((25.0 * r_lowvol[i]) if r["role"] == "defense" else (15.0 * r_m20[i]), 1),
            "near_52w_high": round((10.0 if r["role"] == "defense" else 15.0) * r_52w[i], 1),
            "trend_vol": round((10.0 * align_bonus) if r["role"] == "defense" else (6.0 * align_bonus + 4.0 * r_vol[i]), 1),
        }

    records.sort(key=lambda x: (x["above_ma200"], x["composite_score"]), reverse=True)
    for idx, r in enumerate(records, start=1):
        r["rank"] = idx
    return records


def evaluate_market_regime(records: list[dict[str, Any]], alloc_cfg: dict[str, Any]) -> dict[str, Any]:
    """Macro + Breadth Market Regime Thermometer controlling Offense vs Defense budgets."""
    spy = next((r for r in records if r["symbol"] == "SPY"), None)
    qqq = next((r for r in records if r["symbol"] == "QQQ"), None)
    offense_pool = [r for r in records if r["role"] == "offense"]
    offense_above = sum(1 for r in offense_pool if r["above_ma200"])
    offense_breadth = offense_above / max(1, len(offense_pool))

    spy_up = bool(spy and spy["above_ma200"])
    qqq_up = bool(qqq and qqq["above_ma200"])

    if spy_up and qqq_up and offense_breadth >= 0.50:
        regime_key = "BULL_OFFENSE"
        reason = f"SPY与QQQ均站上200日年线，且进攻宽度 {offense_breadth*100:.0f}% ≥ 50%，允许积极进攻。"
    elif (not spy_up and not qqq_up) or offense_breadth < 0.25:
        regime_key = "BEAR_DEFENSE"
        trigger = "SPY与QQQ双双跌破200日年线" if (not spy_up and not qqq_up) else f"进攻宽度跌破25%（当前 {offense_breadth*100:.0f}%）"
        reason = f"{trigger}（进攻宽度 {offense_breadth*100:.0f}%），切换熊市防守模式，重配避险资产与现金。"
    else:
        regime_key = "NEUTRAL_BALANCE"
        reason = (
            f"大盘分化（SPY{'站上' if spy_up else '跌破'}年线 / QQQ{'站上' if qqq_up else '跌破'}年线，"
            f"进攻宽度 {offense_breadth*100:.0f}%），启动攻守平衡，严控单一方向回撤。"
        )

    reg_cfg = alloc_cfg["regimes"][regime_key]
    return {
        "regime": regime_key,
        "label": reg_cfg["label"],
        "reason": reason,
        "spy_above_ma200": spy_up,
        "qqq_above_ma200": qqq_up,
        "offense_breadth": round(offense_breadth, 4),
        "offense_above_ma200": offense_above,
        "offense_total": len(offense_pool),
        "offense_budget": float(reg_cfg["offense_budget"]),
        "defense_budget": float(reg_cfg["defense_budget"]),
        "cash_floor": float(reg_cfg["cash_floor"]),
    }


def build_offense_defense_allocation(
    scored: list[dict[str, Any]],
    regime: dict[str, Any],
    alloc_cfg: dict[str, Any],
    signal_date: str,
    version: str,
) -> dict[str, Any]:
    """Build rotation-compatible Offense + Defense target_weights (strictly zero leverage)."""
    off_slots = int(alloc_cfg.get("offense_slots", 3))
    def_slots = int(alloc_cfg.get("defense_slots", 2))
    max_per_sec = int(alloc_cfg.get("max_per_sector", 2))
    max_single = float(alloc_cfg.get("max_single_weight", 0.35))

    off_budget = regime["offense_budget"]
    def_budget = regime["defense_budget"]

    # 1. Pick top offense candidates (above MA200, positive Mom60, beat QQQ, not overheated, sector-diversified)
    off_cands = [
        r for r in scored
        if r["role"] == "offense"
        and r["above_ma200"]
        and (r["mom60"] is not None and r["mom60"] > 0.0)
        and (r["rs60_vs_qqq"] is not None and r["rs60_vs_qqq"] > 0.0)
        and (r["rsi14"] is None or r["rsi14"] <= 82.0)
    ]
    off_cands.sort(key=lambda x: x["composite_score"], reverse=True)

    picked_off: list[dict[str, Any]] = []
    sec_counts: dict[str, int] = {}
    for c in off_cands:
        sec = c["sector"]
        if sec_counts.get(sec, 0) >= max_per_sec:
            continue
        picked_off.append(c)
        sec_counts[sec] = sec_counts.get(sec, 0) + 1
        if len(picked_off) >= off_slots:
            break

    # 2. Pick top defense candidates (above MA200, sorted by defense composite_score)
    def_cands = [
        r for r in scored
        if r["role"] == "defense" and r["above_ma200"]
    ]
    def_cands.sort(key=lambda x: x["composite_score"], reverse=True)
    picked_def = def_cands[:def_slots]

    # 3. Assign DVol-capped weights within budgets
    positions: list[dict[str, Any]] = []
    target_weights: dict[str, float] = {}

    if picked_off and off_budget > 0:
        base_w = off_budget / len(picked_off)
        for c in picked_off:
            w = min(max_single, base_w * max(0.35, c["dvol_cap"]))
            w = round(w, 4)
            if w > 0.01:
                target_weights[c["symbol"]] = w
                positions.append({
                    "symbol": c["symbol"],
                    "sleeve": "OFFENSE",
                    "sector": c["sector"],
                    "theme": c["theme"],
                    "weight": w,
                    "score": c["composite_score"],
                    "dvol21": c["dvol21"],
                    "stop_price": c["stop_price"],
                    "discovered": c["discovered"],
                })

    if picked_def and def_budget > 0:
        base_w = def_budget / len(picked_def)
        for c in picked_def:
            w = min(max_single, base_w * max(0.60, c["dvol_cap"]))
            w = round(w, 4)
            if w > 0.01:
                target_weights[c["symbol"]] = w
                positions.append({
                    "symbol": c["symbol"],
                    "sleeve": "DEFENSE",
                    "sector": c["sector"],
                    "theme": c["theme"],
                    "weight": w,
                    "score": c["composite_score"],
                    "dvol21": c["dvol21"],
                    "stop_price": c["stop_price"],
                    "discovered": False,
                })

    # Hard zero-leverage & cash floor invariant
    max_total = max(0.0, 1.0 - regime["cash_floor"])
    total_w = sum(target_weights.values())
    if total_w > max_total and total_w > 0:
        scale = max_total / total_w
        for sym in list(target_weights.keys()):
            target_weights[sym] = round(target_weights[sym] * scale, 4)
        for p in positions:
            p["weight"] = target_weights[p["symbol"]]
        total_w = sum(target_weights.values())

    off_actual = round(sum(p["weight"] for p in positions if p["sleeve"] == "OFFENSE"), 4)
    def_actual = round(sum(p["weight"] for p in positions if p["sleeve"] == "DEFENSE"), 4)
    cash_actual = round(max(0.0, 1.0 - total_w), 4)

    return {
        "version": version,
        "signal_date": signal_date,
        "computed_at": dt.datetime.now().isoformat(timespec="seconds"),
        "regime": regime["regime"],
        "regime_label": regime["label"],
        "selected_symbols": [p["symbol"] for p in positions],
        "target_weights": target_weights,
        "offense_weight": off_actual,
        "defense_weight": def_actual,
        "total_invested_weight": round(total_w, 4),
        "cash_weight": cash_actual,
        "positions": positions,
    }


def apply_preset_filter(records: list[dict[str, Any]], preset: str = "core_momentum") -> list[dict[str, Any]]:
    if preset == "core_momentum":
        out = [
            r for r in records
            if r["role"] == "offense"
            and r["above_ma200"]
            and (r["rs60_vs_qqq"] is not None and r["rs60_vs_qqq"] > 0.0)
            and (r["mom60"] is not None and r["mom60"] > 0.0)
            and (r["rsi14"] is None or r["rsi14"] <= 82.0)
        ]
        out.sort(key=lambda x: x["composite_score"], reverse=True)
        return out
    if preset == "defense_shield":
        out = [r for r in records if r["role"] == "defense"]
        out.sort(key=lambda x: (x["above_ma200"], x["composite_score"]), reverse=True)
        return out
    if preset == "breakout_52w":
        out = [
            r for r in records
            if r["ma_bull_aligned"]
            and (r["drawdown_52w"] is not None and r["drawdown_52w"] >= -0.12)
            and (r["mom20"] is not None and r["mom20"] > 0.0)
        ]
        out.sort(key=lambda x: x["composite_score"], reverse=True)
        return out
    if preset == "market_discovery":
        out = [r for r in records if r["discovered"]]
        out.sort(key=lambda x: (x["above_ma200"], x["composite_score"]), reverse=True)
        return out
    return list(records)


def _append_scan_history(
    signal_date: str,
    preset: str,
    regime: str,
    top_symbols: list[str],
    above200_count: int,
    total_count: int,
) -> None:
    rec = {
        "scanned_at": dt.datetime.now().isoformat(timespec="seconds"),
        "signal_date": signal_date,
        "preset": preset,
        "regime": regime,
        "above_ma200": above200_count,
        "total": total_count,
        "top5": ",".join(top_symbols[:5]),
    }
    try:
        df = pd.read_csv(SCREENER_HISTORY_LOG, encoding="utf-8-sig") if SCREENER_HISTORY_LOG.exists() else pd.DataFrame()
        pd.concat([df, pd.DataFrame([rec])], ignore_index=True).to_csv(
            SCREENER_HISTORY_LOG, index=False, encoding="utf-8-sig"
        )
    except Exception as exc:
        log.warning("Failed to append screener history: %s", exc)


def run_screener(
    preset: str = "core_momentum",
    sector: str = "ALL",
    refresh: bool = False,
    scan_mode: str = "curated",
    record_history: bool = False,
    max_workers: int = 12,
) -> dict[str, Any]:
    cfg = load_screener_config()
    rules = cfg["rules"]
    universe = list(cfg["universe"])
    benchmark = cfg.get("benchmark", "QQQ")

    funnel_meta = {"enabled": False, "us_market_total": 0, "snapshot_checked": 0, "funnel_passed": 0}
    if scan_mode == "full_market":
        try:
            existing_syms = {x["symbol"].upper() for x in universe}
            discovered, fmeta = fetch_us_market_funnel(existing_syms, cfg.get("funnel", {}))
            universe.extend(discovered)
            funnel_meta = dict(fmeta, enabled=True)
        except Exception as exc:
            log.warning("Full-market funnel stage 1 failed (%s); falling back to curated pool", exc)
            funnel_meta["error"] = str(exc)

    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    def _worker(item: dict[str, Any]) -> dict[str, Any]:
        df = load_or_refresh_stock(item, refresh=refresh)
        return compute_stock_factors(df, item, rules)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {pool.submit(_worker, item): item["symbol"] for item in universe}
        for fut in as_completed(future_map):
            sym = future_map[fut]
            try:
                records.append(fut.result())
            except Exception as exc:
                log.warning("Screener skipped %s: %s", sym, exc)
                errors.append({"symbol": sym, "error": str(exc)})

    scored = _score_and_tag(records, rules, benchmark=benchmark)
    regime = evaluate_market_regime(scored, cfg["allocation"])
    signal_date = max((r["date"] for r in scored), default="")
    portfolio_plan = build_offense_defense_allocation(
        scored, regime, cfg["allocation"], signal_date=signal_date, version=cfg["version"]
    )

    # Always persist latest rotation-compatible allocation snapshot
    try:
        SCREENER_ALLOCATION_FILE.write_text(
            json.dumps(portfolio_plan, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        log.warning("Failed to save latest_allocation.json: %s", exc)

    filtered = apply_preset_filter(scored, preset=preset)
    if sector and sector != "ALL":
        filtered = [r for r in filtered if r["sector"] == sector]

    # Sector strength summary (group discovered into "全市场发现" so strip stays clean)
    sector_map: dict[str, list[dict[str, Any]]] = {}
    for r in scored:
        key = "🌐 全市场新发现" if r["discovered"] else r["sector"]
        sector_map.setdefault(key, []).append(r)
    sector_summary = []
    for sec_name, items in sector_map.items():
        m60s = [x["mom60"] for x in items if x["mom60"] is not None]
        avg_score = float(np.mean([x["composite_score"] for x in items]))
        above_cnt = sum(1 for x in items if x["above_ma200"])
        leader = max(items, key=lambda x: x["composite_score"])
        sector_summary.append({
            "sector": sec_name,
            "count": len(items),
            "above_ma200": above_cnt,
            "avg_score": round(avg_score, 1),
            "avg_mom60": round(float(np.mean(m60s)), 4) if m60s else None,
            "leader": leader["symbol"],
            "leader_score": leader["composite_score"],
        })
    sector_summary.sort(key=lambda x: x["avg_score"], reverse=True)

    outside_top = [
        r for r in scored
        if (not r["in_rotation_pool"])
        and r["role"] == "offense"
        and r["above_ma200"]
    ][:5]

    discovered_top = [
        r for r in scored
        if r["discovered"] and r["above_ma200"]
    ][:5]

    above200_cnt = sum(1 for r in scored if r["above_ma200"])
    bull_cnt = sum(1 for r in scored if r["ma_bull_aligned"])
    bench_row = next((r for r in scored if r["symbol"] == benchmark), None)

    if record_history and scored:
        _append_scan_history(
            signal_date, preset, regime["regime"], [r["symbol"] for r in filtered], above200_cnt, len(scored)
        )

    return {
        "ok": True,
        "version": cfg["version"],
        "name": cfg["name"],
        "signal_date": signal_date,
        "computed_at": dt.datetime.now().isoformat(timespec="seconds"),
        "scan_mode": scan_mode,
        "funnel": funnel_meta,
        "preset": preset,
        "sector_filter": sector,
        "presets": cfg["presets"],
        "rules": rules,
        "market_regime": regime,
        "portfolio_plan": portfolio_plan,
        "benchmark": {
            "symbol": benchmark,
            "close": bench_row["close"] if bench_row else None,
            "mom60": bench_row["mom60"] if bench_row else None,
            "above_ma200": bench_row["above_ma200"] if bench_row else None,
        },
        "market_breadth": {
            "total": len(scored),
            "above_ma200": above200_cnt,
            "above_ma200_pct": round(above200_cnt / max(1, len(scored)), 4),
            "ma_bull_aligned": bull_cnt,
            "matched_count": len(filtered),
        },
        "sector_summary": sector_summary,
        "outside_rotation_top5": [
            {
                "symbol": r["symbol"],
                "sector": r["sector"],
                "theme": r["theme"],
                "composite_score": r["composite_score"],
                "mom60": r["mom60"],
                "dvol21": r["dvol21"],
                "grade": r["grade"],
                "discovered": r["discovered"],
            }
            for r in outside_top
        ],
        "discovered_top5": [
            {
                "symbol": r["symbol"],
                "sector": r["sector"],
                "theme": r["theme"],
                "composite_score": r["composite_score"],
                "mom60": r["mom60"],
                "dvol21": r["dvol21"],
                "grade": r["grade"],
            }
            for r in discovered_top
        ],
        "stocks": filtered,
        "all_count": len(scored),
        "errors": errors,
    }
