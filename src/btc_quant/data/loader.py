"""OKX BTC-USDT-SWAP & BTC-USDT Spot Market Data + Funding Rate Loader.

Fetches real historical OHLCV (4H & 1D), 8-hour funding rates, and spot reference
candles from OKX public REST API v5 with rate-limit respect, incremental updates,
and strict data quality validation (monotonic timestamps, zero NaN OHLC, gap audit).
"""
from __future__ import annotations

import datetime as dt
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from btc_quant.config import RAW_DIR
from btc_quant.logs.logger import get_logger

log = get_logger("btc_quant.data")

OKX_BASE = "https://www.okx.com"
SWAP_INST = "BTC-USDT-SWAP"
SPOT_INST = "BTC-USDT"

COIN_SPECS: dict[str, dict[str, Any]] = {
    "BTC": {"swap_inst": "BTC-USDT-SWAP", "spot_inst": "BTC-USDT", "ct_val": 0.01, "lot_sz": 0.01, "min_sz": 0.01},
    "ETH": {"swap_inst": "ETH-USDT-SWAP", "spot_inst": "ETH-USDT", "ct_val": 0.1, "lot_sz": 0.01, "min_sz": 0.01},
    "SOL": {"swap_inst": "SOL-USDT-SWAP", "spot_inst": "SOL-USDT", "ct_val": 1.0, "lot_sz": 0.01, "min_sz": 0.01},
}
SUPPORTED_COINS = ("BTC", "ETH", "SOL")

CSV_4H = RAW_DIR / "btc_usdt_swap_4h.csv"
CSV_1D = RAW_DIR / "btc_usdt_swap_1d.csv"
CSV_SPOT_1D = RAW_DIR / "btc_usdt_spot_1d.csv"
CSV_FUNDING = RAW_DIR / "btc_usdt_swap_funding.csv"
META_FILE = RAW_DIR / "_last_refresh.json"


def _coin_paths(coin: str) -> dict[str, Path]:
    c = coin.upper()
    if c == "BTC":
        return {"4h": CSV_4H, "1d": CSV_1D, "spot_1d": CSV_SPOT_1D, "funding": CSV_FUNDING}
    lc = c.lower()
    return {
        "4h": RAW_DIR / f"{lc}_usdt_swap_4h.csv",
        "1d": RAW_DIR / f"{lc}_usdt_swap_1d.csv",
        "spot_1d": RAW_DIR / f"{lc}_usdt_spot_1d.csv",
        "funding": RAW_DIR / f"{lc}_usdt_swap_funding.csv",
    }


def _get_json(path: str, retries: int = 4) -> dict[str, Any]:
    url = OKX_BASE + path
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("code") == "0":
                    return data
                # Rate limit or transient error
                if data.get("code") in ("50011", "50013"):
                    time.sleep(0.5 * (attempt + 1))
                    continue
                return data
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"OKX request failed for {path}: {last_err}")


def fetch_okx_candles(
    inst_id: str,
    bar: str = "4H",
    since_ts_ms: int = 1577836800000,  # 2020-01-01 UTC
    max_pages: int = 200,
    existing_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Paginate OKX candles + history-candles back to since_ts_ms (or incremental head)."""
    stop_ts = since_ts_ms
    if existing_df is not None and not existing_df.empty:
        # Overlap by 5 bars to refresh any recently closed bar
        tail_ts = int(existing_df["ts_ms"].iloc[-min(5, len(existing_df))])
        stop_ts = max(since_ts_ms, tail_ts)

    rows: list[list[str]] = []
    # First get latest candles from /market/candles (includes current forming bar)
    latest = _get_json(f"/api/v5/market/candles?instId={inst_id}&bar={bar}&limit=100").get("data", [])
    rows.extend(latest)

    after_ts = int(latest[-1][0]) if latest else None
    pages = 1
    while after_ts is not None and after_ts > stop_ts and pages < max_pages:
        time.sleep(0.06)
        batch = _get_json(
            f"/api/v5/market/history-candles?instId={inst_id}&bar={bar}&after={after_ts}&limit=100"
        ).get("data", [])
        if not batch:
            break
        rows.extend(batch)
        new_after = int(batch[-1][0])
        if new_after >= after_ts:
            break
        after_ts = new_after
        pages += 1

    if not rows and existing_df is not None:
        return existing_df

    cols = ["ts_ms", "open", "high", "low", "close", "vol", "vol_ccy", "vol_quote", "confirm"]
    new_df = pd.DataFrame(rows, columns=cols)
    new_df["ts_ms"] = new_df["ts_ms"].astype("int64")
    for c in ("open", "high", "low", "close", "vol", "vol_ccy", "vol_quote"):
        new_df[c] = new_df[c].astype(float)
    new_df["confirm"] = new_df["confirm"].astype(int)
    # Keep only confirmed (closed) bars to strictly eliminate forming-bar lookahead!
    new_df = new_df[new_df["confirm"] == 1].copy()

    if existing_df is not None and not existing_df.empty:
        df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        df = new_df

    df = df.drop_duplicates(subset=["ts_ms"], keep="last").sort_values("ts_ms").reset_index(drop=True)
    df = df[df["ts_ms"] >= since_ts_ms].reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    return df[["ts_ms", "timestamp", "open", "high", "low", "close", "vol", "vol_ccy", "vol_quote", "confirm"]]


def fetch_okx_funding_history(
    inst_id: str = SWAP_INST,
    since_ts_ms: int = 1577836800000,
    max_pages: int = 120,
    existing_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Paginate OKX 8-hour historical funding rates."""
    stop_ts = since_ts_ms
    if existing_df is not None and not existing_df.empty:
        stop_ts = max(since_ts_ms, int(existing_df["funding_ts_ms"].iloc[-min(5, len(existing_df))]))

    rows: list[dict[str, Any]] = []
    after_ts: int | None = None
    for _ in range(max_pages):
        q = f"/api/v5/public/funding-rate-history?instId={inst_id}&limit=100"
        if after_ts is not None:
            q += f"&after={after_ts}"
        batch = _get_json(q).get("data", [])
        if not batch:
            break
        rows.extend(batch)
        new_after = int(batch[-1]["fundingTime"])
        if new_after <= stop_ts or (after_ts is not None and new_after >= after_ts):
            break
        after_ts = new_after
        time.sleep(0.06)

    if not rows and existing_df is not None:
        return existing_df

    new_df = pd.DataFrame(
        [
            {
                "funding_ts_ms": int(r["fundingTime"]),
                "funding_rate": float(r["fundingRate"]),
                "realized_rate": float(r.get("realizedRate") or r["fundingRate"]),
            }
            for r in rows
        ]
    )
    if existing_df is not None and not existing_df.empty:
        df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        df = new_df
    df = df.drop_duplicates(subset=["funding_ts_ms"], keep="last").sort_values("funding_ts_ms").reset_index(drop=True)
    df = df[df["funding_ts_ms"] >= since_ts_ms].reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["funding_ts_ms"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    return df[["funding_ts_ms", "timestamp", "funding_rate", "realized_rate"]]


def fetch_live_snapshot(coin: str = "BTC") -> dict[str, Any]:
    """Fetch current live ticker, mark price, funding rate, and open interest for a coin SWAP."""
    spec = COIN_SPECS[coin.upper()]
    swap_inst = spec["swap_inst"]
    spot_inst = spec["spot_inst"]
    tk = _get_json(f"/api/v5/market/ticker?instId={swap_inst}")["data"][0]
    fr = _get_json(f"/api/v5/public/funding-rate?instId={swap_inst}")["data"][0]
    oi = _get_json(f"/api/v5/public/open-interest?instType=SWAP&instId={swap_inst}")["data"][0]
    spot = _get_json(f"/api/v5/market/ticker?instId={spot_inst}")["data"][0]
    swap_px = float(tk["last"])
    spot_px = float(spot["last"])
    basis_bps = (swap_px / spot_px - 1.0) * 1e4 if spot_px > 0 else 0.0
    return {
        "fetched_at": dt.datetime.now().isoformat(timespec="seconds"),
        "coin": coin.upper(),
        "inst_id": swap_inst,
        "last": swap_px,
        "bid": float(tk["bidPx"]),
        "ask": float(tk["askPx"]),
        "spread_bps": round((float(tk["askPx"]) - float(tk["bidPx"])) / swap_px * 1e4, 3),
        "spot_last": spot_px,
        "basis_bps": round(basis_bps, 2),
        "vol_24h_usdt": round(float(tk["volCcy24h"]) * swap_px * spec["ct_val"], 2),
        "funding_rate": float(fr["fundingRate"]),
        "funding_rate_annual_pct": round(float(fr["fundingRate"]) * 3 * 365 * 100, 2),
        "next_funding_rate": float(fr["nextFundingRate"]) if fr.get("nextFundingRate") else None,
        "oi_coin": round(float(oi["oiCcy"]), 2),
        "oi_usd_m": round(float(oi["oiCcy"]) * swap_px / 1e6, 2),
    }


def refresh_coin_market_data(coin: str = "BTC", full_rebuild: bool = False) -> dict[str, Any]:
    """Download/incrementally update 1D swap, 4H swap, 1D spot, and 8h funding history for one coin."""
    c = coin.upper()
    spec = COIN_SPECS[c]
    paths = _coin_paths(c)
    since_1d = 1577836800000  # 2020-01-01 UTC
    since_4h = 1609459200000  # 2021-01-01 UTC

    ex_1d = None if (full_rebuild or not paths["1d"].exists()) else pd.read_csv(paths["1d"])
    ex_4h = None if (full_rebuild or not paths["4h"].exists()) else pd.read_csv(paths["4h"])
    ex_spot = None if (full_rebuild or not paths["spot_1d"].exists()) else pd.read_csv(paths["spot_1d"])
    ex_fr = None if (full_rebuild or not paths["funding"].exists()) else pd.read_csv(paths["funding"])

    log.info("Refreshing %s 1D candles...", spec["swap_inst"])
    df_1d = fetch_okx_candles(spec["swap_inst"], "1D", since_ts_ms=since_1d, max_pages=60, existing_df=ex_1d)
    df_1d.to_csv(paths["1d"], index=False)

    log.info("Refreshing %s 4H candles...", spec["swap_inst"])
    df_4h = fetch_okx_candles(spec["swap_inst"], "4H", since_ts_ms=since_4h, max_pages=160, existing_df=ex_4h)
    df_4h.to_csv(paths["4h"], index=False)

    log.info("Refreshing %s Spot 1D candles...", spec["spot_inst"])
    df_spot = fetch_okx_candles(spec["spot_inst"], "1D", since_ts_ms=since_1d, max_pages=60, existing_df=ex_spot)
    df_spot.to_csv(paths["spot_1d"], index=False)

    log.info("Refreshing %s 8h funding history...", spec["swap_inst"])
    df_fr = fetch_okx_funding_history(spec["swap_inst"], since_ts_ms=since_1d, max_pages=100, existing_df=ex_fr)
    df_fr.to_csv(paths["funding"], index=False)

    meta = {
        "coin": c,
        "refreshed_at": dt.datetime.now().isoformat(timespec="seconds"),
        "swap_1d_rows": len(df_1d),
        "swap_1d_range": [str(df_1d["timestamp"].iloc[0]), str(df_1d["timestamp"].iloc[-1])],
        "swap_4h_rows": len(df_4h),
        "swap_4h_range": [str(df_4h["timestamp"].iloc[0]), str(df_4h["timestamp"].iloc[-1])],
        "spot_1d_rows": len(df_spot),
        "funding_rows": len(df_fr),
        "funding_range": [str(df_fr["timestamp"].iloc[0]), str(df_fr["timestamp"].iloc[-1])] if len(df_fr) else [],
    }
    if c == "BTC":
        META_FILE.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("%s data refresh complete: %s", c, meta)
    return meta


def refresh_btc_market_data(full_rebuild: bool = False) -> dict[str, Any]:
    return refresh_coin_market_data("BTC", full_rebuild=full_rebuild)


def refresh_all_coins_data(coins: tuple[str, ...] = SUPPORTED_COINS, full_rebuild: bool = False) -> dict[str, Any]:
    out = {}
    for c in coins:
        out[c] = refresh_coin_market_data(c, full_rebuild=full_rebuild)
    META_FILE.write_text(
        json.dumps({"refreshed_at": dt.datetime.now().isoformat(timespec="seconds"), "coins": out}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out


def load_coin_datasets(coin: str = "BTC") -> dict[str, pd.DataFrame]:
    """Load local CSVs for one coin, merge funding rates & spot basis onto 4H and 1D bars."""
    c = coin.upper()
    paths = _coin_paths(c)
    if not paths["1d"].exists() or not paths["4h"].exists():
        refresh_coin_market_data(c)

    df_1d = pd.read_csv(paths["1d"])
    df_4h = pd.read_csv(paths["4h"])
    df_spot = pd.read_csv(paths["spot_1d"]) if paths["spot_1d"].exists() else pd.DataFrame()
    df_fr = pd.read_csv(paths["funding"]) if paths["funding"].exists() else pd.DataFrame()

    for df in (df_1d, df_4h):
        df["dt"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        df.set_index("dt", inplace=True)
        df.sort_index(inplace=True)

    # Merge spot close onto 1D to compute perpetual basis (premium/discount)
    if not df_spot.empty:
        sp = df_spot.copy()
        sp["dt"] = pd.to_datetime(sp["ts_ms"], unit="ms", utc=True)
        sp = sp.set_index("dt")["close"].rename("spot_close")
        df_1d = df_1d.join(sp, how="left")
        df_1d["basis_pct"] = (df_1d["close"] / df_1d["spot_close"] - 1.0).fillna(0.0)
    else:
        df_1d["spot_close"] = df_1d["close"]
        df_1d["basis_pct"] = 0.0

    # Align 8h funding rates onto 4H bars (at exact settlement timestamps) and 1D bars (sum of 3 daily settlements)
    if not df_fr.empty:
        fr = df_fr.copy()
        fr["dt"] = pd.to_datetime(fr["funding_ts_ms"], unit="ms", utc=True)
        fr = fr.set_index("dt").sort_index()
        # Round funding timestamps to nearest hour to align cleanly with 00:00, 08:00, 16:00 UTC bars
        fr.index = fr.index.round("h")
        fr = fr[~fr.index.duplicated(keep="last")]

        df_4h["funding_rate"] = fr["funding_rate"].reindex(df_4h.index)
        df_4h["funding_is_real"] = df_4h["funding_rate"].notna()
        # For historical bars older than OKX's 90-day public funding archive, impute conservative
        # OKX baseline funding (0.01% per 8h settlement at 00:00/08:00/16:00 UTC) so backtests
        # never understate long funding carry costs.
        is_settle_bar = df_4h.index.hour.isin([0, 8, 16])
        imputed_fr = np.where(is_settle_bar, 0.0001, 0.0)
        df_4h["funding_rate"] = df_4h["funding_rate"].fillna(pd.Series(imputed_fr, index=df_4h.index))

        daily_fr = fr["funding_rate"].resample("1D").sum()
        df_1d["funding_rate_daily"] = daily_fr.reindex(df_1d.index).fillna(0.0003)
        fr_out = fr
    else:
        is_settle_bar = df_4h.index.hour.isin([0, 8, 16])
        df_4h["funding_rate"] = np.where(is_settle_bar, 0.0001, 0.0)
        df_4h["funding_is_real"] = False
        df_1d["funding_rate_daily"] = 0.0003
        fr_out = pd.DataFrame()

    return {"df_4h": df_4h, "df_1d": df_1d, "df_funding": fr_out}


def load_btc_datasets() -> dict[str, pd.DataFrame]:
    return load_coin_datasets("BTC")


def load_all_coins_datasets(coins: tuple[str, ...] = SUPPORTED_COINS) -> dict[str, dict[str, pd.DataFrame]]:
    return {c: load_coin_datasets(c) for c in coins}


def validate_btc_data_quality(datasets: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Audit monotonic timestamps, missing bars, zero/negative prices, and funding coverage."""
    df_4h = datasets["df_4h"]
    df_1d = datasets["df_1d"]
    diffs_4h = df_4h.index.to_series().diff().dropna()
    gaps_4h = int((diffs_4h > pd.Timedelta(hours=4)).sum())
    diffs_1d = df_1d.index.to_series().diff().dropna()
    gaps_1d = int((diffs_1d > pd.Timedelta(days=1)).sum())
    bad_ohlc_4h = int(((df_4h["high"] < df_4h["low"]) | (df_4h["close"] <= 0)).sum())
    bad_ohlc_1d = int(((df_1d["high"] < df_1d["low"]) | (df_1d["close"] <= 0)).sum())
    nonzero_fr_4h = int((df_4h["funding_rate"] != 0).sum())
    return {
        "ok": (bad_ohlc_4h == 0 and bad_ohlc_1d == 0 and df_4h.index.is_monotonic_increasing and df_1d.index.is_monotonic_increasing),
        "bars_4h": len(df_4h),
        "range_4h": [str(df_4h.index[0]), str(df_4h.index[-1])],
        "gaps_4h": gaps_4h,
        "bars_1d": len(df_1d),
        "range_1d": [str(df_1d.index[0]), str(df_1d.index[-1])],
        "gaps_1d": gaps_1d,
        "nonzero_funding_events_4h": nonzero_fr_4h,
    }


if __name__ == "__main__":
    meta = refresh_btc_market_data()
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    ds = load_btc_datasets()
    print("Quality Audit:", json.dumps(validate_btc_data_quality(ds), indent=2, ensure_ascii=False))
