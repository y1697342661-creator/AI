"""Stage 6 Frozen Forward Paper-Trading Daily Runner.

Automatically:
1. Refreshes MU + benchmark market data (Yahoo -> Stooq safe fallback).
2. Validates data quality (blocks signal recording if quality check FAILS).
3. Verifies frozen strategy version v2.0.0 SHA-256 integrity.
4. Computes latest T-day close signals (Long primary + Short S3 experimental), target weight, stop price, and simulated order ticket.
5. Appends today's signal to data/paper_trading/signals_log.csv (append-only, NEVER overwrites).

Run:
    python src/mu_quant/paper_trading/daily_runner.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mu_quant.config import VERSIONS_DIR, load_active_config, load_strategy_version
from mu_quant.dashboard.server import build_dashboard_payload
from mu_quant.data.loader import refresh_all_market_data
from mu_quant.logs.logger import get_logger
from mu_quant.paper_trading.engine import record_daily_signal
from mu_quant.paper_trading.forward_tracker import update_forward_scorecard

log = get_logger("mu_quant.daily_runner")


def verify_version_freeze(version: str = "v2.0.0") -> dict:
    vfile = VERSIONS_DIR / f"{version}.json"
    freeze_file = VERSIONS_DIR / f"{version}.freeze.json"
    sha = hashlib.sha256(vfile.read_bytes()).hexdigest()
    if not freeze_file.exists():
        freeze_obj = {
            "version": version,
            "sha256": sha,
            "frozen": True,
            "rule": "前瞻性纸面交易期间冻结参数，严禁根据短期盈亏随意修改策略参数。",
        }
        freeze_file.write_text(json.dumps(freeze_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"ok": True, "created": True, "sha256": sha}
    saved = json.loads(freeze_file.read_text(encoding="utf-8"))
    ok = saved.get("sha256") == sha
    return {"ok": ok, "created": False, "sha256": sha, "expected": saved.get("sha256")}


def main() -> int:
    parser = argparse.ArgumentParser(description="MU 2.0 Daily Auto-Refresh & Paper Signal Runner")
    parser.add_argument("--no-refresh", action="store_true", help="Skip network market data refresh")
    parser.add_argument("--force-refresh", action="store_true", help="Force network refresh ignoring 4h cooldown")
    args = parser.parse_args()

    ac = load_active_config()
    ver = ac.get("active_version", "v2.0.0")
    cfg = load_strategy_version(ver)

    print("=" * 72)
    print(f"MU Specialist Quant System 2.0 · 每日自动更新与纸面信号 ({ver})")
    print("=" * 72)

    fz = verify_version_freeze(ver)
    if not fz["ok"]:
        print(f"[警告] 版本 {ver} 参数指纹与冻结快照不一致！请核查 configs/strategy_versions/{ver}.json")
    else:
        print(f"[版本冻结校验] OK | {ver} | SHA256={fz['sha256'][:16]}...")

    if not args.no_refresh:
        print("[1/3] 正在自动更新市场行情（MU + SMH/SOX/SPY/QQQ/NVDA）...")
        ref = refresh_all_market_data(force=args.force_refresh)
        if ref.get("skipped"):
            print(f"      跳过网络刷新：{ref['reason']}")
        else:
            mu_r = ref.get("symbols", {}).get("MU", {})
            print(f"      MU 更新状态: ok={mu_r.get('ok')} | source={mu_r.get('source')} | last={mu_r.get('last_date', '-')}")

    print("[2/3] 计算最新指标、市场环境与多空信号...")
    payload = build_dashboard_payload(
        long_mode=cfg["long_strategy"]["default_mode"],
        portfolio_mode=ac.get("portfolio_mode", "long_only"),
        start_date="2004-01-01",
    )
    if payload["quality"]["overall"] != "PASS":
        print(f"[阻断] 数据质检未通过：{payload['quality']}")
        return 1

    m = payload["market"]
    sig = payload["signals"]
    rp = payload["risk_plan"]
    tk = payload["simulated_ticket"]

    print("[3/3] 追加记录今日信号到纸面日志（只追加，不覆盖）...")
    rec = record_daily_signal({
        "signal_date": m["date"],
        "strategy_version": ver,
        "portfolio_mode": ac.get("portfolio_mode", "long_only"),
        "close_price": m["close"],
        "regime": payload["regime"]["code"],
        "long_signal": sig["long"]["signal"],
        "short_signal": sig["short"]["signal"],
        "target_weight": sig["target_weight"],
        "stop_price": rp["stop_price"],
        "note": sig["long"]["reason"],
    })
    fwd = update_forward_scorecard(
        market_date=m["date"],
        mu_open=m["open"],
        mu_close=m["close"],
        target_weight=sig["target_weight"],
        strategy_version=ver,
        initial_capital=float(ac.get("initial_cash", 100_000.0)),
    )

    print("-" * 72)
    print(f"最新交易日 : {m['date']}  | MU 收盘价: ${m['close']:.2f} ({m['chg_pct']*100:+.2f}%)")
    print(f"当前环境   : {payload['regime']['label']} (仅风控监控)")
    print(f"做多信号   : {sig['long']['signal']} | 目标仓位 {sig['long']['target_weight']*100:.1f}% ({sig['long']['reason']})")
    print(f"做空信号   : {sig['short']['signal']} (S3实验) | {sig['short']['reason']}")
    print(f"参考止损价 : ${rp['stop_price']:.2f} (止损距离 {rp['stop_distance_pct']*100:.1f}%)")
    print(f"模拟执行单 : {tk['action']} {tk['shares']} 股 @ ~${tk['est_fill_price']} (禁止自动实盘下单)")
    print(f"日志已追加 : {rec['recorded_at']}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
