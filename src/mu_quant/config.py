"""US Stock Screener & Alpaca Paper Executor — Core Paths & Safety Constants."""
from __future__ import annotations

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[2]
CONFIGS_DIR: Path = ROOT / "configs"
LOGS_DIR: Path = ROOT / "logs"

for _d in (CONFIGS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TRADING_DAYS: int = 252
LIVE_AUTO_ORDER_FORBIDDEN: bool = True
