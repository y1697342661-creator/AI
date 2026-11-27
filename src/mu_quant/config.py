"""MU Specialist Quant System 2.0 — Core Configuration & Versioned Parameters."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT: Path = Path(__file__).resolve().parents[2]
CONFIGS_DIR: Path = ROOT / "configs"
VERSIONS_DIR: Path = CONFIGS_DIR / "strategy_versions"
DATA_RAW: Path = ROOT / "data" / "raw"
DATA_PROCESSED: Path = ROOT / "data" / "processed"
DATA_PAPER: Path = ROOT / "data" / "paper_trading"
LOGS_DIR: Path = ROOT / "logs"

for _d in (CONFIGS_DIR, VERSIONS_DIR, DATA_RAW, DATA_PROCESSED, DATA_PAPER, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TRADING_DAYS: int = 252
LIVE_AUTO_ORDER_FORBIDDEN: bool = True


def load_active_config() -> dict[str, Any]:
    p = CONFIGS_DIR / "active_config.json"
    if not p.exists():
        return {"active_version": "v2.0.0", "portfolio_mode": "long_only", "initial_cash": 100000.0, "allow_live_auto_order": False}
    cfg = json.loads(p.read_text(encoding="utf-8-sig"))
    cfg["allow_live_auto_order"] = False  # Hard safety lock
    return cfg


def load_strategy_version(version: str | None = None) -> dict[str, Any]:
    if version is None:
        version = load_active_config().get("active_version", "v2.0.0")
    p = VERSIONS_DIR / f"{version}.json"
    if not p.exists():
        raise FileNotFoundError(f"Strategy version not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8-sig"))
    data.setdefault("execution", {})["allow_live_auto_order"] = False
    return data


def list_strategy_versions() -> list[str]:
    return sorted(p.stem for p in VERSIONS_DIR.glob("*.json"))
