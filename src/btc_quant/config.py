"""Configuration & immutable strategy version management for BTC Perpetual Quant System."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = ROOT / "configs" / "btc"
VERSIONS_DIR = CONFIGS_DIR / "strategy_versions"
ACTIVE_CONFIG_FILE = CONFIGS_DIR / "active_config.json"

DATA_DIR = ROOT / "data" / "btc"
RAW_DIR = DATA_DIR / "raw"
PAPER_DIR = DATA_DIR / "paper_trading"
OKX_DEMO_DIR = DATA_DIR / "okx_demo"
LOGS_DIR = ROOT / "logs"

# Dedicated BTC OKX demo secrets file (gitignored via configs/*.local.json, isolated from MU)
OKX_DEMO_SECRETS_FILE = ROOT / "configs" / "btc_okx_demo_secrets.local.json"

for _d in (CONFIGS_DIR, VERSIONS_DIR, RAW_DIR, PAPER_DIR, OKX_DEMO_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def list_strategy_versions() -> list[str]:
    if not VERSIONS_DIR.exists():
        return []
    return sorted(
        p.stem for p in VERSIONS_DIR.glob("v*.json") if not p.name.endswith(".freeze.json")
    )


def load_strategy_version(version: str) -> dict[str, Any]:
    p = VERSIONS_DIR / f"{version}.json"
    if not p.exists():
        raise FileNotFoundError(f"BTC strategy version not found: {p}")
    return load_json(p)


def load_active_config() -> dict[str, Any]:
    if ACTIVE_CONFIG_FILE.exists():
        active = load_json(ACTIVE_CONFIG_FILE)
        ver = active.get("active_version", "v1.0.0")
        return load_strategy_version(ver)
    versions = list_strategy_versions()
    if versions:
        return load_strategy_version(versions[-1])
    return {}
