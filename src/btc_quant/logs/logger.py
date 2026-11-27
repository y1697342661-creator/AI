"""Structured logger for btc_quant (isolated log file logs/btc_system.log)."""
from __future__ import annotations

import logging
from pathlib import Path

from btc_quant.config import LOGS_DIR

LOG_FILE = LOGS_DIR / "btc_system.log"


def get_logger(name: str = "btc_quant") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger
