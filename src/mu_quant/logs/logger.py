"""Structured logging and exception recording for MU Specialist Quant System 2.0."""
from __future__ import annotations

import logging
from pathlib import Path

from mu_quant.config import LOGS_DIR


def get_logger(name: str = "mu_quant", log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    path = log_file or (LOGS_DIR / "system.log")
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger
