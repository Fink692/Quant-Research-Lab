"""Logging setup for command-line research runs."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(name: str = "quant_research_lab", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a consistent console logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def log_path(logger: logging.Logger, label: str, path: str | Path) -> None:
    """Log a generated artifact path."""
    logger.info("%s saved to %s", label, Path(path).resolve())
