"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the repository root from an installed or source checkout."""
    return Path(__file__).resolve().parents[3]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return a dictionary.

    Parameters
    ----------
    path:
        Absolute path or path relative to the repository root.
    """
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = project_root() / candidate
    if not candidate.exists():
        raise FileNotFoundError(f"Configuration file not found: {candidate}")
    with candidate.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {candidate}")
    return data


def ensure_directories(*paths: str | Path) -> None:
    """Create directories if they do not already exist."""
    for path in paths:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = project_root() / candidate
        candidate.mkdir(parents=True, exist_ok=True)
