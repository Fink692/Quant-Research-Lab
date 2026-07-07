"""Small local cache layer for downloaded research data."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from quant_research_lab.utils.config import project_root


class DataCache:
    """File-backed cache for pandas objects and Python payloads."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else project_root() / "data" / "cache"
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(*parts: object) -> str:
        """Create a stable cache key from arbitrary query parts."""
        payload = "|".join(str(part) for part in parts)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def path(self, key: str, suffix: str) -> Path:
        """Return a cache path for a key and file suffix."""
        return self.root / f"{key}.{suffix.lstrip('.')}"

    def get_pickle(self, key: str) -> Any | None:
        """Load a pickled cache object, returning None on a miss."""
        path = self.path(key, "pkl")
        if not path.exists():
            return None
        with path.open("rb") as file:
            return pickle.load(file)

    def set_pickle(self, key: str, value: Any) -> Path:
        """Persist a Python object in the cache."""
        path = self.path(key, "pkl")
        with path.open("wb") as file:
            pickle.dump(value, file)
        return path

    def get_frame(self, key: str) -> pd.DataFrame | None:
        """Load a cached DataFrame, returning None on a miss."""
        path = self.path(key, "parquet")
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception:
                csv_path = self.path(key, "csv")
                if csv_path.exists():
                    return pd.read_csv(csv_path, index_col=0, parse_dates=True)
                return None
        csv_path = self.path(key, "csv")
        if csv_path.exists():
            return pd.read_csv(csv_path, index_col=0, parse_dates=True)
        return None

    def set_frame(self, key: str, frame: pd.DataFrame) -> Path:
        """Cache a DataFrame as parquet when available, otherwise CSV."""
        parquet_path = self.path(key, "parquet")
        try:
            frame.to_parquet(parquet_path)
            return parquet_path
        except Exception:
            csv_path = self.path(key, "csv")
            frame.to_csv(csv_path)
            return csv_path
