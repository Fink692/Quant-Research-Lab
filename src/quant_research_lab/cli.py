"""Console entry points for Quant Research Lab.

The command wrappers execute the repository scripts so users can choose either
``python scripts/run_*.py`` or installed commands such as ``qrl-backtest``.
"""

from __future__ import annotations

import runpy

from quant_research_lab.utils.config import project_root


def _run_script(script_name: str) -> None:
    script_path = project_root() / "scripts" / script_name
    if not script_path.exists():
        raise FileNotFoundError(
            f"Could not locate {script_path}. Console commands are intended for local "
            "source checkouts or editable installs."
        )
    runpy.run_path(str(script_path), run_name="__main__")


def run_risk_dashboard() -> None:
    """Run the multi-asset portfolio risk dashboard."""
    _run_script("run_risk_dashboard.py")


def run_pairs_trading() -> None:
    """Run the pairs trading research workflow."""
    _run_script("run_pairs_trading.py")


def run_macro_regime_model() -> None:
    """Run the macro regime detection workflow."""
    _run_script("run_macro_regime_model.py")


def run_vol_surface() -> None:
    """Run the implied volatility surface builder."""
    _run_script("run_vol_surface.py")


def run_backtest() -> None:
    """Run the institutional backtesting examples."""
    _run_script("run_backtest.py")
