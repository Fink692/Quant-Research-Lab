"""Run implied volatility surface research."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from quant_research_lab.data.market_data import download_prices
from quant_research_lab.data.options_data import generate_synthetic_options_chain, load_options_chain
from quant_research_lab.derivatives.volatility_surface import (
    build_surface_grid,
    clean_options_chain,
    enrich_options_chain,
    iv_term_structure,
)
from quant_research_lab.utils.config import ensure_directories, load_yaml
from quant_research_lab.utils.logging import log_path, setup_logging
from quant_research_lab.visualization.charts import save_3d_surface, save_line_chart, save_options_smile
from quant_research_lab.visualization.reports import save_dataframe, write_markdown_report


def _spot_from_market(ticker: str, fallback_spot: float) -> float:
    try:
        prices = download_prices([ticker], start="2024-01-01", end=None)
        value = float(prices[ticker].dropna().iloc[-1])
        return value if value > 0 else fallback_spot
    except Exception:
        return fallback_spot


def _save_price_comparison(options: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(options["mid"], options["theoretical_price"], alpha=0.6)
    limit = max(options["mid"].max(), options["theoretical_price"].max())
    ax.plot([0, limit], [0, limit], color="#333333", linestyle="--", linewidth=1)
    ax.set_xlabel("Market Mid Price")
    ax.set_ylabel("Black-Scholes Theoretical Price")
    ax.set_title("Market vs Theoretical Option Prices")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _save_greeks_chart(options: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    atm = options.assign(atm_distance=(options["moneyness"] - 1).abs()).sort_values("atm_distance").head(80)
    melted = atm.melt(id_vars=["moneyness"], value_vars=["delta", "gamma", "theta", "vega", "rho"], var_name="greek", value_name="value")
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.lineplot(data=melted, x="moneyness", y="value", hue="greek", ax=ax)
    ax.set_title("Greeks Near At-The-Money")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    """Load options data, clean it, estimate IVs and Greeks, and save surface outputs."""
    logger = setup_logging("vol_surface")
    ensure_directories("outputs/figures", "outputs/reports")
    cfg = load_yaml("configs/options.yaml")["options"]
    ticker = cfg["ticker"]
    fallback_spot = float(cfg["fallback_spot"])
    raw = load_options_chain(
        ticker=ticker,
        manual_csv_path=cfg.get("manual_csv_path"),
        max_expirations=int(cfg["max_expirations"]),
        fallback_spot=fallback_spot,
        risk_free_rate=float(cfg["risk_free_rate"]),
        dividend_yield=float(cfg["dividend_yield"]),
        valuation_date=cfg.get("valuation_date"),
    )
    using_synthetic = raw.get("contract_symbol", pd.Series(dtype=str)).astype(str).str.startswith("SYN").any()
    spot = fallback_spot if using_synthetic else _spot_from_market(ticker, fallback_spot)
    cleaned = clean_options_chain(
        raw,
        valuation_date=cfg.get("valuation_date"),
        min_days_to_expiry=int(cfg["min_days_to_expiry"]),
        max_days_to_expiry=int(cfg["max_days_to_expiry"]),
        min_open_interest=int(cfg["min_open_interest"]),
        min_volume=int(cfg["min_volume"]),
    )
    if len(cleaned.loc[cleaned["option_type"].eq("c")]) < 4:
        logger.warning("Provider options chain too sparse after cleaning; using synthetic chain.")
        spot = fallback_spot
        raw = generate_synthetic_options_chain(
            spot=spot,
            risk_free_rate=float(cfg["risk_free_rate"]),
            dividend_yield=float(cfg["dividend_yield"]),
            valuation_date=cfg.get("valuation_date"),
        )
        cleaned = clean_options_chain(
            raw,
            valuation_date=cfg.get("valuation_date"),
            min_days_to_expiry=int(cfg["min_days_to_expiry"]),
            max_days_to_expiry=int(cfg["max_days_to_expiry"]),
            min_open_interest=int(cfg["min_open_interest"]),
            min_volume=int(cfg["min_volume"]),
        )
    enriched = enrich_options_chain(
        cleaned,
        spot=spot,
        risk_free_rate=float(cfg["risk_free_rate"]),
        dividend_yield=float(cfg["dividend_yield"]),
        valuation_date=cfg.get("valuation_date"),
    )
    term = iv_term_structure(enriched)
    xx, yy, zz = build_surface_grid(enriched, option_type="c")

    figures = ROOT / "outputs" / "figures"
    reports = ROOT / "outputs" / "reports"
    clean_path = save_dataframe(enriched, reports / "options_cleaned_chain.csv")
    save_dataframe(term, reports / "options_iv_term_structure.csv")
    save_options_smile(enriched, figures / "options_iv_smile.png", option_type="c")
    save_line_chart(
        term.pivot(index="expiration", columns="option_type", values="implied_volatility"),
        figures / "options_iv_term_structure.png",
        "ATM Implied Volatility Term Structure",
        "Implied Volatility",
    )
    save_3d_surface(xx, yy, zz, figures / "options_3d_iv_surface.png")
    _save_greeks_chart(enriched, figures / "options_greeks_chart.png")
    _save_price_comparison(enriched, figures / "options_market_vs_theoretical.png")
    report_path = write_markdown_report(
        "Implied Volatility Surface Builder",
        {
            "Methodology": (
                "The module cleans option quotes, estimates Black-Scholes implied volatility through "
                "root-finding, calculates Greeks, transforms strikes into moneyness, and interpolates "
                "a call implied-volatility surface."
            ),
            "Spot Used": f"{ticker} spot used for calculations: {spot:.2f}.",
        },
        reports / "options_vol_surface_report.md",
    )
    log_path(logger, "Cleaned options chain", clean_path)
    log_path(logger, "Options report", report_path)
    logger.info("Volatility surface build complete.")


if __name__ == "__main__":
    main()
