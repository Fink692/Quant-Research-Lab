# Getting Started

## Installation

```bash
git clone https://github.com/Fink692/Quant-Research-Lab.git
cd Quant-Research-Lab
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

## Run The Research Workflows

The script interface:

```bash
python scripts/run_risk_dashboard.py
python scripts/run_pairs_trading.py
python scripts/run_macro_regime_model.py
python scripts/run_vol_surface.py
python scripts/run_backtest.py
```

The package command interface:

```bash
qrl-risk
qrl-pairs
qrl-macro
qrl-vol
qrl-backtest
```

## Test And Validate

```bash
pytest
ruff check src tests scripts
black --check src tests scripts
mypy src --ignore-missing-imports
```

## Optional API Keys

Copy `.env.example` to `.env` if you have optional credentials.

```bash
copy .env.example .env
```

The project runs without keys by using public endpoints or deterministic fallback data where needed.
