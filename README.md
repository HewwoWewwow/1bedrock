# Austin TIF Model

A comprehensive financial modeling tool for analyzing Tax Increment Financing (TIF) and affordable housing incentives in Austin, Texas.

## Overview

This project provides a detailed discounted cash flow (DCF) model for real estate development projects, with special focus on:

- **TIF (Tax Increment Financing)** analysis - lump sum, abatement, and stream treatments
- **SMART Housing fee waivers** - tier-based fee waiver calculations
- **City tax abatement** for affordable housing units
- **Mixed-income scenario comparison** - market rate vs. affordable housing analysis
- **Property tax scheduling** with Austin's taxing authority stack

## Features

- Full DCF model with monthly granularity
- Sources & Uses analysis
- Construction and permanent debt modeling
- Mezzanine debt and preferred equity support
- IRR, equity multiple, and DSCR calculations
- Interactive Streamlit web UI
- Scenario comparison and sensitivity analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/austin-tif-model.git
cd austin-tif-model

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Usage

### Web UI

```bash
streamlit run ui/app.py
```

### Python API

```python
from src.calculations.detailed_cashflow import generate_detailed_cash_flow
from src.calculations.property_tax import get_austin_tax_stack

# Get Austin tax rates
tax_stack = get_austin_tax_stack()
print(f"Total tax rate: {tax_stack.total_rate_decimal:.4%}")

# Generate detailed cash flow
result = generate_detailed_cash_flow(
    predevelopment_months=6,
    construction_months=18,
    leaseup_months=6,
    operations_months=12,
    land_cost=2_000_000,
    hard_costs=25_000_000,
    soft_cost_pct=0.20,
    # ... additional parameters
)

print(f"Levered IRR: {result.levered_irr:.2%}")
```

## Project Structure

```
austin-tif-model/
├── src/
│   ├── calculations/
│   │   ├── detailed_cashflow.py  # Main DCF engine
│   │   ├── property_tax.py       # TIF and tax calculations
│   │   ├── sources_uses.py       # Sources & Uses
│   │   └── ...
│   └── models/
│       └── inputs.py             # Data models
├── ui/
│   ├── app.py                    # Main Streamlit app
│   └── components/
│       ├── detailed_cashflow_view.py
│       ├── tif_analysis_view.py
│       └── ...
├── tests/
├── config/
└── pyproject.toml
```

## Austin Tax Rates (2024/2025)

| Authority | Rate (per $100) | TIF Participating |
|-----------|-----------------|-------------------|
| City of Austin | $0.5740 | Yes |
| Travis County | $0.3758 | No |
| Austin ISD | $0.9252 | No |
| Austin CC | $0.1034 | No |
| Central Health | $0.0980 | No |
| **Total** | **$2.0764** | |

## License

MIT License
