# 1Bedrock

A transparent, auditable real estate financial modeling platform for multifamily development analysis.

## Why 1Bedrock?

Traditional real estate models are black boxes. You get a number, but you can't see how it was calculated. 1Bedrock is different:

- **Transparent calculations** - Every formula is documented and traceable
- **Audit trail** - See exactly how each value was computed with actual inputs
- **Single source of truth** - One calculation engine, consistent results everywhere

## Features

### Core Financial Modeling
- Full DCF model with monthly granularity
- Sources & Uses with iterative IDC solve
- Construction and permanent debt modeling
- Mezzanine debt and preferred equity support
- IRR, equity multiple, yield-on-cost, and DSCR calculations

### Transparency Engine
- **Formula Registry** - 45+ documented formulas with symbolic definitions
- **Runtime Tracing** - Captures actual values used in each calculation
- **Drill-down UI** - Click any value to see its formula and inputs

### Analysis Tools
- **Monte Carlo Simulation** - Probability distributions for key inputs
- **Sensitivity Analysis** - Identify which inputs matter most
- **TIF Grid Search** - Find optimal TIF parameters to hit IRR targets
- **Scenario Comparison** - Market rate vs. mixed-income analysis

### Austin-Specific Features
- TIF (Tax Increment Financing) - lump sum, abatement, and stream treatments
- SMART Housing fee waivers with tier-based calculations
- City tax abatement for affordable housing
- Austin taxing authority stack (City, County, AISD, ACC, Central Health)

## Installation

```bash
# Clone the repository
git clone https://github.com/HewwoWewwow/1bedrock.git
cd 1bedrock

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
from src.models.project import ProjectInputs, Scenario
from src.calculations.detailed_cashflow import calculate_deal

# Create inputs with defaults
inputs = ProjectInputs(
    target_units=200,
    land_cost=3_000_000,
    hard_cost_per_unit=155_000,
)

# Run the model
result = calculate_deal(inputs, Scenario.MARKET)

# Access results
print(f"TDC: ${result.sources_uses.tdc:,.0f}")
print(f"Levered IRR: {result.levered_irr:.2%}")
print(f"Equity Multiple: {result.equity_multiple:.2f}x")

# View calculation traces
if result.trace_context:
    for key, trace in result.trace_context.traces.items():
        print(f"{key}: {trace.computed_formula}")
```

### Monte Carlo Simulation

```python
from src.calculations.monte_carlo import (
    run_monte_carlo,
    MonteCarloConfig,
    create_full_uncertainty_suite,
)

# Create uncertainty distributions
distributions = create_full_uncertainty_suite()

# Run simulation
config = MonteCarloConfig(iterations=1000, seed=42)
mc_result = run_monte_carlo(inputs, distributions, config)

print(f"IRR P50: {mc_result.percentiles['levered_irr'][50]:.2%}")
print(f"IRR P10: {mc_result.percentiles['levered_irr'][10]:.2%}")
print(f"Probability IRR > 15%: {mc_result.probability_above_threshold('levered_irr', 0.15):.1%}")
```

## Project Structure

```
1bedrock/
├── src/
│   ├── calculations/
│   │   ├── detailed_cashflow.py  # Main DCF engine (SINGLE SOURCE OF TRUTH)
│   │   ├── formula_registry.py   # Formula definitions
│   │   ├── trace.py              # Calculation tracing
│   │   ├── monte_carlo.py        # Simulation engine
│   │   ├── sources_uses.py       # Capital stack
│   │   ├── draw_schedule.py      # Equity-first waterfall
│   │   ├── property_tax.py       # TIF and tax calculations
│   │   └── ...
│   └── models/
│       └── project.py            # ProjectInputs dataclass
├── ui/
│   ├── app.py                    # Main Streamlit app
│   └── components/
│       ├── calculation_trace_view.py  # Transparency UI
│       ├── spreadsheet_debug_view.py  # Debug views
│       └── ...
├── tests/
└── pyproject.toml
```

## Key Concepts

### Single Source of Truth
All calculations flow through `calculate_deal()`. This ensures:
- Consistent results across UI, API, and Monte Carlo
- One place to fix bugs or add features
- Traceable calculation chain

### Equity-First Waterfall
Development costs are funded in order:
1. **Equity** - Developer equity funds first
2. **TIF** - TIF lump sum (if applicable) funds second
3. **Debt** - Construction loan funds last

This matches real-world draw mechanics and ensures accurate IDC calculation.

### Formula Tracing
Every calculation can be traced:
```python
# TDC trace shows:
# Formula: land + hard_costs + soft_costs + idc
# Values: $3.00M + $36.75M + $16.79M + $6.47M = $63.00M
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
