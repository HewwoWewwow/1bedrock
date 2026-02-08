#!/usr/bin/env python3
"""Example Monte Carlo simulation using calibrated inputs from the spreadsheet.

This script demonstrates running a Monte Carlo simulation over the DCF model
to produce IRR distributions with confidence intervals.

Usage:
    python examples/run_monte_carlo.py

The simulation varies key uncertain inputs (costs, cap rate, rates) and
produces:
- Mean, median, std dev of IRRs
- 95% confidence intervals
- Probability of exceeding target returns
- Sensitivity analysis showing which inputs drive IRR variation
"""

import sys
from pathlib import Path
from datetime import date

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.calculations.monte_carlo import (
    DistributionType,
    InputDistribution,
    BaseInputs,
    MonteCarloConfig,
    run_monte_carlo,
    create_full_uncertainty_suite,
)
from src.calculations.detailed_cashflow import AssessedValueBasis


def main():
    """Run Monte Carlo simulation with calibrated spreadsheet inputs."""

    print("=" * 70)
    print("AUSTIN TIF MODEL - MONTE CARLO SIMULATION")
    print("=" * 70)
    print()

    # Base inputs calibrated to match Austin Working TIF Model v.010c.xlsx
    # These are the exact values that produced:
    #   - UNL IRR: 22.14% (spreadsheet: 22.38%)
    #   - LEV IRR: 33.24% (spreadsheet: 33.21%)
    base_inputs = BaseInputs(
        # Land & Construction
        land_cost=2_000_000,
        hard_costs=31_936_000,  # $127,744/unit * 250 units
        soft_cost_pct=0.20,
        construction_ltc=0.70,  # 70% LTC
        construction_rate=0.07,  # 7% construction loan rate

        # Timing (6+18+6+36 = 66 months)
        start_date=date(2025, 1, 1),
        predevelopment_months=6,
        construction_months=18,
        leaseup_months=6,
        operations_months=36,

        # Revenue
        monthly_gpr_at_stabilization=598_740,  # From spreadsheet
        vacancy_rate=0.06,  # 6% stabilized vacancy
        leaseup_pace=0.15,  # 15% per month during lease-up
        max_occupancy=0.94,  # 94% stabilized occupancy

        # Operating
        annual_opex_per_unit=3_940,  # $82,083/month / 250 units * 12
        total_units=250,

        # Property Tax (calibrated)
        existing_assessed_value=1_000_000,
        assessed_value_override=56_000_000,  # Calibrated TDC-based assessment
        assessment_growth_rate=0.02,
        assessed_value_basis=AssessedValueBasis.TDC,

        # Escalations
        market_rent_growth=0.02,  # 2% annual
        opex_growth=0.03,  # 3% annual
        prop_tax_growth=0.02,  # 2% annual

        # Permanent Loan (calibrated)
        perm_rate=0.06,  # 6%
        perm_amort_years=20,  # 20-year amortization
        perm_ltv_max=0.70,
        perm_dscr_min=1.25,
        perm_ltc_max=0.81,  # Calibrated to match spreadsheet (~$36.4M perm loan)

        # Exit
        exit_cap_rate=0.055,  # 5.5% exit cap
        reserves_pct=0.005,  # 0.5% of EGI

        # IRR Calculation (calibrated to spreadsheet methodology)
        exclude_land_from_irr=True,  # Land recovered at sale
        exclude_idc_from_irr=True,   # IDC is financing cost
        cap_perm_at_construction=True,  # No cash-out refinancing
    )

    # Define input distributions for uncertainty analysis
    distributions = [
        # Construction cost uncertainty (±10% CV, lognormal)
        InputDistribution(
            parameter="hard_costs",
            distribution=DistributionType.LOGNORMAL,
            mean=31_936_000,
            std=3_193_600,  # 10% coefficient of variation
            clip_min=25_000_000,  # Floor at ~80%
        ),

        # Soft cost uncertainty (18-25% range)
        InputDistribution(
            parameter="soft_cost_pct",
            distribution=DistributionType.TRIANGULAR,
            min_value=0.18,
            mode=0.20,
            max_value=0.25,
        ),

        # Exit cap rate uncertainty (5.0%-6.5% range)
        InputDistribution(
            parameter="exit_cap_rate",
            distribution=DistributionType.TRIANGULAR,
            min_value=0.050,
            mode=0.055,
            max_value=0.065,
        ),

        # Rent growth uncertainty (1%-4% range)
        InputDistribution(
            parameter="market_rent_growth",
            distribution=DistributionType.TRIANGULAR,
            min_value=0.01,
            mode=0.025,
            max_value=0.04,
        ),

        # Vacancy rate uncertainty (4%-10% range)
        InputDistribution(
            parameter="vacancy_rate",
            distribution=DistributionType.TRIANGULAR,
            min_value=0.04,
            mode=0.06,
            max_value=0.10,
        ),

        # Permanent loan rate uncertainty (5.5%-7.5% range)
        InputDistribution(
            parameter="perm_rate",
            distribution=DistributionType.TRIANGULAR,
            min_value=0.055,
            mode=0.060,
            max_value=0.075,
        ),

        # Construction rate uncertainty (6.5%-8.5% range)
        InputDistribution(
            parameter="construction_rate",
            distribution=DistributionType.TRIANGULAR,
            min_value=0.065,
            mode=0.070,
            max_value=0.085,
        ),

        # Lease-up pace uncertainty (10%-20% per month)
        InputDistribution(
            parameter="leaseup_pace",
            distribution=DistributionType.TRIANGULAR,
            min_value=0.10,
            mode=0.15,
            max_value=0.20,
        ),
    ]

    # Configure simulation
    config = MonteCarloConfig(
        base_inputs=base_inputs,
        distributions=distributions,
        n_iterations=1000,
        seed=42,  # For reproducibility
        confidence_level=0.95,
        parallel=True,
    )

    print("Running Monte Carlo simulation...")
    print(f"  Iterations: {config.n_iterations:,}")
    print(f"  Parameters varied: {len(distributions)}")
    print(f"  Confidence level: {config.confidence_level:.0%}")
    print()

    # Run simulation with progress callback
    def progress(completed, total):
        if completed % 100 == 0 or completed == total:
            pct = completed / total * 100
            print(f"  Progress: {completed:,}/{total:,} ({pct:.0f}%)", end="\r")

    result = run_monte_carlo(
        config,
        target_irr=0.20,  # 20% target for probability calculation
        progress_callback=progress,
    )

    print()
    print()

    # Print results
    print(result.summary())

    # Additional analysis: What drives IRR variance?
    print()
    print("=" * 60)
    print("TOP IRR DRIVERS (by absolute correlation)")
    print("=" * 60)

    # Sort by absolute levered correlation
    sorted_sens = sorted(
        result.sensitivities,
        key=lambda s: abs(s.correlation_levered),
        reverse=True
    )

    print(f"\n{'Parameter':<30} {'Corr (Lev)':<12} {'Corr (Unl)':<12}")
    print("-" * 54)
    for s in sorted_sens:
        print(f"{s.parameter:<30} {s.correlation_levered:>+10.3f}  {s.correlation_unlevered:>+10.3f}")

    # Percentile analysis
    print()
    print("=" * 60)
    print("LEVERED IRR PERCENTILES")
    print("=" * 60)
    print()
    for pct, val in sorted(result.levered_irr_percentiles.items()):
        print(f"  {pct:>3}th percentile: {val:>8.2%}")

    # Scenario analysis: Downside / Base / Upside
    print()
    print("=" * 60)
    print("SCENARIO SUMMARY")
    print("=" * 60)
    print()
    print(f"  Downside (10th %ile): Levered IRR = {result.levered_irr_percentiles[10]:.2%}")
    print(f"  Base Case (50th %ile): Levered IRR = {result.levered_irr_percentiles[50]:.2%}")
    print(f"  Upside (90th %ile):   Levered IRR = {result.levered_irr_percentiles[90]:.2%}")
    print()

    # Risk metrics
    print("=" * 60)
    print("RISK METRICS")
    print("=" * 60)
    print()
    print(f"  Probability of Levered IRR > 20%: {result.prob_levered_above_target:.1%}")
    print(f"  Probability of Levered IRR > 30%: {float(sum(1 for r in result.iterations if r.levered_irr > 0.30)) / len(result.iterations):.1%}")
    print(f"  Probability of Positive IRR: {result.prob_levered_positive:.1%}")

    # Value at Risk (5th percentile)
    var_5 = result.levered_irr_percentiles[5]
    print(f"  Value at Risk (5th %ile): {var_5:.2%}")

    print()
    print("=" * 60)
    print("Simulation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
