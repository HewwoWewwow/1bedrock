#!/usr/bin/env python3
"""Compare Market Rate vs Mixed-Income scenarios using Monte Carlo simulation.

This script runs Monte Carlo simulations on both scenarios to understand
how the IRR difference distribution behaves under uncertainty.

Usage:
    python examples/compare_scenarios_monte_carlo.py
"""

import sys
from pathlib import Path
from datetime import date
from dataclasses import replace
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.calculations.monte_carlo import (
    DistributionType,
    InputDistribution,
    BaseInputs,
    MonteCarloConfig,
    run_monte_carlo,
)
from src.calculations.detailed_cashflow import AssessedValueBasis


def main():
    """Compare market vs mixed-income scenarios with Monte Carlo."""

    print("=" * 70)
    print("SCENARIO COMPARISON: Market Rate vs Mixed-Income")
    print("=" * 70)
    print()

    # Common base inputs (calibrated to spreadsheet)
    common_inputs = {
        "land_cost": 2_000_000,
        "hard_costs": 31_936_000,
        "soft_cost_pct": 0.20,
        "construction_ltc": 0.70,
        "construction_rate": 0.07,
        "start_date": date(2025, 1, 1),
        "predevelopment_months": 6,
        "construction_months": 18,
        "leaseup_months": 6,
        "operations_months": 36,
        "monthly_gpr_at_stabilization": 598_740,
        "vacancy_rate": 0.06,
        "leaseup_pace": 0.15,
        "max_occupancy": 0.94,
        "annual_opex_per_unit": 3_940,
        "total_units": 250,
        "existing_assessed_value": 1_000_000,
        "assessed_value_override": 56_000_000,
        "assessed_value_basis": AssessedValueBasis.TDC,
        "assessment_growth_rate": 0.02,
        "market_rent_growth": 0.02,
        "opex_growth": 0.03,
        "prop_tax_growth": 0.02,
        "perm_rate": 0.06,
        "perm_amort_years": 20,
        "perm_ltv_max": 0.70,
        "perm_dscr_min": 1.25,
        "perm_ltc_max": 0.81,
        "exit_cap_rate": 0.055,
        "reserves_pct": 0.005,
        "exclude_land_from_irr": True,
        "exclude_idc_from_irr": True,
        "cap_perm_at_construction": True,
    }

    # Market Rate scenario: 0% affordable
    market_inputs = BaseInputs(
        **common_inputs,
        affordable_pct=0.0,
        affordable_rent_discount=0.0,
        affordable_rent_growth=0.01,
    )

    # Mixed-Income scenario: 10% affordable at 60% AMI (40% discount)
    mixed_inputs = BaseInputs(
        **common_inputs,
        affordable_pct=0.10,  # 10% affordable units
        affordable_rent_discount=0.40,  # 40% rent discount for affordable
        affordable_rent_growth=0.01,  # 1% growth for affordable rents
    )

    # Common uncertainty distributions
    distributions = [
        InputDistribution(
            parameter="hard_costs",
            distribution=DistributionType.LOGNORMAL,
            mean=31_936_000,
            std=3_193_600,
            clip_min=25_000_000,
        ),
        InputDistribution(
            parameter="exit_cap_rate",
            distribution=DistributionType.TRIANGULAR,
            min_value=0.050,
            mode=0.055,
            max_value=0.065,
        ),
        InputDistribution(
            parameter="market_rent_growth",
            distribution=DistributionType.TRIANGULAR,
            min_value=0.01,
            mode=0.025,
            max_value=0.04,
        ),
        InputDistribution(
            parameter="vacancy_rate",
            distribution=DistributionType.TRIANGULAR,
            min_value=0.04,
            mode=0.06,
            max_value=0.10,
        ),
        InputDistribution(
            parameter="perm_rate",
            distribution=DistributionType.TRIANGULAR,
            min_value=0.055,
            mode=0.060,
            max_value=0.075,
        ),
    ]

    n_iterations = 500
    seed = 42

    print("Running Market Rate scenario...")
    market_config = MonteCarloConfig(
        base_inputs=market_inputs,
        distributions=distributions,
        n_iterations=n_iterations,
        seed=seed,
        parallel=True,
    )
    market_result = run_monte_carlo(market_config, target_irr=0.20)

    print("Running Mixed-Income scenario...")
    mixed_config = MonteCarloConfig(
        base_inputs=mixed_inputs,
        distributions=distributions,
        n_iterations=n_iterations,
        seed=seed,  # Same seed for paired comparison
        parallel=True,
    )
    mixed_result = run_monte_carlo(mixed_config, target_irr=0.20)

    # Calculate IRR differences (paired samples since same seed)
    market_irrs = np.array([r.levered_irr for r in market_result.iterations])
    mixed_irrs = np.array([r.levered_irr for r in mixed_result.iterations])
    irr_diffs = mixed_irrs - market_irrs  # Positive = mixed outperforms

    # Convert to basis points
    irr_diffs_bps = irr_diffs * 10000

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    print("MARKET RATE SCENARIO")
    print("-" * 40)
    print(f"  Mean Levered IRR:   {market_result.levered_irr_mean:>8.2%}")
    print(f"  Median Levered IRR: {market_result.levered_irr_median:>8.2%}")
    print(f"  Std Dev:            {market_result.levered_irr_std:>8.2%}")
    print(f"  95% CI:             [{market_result.levered_irr_ci_lower:>7.2%}, {market_result.levered_irr_ci_upper:>7.2%}]")
    print()

    print("MIXED-INCOME SCENARIO (10% @ 60% AMI)")
    print("-" * 40)
    print(f"  Mean Levered IRR:   {mixed_result.levered_irr_mean:>8.2%}")
    print(f"  Median Levered IRR: {mixed_result.levered_irr_median:>8.2%}")
    print(f"  Std Dev:            {mixed_result.levered_irr_std:>8.2%}")
    print(f"  95% CI:             [{mixed_result.levered_irr_ci_lower:>7.2%}, {mixed_result.levered_irr_ci_upper:>7.2%}]")
    print()

    print("IRR DIFFERENCE (Mixed - Market)")
    print("-" * 40)
    print(f"  Mean Difference:    {np.mean(irr_diffs):>+8.2%} ({np.mean(irr_diffs_bps):>+.0f} bps)")
    print(f"  Median Difference:  {np.median(irr_diffs):>+8.2%} ({np.median(irr_diffs_bps):>+.0f} bps)")
    print(f"  Std Dev:            {np.std(irr_diffs):>8.2%} ({np.std(irr_diffs_bps):.0f} bps)")
    print()

    # Percentiles of difference
    print("IRR DIFFERENCE PERCENTILES")
    print("-" * 40)
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        val = np.percentile(irr_diffs, pct)
        val_bps = np.percentile(irr_diffs_bps, pct)
        print(f"  {pct:>3}th percentile: {val:>+8.2%} ({val_bps:>+6.0f} bps)")
    print()

    # Policy question: What incentive is needed to make mixed-income competitive?
    target_diff_bps = 150  # 150 bps premium target
    prob_meets_target = np.mean(irr_diffs_bps >= -target_diff_bps)  # Within 150 bps of market

    print("=" * 70)
    print("POLICY ANALYSIS")
    print("=" * 70)
    print()
    print(f"  IRR gap (mean):               {np.mean(irr_diffs_bps):>+.0f} bps")
    print(f"  Probability within 150 bps:   {prob_meets_target:.1%}")
    print()

    # What % of scenarios need incentives?
    needs_incentive = np.mean(irr_diffs_bps < 0)
    print(f"  Scenarios where mixed underperforms: {needs_incentive:.1%}")

    # Calculate required incentive to close gap at 95th percentile of difference
    gap_95th = np.percentile(irr_diffs_bps, 5)  # 5th percentile = worst case
    print(f"  Worst case gap (5th %ile):    {gap_95th:>+.0f} bps")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    if np.mean(irr_diffs) < 0:
        print(f"Mixed-income scenario underperforms market by {-np.mean(irr_diffs_bps):.0f} bps on average.")
        print(f"To achieve parity, incentives worth ~{-np.mean(irr_diffs)*100:.2f}% IRR improvement needed.")
    else:
        print(f"Mixed-income scenario outperforms market by {np.mean(irr_diffs_bps):.0f} bps on average.")
    print()


if __name__ == "__main__":
    main()
