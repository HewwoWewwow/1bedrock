#!/usr/bin/env python3
"""TIF Grid Search: Find cap rate and term combinations that meet the 250 bps target.

This script runs a grid search over TIF configurations to find which combinations
of cap rate and term consistently produce IRR gaps within the target tolerance.

Key assumptions:
- Market rate deals do NOT receive TIF
- Mixed-income deals receive TIF lump sum (front-loaded)
- Mixed-income deals require 15% affordable units at 60% AMI
- Target: Mixed-income IRR should be within 250 bps of market IRR

Usage:
    python examples/tif_grid_search.py
"""

import sys
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.calculations.monte_carlo import (
    DistributionType,
    InputDistribution,
    BaseInputs,
    run_tif_grid_search,
)
from src.calculations.detailed_cashflow import (
    AssessedValueBasis,
    generate_detailed_cash_flow,
)
from src.calculations.property_tax import get_austin_tax_stack


def main():
    print("=" * 70)
    print("TIF GRID SEARCH: Finding Optimal Cap Rate and Term Combinations")
    print("=" * 70)
    print()
    print("Configuration:")
    print("  - Market rate deals: NO TIF")
    print("  - Mixed-income deals: TIF lump sum (front-loaded)")
    print("  - Affordable requirement: 15% of units at 60% AMI (40% discount)")
    print("  - Target: Mixed IRR within 250 bps of Market IRR")
    print()

    # Base inputs calibrated to spreadsheet
    base_inputs = BaseInputs(
        land_cost=2_000_000,
        hard_costs=31_936_000,
        soft_cost_pct=0.20,
        construction_ltc=0.70,
        construction_rate=0.07,
        start_date=date(2025, 1, 1),
        predevelopment_months=6,
        construction_months=18,
        leaseup_months=6,
        operations_months=36,
        monthly_gpr_at_stabilization=598_740,
        vacancy_rate=0.06,
        leaseup_pace=0.15,
        max_occupancy=0.94,
        annual_opex_per_unit=3_940,
        total_units=250,
        existing_assessed_value=1_000_000,
        assessed_value_override=56_000_000,
        assessed_value_basis=AssessedValueBasis.TDC,
        assessment_growth_rate=0.02,
        market_rent_growth=0.02,
        opex_growth=0.03,
        prop_tax_growth=0.02,
        perm_rate=0.06,
        perm_amort_years=20,
        perm_ltv_max=0.70,
        perm_dscr_min=1.25,
        perm_ltc_max=0.81,
        exit_cap_rate=0.055,
        reserves_pct=0.005,
        exclude_land_from_irr=True,
        exclude_idc_from_irr=True,
        cap_perm_at_construction=True,
        # Mixed-income configuration: 15% affordable at 60% AMI
        affordable_pct=0.15,
        affordable_rent_discount=0.40,  # 40% discount for 60% AMI
        affordable_rent_growth=0.01,
    )

    # First, calculate the NOI delta (market - mixed) for TIF sizing
    print("Step 1: Calculate NOI delta between scenarios...")
    print()

    tax_stack = get_austin_tax_stack()

    # Market scenario (0% affordable, no TIF)
    market_inputs = base_inputs.to_dict()
    market_inputs["affordable_pct"] = 0.0
    market_inputs["tif_treatment"] = "none"

    market_result = generate_detailed_cash_flow(
        tax_stack=tax_stack,
        **market_inputs
    )

    # Mixed scenario (15% affordable, no TIF yet - to measure gap)
    mixed_inputs = base_inputs.to_dict()
    mixed_inputs["tif_treatment"] = "none"

    mixed_result = generate_detailed_cash_flow(
        tax_stack=tax_stack,
        **mixed_inputs
    )

    # Calculate annual NOI delta at stabilization
    # Use the first full year of operations
    ops_start = market_result.construction_end + market_result.leaseup_end - market_result.construction_end + 1
    market_noi_monthly = market_result.periods[ops_start].operations.noi
    mixed_noi_monthly = mixed_result.periods[ops_start].operations.noi

    noi_delta_annual = (market_noi_monthly - mixed_noi_monthly) * 12

    print(f"  Market Monthly NOI: ${market_noi_monthly:,.0f}")
    print(f"  Mixed Monthly NOI:  ${mixed_noi_monthly:,.0f}")
    print(f"  Annual NOI Delta:   ${noi_delta_annual:,.0f}")
    print()
    print(f"  Market Levered IRR: {market_result.levered_irr:.2%}")
    print(f"  Mixed Levered IRR:  {mixed_result.levered_irr:.2%}")
    print(f"  IRR Gap (no TIF):   {(mixed_result.levered_irr - market_result.levered_irr) * 10000:+.0f} bps")
    print()

    # Define uncertainty distributions for Monte Carlo
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
            parameter="perm_rate",
            distribution=DistributionType.TRIANGULAR,
            min_value=0.055,
            mode=0.060,
            max_value=0.075,
        ),
    ]

    # Define grid of cap rates and terms to test
    cap_rates = [0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08]
    terms = [10, 15, 20, 25]

    print("Step 2: Running TIF Grid Search...")
    print(f"  Cap Rates: {[f'{r:.1%}' for r in cap_rates]}")
    print(f"  Terms: {terms} years")
    print(f"  Total Combinations: {len(cap_rates) * len(terms)}")
    print()

    def progress(completed, total):
        pct = completed / total * 100
        print(f"  Progress: {completed}/{total} ({pct:.0f}%)", end="\r")

    result = run_tif_grid_search(
        base_inputs=base_inputs,
        distributions=distributions,
        cap_rates=cap_rates,
        terms=terms,
        noi_delta_annual=noi_delta_annual,
        target_gap_bps=-250,  # Mixed can underperform by at most 250 bps
        n_iterations=100,  # Per grid point
        seed=42,
        confidence_level=0.95,
        progress_callback=progress,
    )

    print()
    print()
    print(result.summary())

    # Additional analysis
    print()
    print("=" * 70)
    print("DETAILED RESULTS BY CAP RATE AND TERM")
    print("=" * 70)
    print()
    print(f"{'Cap Rate':<10} {'Term':<8} {'TIF Lump Sum':<15} {'Mean Gap':<12} {'5th %ile':<12} {'P(Meet)':<10}")
    print("-" * 70)

    for point in sorted(result.grid_points, key=lambda p: (p.cap_rate, p.term_years)):
        meets = "✓" if point.gap_5th_percentile_bps >= -250 else ""
        print(
            f"{point.cap_rate:>8.2%}  {point.term_years:>5}yr  "
            f"${point.tif_lump_sum:>12,.0f}  {point.irr_gap_mean_bps:>+8.0f} bps  "
            f"{point.gap_5th_percentile_bps:>+8.0f} bps  "
            f"{point.prob_gap_within_tolerance:>8.1%} {meets}"
        )

    # Show TIF amounts needed at each cap rate
    print()
    print("=" * 70)
    print("TIF LUMP SUM BY CAP RATE (NOI Delta / Cap Rate)")
    print("=" * 70)
    print()
    print(f"Annual NOI Delta: ${noi_delta_annual:,.0f}")
    print()
    for cap_rate in cap_rates:
        lump_sum = noi_delta_annual / cap_rate
        per_affordable_unit = lump_sum / (base_inputs.total_units * base_inputs.affordable_pct)
        print(f"  {cap_rate:.1%} cap rate: ${lump_sum:>12,.0f} (${per_affordable_unit:,.0f}/affordable unit)")

    print()
    print("=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print()

    if result.minimum_viable:
        # Find most cost-effective viable option
        cheapest_viable = min(result.minimum_viable, key=lambda p: p.tif_lump_sum)
        print(f"Minimum viable TIF configuration:")
        print(f"  Cap Rate: {cheapest_viable.cap_rate:.2%}")
        print(f"  Term: {cheapest_viable.term_years} years")
        print(f"  TIF Lump Sum: ${cheapest_viable.tif_lump_sum:,.0f}")
        print(f"  Per Affordable Unit: ${cheapest_viable.tif_lump_sum / (base_inputs.total_units * base_inputs.affordable_pct):,.0f}")
        print()
        print(f"This configuration meets the -250 bps target in {cheapest_viable.prob_gap_within_tolerance:.1%} of scenarios.")
    else:
        print("No configuration tested meets the target at 95% confidence.")
        print("Consider testing higher cap rates or finding other incentive sources.")


if __name__ == "__main__":
    main()
