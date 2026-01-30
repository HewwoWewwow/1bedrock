#!/usr/bin/env python3
"""Example script to run the Austin TIF model with spec test inputs."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.project import ProjectInputs, Scenario, UnitMixEntry, TIFStartTiming
from src.models.lookups import ConstructionType, DEFAULT_TAX_RATES
from src.models.incentives import IncentiveTier, IncentiveToggles, get_tier_config
from src.calculations.dcf import run_dcf
from src.calculations.units import allocate_units, get_total_units, get_total_affordable_units
from src.calculations.revenue import calculate_gpr
from src.calculations.metrics import (
    calculate_metrics,
    compare_scenarios,
    format_comparison_table,
)
from src.scenarios import (
    run_scenario_matrix,
    generate_combinations,
    format_matrix_results,
    find_minimum_incentives,
)


def get_spec_inputs() -> ProjectInputs:
    """Get the spec test inputs."""
    from datetime import date

    return ProjectInputs(
        predevelopment_start=date(2026, 1, 1),
        predevelopment_months=18,
        construction_months=24,
        leaseup_months=12,
        operations_months=12,
        land_cost_per_acre=1_000_000,
        target_units=200,
        hard_cost_per_unit=155_000,
        soft_cost_pct=0.30,
        construction_type=ConstructionType.PODIUM_5OVER1,
        unit_mix={
            "studio": UnitMixEntry("studio", gsf=600, allocation=0.12),
            "1br": UnitMixEntry("1br", gsf=750, allocation=0.25),
            "2br": UnitMixEntry("2br", gsf=900, allocation=0.45),
            "3br": UnitMixEntry("3br", gsf=1150, allocation=0.15),
            "4br": UnitMixEntry("4br", gsf=1450, allocation=0.03),
        },
        market_rent_psf=2.50,
        vacancy_rate=0.06,
        leaseup_pace=0.08,
        max_occupancy=0.94,
        market_rent_growth=0.02,
        affordable_rent_growth=0.01,
        opex_growth=0.03,
        property_tax_growth=0.02,
        construction_rate=0.075,
        construction_ltc=0.65,
        perm_rate=0.06,
        perm_amort_years=20,
        perm_ltv_max=0.65,
        perm_dscr_min=1.25,
        existing_assessed_value=5_000_000,
        tax_rates=DEFAULT_TAX_RATES.copy(),
        exit_cap_rate=0.055,
        selected_tier=2,
        affordable_pct=0.20,
        ami_level="50%",
        tif_start_timing=TIFStartTiming.OPERATIONS,
    )


def run_single_comparison():
    """Run a single market vs mixed-income comparison."""
    print("\n" + "=" * 60)
    print("AUSTIN AFFORDABLE HOUSING INCENTIVE MODEL")
    print("Single Scenario Comparison")
    print("=" * 60 + "\n")

    inputs = get_spec_inputs()

    # Market scenario
    print("Running market-rate scenario...")
    market_result = run_dcf(inputs, Scenario.MARKET)

    # Get unit and revenue info for market
    market_allocs = allocate_units(
        target_units=inputs.target_units,
        unit_mix=inputs.unit_mix,
        affordable_pct=0.0,
        ami_level=inputs.ami_level,
        market_rent_psf=inputs.market_rent_psf,
    )
    market_gpr = calculate_gpr(market_allocs)
    market_metrics = calculate_metrics(
        dcf_result=market_result,
        total_units=get_total_units(market_allocs),
        affordable_units=0,
        gpr_annual=market_gpr.total_gpr_annual,
    )

    # Mixed-income scenario with Tier 2 incentives
    print("Running mixed-income scenario (Tier 2 with TIF stream)...")
    toggles = IncentiveToggles(
        smart_fee_waiver=True,
        tax_abatement=False,
        tif_lump_sum=False,
        tif_stream=True,
        interest_buydown=False,
    )
    inputs.incentive_config = get_tier_config(IncentiveTier.TIER_2, toggles)
    inputs.affordable_pct = 0.20
    inputs.ami_level = "50%"

    mixed_result = run_dcf(inputs, Scenario.MIXED_INCOME)

    # Get unit and revenue info for mixed
    mixed_allocs = allocate_units(
        target_units=inputs.target_units,
        unit_mix=inputs.unit_mix,
        affordable_pct=inputs.affordable_pct,
        ami_level=inputs.ami_level,
        market_rent_psf=inputs.market_rent_psf,
    )
    mixed_gpr = calculate_gpr(mixed_allocs)
    mixed_metrics = calculate_metrics(
        dcf_result=mixed_result,
        total_units=get_total_units(mixed_allocs),
        affordable_units=get_total_affordable_units(mixed_allocs),
        gpr_annual=mixed_gpr.total_gpr_annual,
    )

    # Compare
    comparison = compare_scenarios(market_metrics, mixed_metrics)

    print("\n" + format_comparison_table(comparison))

    # Expected vs actual
    print("\n" + "=" * 60)
    print("VALIDATION AGAINST SPEC")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'Expected':>15} {'Actual':>15} {'Diff':>15}")
    print("-" * 70)
    print(f"{'Market TDC':<25} ${'50,400,000':>14} ${market_metrics.tdc:>13,.0f} "
          f"{(market_metrics.tdc - 50_400_000)/1_000_000:>+14.1f}M")
    print(f"{'Market IRR':<25} {'40.3%':>15} {market_metrics.levered_irr:>14.1%} "
          f"{(market_metrics.levered_irr - 0.403)*10000:>+13.0f}bps")
    print(f"{'Mixed TDC':<25} ${'49,200,000':>14} ${mixed_metrics.tdc:>13,.0f} "
          f"{(mixed_metrics.tdc - 49_200_000)/1_000_000:>+14.1f}M")
    print(f"{'Mixed IRR':<25} {'37.4%':>15} {mixed_metrics.levered_irr:>14.1%} "
          f"{(mixed_metrics.levered_irr - 0.374)*10000:>+13.0f}bps")
    print(f"{'IRR Difference':<25} {'-289 bps':>15} {comparison.irr_difference_bps:>+14d}bps")


def run_matrix():
    """Run the full scenario matrix."""
    print("\n" + "=" * 60)
    print("SCENARIO MATRIX ANALYSIS")
    print("Finding incentive combinations that meet +150 bps target")
    print("=" * 60 + "\n")

    inputs = get_spec_inputs()

    # Generate combinations for Tier 2 only (to limit runtime)
    print("Generating incentive combinations for Tier 2...")
    combinations = list(generate_combinations(tiers=[IncentiveTier.TIER_2]))
    print(f"Testing {len(combinations)} combinations...\n")

    # Run matrix
    results = run_scenario_matrix(inputs, iter(combinations))

    # Display results
    print(format_matrix_results(results, show_top_n=15))

    # Find minimum incentives
    print("\n" + "=" * 60)
    print("MINIMUM VIABLE INCENTIVE PACKAGE")
    print("=" * 60 + "\n")

    minimum = find_minimum_incentives(inputs)
    if minimum:
        print(f"Minimum package: {minimum.combination.name}")
        print(f"IRR improvement: {minimum.comparison.irr_difference_bps:+d} bps")
        print(f"Meets +150 bps target: {'YES' if minimum.comparison.meets_target else 'NO'}")
    else:
        print("No combination meets the +150 bps target.")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Austin TIF Model")
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Run full scenario matrix (takes longer)",
    )
    args = parser.parse_args()

    run_single_comparison()

    if args.matrix:
        run_matrix()

    print("\nDone.")


if __name__ == "__main__":
    main()
