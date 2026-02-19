"""Scenario matrix runner for comparing incentive combinations.

Uses the unified calculation engine (calculate_deal) for consistency.
"""

from dataclasses import dataclass
from itertools import product
from typing import Iterator, List

from .models.project import ProjectInputs, Scenario
from .models.incentives import (
    IncentiveTier,
    IncentiveConfig,
    IncentiveToggles,
    get_tier_config,
)
from .calculations.detailed_cashflow import calculate_deal  # SINGLE SOURCE OF TRUTH
from .calculations.units import allocate_units, get_total_units, get_total_affordable_units
from .calculations.revenue import calculate_gpr
from .calculations.metrics import (
    calculate_metrics_from_detailed,  # Unified version
    compare_scenarios,
    ScenarioMetrics,
    ScenarioComparison,
)


@dataclass
class IncentiveCombination:
    """A specific combination of incentives to test."""

    name: str
    tier: IncentiveTier
    smart_fee_waiver: bool
    tax_abatement: bool
    tif_lump_sum: bool
    tif_stream: bool
    interest_buydown: bool


@dataclass
class ScenarioMatrixResult:
    """Result of running a scenario matrix."""

    combination: IncentiveCombination
    market_metrics: ScenarioMetrics
    mixed_metrics: ScenarioMetrics
    comparison: ScenarioComparison


def generate_combinations(
    tiers: List[IncentiveTier] | None = None,
    test_fee_waiver: bool = True,
    test_tax_abatement: bool = True,
    test_tif: bool = True,
    test_buydown: bool = True,
) -> Iterator[IncentiveCombination]:
    """Generate all valid incentive combinations for testing.

    TIF and tax abatement are mutually exclusive. If both would be
    enabled, TIF takes precedence.

    Args:
        tiers: Tiers to test. Defaults to all tiers.
        test_fee_waiver: Include fee waiver variations.
        test_tax_abatement: Include tax abatement variations.
        test_tif: Include TIF variations.
        test_buydown: Include interest buydown variations.

    Yields:
        IncentiveCombination for each valid combination.
    """
    tiers = tiers or list(IncentiveTier)

    fee_options = [True, False] if test_fee_waiver else [False]
    abatement_options = [True, False] if test_tax_abatement else [False]
    tif_lump_options = [True, False] if test_tif else [False]
    tif_stream_options = [True, False] if test_tif else [False]
    buydown_options = [True, False] if test_buydown else [False]

    combo_num = 0
    for tier, fee, abate, tif_lump, tif_stream, buydown in product(
        tiers, fee_options, abatement_options,
        tif_lump_options, tif_stream_options, buydown_options
    ):
        # Skip invalid combinations

        # TIF lump sum and stream are mutually exclusive
        if tif_lump and tif_stream:
            continue

        # Tax abatement and TIF are mutually exclusive
        # Skip combinations where both are enabled
        if abate and (tif_lump or tif_stream):
            continue

        # Skip "no incentives" combination (need at least one)
        if not any([fee, abate, tif_lump, tif_stream, buydown]):
            continue

        combo_num += 1

        # Build name
        parts = [f"T{tier.value}"]
        if fee:
            parts.append("Fee")
        if abate:
            parts.append("Abate")
        if tif_lump:
            parts.append("TIF-L")
        if tif_stream:
            parts.append("TIF-S")
        if buydown:
            parts.append("Buy")

        name = "-".join(parts)

        yield IncentiveCombination(
            name=name,
            tier=tier,
            smart_fee_waiver=fee,
            tax_abatement=abate,
            tif_lump_sum=tif_lump,
            tif_stream=tif_stream,
            interest_buydown=buydown,
        )


def run_single_scenario(
    inputs: ProjectInputs,
    scenario: Scenario,
) -> tuple[ScenarioMetrics, float]:
    """Run a single scenario and return metrics.

    Uses the unified calculation engine (calculate_deal) for consistency.
    This is the SINGLE SOURCE OF TRUTH for all cash flow calculations.

    Args:
        inputs: Project inputs.
        scenario: MARKET or MIXED_INCOME.

    Returns:
        Tuple of (ScenarioMetrics, annual GPR).
    """
    # Run unified calculation engine (SINGLE SOURCE OF TRUTH)
    result = calculate_deal(inputs, scenario)

    # Get unit counts
    affordable_pct = inputs.affordable_pct if scenario == Scenario.MIXED_INCOME else 0.0
    allocations = allocate_units(
        target_units=inputs.target_units,
        unit_mix=inputs.unit_mix,
        affordable_pct=affordable_pct,
        ami_level=inputs.ami_level,
        market_rent_psf=inputs.market_rent_psf,
    )
    total_units = get_total_units(allocations)
    affordable_units = get_total_affordable_units(allocations)

    # Get GPR
    gpr_result = calculate_gpr(allocations)

    # Calculate metrics from detailed result
    metrics = calculate_metrics_from_detailed(
        result=result,
        scenario=scenario,
        total_units=total_units,
        affordable_units=affordable_units,
        gpr_annual=gpr_result.total_gpr_annual,
    )

    return metrics, gpr_result.total_gpr_annual


def run_scenario_matrix(
    base_inputs: ProjectInputs,
    combinations: Iterator[IncentiveCombination] | None = None,
    target_irr_improvement_bps: int = 150,
) -> List[ScenarioMatrixResult]:
    """Run all scenario combinations and return comparison results.

    For each combination:
    1. Run market-rate scenario (baseline)
    2. Run mixed-income scenario with the combination's incentives
    3. Compare and record results

    Args:
        base_inputs: Base project inputs (will be copied for each scenario).
        combinations: Incentive combinations to test. Defaults to all.
        target_irr_improvement_bps: Target IRR improvement (default 150 bps).

    Returns:
        List of ScenarioMatrixResult, sorted by IRR difference (descending).
    """
    if combinations is None:
        combinations = generate_combinations()

    results: List[ScenarioMatrixResult] = []

    # Run market baseline once
    market_inputs = base_inputs.copy()
    market_inputs.affordable_pct = 0.0
    market_inputs.incentive_config = None
    market_metrics, market_gpr = run_single_scenario(market_inputs, Scenario.MARKET)

    for combo in combinations:
        # Build incentive config for this combination
        toggles = IncentiveToggles(
            smart_fee_waiver=combo.smart_fee_waiver,
            tax_abatement=combo.tax_abatement,
            tif_lump_sum=combo.tif_lump_sum,
            tif_stream=combo.tif_stream,
            interest_buydown=combo.interest_buydown,
        )
        incentive_config = get_tier_config(combo.tier, toggles)

        # Set up mixed-income inputs
        mixed_inputs = base_inputs.copy()
        mixed_inputs.affordable_pct = incentive_config.affordable_pct
        mixed_inputs.ami_level = incentive_config.ami_level
        mixed_inputs.incentive_config = incentive_config

        # Run mixed-income scenario
        mixed_metrics, mixed_gpr = run_single_scenario(mixed_inputs, Scenario.MIXED_INCOME)

        # Compare
        comparison = compare_scenarios(
            market=market_metrics,
            mixed_income=mixed_metrics,
            target_irr_improvement_bps=target_irr_improvement_bps,
        )

        results.append(ScenarioMatrixResult(
            combination=combo,
            market_metrics=market_metrics,
            mixed_metrics=mixed_metrics,
            comparison=comparison,
        ))

    # Sort by IRR difference (best first)
    results.sort(key=lambda r: r.comparison.irr_difference_bps, reverse=True)

    return results


def format_matrix_results(
    results: List[ScenarioMatrixResult],
    show_top_n: int = 10,
) -> str:
    """Format matrix results as a text table.

    Args:
        results: Scenario matrix results (assumed sorted).
        show_top_n: Number of top results to show.

    Returns:
        Formatted string table.
    """
    lines = [
        "=" * 90,
        "SCENARIO MATRIX RESULTS (sorted by IRR difference)",
        "=" * 90,
        "",
        f"{'Rank':<6} {'Scenario':<25} {'Mkt IRR':>10} {'Mix IRR':>10} {'Diff (bps)':>12} {'Target':>8}",
        "-" * 90,
    ]

    for i, result in enumerate(results[:show_top_n], 1):
        mkt_irr = result.market_metrics.levered_irr
        mix_irr = result.mixed_metrics.levered_irr
        diff_bps = result.comparison.irr_difference_bps
        meets = "YES" if result.comparison.meets_target else "NO"

        lines.append(
            f"{i:<6} {result.combination.name:<25} "
            f"{mkt_irr:>9.2%} {mix_irr:>9.2%} "
            f"{diff_bps:>+12d} {meets:>8}"
        )

    lines.append("-" * 90)

    # Summary
    meeting_target = sum(1 for r in results if r.comparison.meets_target)
    lines.append(f"\nTotal combinations tested: {len(results)}")
    lines.append(f"Combinations meeting +150 bps target: {meeting_target}")

    if meeting_target > 0:
        best = results[0]
        lines.append(f"\nBest combination: {best.combination.name}")
        lines.append(f"  IRR improvement: {best.comparison.irr_difference_bps:+d} bps")

    lines.append("=" * 90)

    return "\n".join(lines)


def find_minimum_incentives(
    base_inputs: ProjectInputs,
    target_irr_improvement_bps: int = 150,
) -> ScenarioMatrixResult | None:
    """Find the minimum incentive package that meets the target.

    "Minimum" is defined as the fewest incentives enabled that still
    achieves the target IRR improvement.

    Args:
        base_inputs: Base project inputs.
        target_irr_improvement_bps: Target IRR improvement.

    Returns:
        The minimum viable incentive combination, or None if none meet target.
    """
    # Generate all combinations
    all_results = run_scenario_matrix(
        base_inputs,
        target_irr_improvement_bps=target_irr_improvement_bps,
    )

    # Filter to those meeting target
    meeting_target = [r for r in all_results if r.comparison.meets_target]

    if not meeting_target:
        return None

    # Count incentives for each
    def count_incentives(combo: IncentiveCombination) -> int:
        return sum([
            combo.smart_fee_waiver,
            combo.tax_abatement,
            combo.tif_lump_sum,
            combo.tif_stream,
            combo.interest_buydown,
        ])

    # Sort by incentive count (ascending), then by IRR (descending)
    meeting_target.sort(
        key=lambda r: (count_incentives(r.combination), -r.comparison.irr_difference_bps)
    )

    return meeting_target[0]
