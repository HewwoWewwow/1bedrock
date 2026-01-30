"""Metrics calculations and scenario comparison."""

from dataclasses import dataclass

from .dcf import DCFResult
from ..models.project import Scenario


@dataclass
class ScenarioMetrics:
    """Summary metrics for a single scenario."""

    scenario: Scenario

    # Costs
    tdc: float
    tdc_per_unit: float

    # Capital structure
    equity_required: float
    debt_amount: float
    tif_value: float

    # Revenue/NOI
    gpr_annual: float
    noi_annual: float
    yield_on_cost: float  # NOI / TDC

    # Returns
    unlevered_irr: float
    levered_irr: float
    equity_multiple: float

    # Unit counts
    total_units: int
    affordable_units: int
    affordable_pct: float


@dataclass
class ScenarioComparison:
    """Comparison of market vs mixed-income scenarios."""

    market: ScenarioMetrics
    mixed_income: ScenarioMetrics

    # Differences
    irr_difference_bps: int  # Mixed - Market in basis points
    noi_gap_annual: float  # Market NOI - Mixed NOI
    equity_difference: float  # Market equity - Mixed equity
    tdc_difference: float  # Market TDC - Mixed TDC

    # Incentive impact
    total_incentive_value: float  # TIF + fee waivers + other
    incentive_irr_impact_bps: int  # How much IRR improved from incentives

    # Target achievement
    target_irr_improvement_bps: int = 150
    meets_target: bool = False  # True if mixed IRR >= market IRR + target


def calculate_metrics(
    dcf_result: DCFResult,
    total_units: int,
    affordable_units: int,
    gpr_annual: float,
) -> ScenarioMetrics:
    """Calculate summary metrics from DCF result.

    Args:
        dcf_result: Complete DCF analysis result.
        total_units: Total number of units.
        affordable_units: Number of affordable units.
        gpr_annual: Annual Gross Potential Rent.

    Returns:
        ScenarioMetrics with all summary values.
    """
    affordable_pct = affordable_units / total_units if total_units > 0 else 0.0

    return ScenarioMetrics(
        scenario=dcf_result.scenario,
        tdc=dcf_result.tdc,
        tdc_per_unit=dcf_result.tdc / total_units if total_units > 0 else 0.0,
        equity_required=dcf_result.equity_required,
        debt_amount=dcf_result.perm_loan_amount,
        tif_value=dcf_result.tif_capitalized_value,
        gpr_annual=gpr_annual,
        noi_annual=dcf_result.stabilized_noi_annual,
        yield_on_cost=dcf_result.yield_on_cost,
        unlevered_irr=dcf_result.unlevered_irr,
        levered_irr=dcf_result.levered_irr,
        equity_multiple=dcf_result.equity_multiple,
        total_units=total_units,
        affordable_units=affordable_units,
        affordable_pct=affordable_pct,
    )


def compare_scenarios(
    market: ScenarioMetrics,
    mixed_income: ScenarioMetrics,
    target_irr_improvement_bps: int = 150,
) -> ScenarioComparison:
    """Compare market and mixed-income scenarios.

    The key metric is whether the mixed-income scenario achieves
    at least +150 bps of additional levered IRR compared to market.

    Args:
        market: Market-rate scenario metrics.
        mixed_income: Mixed-income scenario metrics.
        target_irr_improvement_bps: Target IRR improvement (default 150 bps).

    Returns:
        ScenarioComparison with all comparison metrics.
    """
    # IRR difference in basis points
    irr_diff_decimal = mixed_income.levered_irr - market.levered_irr
    irr_diff_bps = int(round(irr_diff_decimal * 10_000))

    # NOI gap
    noi_gap = market.noi_annual - mixed_income.noi_annual

    # Equity difference
    equity_diff = market.equity_required - mixed_income.equity_required

    # TDC difference
    tdc_diff = market.tdc - mixed_income.tdc

    # Total incentive value (approximation)
    # TIF + fee savings reflected in lower TDC/equity
    total_incentive = mixed_income.tif_value + max(0, tdc_diff)

    # Estimate IRR impact from incentives
    # This is the improvement from no-incentive mixed to incentivized mixed
    # For simplicity, use the full difference (assumes no incentives = market IRR)
    incentive_impact_bps = irr_diff_bps

    # Check target
    meets_target = irr_diff_bps >= target_irr_improvement_bps

    return ScenarioComparison(
        market=market,
        mixed_income=mixed_income,
        irr_difference_bps=irr_diff_bps,
        noi_gap_annual=noi_gap,
        equity_difference=equity_diff,
        tdc_difference=tdc_diff,
        total_incentive_value=total_incentive,
        incentive_irr_impact_bps=incentive_impact_bps,
        target_irr_improvement_bps=target_irr_improvement_bps,
        meets_target=meets_target,
    )


def format_comparison_table(comparison: ScenarioComparison) -> str:
    """Format comparison as a text table.

    Args:
        comparison: Scenario comparison result.

    Returns:
        Formatted string table.
    """
    market = comparison.market
    mixed = comparison.mixed_income

    lines = [
        "=" * 60,
        "SCENARIO COMPARISON",
        "=" * 60,
        "",
        f"{'Metric':<25} {'Market':>15} {'Mixed-Income':>15}",
        "-" * 60,
        f"{'TDC':<25} ${market.tdc:>13,.0f} ${mixed.tdc:>13,.0f}",
        f"{'TDC/Unit':<25} ${market.tdc_per_unit:>13,.0f} ${mixed.tdc_per_unit:>13,.0f}",
        f"{'Equity Required':<25} ${market.equity_required:>13,.0f} ${mixed.equity_required:>13,.0f}",
        f"{'Perm Loan':<25} ${market.debt_amount:>13,.0f} ${mixed.debt_amount:>13,.0f}",
        f"{'TIF Value':<25} ${market.tif_value:>13,.0f} ${mixed.tif_value:>13,.0f}",
        "",
        f"{'GPR (Annual)':<25} ${market.gpr_annual:>13,.0f} ${mixed.gpr_annual:>13,.0f}",
        f"{'NOI (Annual)':<25} ${market.noi_annual:>13,.0f} ${mixed.noi_annual:>13,.0f}",
        f"{'Yield on Cost':<25} {market.yield_on_cost:>14.2%} {mixed.yield_on_cost:>14.2%}",
        "",
        f"{'Levered IRR':<25} {market.levered_irr:>14.2%} {mixed.levered_irr:>14.2%}",
        f"{'Unlevered IRR':<25} {market.unlevered_irr:>14.2%} {mixed.unlevered_irr:>14.2%}",
        f"{'Equity Multiple':<25} {market.equity_multiple:>14.2f}x {mixed.equity_multiple:>14.2f}x",
        "",
        "-" * 60,
        f"{'IRR Difference':<25} {comparison.irr_difference_bps:>+15d} bps",
        f"{'Target':<25} {comparison.target_irr_improvement_bps:>+15d} bps",
        f"{'Meets Target':<25} {'YES' if comparison.meets_target else 'NO':>15}",
        "=" * 60,
    ]

    return "\n".join(lines)
