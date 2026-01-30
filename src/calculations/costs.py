"""Cost calculations including TDC and iterative IDC."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class TDCResult:
    """Results of Total Development Cost calculation."""

    land_cost: float
    hard_costs: float
    soft_costs: float
    soft_costs_after_waiver: float
    waiver_amount: float
    total_before_idc: float
    idc: float  # Interest During Construction
    tdc: float  # Total Development Cost
    idc_iterations: int  # Number of iterations for IDC convergence


def calculate_idc(
    costs_before_idc: float,
    construction_ltc: float,
    construction_rate: float,
    construction_months: int,
    tolerance: float = 100.0,
    max_iterations: int = 50,
) -> Tuple[float, int]:
    """Calculate Interest During Construction (IDC) iteratively.

    IDC creates a circular dependency:
    - IDC is part of TDC
    - Construction loan is based on TDC (via LTC)
    - Interest accrues on the construction loan
    - Therefore IDC depends on itself

    This function iterates until convergence (change < tolerance).

    The draw schedule assumes:
    - Linear monthly draws over the construction period
    - Interest compounds monthly (unpaid interest accrues to principal)

    Args:
        costs_before_idc: Total costs before adding IDC.
        construction_ltc: Loan-to-cost ratio (e.g., 0.65).
        construction_rate: Annual interest rate (e.g., 0.075).
        construction_months: Duration of construction in months.
        tolerance: Convergence tolerance in dollars (default $100).
        max_iterations: Maximum iterations before returning best estimate.

    Returns:
        Tuple of (IDC amount, number of iterations to converge).

    Example:
        >>> idc, iterations = calculate_idc(
        ...     costs_before_idc=45_000_000,
        ...     construction_ltc=0.65,
        ...     construction_rate=0.075,
        ...     construction_months=24,
        ... )
        >>> idc
        3654321.0  # Example IDC
        >>> iterations
        5  # Converged in 5 iterations
    """
    monthly_rate = construction_rate / 12
    idc_estimate = 0.0

    for iteration in range(max_iterations):
        # Calculate TDC with current IDC estimate
        tdc = costs_before_idc + idc_estimate

        # Construction loan is LTC of TDC
        loan_amount = tdc * construction_ltc

        # Model monthly draws with compounding interest
        # Assumes linear draw schedule (equal draws each month)
        monthly_draw = loan_amount / construction_months
        cumulative_balance = 0.0
        cumulative_interest = 0.0

        for _month in range(construction_months):
            # Draw funds at start of month
            cumulative_balance += monthly_draw

            # Calculate interest at end of month
            month_interest = cumulative_balance * monthly_rate
            cumulative_interest += month_interest

            # Interest capitalizes (accrues to principal for next month)
            cumulative_balance += month_interest

        new_idc = cumulative_interest

        # Check convergence
        if abs(new_idc - idc_estimate) < tolerance:
            return new_idc, iteration + 1

        idc_estimate = new_idc

    # Did not converge within max_iterations - return best estimate
    return idc_estimate, max_iterations


def calculate_tdc(
    land_cost: float,
    target_units: int,
    hard_cost_per_unit: float,
    soft_cost_pct: float,
    construction_ltc: float,
    construction_rate: float,
    construction_months: int,
    fee_waiver_amount: float = 0.0,
) -> TDCResult:
    """Calculate Total Development Cost (TDC) including iterative IDC.

    TDC = Land + Hard Costs + Soft Costs - Waivers + IDC

    Args:
        land_cost: Cost of land.
        target_units: Number of units.
        hard_cost_per_unit: Hard construction cost per unit.
        soft_cost_pct: Soft costs as percentage of hard costs (e.g., 0.30).
        construction_ltc: Construction loan-to-cost ratio.
        construction_rate: Construction loan interest rate.
        construction_months: Construction duration in months.
        fee_waiver_amount: SMART fee waiver amount to subtract from soft costs.

    Returns:
        TDCResult with all cost components and final TDC.

    Example:
        >>> result = calculate_tdc(
        ...     land_cost=3_076_923,
        ...     target_units=200,
        ...     hard_cost_per_unit=155_000,
        ...     soft_cost_pct=0.30,
        ...     construction_ltc=0.65,
        ...     construction_rate=0.075,
        ...     construction_months=24,
        ... )
        >>> result.tdc
        50400000.0  # Approximate
    """
    # Hard costs
    hard_costs = target_units * hard_cost_per_unit

    # Soft costs (before waiver)
    soft_costs = hard_costs * soft_cost_pct

    # Apply fee waiver
    soft_costs_after_waiver = max(0.0, soft_costs - fee_waiver_amount)
    waiver_amount = soft_costs - soft_costs_after_waiver

    # Total before IDC
    total_before_idc = land_cost + hard_costs + soft_costs_after_waiver

    # Calculate IDC iteratively
    idc, iterations = calculate_idc(
        costs_before_idc=total_before_idc,
        construction_ltc=construction_ltc,
        construction_rate=construction_rate,
        construction_months=construction_months,
    )

    # Final TDC
    tdc = total_before_idc + idc

    return TDCResult(
        land_cost=land_cost,
        hard_costs=hard_costs,
        soft_costs=soft_costs,
        soft_costs_after_waiver=soft_costs_after_waiver,
        waiver_amount=waiver_amount,
        total_before_idc=total_before_idc,
        idc=idc,
        tdc=tdc,
        idc_iterations=iterations,
    )


def calculate_opex_monthly(
    total_units: int,
    opex_per_unit_annual: float,
    egi_monthly: float,
    management_pct: float,
    reserves_pct: float,
) -> Tuple[float, float, float]:
    """Calculate monthly operating expenses.

    OpEx has three components:
    1. Fixed per-unit costs (utilities, maintenance, misc)
    2. Management fee (% of EGI)
    3. Replacement reserves (% of EGI)

    Args:
        total_units: Number of units.
        opex_per_unit_annual: Annual per-unit OpEx (fixed portion).
        egi_monthly: Monthly Effective Gross Income.
        management_pct: Management fee as % of EGI.
        reserves_pct: Replacement reserves as % of EGI.

    Returns:
        Tuple of (total monthly opex, management fee, reserves).
    """
    # Fixed per-unit costs (monthly)
    fixed_opex = (total_units * opex_per_unit_annual) / 12

    # Variable costs (% of EGI)
    management_fee = egi_monthly * management_pct
    reserves = egi_monthly * reserves_pct

    total_opex = fixed_opex + management_fee + reserves

    return total_opex, management_fee, reserves


def escalate_opex(
    base_opex: float,
    years_elapsed: int,
    growth_rate: float,
) -> float:
    """Apply annual operating expense escalation.

    Args:
        base_opex: Starting OpEx amount.
        years_elapsed: Number of years since base period.
        growth_rate: Annual growth rate (e.g., 0.03 for 3%).

    Returns:
        Escalated OpEx amount.
    """
    return base_opex * (1 + growth_rate) ** years_elapsed
