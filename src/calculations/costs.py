"""Cost calculations including TDC and iterative IDC.

NOTE: The `calculate_tdc()` function in this module is DEPRECATED.
Use `calculate_deal()` from `detailed_cashflow.py` which computes
TDC as part of its unified SourcesUses calculation.

The helper functions `calculate_idc()`, `calculate_opex_monthly()`,
and `escalate_opex()` are still valid utilities.
"""

import warnings
from dataclasses import dataclass
from typing import Tuple


@dataclass
class TDCResult:
    """Results of Total Development Cost calculation."""

    land_cost: float
    hard_costs: float
    hard_cost_contingency: float
    hard_costs_total: float  # hard_costs + contingency
    soft_costs: float
    soft_cost_contingency: float
    soft_costs_total: float  # soft_costs + contingency (after waiver)
    predevelopment_costs: float  # Separate predevelopment costs
    developer_fee: float
    soft_costs_after_waiver: float
    waiver_amount: float
    total_before_idc: float
    idc: float  # Interest During Construction
    loan_fee: float  # Construction loan origination fee
    financing_costs_total: float  # IDC + loan fee
    operating_reserve: float
    leaseup_reserve: float
    reserves_total: float
    tdc: float  # Total Development Cost
    idc_iterations: int  # Number of iterations for IDC convergence


def calculate_idc(
    costs_before_idc: float,
    construction_ltc: float,
    construction_rate: float,
    construction_months: int,
    leaseup_months: int = 0,
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

    Uses period-by-period interest calculation:
    - During construction: Linear draws with interest on outstanding balance
    - During lease-up: No new draws, but interest continues on full balance

    The construction loan stays in place through lease-up (banks won't close
    perm until DSCR covenants can be met at stabilization).

    Args:
        costs_before_idc: Total costs before adding IDC.
        construction_ltc: Loan-to-cost ratio (e.g., 0.65).
        construction_rate: Annual interest rate (e.g., 0.075).
        construction_months: Duration of construction in months.
        leaseup_months: Duration of lease-up in months (loan still outstanding).
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
        ...     leaseup_months=12,
        ... )
        >>> idc
        4500000.0  # Example IDC
        >>> iterations
        5  # Converged in 5 iterations
    """
    # Use actual draw schedule for accurate IDC calculation
    from src.calculations.draw_schedule import generate_draw_schedule

    idc_estimate = 0.0

    for iteration in range(max_iterations):
        # Calculate TDC with current IDC estimate
        tdc = costs_before_idc + idc_estimate

        # Construction loan is LTC of TDC
        loan_amount = tdc * construction_ltc
        equity = tdc - loan_amount

        # Generate draw schedule with S-curve (period-by-period interest)
        draw_schedule = generate_draw_schedule(
            tdc=tdc,
            equity=equity,
            construction_loan=loan_amount,
            predevelopment_months=0,  # IDC calculation doesn't need predev
            construction_months=construction_months,
            construction_rate=construction_rate,
            leaseup_months=leaseup_months,
            predev_costs=0.0,
            construction_costs=costs_before_idc + idc_estimate,
        )

        new_idc = draw_schedule.total_idc_actual

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
    leaseup_months: int = 0,
    predevelopment_cost_pct: float = 0.0,
    fee_waiver_amount: float = 0.0,
    hard_cost_contingency_pct: float = 0.0,
    soft_cost_contingency_pct: float = 0.0,
    developer_fee_pct: float = 0.0,
    construction_loan_fee_pct: float = 0.01,
    monthly_opex: float = 0.0,
    monthly_debt_service: float = 0.0,
    operating_reserve_months: int = 0,
    leaseup_reserve_months: int = 0,
) -> TDCResult:
    """Calculate Total Development Cost (TDC) including iterative IDC.

    .. deprecated::
        This function is DEPRECATED. Use `calculate_deal()` from
        `detailed_cashflow.py` instead, which computes TDC as part
        of its unified SourcesUses calculation. The SourcesUses
        dataclass provides detailed cost breakdown.

    TDC = Land + Hard Costs (with contingency) + Predevelopment + Soft Costs
          (with contingency) + Financing (IDC + loan fee) + Reserves

    IDC is calculated over construction + lease-up period since the construction
    loan remains in place until stabilization (when perm can close).

    Args:
        land_cost: Cost of land.
        target_units: Number of units.
        hard_cost_per_unit: Hard construction cost per unit.
        soft_cost_pct: Soft costs as percentage of hard costs (e.g., 0.30).
        construction_ltc: Construction loan-to-cost ratio.
        construction_rate: Construction loan interest rate.
        construction_months: Construction duration in months.
        leaseup_months: Duration of lease-up (construction loan still outstanding).
        predevelopment_cost_pct: Predevelopment costs as percentage of hard costs.
        fee_waiver_amount: SMART fee waiver amount to subtract from soft costs.
        hard_cost_contingency_pct: Contingency on hard costs (e.g., 0.05 for 5%).
        soft_cost_contingency_pct: Contingency on soft costs (e.g., 0.05 for 5%).
        developer_fee_pct: Developer fee as percentage of hard costs.
        construction_loan_fee_pct: Loan origination fee (e.g., 0.01 for 1%).
        monthly_opex: Monthly operating expenses for reserve calculation.
        monthly_debt_service: Monthly debt service for lease-up reserve.
        operating_reserve_months: Months of OpEx to hold in reserve.
        leaseup_reserve_months: Months of debt service reserve during lease-up.

    Returns:
        TDCResult with all cost components and final TDC.
    """
    warnings.warn(
        "calculate_tdc() is deprecated. Use calculate_deal() from detailed_cashflow.py instead. "
        "The SourcesUses dataclass in the result provides the same cost breakdown.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Hard costs with contingency
    hard_costs = target_units * hard_cost_per_unit
    hard_cost_contingency = hard_costs * hard_cost_contingency_pct
    hard_costs_total = hard_costs + hard_cost_contingency

    # Predevelopment costs (based on hard costs before contingency, per Excel)
    predevelopment_costs = hard_costs * predevelopment_cost_pct

    # Soft costs with contingency
    # Soft costs base = soft_cost_pct of hard_costs_total (with contingency)
    soft_costs_base = hard_costs_total * soft_cost_pct
    developer_fee = hard_costs_total * developer_fee_pct
    soft_costs = soft_costs_base + developer_fee

    # Apply fee waiver to soft costs
    soft_costs_after_waiver = max(0.0, soft_costs - fee_waiver_amount)
    waiver_amount = soft_costs - soft_costs_after_waiver

    # Soft cost contingency (on soft costs after waiver)
    soft_cost_contingency = soft_costs_after_waiver * soft_cost_contingency_pct
    soft_costs_total = soft_costs_after_waiver + soft_cost_contingency

    # Reserves
    operating_reserve = monthly_opex * operating_reserve_months
    leaseup_reserve = monthly_debt_service * leaseup_reserve_months
    reserves_total = operating_reserve + leaseup_reserve

    # Total before IDC (includes predevelopment and reserves)
    total_before_idc = (
        land_cost +
        hard_costs_total +
        predevelopment_costs +
        soft_costs_total +
        reserves_total
    )

    # Calculate IDC iteratively (includes lease-up period)
    idc, iterations = calculate_idc(
        costs_before_idc=total_before_idc,
        construction_ltc=construction_ltc,
        construction_rate=construction_rate,
        construction_months=construction_months,
        leaseup_months=leaseup_months,
    )

    # Loan fee based on construction loan amount (which is LTC of TDC)
    # This is circular but we approximate using TDC before loan fee
    tdc_before_loan_fee = total_before_idc + idc
    construction_loan = tdc_before_loan_fee * construction_ltc
    loan_fee = construction_loan * construction_loan_fee_pct

    # Financing costs total
    financing_costs_total = idc + loan_fee

    # Final TDC
    tdc = total_before_idc + financing_costs_total

    return TDCResult(
        land_cost=land_cost,
        hard_costs=hard_costs,
        hard_cost_contingency=hard_cost_contingency,
        hard_costs_total=hard_costs_total,
        soft_costs=soft_costs,
        soft_cost_contingency=soft_cost_contingency,
        soft_costs_total=soft_costs_total,
        predevelopment_costs=predevelopment_costs,
        developer_fee=developer_fee,
        soft_costs_after_waiver=soft_costs_after_waiver,
        waiver_amount=waiver_amount,
        total_before_idc=total_before_idc,
        idc=idc,
        loan_fee=loan_fee,
        financing_costs_total=financing_costs_total,
        operating_reserve=operating_reserve,
        leaseup_reserve=leaseup_reserve,
        reserves_total=reserves_total,
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
