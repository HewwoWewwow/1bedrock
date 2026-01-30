"""Revenue calculations for Gross Potential Rent (GPR) and Effective Gross Income (EGI)."""

from dataclasses import dataclass

from .units import UnitAllocation


@dataclass
class GPRResult:
    """Results of GPR calculation."""

    market_gpr_annual: float  # Annual GPR from market units
    affordable_gpr_annual: float  # Annual GPR from affordable units
    total_gpr_annual: float  # Total annual GPR
    market_gpr_monthly: float
    affordable_gpr_monthly: float
    total_gpr_monthly: float


def calculate_gpr(allocations: list[UnitAllocation]) -> GPRResult:
    """Calculate annual Gross Potential Rent (GPR) for market and affordable units.

    GPR is the total rental income assuming 100% occupancy.
    - Market units: GSF x $/PSF x 12 months x count
    - Affordable units: AMI rent x 12 months x count

    Args:
        allocations: List of unit allocations with rents.

    Returns:
        GPRResult with market, affordable, and total GPR (monthly and annual).

    Example:
        >>> gpr = calculate_gpr(allocations)
        >>> gpr.total_gpr_annual
        7200000.0  # Example: $7.2M annual GPR
    """
    market_gpr_monthly = 0.0
    affordable_gpr_monthly = 0.0

    for allocation in allocations:
        # Market units: monthly rent x unit count
        market_gpr_monthly += allocation.market_rent_monthly * allocation.market_units

        # Affordable units: monthly rent x unit count
        affordable_gpr_monthly += allocation.affordable_rent_monthly * allocation.affordable_units

    total_gpr_monthly = market_gpr_monthly + affordable_gpr_monthly

    return GPRResult(
        market_gpr_annual=market_gpr_monthly * 12,
        affordable_gpr_annual=affordable_gpr_monthly * 12,
        total_gpr_annual=total_gpr_monthly * 12,
        market_gpr_monthly=market_gpr_monthly,
        affordable_gpr_monthly=affordable_gpr_monthly,
        total_gpr_monthly=total_gpr_monthly,
    )


def calculate_egi(gpr_monthly: float, vacancy_rate: float) -> float:
    """Calculate monthly Effective Gross Income (EGI).

    EGI = GPR x (1 - vacancy rate)

    Args:
        gpr_monthly: Monthly Gross Potential Rent.
        vacancy_rate: Vacancy rate as decimal (e.g., 0.06 for 6%).

    Returns:
        Monthly EGI.
    """
    return gpr_monthly * (1 - vacancy_rate)


def calculate_egi_with_leaseup(
    gpr_monthly: float,
    month_in_leaseup: int,
    leaseup_months: int,
    leaseup_pace: float,
    max_occupancy: float,
) -> tuple[float, float]:
    """Calculate EGI during lease-up period with ramping occupancy.

    Occupancy ramps linearly from 0% to max_occupancy over the lease-up period,
    constrained by the leaseup_pace (max units leased per month as % of total).

    Args:
        gpr_monthly: Monthly Gross Potential Rent at full occupancy.
        month_in_leaseup: Month number within lease-up (1-indexed).
        leaseup_months: Total lease-up period in months.
        leaseup_pace: Maximum occupancy gain per month (e.g., 0.08 for 8%).
        max_occupancy: Target stabilized occupancy (e.g., 0.94 for 94%).

    Returns:
        Tuple of (EGI for this month, occupancy rate for this month).
    """
    # Linear ramp: occupancy increases each month
    # Constrained by pace and max occupancy
    target_occupancy = min(month_in_leaseup * leaseup_pace, max_occupancy)
    occupancy = min(target_occupancy, max_occupancy)

    egi = gpr_monthly * occupancy
    return egi, occupancy


def escalate_rent(
    base_rent: float,
    years_elapsed: int,
    growth_rate: float,
) -> float:
    """Apply annual rent escalation.

    Args:
        base_rent: Starting rent amount.
        years_elapsed: Number of years since base period.
        growth_rate: Annual growth rate (e.g., 0.02 for 2%).

    Returns:
        Escalated rent amount.
    """
    return base_rent * (1 + growth_rate) ** years_elapsed
