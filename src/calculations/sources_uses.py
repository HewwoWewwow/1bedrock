"""Sources and Uses calculation with iterative IDC solve."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SourcesUses:
    """Capital stack showing sources and uses of funds.

    Uses:
        land: Land acquisition cost
        hard_costs: Construction hard costs (base)
        hard_cost_contingency: Contingency on hard costs
        soft_costs: Soft costs (design, legal, etc.) including developer fee
        soft_cost_contingency: Contingency on soft costs
        predevelopment_costs: Predevelopment costs
        idc: Interest during construction (capitalized)
        loan_fee: Construction loan origination fee
        reserves: Operating and leaseup reserves
        tdc: Total development cost (sum of all uses)

    Sources:
        equity: Developer equity contribution
        construction_loan: Construction debt
        total_sources: Total sources (must equal TDC)

    Metrics:
        ltc: Loan-to-cost ratio (construction_loan / tdc)
        equity_pct: Equity percentage (equity / tdc)
    """
    # Uses - totals
    land: float
    hard_costs: float  # Base hard costs (before contingency)
    soft_costs: float  # Base soft costs including developer fee (before contingency)
    idc: float
    tdc: float

    # Sources
    equity: float
    construction_loan: float
    total_sources: float

    # Metrics
    ltc: float
    equity_pct: float

    # Breakdown for tracking
    soft_cost_pct: float  # Soft costs as % of hard costs

    # Detailed breakdown (optional - for display)
    hard_cost_contingency: float = 0.0
    soft_cost_contingency: float = 0.0
    predevelopment_costs: float = 0.0
    developer_fee: float = 0.0
    loan_fee: float = 0.0
    reserves: float = 0.0

    def __post_init__(self):
        """Validate sources equal uses."""
        if abs(self.tdc - self.total_sources) > 1.0:  # Allow $1 rounding
            raise ValueError(
                f"Sources ({self.total_sources:,.0f}) must equal "
                f"Uses ({self.tdc:,.0f})"
            )


def calculate_sources_uses(
    land_cost: float,
    hard_costs: float,
    soft_cost_pct: float,
    construction_ltc: float,
    construction_rate: float,
    construction_months: int,
    predevelopment_months: int,
    leaseup_months: int = 0,  # Months of lease-up (interest continues to accrue)
    tolerance: float = 100.0,
    max_iterations: int = 50,
) -> SourcesUses:
    """Calculate sources and uses with iterative IDC solve.

    The circular reference is:
    - TDC = Land + Hard + Soft + IDC
    - Construction Loan = TDC × LTC
    - IDC = f(Loan Balance, Rate, Time) - calculated period-by-period
    - But Loan Balance depends on TDC...

    We solve by iterating until IDC converges, using actual period-by-period
    interest calculation from the draw schedule (not an approximation).

    Args:
        land_cost: Land acquisition cost
        hard_costs: Total hard construction costs
        soft_cost_pct: Soft costs as percentage of hard costs
        construction_ltc: Loan-to-cost ratio for construction loan
        construction_rate: Annual interest rate on construction loan
        construction_months: Number of months of construction
        predevelopment_months: Number of months of predevelopment
        leaseup_months: Months of lease-up (construction loan stays outstanding)
        tolerance: Convergence tolerance for IDC (dollars)
        max_iterations: Maximum iterations before giving up

    Returns:
        SourcesUses object with complete capital stack

    Raises:
        ValueError: If IDC doesn't converge
    """
    # Import here to avoid circular dependency
    from src.calculations.draw_schedule import generate_draw_schedule

    # Fixed costs (don't depend on IDC)
    soft_costs = hard_costs * soft_cost_pct
    predev_costs = land_cost + soft_costs  # Land + soft costs in predevelopment

    # Initial estimate: IDC = 0
    idc = 0.0

    for iteration in range(max_iterations):
        # Calculate TDC with current IDC estimate
        # TDC = predev costs + hard costs + IDC
        tdc = predev_costs + hard_costs + idc

        # Size the construction loan
        construction_loan = tdc * construction_ltc

        # Calculate equity (funds first, before debt)
        equity = tdc - construction_loan

        # Generate draw schedule to get actual period-by-period IDC
        # This calculates interest based on actual loan balances as costs are drawn
        # Pass actual cost breakdown instead of using heuristics
        draw_schedule = generate_draw_schedule(
            tdc=tdc,
            equity=equity,
            construction_loan=construction_loan,
            predevelopment_months=predevelopment_months,
            construction_months=construction_months,
            construction_rate=construction_rate,
            leaseup_months=leaseup_months,
            predev_costs=predev_costs,  # Actual: land + soft
            construction_costs=hard_costs + idc,  # Actual: hard + current IDC estimate
        )

        # Get actual IDC from the draw schedule
        new_idc = draw_schedule.total_idc_actual

        # Check convergence
        if abs(new_idc - idc) < tolerance:
            # Converged - build final result
            final_tdc = predev_costs + hard_costs + new_idc
            final_loan = final_tdc * construction_ltc
            final_equity = final_tdc - final_loan

            return SourcesUses(
                land=land_cost,
                hard_costs=hard_costs,
                soft_costs=soft_costs,
                idc=new_idc,
                tdc=final_tdc,
                equity=final_equity,
                construction_loan=final_loan,
                total_sources=final_equity + final_loan,
                ltc=construction_ltc,
                equity_pct=1 - construction_ltc,
                soft_cost_pct=soft_cost_pct,
            )

        # Update for next iteration
        idc = new_idc

    raise ValueError(
        f"IDC calculation did not converge after {max_iterations} iterations. "
        f"Last IDC: ${idc:,.0f}, tolerance: ${tolerance:,.0f}"
    )
