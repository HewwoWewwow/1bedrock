"""Sources and Uses calculation with iterative IDC solve."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SourcesUses:
    """Capital stack showing sources and uses of funds.

    Uses:
        land: Land acquisition cost
        hard_costs: Construction hard costs
        soft_costs: Soft costs (design, legal, etc.)
        idc: Interest during construction (capitalized)
        tdc: Total development cost (sum of all uses)

    Sources:
        equity: Developer equity contribution
        construction_loan: Construction debt
        total_sources: Total sources (must equal TDC)

    Metrics:
        ltc: Loan-to-cost ratio (construction_loan / tdc)
        equity_pct: Equity percentage (equity / tdc)
    """
    # Uses
    land: float
    hard_costs: float
    soft_costs: float
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
    tolerance: float = 100.0,
    max_iterations: int = 50,
) -> SourcesUses:
    """Calculate sources and uses with iterative IDC solve.

    The circular reference is:
    - TDC = Land + Hard + Soft + IDC
    - Construction Loan = TDC × LTC
    - IDC = f(Loan Balance, Rate, Time)
    - But Loan Balance depends on TDC...

    We solve by iterating until IDC converges.

    Args:
        land_cost: Land acquisition cost
        hard_costs: Total hard construction costs
        soft_cost_pct: Soft costs as percentage of hard costs
        construction_ltc: Loan-to-cost ratio for construction loan
        construction_rate: Annual interest rate on construction loan
        construction_months: Number of months of construction
        predevelopment_months: Number of months of predevelopment
        tolerance: Convergence tolerance for IDC (dollars)
        max_iterations: Maximum iterations before giving up

    Returns:
        SourcesUses object with complete capital stack

    Raises:
        ValueError: If IDC doesn't converge
    """
    # Fixed costs (don't depend on IDC)
    soft_costs = hard_costs * soft_cost_pct
    costs_before_idc = land_cost + hard_costs + soft_costs

    # Monthly rate
    monthly_rate = construction_rate / 12

    # Initial estimate: IDC = 0
    idc = 0.0

    for iteration in range(max_iterations):
        # Calculate TDC with current IDC estimate
        tdc = costs_before_idc + idc

        # Size the construction loan
        construction_loan = tdc * construction_ltc

        # Calculate equity (funds first, before debt)
        equity = tdc - construction_loan

        # Calculate new IDC estimate
        # IDC accrues on the construction loan balance during construction
        # Using average balance method for S-curve approximation:
        # - First half of construction: loan ramps up
        # - Second half: loan at full draw
        # Average balance ≈ 65% of full loan amount for S-curve
        average_balance_factor = 0.65
        average_balance = construction_loan * average_balance_factor

        # Interest accrues for construction period
        # (predevelopment is equity-funded, no debt interest)
        new_idc = average_balance * monthly_rate * construction_months

        # Check convergence
        if abs(new_idc - idc) < tolerance:
            # Converged - build final result
            final_tdc = costs_before_idc + new_idc
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


def calculate_sources_uses_with_incentives(
    land_cost: float,
    hard_costs: float,
    soft_cost_pct: float,
    construction_ltc: float,
    construction_rate: float,
    construction_months: int,
    predevelopment_months: int,
    fee_waiver_amount: float = 0.0,
    rate_buydown_bps: int = 0,
    tolerance: float = 100.0,
    max_iterations: int = 50,
) -> SourcesUses:
    """Calculate sources and uses with incentives applied.

    Args:
        land_cost: Land acquisition cost
        hard_costs: Total hard construction costs
        soft_cost_pct: Soft costs as percentage of hard costs
        construction_ltc: Loan-to-cost ratio for construction loan
        construction_rate: Annual interest rate on construction loan
        construction_months: Number of months of construction
        predevelopment_months: Number of months of predevelopment
        fee_waiver_amount: SMART fee waiver reducing soft costs
        rate_buydown_bps: Interest rate buydown in basis points
        tolerance: Convergence tolerance for IDC (dollars)
        max_iterations: Maximum iterations before giving up

    Returns:
        SourcesUses object with complete capital stack
    """
    # Reduce soft costs by fee waiver
    soft_costs = hard_costs * soft_cost_pct - fee_waiver_amount
    soft_costs = max(0, soft_costs)  # Can't go negative

    # Adjust effective soft cost percentage
    effective_soft_cost_pct = soft_costs / hard_costs if hard_costs > 0 else 0

    # Reduce construction rate by buydown
    effective_rate = construction_rate - (rate_buydown_bps / 10000)
    effective_rate = max(0.01, effective_rate)  # Floor at 1%

    return calculate_sources_uses(
        land_cost=land_cost,
        hard_costs=hard_costs,
        soft_cost_pct=effective_soft_cost_pct,
        construction_ltc=construction_ltc,
        construction_rate=effective_rate,
        construction_months=construction_months,
        predevelopment_months=predevelopment_months,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
