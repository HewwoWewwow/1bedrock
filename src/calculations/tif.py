"""TIF (Tax Increment Financing) calculations."""

from dataclasses import dataclass

import numpy_financial as npf


@dataclass
class TIFResult:
    """Results of TIF calculation."""

    city_increment_annual: float  # Annual city tax increment
    tif_term_years: int
    discount_rate: float
    cap_rate: float

    # Capitalized value (NPV of TIF stream)
    capitalized_value: float

    # Stream details (if TIF stream mode)
    monthly_payment: float
    annual_payment: float

    # Timing
    start_month: int  # Month when TIF payments begin
    end_month: int  # Month when TIF payments end

    # Repayment analysis
    repayment_months: int = 0  # Months of stream needed to equal lump sum value


def calculate_tif_value(
    city_increment_annual: float,
    tif_term_years: int,
    tif_rate: float,
    tif_cap_rate: float,
    tif_start_month: int,
    tif_type: str = "stream",  # "stream" or "lump_sum"
    escalation_rate: float = 0.015,  # 1.5% annual escalation
) -> TIFResult:
    """Calculate TIF value based on city property tax increment.

    TIF captures the increment in city property taxes (new taxes above
    the existing base) and returns it to the developer.

    Two modes:
    - Lump Sum: PV of stream provided upfront as development capital
    - Stream: Monthly reimbursement during operations

    The capitalized value uses the TIF cap rate (different from exit cap).

    Args:
        city_increment_annual: Annual city property tax increment.
        tif_term_years: Duration of TIF agreement.
        tif_rate: Discount rate for NPV calculation.
        tif_cap_rate: Cap rate for simple capitalization.
        tif_start_month: Month when TIF payments begin.
        tif_type: "stream" or "lump_sum".
        escalation_rate: Annual escalation of TIF payments.

    Returns:
        TIFResult with capitalized value and payment details.

    Example:
        >>> result = calculate_tif_value(
        ...     city_increment_annual=300_000,
        ...     tif_term_years=15,
        ...     tif_rate=0.02,
        ...     tif_cap_rate=0.065,
        ...     tif_start_month=55,
        ... )
        >>> result.capitalized_value
        4615384.62  # 300,000 / 0.065
    """
    # Monthly payment (before escalation)
    monthly_payment = city_increment_annual / 12
    annual_payment = city_increment_annual

    # End month
    end_month = tif_start_month + (tif_term_years * 12) - 1

    if tif_type == "lump_sum":
        # Simple capitalization: Annual Increment / Cap Rate
        capitalized_value = city_increment_annual / tif_cap_rate
    else:
        # Stream: NPV of escalating payments
        # For simplicity, use PV of annuity if no escalation
        # With escalation, sum the discounted cash flows
        if escalation_rate > 0 and tif_rate > 0:
            # Gordon growth model approximation for growing perpetuity
            # truncated to term years
            # PV = C / (r - g) * [1 - ((1+g)/(1+r))^n]
            if tif_rate != escalation_rate:
                growth_factor = (1 + escalation_rate) / (1 + tif_rate)
                npv_factor = (1 - growth_factor ** tif_term_years) / (tif_rate - escalation_rate)
                capitalized_value = city_increment_annual * npv_factor
            else:
                # When r == g, use limit formula
                capitalized_value = city_increment_annual * tif_term_years / (1 + tif_rate)
        else:
            # No escalation: simple PV of annuity
            capitalized_value = -npf.pv(
                rate=tif_rate,
                nper=tif_term_years,
                pmt=city_increment_annual,
                fv=0,
            )

    # Calculate how many months of stream payments equal the lump sum value
    repayment_months = calculate_repayment_term(
        capitalized_value, monthly_payment, escalation_rate,
    )

    return TIFResult(
        city_increment_annual=city_increment_annual,
        tif_term_years=tif_term_years,
        discount_rate=tif_rate,
        cap_rate=tif_cap_rate,
        capitalized_value=capitalized_value,
        monthly_payment=monthly_payment,
        annual_payment=annual_payment,
        start_month=tif_start_month,
        end_month=end_month,
        repayment_months=repayment_months,
    )


def get_tif_payment_for_month(
    tif_result: TIFResult,
    month: int,
    years_since_start: int = 0,
    escalation_rate: float = 0.015,
) -> float:
    """Get TIF payment for a specific month.

    Returns 0 if month is outside the TIF term.

    Args:
        tif_result: TIF calculation result.
        month: Month number (1-indexed).
        years_since_start: Years elapsed since TIF start (for escalation).
        escalation_rate: Annual escalation rate.

    Returns:
        TIF payment for this month.
    """
    if month < tif_result.start_month or month > tif_result.end_month:
        return 0.0

    # Apply escalation
    escalated_payment = tif_result.monthly_payment * (1 + escalation_rate) ** years_since_start

    return escalated_payment


def calculate_repayment_term(
    principal: float,
    monthly_payment: float,
    escalation_rate: float = 0.015,
    max_years: int = 30,
) -> int:
    """Calculate months needed to repay principal with escalating payments.

    Used to determine how long TIF stream needs to run to "repay"
    a lump sum capitalized value.

    Args:
        principal: Amount to repay.
        monthly_payment: Starting monthly payment.
        escalation_rate: Annual payment escalation.
        max_years: Maximum term to consider.

    Returns:
        Number of months to repay, or max_years * 12 if not repaid.
    """
    annual_escalation = 1 + escalation_rate
    remaining = principal
    current_payment = monthly_payment

    for month in range(1, max_years * 12 + 1):
        remaining -= current_payment

        if remaining <= 0:
            return month

        # Escalate payment annually
        if month % 12 == 0:
            current_payment *= annual_escalation

    return max_years * 12
