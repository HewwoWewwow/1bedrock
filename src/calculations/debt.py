"""Debt calculations for construction and permanent loans."""

from dataclasses import dataclass

import numpy_financial as npf


@dataclass
class ConstructionLoan:
    """Construction loan details."""

    loan_amount: float
    ltc_ratio: float
    interest_rate: float
    term_months: int


@dataclass
class PermanentLoan:
    """Permanent loan details including constraint analysis."""

    loan_amount: float
    ltv_constrained_amount: float  # Max loan from LTV constraint
    dscr_constrained_amount: float  # Max loan from DSCR constraint
    binding_constraint: str  # "LTV" or "DSCR"
    interest_rate: float  # After any buydown
    amortization_months: int
    monthly_payment: float
    annual_debt_service: float
    actual_ltv: float
    actual_dscr: float


def size_construction_loan(
    tdc: float,
    ltc_ratio: float,
    interest_rate: float,
    construction_months: int,
) -> ConstructionLoan:
    """Size construction loan based on loan-to-cost ratio.

    Args:
        tdc: Total Development Cost.
        ltc_ratio: Loan-to-cost ratio (e.g., 0.65).
        interest_rate: Annual interest rate.
        construction_months: Construction duration in months.

    Returns:
        ConstructionLoan with loan amount and terms.
    """
    loan_amount = tdc * ltc_ratio

    return ConstructionLoan(
        loan_amount=loan_amount,
        ltc_ratio=ltc_ratio,
        interest_rate=interest_rate,
        term_months=construction_months,
    )


def size_permanent_loan(
    stabilized_value: float,
    stabilized_noi: float,
    ltv_max: float,
    dscr_min: float,
    perm_rate: float,
    amort_years: int,
    rate_buydown_bps: int = 0,
) -> PermanentLoan:
    """Size permanent loan as the lesser of LTV and DSCR constraints.

    The permanent loan is sized using two constraints:
    1. LTV Constraint: Loan <= Value x LTV_max
    2. DSCR Constraint: Loan <= Max loan where NOI / Debt_Service >= DSCR_min

    The binding constraint is whichever produces the smaller loan.

    For DSCR constraint:
        Max annual debt service = NOI / DSCR_min
        Max loan = PV of annuity at that payment level

    Args:
        stabilized_value: Property value at stabilization.
        stabilized_noi: Annual NOI at stabilization.
        ltv_max: Maximum loan-to-value ratio (e.g., 0.65).
        dscr_min: Minimum debt service coverage ratio (e.g., 1.25).
        perm_rate: Annual interest rate (before buydown).
        amort_years: Amortization period in years.
        rate_buydown_bps: Interest rate reduction in basis points.

    Returns:
        PermanentLoan with loan amount, constraints, and actual ratios.

    Example:
        >>> loan = size_permanent_loan(
        ...     stabilized_value=90_000_000,
        ...     stabilized_noi=4_900_000,
        ...     ltv_max=0.65,
        ...     dscr_min=1.25,
        ...     perm_rate=0.06,
        ...     amort_years=20,
        ... )
        >>> loan.loan_amount
        52000000.0  # Constrained by DSCR
        >>> loan.binding_constraint
        "DSCR"
    """
    # Apply rate buydown
    effective_rate = perm_rate - (rate_buydown_bps / 10_000)
    monthly_rate = effective_rate / 12
    amort_months = amort_years * 12

    # LTV Constraint
    ltv_constrained = stabilized_value * ltv_max

    # DSCR Constraint
    # Max annual debt service = NOI / DSCR_min
    max_annual_ds = stabilized_noi / dscr_min
    max_monthly_ds = max_annual_ds / 12

    # Find max loan using PV of annuity formula
    # PV = PMT x [(1 - (1 + r)^-n) / r]
    # Using numpy_financial.pv (returns negative, so negate)
    dscr_constrained = -npf.pv(
        rate=monthly_rate,
        nper=amort_months,
        pmt=max_monthly_ds,
        fv=0,
    )

    # Use the binding (smaller) constraint
    if ltv_constrained <= dscr_constrained:
        loan_amount = ltv_constrained
        binding_constraint = "LTV"
    else:
        loan_amount = dscr_constrained
        binding_constraint = "DSCR"

    # Calculate actual debt service
    monthly_payment = -npf.pmt(
        rate=monthly_rate,
        nper=amort_months,
        pv=loan_amount,
        fv=0,
    )
    annual_ds = monthly_payment * 12

    # Calculate actual ratios
    actual_ltv = loan_amount / stabilized_value if stabilized_value > 0 else 0
    actual_dscr = stabilized_noi / annual_ds if annual_ds > 0 else 0

    return PermanentLoan(
        loan_amount=loan_amount,
        ltv_constrained_amount=ltv_constrained,
        dscr_constrained_amount=dscr_constrained,
        binding_constraint=binding_constraint,
        interest_rate=effective_rate,
        amortization_months=amort_months,
        monthly_payment=monthly_payment,
        annual_debt_service=annual_ds,
        actual_ltv=actual_ltv,
        actual_dscr=actual_dscr,
    )


def calculate_loan_balance(
    original_principal: float,
    monthly_rate: float,
    monthly_payment: float,
    months_elapsed: int,
) -> float:
    """Calculate remaining loan balance after a number of payments.

    Uses the loan balance formula:
    Balance = P x (1 + r)^n - PMT x [((1 + r)^n - 1) / r]

    Args:
        original_principal: Original loan amount.
        monthly_rate: Monthly interest rate.
        monthly_payment: Monthly P&I payment.
        months_elapsed: Number of payments made.

    Returns:
        Remaining loan balance.
    """
    if monthly_rate == 0:
        # Simple case: no interest
        return original_principal - (monthly_payment * months_elapsed)

    # Standard amortization formula
    growth_factor = (1 + monthly_rate) ** months_elapsed
    balance = (
        original_principal * growth_factor
        - monthly_payment * ((growth_factor - 1) / monthly_rate)
    )

    return max(0.0, balance)  # Balance can't go negative


def calculate_interest_portion(
    balance: float,
    monthly_rate: float,
) -> float:
    """Calculate interest portion of a payment given current balance.

    Args:
        balance: Current loan balance.
        monthly_rate: Monthly interest rate.

    Returns:
        Interest amount for this period.
    """
    return balance * monthly_rate


def calculate_principal_portion(
    monthly_payment: float,
    interest: float,
) -> float:
    """Calculate principal portion of a payment.

    Args:
        monthly_payment: Total monthly payment.
        interest: Interest portion of payment.

    Returns:
        Principal portion of payment.
    """
    return monthly_payment - interest
