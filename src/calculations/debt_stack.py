"""Debt stack calculations for multiple tranches (senior, mezzanine, etc.)."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict


class DebtType(Enum):
    """Type of debt instrument."""
    SENIOR = "senior"           # Construction / permanent loan
    MEZZANINE = "mezzanine"     # Junior debt, higher rate
    PREFERRED = "preferred"     # Preferred equity (debt-like)


@dataclass
class DebtTranche:
    """A single debt tranche with its terms."""
    name: str
    debt_type: DebtType
    principal: float = 0.0
    interest_rate: float = 0.0          # Annual rate
    amortization_years: int = 0         # 0 = interest-only
    term_months: int = 0                # Loan term
    is_active: bool = False

    # Calculated fields
    monthly_rate: float = field(init=False)
    monthly_payment: float = field(init=False)
    monthly_interest_only: float = field(init=False)

    def __post_init__(self):
        self.monthly_rate = self.interest_rate / 12
        self.monthly_interest_only = self.principal * self.monthly_rate

        if self.amortization_years > 0 and self.principal > 0:
            # Amortizing payment calculation
            n_payments = self.amortization_years * 12
            if self.monthly_rate > 0:
                self.monthly_payment = (
                    self.principal *
                    (self.monthly_rate * (1 + self.monthly_rate) ** n_payments) /
                    ((1 + self.monthly_rate) ** n_payments - 1)
                )
            else:
                self.monthly_payment = self.principal / n_payments
        else:
            # Interest-only
            self.monthly_payment = self.monthly_interest_only


@dataclass
class DebtServicePeriod:
    """Debt service for a single period across all tranches."""
    period: int
    senior_interest: float = 0.0
    senior_principal: float = 0.0
    senior_payment: float = 0.0
    senior_balance: float = 0.0

    mezzanine_interest: float = 0.0
    mezzanine_principal: float = 0.0
    mezzanine_payment: float = 0.0
    mezzanine_balance: float = 0.0

    preferred_dividend: float = 0.0
    preferred_balance: float = 0.0

    total_debt_service: float = 0.0


@dataclass
class DebtStack:
    """Complete debt stack with all tranches."""
    senior: Optional[DebtTranche] = None
    mezzanine: Optional[DebtTranche] = None
    preferred: Optional[DebtTranche] = None

    @property
    def total_debt(self) -> float:
        """Total debt across all tranches."""
        total = 0.0
        if self.senior and self.senior.is_active:
            total += self.senior.principal
        if self.mezzanine and self.mezzanine.is_active:
            total += self.mezzanine.principal
        if self.preferred and self.preferred.is_active:
            total += self.preferred.principal
        return total

    @property
    def active_tranches(self) -> List[DebtTranche]:
        """List of active debt tranches."""
        tranches = []
        if self.senior and self.senior.is_active:
            tranches.append(self.senior)
        if self.mezzanine and self.mezzanine.is_active:
            tranches.append(self.mezzanine)
        if self.preferred and self.preferred.is_active:
            tranches.append(self.preferred)
        return tranches


def calculate_debt_service_schedule(
    debt_stack: DebtStack,
    num_periods: int,
    start_period: int = 1,  # When debt service begins (e.g., after construction)
) -> List[DebtServicePeriod]:
    """Calculate debt service schedule for all tranches.

    Args:
        debt_stack: DebtStack with configured tranches
        num_periods: Total number of periods
        start_period: Period when debt service begins

    Returns:
        List of DebtServicePeriod for each period
    """
    schedule = []

    # Initialize balances
    senior_balance = debt_stack.senior.principal if debt_stack.senior and debt_stack.senior.is_active else 0
    mezz_balance = debt_stack.mezzanine.principal if debt_stack.mezzanine and debt_stack.mezzanine.is_active else 0
    pref_balance = debt_stack.preferred.principal if debt_stack.preferred and debt_stack.preferred.is_active else 0

    for period in range(1, num_periods + 1):
        ds = DebtServicePeriod(period=period)

        if period >= start_period:
            # Senior debt service
            if debt_stack.senior and debt_stack.senior.is_active and senior_balance > 0:
                ds.senior_interest = senior_balance * debt_stack.senior.monthly_rate
                if debt_stack.senior.amortization_years > 0:
                    ds.senior_payment = min(debt_stack.senior.monthly_payment, senior_balance + ds.senior_interest)
                    ds.senior_principal = ds.senior_payment - ds.senior_interest
                else:
                    ds.senior_payment = ds.senior_interest
                    ds.senior_principal = 0
                senior_balance = max(0, senior_balance - ds.senior_principal)
                ds.senior_balance = senior_balance

            # Mezzanine debt service
            if debt_stack.mezzanine and debt_stack.mezzanine.is_active and mezz_balance > 0:
                ds.mezzanine_interest = mezz_balance * debt_stack.mezzanine.monthly_rate
                if debt_stack.mezzanine.amortization_years > 0:
                    ds.mezzanine_payment = min(debt_stack.mezzanine.monthly_payment, mezz_balance + ds.mezzanine_interest)
                    ds.mezzanine_principal = ds.mezzanine_payment - ds.mezzanine_interest
                else:
                    ds.mezzanine_payment = ds.mezzanine_interest
                    ds.mezzanine_principal = 0
                mezz_balance = max(0, mezz_balance - ds.mezzanine_principal)
                ds.mezzanine_balance = mezz_balance

            # Preferred equity dividend (treated like interest-only)
            if debt_stack.preferred and debt_stack.preferred.is_active and pref_balance > 0:
                ds.preferred_dividend = pref_balance * debt_stack.preferred.monthly_rate
                ds.preferred_balance = pref_balance  # Preferred doesn't amortize

            # Total debt service
            ds.total_debt_service = (
                ds.senior_payment +
                ds.mezzanine_payment +
                ds.preferred_dividend
            )
        else:
            # Before debt service starts, just track balances
            ds.senior_balance = senior_balance
            ds.mezzanine_balance = mezz_balance
            ds.preferred_balance = pref_balance

        schedule.append(ds)

    return schedule


def create_debt_stack(
    senior_principal: float = 0.0,
    senior_rate: float = 0.06,
    senior_amort_years: int = 25,
    senior_term_months: int = 120,

    mezzanine_principal: float = 0.0,
    mezzanine_rate: float = 0.12,
    mezzanine_amort_years: int = 0,  # Usually I/O
    mezzanine_term_months: int = 60,

    preferred_principal: float = 0.0,
    preferred_rate: float = 0.10,
) -> DebtStack:
    """Create a debt stack with the specified tranches.

    Args:
        senior_principal: Senior loan amount
        senior_rate: Senior loan annual interest rate
        senior_amort_years: Senior loan amortization (0 = I/O)
        senior_term_months: Senior loan term

        mezzanine_principal: Mezzanine loan amount
        mezzanine_rate: Mezzanine loan annual interest rate
        mezzanine_amort_years: Mezzanine amortization (0 = I/O)
        mezzanine_term_months: Mezzanine term

        preferred_principal: Preferred equity amount
        preferred_rate: Preferred return rate

    Returns:
        Configured DebtStack
    """
    senior = DebtTranche(
        name="Senior Loan",
        debt_type=DebtType.SENIOR,
        principal=senior_principal,
        interest_rate=senior_rate,
        amortization_years=senior_amort_years,
        term_months=senior_term_months,
        is_active=senior_principal > 0,
    ) if senior_principal > 0 else None

    mezzanine = DebtTranche(
        name="Mezzanine Debt",
        debt_type=DebtType.MEZZANINE,
        principal=mezzanine_principal,
        interest_rate=mezzanine_rate,
        amortization_years=mezzanine_amort_years,
        term_months=mezzanine_term_months,
        is_active=mezzanine_principal > 0,
    ) if mezzanine_principal > 0 else None

    preferred = DebtTranche(
        name="Preferred Equity",
        debt_type=DebtType.PREFERRED,
        principal=preferred_principal,
        interest_rate=preferred_rate,
        amortization_years=0,  # Always I/O
        term_months=0,
        is_active=preferred_principal > 0,
    ) if preferred_principal > 0 else None

    return DebtStack(
        senior=senior,
        mezzanine=mezzanine,
        preferred=preferred,
    )
