"""Tests for debt calculations including DSCR constraint."""

import pytest

from src.calculations.debt import (
    size_construction_loan,
    size_permanent_loan,
    calculate_loan_balance,
)


class TestSizeConstructionLoan:
    """Tests for construction loan sizing."""

    def test_construction_loan_is_ltc_of_tdc(self):
        """Construction loan should be LTC ratio of TDC."""
        loan = size_construction_loan(
            tdc=50_000_000,
            ltc_ratio=0.65,
            interest_rate=0.075,
            construction_months=24,
        )

        expected = 50_000_000 * 0.65
        assert loan.loan_amount == expected
        assert loan.ltc_ratio == 0.65


class TestSizePermanentLoan:
    """Tests for permanent loan sizing with dual constraints."""

    def test_ltv_constrains_when_value_is_low(self):
        """LTV should be binding when value is low relative to NOI."""
        loan = size_permanent_loan(
            stabilized_value=60_000_000,  # Low value
            stabilized_noi=5_000_000,  # High NOI
            ltv_max=0.65,
            dscr_min=1.25,
            perm_rate=0.06,
            amort_years=20,
        )

        assert loan.binding_constraint == "LTV"
        assert loan.loan_amount == loan.ltv_constrained_amount
        assert abs(loan.loan_amount - 39_000_000) < 1  # 60M * 0.65

    def test_dscr_constrains_when_noi_is_low(self):
        """DSCR should be binding when NOI is low relative to value."""
        loan = size_permanent_loan(
            stabilized_value=100_000_000,  # High value
            stabilized_noi=4_000_000,  # Lower NOI
            ltv_max=0.65,
            dscr_min=1.25,
            perm_rate=0.06,
            amort_years=20,
        )

        assert loan.binding_constraint == "DSCR"
        assert loan.loan_amount == loan.dscr_constrained_amount
        assert loan.loan_amount < loan.ltv_constrained_amount

    def test_actual_dscr_meets_minimum(self):
        """Actual DSCR should meet or exceed minimum."""
        loan = size_permanent_loan(
            stabilized_value=90_000_000,
            stabilized_noi=4_900_000,
            ltv_max=0.65,
            dscr_min=1.25,
            perm_rate=0.06,
            amort_years=20,
        )

        assert loan.actual_dscr >= 1.25 - 0.01  # Allow small rounding

    def test_actual_ltv_within_maximum(self):
        """Actual LTV should not exceed maximum."""
        loan = size_permanent_loan(
            stabilized_value=90_000_000,
            stabilized_noi=4_900_000,
            ltv_max=0.65,
            dscr_min=1.25,
            perm_rate=0.06,
            amort_years=20,
        )

        assert loan.actual_ltv <= 0.65 + 0.001  # Allow small rounding

    def test_rate_buydown_increases_loan_capacity(self):
        """Lower rate from buydown should allow larger DSCR-constrained loan."""
        loan_no_buydown = size_permanent_loan(
            stabilized_value=100_000_000,
            stabilized_noi=4_000_000,
            ltv_max=0.65,
            dscr_min=1.25,
            perm_rate=0.06,
            amort_years=20,
            rate_buydown_bps=0,
        )

        loan_with_buydown = size_permanent_loan(
            stabilized_value=100_000_000,
            stabilized_noi=4_000_000,
            ltv_max=0.65,
            dscr_min=1.25,
            perm_rate=0.06,
            amort_years=20,
            rate_buydown_bps=150,  # 1.5% buydown
        )

        # With DSCR binding, lower rate = higher loan capacity
        if loan_no_buydown.binding_constraint == "DSCR":
            assert loan_with_buydown.loan_amount >= loan_no_buydown.loan_amount

    def test_monthly_payment_is_correct(self):
        """Monthly payment should service the loan correctly."""
        loan = size_permanent_loan(
            stabilized_value=90_000_000,
            stabilized_noi=4_900_000,
            ltv_max=0.65,
            dscr_min=1.25,
            perm_rate=0.06,
            amort_years=20,
        )

        # Annual debt service should equal 12 * monthly payment
        assert abs(loan.annual_debt_service - loan.monthly_payment * 12) < 1


class TestCalculateLoanBalance:
    """Tests for loan balance calculation."""

    def test_balance_decreases_over_time(self):
        """Loan balance should decrease with payments."""
        principal = 50_000_000
        monthly_rate = 0.06 / 12
        monthly_payment = 430_000  # Approximate

        balance_12 = calculate_loan_balance(principal, monthly_rate, monthly_payment, 12)
        balance_24 = calculate_loan_balance(principal, monthly_rate, monthly_payment, 24)

        assert balance_24 < balance_12 < principal

    def test_balance_is_zero_at_maturity(self):
        """Balance should be zero at end of amortization."""
        import numpy_financial as npf

        principal = 50_000_000
        rate = 0.06
        years = 20
        monthly_rate = rate / 12
        months = years * 12

        # Calculate correct payment
        payment = -npf.pmt(monthly_rate, months, principal)

        balance = calculate_loan_balance(principal, monthly_rate, payment, months)

        assert abs(balance) < 1  # Should be essentially zero

    def test_balance_never_negative(self):
        """Balance should never go below zero."""
        balance = calculate_loan_balance(
            original_principal=1_000_000,
            monthly_rate=0.005,
            monthly_payment=50_000,  # Very high payment
            months_elapsed=100,  # Way past payoff
        )

        assert balance >= 0
