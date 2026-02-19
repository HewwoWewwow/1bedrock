"""Tests for cost calculations including iterative IDC."""

import pytest

from src.calculations.costs import calculate_idc, calculate_tdc


class TestCalculateIDC:
    """Tests for iterative IDC calculation."""

    def test_idc_converges(self):
        """IDC calculation should converge within tolerance."""
        idc, iterations = calculate_idc(
            costs_before_idc=45_000_000,
            construction_ltc=0.65,
            construction_rate=0.075,
            construction_months=24,
            tolerance=100.0,
        )

        # Should converge before max iterations
        assert iterations < 50
        # IDC should be positive and reasonable
        # With S-curve draws (debt builds gradually), IDC is ~4% of costs
        # For $45M at 65% LTC over 24 months at 7.5%, expect ~$1.5-2.5M
        assert 1_000_000 < idc < 3_000_000

    def test_idc_increases_with_longer_construction(self):
        """Longer construction period should increase IDC."""
        idc_24mo, _ = calculate_idc(
            costs_before_idc=45_000_000,
            construction_ltc=0.65,
            construction_rate=0.075,
            construction_months=24,
        )

        idc_30mo, _ = calculate_idc(
            costs_before_idc=45_000_000,
            construction_ltc=0.65,
            construction_rate=0.075,
            construction_months=30,
        )

        assert idc_30mo > idc_24mo

    def test_idc_increases_with_higher_rate(self):
        """Higher interest rate should increase IDC."""
        idc_low, _ = calculate_idc(
            costs_before_idc=45_000_000,
            construction_ltc=0.65,
            construction_rate=0.06,
            construction_months=24,
        )

        idc_high, _ = calculate_idc(
            costs_before_idc=45_000_000,
            construction_ltc=0.65,
            construction_rate=0.10,
            construction_months=24,
        )

        assert idc_high > idc_low

    def test_idc_increases_with_higher_ltc(self):
        """Higher LTC ratio should increase IDC."""
        idc_low_ltc, _ = calculate_idc(
            costs_before_idc=45_000_000,
            construction_ltc=0.50,
            construction_rate=0.075,
            construction_months=24,
        )

        idc_high_ltc, _ = calculate_idc(
            costs_before_idc=45_000_000,
            construction_ltc=0.75,
            construction_rate=0.075,
            construction_months=24,
        )

        assert idc_high_ltc > idc_low_ltc

    def test_idc_is_stable(self):
        """Re-running with converged IDC should give same result."""
        idc1, _ = calculate_idc(
            costs_before_idc=45_000_000,
            construction_ltc=0.65,
            construction_rate=0.075,
            construction_months=24,
        )

        # Run again - should get essentially same result
        idc2, _ = calculate_idc(
            costs_before_idc=45_000_000,
            construction_ltc=0.65,
            construction_rate=0.075,
            construction_months=24,
        )

        assert abs(idc2 - idc1) < 100  # Within tolerance


class TestCalculateTDC:
    """Tests for TDC calculation."""

    def test_tdc_components_sum_correctly(self):
        """TDC should equal sum of components."""
        result = calculate_tdc(
            land_cost=3_000_000,
            target_units=200,
            hard_cost_per_unit=155_000,
            soft_cost_pct=0.30,
            construction_ltc=0.65,
            construction_rate=0.075,
            construction_months=24,
        )

        # TDC = land + hard + soft + financing_costs_total (which includes IDC + loan_fee)
        expected_sum = (
            result.land_cost
            + result.hard_costs
            + result.soft_costs_after_waiver
            + result.financing_costs_total  # IDC + loan_fee
        )
        assert abs(result.tdc - expected_sum) < 1  # Within $1 rounding

    def test_fee_waiver_reduces_tdc(self):
        """Fee waiver should reduce TDC."""
        tdc_no_waiver = calculate_tdc(
            land_cost=3_000_000,
            target_units=200,
            hard_cost_per_unit=155_000,
            soft_cost_pct=0.30,
            construction_ltc=0.65,
            construction_rate=0.075,
            construction_months=24,
            fee_waiver_amount=0,
        )

        tdc_with_waiver = calculate_tdc(
            land_cost=3_000_000,
            target_units=200,
            hard_cost_per_unit=155_000,
            soft_cost_pct=0.30,
            construction_ltc=0.65,
            construction_rate=0.075,
            construction_months=24,
            fee_waiver_amount=700_000,  # ~$3500/unit * 200 affordable units
        )

        assert tdc_with_waiver.tdc < tdc_no_waiver.tdc
        assert tdc_with_waiver.waiver_amount == 700_000

    def test_soft_costs_are_percentage_of_hard(self):
        """Soft costs should be correct percentage of hard costs."""
        result = calculate_tdc(
            land_cost=3_000_000,
            target_units=200,
            hard_cost_per_unit=155_000,
            soft_cost_pct=0.30,
            construction_ltc=0.65,
            construction_rate=0.075,
            construction_months=24,
        )

        expected_soft = result.hard_costs * 0.30
        assert abs(result.soft_costs - expected_soft) < 1

    def test_spec_inputs_tdc_approximately_correct(self, spec_inputs):
        """TDC with spec inputs should be approximately $50.4M."""
        from src.calculations.detailed_cashflow import calculate_deal
        from src.models.project import Scenario

        # Use calculate_deal as SINGLE SOURCE OF TRUTH
        result = calculate_deal(spec_inputs, Scenario.MARKET)

        # Should be within 10% of expected $50.4M
        from tests.fixtures.test_inputs import EXPECTED_MARKET_TDC
        tolerance = EXPECTED_MARKET_TDC * 0.10
        assert abs(result.sources_uses.tdc - EXPECTED_MARKET_TDC) < tolerance
