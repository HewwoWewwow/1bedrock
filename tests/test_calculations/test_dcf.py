"""Tests for DCF engine using calculate_deal (SINGLE SOURCE OF TRUTH)."""

import pytest

from src.models.project import ProjectInputs, Scenario
from src.calculations.detailed_cashflow import calculate_deal
from src.calculations.draw_schedule import Phase


class TestCalculateDeal:
    """Tests for the unified calculation engine."""

    def test_market_scenario_produces_results(self, spec_inputs):
        """Market scenario should produce valid results."""
        result = calculate_deal(spec_inputs, Scenario.MARKET)

        assert result.sources_uses.tdc > 0
        assert result.sources_uses.equity > 0
        assert result.sources_uses.construction_loan > 0
        assert result.total_noi > 0
        assert len(result.periods) == spec_inputs.total_months

    def test_mixed_scenario_produces_results(self, spec_inputs_with_incentives):
        """Mixed-income scenario should produce valid results."""
        result = calculate_deal(spec_inputs_with_incentives, Scenario.MIXED_INCOME)

        assert result.sources_uses.tdc > 0
        assert result.sources_uses.equity > 0
        assert result.sources_uses.construction_loan > 0
        assert len(result.periods) == spec_inputs_with_incentives.total_months

    def test_mixed_has_lower_tdc_with_fee_waiver(self, spec_inputs, spec_inputs_with_incentives):
        """Mixed-income with fee waiver should have lower TDC."""
        market_result = calculate_deal(spec_inputs, Scenario.MARKET)
        mixed_result = calculate_deal(spec_inputs_with_incentives, Scenario.MIXED_INCOME)

        # With fee waivers, TDC should be lower
        assert mixed_result.sources_uses.tdc <= market_result.sources_uses.tdc

    def test_periods_cover_full_timeline(self, spec_inputs):
        """Periods should cover predevelopment through reversion."""
        result = calculate_deal(spec_inputs, Scenario.MARKET)

        # Check we have all phases
        has_predev = any(p.header.is_predevelopment for p in result.periods)
        has_construction = any(p.header.is_construction for p in result.periods)
        has_leaseup = any(p.header.is_leaseup for p in result.periods)
        has_operations = any(p.header.is_operations for p in result.periods)
        has_reversion = any(p.header.is_reversion for p in result.periods)

        assert has_predev, "Missing predevelopment periods"
        assert has_construction, "Missing construction periods"
        assert has_leaseup, "Missing lease-up periods"
        assert has_operations, "Missing operations periods"
        assert has_reversion, "Missing reversion period"

    def test_reversion_has_sale_proceeds(self, spec_inputs):
        """Reversion month should have sale proceeds."""
        result = calculate_deal(spec_inputs, Scenario.MARKET)

        reversion_periods = [p for p in result.periods if p.header.is_reversion]
        assert len(reversion_periods) == 1

        reversion_cf = reversion_periods[0]
        assert reversion_cf.investment.reversion > 0, "Reversion should have sale proceeds"

    def test_operations_has_positive_noi(self, spec_inputs):
        """Operations period should have positive NOI."""
        result = calculate_deal(spec_inputs, Scenario.MARKET)

        ops_periods = [p for p in result.periods if p.header.is_operations]
        assert len(ops_periods) > 0, "Should have operations periods"

        for cf in ops_periods:
            assert cf.operations.noi > 0, f"Period {cf.header.period} has non-positive NOI"

    def test_leaseup_occupancy_ramps(self, spec_inputs):
        """Occupancy should ramp during lease-up."""
        result = calculate_deal(spec_inputs, Scenario.MARKET)

        leaseup_periods = [p for p in result.periods if p.header.is_leaseup]
        assert len(leaseup_periods) > 0, "Should have lease-up periods"

        # First month should have lower occupancy than last
        first_occ = leaseup_periods[0].operations.leaseup_pct
        last_occ = leaseup_periods[-1].operations.leaseup_pct
        assert first_occ < last_occ, "Occupancy should increase during lease-up"

        # Occupancy should increase monotonically
        for i in range(1, len(leaseup_periods)):
            curr_occ = leaseup_periods[i].operations.leaseup_pct
            prev_occ = leaseup_periods[i-1].operations.leaseup_pct
            assert curr_occ >= prev_occ, f"Occupancy should not decrease (period {i})"

    def test_irr_calculation_completes(self, spec_inputs):
        """IRR calculation should complete without error."""
        result = calculate_deal(spec_inputs, Scenario.MARKET)

        # IRR should be a valid number (not NaN or infinity)
        import math
        assert not math.isnan(result.levered_irr), "Levered IRR should not be NaN"
        assert not math.isinf(result.levered_irr), "Levered IRR should not be infinite"
        assert not math.isnan(result.unlevered_irr), "Unlevered IRR should not be NaN"
        assert not math.isinf(result.unlevered_irr), "Unlevered IRR should not be infinite"

    def test_yield_on_cost_is_reasonable(self, spec_inputs):
        """Yield on cost should be in reasonable range."""
        result = calculate_deal(spec_inputs, Scenario.MARKET)

        # Calculate yield on cost: stabilized NOI / TDC
        yoc = result.total_noi / result.sources_uses.tdc if result.sources_uses.tdc > 0 else 0

        # YoC typically 4-12% for multifamily new construction
        assert 0.03 < yoc < 0.15, \
            f"Yield on cost {yoc:.1%} outside reasonable range (3-15%)"


class TestDCFSpecValidation:
    """Tests validating deal economics."""

    def test_market_produces_valid_metrics(self, spec_inputs):
        """Market scenario should produce valid financial metrics."""
        result = calculate_deal(spec_inputs, Scenario.MARKET)

        # Basic validity checks
        assert result.sources_uses.tdc > 0
        assert result.sources_uses.equity > 0
        assert result.total_noi > 0
        assert result.reversion_value > 0

        # LTC should be around 65%
        ltc = result.sources_uses.construction_loan / result.sources_uses.tdc
        assert 0.60 < ltc < 0.70, f"LTC {ltc:.1%} outside expected range"

    def test_mixed_income_has_lower_revenue(self, spec_inputs, spec_inputs_with_incentives):
        """Mixed-income should have lower GPR due to affordable units."""
        market_result = calculate_deal(spec_inputs, Scenario.MARKET)
        mixed_result = calculate_deal(spec_inputs_with_incentives, Scenario.MIXED_INCOME)

        # Get stabilized GPR from operations periods
        market_ops = [p for p in market_result.periods if p.header.is_operations]
        mixed_ops = [p for p in mixed_result.periods if p.header.is_operations]

        if market_ops and mixed_ops:
            market_gpr = market_ops[0].operations.gpr
            mixed_gpr = mixed_ops[0].operations.gpr

            # Mixed should have lower GPR due to affordable rents
            assert mixed_gpr < market_gpr, \
                f"Mixed GPR ({mixed_gpr:,.0f}) should be lower than market ({market_gpr:,.0f})"

    def test_equity_plus_debt_equals_tdc(self, spec_inputs):
        """Sources should equal uses (equity + debt = TDC)."""
        result = calculate_deal(spec_inputs, Scenario.MARKET)

        sources = result.sources_uses.equity + result.sources_uses.construction_loan
        uses = result.sources_uses.tdc

        # Allow small rounding tolerance
        assert abs(sources - uses) < 100, \
            f"Sources ({sources:,.0f}) should equal uses ({uses:,.0f})"


class TestPhaseTimings:
    """Tests for correct phase boundaries."""

    def test_predevelopment_period_count(self, spec_inputs):
        """Should have correct number of predevelopment periods."""
        result = calculate_deal(spec_inputs, Scenario.MARKET)

        predev_periods = [p for p in result.periods if p.header.is_predevelopment]
        assert len(predev_periods) == spec_inputs.predevelopment_months

    def test_construction_period_count(self, spec_inputs):
        """Should have correct number of construction periods."""
        result = calculate_deal(spec_inputs, Scenario.MARKET)

        construction_periods = [p for p in result.periods if p.header.is_construction]
        assert len(construction_periods) == spec_inputs.construction_months

    def test_leaseup_period_count(self, spec_inputs):
        """Should have correct number of lease-up periods."""
        result = calculate_deal(spec_inputs, Scenario.MARKET)

        leaseup_periods = [p for p in result.periods if p.header.is_leaseup]
        assert len(leaseup_periods) == spec_inputs.leaseup_months

    def test_operations_period_count(self, spec_inputs):
        """Should have correct number of operations periods."""
        result = calculate_deal(spec_inputs, Scenario.MARKET)

        operations_periods = [p for p in result.periods if p.header.is_operations]
        assert len(operations_periods) == spec_inputs.operations_months

    def test_reversion_is_final_period(self, spec_inputs):
        """Reversion should be the final period."""
        result = calculate_deal(spec_inputs, Scenario.MARKET)

        final_period = result.periods[-1]
        assert final_period.header.is_reversion, "Final period should be reversion"
        assert final_period.header.period == spec_inputs.total_months
