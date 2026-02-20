"""Tests for the calculation tracing system."""

import pytest
from src.calculations.trace import TraceContext, trace, TracedValue
from src.calculations.formula_registry import FormulaRegistry, FormulaCategory


class TestFormulaRegistry:
    """Test the formula registry."""

    def test_formulas_are_registered(self):
        """Verify that formulas are registered at import time."""
        # Get all formulas
        all_formulas = FormulaRegistry.get_all()

        # Should have many formulas registered
        assert len(all_formulas) >= 40, "Expected at least 40 formulas registered"

    def test_can_get_formula_by_path(self):
        """Can retrieve a specific formula."""
        formula = FormulaRegistry.get("sources_uses.tdc")

        assert formula is not None
        assert formula.name == "Total Development Cost"
        assert "land" in formula.formula.lower()

    def test_can_get_by_category(self):
        """Can filter formulas by category."""
        dev_formulas = FormulaRegistry.get_by_category(FormulaCategory.DEVELOPMENT)

        assert len(dev_formulas) > 0
        for formula in dev_formulas:
            assert formula.category == FormulaCategory.DEVELOPMENT

    def test_can_get_dependents(self):
        """Can find formulas that depend on a given field."""
        # TDC should be used by other formulas
        dependents = FormulaRegistry.get_dependents("sources_uses.tdc")

        # At least equity and construction loan depend on TDC
        assert len(dependents) >= 2


class TestTraceContext:
    """Test the trace context manager."""

    def test_trace_context_captures_traces(self):
        """TraceContext captures trace calls."""
        with TraceContext() as ctx:
            trace("test.value", 100.0, {"input_a": 50.0, "input_b": 50.0})

        assert "test.value" in ctx.traces
        traced = ctx.traces["test.value"]
        assert traced.value == 100.0
        assert traced.input_values["input_a"] == 50.0

    def test_trace_context_can_be_disabled(self):
        """Disabled TraceContext does not capture traces."""
        with TraceContext(enabled=False) as ctx:
            trace("test.value", 100.0, {"input_a": 50.0})

        assert len(ctx.traces) == 0

    def test_trace_with_period(self):
        """Traces can include period information."""
        with TraceContext() as ctx:
            trace("test.value", 100.0, {"input": 100.0}, period=5)

        assert "test.value:5" in ctx.traces
        traced = ctx.traces["test.value:5"]
        assert traced.period == 5

    def test_trace_returns_value(self):
        """trace() returns the value for inline usage."""
        with TraceContext() as ctx:
            result = trace("test.value", 42.0, {"x": 42.0})

        assert result == 42.0

    def test_nested_context_not_supported(self):
        """Only one TraceContext can be active at a time."""
        with TraceContext() as outer:
            # Inner context replaces outer
            with TraceContext() as inner:
                trace("inner.value", 1.0, {})

            # Outer is no longer current
            assert TraceContext.current() is None

        # After both exit, no current context
        assert TraceContext.current() is None


class TestTraceIntegration:
    """Integration tests for tracing with calculate_deal."""

    def test_calculate_deal_captures_traces(self):
        """calculate_deal captures traces in its result."""
        from src.models.project import ProjectInputs, Scenario
        from src.calculations.detailed_cashflow import calculate_deal

        inputs = ProjectInputs()
        result = calculate_deal(inputs, Scenario.MARKET)

        # Result should have trace context
        assert result.trace_context is not None
        assert len(result.trace_context.traces) > 0

    def test_key_formulas_are_traced(self):
        """Key formulas should have traces."""
        from src.models.project import ProjectInputs, Scenario
        from src.calculations.detailed_cashflow import calculate_deal

        inputs = ProjectInputs()
        result = calculate_deal(inputs, Scenario.MARKET)

        ctx = result.trace_context
        assert ctx is not None

        # Check key traces exist
        key_traces = [
            "sources_uses.tdc",
            "sources_uses.equity",
            "gpr.gpr_total",
            "operations.egi",
            "perm_loan.amount",
            "returns.levered_irr",
        ]

        for trace_key in key_traces:
            assert trace_key in ctx.traces, f"Missing trace: {trace_key}"

    def test_traced_values_match_results(self):
        """Traced values should match actual result values."""
        from src.models.project import ProjectInputs, Scenario
        from src.calculations.detailed_cashflow import calculate_deal

        inputs = ProjectInputs()
        result = calculate_deal(inputs, Scenario.MARKET)

        ctx = result.trace_context
        assert ctx is not None

        # TDC trace should match result
        tdc_trace = ctx.traces.get("sources_uses.tdc")
        assert tdc_trace is not None
        assert abs(tdc_trace.value - result.sources_uses.tdc) < 1.0

        # IRR trace should match result
        irr_trace = ctx.traces.get("returns.levered_irr")
        assert irr_trace is not None
        assert abs(irr_trace.value - result.levered_irr) < 0.0001
