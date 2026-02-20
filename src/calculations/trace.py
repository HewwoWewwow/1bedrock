"""Calculation tracing for transparent audit trails.

This module provides runtime tracing of calculations, capturing
the actual values used in each formula for debugging and auditing.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import re

from .formula_registry import FormulaRegistry, FormulaDefinition


@dataclass
class TracedValue:
    """A single traced calculation.

    Captures the formula definition, actual input values,
    computed result, and formatted formula string.
    """
    field_path: str
    value: float
    formula_def: Optional[FormulaDefinition]
    input_values: Dict[str, float]
    computed_formula: str  # Formula with values substituted
    timestamp: datetime = field(default_factory=datetime.now)
    period: Optional[int] = None
    notes: str = ""

    def format_inputs(self) -> str:
        """Format input values for display."""
        parts = []
        for name, val in self.input_values.items():
            short_name = name.split(".")[-1]
            if isinstance(val, float):
                if abs(val) >= 1_000_000:
                    parts.append(f"{short_name}=${val/1_000_000:,.2f}M")
                elif abs(val) >= 1_000:
                    parts.append(f"{short_name}=${val/1_000:,.1f}K")
                elif abs(val) < 1 and val != 0:
                    parts.append(f"{short_name}={val:.2%}")
                else:
                    parts.append(f"{short_name}=${val:,.0f}")
            else:
                parts.append(f"{short_name}={val}")
        return ", ".join(parts)


class TraceContext:
    """Context manager for capturing calculation traces.

    Usage:
        with TraceContext() as ctx:
            result = calculate_deal(inputs, scenario)
            # ctx.traces now contains all traced calculations

    The context is stored in a class variable so trace() calls
    can access it from anywhere in the call stack.
    """
    _current: Optional['TraceContext'] = None

    def __init__(self, enabled: bool = True):
        """Initialize trace context.

        Args:
            enabled: If False, trace() calls are no-ops for performance.
        """
        self.enabled = enabled
        self.traces: Dict[str, TracedValue] = {}
        self._start_time = datetime.now()

    def __enter__(self) -> 'TraceContext':
        TraceContext._current = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        TraceContext._current = None

    def trace(
        self,
        field_path: str,
        value: float,
        input_values: Dict[str, float],
        period: Optional[int] = None,
        notes: str = "",
    ) -> None:
        """Record a traced calculation.

        Args:
            field_path: The formula field path (e.g., "sources_uses.tdc")
            value: The calculated result
            input_values: Dict of input name -> value used in calculation
            period: Optional period number for period-specific values
            notes: Optional notes about this specific calculation
        """
        if not self.enabled:
            return

        formula_def = FormulaRegistry.get(field_path)
        computed_formula = self._substitute_values(
            formula_def.formula if formula_def else field_path,
            input_values,
            value
        )

        # Create unique key for period-specific values
        trace_key = f"{field_path}:{period}" if period is not None else field_path

        self.traces[trace_key] = TracedValue(
            field_path=field_path,
            value=value,
            formula_def=formula_def,
            input_values=input_values,
            computed_formula=computed_formula,
            timestamp=datetime.now(),
            period=period,
            notes=notes,
        )

    def _substitute_values(
        self,
        formula: str,
        input_values: Dict[str, float],
        result: float
    ) -> str:
        """Substitute actual values into a formula string.

        Args:
            formula: Symbolic formula (e.g., "land + hard_costs + soft_costs")
            input_values: Dict of values to substitute
            result: The calculated result

        Returns:
            Formatted string like "$3M + $31M + $9.3M = $43.3M"
        """
        if not input_values:
            return f"{formula} = {self._format_value(result)}"

        # Build value string
        value_parts = []
        for name, val in input_values.items():
            value_parts.append(self._format_value(val))

        values_str = " + ".join(value_parts) if len(value_parts) > 1 else value_parts[0] if value_parts else ""
        result_str = self._format_value(result)

        return f"{formula} = {values_str} = {result_str}"

    def _format_value(self, value: float) -> str:
        """Format a value for display."""
        if abs(value) >= 1_000_000:
            return f"${value/1_000_000:,.2f}M"
        elif abs(value) >= 1_000:
            return f"${value/1_000:,.1f}K"
        elif abs(value) < 1 and value != 0:
            return f"{value:.2%}"
        elif value == 0:
            return "$0"
        else:
            return f"${value:,.0f}"

    def get_trace(self, field_path: str, period: Optional[int] = None) -> Optional[TracedValue]:
        """Get a specific trace by field path and optional period."""
        trace_key = f"{field_path}:{period}" if period is not None else field_path
        return self.traces.get(trace_key)

    def get_traces_for_period(self, period: int) -> Dict[str, TracedValue]:
        """Get all traces for a specific period."""
        return {
            k: v for k, v in self.traces.items()
            if v.period == period
        }

    def get_traces_by_category(self, category: str) -> Dict[str, TracedValue]:
        """Get all traces in a specific category."""
        return {
            k: v for k, v in self.traces.items()
            if v.formula_def and v.formula_def.category.value == category
        }

    def get_calculation_chain(self, field_path: str, period: Optional[int] = None) -> List[TracedValue]:
        """Get the full calculation chain for a value (all upstream traces).

        Returns traces in order from inputs to final value.
        """
        chain = []
        visited = set()

        def _collect_chain(path: str, per: Optional[int]):
            trace_key = f"{path}:{per}" if per is not None else path
            if trace_key in visited:
                return
            visited.add(trace_key)

            trace = self.get_trace(path, per)
            if trace:
                # First collect all inputs
                for input_path in trace.input_values.keys():
                    _collect_chain(input_path, per)
                # Then add this trace
                chain.append(trace)

        _collect_chain(field_path, period)
        return chain

    def summary(self) -> str:
        """Generate a summary of all traces."""
        lines = [
            f"Trace Summary ({len(self.traces)} calculations traced)",
            f"Duration: {datetime.now() - self._start_time}",
            "",
        ]

        # Group by category
        by_category: Dict[str, List[TracedValue]] = {}
        for trace in self.traces.values():
            cat = trace.formula_def.category.value if trace.formula_def else "Unknown"
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(trace)

        for category, traces in sorted(by_category.items()):
            lines.append(f"=== {category} ({len(traces)} traces) ===")
            for trace in traces[:5]:  # Show first 5 per category
                lines.append(f"  {trace.field_path}: {trace.computed_formula}")
            if len(traces) > 5:
                lines.append(f"  ... and {len(traces) - 5} more")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def current() -> Optional['TraceContext']:
        """Get the current active trace context."""
        return TraceContext._current


def trace(
    field_path: str,
    value: float,
    input_values: Dict[str, float],
    period: Optional[int] = None,
    notes: str = "",
) -> float:
    """Convenience function to trace a calculation and return the value.

    This can be used inline in calculations:
        noi = trace("operations.noi", egi - opex, {"egi": egi, "opex": opex})

    Args:
        field_path: The formula field path
        value: The calculated result
        input_values: Dict of input name -> value
        period: Optional period number
        notes: Optional notes

    Returns:
        The value (unchanged), allowing inline usage
    """
    ctx = TraceContext.current()
    if ctx:
        ctx.trace(field_path, value, input_values, period, notes)
    return value
