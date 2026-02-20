"""Calculation Trace View - Interactive drill-down into formulas and values.

This component displays traced calculations, allowing users to see:
- Formula definitions (symbolic)
- Actual values used in each formula
- Drill-down navigation through calculation dependencies
"""

import streamlit as st
from typing import Optional, Dict, List

from src.calculations.trace import TraceContext, TracedValue
from src.calculations.formula_registry import FormulaRegistry, FormulaCategory


def _format_value(value: float) -> str:
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


def render_trace_summary(trace_context: Optional[TraceContext]) -> None:
    """Render summary of all traced calculations.

    Args:
        trace_context: The TraceContext containing traced values
    """
    if trace_context is None or not trace_context.traces:
        st.warning("No calculation traces available. Ensure tracing is enabled.")
        return

    st.subheader("Calculation Trace Summary")
    st.caption(f"{len(trace_context.traces)} calculations traced")

    # Group traces by category
    by_category: Dict[str, List[TracedValue]] = {}
    for trace in trace_context.traces.values():
        if trace.formula_def:
            cat = trace.formula_def.category.value
        else:
            cat = "Uncategorized"
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(trace)

    # Display by category in expandable sections
    for category, traces in sorted(by_category.items()):
        with st.expander(f"{category} ({len(traces)} calculations)", expanded=False):
            for trace in traces:
                col1, col2 = st.columns([2, 3])
                with col1:
                    # Show field name and result
                    name = trace.formula_def.name if trace.formula_def else trace.field_path
                    st.markdown(f"**{name}**")
                    st.write(f"= {_format_value(trace.value)}")
                with col2:
                    # Show formula with values
                    if trace.formula_def:
                        st.code(trace.formula_def.formula, language=None)
                    st.caption(trace.computed_formula[:100] + "..." if len(trace.computed_formula) > 100 else trace.computed_formula)


def render_single_trace(trace: TracedValue, trace_context: TraceContext) -> None:
    """Render detailed view of a single traced calculation.

    Args:
        trace: The TracedValue to display
        trace_context: Context for looking up input traces
    """
    # Header
    if trace.formula_def:
        st.markdown(f"### {trace.formula_def.name}")
        st.caption(f"`{trace.field_path}`")
    else:
        st.markdown(f"### {trace.field_path}")

    # Result value prominently displayed
    st.metric("Result", _format_value(trace.value))

    # Formula definition
    st.markdown("**Formula:**")
    if trace.formula_def:
        st.code(trace.formula_def.formula, language=None)
        if trace.formula_def.notes:
            st.info(trace.formula_def.notes)

    # Computed formula with values
    st.markdown("**With Values:**")
    st.code(trace.computed_formula, language=None)

    # Input values table
    if trace.input_values:
        st.markdown("**Input Values:**")
        input_data = []
        for input_path, value in trace.input_values.items():
            # Check if we have a trace for this input (for drill-down)
            has_trace = input_path in trace_context.traces
            input_data.append({
                "Input": input_path.split(".")[-1],
                "Full Path": input_path,
                "Value": _format_value(value),
                "Traceable": "Yes" if has_trace else "No"
            })

        st.dataframe(input_data, use_container_width=True, hide_index=True)

    # Notes
    if trace.notes:
        st.markdown(f"**Notes:** {trace.notes}")

    # Period info if available
    if trace.period is not None:
        st.caption(f"Period: {trace.period}")


def render_calculation_trace_view(trace_context: Optional[TraceContext]) -> None:
    """Render interactive calculation trace viewer.

    Provides:
    - Summary view of all traces by category
    - Drill-down into individual calculations
    - Navigation through calculation dependencies

    Args:
        trace_context: The TraceContext from a calculation result
    """
    if trace_context is None or not trace_context.traces:
        st.info("No calculation traces available.")
        st.caption(
            "Traces are captured when calculate_deal() runs. "
            "Make sure you've run a calculation first."
        )
        return

    st.header("Calculation Trace Viewer")
    st.markdown(
        "Explore how each value was calculated. "
        "Click on a calculation to see its formula and inputs."
    )

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["By Category", "Search", "Formula Registry"])

    with tab1:
        render_trace_summary(trace_context)

    with tab2:
        # Search/select a specific trace
        trace_keys = sorted(trace_context.traces.keys())
        selected_trace = st.selectbox(
            "Select Calculation",
            trace_keys,
            format_func=lambda k: f"{k}: {_format_value(trace_context.traces[k].value)}"
        )

        if selected_trace:
            trace = trace_context.traces[selected_trace]
            render_single_trace(trace, trace_context)

            # Drill-down buttons for traceable inputs
            if trace.input_values:
                st.markdown("---")
                st.markdown("**Drill Down into Inputs:**")
                cols = st.columns(min(len(trace.input_values), 4))
                for i, (input_path, _) in enumerate(trace.input_values.items()):
                    if input_path in trace_context.traces:
                        with cols[i % len(cols)]:
                            if st.button(
                                input_path.split(".")[-1],
                                key=f"drill_{input_path}_{selected_trace}",
                                use_container_width=True
                            ):
                                st.session_state["selected_trace"] = input_path
                                st.rerun()

    with tab3:
        # Show all registered formulas
        st.subheader("Formula Registry")
        st.caption("All formula definitions available in the system")

        # Get all formulas from registry
        all_formulas = FormulaRegistry.get_all()

        if not all_formulas:
            st.info("No formulas registered.")
            return

        # Group by category
        by_cat: Dict[str, list] = {}
        for field_path, formula_def in all_formulas.items():
            cat = formula_def.category.value
            if cat not in by_cat:
                by_cat[cat] = []
            by_cat[cat].append((field_path, formula_def))

        for category, formulas in sorted(by_cat.items()):
            with st.expander(f"{category} ({len(formulas)} formulas)", expanded=False):
                for field_path, formula_def in formulas:
                    st.markdown(f"**{formula_def.name}** (`{field_path}`)")
                    st.code(formula_def.formula, language=None)
                    if formula_def.inputs:
                        st.caption(f"Inputs: {', '.join(formula_def.inputs)}")
                    if formula_def.notes:
                        st.caption(f"Note: {formula_def.notes}")
                    st.markdown("---")


def render_value_with_formula_tooltip(
    value: float,
    field_path: str,
    trace_context: Optional[TraceContext],
    period: Optional[int] = None,
) -> None:
    """Render a value with a popover showing its formula.

    This can be used inline in any view to add formula tooltips to values.

    Args:
        value: The numeric value to display
        field_path: The formula field path
        trace_context: TraceContext for lookup
        period: Optional period for period-specific traces
    """
    # Get trace if available
    trace_key = f"{field_path}:{period}" if period is not None else field_path
    trace = trace_context.traces.get(trace_key) if trace_context else None

    if trace:
        with st.popover(_format_value(value)):
            st.markdown(f"**{trace.formula_def.name if trace.formula_def else field_path}**")
            if trace.formula_def:
                st.code(trace.formula_def.formula, language=None)
            st.markdown("**Calculated as:**")
            st.code(trace.computed_formula, language=None)
            if trace.input_values:
                st.markdown("**Inputs:**")
                for name, val in trace.input_values.items():
                    st.write(f"  {name.split('.')[-1]}: {_format_value(val)}")
    else:
        # No trace available - just show the value
        st.write(_format_value(value))
