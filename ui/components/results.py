"""Results display components for the Streamlit UI."""

import streamlit as st
import pandas as pd
from typing import List

from src.calculations.metrics import ScenarioMetrics, ScenarioComparison
from src.calculations.dcf import DCFResult, MonthlyCashFlow


def render_comparison_table(
    market_metrics: ScenarioMetrics,
    mixed_metrics: ScenarioMetrics,
) -> None:
    """Render side-by-side comparison table.

    Args:
        market_metrics: Metrics for market-rate scenario.
        mixed_metrics: Metrics for mixed-income scenario.
    """
    comparison_data = {
        "Metric": [
            "Total Development Cost",
            "TDC per Unit",
            "Equity Required",
            "Permanent Loan",
            "TIF Value",
            "",
            "Gross Potential Rent (Annual)",
            "Net Operating Income (Annual)",
            "Yield on Cost",
            "",
            "Levered IRR",
            "Unlevered IRR",
            "Equity Multiple",
        ],
        "Market Rate": [
            f"${market_metrics.tdc:,.0f}",
            f"${market_metrics.tdc_per_unit:,.0f}",
            f"${market_metrics.equity_required:,.0f}",
            f"${market_metrics.debt_amount:,.0f}",
            f"${market_metrics.tif_value:,.0f}",
            "",
            f"${market_metrics.gpr_annual:,.0f}",
            f"${market_metrics.noi_annual:,.0f}",
            f"{market_metrics.yield_on_cost:.2%}",
            "",
            f"{market_metrics.levered_irr:.2%}",
            f"{market_metrics.unlevered_irr:.2%}",
            f"{market_metrics.equity_multiple:.2f}x",
        ],
        "Mixed Income": [
            f"${mixed_metrics.tdc:,.0f}",
            f"${mixed_metrics.tdc_per_unit:,.0f}",
            f"${mixed_metrics.equity_required:,.0f}",
            f"${mixed_metrics.debt_amount:,.0f}",
            f"${mixed_metrics.tif_value:,.0f}",
            "",
            f"${mixed_metrics.gpr_annual:,.0f}",
            f"${mixed_metrics.noi_annual:,.0f}",
            f"{mixed_metrics.yield_on_cost:.2%}",
            "",
            f"{mixed_metrics.levered_irr:.2%}",
            f"{mixed_metrics.unlevered_irr:.2%}",
            f"{mixed_metrics.equity_multiple:.2f}x",
        ],
    }

    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_irr_callout(comparison: ScenarioComparison) -> None:
    """Render the IRR difference callout.

    Args:
        comparison: Scenario comparison result.
    """
    irr_diff = comparison.irr_difference_bps
    meets_target = comparison.meets_target

    if meets_target:
        st.success(f"""
        ### ✅ Target Met!
        **IRR Improvement: {irr_diff:+d} bps**

        The mixed-income scenario with selected incentives
        achieves at least +150 bps vs market rate.
        """)
    else:
        st.error(f"""
        ### ❌ Target Not Met
        **IRR Difference: {irr_diff:+d} bps**

        Need +150 bps improvement. Consider enabling
        additional incentives (TIF Lump Sum is most impactful).
        """)


def render_cash_flow_table(
    cash_flows: List[MonthlyCashFlow],
    show_all: bool = False,
) -> None:
    """Render cash flow table.

    Args:
        cash_flows: List of monthly cash flows.
        show_all: Whether to show all months or summary.
    """
    if show_all:
        data = []
        for cf in cash_flows:
            data.append({
                "Month": cf.month,
                "Phase": cf.phase.value,
                "GPR": f"${cf.gross_potential_rent:,.0f}",
                "EGI": f"${cf.effective_gross_income:,.0f}",
                "OpEx": f"${cf.operating_expenses:,.0f}",
                "NOI": f"${cf.noi:,.0f}",
                "Debt Service": f"${cf.perm_debt_service:,.0f}",
                "Cash Flow": f"${cf.levered_cf:,.0f}",
            })
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True, height=400)
    else:
        # Summary by phase
        from collections import defaultdict
        phase_totals = defaultdict(lambda: {"noi": 0, "cash_flow": 0, "months": 0})

        for cf in cash_flows:
            phase_totals[cf.phase.value]["noi"] += cf.noi
            phase_totals[cf.phase.value]["cash_flow"] += cf.levered_cf
            phase_totals[cf.phase.value]["months"] += 1

        summary_data = []
        for phase, totals in phase_totals.items():
            summary_data.append({
                "Phase": phase.title(),
                "Months": totals["months"],
                "Total NOI": f"${totals['noi']:,.0f}",
                "Total Cash Flow": f"${totals['cash_flow']:,.0f}",
            })

        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_key_metrics(
    metrics: ScenarioMetrics,
    label: str = "Scenario",
) -> None:
    """Render key metrics in a compact format.

    Args:
        metrics: Scenario metrics to display.
        label: Label for the scenario.
    """
    st.subheader(label)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("TDC", f"${metrics.tdc/1_000_000:.1f}M")

    with col2:
        st.metric("Equity", f"${metrics.equity_required/1_000_000:.1f}M")

    with col3:
        st.metric("Levered IRR", f"{metrics.levered_irr:.1%}")

    with col4:
        st.metric("Yield on Cost", f"{metrics.yield_on_cost:.2%}")
