"""Chart components for the Streamlit UI."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import List

from src.calculations.dcf import DCFResult, MonthlyCashFlow


def render_cash_flow_chart(
    market_result: DCFResult,
    mixed_result: DCFResult,
) -> None:
    """Render cash flow comparison chart.

    Args:
        market_result: Market scenario DCF result.
        mixed_result: Mixed-income scenario DCF result.
    """
    market_cfs = [cf.levered_cf for cf in market_result.monthly_cash_flows]
    mixed_cfs = [cf.levered_cf for cf in mixed_result.monthly_cash_flows]
    months = list(range(1, len(market_cfs) + 1))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=months,
        y=market_cfs,
        mode='lines',
        name='Market Rate',
        line=dict(color='#1f77b4', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=months,
        y=mixed_cfs,
        mode='lines',
        name='Mixed Income',
        line=dict(color='#ff7f0e', width=2)
    ))

    fig.update_layout(
        title="Levered Cash Flows by Month",
        xaxis_title="Month",
        yaxis_title="Cash Flow ($)",
        yaxis_tickformat="$,.0f",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        height=400,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_cost_breakdown_chart(
    land_cost: float,
    hard_costs: float,
    soft_costs: float,
    idc: float,
) -> None:
    """Render cost breakdown pie chart.

    Args:
        land_cost: Land cost.
        hard_costs: Hard construction costs.
        soft_costs: Soft costs.
        idc: Interest during construction.
    """
    labels = ['Land', 'Hard Costs', 'Soft Costs', 'IDC']
    values = [land_cost, hard_costs, soft_costs, idc]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        textinfo='label+percent',
        textposition='outside',
        marker=dict(colors=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'])
    )])

    fig.update_layout(
        title="Cost Breakdown",
        height=350,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )

    st.plotly_chart(fig, use_container_width=True)


def render_noi_chart(
    cash_flows: List[MonthlyCashFlow],
) -> None:
    """Render NOI over time chart.

    Args:
        cash_flows: List of monthly cash flows.
    """
    # Filter to operating periods
    ops_cfs = [cf for cf in cash_flows if cf.noi > 0]

    if not ops_cfs:
        st.info("No operating cash flows to display")
        return

    months = [cf.month for cf in ops_cfs]
    nois = [cf.noi for cf in ops_cfs]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=months,
        y=nois,
        name='NOI',
        marker_color='#2ca02c'
    ))

    fig.update_layout(
        title="Monthly NOI",
        xaxis_title="Month",
        yaxis_title="NOI ($)",
        yaxis_tickformat="$,.0f",
        height=300,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_scenario_matrix_chart(
    results: list,
) -> None:
    """Render scenario matrix results as a bar chart.

    Args:
        results: List of ScenarioMatrixResult objects.
    """
    if not results:
        st.info("No results to display")
        return

    # Prepare data
    scenarios = [r.combination.name for r in results[:15]]  # Top 15
    irr_diffs = [r.comparison.irr_difference_bps for r in results[:15]]
    colors = ['#28a745' if r.comparison.meets_target else '#dc3545' for r in results[:15]]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=scenarios,
        y=irr_diffs,
        marker_color=colors,
        text=[f"{d:+d}" for d in irr_diffs],
        textposition='outside',
    ))

    # Add target line
    fig.add_hline(
        y=150,
        line_dash="dash",
        line_color="green",
        annotation_text="+150 bps Target",
        annotation_position="right"
    )

    fig.update_layout(
        title="IRR Improvement by Scenario (bps)",
        xaxis_title="Scenario",
        yaxis_title="IRR Difference (bps)",
        height=400,
        xaxis_tickangle=-45,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_waterfall_chart(
    market_irr: float,
    incentive_impacts: dict,
    final_irr: float,
) -> None:
    """Render IRR waterfall chart showing incentive impacts.

    Args:
        market_irr: Base market IRR.
        incentive_impacts: Dictionary of incentive name to IRR impact.
        final_irr: Final mixed-income IRR.
    """
    # Build waterfall data
    measure = ['absolute']
    x = ['Market IRR']
    y = [market_irr * 100]

    for incentive, impact in incentive_impacts.items():
        measure.append('relative')
        x.append(incentive)
        y.append(impact * 100)

    measure.append('total')
    x.append('Mixed IRR')
    y.append(final_irr * 100)

    fig = go.Figure(go.Waterfall(
        name="IRR",
        orientation="v",
        measure=measure,
        x=x,
        y=y,
        textposition="outside",
        text=[f"{v:.1f}%" for v in y],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))

    fig.update_layout(
        title="IRR Waterfall",
        showlegend=False,
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)
