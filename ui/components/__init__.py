"""UI components for the Austin TIF Model."""

from .inputs import render_sidebar_inputs, render_project_inputs, render_incentive_inputs
from .results import render_comparison_table, render_irr_callout, render_cash_flow_table
from .charts import render_cash_flow_chart, render_cost_breakdown_chart
from .unit_mix import render_unit_mix_tab, get_unit_mix_from_session_state, get_efficiency
from .tif_analysis_view import (
    render_tax_rate_breakdown,
    render_tif_lump_sum_inputs,
    render_tif_lump_sum_result,
    render_tif_loan_schedule,
    render_smart_fee_waiver,
    render_city_tax_abatement,
    render_tif_analysis_tab,
)

__all__ = [
    "render_sidebar_inputs",
    "render_project_inputs",
    "render_incentive_inputs",
    "render_comparison_table",
    "render_irr_callout",
    "render_cash_flow_table",
    "render_cash_flow_chart",
    "render_cost_breakdown_chart",
    "render_unit_mix_tab",
    "get_unit_mix_from_session_state",
    "get_efficiency",
    # TIF Analysis
    "render_tax_rate_breakdown",
    "render_tif_lump_sum_inputs",
    "render_tif_lump_sum_result",
    "render_tif_loan_schedule",
    "render_smart_fee_waiver",
    "render_city_tax_abatement",
    "render_tif_analysis_tab",
]
