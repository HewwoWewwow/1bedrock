"""Spreadsheet-style debug view for detailed cash flow verification.

This component displays every calculation row period-by-period in a format
that can be directly compared with the Excel model for validation.
"""

import streamlit as st
import pandas as pd
from typing import List, Optional
from dataclasses import fields

from src.calculations.detailed_cashflow import (
    DetailedCashFlowResult, DetailedPeriodCashFlow,
)
from ui.components.calculation_trace_view import render_calculation_trace_view
from ui.components.dependency_graph_view import render_dependency_graph_view
from src.export.audit_report import generate_audit_excel, AuditReportConfig


def _format_dollar(value: float) -> str:
    """Format as dollar amount."""
    if abs(value) >= 1_000_000:
        return f"${value/1_000_000:,.2f}M"
    elif abs(value) >= 1_000:
        return f"${value/1_000:,.1f}K"
    elif value == 0:
        return "-"
    else:
        return f"${value:,.0f}"


def _format_pct(value: float) -> str:
    """Format as percentage."""
    if value == 0:
        return "-"
    return f"{value:.2%}"


def _format_number(value: float, decimals: int = 0) -> str:
    """Format as plain number."""
    if value == 0:
        return "-"
    if decimals == 0:
        return f"{value:,.0f}"
    return f"{value:,.{decimals}f}"


def _get_phase_label(period: DetailedPeriodCashFlow) -> str:
    """Get phase label for a period."""
    h = period.header
    if h.is_reversion:
        return "REVERSION"
    elif h.is_operations:
        return "OPS"
    elif h.is_leaseup:
        return "LEASEUP"
    elif h.is_construction:
        return "CONSTR"
    elif h.is_predevelopment:
        return "PREDEV"
    return "-"


def create_spreadsheet_dataframe(result: DetailedCashFlowResult) -> pd.DataFrame:
    """Create a spreadsheet-style DataFrame with all calculations.

    Rows are calculation items, columns are periods.
    This matches typical Excel cash flow model layout.
    """
    periods = result.periods
    n_periods = len(periods)

    # Build column headers (period numbers)
    columns = [f"P{p.header.period}" for p in periods]

    # Initialize data dictionary - each key becomes a row
    data = {}

    # === HEADER INFO ===
    data["Period"] = [p.header.period for p in periods]
    data["Phase"] = [_get_phase_label(p) for p in periods]
    data["Date From"] = [p.header.date_from.strftime("%Y-%m") for p in periods]

    # === ESCALATION SECTION ===
    data["--- ESCALATION ---"] = ["-"] * n_periods
    data["Escalation Year"] = [p.escalation.active_count for p in periods]
    data["Market Rent Bump (cum)"] = [p.escalation.market_revenue_bump for p in periods]
    data["Affordable Rent Bump (cum)"] = [p.escalation.affordable_revenue_bump for p in periods]
    data["OpEx Bump (cum)"] = [p.escalation.opex_bump for p in periods]
    data["Prop Tax Bump (cum)"] = [p.escalation.prop_tax_bump for p in periods]

    # === DEVELOPMENT SECTION ===
    data["--- DEVELOPMENT ---"] = ["-"] * n_periods
    data["TDC Total"] = [p.development.tdc_total for p in periods]
    data["TDC BOP"] = [p.development.tdc_bop for p in periods]
    data["Draw % Total"] = [p.development.draw_pct_total for p in periods]
    data["Draw % Predev"] = [p.development.draw_pct_predev for p in periods]
    data["Draw % Construction"] = [p.development.draw_pct_construction for p in periods]
    data["Draw $ Total"] = [p.development.draw_dollars_total for p in periods]
    data["Draw $ Predev"] = [p.development.draw_dollars_predev for p in periods]
    data["Draw $ Construction"] = [p.development.draw_dollars_construction for p in periods]
    data["TDC EOP"] = [p.development.tdc_eop for p in periods]
    data["TDC To Be Funded"] = [p.development.tdc_to_be_funded for p in periods]

    # === EQUITY SECTION ===
    data["--- EQUITY ---"] = ["-"] * n_periods
    data["Equity BOP"] = [p.equity.equity_bop for p in periods]
    data["Equity Drawn"] = [p.equity.equity_drawn for p in periods]
    data["Equity EOP"] = [p.equity.equity_eop for p in periods]

    # === DEBT SOURCES SECTION ===
    data["--- DEBT SOURCES ---"] = ["-"] * n_periods
    data["Const Debt BOP"] = [p.debt_source.debt_bop for p in periods]
    data["Const Debt To Finance"] = [p.debt_source.to_be_financed for p in periods]
    data["Const Debt EOP"] = [p.debt_source.debt_eop for p in periods]

    # === GPR SECTION ===
    data["--- GPR ---"] = ["-"] * n_periods
    data["GPR All Market"] = [p.gpr.gpr_all_market for p in periods]
    data["GPR Mixed - Market Portion"] = [p.gpr.gpr_mixed_market for p in periods]
    data["GPR Mixed - Affordable Portion"] = [p.gpr.gpr_mixed_affordable for p in periods]
    data["GPR Total (Blended)"] = [p.gpr.gpr_total for p in periods]

    # === OPEX SECTION ===
    data["--- OPEX ---"] = ["-"] * n_periods
    data["OpEx ex Prop Taxes"] = [p.opex.opex_ex_prop_taxes for p in periods]
    data["Property Taxes"] = [p.opex.property_taxes for p in periods]
    data["Total OpEx"] = [p.opex.total_opex for p in periods]

    # === OPERATIONS / NOI BUILDUP ===
    data["--- NOI BUILDUP ---"] = ["-"] * n_periods
    data["Lease-up %"] = [p.operations.leaseup_pct for p in periods]
    data["GPR (Operations)"] = [p.operations.gpr for p in periods]
    data["Vacancy Rate"] = [p.operations.vacancy_rate for p in periods]
    data["Less: Vacancy"] = [p.operations.less_vacancy for p in periods]
    data["EGI"] = [p.operations.egi for p in periods]
    data["Less: Management Fee"] = [p.operations.less_management_fee for p in periods]
    data["Less: OpEx ex Taxes"] = [p.operations.less_opex_ex_taxes for p in periods]
    data["Less: Prop Taxes"] = [p.operations.less_property_taxes for p in periods]
    data["NOI"] = [p.operations.noi for p in periods]
    data["NOI Reserve in TDC"] = [p.operations.noi_reserve_in_tdc for p in periods]

    # TIF data from operations
    data["--- TIF ---"] = ["-"] * n_periods
    data["Gross Prop Taxes (pre-TIF)"] = [
        p.operations.tif.gross_property_taxes if p.operations.tif else 0
        for p in periods
    ]
    data["TIF Abatement"] = [
        p.operations.tif.abatement_amount if p.operations.tif else 0
        for p in periods
    ]
    data["Net Prop Taxes (post-abate)"] = [
        p.operations.tif.net_property_taxes if p.operations.tif else 0
        for p in periods
    ]
    data["TIF Reimbursement"] = [
        p.operations.tif.tif_reimbursement if p.operations.tif else 0
        for p in periods
    ]
    data["Net TIF Benefit"] = [
        p.operations.tif.net_tif_benefit if p.operations.tif else 0
        for p in periods
    ]
    data["TIF Reimb (in NOI)"] = [p.operations.tif_reimbursement for p in periods]

    # === INVESTMENT / UNLEVERED CF ===
    data["--- UNLEVERED CF ---"] = ["-"] * n_periods
    data["Dev Cost (outflow)"] = [p.investment.dev_cost for p in periods]
    data["Reserves"] = [p.investment.reserves for p in periods]
    data["Reversion"] = [p.investment.reversion for p in periods]
    data["Unlevered CF"] = [p.investment.unlevered_cf for p in periods]

    # === CONSTRUCTION DEBT ===
    data["--- CONSTRUCTION DEBT ---"] = ["-"] * n_periods
    data["Const Loan Principal BOP"] = [p.construction_debt.principal_bop for p in periods]
    data["Const Loan Debt Added"] = [p.construction_debt.debt_added for p in periods]
    data["Const Loan Interest"] = [p.construction_debt.interest_in_period for p in periods]
    data["Const Loan Repaid"] = [p.construction_debt.repaid for p in periods]
    data["Const Loan Principal EOP"] = [p.construction_debt.principal_eop for p in periods]
    data["Const Loan Net CF"] = [p.construction_debt.net_cf for p in periods]

    # === PERMANENT DEBT ===
    data["--- PERMANENT DEBT ---"] = ["-"] * n_periods
    data["Perm Loan Principal BOP"] = [p.permanent_debt.principal_bop for p in periods]
    data["Perm Loan Payment"] = [p.permanent_debt.pmt_in_period for p in periods]
    data["Perm Loan Interest Pmt"] = [p.permanent_debt.interest_pmt for p in periods]
    data["Perm Loan Principal Pmt"] = [p.permanent_debt.principal_pmt for p in periods]
    data["Perm Loan Payoff"] = [p.permanent_debt.payoff for p in periods]
    data["Perm Loan Principal EOP"] = [p.permanent_debt.principal_eop for p in periods]
    data["Perm Loan Net CF"] = [p.permanent_debt.net_cf for p in periods]

    # === MEZZANINE DEBT ===
    data["--- MEZZANINE DEBT ---"] = ["-"] * n_periods
    data["Mezz Active"] = [
        "Yes" if (p.mezzanine_debt and p.mezzanine_debt.is_active) else "No"
        for p in periods
    ]
    data["Mezz Principal BOP"] = [
        p.mezzanine_debt.principal_bop if p.mezzanine_debt else 0
        for p in periods
    ]
    data["Mezz Interest Rate"] = [
        p.mezzanine_debt.interest_rate if p.mezzanine_debt else 0
        for p in periods
    ]
    data["Mezz Payment"] = [
        p.mezzanine_debt.pmt_in_period if p.mezzanine_debt else 0
        for p in periods
    ]
    data["Mezz Interest Pmt"] = [
        p.mezzanine_debt.interest_pmt if p.mezzanine_debt else 0
        for p in periods
    ]
    data["Mezz Principal Pmt"] = [
        p.mezzanine_debt.principal_pmt if p.mezzanine_debt else 0
        for p in periods
    ]
    data["Mezz Payoff"] = [
        p.mezzanine_debt.payoff if p.mezzanine_debt else 0
        for p in periods
    ]
    data["Mezz Principal EOP"] = [
        p.mezzanine_debt.principal_eop if p.mezzanine_debt else 0
        for p in periods
    ]
    data["Mezz Net CF"] = [
        p.mezzanine_debt.net_cf if p.mezzanine_debt else 0
        for p in periods
    ]

    # === PREFERRED EQUITY ===
    data["--- PREFERRED EQUITY ---"] = ["-"] * n_periods
    data["Pref Active"] = [
        "Yes" if (p.preferred_equity and p.preferred_equity.is_active) else "No"
        for p in periods
    ]
    data["Pref Balance BOP"] = [
        p.preferred_equity.balance_bop if p.preferred_equity else 0
        for p in periods
    ]
    data["Pref Return Rate"] = [
        p.preferred_equity.preferred_return_rate if p.preferred_equity else 0
        for p in periods
    ]
    data["Pref Accrued Return"] = [
        p.preferred_equity.accrued_return if p.preferred_equity else 0
        for p in periods
    ]
    data["Pref Paid Return"] = [
        p.preferred_equity.paid_return if p.preferred_equity else 0
        for p in periods
    ]
    data["Pref Payoff"] = [
        p.preferred_equity.payoff if p.preferred_equity else 0
        for p in periods
    ]
    data["Pref Balance EOP"] = [
        p.preferred_equity.balance_eop if p.preferred_equity else 0
        for p in periods
    ]
    data["Pref Net CF"] = [
        p.preferred_equity.net_cf if p.preferred_equity else 0
        for p in periods
    ]

    # === FINAL CASH FLOWS ===
    data["--- FINAL CASH FLOWS ---"] = ["-"] * n_periods
    data["Net Senior Debt CF"] = [p.net_senior_debt_cf for p in periods]
    data["Net Mezz Debt CF"] = [p.net_mezz_debt_cf for p in periods]
    data["Net Preferred CF"] = [p.net_preferred_cf for p in periods]
    data["Net Debt CF (Total)"] = [p.net_debt_cf for p in periods]
    data["LEVERED CF"] = [p.levered_cf for p in periods]

    # Create DataFrame with row labels as index
    df = pd.DataFrame(data)
    df = df.T  # Transpose so periods are columns
    df.columns = columns

    return df


def render_spreadsheet_debug_view(
    result: DetailedCashFlowResult,
    scenario_name: str = "Scenario",
    show_all_periods: bool = False,
) -> None:
    """Render the spreadsheet-style debug view.

    Args:
        result: DetailedCashFlowResult from calculate_deal()
        scenario_name: Name to display
        show_all_periods: If False, allow user to select period range
    """
    st.subheader(f"Spreadsheet Debug View: {scenario_name}")

    st.info(
        "This view shows every calculation row period-by-period, matching the Excel model structure. "
        "Use this to verify calculations by comparing directly with your Excel model."
    )

    # Summary metrics at top
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("TDC", f"${result.sources_uses.tdc:,.0f}")
    with col2:
        st.metric("Equity", f"${result.sources_uses.equity:,.0f}")
    with col3:
        st.metric("Const Loan", f"${result.sources_uses.construction_loan:,.0f}")
    with col4:
        st.metric("Total Periods", result.total_periods)
    with col5:
        st.metric("Levered IRR", f"{result.levered_irr:.2%}")
    with col6:
        st.metric("Unlevered IRR", f"{result.unlevered_irr:.2%}")

    st.divider()

    # Period range selector
    if not show_all_periods:
        # Callback functions for quick select buttons
        # These run BEFORE widgets are created on the next run, avoiding the session state error
        def set_range(start: int, end: int, scenario: str):
            """Set period range in session state. Called by button callbacks."""
            st.session_state[f"debug_start_{scenario}"] = start
            st.session_state[f"debug_end_{scenario}"] = end

        # Initialize session state if not present (before widgets are created)
        start_key = f"debug_start_{scenario_name}"
        end_key = f"debug_end_{scenario_name}"
        if start_key not in st.session_state:
            st.session_state[start_key] = 1
        if end_key not in st.session_state:
            st.session_state[end_key] = min(24, result.total_periods)

        col1, col2 = st.columns(2)
        with col1:
            start_period = st.number_input(
                "Start Period",
                min_value=1,
                max_value=result.total_periods,
                key=start_key
            )
        with col2:
            end_period = st.number_input(
                "End Period",
                min_value=1,
                max_value=result.total_periods,
                key=end_key
            )

        # Quick select buttons using on_click callbacks
        # The callback runs BEFORE the page rerenders, so session state can be modified
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.button(
                "Predev",
                key=f"btn_predev_{scenario_name}",
                on_click=set_range,
                args=(1, result.predevelopment_end, scenario_name)
            )
        with col2:
            st.button(
                "Construction",
                key=f"btn_constr_{scenario_name}",
                on_click=set_range,
                args=(result.predevelopment_end + 1, result.construction_end, scenario_name)
            )
        with col3:
            st.button(
                "Lease-up",
                key=f"btn_leaseup_{scenario_name}",
                on_click=set_range,
                args=(result.construction_end + 1, result.leaseup_end, scenario_name)
            )
        with col4:
            st.button(
                "Operations",
                key=f"btn_ops_{scenario_name}",
                on_click=set_range,
                args=(result.leaseup_end + 1, result.total_periods, scenario_name)
            )
    else:
        start_period = 1
        end_period = result.total_periods

    # Create the full dataframe
    df = create_spreadsheet_dataframe(result)

    # Filter to selected period range
    period_cols = [f"P{i}" for i in range(start_period, end_period + 1)]
    available_cols = [c for c in period_cols if c in df.columns]
    df_filtered = df[available_cols]

    # Display phase boundaries
    st.caption(
        f"Phase boundaries: "
        f"Predev ends P{result.predevelopment_end}, "
        f"Construction ends P{result.construction_end}, "
        f"Lease-up ends P{result.leaseup_end}, "
        f"Operations ends P{result.operations_end}"
    )

    # Section selector for easier navigation
    sections = [
        "All Sections",
        "ESCALATION",
        "DEVELOPMENT",
        "EQUITY",
        "DEBT SOURCES",
        "GPR",
        "OPEX",
        "NOI BUILDUP",
        "TIF",
        "UNLEVERED CF",
        "CONSTRUCTION DEBT",
        "PERMANENT DEBT",
        "MEZZANINE DEBT",
        "PREFERRED EQUITY",
        "FINAL CASH FLOWS",
    ]

    selected_section = st.selectbox(
        "Jump to Section",
        sections,
        key=f"section_select_{scenario_name}"
    )

    # Filter rows if a specific section is selected
    if selected_section != "All Sections":
        # Find rows in the selected section
        section_marker = f"--- {selected_section} ---"
        in_section = False
        rows_to_show = []

        for idx in df_filtered.index:
            if idx == section_marker:
                in_section = True
                rows_to_show.append(idx)
            elif idx.startswith("---") and in_section:
                # Next section started
                break
            elif in_section:
                rows_to_show.append(idx)

        df_display = df_filtered.loc[rows_to_show]
    else:
        df_display = df_filtered

    # Format options
    show_raw = st.checkbox(
        "Show raw numbers (no formatting)",
        value=False,
        key=f"raw_numbers_{scenario_name}"
    )

    if not show_raw:
        # Apply formatting based on row type
        def format_cell(val, row_name: str):
            if isinstance(val, str):
                return val
            if val is None:
                return "-"

            # Determine format based on row name
            row_lower = row_name.lower()

            if "%" in row_name or "rate" in row_lower or "pct" in row_lower or "bump" in row_lower or "vacancy" in row_lower:
                return f"{val:.2%}" if val != 0 else "-"
            elif row_name.startswith("---"):
                return val
            elif "year" in row_lower or "period" in row_lower or "month" in row_lower:
                return f"{val:.0f}" if val != 0 else "-"
            elif "active" in row_lower:
                return val
            elif row_name in ["Phase", "Date From"]:
                return val
            else:
                # Dollar amounts
                if val == 0:
                    return "-"
                elif abs(val) >= 1_000_000:
                    return f"${val:,.0f}"
                else:
                    return f"${val:,.0f}"

        # Apply formatting
        df_formatted = df_display.copy()
        for idx in df_formatted.index:
            for col in df_formatted.columns:
                df_formatted.loc[idx, col] = format_cell(df_display.loc[idx, col], idx)

        st.dataframe(
            df_formatted,
            use_container_width=True,
            height=800,
        )
    else:
        st.dataframe(
            df_display,
            use_container_width=True,
            height=800,
        )

    # Export option
    st.divider()

    # Export to CSV
    csv_data = df.to_csv()
    st.download_button(
        label="Download Full Spreadsheet as CSV",
        data=csv_data,
        file_name=f"cashflow_debug_{scenario_name.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        key=f"download_csv_{scenario_name}"
    )


def render_irr_debug_view(result: DetailedCashFlowResult, scenario_name: str = "Scenario") -> None:
    """Render IRR calculation debug view showing the cash flow series used for IRR.

    This helps verify that the correct cash flows are being used in IRR calculation.
    """
    st.subheader(f"IRR Calculation Debug: {scenario_name}")

    # Show the raw cash flow series
    periods = result.periods

    data = {
        "Period": [p.header.period for p in periods],
        "Phase": [_get_phase_label(p) for p in periods],
        "Unlevered CF": [p.investment.unlevered_cf for p in periods],
        "Levered CF": [p.levered_cf for p in periods],
        "Equity Drawn": [p.equity.equity_drawn for p in periods],
        "NOI": [p.operations.noi for p in periods],
        "Reversion": [p.investment.reversion for p in periods],
        "Dev Cost": [p.investment.dev_cost for p in periods],
    }

    df = pd.DataFrame(data)

    # Summary
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Calculated Levered IRR", f"{result.levered_irr:.4%}")
        st.metric("Sum of Levered CFs", f"${sum(p.levered_cf for p in periods):,.0f}")
    with col2:
        st.metric("Calculated Unlevered IRR", f"{result.unlevered_irr:.4%}")
        st.metric("Sum of Unlevered CFs", f"${sum(p.investment.unlevered_cf for p in periods):,.0f}")

    # Totals
    st.caption("Totals:")
    totals = {
        "Total Equity Drawn": sum(p.equity.equity_drawn for p in periods),
        "Total NOI": sum(p.operations.noi for p in periods),
        "Total Reversion": sum(p.investment.reversion for p in periods),
        "Total Dev Cost": sum(p.investment.dev_cost for p in periods),
    }

    for label, value in totals.items():
        st.write(f"- {label}: ${value:,.0f}")

    st.divider()

    # Show the DataFrame
    st.dataframe(df, use_container_width=True, height=400)

    # Show levered CF array (what goes into npf.irr)
    with st.expander("Raw Levered CF Array (for IRR calculation)"):
        levered_cfs = [p.levered_cf for p in periods]

        # Note: The actual IRR calculation may adjust period 0 for TIF lump sum
        # and exclude land/IDC from unlevered. The stored result.levered_irr
        # already includes these adjustments.
        st.caption("⚠️ Note: Period levered_cf values shown below. Actual IRR may differ if TIF lump sum was applied (added to period 1).")

        st.code(f"levered_cfs = {[f'{cf:,.0f}' for cf in levered_cfs[:10]]} ... (first 10)")

        # Manual IRR verification (without adjustments - may differ from result.levered_irr)
        try:
            import numpy_financial as npf
            monthly_irr = npf.irr(levered_cfs)
            annual_irr = (1 + monthly_irr) ** 12 - 1
            st.write(f"IRR from per-period levered_cf: {annual_irr:.4%}")
            st.write(f"Stored result.levered_irr: {result.levered_irr:.4%}")
            if abs(annual_irr - result.levered_irr) > 0.001:
                st.warning("⚠️ Difference suggests TIF lump sum or other adjustment was applied during IRR calculation.")
        except Exception as e:
            st.error(f"IRR calculation error: {e}")


def render_sources_uses_debug(result: DetailedCashFlowResult) -> None:
    """Render Sources & Uses debug view."""
    st.subheader("Sources & Uses Debug")

    su = result.sources_uses

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Uses")
        uses_data = {
            "Item": ["Land", "Hard Costs", "Soft Costs", "IDC", "TDC (Total)"],
            "Amount": [
                f"${su.land:,.0f}",
                f"${su.hard_costs:,.0f}",
                f"${su.soft_costs:,.0f}",
                f"${su.idc:,.0f}",
                f"${su.tdc:,.0f}"
            ],
            "% of TDC": [
                f"{su.land/su.tdc:.1%}" if su.tdc > 0 else "-",
                f"{su.hard_costs/su.tdc:.1%}" if su.tdc > 0 else "-",
                f"{su.soft_costs/su.tdc:.1%}" if su.tdc > 0 else "-",
                f"{su.idc/su.tdc:.1%}" if su.tdc > 0 else "-",
                "100.0%"
            ],
        }
        st.dataframe(pd.DataFrame(uses_data), use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### Sources")
        sources_data = {
            "Item": ["Equity", "Construction Loan", "Total Sources"],
            "Amount": [
                f"${su.equity:,.0f}",
                f"${su.construction_loan:,.0f}",
                f"${su.total_sources:,.0f}"
            ],
            "% of TDC": [
                f"{su.equity_pct:.1%}",
                f"{su.ltc:.1%}",
                "100.0%"
            ],
        }
        st.dataframe(pd.DataFrame(sources_data), use_container_width=True, hide_index=True)

    # Verify balance
    diff = abs(su.tdc - su.total_sources)
    if diff < 1:
        st.success(f"Sources = Uses (balanced)")
    else:
        st.error(f"Imbalance: ${diff:,.0f}")

    st.divider()

    # Draw Schedule Debug
    st.markdown("#### Draw Schedule")
    st.caption("Shows how TDC is drawn down over development period")

    ds = result.draw_schedule
    if ds:
        draw_data = []
        for period_draw in ds.periods:
            draw_data.append({
                "Period": period_draw.period,
                "Phase": period_draw.phase.value,
                "TDC Draw %": f"{period_draw.tdc_draw_pct:.2%}",
                "TDC Draw $": f"${period_draw.tdc_draw_total:,.0f}",
                "Equity Draw": f"${period_draw.equity_draw:,.0f}",
                "Debt Draw": f"${period_draw.const_debt_draw:,.0f}",
                "TDC EOP": f"${period_draw.tdc_eop:,.0f}",
            })
        df_draws = pd.DataFrame(draw_data)
        st.dataframe(df_draws, use_container_width=True, height=300, hide_index=True)

        # Totals
        st.caption("Draw Totals:")
        total_equity = sum(p.equity_draw for p in ds.periods)
        total_debt = sum(p.const_debt_draw for p in ds.periods)
        total_tdc = sum(p.tdc_draw_total for p in ds.periods)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Equity Drawn", f"${total_equity:,.0f}")
        with col2:
            st.metric("Total Debt Drawn", f"${total_debt:,.0f}")
        with col3:
            st.metric("Total TDC Drawn", f"${total_tdc:,.0f}")


def render_key_inputs_debug() -> None:
    """Render key inputs from session state for debugging.

    Shows both Market Rate and Mixed Income values side-by-side
    so users can verify each scenario is using the correct inputs.
    """
    st.subheader("Key Inputs Comparison: Market Rate vs Mixed Income")

    st.caption(
        "These are the actual session state values used for each scenario. "
        "Market Rate uses non-prefixed keys, Mixed Income uses `mixed_` prefixed keys."
    )

    def get_market(key: str, default=0):
        """Get market rate value (non-prefixed key)."""
        return st.session_state.get(key, default)

    def get_mixed(key: str, default=0):
        """Get mixed income value (mixed_ prefixed key, falls back to non-prefixed)."""
        return st.session_state.get(f"mixed_{key}", st.session_state.get(key, default))

    def fmt_pct(val) -> str:
        """Format value as percentage (values stored as 5.0 meaning 5%)."""
        if val is None or val == "N/A":
            return "N/A"
        return f"{val:.1f}%"

    def fmt_dollar(val) -> str:
        """Format value as dollar amount."""
        if val is None or val == "N/A":
            return "N/A"
        return f"${val:,.0f}"

    def fmt_decimal_pct(val) -> str:
        """Format decimal value as percentage (values stored as 0.05 meaning 5%)."""
        if val is None or val == "N/A":
            return "N/A"
        return f"{val:.1%}"

    # Create comparison table
    st.markdown("##### Project & Timing (Shared)")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"- Target Units: **{get_market('target_units', 200)}**")
        st.write(f"- Predevelopment Months: **{get_market('predevelopment_months', 18)}**")
        st.write(f"- Construction Months: **{get_market('construction_months', 24)}**")
        st.write(f"- Lease-up Months: **{get_market('leaseup_months', 12)}**")
        st.write(f"- Operations Months: **{get_market('operations_months', 60)}**")

    st.divider()

    # Scenario-specific inputs comparison
    st.markdown("##### Cost Inputs (Per-Scenario)")
    st.markdown("| Input | Market Rate | Mixed Income |")
    st.markdown("|-------|-------------|--------------|")
    cost_keys = [
        ("land_cost", "Land Cost", fmt_dollar),
        ("hard_cost_per_unit", "Hard Cost/Unit", fmt_dollar),
        ("soft_cost_pct", "Soft Cost %", fmt_pct),
        ("hard_cost_contingency_pct", "Hard Cost Contingency %", fmt_pct),
        ("soft_cost_contingency_pct", "Soft Cost Contingency %", fmt_pct),
        ("predevelopment_cost_pct", "Predevelopment Cost %", fmt_pct),
        ("developer_fee_pct", "Developer Fee %", fmt_pct),
    ]
    for key, label, fmt in cost_keys:
        market_val = fmt(get_market(key, 0))
        mixed_val = fmt(get_mixed(key, 0))
        # Highlight differences
        diff_marker = " **" if market_val != mixed_val else ""
        st.markdown(f"| {label} | {market_val} | {mixed_val}{diff_marker} |")

    st.markdown("##### Operating Inputs (Per-Scenario)")
    st.markdown("| Input | Market Rate | Mixed Income |")
    st.markdown("|-------|-------------|--------------|")
    opex_keys = [
        ("vacancy_rate_pct", "Vacancy Rate %", fmt_pct),
        ("leaseup_pace_pct", "Lease-up Pace %/mo", fmt_pct),
        ("opex_utilities", "Utilities/Unit/Yr", fmt_dollar),
        ("opex_maintenance", "Maintenance/Unit/Yr", fmt_dollar),
        ("opex_management_pct", "Management Fee %", fmt_pct),
        ("market_rent_growth_pct", "Market Rent Growth %", fmt_pct),
        ("opex_growth_pct", "OpEx Growth %", fmt_pct),
        ("property_tax_growth_pct", "Property Tax Growth %", fmt_pct),
    ]
    for key, label, fmt in opex_keys:
        market_val = fmt(get_market(key, 0))
        mixed_val = fmt(get_mixed(key, 0))
        diff_marker = " **" if market_val != mixed_val else ""
        st.markdown(f"| {label} | {market_val} | {mixed_val}{diff_marker} |")

    st.markdown("##### Financing Inputs (Per-Scenario)")
    st.markdown("| Input | Market Rate | Mixed Income |")
    st.markdown("|-------|-------------|--------------|")
    financing_keys = [
        ("construction_rate_pct", "Construction Rate %", fmt_pct),
        ("construction_ltc_pct", "Construction LTC %", fmt_pct),
        ("perm_rate_pct", "Perm Rate %", fmt_pct),
        ("perm_amort_years", "Perm Amort Years", lambda x: str(x)),
        ("perm_ltv_max_pct", "Perm LTV Max %", fmt_pct),
        ("perm_dscr_min", "Perm DSCR Min", lambda x: f"{x:.2f}x"),
        ("exit_cap_rate_pct", "Exit Cap Rate %", fmt_pct),
    ]
    for key, label, fmt in financing_keys:
        market_val = fmt(get_market(key, 0))
        mixed_val = fmt(get_mixed(key, 0))
        diff_marker = " **" if market_val != mixed_val else ""
        st.markdown(f"| {label} | {market_val} | {mixed_val}{diff_marker} |")

    st.markdown("##### Scenario-Specific Settings")
    st.markdown("| Setting | Market Rate | Mixed Income |")
    st.markdown("|---------|-------------|--------------|")
    affordable_pct = get_market('affordable_pct', 20.0)
    st.markdown(f"| Affordable % | **0%** (always) | **{fmt_pct(affordable_pct)}** |")
    st.markdown(f"| AMI Level | N/A | **{get_market('ami_level', '50%')}** |")

    st.info(
        "Values marked with ** indicate a difference between scenarios. "
        "If all values are the same except Affordable %, the mixed income inputs may not have been customized yet."
    )


def render_full_debug_page(
    market_result: Optional[DetailedCashFlowResult],
    mixed_result: Optional[DetailedCashFlowResult],
) -> None:
    """Render the full debug page with tabs for each scenario."""

    st.header("Spreadsheet Debug View")
    st.markdown(
        "Use this page to compare Python model calculations with your Excel model line-by-line. "
        "Every calculation row is displayed period-by-period."
    )

    if market_result is None and mixed_result is None:
        st.warning("No results available. Run analysis first.")
        return

    # Create tabs for different debug views
    debug_tabs = st.tabs([
        "Key Inputs",
        "Market Rate Spreadsheet",
        "Mixed Income Spreadsheet",
        "Calculation Traces",
        "Dependency Graph",
        "Audit Export",
        "IRR Debug",
        "Sources & Uses Debug",
    ])

    with debug_tabs[0]:
        render_key_inputs_debug()

    with debug_tabs[1]:
        if market_result:
            render_spreadsheet_debug_view(market_result, "Market Rate")
        else:
            st.info("Market rate results not available")

    with debug_tabs[2]:
        if mixed_result:
            render_spreadsheet_debug_view(mixed_result, "Mixed Income")
        else:
            st.info("Mixed income results not available")

    with debug_tabs[3]:
        st.subheader("Calculation Traces")
        st.markdown(
            "View formulas and traced values for each calculation. "
            "This shows exactly how each value was computed."
        )

        trace_scenario = st.radio(
            "Select Scenario",
            ["Market Rate", "Mixed Income"],
            horizontal=True,
            key="trace_scenario"
        )

        if trace_scenario == "Market Rate" and market_result:
            render_calculation_trace_view(market_result.trace_context)
        elif trace_scenario == "Mixed Income" and mixed_result:
            render_calculation_trace_view(mixed_result.trace_context)
        else:
            st.info("Results not available for selected scenario")

    with debug_tabs[4]:
        render_dependency_graph_view()

    with debug_tabs[5]:
        st.subheader("Audit Export")
        st.markdown(
            "Generate a comprehensive Excel report documenting all calculations, "
            "formulas, and traced values for audit and review purposes."
        )

        export_scenario = st.radio(
            "Select Scenario",
            ["Market Rate", "Mixed Income"],
            horizontal=True,
            key="audit_export_scenario"
        )

        result_to_export = market_result if export_scenario == "Market Rate" else mixed_result

        if result_to_export:
            col1, col2 = st.columns(2)

            with col1:
                project_name = st.text_input(
                    "Project Name",
                    value="Real Estate Development",
                    key="audit_project_name"
                )

            with col2:
                scenario_label = st.text_input(
                    "Scenario Label",
                    value=export_scenario,
                    key="audit_scenario_label"
                )

            st.markdown("**Include in Report:**")
            col1, col2 = st.columns(2)
            with col1:
                include_summary = st.checkbox("Summary", value=True, key="audit_inc_summary")
                include_sources_uses = st.checkbox("Sources & Uses", value=True, key="audit_inc_su")
                include_formulas = st.checkbox("Formula Registry", value=True, key="audit_inc_formulas")
            with col2:
                include_traces = st.checkbox("Traced Calculations", value=True, key="audit_inc_traces")
                include_cashflows = st.checkbox("Cash Flows", value=True, key="audit_inc_cf")

            if st.button("Generate Audit Report", type="primary", key="generate_audit"):
                config = AuditReportConfig(
                    project_name=project_name,
                    scenario_name=scenario_label,
                    include_summary=include_summary,
                    include_sources_uses=include_sources_uses,
                    include_formula_registry=include_formulas,
                    include_traced_values=include_traces,
                    include_cash_flows=include_cashflows,
                )

                excel_bytes = generate_audit_excel(result_to_export, config)

                st.download_button(
                    label="Download Excel Report",
                    data=excel_bytes,
                    file_name=f"audit_report_{scenario_label.lower().replace(' ', '_')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_audit_excel"
                )

                st.success("Audit report generated successfully!")
        else:
            st.info("No results available for selected scenario")

    with debug_tabs[6]:
        st.subheader("IRR Calculation Verification")

        irr_scenario = st.radio(
            "Select Scenario",
            ["Market Rate", "Mixed Income"],
            horizontal=True,
            key="irr_debug_scenario"
        )

        if irr_scenario == "Market Rate" and market_result:
            render_irr_debug_view(market_result, "Market Rate")
        elif irr_scenario == "Mixed Income" and mixed_result:
            render_irr_debug_view(mixed_result, "Mixed Income")
        else:
            st.info("Results not available for selected scenario")

    with debug_tabs[7]:
        st.subheader("Sources & Uses Verification")

        su_scenario = st.radio(
            "Select Scenario",
            ["Market Rate", "Mixed Income"],
            horizontal=True,
            key="su_debug_scenario"
        )

        if su_scenario == "Market Rate" and market_result:
            render_sources_uses_debug(market_result)
        elif su_scenario == "Mixed Income" and mixed_result:
            render_sources_uses_debug(mixed_result)
        else:
            st.info("Results not available for selected scenario")
