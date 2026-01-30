"""Detailed Sources & Uses UI component."""

import streamlit as st
import pandas as pd
from typing import Optional

from src.calculations.sources_uses_detailed import (
    SourcesUsesDetailed,
    UsesDetail,
    SourcesDetail,
    LandCostMethod,
    calculate_sources_uses_detailed,
)


def render_sources_uses_inputs() -> dict:
    """Render input controls for Sources & Uses and return values."""
    inputs = {}

    st.subheader("Sources & Uses Configuration")

    # === LAND SECTION ===
    st.markdown("#### Land & Acquisition")

    col1, col2 = st.columns(2)
    with col1:
        land_method = st.radio(
            "Land Cost Method",
            options=["Direct Input", "Calculate from $/Acre"],
            key="land_cost_method",
            horizontal=True,
        )
        inputs["land_cost_method"] = (
            LandCostMethod.DIRECT if land_method == "Direct Input"
            else LandCostMethod.PER_ACRE
        )

    with col2:
        if inputs["land_cost_method"] == LandCostMethod.DIRECT:
            inputs["land_direct"] = st.number_input(
                "Total Land Cost ($)",
                min_value=0,
                max_value=100_000_000,
                value=st.session_state.get("land_direct", 2_500_000),
                step=100_000,
                format="%d",
                key="land_direct",
            )
            # Show implied $/acre for reference
            land_per_acre = st.session_state.get("land_cost_per_acre", 1_000_000)
            if land_per_acre > 0:
                implied_acres = inputs["land_direct"] / land_per_acre
                st.caption(f"≈ {implied_acres:.2f} acres at ${land_per_acre:,.0f}/acre")
        else:
            inputs["land_per_acre"] = st.session_state.get("land_cost_per_acre", 1_000_000)
            # Acres will be calculated from units / units_per_acre
            st.caption("Land cost calculated from $/Acre × Required Acres")

    st.divider()

    # === HARD COSTS SECTION ===
    st.markdown("#### Hard Costs")

    col1, col2, col3 = st.columns(3)
    with col1:
        inputs["hard_cost_per_unit"] = st.number_input(
            "Hard Cost $/Unit",
            min_value=50_000,
            max_value=500_000,
            value=st.session_state.get("hard_cost_per_unit", 155_000),
            step=5_000,
            format="%d",
            key="hard_cost_per_unit_su",
        )

    with col2:
        inputs["hard_cost_contingency_pct"] = st.slider(
            "Hard Cost Contingency",
            min_value=0.0,
            max_value=15.0,
            value=5.0,
            step=0.5,
            format="%.1f%%",
            key="hard_cost_contingency_pct",
        ) / 100

    with col3:
        target_units = st.session_state.get("target_units", 200)
        hard_subtotal = inputs["hard_cost_per_unit"] * target_units
        hard_contingency = hard_subtotal * inputs["hard_cost_contingency_pct"]
        st.metric("Hard Costs Total", f"${hard_subtotal + hard_contingency:,.0f}")

    st.divider()

    # === SOFT COSTS SECTION ===
    st.markdown("#### Soft Costs")

    col1, col2, col3 = st.columns(3)
    with col1:
        inputs["soft_cost_pct"] = st.slider(
            "Soft Costs (% of Hard)",
            min_value=0.0,
            max_value=60.0,
            value=st.session_state.get("soft_cost_pct", 30.0),
            step=1.0,
            format="%.0f%%",
            key="soft_cost_pct_su",
        ) / 100

    with col2:
        inputs["developer_fee_pct"] = st.slider(
            "Developer Fee (% of Hard)",
            min_value=0.0,
            max_value=10.0,
            value=4.0,
            step=0.5,
            format="%.1f%%",
            key="developer_fee_pct",
        ) / 100

    with col3:
        inputs["soft_cost_contingency_pct"] = st.slider(
            "Soft Cost Contingency",
            min_value=0.0,
            max_value=15.0,
            value=5.0,
            step=0.5,
            format="%.1f%%",
            key="soft_cost_contingency_pct",
        ) / 100

    st.divider()

    # === FINANCING SECTION ===
    st.markdown("#### Financing Costs")

    col1, col2 = st.columns(2)
    with col1:
        inputs["construction_loan_fee_pct"] = st.slider(
            "Construction Loan Fee",
            min_value=0.0,
            max_value=3.0,
            value=1.0,
            step=0.25,
            format="%.2f%%",
            key="construction_loan_fee_pct",
        ) / 100

    with col2:
        st.caption("IDC calculated from draw schedule")
        st.caption("(Rate × Balance × Time)")

    st.divider()

    # === RESERVES SECTION ===
    st.markdown("#### Reserves")

    col1, col2 = st.columns(2)
    with col1:
        inputs["operating_reserve_months"] = st.number_input(
            "Operating Reserve (months)",
            min_value=0,
            max_value=12,
            value=3,
            step=1,
            key="operating_reserve_months",
        )

    with col2:
        inputs["leaseup_reserve_months"] = st.number_input(
            "Lease-up Reserve (months of debt service)",
            min_value=0,
            max_value=12,
            value=6,
            step=1,
            key="leaseup_reserve_months",
        )

    st.divider()

    # === SOURCES SECTION ===
    st.markdown("#### Sources Configuration")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Senior Debt**")
        inputs["construction_ltc_cap"] = st.slider(
            "Construction Loan LTC Cap",
            min_value=0.0,
            max_value=90.0,
            value=st.session_state.get("construction_ltc_pct", 65.0),
            step=5.0,
            format="%.0f%%",
            key="construction_ltc_cap",
        ) / 100

    with col2:
        st.markdown("**Gap Funding / Incentives**")

        # TIF Lump Sum with auto-calculate option
        tif_col1, tif_col2 = st.columns([2, 1])
        with tif_col1:
            inputs["tif_lump_sum"] = st.number_input(
                "TIF Lump Sum ($)",
                min_value=0,
                max_value=50_000_000,
                value=st.session_state.get("tif_lump_sum_calculated", 0),
                step=100_000,
                format="%d",
                key="tif_lump_sum_source",
            )
        with tif_col2:
            if st.button("Calculate", key="calc_tif_btn", help="Calculate from Property Tax tab settings"):
                # Import and calculate TIF
                from src.calculations.property_tax import get_austin_tax_stack

                tax_stack = get_austin_tax_stack()
                target_units = st.session_state.get("target_units", 200)
                hard_cost = st.session_state.get("hard_cost_per_unit", 155000) * target_units
                soft_cost = hard_cost * st.session_state.get("soft_cost_pct", 0.30)
                land_cost = st.session_state.get("land_cost_per_acre", 1000000) * 3  # ~3 acres

                # Estimate TDC
                estimated_tdc = land_cost + hard_cost + soft_cost * 1.3  # Include IDC estimate
                base_value = st.session_state.get("existing_assessed_value", 5000000)

                # Calculate increment
                increment = max(0, estimated_tdc - base_value)
                city_annual_increment = increment * tax_stack.tif_participating_rate_decimal

                # TIF cap rate from property tax tab or default
                tif_cap_rate = st.session_state.get("prop_tif_cap_rate", 9.5) / 100
                if tif_cap_rate <= 0:
                    tif_cap_rate = 0.095

                calculated_tif = city_annual_increment / tif_cap_rate
                st.session_state["tif_lump_sum_calculated"] = int(calculated_tif)
                st.session_state["tif_lump_sum_source"] = int(calculated_tif)
                st.rerun()

        # SMART Fee Waiver with auto-calculate
        smart_col1, smart_col2 = st.columns([2, 1])
        with smart_col1:
            inputs["fee_waivers"] = st.number_input(
                "SMART Fee Waivers ($)",
                min_value=0,
                max_value=20_000_000,
                value=st.session_state.get("smart_waiver_calculated", 0),
                step=50_000,
                format="%d",
                key="fee_waivers_source",
            )
        with smart_col2:
            if st.button("Calculate", key="calc_smart_btn", help="Calculate from SMART tier"):
                from src.calculations.property_tax import calculate_smart_fee_waiver

                tier = st.session_state.get("smart_tier", 2)
                total_units = st.session_state.get("target_units", 200)
                waiver = calculate_smart_fee_waiver(tier=tier, total_units=total_units)

                st.session_state["smart_waiver_calculated"] = int(waiver.total_waiver)
                st.session_state["fee_waivers_source"] = int(waiver.total_waiver)
                st.rerun()

        inputs["grants"] = st.number_input(
            "Grants ($)",
            min_value=0,
            max_value=50_000_000,
            value=0,
            step=100_000,
            format="%d",
            key="grants_source",
        )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Mezzanine Debt**")
        inputs["mezzanine_debt"] = st.number_input(
            "Mezzanine Amount ($)",
            min_value=0,
            max_value=100_000_000,
            value=0,
            step=100_000,
            format="%d",
            key="mezzanine_debt",
        )
        inputs["mezzanine_rate"] = st.slider(
            "Mezzanine Rate",
            min_value=0.0,
            max_value=25.0,
            value=st.session_state.get("mezzanine_rate_pct", 12.0),
            step=0.5,
            format="%.1f%%",
            key="mezzanine_rate_pct",
        ) / 100

        st.markdown("**Preferred Equity**")
        inputs["preferred_equity"] = st.number_input(
            "Preferred Amount ($)",
            min_value=0,
            max_value=100_000_000,
            value=0,
            step=100_000,
            format="%d",
            key="preferred_equity",
        )
        inputs["preferred_return"] = st.slider(
            "Preferred Return",
            min_value=0.0,
            max_value=20.0,
            value=st.session_state.get("preferred_return_pct", 10.0),
            step=0.5,
            format="%.1f%%",
            key="preferred_return_pct",
        ) / 100

    with col2:
        st.markdown("**Deferred Items**")
        inputs["deferred_developer_fee_pct"] = st.slider(
            "Deferred Developer Fee (%)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=10.0,
            format="%.0f%%",
            key="deferred_developer_fee_pct",
        ) / 100

    return inputs


def render_sources_uses_detailed_table(su: SourcesUsesDetailed) -> None:
    """Render the detailed Sources & Uses tables with $/unit column."""

    uses = su.uses
    sources = su.sources
    units = uses.target_units
    tdc = uses.total_development_cost

    st.subheader("Sources & Uses Summary")

    # Check balance
    if not su.is_balanced:
        st.error(f"⚠️ Sources and Uses don't balance! Difference: ${su.balance_difference:,.0f}")

    col1, col2 = st.columns(2)

    # === USES TABLE ===
    with col1:
        st.markdown("**USES**")

        uses_rows = []

        def add_uses_row(category: str, subcategory: str, amount: float, is_total: bool = False):
            pct_tdc = amount / tdc if tdc > 0 else 0
            per_unit = amount / units if units > 0 else 0
            if is_total:
                uses_rows.append({
                    "Category": f"**{category}**",
                    "Subcategory": f"**{subcategory}**",
                    "Amount": f"**${amount:,.0f}**",
                    "% TDC": f"**{pct_tdc:.1%}**",
                    "$/Unit": f"**${per_unit:,.0f}**",
                })
            else:
                uses_rows.append({
                    "Category": category,
                    "Subcategory": subcategory,
                    "Amount": f"${amount:,.0f}",
                    "% TDC": f"{pct_tdc:.1%}",
                    "$/Unit": f"${per_unit:,.0f}",
                })

        # Land & Acquisition
        add_uses_row("Land & Acquisition", "Land", uses.land_total)
        add_uses_row("", "Total Land", uses.total_land_acquisition, is_total=True)

        # Hard Costs
        add_uses_row("Hard Costs", "Construction", uses.hard_costs_subtotal)
        add_uses_row("", f"Contingency ({uses.hard_cost_contingency_pct:.0%})", uses.hard_cost_contingency)
        add_uses_row("", "Total Hard Costs", uses.hard_costs_total, is_total=True)

        # Soft Costs
        add_uses_row("Soft Costs", "Soft Costs", uses.soft_costs_subtotal)
        add_uses_row("", f"Contingency ({uses.soft_cost_contingency_pct:.0%})", uses.soft_cost_contingency)
        add_uses_row("", "Total Soft Costs", uses.soft_costs_total, is_total=True)

        # Financing
        add_uses_row("Financing", "IDC", uses.idc)
        add_uses_row("", f"Loan Fee ({uses.construction_loan_fee_pct:.1%})", uses.construction_loan_fee)
        add_uses_row("", "Total Financing", uses.financing_costs_total, is_total=True)

        # Reserves
        add_uses_row("Reserves", f"Operating ({uses.operating_reserve_months} mo)", uses.operating_reserve)
        add_uses_row("", f"Lease-up ({uses.leaseup_reserve_months} mo)", uses.leaseup_reserve)
        add_uses_row("", "Total Reserves", uses.reserves_total, is_total=True)

        # TDC
        add_uses_row("", "TOTAL DEVELOPMENT COST", tdc, is_total=True)

        df_uses = pd.DataFrame(uses_rows)
        st.dataframe(df_uses, use_container_width=True, hide_index=True)

    # === SOURCES TABLE ===
    with col2:
        st.markdown("**SOURCES**")

        sources_rows = []

        def add_sources_row(category: str, subcategory: str, amount: float, is_total: bool = False):
            pct_tdc = amount / tdc if tdc > 0 else 0
            per_unit = amount / units if units > 0 else 0
            if is_total:
                sources_rows.append({
                    "Category": f"**{category}**",
                    "Subcategory": f"**{subcategory}**",
                    "Amount": f"**${amount:,.0f}**",
                    "% TDC": f"**{pct_tdc:.1%}**",
                    "$/Unit": f"**${per_unit:,.0f}**",
                })
            else:
                sources_rows.append({
                    "Category": category,
                    "Subcategory": subcategory,
                    "Amount": f"${amount:,.0f}",
                    "% TDC": f"{pct_tdc:.1%}",
                    "$/Unit": f"${per_unit:,.0f}",
                })

        # Senior Debt
        add_sources_row("Senior Debt", f"Construction Loan ({sources.construction_loan_ltc_cap:.0%} LTC)",
                       sources.construction_loan)

        # Mezzanine / Preferred
        if sources.mezzanine_debt > 0:
            add_sources_row("Mezzanine", f"Mezzanine Debt ({sources.mezzanine_rate:.1%})", sources.mezzanine_debt)
        if sources.preferred_equity > 0:
            add_sources_row("Preferred", f"Preferred Equity ({sources.preferred_return:.1%})", sources.preferred_equity)

        # Gap Funding
        if sources.gap_funding_total > 0:
            if sources.tif_lump_sum > 0:
                add_sources_row("Gap Funding", "TIF Lump Sum", sources.tif_lump_sum)
            if sources.grants > 0:
                add_sources_row("", "Grants", sources.grants)
            if sources.fee_waivers > 0:
                add_sources_row("", "Fee Waivers", sources.fee_waivers)
            add_sources_row("", "Total Gap Funding", sources.gap_funding_total, is_total=True)

        # Deferred
        if sources.deferred_developer_fee > 0:
            add_sources_row("Deferred", f"Developer Fee ({sources.deferred_developer_fee_pct:.0%})",
                          sources.deferred_developer_fee)

        # Equity
        add_sources_row("Equity", "Required Equity", sources.equity_required)

        # Total
        add_sources_row("", "TOTAL SOURCES", sources.total_sources, is_total=True)

        df_sources = pd.DataFrame(sources_rows)
        st.dataframe(df_sources, use_container_width=True, hide_index=True)

    # === KEY METRICS ===
    st.divider()
    st.markdown("**Key Metrics**")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("TDC", f"${tdc:,.0f}")
        st.metric("TDC/Unit", f"${tdc/units:,.0f}" if units > 0 else "$0")
    with col2:
        st.metric("Senior Debt LTC", f"{sources.construction_loan_ltc_actual:.1%}")
        st.metric("Construction Loan", f"${sources.construction_loan:,.0f}")
    with col3:
        st.metric("Equity Required", f"${sources.equity_required:,.0f}")
        st.metric("Equity %", f"{sources.equity_pct_of_tdc:.1%}")
    with col4:
        if uses.land_acres > 0:
            st.metric("Land (acres)", f"{uses.land_acres:.2f}")
        st.metric("Land Cost", f"${uses.land_total:,.0f}")

    # === CAPITAL STACK VISUALIZATION ===
    st.divider()
    st.markdown("**Capital Stack**")

    # Build capital stack data
    stack_items = []
    stack_colors = []

    if sources.construction_loan > 0:
        stack_items.append(("Senior Debt", sources.construction_loan, sources.construction_loan / tdc))
        stack_colors.append("#1f77b4")  # Blue

    if sources.mezzanine_debt > 0:
        stack_items.append(("Mezzanine Debt", sources.mezzanine_debt, sources.mezzanine_debt / tdc))
        stack_colors.append("#ff7f0e")  # Orange

    if sources.preferred_equity > 0:
        stack_items.append(("Preferred Equity", sources.preferred_equity, sources.preferred_equity / tdc))
        stack_colors.append("#9467bd")  # Purple

    if sources.gap_funding_total > 0:
        stack_items.append(("Gap Funding/Incentives", sources.gap_funding_total, sources.gap_funding_total / tdc))
        stack_colors.append("#2ca02c")  # Green

    if sources.deferred_developer_fee > 0:
        stack_items.append(("Deferred Fee", sources.deferred_developer_fee, sources.deferred_developer_fee / tdc))
        stack_colors.append("#bcbd22")  # Yellow-green

    if sources.equity_required > 0:
        stack_items.append(("Common Equity", sources.equity_required, sources.equity_required / tdc))
        stack_colors.append("#d62728")  # Red

    # Display as horizontal bar chart using streamlit
    col1, col2 = st.columns([2, 1])

    with col1:
        # Simple text-based stacked bar
        stack_html = '<div style="display: flex; width: 100%; height: 40px; border-radius: 4px; overflow: hidden;">'
        for (name, amount, pct), color in zip(stack_items, stack_colors):
            width_pct = pct * 100
            if width_pct >= 5:  # Only show label if segment is wide enough
                stack_html += f'<div style="width: {width_pct}%; background-color: {color}; display: flex; align-items: center; justify-content: center; color: white; font-size: 11px; font-weight: bold;">{pct:.0%}</div>'
            else:
                stack_html += f'<div style="width: {width_pct}%; background-color: {color};"></div>'
        stack_html += '</div>'
        st.markdown(stack_html, unsafe_allow_html=True)

    with col2:
        # Legend
        legend_html = '<div style="font-size: 12px;">'
        for (name, amount, pct), color in zip(stack_items, stack_colors):
            legend_html += f'<div style="margin: 2px 0;"><span style="display: inline-block; width: 12px; height: 12px; background-color: {color}; margin-right: 4px; border-radius: 2px;"></span>{name}</div>'
        legend_html += '</div>'
        st.markdown(legend_html, unsafe_allow_html=True)

    # Summary table
    stack_df_data = []
    for (name, amount, pct), color in zip(stack_items, stack_colors):
        stack_df_data.append({
            "Source": name,
            "Amount": f"${amount:,.0f}",
            "% TDC": f"{pct:.1%}",
            "$/Unit": f"${amount/units:,.0f}" if units > 0 else "-",
        })

    if stack_df_data:
        df_stack = pd.DataFrame(stack_df_data)
        st.dataframe(df_stack, use_container_width=True, hide_index=True)

    # === UNDERWRITING ALERTS ===
    alerts = []

    # Check minimum equity
    if sources.equity_pct_of_tdc < 0.10:
        alerts.append(("warning", f"Equity at {sources.equity_pct_of_tdc:.1%} is below typical 10% minimum"))
    elif sources.equity_pct_of_tdc < 0.20:
        alerts.append(("info", f"Equity at {sources.equity_pct_of_tdc:.1%} is below typical 20-25% target"))

    # Check senior debt LTC
    if sources.construction_loan_ltc_actual > 0.75:
        alerts.append(("warning", f"Senior debt LTC at {sources.construction_loan_ltc_actual:.1%} exceeds typical 70-75% max"))

    # Check combined leverage
    total_debt = sources.construction_loan + sources.mezzanine_debt + sources.preferred_equity
    combined_ltc = total_debt / tdc if tdc > 0 else 0
    if combined_ltc > 0.90:
        alerts.append(("error", f"Combined leverage at {combined_ltc:.1%} is very high (>90%)"))
    elif combined_ltc > 0.80:
        alerts.append(("warning", f"Combined leverage at {combined_ltc:.1%} exceeds typical 80% target"))

    # Check TDC per unit
    tdc_per_unit = tdc / units if units > 0 else 0
    if tdc_per_unit > 400_000:
        alerts.append(("warning", f"TDC/Unit at ${tdc_per_unit:,.0f} is very high"))

    # Display alerts
    if alerts:
        st.divider()
        st.markdown("**Underwriting Alerts**")
        for alert_type, message in alerts:
            if alert_type == "error":
                st.error(message)
            elif alert_type == "warning":
                st.warning(message)
            else:
                st.info(message)
