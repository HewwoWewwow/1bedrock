"""TIF Analysis UI component.

Shows detailed TIF calculations including:
- Tax rate breakdown by entity
- TIF lump sum calculation
- TIF-as-loan repayment schedule
- SMART fee waiver
- City tax abatement
"""

import streamlit as st
import pandas as pd
from typing import Optional

from src.calculations.property_tax import (
    TaxingAuthorityStack,
    get_austin_tax_stack,
    TIFLumpSumResult,
    calculate_tif_lump_sum,
    TIFLoanSchedule,
    calculate_tif_loan_schedule,
    solve_tif_term,
    SMARTFeeWaiver,
    calculate_smart_fee_waiver,
    CityTaxAbatement,
    calculate_city_tax_abatement,
)
from src.models.incentives import IncentiveTier, TIF_PARAMS, TIER_REQUIREMENTS
from src.calculations.tif_calculator import (
    calculate_tif_lump_sum_from_tier,
    get_tier_tif_defaults,
    TIFCalculationResult,
)


def parse_currency_input(value: str) -> float:
    """Parse a currency string (with commas, $, etc.) to float."""
    if not value:
        return 0.0
    # Remove $, commas, spaces
    cleaned = value.replace("$", "").replace(",", "").replace(" ", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def format_currency_input(value: float) -> str:
    """Format a float as currency string with commas (no $ sign for input)."""
    return f"{value:,.0f}"


def render_tax_rate_breakdown(tax_stack: Optional[TaxingAuthorityStack] = None) -> None:
    """Render property tax rate breakdown by taxing entity."""
    if tax_stack is None:
        tax_stack = get_austin_tax_stack()

    st.markdown("### Property Tax Rates by Entity")

    data = []
    for auth in tax_stack.authorities:
        data.append({
            "Entity": auth.name,
            "Code": auth.code,
            "Rate (per $100)": f"${auth.rate_per_100:.4f}",
            "Rate (%)": f"{auth.rate_decimal:.4%}",
            "TIF Participating": "Yes" if auth.participates_in_tif else "No",
        })

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rate", f"{tax_stack.total_rate_per_100:.4f}%")
    with col2:
        st.metric("TIF Participating", f"{tax_stack.tif_participating_rate_per_100:.4f}%")
    with col3:
        st.metric("Non-TIF Rate", f"{tax_stack.non_tif_rate_per_100:.4f}%")


def render_tif_lump_sum_inputs() -> dict:
    """Render TIF lump sum calculation inputs and return values.

    Shows the calculated TIF from Scenarios tab by default, with optional manual overrides.
    """
    st.markdown("### TIF Lump Sum Calculation")

    # Get selected tier from session state
    selected_tier_num = st.session_state.get("selected_tier", 2)
    tier = IncentiveTier(selected_tier_num)
    tier_defaults = get_tier_tif_defaults(tier)
    tier_reqs = TIER_REQUIREMENTS[tier]

    # Get project values from session state (same as Scenarios tab uses)
    target_units = st.session_state.get("target_units", 200)
    hard_cost_per_unit = st.session_state.get("hard_cost_per_unit", 155_000)
    soft_cost_pct = st.session_state.get("soft_cost_pct", 30.0) / 100
    land_cost = st.session_state.get("land_cost_per_acre", 1_000_000)
    base_av_default = st.session_state.get("existing_assessed_value", 5_000_000)

    # Get affordable_pct - handle both percentage (20) and decimal (0.20) formats
    affordable_pct_raw = st.session_state.get("affordable_pct", 20.0)
    affordable_pct = affordable_pct_raw / 100.0 if affordable_pct_raw > 1 else affordable_pct_raw
    affordable_units_default = max(1, int(target_units * affordable_pct))

    # Estimate TDC (same formula as Scenarios tab)
    hard_costs = target_units * hard_cost_per_unit
    soft_costs = hard_costs * soft_cost_pct
    estimated_tdc = land_cost + hard_costs + soft_costs

    # Show what Scenarios tab calculated (if available)
    scenarios_tif = st.session_state.get("calculated_tif_lump_sum", None)

    # Show tier info
    st.info(f"**Current Tier: {selected_tier_num}** - Cap Rate: {tier_defaults['cap_rate']:.1%}, "
            f"Term: {tier_defaults['term_years']} years, Escalation: {tier_defaults['escalation_rate']:.1%}")

    if scenarios_tif:
        st.success(f"**TIF from Scenarios Tab: ${scenarios_tif:,.0f}**")

    use_overrides = st.checkbox(
        "Enable Manual Overrides (What-If Analysis)",
        value=st.session_state.get("tif_use_overrides", False),
        key="tif_use_overrides",
        help="Override the calculated TIF with custom parameters for what-if analysis"
    )

    if use_overrides:
        st.warning("Manual overrides enabled. Values below will differ from Scenarios tab calculation.")

        with st.expander("TIF Manual Overrides", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                # New Assessed Value - format with commas
                # Initialize or reformat existing session state value
                if "tif_new_av_text" not in st.session_state:
                    st.session_state["tif_new_av_text"] = f"{estimated_tdc:,.0f}"
                else:
                    # Reformat on each render to ensure commas
                    current_val = parse_currency_input(st.session_state["tif_new_av_text"])
                    st.session_state["tif_new_av_text"] = f"{current_val:,.0f}"

                new_av_str = st.text_input(
                    "New Assessed Value (TDC)",
                    key="tif_new_av_text",
                    help="Typically equal to TDC or income-based value",
                )
                new_assessed_value = parse_currency_input(new_av_str)

                # Base Assessed Value - format with commas
                if "tif_base_av_text" not in st.session_state:
                    st.session_state["tif_base_av_text"] = f"{base_av_default:,.0f}"
                else:
                    # Reformat on each render to ensure commas
                    current_val = parse_currency_input(st.session_state["tif_base_av_text"])
                    st.session_state["tif_base_av_text"] = f"{current_val:,.0f}"

                base_av_str = st.text_input(
                    "Base Assessed Value",
                    key="tif_base_av_text",
                    help="Existing assessed value before development",
                )
                base_assessed_value = parse_currency_input(base_av_str)

                affordable_units = st.number_input(
                    "Affordable Units",
                    min_value=1,
                    value=affordable_units_default,
                    step=1,
                    key="tif_override_aff_units",
                )

            with col2:
                tif_cap_rate = st.slider(
                    "TIF Cap Rate (Override)",
                    min_value=5.0,
                    max_value=15.0,
                    value=tier_defaults['cap_rate'] * 100,
                    step=0.25,
                    format="%.2f%%",
                    key="tif_override_cap_rate",
                    help="Cap rate for capitalizing tax increment",
                ) / 100

                discount_rate = st.slider(
                    "Escalation Rate (Override)",
                    min_value=0.0,
                    max_value=10.0,
                    value=tier_defaults['escalation_rate'] * 100,
                    step=0.25,
                    format="%.2f%%",
                    key="tif_override_discount",
                    help="Annual escalation rate for tax increment",
                ) / 100

                tif_term = st.number_input(
                    "TIF Term (years) (Override)",
                    min_value=5,
                    max_value=40,
                    value=tier_defaults['term_years'],
                    step=1,
                    key="tif_override_term",
                )
    else:
        # Use same values as Scenarios tab (tier defaults applied to project inputs)
        new_assessed_value = estimated_tdc
        base_assessed_value = base_av_default
        affordable_units = affordable_units_default
        tif_cap_rate = tier_defaults['cap_rate']
        discount_rate = tier_defaults['escalation_rate']
        tif_term = tier_defaults['term_years']

        st.caption(f"Using same calculation as Scenarios tab (Tier {selected_tier_num} defaults). "
                   "Enable overrides above for what-if analysis.")

    return {
        "new_assessed_value": new_assessed_value,
        "base_assessed_value": base_assessed_value,
        "affordable_units": affordable_units,
        "tif_cap_rate": tif_cap_rate,
        "discount_rate": discount_rate,
        "tif_term": tif_term,
        "tier": tier,
        "use_overrides": use_overrides,
    }


def render_tif_lump_sum_result(
    new_assessed_value: float,
    base_assessed_value: float,
    affordable_units: int,
    tif_cap_rate: float,
    discount_rate: float,
    tif_term: int,
    tax_stack: Optional[TaxingAuthorityStack] = None,
    tier: Optional[IncentiveTier] = None,
) -> Optional[float]:
    """Calculate and render TIF lump sum result. Returns lump sum amount."""
    if tax_stack is None:
        tax_stack = get_austin_tax_stack()

    if tier is None:
        tier = IncentiveTier(st.session_state.get("selected_tier", 2))

    # Use the calculator for consistent logic
    tif_result = calculate_tif_lump_sum_from_tier(
        new_assessed_value=new_assessed_value,
        base_assessed_value=base_assessed_value,
        tier=tier,
        affordable_units=affordable_units,
        tax_stack=tax_stack,
        override_cap_rate=tif_cap_rate,
        override_term_years=tif_term,
        override_escalation_rate=discount_rate,
    )

    st.markdown("### TIF Lump Sum Result")

    # Show calculation steps
    with st.expander("Calculation Details", expanded=False):
        st.markdown(f"""
**Step 1: Calculate Incremental Value**
- New Assessed Value: ${new_assessed_value:,.0f}
- Base Assessed Value: ${base_assessed_value:,.0f}
- **Incremental Value**: ${tif_result.incremental_value:,.0f}

**Step 2: Calculate Annual City Increment**
- Incremental Value: ${tif_result.incremental_value:,.0f}
- City (TIF) Rate: {tif_result.city_rate_decimal:.4%}
- **Annual City Increment**: ${tif_result.annual_city_increment:,.0f}

**Step 3: Capitalize to Lump Sum**
- Annual City Increment: ${tif_result.annual_city_increment:,.0f}
- Cap Rate: {tif_cap_rate:.2%}
- **TIF Lump Sum**: ${tif_result.tif_lump_sum:,.0f}
        """)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("TIF Lump Sum", f"${tif_result.tif_lump_sum:,.0f}")
    with col2:
        st.metric("Per Affordable Unit", f"${tif_result.per_affordable_unit:,.0f}")
    with col3:
        pct_of_tdc = tif_result.tif_lump_sum / new_assessed_value if new_assessed_value > 0 else 0
        st.metric("% of TDC", f"{pct_of_tdc:.2%}")
    with col4:
        st.metric("Annual Increment", f"${tif_result.annual_city_increment:,.0f}")

    # TIF Increment Buildup Schedule
    st.markdown("### TIF Increment Buildup")
    st.caption("Shows how the cumulative tax increment builds up over the TIF term.")

    # Create buildup table
    buildup_data = []
    for year in range(1, tif_term + 1):
        year_increment = tif_result.annual_city_increment * ((1 + discount_rate) ** (year - 1))
        nominal_cumulative = tif_result.buildup_schedule_nominal.get(year, 0)
        real_cumulative = tif_result.buildup_schedule_real.get(year, 0)

        buildup_data.append({
            "Year": year,
            "Annual Increment": year_increment,
            "Cumulative (Nominal)": nominal_cumulative,
            "Cumulative (Real $)": real_cumulative,
        })

    df = pd.DataFrame(buildup_data)

    # Show chart
    col1, col2 = st.columns(2)

    with col1:
        # Format for display
        df_display = df.copy()
        df_display["Annual Increment"] = df_display["Annual Increment"].apply(lambda x: f"${x:,.0f}")
        df_display["Cumulative (Nominal)"] = df_display["Cumulative (Nominal)"].apply(lambda x: f"${x:,.0f}")
        df_display["Cumulative (Real $)"] = df_display["Cumulative (Real $)"].apply(lambda x: f"${x:,.0f}")

        st.dataframe(df_display, use_container_width=True, hide_index=True, height=300)

    with col2:
        # Chart
        chart_df = df[["Year", "Cumulative (Nominal)", "Cumulative (Real $)"]].copy()
        chart_df = chart_df.set_index("Year")
        st.line_chart(chart_df, height=300)

    # Summary row
    total_nominal = tif_result.buildup_schedule_nominal.get(tif_term, 0)
    total_real = tif_result.buildup_schedule_real.get(tif_term, 0)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Increment (Nominal)", f"${total_nominal:,.0f}")
    with col2:
        st.metric("Total Increment (Real $)", f"${total_real:,.0f}")
    with col3:
        st.metric("Lump Sum Equivalent", f"${tif_result.tif_lump_sum:,.0f}",
                 help="Capitalized value of the increment stream")

    return tif_result.tif_lump_sum


def render_tif_loan_schedule(
    principal: float,
    annual_rate: float,
    monthly_payment: float,
    escalation_rate: float = 0.015,
) -> None:
    """Render TIF-as-loan repayment schedule."""
    st.markdown("### TIF as Amortizing Loan")

    st.caption("""
    The TIF lump sum can be viewed as a loan from the city to the developer.
    The developer's regular property tax payments serve as loan payments,
    with principal and interest components.
    """)

    # Solve for term
    term_months = solve_tif_term(
        principal=principal,
        annual_rate=annual_rate,
        monthly_payment=monthly_payment,
        escalation_rate=escalation_rate,
    )

    if term_months == float('inf'):
        st.error("Payment amount is too low to repay principal.")
        return

    # Generate schedule
    schedule = calculate_tif_loan_schedule(
        principal=principal,
        annual_rate=annual_rate,
        monthly_payment=monthly_payment,
        max_term_months=min(int(term_months) + 12, 360),
        escalation_rate=escalation_rate,
    )

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Principal", f"${principal:,.0f}")
    with col2:
        st.metric("Term", f"{schedule.term_years:.1f} years")
    with col3:
        st.metric("Total Interest", f"${schedule.total_interest:,.0f}")
    with col4:
        interest_ratio = schedule.total_interest / principal if principal > 0 else 0
        st.metric("Interest / Principal", f"{interest_ratio:.2%}")

    # Show annual summary
    with st.expander("Annual Repayment Summary", expanded=False):
        # Aggregate by year
        annual_data = {}
        for pmt in schedule.payments:
            year = (pmt.period - 1) // 12 + 1
            if year not in annual_data:
                annual_data[year] = {
                    "Year": year,
                    "BoP Balance": pmt.principal_bop,
                    "Total Payment": 0,
                    "Interest": 0,
                    "Principal": 0,
                    "EoP Balance": pmt.principal_eop,
                }
            annual_data[year]["Total Payment"] += pmt.payment
            annual_data[year]["Interest"] += pmt.interest
            annual_data[year]["Principal"] += pmt.principal_paid
            annual_data[year]["EoP Balance"] = pmt.principal_eop

        df_data = []
        for year, data in annual_data.items():
            df_data.append({
                "Year": year,
                "BoP Balance": f"${data['BoP Balance']:,.0f}",
                "Payment": f"${data['Total Payment']:,.0f}",
                "Interest": f"${data['Interest']:,.0f}",
                "Principal": f"${data['Principal']:,.0f}",
                "EoP Balance": f"${data['EoP Balance']:,.0f}",
            })

        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_smart_fee_waiver(
    total_units: int,
) -> Optional[SMARTFeeWaiver]:
    """Render SMART fee waiver calculation.

    Uses the tier selected on the Scenarios tab (no separate selector here).
    Only shows values if SMART fee waiver is enabled on Scenarios tab.
    """
    st.markdown("### SMART Housing Fee Waiver")

    # Check if SMART fee waiver is enabled on Scenarios tab
    smart_enabled = st.session_state.get("smart_fee_waiver", False)

    if not smart_enabled:
        st.info("SMART Fee Waiver is not enabled. Enable it on the **Scenarios** tab to see calculations.")
        return None

    # Use tier from Scenarios tab
    selected_tier = st.session_state.get("selected_tier", 2)

    waiver = calculate_smart_fee_waiver(tier=selected_tier, total_units=total_units)

    st.success(f"**Using Tier {selected_tier}** from Scenarios tab")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        **Tier {selected_tier} Requirements:**
        - Affordable %: {waiver.affordable_pct:.0%}
        - AMI Level: {waiver.ami_level:.0%}
        - Waiver: {waiver.waiver_pct:.0%}
        """)
    with col2:
        st.markdown(f"""
        **Unit Breakdown:**
        - Total Units: {total_units}
        - Affordable Units: {waiver.affordable_units}
        """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Per Unit Waiver", f"${waiver.per_unit_amount:,.0f}")
    with col2:
        st.metric("Total Waiver", f"${waiver.total_waiver:,.0f}")
    with col3:
        st.metric("Waiver %", f"{waiver.waiver_pct:.0%}")

    return waiver


def render_city_tax_abatement(
    assessed_value: float,
    affordable_units: int,
    total_units: int,
) -> Optional[CityTaxAbatement]:
    """Render city tax abatement calculation.

    Tax abatement and TIF are mutually exclusive - can't have both.
    Only shows values if tax abatement is enabled on Scenarios tab and TIF is not selected.
    """
    st.markdown("### City Property Tax Abatement")

    # Check if tax abatement is enabled on Scenarios tab
    abatement_enabled = st.session_state.get("tax_abatement", False)

    # Check if TIF is selected (mutually exclusive with abatement)
    tif_lump_sum_selected = st.session_state.get("tif_lump_sum", False)
    tif_stream_selected = st.session_state.get("tif_stream", False)
    tif_selected = tif_lump_sum_selected or tif_stream_selected

    if tif_selected and abatement_enabled:
        st.warning("**Tax Abatement and TIF are mutually exclusive.** You have both selected on the Scenarios tab. "
                  "Please choose one or the other. TIF captures the city's increment, so abatement cannot also apply.")
        return None

    if not abatement_enabled:
        st.info("Tax Abatement is not enabled. Enable it on the **Scenarios** tab to see calculations. "
               "Note: Tax Abatement and TIF Lump Sum are mutually exclusive.")
        return None

    st.caption("Abatement reduces 100% of the City's property tax increment for the pro rata portion "
              "attributable to affordable units, based on the selected incentive tier.")

    # Get tier info
    selected_tier = st.session_state.get("selected_tier", 2)
    affordable_pct_raw = st.session_state.get("affordable_pct", 20.0)
    affordable_pct = affordable_pct_raw / 100.0 if affordable_pct_raw > 1 else affordable_pct_raw

    st.success(f"**Using Tier {selected_tier}** from Scenarios tab ({affordable_pct:.0%} affordable)")

    col1, col2 = st.columns(2)

    with col1:
        abatement_pct = st.slider(
            "Abatement Percentage",
            min_value=0,
            max_value=100,
            value=100,  # Default to 100% abatement
            step=5,
            format="%d%%",
            key="abate_pct",
            help="Percentage of city tax on affordable units that is abated",
        ) / 100

    with col2:
        term_years = st.number_input(
            "Abatement Term (years)",
            min_value=1,
            max_value=30,
            value=15,
            step=1,
            key="abate_term",
            help="Number of years the abatement applies",
        )

    # Calculate affordable units based on tier
    calculated_affordable_units = max(1, int(total_units * affordable_pct))

    abatement = calculate_city_tax_abatement(
        assessed_value=assessed_value,
        base_assessed_value=0,  # Use full value for abatement calc
        abatement_pct=abatement_pct,
        affordable_units=calculated_affordable_units,
        total_units=total_units,
        term_years=term_years,
    )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        **Calculation Basis:**
        - City Rate: {abatement.city_rate:.4%}
        - Affordable Share: {abatement.affordable_share:.1%}
        - Affordable Units: {calculated_affordable_units}
        """)
    with col2:
        st.markdown(f"""
        **Abatement Terms:**
        - Abatement %: {abatement_pct:.0%}
        - Term: {term_years} years
        """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Annual Abatement", f"${abatement.annual_abatement:,.0f}")
    with col2:
        st.metric("Monthly Abatement", f"${abatement.monthly_abatement:,.0f}")
    with col3:
        st.metric("Total Over Term", f"${abatement.total_abatement:,.0f}")

    return abatement


def render_tif_analysis_tab() -> None:
    """Render complete TIF analysis tab."""
    st.header("TIF & Incentive Analysis")

    # Tax rate breakdown
    tax_stack = get_austin_tax_stack()
    render_tax_rate_breakdown(tax_stack)

    st.divider()

    # TIF inputs
    inputs = render_tif_lump_sum_inputs()

    # TIF lump sum result
    tif_lump_sum = render_tif_lump_sum_result(
        new_assessed_value=inputs["new_assessed_value"],
        base_assessed_value=inputs["base_assessed_value"],
        affordable_units=inputs["affordable_units"],
        tif_cap_rate=inputs["tif_cap_rate"],
        discount_rate=inputs["discount_rate"],
        tif_term=inputs["tif_term"],
        tax_stack=tax_stack,
        tier=inputs.get("tier"),
    )

    # Store calculated TIF lump sum in session state for use by other tabs
    st.session_state["calculated_tif_lump_sum"] = tif_lump_sum

    st.divider()

    # TIF as loan
    if tif_lump_sum and tif_lump_sum > 0:
        incremental_value = inputs["new_assessed_value"] - inputs["base_assessed_value"]
        city_monthly_increment = (
            incremental_value * tax_stack.tif_participating_rate_decimal / 12
        )

        render_tif_loan_schedule(
            principal=tif_lump_sum,
            annual_rate=inputs["discount_rate"],
            monthly_payment=city_monthly_increment,
        )

    st.divider()

    # SMART fee waiver
    total_units = st.session_state.get("target_units", 200)
    render_smart_fee_waiver(total_units=total_units)

    st.divider()

    # City tax abatement
    render_city_tax_abatement(
        assessed_value=inputs["new_assessed_value"],
        affordable_units=inputs["affordable_units"],
        total_units=total_units,
    )
