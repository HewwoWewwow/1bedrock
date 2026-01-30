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
    """Render TIF lump sum calculation inputs and return values."""
    st.markdown("### TIF Lump Sum Calculation")

    with st.expander("TIF Lump Sum Inputs", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            # Assessed values
            new_assessed_value = st.number_input(
                "New Assessed Value (TDC)",
                min_value=0.0,
                value=st.session_state.get("tif_new_av", 55000000.0),
                step=1000000.0,
                format="%.0f",
                key="tif_new_av",
                help="Typically equal to TDC or income-based value",
            )

            base_assessed_value = st.number_input(
                "Base Assessed Value",
                min_value=0.0,
                value=st.session_state.get("tif_base_av", 5000000.0),
                step=100000.0,
                format="%.0f",
                key="tif_base_av",
                help="Existing assessed value before development",
            )

            affordable_units = st.number_input(
                "Affordable Units",
                min_value=1,
                value=st.session_state.get("tif_aff_units", 20),
                step=1,
                key="tif_aff_units",
            )

        with col2:
            tif_cap_rate = st.slider(
                "TIF Cap Rate",
                min_value=5.0,
                max_value=15.0,
                value=st.session_state.get("tif_cap_rate", 9.5),
                step=0.25,
                format="%.2f%%",
                key="tif_cap_rate",
                help="Cap rate for capitalizing tax increment (typically higher than exit cap)",
            ) / 100

            discount_rate = st.slider(
                "Discount Rate",
                min_value=1.0,
                max_value=10.0,
                value=st.session_state.get("tif_discount", 3.0),
                step=0.25,
                format="%.2f%%",
                key="tif_discount",
                help="Rate for discounting future payments / loan interest rate",
            ) / 100

            tif_term = st.number_input(
                "TIF Term (years)",
                min_value=5,
                max_value=40,
                value=st.session_state.get("tif_term", 20),
                step=1,
                key="tif_term",
            )

    return {
        "new_assessed_value": new_assessed_value,
        "base_assessed_value": base_assessed_value,
        "affordable_units": affordable_units,
        "tif_cap_rate": tif_cap_rate,
        "discount_rate": discount_rate,
        "tif_term": tif_term,
    }


def render_tif_lump_sum_result(
    new_assessed_value: float,
    base_assessed_value: float,
    affordable_units: int,
    tif_cap_rate: float,
    discount_rate: float,
    tif_term: int,
    tax_stack: Optional[TaxingAuthorityStack] = None,
) -> Optional[float]:
    """Calculate and render TIF lump sum result. Returns lump sum amount."""
    if tax_stack is None:
        tax_stack = get_austin_tax_stack()

    # Calculate increment
    incremental_value = max(0, new_assessed_value - base_assessed_value)

    # City's annual increment (using participating rate)
    city_annual_increment = incremental_value * tax_stack.tif_participating_rate_decimal

    # TIF lump sum = annual increment / cap rate
    # This is the spreadsheet's approach
    tif_lump_sum = city_annual_increment / tif_cap_rate if tif_cap_rate > 0 else 0

    st.markdown("### TIF Lump Sum Result")

    # Show calculation steps
    with st.expander("Calculation Details", expanded=False):
        st.markdown(f"""
**Step 1: Calculate Incremental Value**
- New Assessed Value: ${new_assessed_value:,.0f}
- Base Assessed Value: ${base_assessed_value:,.0f}
- **Incremental Value**: ${incremental_value:,.0f}

**Step 2: Calculate Annual City Increment**
- Incremental Value: ${incremental_value:,.0f}
- City (TIF) Rate: {tax_stack.tif_participating_rate_per_100:.4f}%
- **Annual City Increment**: ${city_annual_increment:,.0f}

**Step 3: Capitalize to Lump Sum**
- Annual City Increment: ${city_annual_increment:,.0f}
- Cap Rate: {tif_cap_rate:.2%}
- **TIF Lump Sum**: ${tif_lump_sum:,.0f}
        """)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("TIF Lump Sum", f"${tif_lump_sum:,.0f}")
    with col2:
        per_unit = tif_lump_sum / affordable_units if affordable_units > 0 else 0
        st.metric("Per Affordable Unit", f"${per_unit:,.0f}")
    with col3:
        pct_of_tdc = tif_lump_sum / new_assessed_value if new_assessed_value > 0 else 0
        st.metric("% of TDC", f"{pct_of_tdc:.2%}")
    with col4:
        st.metric("Annual Increment", f"${city_annual_increment:,.0f}")

    return tif_lump_sum


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
    tier: int = 2,
) -> SMARTFeeWaiver:
    """Render SMART fee waiver calculation."""
    st.markdown("### SMART Housing Fee Waiver")

    col1, col2 = st.columns(2)

    with col1:
        selected_tier = st.radio(
            "SMART Tier",
            options=[1, 2, 3],
            index=tier - 1,
            horizontal=True,
            key="smart_tier",
            help="Tier determines affordable % required and waiver amount",
        )

    waiver = calculate_smart_fee_waiver(tier=selected_tier, total_units=total_units)

    with col2:
        st.markdown(f"""
        **Tier {selected_tier} Requirements:**
        - Affordable %: {waiver.affordable_pct:.0%}
        - AMI Level: {waiver.ami_level:.0%}
        - Waiver: {waiver.waiver_pct:.0%}
        """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Affordable Units", f"{waiver.affordable_units}")
    with col2:
        st.metric("Per Unit Waiver", f"${waiver.per_unit_amount:,.0f}")
    with col3:
        st.metric("Total Waiver", f"${waiver.total_waiver:,.0f}")

    return waiver


def render_city_tax_abatement(
    assessed_value: float,
    affordable_units: int,
    total_units: int,
) -> Optional[CityTaxAbatement]:
    """Render city tax abatement calculation."""
    st.markdown("### City Property Tax Abatement")

    st.caption("Abatement applies only to the City of Austin's portion of property tax, "
              "and only on the share attributable to affordable units.")

    col1, col2, col3 = st.columns(3)

    with col1:
        abatement_pct = st.slider(
            "Abatement Percentage",
            min_value=0,
            max_value=100,
            value=st.session_state.get("abate_pct", 100),
            step=5,
            format="%d%%",
            key="abate_pct",
        ) / 100

    with col2:
        term_years = st.number_input(
            "Abatement Term (years)",
            min_value=1,
            max_value=30,
            value=st.session_state.get("abate_term", 15),
            step=1,
            key="abate_term",
        )

    abatement = calculate_city_tax_abatement(
        assessed_value=assessed_value,
        base_assessed_value=0,  # Use full value for abatement calc
        abatement_pct=abatement_pct,
        affordable_units=affordable_units,
        total_units=total_units,
        term_years=term_years,
    )

    with col3:
        st.markdown(f"""
        **Calculation:**
        - City Rate: {abatement.city_rate:.4%}
        - Affordable Share: {abatement.affordable_share:.1%}
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
    )

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
