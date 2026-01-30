"""Input components for the Streamlit UI."""

import streamlit as st
from typing import Dict, Any


def render_sidebar_inputs() -> Dict[str, Any]:
    """Render sidebar inputs and return values.

    Returns:
        Dictionary of input values from sidebar.
    """
    st.sidebar.header("Quick Settings")

    inputs = {}

    st.sidebar.subheader("Project Size")
    inputs["target_units"] = st.sidebar.slider(
        "Target Units",
        min_value=50, max_value=500, value=200, step=10,
        help="Total number of units in the development"
    )

    inputs["construction_type"] = st.sidebar.selectbox(
        "Construction Type",
        options=["garden", "small_courtyard", "corridor", "podium_midrise_5over1", "wrap", "highrise"],
        index=3,  # Default to Podium 5-over-1
        format_func=lambda x: {
            "garden": "Garden (2-3 stories)",
            "small_courtyard": "Small Courtyard (2-4 stories)",
            "corridor": "Double-Loaded Corridor (4-7 stories)",
            "podium_midrise_5over1": "Podium Midrise 5-over-1",
            "wrap": "Wrap (around garage)",
            "highrise": "High-Rise (15+ stories)",
        }.get(x, x.replace("_", " ").title())
    )

    st.sidebar.subheader("Affordability")
    inputs["affordable_pct"] = st.sidebar.slider(
        "Affordable %",
        min_value=0.0, max_value=50.0, value=20.0, step=5.0,
        format="%.0f%%",
        help="Percentage of units designated as affordable"
    ) / 100

    inputs["ami_level"] = st.sidebar.selectbox(
        "AMI Level",
        options=["30%", "50%", "60%", "80%", "100%"],
        index=1
    )

    inputs["selected_tier"] = st.sidebar.radio(
        "Incentive Tier",
        options=[1, 2, 3],
        index=1,
        format_func=lambda x: f"Tier {x}"
    )

    st.sidebar.subheader("Incentives")
    inputs["smart_fee_waiver"] = st.sidebar.checkbox("SMART Fee Waiver", value=True)
    inputs["tif_stream"] = st.sidebar.checkbox("TIF Stream", value=True)
    inputs["tif_lump_sum"] = st.sidebar.checkbox("TIF Lump Sum", value=False)
    inputs["tax_abatement"] = st.sidebar.checkbox("Tax Abatement", value=False)
    inputs["interest_buydown"] = st.sidebar.checkbox("Interest Buydown", value=False)

    return inputs


def render_project_inputs() -> Dict[str, Any]:
    """Render detailed project inputs.

    Returns:
        Dictionary of project input values.
    """
    inputs = {}

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Timing")
        inputs["predevelopment_months"] = st.number_input(
            "Predevelopment (months)", 6, 36, 18
        )
        inputs["construction_months"] = st.number_input(
            "Construction (months)", 12, 48, 24
        )
        inputs["leaseup_months"] = st.number_input(
            "Lease-up (months)", 6, 24, 12
        )
        inputs["operations_months"] = st.number_input(
            "Operations (months)", 12, 36, 12
        )

    with col2:
        st.subheader("Land & Construction")
        inputs["land_cost_per_acre"] = st.number_input(
            "Land Cost/Acre ($)", 100_000, 5_000_000, 1_000_000,
            step=100_000, format="%d"
        )
        inputs["hard_cost_per_unit"] = st.number_input(
            "Hard Cost/Unit ($)", 100_000, 300_000, 155_000,
            step=5_000, format="%d"
        )
        inputs["soft_cost_pct"] = st.slider(
            "Soft Cost %", 0.0, 60.0, 30.0, 1.0,
            format="%.0f%%"
        ) / 100

    with col3:
        st.subheader("Financing")
        inputs["construction_rate"] = st.slider(
            "Construction Rate", 0.0, 18.0, 7.5, 0.5,
            format="%.1f%%"
        ) / 100
        inputs["perm_rate"] = st.slider(
            "Perm Rate", 0.0, 18.0, 6.0, 0.5,
            format="%.1f%%"
        ) / 100
        inputs["exit_cap_rate"] = st.slider(
            "Exit Cap Rate", 3.0, 15.0, 5.5, 0.5,
            format="%.1f%%"
        ) / 100

    return inputs


def render_incentive_inputs() -> Dict[str, Any]:
    """Render incentive configuration inputs.

    Returns:
        Dictionary of incentive input values.
    """
    inputs = {}

    st.subheader("Incentive Configuration")

    col1, col2 = st.columns(2)

    with col1:
        inputs["selected_tier"] = st.radio(
            "Incentive Tier",
            options=[1, 2, 3],
            index=1,
            format_func=lambda x: f"Tier {x}",
            horizontal=True
        )

        st.caption("""
        - **Tier 1**: 5% affordable at 30% AMI, 10-year term
        - **Tier 2**: 20% affordable at 50% AMI, 20-year term
        - **Tier 3**: 10% affordable at 50% AMI, 20-year term
        """)

    with col2:
        st.write("**Enable Incentives:**")
        inputs["smart_fee_waiver"] = st.checkbox("SMART Fee Waiver", value=True)
        inputs["tif_stream"] = st.checkbox("TIF Stream", value=True)
        inputs["tif_lump_sum"] = st.checkbox("TIF Lump Sum", value=False)
        inputs["tax_abatement"] = st.checkbox("Tax Abatement", value=False)
        inputs["interest_buydown"] = st.checkbox("Interest Buydown", value=False)

        if inputs["tif_lump_sum"] and inputs["tif_stream"]:
            st.warning("TIF Lump Sum and Stream are mutually exclusive. Lump Sum will be used.")

        if (inputs["tif_lump_sum"] or inputs["tif_stream"]) and inputs["tax_abatement"]:
            st.warning("TIF and Tax Abatement are mutually exclusive. TIF will take precedence.")

    return inputs
