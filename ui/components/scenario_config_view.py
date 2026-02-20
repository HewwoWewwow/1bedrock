"""Scenario configuration UI component."""

import streamlit as st
from typing import Tuple, Optional

from src.models.scenario_config import (
    ModelMode, ProjectType, TIFTreatment, TIFConfig,
    ScenarioInputs, ModelConfig, SharedInputs,
)
from src.models.incentives import IncentiveTier, TIER_REQUIREMENTS


def render_mode_selector() -> ModelMode:
    """Render the model mode selector and return selected mode."""
    st.subheader("Analysis Mode")

    mode = st.radio(
        "Select analysis mode",
        options=[ModelMode.SINGLE_PROJECT, ModelMode.COMPARISON],
        format_func=lambda x: {
            ModelMode.SINGLE_PROJECT: "Single Project",
            ModelMode.COMPARISON: "Comparison (Market vs Mixed-Income)",
        }[x],
        key="model_mode",
        horizontal=True,
        help="Single Project: Analyze one scenario. Comparison: Compare market-rate vs mixed-income side-by-side."
    )

    return mode


def render_incentive_config(prefix: str = "mixed") -> dict:
    """Render the incentive configuration panel.

    This is the main configuration for mixed-income scenarios, including:
    - Tier selection with editable affordable % and AMI for each tier
    - Incentive toggles
    - Calculated TIF lump sum

    Args:
        prefix: Session state key prefix

    Returns:
        Dict with all incentive configuration values
    """
    st.markdown("#### Select Incentive Tier")
    st.caption("Each tier has editable affordability requirements. Select the tier to use.")

    # Get target units for calculations
    target_units = st.session_state.get("target_units", 200)

    # AMI options
    ami_options = ["30%", "50%", "60%", "80%"]

    # Initialize tier values from defaults if not set
    for tier_num in [1, 2, 3]:
        tier_enum = IncentiveTier(tier_num)
        tier_reqs = TIER_REQUIREMENTS[tier_enum]
        default_pct = int(tier_reqs["affordable_pct"] * 100)
        default_ami = str(tier_reqs["ami_level"])

        if f"{prefix}_tier{tier_num}_pct" not in st.session_state:
            st.session_state[f"{prefix}_tier{tier_num}_pct"] = default_pct
        if f"{prefix}_tier{tier_num}_ami" not in st.session_state:
            st.session_state[f"{prefix}_tier{tier_num}_ami"] = default_ami

    # Tier descriptions
    tier_names = {
        1: "Tier 1 - Deep Affordability",
        2: "Tier 2 - Moderate (More Units)",
        3: "Tier 3 - Moderate (Balanced)",
    }

    # Create columns for each tier
    cols = st.columns(3)

    for i, tier_num in enumerate([1, 2, 3]):
        with cols[i]:
            # Tier header with radio selection
            is_selected = st.session_state.get("selected_tier", 2) == tier_num

            # Selection radio (using container for styling)
            if st.button(
                f"{'✓ ' if is_selected else '○ '}{tier_names[tier_num]}",
                key=f"{prefix}_select_tier_{tier_num}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
            ):
                st.session_state["selected_tier"] = tier_num
                st.rerun()

            # Editable inputs for this tier
            # Note: value comes from session state (pre-initialized above), so don't set value param
            tier_pct = st.number_input(
                "Affordable %",
                min_value=0,
                max_value=50,
                step=5,
                key=f"{prefix}_tier{tier_num}_pct",
                label_visibility="visible",
            )

            # Note: selectbox value comes from session state (pre-initialized above)
            tier_ami = st.selectbox(
                "AMI Level",
                options=ami_options,
                key=f"{prefix}_tier{tier_num}_ami",
                label_visibility="visible",
            )

            # Show unit counts for this tier
            tier_affordable = int(target_units * tier_pct / 100)
            tier_market = target_units - tier_affordable
            st.caption(f"{tier_affordable} affordable / {tier_market} market")

    # Get selected tier values
    selected_tier = st.session_state.get("selected_tier", 2)
    affordable_pct = st.session_state.get(f"{prefix}_tier{selected_tier}_pct", 20)
    ami_level = st.session_state.get(f"{prefix}_tier{selected_tier}_ami", "50%")

    # Store for use elsewhere
    st.session_state["affordable_pct"] = float(affordable_pct)
    st.session_state["ami_level"] = ami_level

    # Get tier enum for later use
    tier = IncentiveTier(selected_tier)

    # Show selected tier summary
    affordable_units = int(target_units * affordable_pct / 100)
    market_units = target_units - affordable_units

    st.divider()
    st.markdown(f"**Selected: {tier_names[selected_tier]}**")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Units", f"{target_units}")
    with col2:
        st.metric("Affordable Units", f"{affordable_units}")
    with col3:
        st.metric("Market Units", f"{market_units}")

    st.divider()

    # Incentive toggles
    st.markdown("#### Incentive Selection")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Fee Waivers**")
        smart_fee_waiver = st.checkbox(
            "SMART Fee Waiver",
            value=st.session_state.get("smart_fee_waiver", True),
            key="smart_fee_waiver",
            help="Waives development fees for affordable units"
        )

    with col2:
        st.markdown("**TIF Options**")
        tif_lump_sum = st.checkbox(
            "TIF Lump Sum (Upfront)",
            value=st.session_state.get("tif_lump_sum", True),
            key="tif_lump_sum",
            help="Receive TIF as upfront capital grant"
        )
        tif_stream = st.checkbox(
            "TIF Stream (Annual)",
            value=st.session_state.get("tif_stream", False),
            key="tif_stream",
            help="Receive annual TIF payments over time"
        )

        # Mutual exclusivity warning
        if tif_lump_sum and tif_stream:
            st.warning("Select only one TIF option")

    with col3:
        st.markdown("**Other Incentives**")
        tax_abatement = st.checkbox(
            "Tax Abatement",
            value=st.session_state.get("tax_abatement", False),
            key="tax_abatement",
            help="Property tax abatement on affordable units"
        )
        interest_buydown = st.checkbox(
            "Interest Buydown",
            value=st.session_state.get("interest_buydown", False),
            key="interest_buydown",
            help="Reduced interest rate on construction loan"
        )

    # Determine TIF treatment
    if tif_lump_sum:
        tif_treatment = TIFTreatment.LUMP_SUM_CAPITAL
    elif tif_stream:
        tif_treatment = TIFTreatment.TIF_STREAM
    elif tax_abatement:
        tif_treatment = TIFTreatment.TAX_ABATEMENT
    else:
        tif_treatment = TIFTreatment.NONE

    st.session_state["scenario_b_tif_treatment"] = tif_treatment

    # If TIF lump sum is selected, show note about Property Tax tab
    if tif_lump_sum:
        st.info("TIF Lump Sum is configured on the **Property Tax** tab under **TIF Lump Sum Analysis**.")

    # Get calculated TIF from Property Tax tab (if available)
    calculated_tif = st.session_state.get("calculated_tif_lump_sum", 0)
    if tif_lump_sum:
        st.session_state["scenario_b_lump_sum_amount"] = int(calculated_tif)

    return {
        "tier": selected_tier,
        "affordable_pct": affordable_pct / 100.0,  # Return as decimal
        "ami_level": ami_level,
        "smart_fee_waiver": smart_fee_waiver,
        "tif_lump_sum": tif_lump_sum,
        "tif_stream": tif_stream,
        "tax_abatement": tax_abatement,
        "interest_buydown": interest_buydown,
        "tif_treatment": tif_treatment,
        "calculated_tif": calculated_tif,
    }


def render_single_project_config() -> Tuple[ScenarioInputs, ProjectType]:
    """Render configuration for single project mode.

    Returns:
        Tuple of (ScenarioInputs, ProjectType)
    """
    st.subheader("Project Type")

    project_type = st.radio(
        "Select project type",
        options=[ProjectType.MARKET_RATE, ProjectType.MIXED_INCOME],
        format_func=lambda x: {
            ProjectType.MARKET_RATE: "Market Rate (No Affordable Units)",
            ProjectType.MIXED_INCOME: "Mixed Income (With Affordable Units)",
        }[x],
        key="single_project_type",
        horizontal=True,
    )

    is_market = project_type == ProjectType.MARKET_RATE

    if is_market:
        st.info("Market-rate project with no affordable units or incentives. Configure project details on the **Project Inputs** tab.")

        # Build minimal scenario
        scenario = ScenarioInputs(
            name="Market Rate",
            total_units=st.session_state.get("target_units", 200),
            affordable_pct=0.0,
            ami_level="50%",
            smart_fee_waiver=False,
            tif_config=TIFConfig(treatment=TIFTreatment.NONE),
        )
    else:
        st.divider()
        st.subheader("Mixed-Income Configuration")

        # Render full incentive configuration
        config = render_incentive_config(prefix="single")

        # Build scenario from config
        tif_config = TIFConfig(
            treatment=config["tif_treatment"],
            lump_sum_amount=int(config["calculated_tif"]) if config["tif_lump_sum"] else 0,
        )

        scenario = ScenarioInputs(
            name="Mixed Income",
            total_units=st.session_state.get("target_units", 200),
            affordable_pct=config["affordable_pct"],
            ami_level=config["ami_level"],
            smart_fee_waiver=config["smart_fee_waiver"],
            tif_config=tif_config,
        )

    return scenario, project_type


def render_comparison_config() -> Tuple[ScenarioInputs, ScenarioInputs]:
    """Render configuration for comparison mode.

    Returns:
        Tuple of (scenario_a, scenario_b)
    """
    # Get shared unit count
    target_units = st.session_state.get("target_units", 200)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Scenario A: Market Rate")
        st.info(f"""
        **{target_units} units** at market rate
        - No affordable units
        - No incentives
        - No TIF
        """)

        # Build market scenario
        scenario_a = ScenarioInputs(
            name="Market Rate",
            total_units=target_units,
            affordable_pct=0.0,
            ami_level="50%",
            smart_fee_waiver=False,
            tif_config=TIFConfig(treatment=TIFTreatment.NONE),
        )

    with col2:
        st.subheader("Scenario B: Mixed Income")

        # Render full incentive configuration
        config = render_incentive_config(prefix="scenario_b")

        # Build mixed-income scenario from config
        tif_config = TIFConfig(
            treatment=config["tif_treatment"],
            lump_sum_amount=int(config["calculated_tif"]) if config["tif_lump_sum"] else 0,
        )

        scenario_b = ScenarioInputs(
            name="Mixed Income",
            total_units=target_units,
            affordable_pct=config["affordable_pct"],
            ami_level=config["ami_level"],
            smart_fee_waiver=config["smart_fee_waiver"],
            tif_config=tif_config,
        )

    return scenario_a, scenario_b


def render_scenario_summary(scenario: ScenarioInputs) -> None:
    """Render a compact summary of a scenario configuration."""
    st.markdown(f"**{scenario.name}**")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Units", f"{scenario.total_units}")
    with col2:
        st.metric("Affordable", f"{scenario.affordable_pct:.0%}")
    with col3:
        tif_label = {
            TIFTreatment.NONE: "None",
            TIFTreatment.LUMP_SUM_CAPITAL: "Lump Sum",
            TIFTreatment.TAX_ABATEMENT: "Abatement",
            TIFTreatment.TIF_STREAM: "TIF Stream",
        }[scenario.tif_config.treatment]
        st.metric("TIF", tif_label)


def get_model_config_from_session() -> ModelConfig:
    """Build ModelConfig from session state."""
    mode = st.session_state.get("model_mode", ModelMode.COMPARISON)
    target_units = st.session_state.get("target_units", 200)

    config = ModelConfig(mode=mode)

    if mode == ModelMode.SINGLE_PROJECT:
        project_type = st.session_state.get("single_project_type", ProjectType.MIXED_INCOME)
        config.single_project_type = project_type

        if project_type == ProjectType.MARKET_RATE:
            config.single_scenario = ScenarioInputs(
                name="Market Rate",
                total_units=target_units,
                affordable_pct=0.0,
                ami_level="50%",
                smart_fee_waiver=False,
                tif_config=TIFConfig(treatment=TIFTreatment.NONE),
            )
        else:
            # Mixed income - get values from session state
            affordable_pct_raw = st.session_state.get("affordable_pct", 20.0)
            affordable_pct = affordable_pct_raw / 100.0 if affordable_pct_raw > 1 else affordable_pct_raw

            tif_treatment = st.session_state.get("scenario_b_tif_treatment", TIFTreatment.NONE)

            config.single_scenario = ScenarioInputs(
                name="Mixed Income",
                total_units=target_units,
                affordable_pct=affordable_pct,
                ami_level=st.session_state.get("ami_level", "50%"),
                smart_fee_waiver=st.session_state.get("smart_fee_waiver", True),
                tif_config=TIFConfig(
                    treatment=tif_treatment,
                    lump_sum_amount=st.session_state.get("scenario_b_lump_sum_amount", 0),
                ),
            )
    else:
        # Comparison mode
        affordable_pct_raw = st.session_state.get("affordable_pct", 20.0)
        affordable_pct = affordable_pct_raw / 100.0 if affordable_pct_raw > 1 else affordable_pct_raw

        tif_treatment = st.session_state.get("scenario_b_tif_treatment", TIFTreatment.NONE)

        # Scenario A: Market Rate
        config.scenario_a = ScenarioInputs(
            name="Market Rate",
            total_units=target_units,
            affordable_pct=0.0,
            ami_level="50%",
            smart_fee_waiver=False,
            tif_config=TIFConfig(treatment=TIFTreatment.NONE),
        )

        # Scenario B: Mixed Income
        config.scenario_b = ScenarioInputs(
            name="Mixed Income",
            total_units=target_units,
            affordable_pct=affordable_pct,
            ami_level=st.session_state.get("ami_level", "50%"),
            smart_fee_waiver=st.session_state.get("smart_fee_waiver", True),
            tif_config=TIFConfig(
                treatment=tif_treatment,
                lump_sum_amount=st.session_state.get("scenario_b_lump_sum_amount", 0),
            ),
        )

    return config
