"""Scenario configuration UI component."""

import streamlit as st
from typing import Tuple

from src.models.scenario_config import (
    ModelMode, ProjectType, TIFTreatment, TIFConfig,
    ScenarioInputs, ModelConfig, SharedInputs,
    create_default_market_scenario, create_default_mixed_income_scenario,
)


def render_mode_selector() -> ModelMode:
    """Render the model mode selector and return selected mode."""
    st.subheader("Analysis Mode")

    mode = st.radio(
        "Select analysis mode",
        options=[ModelMode.SINGLE_PROJECT, ModelMode.COMPARISON],
        format_func=lambda x: {
            ModelMode.SINGLE_PROJECT: "Single Project",
            ModelMode.COMPARISON: "Comparison (Side-by-Side)",
        }[x],
        key="model_mode",
        horizontal=True,
        help="Single Project: Analyze one scenario. Comparison: Compare two scenarios side-by-side."
    )

    return mode


def render_tif_config(prefix: str, default_treatment: TIFTreatment = TIFTreatment.NONE) -> TIFConfig:
    """Render TIF configuration inputs.

    Args:
        prefix: Key prefix for session state (e.g., 'scenario_a' or 'scenario_b')
        default_treatment: Default TIF treatment

    Returns:
        TIFConfig with current settings
    """
    st.markdown("**TIF / Incentive Treatment**")

    treatment = st.selectbox(
        "TIF Treatment",
        options=[t for t in TIFTreatment],
        format_func=lambda x: {
            TIFTreatment.NONE: "None - No TIF Benefits",
            TIFTreatment.LUMP_SUM_CAPITAL: "Lump Sum - Upfront Capital Grant",
            TIFTreatment.TAX_ABATEMENT: "Tax Abatement - Reduced Property Taxes",
            TIFTreatment.TIF_STREAM: "TIF Stream - Increment Reimbursement",
        }[x],
        key=f"{prefix}_tif_treatment",
        index=[t for t in TIFTreatment].index(default_treatment),
    )

    config = TIFConfig(treatment=treatment)

    if treatment == TIFTreatment.LUMP_SUM_CAPITAL:
        st.caption("Upfront capital grant reduces equity needed. Project then pays standard property taxes.")

        col1, col2 = st.columns(2)
        with col1:
            input_method = st.radio(
                "Input Method",
                options=["Dollar Amount", "% of TDC"],
                key=f"{prefix}_lump_sum_method",
                horizontal=True,
            )

        with col2:
            if input_method == "Dollar Amount":
                config.lump_sum_amount = st.number_input(
                    "Lump Sum Amount ($)",
                    min_value=0,
                    max_value=50_000_000,
                    value=st.session_state.get(f"{prefix}_lump_sum_amount", 0),
                    step=100_000,
                    format="%d",
                    key=f"{prefix}_lump_sum_amount",
                )
                config.use_lump_sum_pct = False
            else:
                config.lump_sum_pct_of_tdc = st.slider(
                    "Lump Sum (% of TDC)",
                    min_value=0.0,
                    max_value=30.0,
                    value=st.session_state.get(f"{prefix}_lump_sum_pct", 10.0),
                    step=1.0,
                    format="%.0f%%",
                    key=f"{prefix}_lump_sum_pct",
                ) / 100
                config.use_lump_sum_pct = True

    elif treatment == TIFTreatment.TAX_ABATEMENT:
        st.caption("Property taxes are reduced for a period. No upfront capital benefit.")

        col1, col2 = st.columns(2)
        with col1:
            config.abatement_pct = st.slider(
                "Abatement Percentage",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.get(f"{prefix}_abatement_pct", 100.0),
                step=5.0,
                format="%.0f%%",
                key=f"{prefix}_abatement_pct",
            ) / 100

        with col2:
            config.abatement_years = st.number_input(
                "Abatement Duration (years)",
                min_value=0,
                max_value=30,
                value=st.session_state.get(f"{prefix}_abatement_years", 10),
                step=1,
                key=f"{prefix}_abatement_years",
            )

        config.abatement_participating_only = st.checkbox(
            "Participating entities only",
            value=st.session_state.get(f"{prefix}_abatement_participating", True),
            key=f"{prefix}_abatement_participating",
            help="If checked, only TIF-participating entities provide abatement"
        )

    elif treatment == TIFTreatment.TIF_STREAM:
        st.caption("Project pays full property taxes, but receives reimbursement of the tax increment from participating entities.")

        col1, col2 = st.columns(2)
        with col1:
            config.stream_pct_of_increment = st.slider(
                "Capture Rate (% of Increment)",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.get(f"{prefix}_stream_pct", 100.0),
                step=5.0,
                format="%.0f%%",
                key=f"{prefix}_stream_pct",
                help="Percentage of the tax increment that is reimbursed"
            ) / 100

        with col2:
            config.stream_years = st.number_input(
                "TIF Term (years)",
                min_value=0,
                max_value=30,
                value=st.session_state.get(f"{prefix}_stream_years", 20),
                step=1,
                key=f"{prefix}_stream_years",
            )

    if treatment != TIFTreatment.NONE:
        config.tif_start_delay_months = st.number_input(
            "Delay from Stabilization (months)",
            min_value=0,
            max_value=24,
            value=st.session_state.get(f"{prefix}_tif_delay", 0),
            step=1,
            key=f"{prefix}_tif_delay",
            help="Months after stabilization before TIF benefits begin"
        )

    return config


def render_scenario_inputs(
    prefix: str,
    scenario_name: str,
    is_market_default: bool = False,
    shared_units: int = 200,
) -> ScenarioInputs:
    """Render inputs for a single scenario.

    Args:
        prefix: Key prefix for session state
        scenario_name: Display name for the scenario
        is_market_default: If True, defaults to market rate settings
        shared_units: Base unit count from shared inputs

    Returns:
        ScenarioInputs with current settings
    """
    # Scenario name
    name = st.text_input(
        "Scenario Name",
        value=st.session_state.get(f"{prefix}_name", scenario_name),
        key=f"{prefix}_name",
    )

    # Unit configuration
    st.markdown("**Unit Configuration**")

    total_units = st.number_input(
        "Total Units",
        min_value=10,
        max_value=1000,
        value=st.session_state.get(f"{prefix}_units", shared_units),
        step=10,
        key=f"{prefix}_units",
        help="Unit count should be determined from site/massing analysis"
    )

    st.caption("Unit count should reflect site constraints, zoning, parking requirements, "
              "and any applicable bonuses. Use separate test-fit analysis to determine feasible unit count.")

    # Affordable configuration
    st.markdown("**Affordable Housing**")

    default_affordable = 0.0 if is_market_default else 20.0
    affordable_pct = st.slider(
        "Affordable Percentage",
        min_value=0.0,
        max_value=100.0,
        value=st.session_state.get(f"{prefix}_affordable_pct", default_affordable),
        step=5.0,
        format="%.0f%%",
        key=f"{prefix}_affordable_pct",
    ) / 100

    affordable_units = int(total_units * affordable_pct)
    market_units = total_units - affordable_units

    if affordable_pct > 0:
        col1, col2 = st.columns(2)
        with col1:
            ami_level = st.selectbox(
                "AMI Level",
                options=["30%", "50%", "60%", "80%", "100%"],
                index=1,  # Default to 50%
                key=f"{prefix}_ami_level",
            )
        with col2:
            st.caption(f"Affordable Units: **{affordable_units}**")
            st.caption(f"Market Units: **{market_units}**")
    else:
        ami_level = "50%"
        st.caption("No affordable units in this scenario")

    st.divider()

    # TIF Configuration
    default_tif = TIFTreatment.NONE if is_market_default else TIFTreatment.TIF_STREAM
    tif_config = render_tif_config(prefix, default_tif)

    st.divider()

    # Other incentives
    st.markdown("**Other Incentives**")
    smart_fee_waiver = st.checkbox(
        "SMART Fee Waiver",
        value=st.session_state.get(f"{prefix}_smart_fee", not is_market_default),
        key=f"{prefix}_smart_fee",
        help="Fee waiver for affordable housing projects"
    )

    # Build and return ScenarioInputs
    return ScenarioInputs(
        name=name,
        total_units=total_units,
        affordable_pct=affordable_pct,
        ami_level=ami_level,
        tif_config=tif_config,
        smart_fee_waiver=smart_fee_waiver,
    )


def render_single_project_config() -> Tuple[ScenarioInputs, ProjectType]:
    """Render configuration for single project mode.

    Returns:
        Tuple of (ScenarioInputs, ProjectType)
    """
    st.subheader("Project Configuration")

    # Project type selector
    project_type = st.radio(
        "Project Type",
        options=[ProjectType.MARKET_RATE, ProjectType.MIXED_INCOME],
        format_func=lambda x: {
            ProjectType.MARKET_RATE: "Market Rate",
            ProjectType.MIXED_INCOME: "Mixed Income (with Affordable)",
        }[x],
        key="single_project_type",
        horizontal=True,
    )

    is_market = project_type == ProjectType.MARKET_RATE
    scenario = render_scenario_inputs(
        prefix="single",
        scenario_name="Market Rate" if is_market else "Mixed Income",
        is_market_default=is_market,
    )

    return scenario, project_type


def render_comparison_config() -> Tuple[ScenarioInputs, ScenarioInputs]:
    """Render configuration for comparison mode.

    Returns:
        Tuple of (scenario_a, scenario_b)
    """
    st.subheader("Comparison Configuration")
    st.caption("Configure two scenarios to compare side-by-side")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Scenario A")
        scenario_a = render_scenario_inputs(
            prefix="scenario_a",
            scenario_name="Market Rate",
            is_market_default=True,
        )

    with col2:
        st.markdown("### Scenario B")
        scenario_b = render_scenario_inputs(
            prefix="scenario_b",
            scenario_name="Mixed Income",
            is_market_default=False,
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

    config = ModelConfig(mode=mode)

    if mode == ModelMode.SINGLE_PROJECT:
        # Get single project config
        project_type = st.session_state.get("single_project_type", ProjectType.MIXED_INCOME)
        config.single_project_type = project_type

        # Build scenario from session state
        config.single_scenario = ScenarioInputs(
            name=st.session_state.get("single_name", "Project"),
            total_units=st.session_state.get("single_units", 200),
            affordable_pct=st.session_state.get("single_affordable_pct", 0.0) / 100,
            ami_level=st.session_state.get("single_ami_level", "50%"),
            smart_fee_waiver=st.session_state.get("single_smart_fee", False),
            tif_config=TIFConfig(
                treatment=st.session_state.get("single_tif_treatment", TIFTreatment.NONE),
                lump_sum_amount=st.session_state.get("single_lump_sum_amount", 0),
                lump_sum_pct_of_tdc=st.session_state.get("single_lump_sum_pct", 0) / 100,
                abatement_pct=st.session_state.get("single_abatement_pct", 0) / 100,
                abatement_years=st.session_state.get("single_abatement_years", 0),
                stream_pct_of_increment=st.session_state.get("single_stream_pct", 100) / 100,
                stream_years=st.session_state.get("single_stream_years", 20),
            ),
        )
    else:
        # Build scenario A
        config.scenario_a = ScenarioInputs(
            name=st.session_state.get("scenario_a_name", "Market Rate"),
            total_units=st.session_state.get("scenario_a_units", 200),
            affordable_pct=st.session_state.get("scenario_a_affordable_pct", 0.0) / 100,
            ami_level=st.session_state.get("scenario_a_ami_level", "50%"),
            smart_fee_waiver=st.session_state.get("scenario_a_smart_fee", False),
            tif_config=TIFConfig(
                treatment=st.session_state.get("scenario_a_tif_treatment", TIFTreatment.NONE),
                lump_sum_amount=st.session_state.get("scenario_a_lump_sum_amount", 0),
                lump_sum_pct_of_tdc=st.session_state.get("scenario_a_lump_sum_pct", 0) / 100,
                abatement_pct=st.session_state.get("scenario_a_abatement_pct", 0) / 100,
                abatement_years=st.session_state.get("scenario_a_abatement_years", 0),
                stream_pct_of_increment=st.session_state.get("scenario_a_stream_pct", 100) / 100,
                stream_years=st.session_state.get("scenario_a_stream_years", 20),
            ),
        )

        # Build scenario B
        config.scenario_b = ScenarioInputs(
            name=st.session_state.get("scenario_b_name", "Mixed Income"),
            total_units=st.session_state.get("scenario_b_units", 200),
            affordable_pct=st.session_state.get("scenario_b_affordable_pct", 20.0) / 100,
            ami_level=st.session_state.get("scenario_b_ami_level", "50%"),
            smart_fee_waiver=st.session_state.get("scenario_b_smart_fee", True),
            tif_config=TIFConfig(
                treatment=st.session_state.get("scenario_b_tif_treatment", TIFTreatment.TIF_STREAM),
                lump_sum_amount=st.session_state.get("scenario_b_lump_sum_amount", 0),
                lump_sum_pct_of_tdc=st.session_state.get("scenario_b_lump_sum_pct", 0) / 100,
                abatement_pct=st.session_state.get("scenario_b_abatement_pct", 0) / 100,
                abatement_years=st.session_state.get("scenario_b_abatement_years", 0),
                stream_pct_of_increment=st.session_state.get("scenario_b_stream_pct", 100) / 100,
                stream_years=st.session_state.get("scenario_b_stream_years", 20),
            ),
        )

    return config
