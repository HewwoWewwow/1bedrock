"""Main Streamlit application for Austin TIF Model."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
from datetime import date

from src.models.project import ProjectInputs, Scenario, UnitMixEntry, TIFStartTiming
from src.models.lookups import ConstructionType, DEFAULT_TAX_RATES
from src.models.incentives import IncentiveTier, IncentiveToggles, get_tier_config
from src.calculations.dcf import run_dcf
from src.calculations.units import allocate_units, get_total_units, get_total_affordable_units
from src.calculations.revenue import calculate_gpr
from src.calculations.metrics import calculate_metrics, compare_scenarios
from src.scenarios import run_scenario_matrix, generate_combinations
from src.calculations.detailed_cashflow import generate_detailed_cash_flow
from src.calculations.sources_uses import calculate_sources_uses
from src.calculations.property_tax import get_austin_tax_stack, analyze_tif
from ui.components.detailed_cashflow_view import (
    render_sources_uses, render_detailed_cashflow_table,
    render_irr_summary, render_property_tax_engine, render_sensitivity_analysis,
    render_deal_summary_header, export_cashflow_to_csv, render_exit_waterfall,
    render_operating_statement
)
from ui.components.unit_mix import (
    render_unit_mix_tab, get_unit_mix_from_session_state, get_efficiency
)
from ui.components.sources_uses_detailed_view import (
    render_sources_uses_inputs, render_sources_uses_detailed_table
)
from src.calculations.sources_uses_detailed import (
    LandCostMethod, calculate_sources_uses_detailed
)
from src.models.scenario_config import (
    ModelMode, ProjectType, TIFTreatment, TIFConfig,
    ScenarioInputs, ModelConfig, SharedInputs,
)
from ui.components.scenario_config_view import (
    render_mode_selector, render_single_project_config, render_comparison_config,
    render_scenario_summary, get_model_config_from_session,
)

# Page configuration
st.set_page_config(
    page_title="Austin Affordable Housing Incentive Calculator",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .big-metric {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
        text-align: center;
    }
    .positive { color: #28a745; }
    .negative { color: #dc3545; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
</style>
""", unsafe_allow_html=True)


def _get_tif_treatment_str() -> str:
    """Get TIF treatment as string for cash flow engine.

    Reads from session state and converts TIFTreatment enum to string.
    """
    # Check if in comparison mode and which scenario's TIF to use
    mode = st.session_state.get("model_mode", ModelMode.COMPARISON)

    if mode == ModelMode.SINGLE_PROJECT:
        treatment = st.session_state.get("single_tif_treatment", TIFTreatment.NONE)
    else:
        # In comparison mode, use scenario B's TIF (the mixed income scenario)
        treatment = st.session_state.get("scenario_b_tif_treatment", TIFTreatment.NONE)

    # Convert enum to string
    if treatment == TIFTreatment.LUMP_SUM_CAPITAL:
        return "lump_sum"
    elif treatment == TIFTreatment.TAX_ABATEMENT:
        return "abatement"
    elif treatment == TIFTreatment.TIF_STREAM:
        return "stream"
    else:
        return "none"


def get_inputs_for_scenario(scenario: ScenarioInputs = None) -> ProjectInputs:
    """Build ProjectInputs from session state, optionally overriding with scenario-specific values.

    Args:
        scenario: Optional ScenarioInputs to override unit count, affordable %, etc.

    Returns:
        ProjectInputs configured for the scenario
    """
    # Get efficiency from construction type
    construction_type = st.session_state.get("construction_type", "podium_midrise_5over1")
    efficiency = get_efficiency(construction_type)

    # Unit mix from session state (using new component)
    unit_mix = get_unit_mix_from_session_state(efficiency)

    # Determine unit count and affordable % from scenario if provided
    if scenario:
        target_units = scenario.total_units
        affordable_pct = scenario.affordable_pct
        ami_level = scenario.ami_level
    else:
        target_units = st.session_state.get("target_units", 200)
        affordable_pct = st.session_state.get("affordable_pct", 20.0) / 100
        ami_level = st.session_state.get("ami_level", "50%")

    # Build incentive config
    incentive_config = None
    if scenario and scenario.tif_config.treatment != TIFTreatment.NONE:
        tier = IncentiveTier(st.session_state.get("selected_tier", 2))
        toggles = IncentiveToggles(
            smart_fee_waiver=scenario.smart_fee_waiver,
            tax_abatement=scenario.tif_config.treatment == TIFTreatment.TAX_ABATEMENT,
            tif_lump_sum=scenario.tif_config.treatment == TIFTreatment.LUMP_SUM_CAPITAL,
            tif_stream=scenario.tif_config.treatment == TIFTreatment.TIF_STREAM,
            interest_buydown=False,
        )
        incentive_config = get_tier_config(tier, toggles)
    elif st.session_state.get("run_mixed", True):
        tier = IncentiveTier(st.session_state.get("selected_tier", 2))
        toggles = IncentiveToggles(
            smart_fee_waiver=st.session_state.get("smart_fee_waiver", True),
            tax_abatement=st.session_state.get("tax_abatement", False),
            tif_lump_sum=st.session_state.get("tif_lump_sum", False),
            tif_stream=st.session_state.get("tif_stream", True),
            interest_buydown=st.session_state.get("interest_buydown", False),
        )
        incentive_config = get_tier_config(tier, toggles)

    return ProjectInputs(
        predevelopment_start=date(2026, 1, 1),
        predevelopment_months=st.session_state.get("predevelopment_months", 18),
        construction_months=st.session_state.get("construction_months", 24),
        leaseup_months=st.session_state.get("leaseup_months", 12),
        operations_months=st.session_state.get("operations_months", 12),
        land_cost_per_acre=st.session_state.get("land_cost_per_acre", 1_000_000),
        target_units=target_units,
        hard_cost_per_unit=st.session_state.get("hard_cost_per_unit", 155_000),
        soft_cost_pct=st.session_state.get("soft_cost_pct", 30.0) / 100,
        construction_type=ConstructionType(st.session_state.get("construction_type", "podium_midrise_5over1")),
        unit_mix=unit_mix,
        market_rent_psf=st.session_state.get("market_rent_psf", 2.50),
        vacancy_rate=st.session_state.get("vacancy_rate_pct", 6.0) / 100,
        leaseup_pace=st.session_state.get("leaseup_pace_pct", 8.0) / 100,
        max_occupancy=st.session_state.get("max_occupancy", 0.94),
        opex_utilities=st.session_state.get("opex_utilities", 1200),
        opex_management_pct=st.session_state.get("opex_management_pct", 5.0) / 100,
        opex_maintenance=st.session_state.get("opex_maintenance", 1500),
        opex_misc=st.session_state.get("opex_misc", 650),
        reserves_pct=st.session_state.get("reserves_pct", 0.02),
        market_rent_growth=st.session_state.get("market_rent_growth_pct", 2.0) / 100,
        affordable_rent_growth=st.session_state.get("affordable_rent_growth_pct", 1.0) / 100,
        opex_growth=st.session_state.get("opex_growth_pct", 3.0) / 100,
        property_tax_growth=st.session_state.get("property_tax_growth_pct", 2.0) / 100,
        construction_rate=st.session_state.get("construction_rate_pct", 7.5) / 100,
        construction_ltc=st.session_state.get("construction_ltc_pct", 65.0) / 100,
        perm_rate=st.session_state.get("perm_rate_pct", 6.0) / 100,
        perm_amort_years=st.session_state.get("perm_amort_years", 20),
        perm_ltv_max=st.session_state.get("perm_ltv_max_pct", 65.0) / 100,
        perm_dscr_min=st.session_state.get("perm_dscr_min", 1.25),
        existing_assessed_value=st.session_state.get("existing_assessed_value", 5_000_000),
        tax_rates=DEFAULT_TAX_RATES.copy(),
        exit_cap_rate=st.session_state.get("exit_cap_rate_pct", 5.5) / 100,
        affordable_pct=affordable_pct,
        ami_level=ami_level,
        incentive_config=incentive_config,
        tif_start_timing=TIFStartTiming.OPERATIONS,
    )


def get_session_state_inputs() -> ProjectInputs:
    """Build ProjectInputs from session state."""
    # Get efficiency from construction type
    construction_type = st.session_state.get("construction_type", "podium_midrise_5over1")
    efficiency = get_efficiency(construction_type)

    # Unit mix from session state (using new component)
    unit_mix = get_unit_mix_from_session_state(efficiency)

    # Build incentive config if in mixed-income mode
    incentive_config = None
    if st.session_state.get("run_mixed", True):
        tier = IncentiveTier(st.session_state.get("selected_tier", 2))
        toggles = IncentiveToggles(
            smart_fee_waiver=st.session_state.get("smart_fee_waiver", True),
            tax_abatement=st.session_state.get("tax_abatement", False),
            tif_lump_sum=st.session_state.get("tif_lump_sum", False),
            tif_stream=st.session_state.get("tif_stream", True),
            interest_buydown=st.session_state.get("interest_buydown", False),
        )
        incentive_config = get_tier_config(tier, toggles)

    return ProjectInputs(
        predevelopment_start=date(2026, 1, 1),
        predevelopment_months=st.session_state.get("predevelopment_months", 18),
        construction_months=st.session_state.get("construction_months", 24),
        leaseup_months=st.session_state.get("leaseup_months", 12),
        operations_months=st.session_state.get("operations_months", 12),
        land_cost_per_acre=st.session_state.get("land_cost_per_acre", 1_000_000),
        target_units=st.session_state.get("target_units", 200),
        hard_cost_per_unit=st.session_state.get("hard_cost_per_unit", 155_000),
        soft_cost_pct=st.session_state.get("soft_cost_pct", 30.0) / 100,
        construction_type=ConstructionType(st.session_state.get("construction_type", "podium_midrise_5over1")),
        unit_mix=unit_mix,
        market_rent_psf=st.session_state.get("market_rent_psf", 2.50),
        vacancy_rate=st.session_state.get("vacancy_rate_pct", 6.0) / 100,
        leaseup_pace=st.session_state.get("leaseup_pace_pct", 8.0) / 100,
        max_occupancy=st.session_state.get("max_occupancy", 0.94),
        opex_utilities=st.session_state.get("opex_utilities", 1200),
        opex_management_pct=st.session_state.get("opex_management_pct", 5.0) / 100,
        opex_maintenance=st.session_state.get("opex_maintenance", 1500),
        opex_misc=st.session_state.get("opex_misc", 650),
        reserves_pct=st.session_state.get("reserves_pct", 0.02),
        market_rent_growth=st.session_state.get("market_rent_growth_pct", 2.0) / 100,
        affordable_rent_growth=st.session_state.get("affordable_rent_growth_pct", 1.0) / 100,
        opex_growth=st.session_state.get("opex_growth_pct", 3.0) / 100,
        property_tax_growth=st.session_state.get("property_tax_growth_pct", 2.0) / 100,
        construction_rate=st.session_state.get("construction_rate_pct", 7.5) / 100,
        construction_ltc=st.session_state.get("construction_ltc_pct", 65.0) / 100,
        perm_rate=st.session_state.get("perm_rate_pct", 6.0) / 100,
        perm_amort_years=st.session_state.get("perm_amort_years", 20),
        perm_ltv_max=st.session_state.get("perm_ltv_max_pct", 65.0) / 100,
        perm_dscr_min=st.session_state.get("perm_dscr_min", 1.25),
        existing_assessed_value=st.session_state.get("existing_assessed_value", 5_000_000),
        tax_rates=DEFAULT_TAX_RATES.copy(),
        exit_cap_rate=st.session_state.get("exit_cap_rate_pct", 5.5) / 100,
        affordable_pct=st.session_state.get("affordable_pct", 20.0) / 100,
        ami_level=st.session_state.get("ami_level", "50%"),
        incentive_config=incentive_config,
        tif_start_timing=TIFStartTiming.OPERATIONS,
    )


def run_analysis(inputs: ProjectInputs, scenario_a: ScenarioInputs = None, scenario_b: ScenarioInputs = None):
    """Run DCF analysis for both scenarios.

    Args:
        inputs: Base ProjectInputs (used if scenarios not provided)
        scenario_a: Optional scenario A configuration (market rate default)
        scenario_b: Optional scenario B configuration (mixed income default)

    Returns:
        Tuple of (market_result, mixed_result, market_metrics, mixed_metrics, comparison)
    """
    # Build inputs for each scenario if provided
    if scenario_a:
        inputs_a = get_inputs_for_scenario(scenario_a)
    else:
        inputs_a = inputs

    if scenario_b:
        inputs_b = get_inputs_for_scenario(scenario_b)
    else:
        inputs_b = inputs

    # Market scenario (scenario A or base inputs with 0% affordable)
    market_result = run_dcf(inputs_a, Scenario.MARKET)
    market_allocs = allocate_units(
        inputs_a.target_units, inputs_a.unit_mix, 0.0, inputs_a.ami_level, inputs_a.market_rent_psf
    )
    market_gpr = calculate_gpr(market_allocs)
    market_metrics = calculate_metrics(
        market_result, get_total_units(market_allocs), 0, market_gpr.total_gpr_annual
    )

    # Mixed-income scenario (scenario B or base inputs with affordable %)
    mixed_result = run_dcf(inputs_b, Scenario.MIXED_INCOME)
    mixed_allocs = allocate_units(
        inputs_b.target_units, inputs_b.unit_mix, inputs_b.affordable_pct,
        inputs_b.ami_level, inputs_b.market_rent_psf
    )
    mixed_gpr = calculate_gpr(mixed_allocs)
    mixed_metrics = calculate_metrics(
        mixed_result, get_total_units(mixed_allocs),
        get_total_affordable_units(mixed_allocs), mixed_gpr.total_gpr_annual
    )

    comparison = compare_scenarios(market_metrics, mixed_metrics)

    return market_result, mixed_result, market_metrics, mixed_metrics, comparison


def currency_input(label: str, key: str, default: int, min_val: int = 0, max_val: int = 100_000_000) -> int:
    """Render a currency input with comma formatting.

    Args:
        label: Input label
        key: Session state key
        default: Default value
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        The numeric value entered
    """
    # Get current value from session state or use default
    current_val = st.session_state.get(key, default)

    # Format for display
    formatted = f"${current_val:,}"

    # Render text input
    user_input = st.text_input(label, value=formatted, key=f"{key}_text")

    # Parse the input - strip $ and commas
    try:
        cleaned = user_input.replace("$", "").replace(",", "").strip()
        parsed_val = int(float(cleaned)) if cleaned else default
        # Clamp to valid range
        parsed_val = max(min_val, min(max_val, parsed_val))
    except ValueError:
        parsed_val = current_val

    # Store in session state
    st.session_state[key] = parsed_val

    return parsed_val


def render_sidebar():
    """Render the sidebar with key inputs."""
    st.sidebar.header("Quick Settings")

    # Show current mode
    mode = st.session_state.get("model_mode", ModelMode.COMPARISON)
    if mode == ModelMode.SINGLE_PROJECT:
        project_type = st.session_state.get("single_project_type", ProjectType.MIXED_INCOME)
        st.sidebar.info(f"Mode: **Single Project** ({project_type.value.replace('_', ' ').title()})")
    else:
        st.sidebar.info("Mode: **Comparison** (Market vs Mixed Income)")

    st.sidebar.divider()

    # Try to show live metrics at the top
    try:
        inputs = get_session_state_inputs()
        market_result, mixed_result, market_metrics, mixed_metrics, comparison = run_analysis(inputs)

        # Display key metrics
        st.sidebar.subheader("Returns")

        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Market IRR", f"{market_metrics.levered_irr:.1%}")
        with col2:
            st.metric("Mixed IRR", f"{mixed_metrics.levered_irr:.1%}")

        # IRR difference
        irr_diff = comparison.irr_difference_bps
        if comparison.meets_target:
            st.sidebar.success(f"IRR Δ: **{irr_diff:+d} bps** ✓")
        else:
            st.sidebar.warning(f"IRR Δ: **{irr_diff:+d} bps**")

        # Calculate NPV (sum of discounted cash flows)
        def calc_npv(cash_flows, rate):
            monthly_rate = (1 + rate) ** (1/12) - 1
            npv = 0
            for i, cf in enumerate(cash_flows):
                npv += cf.levered_cf / ((1 + monthly_rate) ** i)
            return npv

        # NPV side by side with discount sliders underneath
        col1, col2 = st.sidebar.columns(2)
        with col1:
            market_discount = st.slider(
                "Disc %", min_value=0, max_value=30, value=15, step=1,
                key="market_discount_rate", format="%d%%"
            ) / 100.0
            market_npv = calc_npv(market_result.monthly_cash_flows, market_discount)
            st.metric("Market NPV", f"${market_npv/1e6:.1f}M")
        with col2:
            mixed_discount = st.slider(
                "Disc %", min_value=0, max_value=30, value=15, step=1,
                key="mixed_discount_rate", format="%d%%"
            ) / 100.0
            mixed_npv = calc_npv(mixed_result.monthly_cash_flows, mixed_discount)
            st.metric("Mixed NPV", f"${mixed_npv/1e6:.1f}M")

        st.sidebar.divider()

    except Exception:
        # Show placeholder on first run or if analysis fails
        st.sidebar.info("Adjust inputs to see returns")
        st.sidebar.divider()

    st.sidebar.subheader("Construction")
    st.sidebar.selectbox(
        "Construction Type",
        options=[ct.value for ct in ConstructionType],
        index=2,  # PODIUM_5OVER1
        key="construction_type",
        format_func=lambda x: x.replace("_", " ").title()
    )

    st.sidebar.subheader("Project Size")
    st.sidebar.slider(
        "Total Units",
        min_value=50, max_value=500, value=200, step=10,
        key="target_units",
        help="Total number of units in the development"
    )

    st.sidebar.slider(
        "Affordable %",
        min_value=0.0, max_value=50.0, value=20.0, step=5.0,
        key="affordable_pct",
        format="%.0f%%",
        help="Percentage of units designated as affordable"
    )

    st.sidebar.subheader("Incentive Tier")
    st.sidebar.radio(
        "Select Tier",
        options=[1, 2, 3],
        index=1,  # Tier 2
        key="selected_tier",
        format_func=lambda x: {
            1: "Tier 1: 5% @ 30% AMI",
            2: "Tier 2: 20% @ 50% AMI",
            3: "Tier 3: 10% @ 50% AMI",
        }[x]
    )

    st.sidebar.subheader("Incentives")
    st.sidebar.checkbox("SMART Fee Waiver", value=True, key="smart_fee_waiver")
    st.sidebar.checkbox("TIF Stream", value=True, key="tif_stream")
    st.sidebar.checkbox("TIF Lump Sum", value=False, key="tif_lump_sum")
    st.sidebar.checkbox("Tax Abatement", value=False, key="tax_abatement")
    st.sidebar.checkbox("Interest Buydown", value=False, key="interest_buydown")


def render_scenarios_tab():
    """Render the scenarios configuration tab."""
    st.header("Scenario Configuration")

    # Mode selector
    mode = render_mode_selector()

    st.divider()

    if mode == ModelMode.SINGLE_PROJECT:
        scenario, project_type = render_single_project_config()

        # Show summary
        st.divider()
        st.markdown("### Scenario Summary")
        render_scenario_summary(scenario)

    else:  # Comparison mode
        scenario_a, scenario_b = render_comparison_config()

        # Show comparison summary
        st.divider()
        st.markdown("### Comparison Summary")
        col1, col2 = st.columns(2)
        with col1:
            render_scenario_summary(scenario_a)
        with col2:
            render_scenario_summary(scenario_b)


def render_project_inputs_tab():
    """Render the project inputs tab."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Timing")
        st.number_input("Predevelopment (months)", 6, 36, 18, key="predevelopment_months")
        st.number_input("Construction (months)", 12, 48, 24, key="construction_months")
        st.number_input("Lease-up (months)", 6, 24, 12, key="leaseup_months")
        st.number_input("Operations (months)", 12, 120, 60, key="operations_months")

        st.subheader("Land & Construction")
        currency_input("Land Cost/Acre", "land_cost_per_acre", 1_000_000, 100_000, 10_000_000)
        currency_input("Hard Cost/Unit", "hard_cost_per_unit", 155_000, 100_000, 500_000)
        st.slider("Soft Cost % (of hard costs)", 0.0, 60.0, 30.0, 1.0, key="soft_cost_pct",
                 format="%.0f%%", help="As percentage of hard costs")

    with col2:
        st.subheader("Operations")
        st.slider("Vacancy Rate", 0.0, 30.0, 6.0, 1.0, key="vacancy_rate_pct", format="%.0f%%")
        st.slider("Lease-up Pace (monthly)", 0.0, 20.0, 8.0, 1.0, key="leaseup_pace_pct",
                 format="%.0f%%", help="Monthly absorption during lease-up")

        st.subheader("Operating Expenses")
        currency_input("Utilities ($/unit/yr)", "opex_utilities", 1_200, 500, 5_000)
        currency_input("Maintenance ($/unit/yr)", "opex_maintenance", 1_500, 500, 5_000)
        st.slider("Management Fee % (of EGI)", 0.0, 8.0, 5.0, 0.5, key="opex_management_pct",
                 format="%.1f%%")

        st.subheader("Annual Escalations")
        st.slider("Market Rent Growth", 0.0, 5.0, 2.0, 0.5, key="market_rent_growth_pct",
                 format="%.1f%%", help="Annual growth rate for market rents")
        st.slider("Affordable Rent Growth", 0.0, 3.0, 1.0, 0.5, key="affordable_rent_growth_pct",
                 format="%.1f%%", help="Annual growth rate for affordable rents")
        st.slider("OpEx Growth", 0.0, 5.0, 3.0, 0.5, key="opex_growth_pct",
                 format="%.1f%%", help="Annual growth rate for operating expenses")
        st.slider("Property Tax Growth", 0.0, 5.0, 2.0, 0.5, key="property_tax_growth_pct",
                 format="%.1f%%", help="Annual growth rate for property taxes")

    with col3:
        st.subheader("Financing")
        st.slider("Construction Rate", 0.0, 15.0, 7.5, 0.5, key="construction_rate_pct",
                 format="%.1f%%")
        st.slider("Construction LTC", 0.0, 90.0, 65.0, 5.0, key="construction_ltc_pct",
                 format="%.0f%%")
        st.slider("Perm Rate", 0.0, 18.0, 6.0, 0.5, key="perm_rate_pct",
                 format="%.1f%%")
        st.selectbox("Perm Amortization", [15, 20, 25, 30], index=1,
                    key="perm_amort_years", format_func=lambda x: f"{x} years")
        st.slider("Max LTV", 0.0, 90.0, 65.0, 5.0, key="perm_ltv_max_pct",
                 format="%.0f%%")
        st.slider("Min DSCR", 1.00, 1.50, 1.25, 0.05,
                 key="perm_dscr_min")

        st.subheader("Exit")
        st.slider("Exit Cap Rate", 3.0, 15.0, 5.5, 0.5, key="exit_cap_rate_pct",
                 format="%.1f%%")

        st.subheader("Property Tax")
        currency_input("Existing Assessed Value", "existing_assessed_value", 5_000_000, 0, 50_000_000,
                      help="Pre-development assessed value of the land (baseline for TIF)")




def render_single_project_results(metrics, result, scenario_name: str = "Project"):
    """Render results for single project mode."""
    st.subheader(f"{scenario_name} Results")

    # Key metrics in a row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Levered IRR", f"{metrics.levered_irr:.2%}")
    with col2:
        st.metric("Equity Multiple", f"{metrics.equity_multiple:.2f}x")
    with col3:
        st.metric("TDC", f"${metrics.tdc:,.0f}")
    with col4:
        st.metric("Equity Required", f"${metrics.equity_required:,.0f}")

    st.divider()

    # Full metrics table
    st.subheader("Project Metrics")

    metrics_data = {
        "Category": [
            "Development",
            "",
            "",
            "",
            "",
            "Revenue & Operations",
            "",
            "",
            "",
            "Returns",
            "",
            "",
        ],
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
        "Value": [
            f"${metrics.tdc:,.0f}",
            f"${metrics.tdc_per_unit:,.0f}",
            f"${metrics.equity_required:,.0f}",
            f"${metrics.debt_amount:,.0f}",
            f"${metrics.tif_value:,.0f}",
            "",
            f"${metrics.gpr_annual:,.0f}",
            f"${metrics.noi_annual:,.0f}",
            f"{metrics.yield_on_cost:.2%}",
            "",
            f"{metrics.levered_irr:.2%}",
            f"{metrics.unlevered_irr:.2%}",
            f"{metrics.equity_multiple:.2f}x",
        ],
    }

    df = pd.DataFrame(metrics_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Cash flow chart
    st.subheader("Monthly Cash Flows")

    import plotly.graph_objects as go

    cfs = [cf.levered_cf for cf in result.monthly_cash_flows]
    months = list(range(1, len(cfs) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months, y=cfs,
        mode='lines', name=scenario_name,
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
    ))

    fig.update_layout(
        title="Levered Cash Flows by Month",
        xaxis_title="Month",
        yaxis_title="Cash Flow ($)",
        yaxis_tickformat="$,.0f",
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_results_tab(market_metrics, mixed_metrics, comparison, market_result, mixed_result):
    """Render the results comparison tab."""
    # IRR Difference callout
    irr_diff = comparison.irr_difference_bps
    meets_target = comparison.meets_target

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
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

    st.divider()

    # Comparison table
    st.subheader("Scenario Comparison")

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

    # Cash flow chart
    st.subheader("Monthly Cash Flows")

    import plotly.graph_objects as go

    market_cfs = [cf.levered_cf for cf in market_result.monthly_cash_flows]
    mixed_cfs = [cf.levered_cf for cf in mixed_result.monthly_cash_flows]
    months = list(range(1, len(market_cfs) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months, y=market_cfs,
        mode='lines', name='Market Rate',
        line=dict(color='#1f77b4', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=months, y=mixed_cfs,
        mode='lines', name='Mixed Income',
        line=dict(color='#ff7f0e', width=2)
    ))

    fig.update_layout(
        title="Levered Cash Flows by Month",
        xaxis_title="Month",
        yaxis_title="Cash Flow ($)",
        yaxis_tickformat="$,.0f",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_matrix_tab(inputs: ProjectInputs):
    """Render the scenario matrix tab."""
    st.subheader("Scenario Matrix Analysis")
    st.caption("Test all incentive combinations to find the best package")

    tier_options = st.multiselect(
        "Tiers to Test",
        options=[1, 2, 3],
        default=[st.session_state.get("selected_tier", 2)],
        format_func=lambda x: f"Tier {x}"
    )

    if st.button("Run Scenario Matrix", type="primary"):
        if not tier_options:
            st.warning("Please select at least one tier to test")
            return

        with st.spinner("Running scenario matrix... This may take a moment."):
            tiers = [IncentiveTier(t) for t in tier_options]
            combinations = list(generate_combinations(tiers=tiers))

            # Run matrix
            results = run_scenario_matrix(inputs, iter(combinations))

            # Store results in session state
            st.session_state["matrix_results"] = results

    # Display results if available
    if "matrix_results" in st.session_state:
        results = st.session_state["matrix_results"]

        # Summary metrics
        meeting_target = sum(1 for r in results if r.comparison.meets_target)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Combinations Tested", len(results))
        with col2:
            st.metric("Meeting +150 bps Target", meeting_target)
        with col3:
            if results:
                st.metric("Best IRR Improvement", f"{results[0].comparison.irr_difference_bps:+d} bps")

        st.divider()

        # Results table
        table_data = []
        for i, r in enumerate(results, 1):
            table_data.append({
                "Rank": i,
                "Scenario": r.combination.name,
                "Market IRR": f"{r.market_metrics.levered_irr:.2%}",
                "Mixed IRR": f"{r.mixed_metrics.levered_irr:.2%}",
                "Difference (bps)": r.comparison.irr_difference_bps,
                "Meets Target": "✅" if r.comparison.meets_target else "❌",
            })

        df = pd.DataFrame(table_data)

        # Style the dataframe
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", width="small"),
                "Scenario": st.column_config.TextColumn("Scenario", width="medium"),
                "Market IRR": st.column_config.TextColumn("Market IRR", width="small"),
                "Mixed IRR": st.column_config.TextColumn("Mixed IRR", width="small"),
                "Difference (bps)": st.column_config.NumberColumn(
                    "Diff (bps)", width="small", format="%+d"
                ),
                "Meets Target": st.column_config.TextColumn("Target", width="small"),
            }
        )

        # Best combination details
        if results and results[0].comparison.meets_target:
            st.success(f"""
            **Best Combination: {results[0].combination.name}**

            This combination achieves **{results[0].comparison.irr_difference_bps:+d} bps**
            improvement over market rate.
            """)


def render_detailed_cashflow_tab(inputs: ProjectInputs):
    """Render the detailed cash flow tab."""
    st.subheader("Detailed Cash Flow Analysis")

    # Export button in top-right corner
    col1, col2 = st.columns([4, 1])
    with col2:
        export_placeholder = st.empty()

    # Calculate average GSF for revenue estimation
    total_gsf = sum(
        inputs.unit_mix[ut].gsf * inputs.unit_mix[ut].allocation
        for ut in inputs.unit_mix
    )
    avg_gsf = total_gsf
    monthly_gpr = inputs.target_units * avg_gsf * inputs.market_rent_psf

    # Calculate hard costs
    hard_costs = inputs.target_units * inputs.hard_cost_per_unit

    # Calculate annual opex per unit
    annual_opex_per_unit = inputs.opex_utilities + inputs.opex_maintenance + inputs.opex_misc

    # Get mezzanine and preferred from session state (set in Sources & Uses tab)
    mezzanine_debt = st.session_state.get("mezzanine_debt", 0)
    mezzanine_rate = st.session_state.get("mezzanine_rate_pct", 12.0) / 100
    preferred_equity = st.session_state.get("preferred_equity", 0)
    preferred_return = st.session_state.get("preferred_return_pct", 10.0) / 100

    try:
        # Generate detailed cash flow
        result = generate_detailed_cash_flow(
            land_cost=inputs.land_cost_per_acre,  # Simplified: 1 acre
            hard_costs=hard_costs,
            soft_cost_pct=inputs.soft_cost_pct,
            construction_ltc=inputs.construction_ltc,
            construction_rate=inputs.construction_rate,
            start_date=inputs.predevelopment_start,
            predevelopment_months=inputs.predevelopment_months,
            construction_months=inputs.construction_months,
            leaseup_months=inputs.leaseup_months,
            operations_months=inputs.operations_months,
            monthly_gpr_at_stabilization=monthly_gpr,
            vacancy_rate=inputs.vacancy_rate,
            leaseup_pace=inputs.leaseup_pace,
            max_occupancy=inputs.max_occupancy,
            annual_opex_per_unit=annual_opex_per_unit,
            total_units=inputs.target_units,
            existing_assessed_value=inputs.existing_assessed_value,
            market_rent_growth=inputs.market_rent_growth,
            opex_growth=inputs.opex_growth,
            prop_tax_growth=inputs.property_tax_growth,
            perm_rate=inputs.perm_rate,
            perm_amort_years=inputs.perm_amort_years,
            perm_ltv_max=inputs.perm_ltv_max,
            perm_dscr_min=inputs.perm_dscr_min,
            exit_cap_rate=inputs.exit_cap_rate,
            reserves_pct=inputs.reserves_pct,
            affordable_pct=inputs.affordable_pct,
            # Mezzanine debt
            mezzanine_amount=mezzanine_debt,
            mezzanine_rate=mezzanine_rate,
            mezzanine_io=True,  # Interest-only
            # Preferred equity
            preferred_amount=preferred_equity,
            preferred_return=preferred_return,
            # TIF treatment from scenario config
            tif_treatment=_get_tif_treatment_str(),
            tif_lump_sum=st.session_state.get("tif_lump_sum_source", 0),
            tif_abatement_pct=st.session_state.get("abate_pct", 0) / 100,
            tif_abatement_years=st.session_state.get("abate_term", 0),
            tif_stream_pct=st.session_state.get("scenario_b_stream_pct", 100) / 100,
            tif_stream_years=st.session_state.get("scenario_b_stream_years", 20),
            tif_start_delay_months=st.session_state.get("scenario_b_tif_delay", 0),
        )

        # Calculate equity multiple for header
        total_equity_out = abs(sum(p.levered_cf for p in result.periods if p.levered_cf < 0))
        total_distributions = sum(p.levered_cf for p in result.periods if p.levered_cf > 0)
        equity_multiple = total_distributions / total_equity_out if total_equity_out > 0 else 0

        # Deal summary header
        render_deal_summary_header(
            tdc=result.sources_uses.tdc,
            units=inputs.target_units,
            equity=result.sources_uses.equity,
            senior_debt=result.sources_uses.construction_loan,
            levered_irr=result.levered_irr,
            equity_multiple=equity_multiple,
        )

        st.divider()

        # Export button
        with export_placeholder:
            csv_data = export_cashflow_to_csv(result)
            st.download_button(
                label="Export CSV",
                data=csv_data,
                file_name="detailed_cashflow.csv",
                mime="text/csv",
            )

        # Render Sources & Uses
        render_sources_uses(result.sources_uses)

        st.divider()

        # Render IRR Summary
        render_irr_summary(result)

        st.divider()

        # Stabilized Operating Statement
        render_operating_statement(result)

        st.divider()

        # Period aggregation selector
        aggregation = st.radio(
            "View by",
            options=["monthly", "quarterly", "annual"],
            index=0,
            key="cf_aggregation",
            horizontal=True
        )

        # Render detailed cash flow table with native horizontal scroll
        render_detailed_cashflow_table(
            result,
            aggregation=aggregation,
        )

        st.divider()

        # Exit waterfall
        render_exit_waterfall(result)

        st.divider()

        # Sensitivity analysis
        render_sensitivity_analysis(result, {})

    except Exception as e:
        st.error(f"Error generating detailed cash flow: {e}")
        import traceback
        st.code(traceback.format_exc())


def render_property_tax_tab(inputs: ProjectInputs):
    """Render the property tax engine tab with TIF analysis."""
    # Import TIF analysis components
    from ui.components.tif_analysis_view import (
        render_tax_rate_breakdown,
        render_tif_lump_sum_result,
        render_tif_loan_schedule,
        render_smart_fee_waiver,
        render_city_tax_abatement,
    )
    from src.calculations.property_tax import solve_tif_term

    st.subheader("Property Tax & TIF Analysis")

    # Get Austin tax stack
    tax_stack = get_austin_tax_stack()

    # Calculate TDC (Total Development Cost)
    hard_costs = inputs.target_units * inputs.hard_cost_per_unit
    soft_costs = hard_costs * inputs.soft_cost_pct
    land_cost = inputs.land_cost_per_acre  # Simplified: 1 acre

    # Estimate IDC (Interest During Construction)
    construction_loan = (land_cost + hard_costs + soft_costs) * inputs.construction_ltc
    avg_balance = construction_loan * 0.5
    months = inputs.predevelopment_months + inputs.construction_months
    monthly_rate = inputs.construction_rate / 12
    idc = avg_balance * monthly_rate * months

    tdc = land_cost + hard_costs + soft_costs + idc

    # Calculate stabilized value from NOI
    total_gsf = sum(
        inputs.unit_mix[ut].gsf * inputs.unit_mix[ut].allocation
        for ut in inputs.unit_mix
    )
    monthly_gpr = inputs.target_units * total_gsf * inputs.market_rent_psf
    annual_noi = monthly_gpr * 12 * (1 - inputs.vacancy_rate) * 0.60  # Rough margin
    stabilized_value = annual_noi / inputs.exit_cap_rate

    # Tabs for different views
    prop_tab1, prop_tab2, prop_tab3 = st.tabs([
        "Tax Rates & Assessment",
        "TIF Lump Sum Analysis",
        "Incentives (SMART / Abatement)"
    ])

    with prop_tab1:
        # Show tax rate breakdown
        render_tax_rate_breakdown(tax_stack)

        st.divider()

        # Render the property tax engine with TDC
        render_property_tax_engine(
            tax_stack=tax_stack,
            baseline_value=inputs.existing_assessed_value,
            stabilized_value=stabilized_value,
            tdc=tdc,
        )

    with prop_tab2:
        st.markdown("### TIF Lump Sum Calculator")
        st.caption("Calculate the TIF lump sum value based on capitalized tax increment.")

        # Inputs for TIF calculation
        col1, col2, col3 = st.columns(3)

        with col1:
            tif_base_value = st.number_input(
                "Base Assessed Value",
                min_value=0.0,
                value=float(inputs.existing_assessed_value),
                step=100000.0,
                format="%.0f",
                key="tif_base_value",
                help="Assessed value before development",
            )

        with col2:
            tif_new_value = st.number_input(
                "New Assessed Value (TDC)",
                min_value=0.0,
                value=float(tdc),
                step=100000.0,
                format="%.0f",
                key="tif_new_value",
                help="Typically equals TDC",
            )

        with col3:
            tif_affordable_units = st.number_input(
                "Affordable Units",
                min_value=1,
                value=max(1, int(inputs.target_units * inputs.affordable_pct)),
                step=1,
                key="tif_affordable_units",
            )

        col1, col2 = st.columns(2)
        with col1:
            tif_cap_rate = st.slider(
                "TIF Cap Rate",
                min_value=5.0,
                max_value=15.0,
                value=9.5,
                step=0.25,
                format="%.2f%%",
                key="prop_tif_cap_rate",
                help="Cap rate for capitalizing tax increment",
            ) / 100

        with col2:
            tif_discount_rate = st.slider(
                "Discount / Interest Rate",
                min_value=1.0,
                max_value=10.0,
                value=3.0,
                step=0.25,
                format="%.2f%%",
                key="prop_tif_discount",
            ) / 100

        st.divider()

        # Calculate and display TIF lump sum
        tif_lump_sum = render_tif_lump_sum_result(
            new_assessed_value=tif_new_value,
            base_assessed_value=tif_base_value,
            affordable_units=tif_affordable_units,
            tif_cap_rate=tif_cap_rate,
            discount_rate=tif_discount_rate,
            tif_term=20,
            tax_stack=tax_stack,
        )

        st.divider()

        # TIF as loan schedule
        if tif_lump_sum and tif_lump_sum > 0:
            incremental_value = tif_new_value - tif_base_value
            city_monthly_increment = (
                incremental_value * tax_stack.tif_participating_rate_decimal / 12
            )
            render_tif_loan_schedule(
                principal=tif_lump_sum,
                annual_rate=tif_discount_rate,
                monthly_payment=city_monthly_increment,
            )

    with prop_tab3:
        st.markdown("### Incentive Programs")

        # SMART fee waiver
        st.markdown("#### SMART Housing Fee Waiver")
        render_smart_fee_waiver(total_units=inputs.target_units)

        st.divider()

        # City tax abatement
        st.markdown("#### City Property Tax Abatement")
        affordable_units = max(1, int(inputs.target_units * inputs.affordable_pct))
        render_city_tax_abatement(
            assessed_value=tdc,
            affordable_units=affordable_units,
            total_units=inputs.target_units,
        )


def render_sources_uses_tab(inputs: ProjectInputs):
    """Render the detailed Sources & Uses tab."""
    st.subheader("Sources & Uses")

    # Get construction type params for units_per_acre
    from src.models.lookups import CONSTRUCTION_TYPE_PARAMS
    ct_params = CONSTRUCTION_TYPE_PARAMS.get(inputs.construction_type)
    units_per_acre = ct_params.units_per_acre if ct_params else 66

    # Calculate total GSF
    total_gsf = 0
    for ut, entry in inputs.unit_mix.items():
        units_of_type = round(inputs.target_units * entry.allocation)
        total_gsf += entry.gsf * units_of_type

    # Render inputs section
    su_inputs = render_sources_uses_inputs()

    st.divider()

    # Determine land cost method and values
    land_method = su_inputs.get("land_cost_method", LandCostMethod.PER_ACRE)
    land_direct = su_inputs.get("land_direct", 0)
    land_per_acre = inputs.land_cost_per_acre

    # Calculate monthly opex for reserves
    monthly_opex = (
        inputs.opex_utilities +
        inputs.opex_maintenance +
        inputs.opex_misc
    ) / 12 * inputs.target_units

    # Estimate monthly debt service for lease-up reserve
    # (rough estimate based on construction loan)
    hard_costs = su_inputs.get("hard_cost_per_unit", inputs.hard_cost_per_unit) * inputs.target_units
    soft_costs = hard_costs * su_inputs.get("soft_cost_pct", inputs.soft_cost_pct)
    estimated_tdc = hard_costs + soft_costs + (land_direct if land_method == LandCostMethod.DIRECT else land_per_acre * (inputs.target_units / units_per_acre))
    construction_loan = estimated_tdc * su_inputs.get("construction_ltc_cap", inputs.construction_ltc)
    monthly_debt_service = construction_loan * (inputs.perm_rate / 12) * 1.5  # Rough P&I estimate

    # Calculate detailed S&U
    try:
        su_detailed = calculate_sources_uses_detailed(
            target_units=inputs.target_units,
            total_gsf=total_gsf,
            units_per_acre=units_per_acre,

            # Land
            land_cost_method=land_method,
            land_direct=land_direct,
            land_per_acre=land_per_acre,

            # Hard costs
            hard_cost_per_unit=su_inputs.get("hard_cost_per_unit", inputs.hard_cost_per_unit),
            hard_cost_contingency_pct=su_inputs.get("hard_cost_contingency_pct", 0.05),

            # Soft costs
            soft_cost_pct_of_hard=su_inputs.get("soft_cost_pct", inputs.soft_cost_pct),
            developer_fee_pct=su_inputs.get("developer_fee_pct", 0.04),
            soft_cost_contingency_pct=su_inputs.get("soft_cost_contingency_pct", 0.05),

            # Financing
            construction_ltc_cap=su_inputs.get("construction_ltc_cap", inputs.construction_ltc),
            construction_rate=inputs.construction_rate,
            construction_months=inputs.construction_months,
            construction_loan_fee_pct=su_inputs.get("construction_loan_fee_pct", 0.01),

            # Reserves
            monthly_opex=monthly_opex,
            monthly_debt_service=monthly_debt_service,
            operating_reserve_months=su_inputs.get("operating_reserve_months", 3),
            leaseup_reserve_months=su_inputs.get("leaseup_reserve_months", 6),

            # Gap funding
            tif_lump_sum=su_inputs.get("tif_lump_sum", 0),
            grants=su_inputs.get("grants", 0),
            fee_waivers=0,  # Calculated from incentives

            # Mezzanine / preferred
            mezzanine_debt=su_inputs.get("mezzanine_debt", 0),
            mezzanine_rate=su_inputs.get("mezzanine_rate", 0.12),
            preferred_equity=su_inputs.get("preferred_equity", 0),
            preferred_return=su_inputs.get("preferred_return", 0.10),

            # Deferred
            deferred_developer_fee_pct=su_inputs.get("deferred_developer_fee_pct", 0),
        )

        # Render the detailed table
        render_sources_uses_detailed_table(su_detailed)

    except Exception as e:
        st.error(f"Error calculating Sources & Uses: {e}")
        import traceback
        st.code(traceback.format_exc())


def main():
    """Main application entry point."""
    st.title("🏠 Austin Affordable Housing Incentive Calculator")
    st.caption("Model multifamily developments and analyze incentive impacts on returns")

    # Render sidebar
    render_sidebar()

    # Main content tabs
    tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🎯 Scenarios",
        "⚙️ Project Inputs",
        "🏢 Unit Mix",
        "🏛️ Property Tax",
        "💵 Sources & Uses",
        "🎁 Incentives",
        "💰 Detailed Cash Flows",
        "📊 Results"
    ])

    # Get inputs from session state
    inputs = get_session_state_inputs()

    # Get scenario configuration
    model_config = get_model_config_from_session()

    # Run analysis based on mode
    try:
        if model_config.mode == ModelMode.COMPARISON:
            # Use scenario-specific configurations
            market_result, mixed_result, market_metrics, mixed_metrics, comparison = run_analysis(
                inputs,
                scenario_a=model_config.scenario_a,
                scenario_b=model_config.scenario_b,
            )
        else:
            # Single project mode - run both but use the single scenario config
            market_result, mixed_result, market_metrics, mixed_metrics, comparison = run_analysis(
                inputs,
                scenario_a=model_config.single_scenario if model_config.single_project_type == ProjectType.MARKET_RATE else None,
                scenario_b=model_config.single_scenario if model_config.single_project_type == ProjectType.MIXED_INCOME else None,
            )
        analysis_success = True
    except Exception as e:
        st.error(f"Error running analysis: {e}")
        analysis_success = False

    with tab0:
        render_scenarios_tab()

    with tab1:
        render_project_inputs_tab()

    with tab2:
        render_unit_mix_tab()

    with tab3:
        render_property_tax_tab(inputs)

    with tab4:
        render_sources_uses_tab(inputs)

    with tab5:
        render_matrix_tab(inputs)

    with tab6:
        render_detailed_cashflow_tab(inputs)

    with tab7:
        if analysis_success:
            # Check mode and render appropriate view
            mode = st.session_state.get("model_mode", ModelMode.COMPARISON)

            if mode == ModelMode.SINGLE_PROJECT:
                # In single project mode, show only the relevant scenario
                project_type = st.session_state.get("single_project_type", ProjectType.MIXED_INCOME)
                if project_type == ProjectType.MARKET_RATE:
                    render_single_project_results(market_metrics, market_result, "Market Rate")
                else:
                    render_single_project_results(mixed_metrics, mixed_result, "Mixed Income")
            else:
                # Comparison mode - show both scenarios
                render_results_tab(market_metrics, mixed_metrics, comparison, market_result, mixed_result)
        else:
            st.warning("Fix input errors to see results")


if __name__ == "__main__":
    main()
