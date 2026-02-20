"""Main Streamlit application for Austin TIF Model."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

from src.models.project import ProjectInputs, Scenario, UnitMixEntry, TIFStartTiming
from src.models.lookups import ConstructionType, DEFAULT_TAX_RATES
from src.models.incentives import IncentiveTier, IncentiveToggles, get_tier_config
from src.calculations.dcf import run_dcf
from src.calculations.units import allocate_units, get_total_units, get_total_affordable_units
from src.calculations.revenue import calculate_gpr
from src.calculations.metrics import calculate_metrics, calculate_metrics_from_detailed, compare_scenarios
from src.scenarios import run_scenario_matrix, generate_combinations
from src.calculations.detailed_cashflow import generate_detailed_cash_flow, calculate_deal
from src.calculations.sources_uses import calculate_sources_uses
from src.calculations.property_tax import get_austin_tax_stack
from ui.components.detailed_cashflow_view import (
    render_sources_uses, render_detailed_cashflow_table,
    render_irr_summary, render_property_tax_engine, render_sensitivity_analysis,
    render_deal_summary_header, export_cashflow_to_csv, export_cashflow_to_excel,
    render_exit_waterfall, render_operating_statement
)
from ui.components.unit_mix import (
    render_unit_mix_tab, get_unit_mix_from_session_state, get_efficiency
)
from ui.components.spreadsheet_debug_view import render_full_debug_page
from src.models.scenario_config import (
    ModelMode, ProjectType, TIFTreatment, TIFConfig,
    ScenarioInputs, ModelConfig, SharedInputs,
)
from ui.components.scenario_config_view import (
    render_mode_selector, render_single_project_config, render_comparison_config,
    render_scenario_summary, get_model_config_from_session,
)
from src.calculations.monte_carlo import (
    DistributionType, InputDistribution, BaseInputs,
    MonteCarloConfig, run_monte_carlo, run_tif_grid_search,
)
from src.calculations.detailed_cashflow import AssessedValueBasis
import numpy as np

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


def get_inputs_for_scenario(scenario: ScenarioInputs = None, is_mixed_income: bool = False) -> ProjectInputs:
    """Build ProjectInputs from session state, optionally overriding with scenario-specific values.

    Args:
        scenario: Optional ScenarioInputs to override unit count, affordable %, etc.
        is_mixed_income: If True, read from mixed income inputs (mixed_ prefix)

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
        # Determine if this is mixed income based on affordable_pct
        is_mixed_income = affordable_pct > 0
    else:
        target_units = st.session_state.get("target_units", 200)
        # Market rate scenarios have 0% affordable, mixed income reads from session state
        if is_mixed_income:
            affordable_pct = st.session_state.get("affordable_pct", 20.0) / 100
            ami_level = st.session_state.get("ami_level", "50%")
        else:
            affordable_pct = 0.0
            ami_level = "50%"  # Not used for market rate

    # Determine prefix for reading inputs
    prefix = "mixed_" if is_mixed_income else ""

    # Helper to get value from appropriate source
    def get_val(key: str, default, as_pct: bool = False):
        """Get value from market or mixed income inputs based on scenario."""
        full_key = f"{prefix}{key}" if prefix else key
        val = st.session_state.get(full_key, st.session_state.get(key, default))
        return val / 100 if as_pct else val

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
    elif st.session_state.get("run_mixed", True) and is_mixed_income:
        tier = IncentiveTier(st.session_state.get("selected_tier", 2))
        # NOTE: Defaults must match get_session_state_inputs() defaults
        # Default: TIF lump sum ON, TIF stream OFF, SMART ON
        toggles = IncentiveToggles(
            smart_fee_waiver=st.session_state.get("smart_fee_waiver", True),
            tax_abatement=st.session_state.get("tax_abatement", False),
            tif_lump_sum=st.session_state.get("tif_lump_sum", True),
            tif_stream=st.session_state.get("tif_stream", False),
            interest_buydown=st.session_state.get("interest_buydown", False),
        )
        incentive_config = get_tier_config(tier, toggles)

    return ProjectInputs(
        predevelopment_start=date(2026, 1, 1),
        predevelopment_months=get_val("predevelopment_months", 18),
        construction_months=get_val("construction_months", 24),
        leaseup_months=get_val("leaseup_months", 12),
        operations_months=get_val("operations_months", 12),
        land_cost=get_val("land_cost", 3_000_000),
        target_units=target_units,
        hard_cost_per_unit=get_val("hard_cost_per_unit", 175_000),
        soft_cost_pct=get_val("soft_cost_pct", 30.0, as_pct=True),
        predevelopment_cost_pct=get_val("predevelopment_cost_pct", 9.72, as_pct=True),
        hard_cost_contingency_pct=get_val("hard_cost_contingency_pct", 5.0, as_pct=True),
        soft_cost_contingency_pct=get_val("soft_cost_contingency_pct", 5.0, as_pct=True),
        developer_fee_pct=get_val("developer_fee_pct", 4.0, as_pct=True),
        construction_type=ConstructionType(st.session_state.get("construction_type", "podium_midrise_5over1")),
        unit_mix=unit_mix,
        market_rent_psf=st.session_state.get("market_rent_psf", 2.50),
        vacancy_rate=get_val("vacancy_rate_pct", 6.0, as_pct=True),
        leaseup_pace=get_val("leaseup_pace_pct", 8.0, as_pct=True),
        max_occupancy=st.session_state.get("max_occupancy", 0.94),
        opex_utilities=get_val("opex_utilities", 1200),
        opex_management_pct=get_val("opex_management_pct", 5.0, as_pct=True),
        opex_maintenance=get_val("opex_maintenance", 1500),
        opex_misc=st.session_state.get("opex_misc", 650),
        reserves_pct=st.session_state.get("reserves_pct", 0.02),
        market_rent_growth=get_val("market_rent_growth_pct", 2.0, as_pct=True),
        affordable_rent_growth=get_val("affordable_rent_growth_pct", 1.0, as_pct=True),
        opex_growth=get_val("opex_growth_pct", 3.0, as_pct=True),
        property_tax_growth=get_val("property_tax_growth_pct", 2.0, as_pct=True),
        construction_rate=get_val("construction_rate_pct", 7.5, as_pct=True),
        construction_ltc=get_val("construction_ltc_pct", 65.0, as_pct=True),
        operating_reserve_months=get_val("operating_reserve_months", 3),
        leaseup_reserve_months=get_val("leaseup_reserve_months", 6),
        perm_rate=get_val("perm_rate_pct", 6.0, as_pct=True),
        perm_amort_years=get_val("perm_amort_years", 20),
        perm_ltv_max=get_val("perm_ltv_max_pct", 65.0, as_pct=True),
        perm_dscr_min=get_val("perm_dscr_min", 1.25),
        existing_assessed_value=get_val("existing_assessed_value", 5_000_000),
        tax_rates=DEFAULT_TAX_RATES.copy(),
        exit_cap_rate=get_val("exit_cap_rate_pct", 5.5, as_pct=True),
        affordable_pct=affordable_pct,
        ami_level=ami_level,
        incentive_config=incentive_config,
        tif_start_timing=TIFStartTiming.OPERATIONS,
    )


def get_session_state_inputs() -> ProjectInputs:
    """Build ProjectInputs from session state."""
    from src.models.incentives import TIER_REQUIREMENTS

    # Get efficiency from construction type
    construction_type = st.session_state.get("construction_type", "podium_midrise_5over1")
    efficiency = get_efficiency(construction_type)

    # Unit mix from session state (using new component)
    unit_mix = get_unit_mix_from_session_state(efficiency)

    # Sync affordable_pct with selected tier to ensure consistency
    # This is needed because the Scenarios tab might not have run yet
    selected_tier = st.session_state.get("selected_tier", 2)
    tier_enum = IncentiveTier(selected_tier)
    tier_reqs = TIER_REQUIREMENTS[tier_enum]

    # Get affordable_pct from tier-specific key, or fall back to tier default
    tier_key = f"scenario_b_tier{selected_tier}_pct"
    if tier_key in st.session_state:
        affordable_pct_value = st.session_state[tier_key]
    else:
        affordable_pct_value = tier_reqs["affordable_pct"] * 100  # Convert to percentage

    # Get AMI level from tier-specific key, or fall back to tier default
    ami_key = f"scenario_b_tier{selected_tier}_ami"
    if ami_key in st.session_state:
        ami_level_value = st.session_state[ami_key]
    else:
        ami_level_value = str(tier_reqs["ami_level"])

    # Update the global values to match the selected tier
    st.session_state["affordable_pct"] = float(affordable_pct_value)
    st.session_state["ami_level"] = ami_level_value

    # Build incentive config if in mixed-income mode
    incentive_config = None
    if st.session_state.get("run_mixed", True):
        tier = IncentiveTier(selected_tier)
        # NOTE: Defaults here must match scenario_config_view.py defaults
        # Default: TIF lump sum ON, TIF stream OFF, SMART ON
        toggles = IncentiveToggles(
            smart_fee_waiver=st.session_state.get("smart_fee_waiver", True),
            tax_abatement=st.session_state.get("tax_abatement", False),
            tif_lump_sum=st.session_state.get("tif_lump_sum", True),
            tif_stream=st.session_state.get("tif_stream", False),
            interest_buydown=st.session_state.get("interest_buydown", False),
        )
        incentive_config = get_tier_config(tier, toggles)

    return ProjectInputs(
        predevelopment_start=date(2026, 1, 1),
        predevelopment_months=st.session_state.get("predevelopment_months", 18),
        construction_months=st.session_state.get("construction_months", 24),
        leaseup_months=st.session_state.get("leaseup_months", 12),
        operations_months=st.session_state.get("operations_months", 12),
        land_cost=st.session_state.get("land_cost", 3_000_000),
        target_units=st.session_state.get("target_units", 200),
        hard_cost_per_unit=st.session_state.get("hard_cost_per_unit", 175_000),
        soft_cost_pct=st.session_state.get("soft_cost_pct", 30.0) / 100,
        predevelopment_cost_pct=st.session_state.get("predevelopment_cost_pct", 9.72) / 100,
        hard_cost_contingency_pct=st.session_state.get("hard_cost_contingency_pct", 5.0) / 100,
        soft_cost_contingency_pct=st.session_state.get("soft_cost_contingency_pct", 5.0) / 100,
        developer_fee_pct=st.session_state.get("developer_fee_pct", 4.0) / 100,
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
        operating_reserve_months=st.session_state.get("operating_reserve_months", 3),
        leaseup_reserve_months=st.session_state.get("leaseup_reserve_months", 6),
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
    """Run unified deal analysis for both scenarios.

    Uses calculate_deal() as the SINGLE SOURCE OF TRUTH for all calculations.
    All metrics are derived from the period-by-period cash flows.

    Args:
        inputs: Base ProjectInputs (used if scenarios not provided)
        scenario_a: Optional scenario A configuration (market rate default)
        scenario_b: Optional scenario B configuration (mixed income default)

    Returns:
        Tuple of (market_result, mixed_result, market_metrics, mixed_metrics, comparison)
        All results are DetailedCashFlowResult with full period data.
    """
    # Build inputs for each scenario
    # Scenario A is market rate (no mixed income overrides, reads from non-prefixed keys)
    if scenario_a:
        inputs_a = get_inputs_for_scenario(scenario_a, is_mixed_income=False)
    else:
        # No scenario provided - build from session state using market rate keys
        inputs_a = get_inputs_for_scenario(None, is_mixed_income=False)

    # Scenario B is mixed income (use mixed income overrides, reads from mixed_ prefixed keys)
    if scenario_b:
        inputs_b = get_inputs_for_scenario(scenario_b, is_mixed_income=True)
    else:
        # No scenario provided - build from session state using mixed income keys
        inputs_b = get_inputs_for_scenario(None, is_mixed_income=True)

    # Get TIF parameters from session state (for mixed income)
    tif_lump_sum = st.session_state.get("calculated_tif_lump_sum", 0)
    tif_enabled = st.session_state.get("tif_lump_sum", True)

    # Market scenario - no incentives, 0% affordable
    market_result = calculate_deal(
        inputs=inputs_a,
        scenario=Scenario.MARKET,
        tif_lump_sum=0,
        tif_treatment="none",
    )

    # Calculate market GPR for metrics (need unit counts)
    market_allocs = allocate_units(
        inputs_a.target_units, inputs_a.unit_mix, 0.0, inputs_a.ami_level, inputs_a.market_rent_psf
    )
    market_gpr = calculate_gpr(market_allocs)
    market_metrics = calculate_metrics_from_detailed(
        market_result,
        Scenario.MARKET,
        get_total_units(market_allocs),
        0,
        market_gpr.total_gpr_annual
    )

    # Mixed-income scenario - with incentives
    mixed_result = calculate_deal(
        inputs=inputs_b,
        scenario=Scenario.MIXED_INCOME,
        tif_lump_sum=tif_lump_sum if tif_enabled else 0,
        tif_treatment="lump_sum" if tif_enabled and tif_lump_sum > 0 else "none",
    )

    # Calculate mixed income GPR for metrics
    mixed_allocs = allocate_units(
        inputs_b.target_units, inputs_b.unit_mix, inputs_b.affordable_pct,
        inputs_b.ami_level, inputs_b.market_rent_psf
    )
    mixed_gpr = calculate_gpr(mixed_allocs)
    mixed_metrics = calculate_metrics_from_detailed(
        mixed_result,
        Scenario.MIXED_INCOME,
        get_total_units(mixed_allocs),
        get_total_affordable_units(mixed_allocs),
        mixed_gpr.total_gpr_annual
    )

    comparison = compare_scenarios(market_metrics, mixed_metrics)

    return market_result, mixed_result, market_metrics, mixed_metrics, comparison


def currency_input(label: str, key: str, default: int, min_val: int = 0, max_val: int = 100_000_000, help: str = None) -> int:
    """Render a currency input with comma formatting.

    Args:
        label: Input label
        key: Session state key
        default: Default value
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        help: Optional help text tooltip

    Returns:
        The numeric value entered
    """
    # Get current value from session state or use default
    current_val = st.session_state.get(key, default)

    # Format for display
    formatted = f"${current_val:,}"

    # Render text input
    user_input = st.text_input(label, value=formatted, key=f"{key}_text", help=help)

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
    """Render the sidebar as a two-column Deal Summary (Market | Mixed Income)."""
    st.sidebar.header("Deal Summary")

    # Check mode
    mode = st.session_state.get("model_mode", ModelMode.COMPARISON)
    is_comparison = mode == ModelMode.COMPARISON

    # Try to calculate metrics
    try:
        # Initialize mixed income inputs BEFORE running analysis
        # This ensures mixed_ prefixed keys exist so mixed income uses its own values
        _initialize_mixed_income_inputs()

        inputs = get_session_state_inputs()

        # DEBUG: Show key input values being used
        with st.sidebar.expander("Debug: Input Values", expanded=False):
            st.caption(f"const_rate: {inputs.construction_rate:.2%}")
            st.caption(f"perm_rate: {inputs.perm_rate:.2%}")
            st.caption(f"exit_cap: {inputs.exit_cap_rate:.2%}")
            st.caption(f"target_units: {inputs.target_units}")
            st.caption(f"hard_cost: ${inputs.hard_cost_per_unit:,}")

        market_result, mixed_result, market_metrics, mixed_metrics, comparison = run_analysis(inputs)

        # DEBUG: Show calculated values
        with st.sidebar.expander("Debug: Calculated", expanded=False):
            st.caption(f"Market IRR: {market_metrics.levered_irr:.2%}")
            st.caption(f"Mixed IRR: {mixed_metrics.levered_irr:.2%}")
            st.caption(f"Market TDC: ${market_metrics.tdc:,.0f}")
            st.caption(f"Diff bps: {comparison.irr_difference_bps}")

        # Store results in session state for use by other tabs (single source of truth)
        st.session_state["_cached_market_result"] = market_result
        st.session_state["_cached_mixed_result"] = mixed_result
        st.session_state["_cached_market_metrics"] = market_metrics
        st.session_state["_cached_mixed_metrics"] = mixed_metrics
        st.session_state["_cached_inputs"] = inputs

        # For single project mode, determine which scenario is active
        if not is_comparison:
            project_type = st.session_state.get("single_project_type", ProjectType.MIXED_INCOME)
            is_market_only = project_type == ProjectType.MARKET_RATE
        else:
            is_market_only = False

        # ========== COLUMN HEADERS ==========
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.markdown("**Market Rate**")
        with col2:
            st.markdown("**Mixed Income**")

        st.sidebar.divider()

        # ========== RETURNS ==========
        # IRR
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("IRR", f"{market_metrics.levered_irr:.1%}")
        with col2:
            if is_comparison or not is_market_only:
                st.metric("IRR", f"{mixed_metrics.levered_irr:.1%}")
            else:
                st.metric("IRR", "—")

        # Equity Multiple
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Equity Mult", f"{market_metrics.equity_multiple:.2f}x")
        with col2:
            if is_comparison or not is_market_only:
                st.metric("Equity Mult", f"{mixed_metrics.equity_multiple:.2f}x")
            else:
                st.metric("Equity Mult", "—")

        # IRR Difference (comparison mode only)
        if is_comparison:
            irr_diff = comparison.irr_difference_bps
            if comparison.meets_target:
                st.sidebar.success(f"**Δ {irr_diff:+d} bps** ✓")
            else:
                st.sidebar.warning(f"**Δ {irr_diff:+d} bps** (need +150)")

        st.sidebar.divider()

        # ========== UNITS ==========
        total_units = st.session_state.get("target_units", 200)
        affordable_pct = st.session_state.get("affordable_pct", 20.0)
        if affordable_pct > 1:
            affordable_pct = affordable_pct / 100
        affordable_units = int(total_units * affordable_pct)
        market_units_mixed = total_units - affordable_units
        ami_level = st.session_state.get("ami_level", "50%")

        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.caption("Units")
            st.markdown(f"{total_units} total")
            st.markdown("0 affordable")
        with col2:
            st.caption("Units")
            if is_comparison or not is_market_only:
                st.markdown(f"{market_units_mixed} market")
                st.markdown(f"{affordable_units} @ {ami_level}")
            else:
                st.markdown("—")
                st.markdown("—")

        st.sidebar.divider()

        # ========== CAPITAL STACK ==========
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.caption("Capital")
            st.markdown(f"TDC ${market_metrics.tdc/1e6:.1f}M")
            st.markdown(f"Debt ${market_metrics.debt_amount/1e6:.1f}M")
            st.markdown(f"Equity ${market_metrics.equity_required/1e6:.1f}M")
            st.markdown("TIF $0")
        with col2:
            st.caption("Capital")
            if is_comparison or not is_market_only:
                st.markdown(f"TDC ${mixed_metrics.tdc/1e6:.1f}M")
                st.markdown(f"Debt ${mixed_metrics.debt_amount/1e6:.1f}M")
                st.markdown(f"Equity ${mixed_metrics.equity_required/1e6:.1f}M")
                tif_val = mixed_metrics.tif_value if hasattr(mixed_metrics, 'tif_value') else 0
                st.markdown(f"TIF ${tif_val/1e6:.1f}M")
            else:
                st.markdown("—")
                st.markdown("—")
                st.markdown("—")
                st.markdown("—")

        st.sidebar.divider()

        # ========== OPERATING METRICS ==========
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.caption("Operations")
            st.markdown(f"GPR ${market_metrics.gpr_annual/1e6:.2f}M")
            st.markdown(f"NOI ${market_metrics.noi_annual/1e6:.2f}M")
            st.markdown(f"YoC {market_metrics.yield_on_cost:.1%}")
        with col2:
            st.caption("Operations")
            if is_comparison or not is_market_only:
                st.markdown(f"GPR ${mixed_metrics.gpr_annual/1e6:.2f}M")
                st.markdown(f"NOI ${mixed_metrics.noi_annual/1e6:.2f}M")
                st.markdown(f"YoC {mixed_metrics.yield_on_cost:.1%}")
            else:
                st.markdown("—")
                st.markdown("—")
                st.markdown("—")

        st.sidebar.divider()

        # ========== NPV (EDITABLE DISCOUNT RATE) ==========
        npv_discount = st.sidebar.slider(
            "NPV Discount Rate",
            min_value=5, max_value=25, value=15, step=1,
            key="npv_discount_rate", format="%d%%"
        ) / 100.0

        def calc_npv(cash_flows, rate):
            monthly_rate = (1 + rate) ** (1/12) - 1
            npv = 0
            for i, cf in enumerate(cash_flows):
                npv += cf.levered_cf / ((1 + monthly_rate) ** i)
            return npv

        market_npv = calc_npv(market_result.periods, npv_discount)
        mixed_npv = calc_npv(mixed_result.periods, npv_discount)

        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("NPV", f"${market_npv/1e6:.1f}M")
        with col2:
            if is_comparison or not is_market_only:
                st.metric("NPV", f"${mixed_npv/1e6:.1f}M")
            else:
                st.metric("NPV", "—")

        st.sidebar.divider()

        # ========== KEY ASSUMPTIONS (SHARED) ==========
        st.sidebar.caption("**Key Assumptions**")

        const_type = st.session_state.get("construction_type", "podium_5over1")
        const_rate = st.session_state.get("construction_rate_pct", 7.5)
        const_ltc = st.session_state.get("construction_ltc_pct", 65.0)
        perm_rate = st.session_state.get("perm_rate_pct", 6.5)
        exit_cap = st.session_state.get("exit_cap_rate_pct", 5.5)

        st.sidebar.markdown(f"{const_type.replace('_', ' ').title()}")
        st.sidebar.markdown(f"Const: {const_rate:.1f}% @ {const_ltc:.0f}% LTC")
        st.sidebar.markdown(f"Perm: {perm_rate:.1f}% | Exit: {exit_cap:.1f}%")

        # Incentive tier (comparison mode)
        if is_comparison:
            selected_tier = st.session_state.get("selected_tier", 2)
            tier_names = {1: "Tier 1", 2: "Tier 2", 3: "Tier 3"}
            active = []
            if st.session_state.get("tif_lump_sum", True):
                active.append("TIF")
            if st.session_state.get("smart_fee_waiver", True):
                active.append("SMART")
            incentive_str = " + ".join(active) if active else "None"
            st.sidebar.markdown(f"{tier_names.get(selected_tier, 'Tier 2')} | {incentive_str}")

    except Exception as e:
        st.sidebar.info("Configure project to see summary")
        import traceback
        st.sidebar.caption(f"Error: {e}")

    # ========== DARK MODE TOGGLE (at bottom of sidebar) ==========
    st.sidebar.divider()
    dark_mode = st.sidebar.toggle(
        "Dark Mode",
        value=st.session_state.get("dark_mode", False),
        key="dark_mode",
        help="Toggle between light and dark color scheme"
    )

    # Apply dark mode CSS
    if dark_mode:
        st.markdown("""
        <style>
            /* Dark mode overrides */
            .stApp {
                background-color: #0e1117;
                color: #fafafa;
            }
            .stSidebar {
                background-color: #262730;
            }
            .stSidebar [data-testid="stSidebarContent"] {
                background-color: #262730;
            }
            /* Metric cards */
            [data-testid="stMetricValue"] {
                color: #fafafa;
            }
            [data-testid="stMetricLabel"] {
                color: #b0b0b0;
            }
            /* Headers */
            h1, h2, h3, h4, h5, h6 {
                color: #fafafa !important;
            }
            /* Text */
            p, span, label, .stMarkdown {
                color: #fafafa;
            }
            /* Dataframes */
            .stDataFrame {
                background-color: #1e1e1e;
            }
            .stDataFrame [data-testid="stDataFrameResizable"] {
                background-color: #1e1e1e;
            }
            /* Tabs */
            .stTabs [data-baseweb="tab-list"] {
                background-color: #262730;
            }
            .stTabs [data-baseweb="tab"] {
                color: #fafafa;
            }
            /* Inputs */
            .stNumberInput input, .stTextInput input, .stSelectbox select {
                background-color: #262730;
                color: #fafafa;
            }
            /* Expander */
            .streamlit-expanderHeader {
                background-color: #262730;
                color: #fafafa;
            }
            /* Success/Warning/Error boxes */
            .stSuccess, .stWarning, .stError, .stInfo {
                color: #fafafa;
            }
            /* Captions */
            .stCaption {
                color: #b0b0b0;
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        # Light mode (default) - explicit light colors
        st.markdown("""
        <style>
            /* Light mode - explicit reset */
            .stApp {
                background-color: #ffffff;
                color: #262730;
            }
            .stSidebar {
                background-color: #f0f2f6;
            }
            .stSidebar [data-testid="stSidebarContent"] {
                background-color: #f0f2f6;
            }
            /* Metric cards */
            [data-testid="stMetricValue"] {
                color: #262730;
            }
            [data-testid="stMetricLabel"] {
                color: #555555;
            }
            /* Headers */
            h1, h2, h3, h4, h5, h6 {
                color: #262730 !important;
            }
            /* Text */
            p, span, label, .stMarkdown {
                color: #262730;
            }
            /* Dataframes */
            .stDataFrame {
                background-color: #ffffff;
            }
            .stDataFrame [data-testid="stDataFrameResizable"] {
                background-color: #ffffff;
            }
            /* Tabs */
            .stTabs [data-baseweb="tab-list"] {
                background-color: #f0f2f6;
            }
            .stTabs [data-baseweb="tab"] {
                color: #262730;
            }
            /* Number inputs */
            .stNumberInput input {
                background-color: #ffffff !important;
                color: #262730 !important;
            }
            .stNumberInput [data-baseweb="input"] {
                background-color: #ffffff !important;
            }
            .stNumberInput button {
                background-color: #f0f2f6 !important;
                color: #262730 !important;
            }
            /* Text inputs */
            .stTextInput input {
                background-color: #ffffff !important;
                color: #262730 !important;
            }
            .stTextInput [data-baseweb="input"] {
                background-color: #ffffff !important;
            }
            /* Selectbox / Dropdown */
            .stSelectbox [data-baseweb="select"] {
                background-color: #ffffff !important;
            }
            .stSelectbox [data-baseweb="select"] > div {
                background-color: #ffffff !important;
                color: #262730 !important;
            }
            .stSelectbox svg {
                fill: #262730 !important;
            }
            /* Dropdown menu */
            [data-baseweb="popover"] {
                background-color: #ffffff !important;
            }
            [data-baseweb="menu"] {
                background-color: #ffffff !important;
            }
            [data-baseweb="menu"] li {
                background-color: #ffffff !important;
                color: #262730 !important;
            }
            [data-baseweb="menu"] li:hover {
                background-color: #f0f2f6 !important;
            }
            /* Expander */
            .streamlit-expanderHeader {
                background-color: #f0f2f6;
                color: #262730;
            }
            /* Captions */
            .stCaption {
                color: #555555;
            }
            /* Radio buttons */
            .stRadio label {
                color: #262730 !important;
            }
            /* Checkboxes */
            .stCheckbox label {
                color: #262730 !important;
            }
            /* Sliders */
            .stSlider label {
                color: #262730 !important;
            }
            .stSlider [data-baseweb="slider"] div {
                color: #262730 !important;
            }
            /* Buttons - ensure visible in light mode */
            .stButton button {
                background-color: #f0f2f6 !important;
                color: #262730 !important;
                border: 1px solid #d0d0d0 !important;
            }
            .stButton button:hover {
                background-color: #e0e2e6 !important;
                border-color: #b0b0b0 !important;
            }
            /* Primary buttons */
            .stButton button[kind="primary"] {
                background-color: #ff4b4b !important;
                color: #ffffff !important;
                border: none !important;
            }
            .stButton button[kind="primary"]:hover {
                background-color: #ff3333 !important;
            }
            /* Secondary buttons */
            .stButton button[kind="secondary"] {
                background-color: #ffffff !important;
                color: #262730 !important;
                border: 1px solid #d0d0d0 !important;
            }
            /* Toggle */
            .stToggle label {
                color: #262730 !important;
            }
            /* DataFrame cells - ensure text is dark */
            [data-testid="stDataFrame"] td {
                color: #262730 !important;
            }
            [data-testid="stDataFrame"] th {
                color: #262730 !important;
                background-color: #f0f2f6 !important;
            }
            /* DataFrame styling for spreadsheet view */
            .dataframe {
                background-color: #ffffff !important;
            }
            .dataframe td, .dataframe th {
                color: #262730 !important;
            }
            /* Tables in markdown */
            table {
                background-color: #ffffff !important;
            }
            table td, table th {
                color: #262730 !important;
                border-color: #d0d0d0 !important;
            }
        </style>
        """, unsafe_allow_html=True)


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


def render_inputs_form(prefix: str = "", is_mixed_income: bool = False):
    """Render the project inputs form.

    Args:
        prefix: Key prefix for session state (empty for market rate, "mixed_" for mixed income)
        is_mixed_income: If True, show affordable rent growth input
    """
    # Helper to get key with prefix
    def key(name: str) -> str:
        return f"{prefix}{name}" if prefix else name

    # Initialize session state for all keys before rendering widgets
    # This avoids the "default value but also set via Session State API" error
    defaults = {
        "predevelopment_months": 18,
        "construction_months": 24,
        "leaseup_months": 12,
        "operations_months": 60,
        "land_cost": 3_000_000,
        "hard_cost_per_unit": 175_000,
        "soft_cost_pct": 30.0,
        "hard_cost_contingency_pct": 5.0,
        "soft_cost_contingency_pct": 5.0,
        "developer_fee_pct": 4.0,
        "predevelopment_cost_pct": 9.72,
        "operating_reserve_months": 3,
        "leaseup_reserve_months": 6,
        "vacancy_rate_pct": 6.0,
        "leaseup_pace_pct": 8.0,
        "opex_utilities": 1_200,
        "opex_maintenance": 1_500,
        "opex_management_pct": 5.0,
        "market_rent_growth_pct": 2.0,
        "affordable_rent_growth_pct": 1.0,
        "opex_growth_pct": 3.0,
        "property_tax_growth_pct": 2.0,
        "construction_rate_pct": 7.5,
        "construction_ltc_pct": 65.0,
        "perm_rate_pct": 6.0,
        "perm_amort_years": 20,
        "perm_ltv_max_pct": 65.0,
        "perm_dscr_min": 1.25,
        "exit_cap_rate_pct": 5.5,
        "existing_assessed_value": 5_000_000,
    }

    for name, default in defaults.items():
        k = key(name)
        if k not in st.session_state:
            # For mixed income, inherit from market rate if available
            if prefix:
                st.session_state[k] = st.session_state.get(name, default)
            else:
                st.session_state[k] = default

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Timing")
        st.number_input("Predevelopment (months)", min_value=6, max_value=36,
                       key=key("predevelopment_months"))
        st.number_input("Construction (months)", min_value=12, max_value=48,
                       key=key("construction_months"))
        st.number_input("Lease-up (months)", min_value=6, max_value=24,
                       key=key("leaseup_months"))
        st.number_input("Operations (months)", min_value=12, max_value=120,
                       key=key("operations_months"))

        st.subheader("Land & Construction")
        currency_input("Land Cost (Total)", key("land_cost"),
                      st.session_state[key("land_cost")], 500_000, 20_000_000)
        currency_input("Hard Cost/Unit", key("hard_cost_per_unit"),
                      st.session_state[key("hard_cost_per_unit")], 100_000, 500_000)
        st.slider("Soft Cost % (of hard costs)", min_value=0.0, max_value=60.0, step=1.0,
                 key=key("soft_cost_pct"), format="%.0f%%",
                 help="As percentage of hard costs")

        st.subheader("Contingencies & Fees")
        st.slider("Hard Cost Contingency", min_value=0.0, max_value=15.0, step=0.5,
                 key=key("hard_cost_contingency_pct"), format="%.1f%%")
        st.slider("Soft Cost Contingency", min_value=0.0, max_value=15.0, step=0.5,
                 key=key("soft_cost_contingency_pct"), format="%.1f%%")
        st.slider("Predevelopment (% of Hard)", min_value=0.0, max_value=20.0, step=0.1,
                 key=key("predevelopment_cost_pct"), format="%.1f%%",
                 help="Design, entitlement, and other costs before construction")
        st.slider("Developer Fee (% of Hard)", min_value=0.0, max_value=10.0, step=0.5,
                 key=key("developer_fee_pct"), format="%.1f%%")

        st.subheader("Reserves")
        st.number_input("Operating Reserve (months)", min_value=0, max_value=12,
                       key=key("operating_reserve_months"),
                       help="Months of operating expenses held in reserve")
        st.number_input("Lease-up Reserve (months)", min_value=0, max_value=12,
                       key=key("leaseup_reserve_months"),
                       help="Months of debt service held in reserve during lease-up")

    with col2:
        st.subheader("Operations")
        st.slider("Vacancy Rate", min_value=0.0, max_value=30.0, step=1.0,
                 key=key("vacancy_rate_pct"), format="%.0f%%")
        st.slider("Lease-up Pace (monthly)", min_value=0.0, max_value=20.0, step=1.0,
                 key=key("leaseup_pace_pct"), format="%.0f%%",
                 help="Monthly absorption during lease-up")

        st.subheader("Operating Expenses")
        currency_input("Utilities ($/unit/yr)", key("opex_utilities"),
                      st.session_state[key("opex_utilities")], 500, 5_000)
        currency_input("Maintenance ($/unit/yr)", key("opex_maintenance"),
                      st.session_state[key("opex_maintenance")], 500, 5_000)
        st.slider("Management Fee % (of EGI)", min_value=0.0, max_value=8.0, step=0.5,
                 key=key("opex_management_pct"), format="%.1f%%")

        st.subheader("Annual Escalations")
        st.slider("Market Rent Growth", min_value=0.0, max_value=5.0, step=0.5,
                 key=key("market_rent_growth_pct"), format="%.1f%%",
                 help="Annual growth rate for market rents")
        if is_mixed_income:
            st.slider("Affordable Rent Growth", min_value=0.0, max_value=3.0, step=0.5,
                     key=key("affordable_rent_growth_pct"), format="%.1f%%",
                     help="Annual growth rate for affordable rents")
        st.slider("OpEx Growth", min_value=0.0, max_value=5.0, step=0.5,
                 key=key("opex_growth_pct"), format="%.1f%%",
                 help="Annual growth rate for operating expenses")
        st.slider("Property Tax Growth", min_value=0.0, max_value=5.0, step=0.5,
                 key=key("property_tax_growth_pct"), format="%.1f%%",
                 help="Annual growth rate for property taxes")

    with col3:
        st.subheader("Financing")
        st.slider("Construction Rate", min_value=0.0, max_value=15.0, step=0.5,
                 key=key("construction_rate_pct"), format="%.1f%%")
        st.slider("Construction LTC", min_value=0.0, max_value=90.0, step=5.0,
                 key=key("construction_ltc_pct"), format="%.0f%%")
        st.slider("Perm Rate", min_value=0.0, max_value=18.0, step=0.5,
                 key=key("perm_rate_pct"), format="%.1f%%")
        amort_options = [15, 20, 25, 30]
        st.selectbox("Perm Amortization", amort_options,
                    key=key("perm_amort_years"), format_func=lambda x: f"{x} years")
        st.slider("Max LTV", min_value=0.0, max_value=90.0, step=5.0,
                 key=key("perm_ltv_max_pct"), format="%.0f%%")
        st.slider("Min DSCR", min_value=1.00, max_value=1.50, step=0.05,
                 key=key("perm_dscr_min"))

        st.subheader("Exit")
        st.slider("Exit Cap Rate", min_value=3.0, max_value=15.0, step=0.5,
                 key=key("exit_cap_rate_pct"), format="%.1f%%")

        st.subheader("Property Tax")
        currency_input("Existing Assessed Value", key("existing_assessed_value"),
                      st.session_state[key("existing_assessed_value")], 0, 50_000_000,
                      help="Pre-development assessed value of the land (baseline for TIF)")


def render_tier_and_incentives():
    """Render tier selection and incentive toggles for mixed-income scenarios."""
    from src.models.incentives import IncentiveTier, TIER_REQUIREMENTS

    st.subheader("Affordability Tier & Incentives")

    # Tier defaults
    tier_defaults = {
        1: {"affordable_pct": 5, "ami_level": "30%", "name": "Tier 1 - Deep Affordability (5% @ 30% AMI)"},
        2: {"affordable_pct": 20, "ami_level": "50%", "name": "Tier 2 - Moderate Volume (20% @ 50% AMI)"},
        3: {"affordable_pct": 10, "ami_level": "50%", "name": "Tier 3 - Balanced (10% @ 50% AMI)"},
    }

    col1, col2 = st.columns([1, 1])

    with col1:
        # Tier Selection
        selected_tier = st.radio(
            "Select Incentive Tier",
            options=[1, 2, 3],
            index=st.session_state.get("selected_tier", 2) - 1,
            format_func=lambda x: tier_defaults[x]["name"],
            key="selected_tier_radio",
            help="Each tier has different affordability requirements and incentive levels"
        )
        # Sync to session state
        st.session_state["selected_tier"] = selected_tier

        # Get tier defaults
        tier_info = tier_defaults[selected_tier]

        # Affordable % (editable)
        affordable_pct = st.number_input(
            "Affordable Units (%)",
            min_value=0,
            max_value=50,
            value=st.session_state.get(f"tier{selected_tier}_affordable_pct", tier_info["affordable_pct"]),
            step=5,
            key=f"tier{selected_tier}_affordable_pct_input",
            help="Percentage of units set aside as affordable"
        )
        st.session_state["affordable_pct"] = float(affordable_pct)

        # AMI Level
        ami_options = ["30%", "50%", "60%", "80%"]
        default_ami_idx = ami_options.index(tier_info["ami_level"]) if tier_info["ami_level"] in ami_options else 1
        ami_level = st.selectbox(
            "AMI Level",
            options=ami_options,
            index=default_ami_idx,
            key=f"tier{selected_tier}_ami_input",
            help="Area Median Income level for affordable units"
        )
        st.session_state["ami_level"] = ami_level

        # Show unit breakdown
        total_units = st.session_state.get("target_units", 200)
        affordable_units = int(total_units * affordable_pct / 100)
        market_units = total_units - affordable_units
        st.caption(f"**{affordable_units} affordable** / {market_units} market rate units")

    with col2:
        # Incentive Toggles
        st.markdown("**Available Incentives**")

        tif_lump_sum = st.checkbox(
            "TIF Lump Sum (Upfront Capital)",
            value=st.session_state.get("tif_lump_sum", True),
            key="tif_lump_sum_toggle",
            help="Receive TIF as upfront capital contribution"
        )
        st.session_state["tif_lump_sum"] = tif_lump_sum

        tif_stream = st.checkbox(
            "TIF Stream (Annual Payments)",
            value=st.session_state.get("tif_stream", False),
            key="tif_stream_toggle",
            help="Receive annual TIF reimbursement payments"
        )
        st.session_state["tif_stream"] = tif_stream

        # TIF options are mutually exclusive
        if tif_lump_sum and tif_stream:
            st.warning("TIF Lump Sum and TIF Stream are mutually exclusive. Select one.")

        smart_fee_waiver = st.checkbox(
            "SMART Fee Waiver",
            value=st.session_state.get("smart_fee_waiver", True),
            key="smart_fee_waiver_toggle",
            help="Waives development fees for affordable units"
        )
        st.session_state["smart_fee_waiver"] = smart_fee_waiver

        tax_abatement = st.checkbox(
            "Tax Abatement",
            value=st.session_state.get("tax_abatement", False),
            key="tax_abatement_toggle",
            help="Property tax abatement (mutually exclusive with TIF)"
        )
        st.session_state["tax_abatement"] = tax_abatement

        # TIF and abatement are mutually exclusive
        if tax_abatement and (tif_lump_sum or tif_stream):
            st.warning("Tax Abatement and TIF are mutually exclusive. Select one or the other.")

        # Show calculated TIF value if lump sum selected
        if tif_lump_sum:
            calculated_tif = st.session_state.get("calculated_tif_lump_sum", 0)
            if calculated_tif > 0:
                st.info(f"Calculated TIF Lump Sum: **${calculated_tif:,.0f}**")
            else:
                st.caption("TIF value calculated on Property Tax tab")


def render_project_inputs_tab():
    """Render the project inputs tab with mode selection at top."""

    # ========== PROJECT SETUP SECTION ==========
    st.header("Project Setup")

    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        # Analysis Mode
        mode = st.radio(
            "Analysis Mode",
            options=[ModelMode.SINGLE_PROJECT, ModelMode.COMPARISON],
            format_func=lambda x: {
                ModelMode.SINGLE_PROJECT: "Single Project",
                ModelMode.COMPARISON: "Comparison",
            }[x],
            key="model_mode",
            horizontal=True,
            help="Single: Analyze one scenario. Comparison: Market vs Mixed-Income side-by-side."
        )

    with col2:
        # Construction Type
        st.selectbox(
            "Construction Type",
            options=[ct.value for ct in ConstructionType],
            index=2,  # PODIUM_5OVER1
            key="construction_type",
            format_func=lambda x: x.replace("_", " ").title()
        )

    with col3:
        # Total Units
        st.number_input(
            "Total Units",
            min_value=50, max_value=500, value=200, step=10,
            key="target_units",
            help="Total number of units in the development"
        )

    st.divider()

    # ========== PROJECT INPUTS SECTION ==========
    st.header("Project Inputs")

    # Check mode for input display
    is_comparison = mode == ModelMode.COMPARISON
    is_single_mixed = (mode == ModelMode.SINGLE_PROJECT and
                       st.session_state.get("single_project_type") == ProjectType.MIXED_INCOME)

    if is_comparison:
        # Show sub-tabs for Market Rate and Mixed Income
        input_tab1, input_tab2 = st.tabs(["Market Rate Inputs", "Mixed Income Inputs"])

        with input_tab1:
            st.caption("Inputs for the market rate scenario.")
            render_inputs_form(prefix="", is_mixed_income=False)

        with input_tab2:
            st.caption("Inputs for the mixed income scenario. Values default to market rate on first load.")
            # Render tier selection and incentives first
            render_tier_and_incentives()
            st.divider()
            # Initialize mixed income values from market rate if not set
            _initialize_mixed_income_inputs()
            render_inputs_form(prefix="mixed_", is_mixed_income=True)

    elif is_single_mixed:
        # Single project mixed income - show sub-tabs
        input_tab1, input_tab2 = st.tabs(["Market Rate Inputs", "Mixed Income Inputs"])

        with input_tab1:
            st.caption("Base market rate inputs (for comparison reference).")
            render_inputs_form(prefix="", is_mixed_income=False)

        with input_tab2:
            st.caption("Inputs for your mixed income project.")
            # Render tier selection and incentives first
            render_tier_and_incentives()
            st.divider()
            _initialize_mixed_income_inputs()
            render_inputs_form(prefix="mixed_", is_mixed_income=True)

    else:
        # Single project market rate - just show one form
        st.caption("Project inputs for market rate scenario.")
        render_inputs_form(prefix="", is_mixed_income=False)


def _initialize_mixed_income_inputs():
    """Initialize mixed income inputs from market rate values if not already set."""
    # List of keys to copy from market rate to mixed income
    keys_to_copy = [
        "predevelopment_months", "construction_months", "leaseup_months", "operations_months",
        "land_cost", "hard_cost_per_unit", "soft_cost_pct",
        "predevelopment_cost_pct", "hard_cost_contingency_pct", "soft_cost_contingency_pct",
        "developer_fee_pct", "operating_reserve_months", "leaseup_reserve_months",
        "vacancy_rate_pct", "leaseup_pace_pct",
        "opex_utilities", "opex_maintenance", "opex_management_pct",
        "market_rent_growth_pct", "opex_growth_pct", "property_tax_growth_pct",
        "construction_rate_pct", "construction_ltc_pct",
        "perm_rate_pct", "perm_amort_years", "perm_ltv_max_pct", "perm_dscr_min",
        "exit_cap_rate_pct", "existing_assessed_value",
    ]

    for key in keys_to_copy:
        mixed_key = f"mixed_{key}"
        if mixed_key not in st.session_state and key in st.session_state:
            st.session_state[mixed_key] = st.session_state[key]

    # Set default for affordable rent growth if not present
    if "mixed_affordable_rent_growth_pct" not in st.session_state:
        st.session_state["mixed_affordable_rent_growth_pct"] = 1.0




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
            "Gross Potential Rent (Annual)",
            "Net Operating Income (Annual)",
            "Yield on Cost",
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
            f"${metrics.gpr_annual:,.0f}",
            f"${metrics.noi_annual:,.0f}",
            f"{metrics.yield_on_cost:.2%}",
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

    cfs = [cf.levered_cf for cf in result.periods]
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
    """Render the results comparison tab with granular developer metrics."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

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

    st.divider()

    # ========== DEVELOPER METRICS SECTION ==========
    st.subheader("Developer Metrics Analysis")

    # Helper function to calculate metrics from cash flows
    def calc_developer_metrics(result, metrics, label):
        """Calculate detailed developer metrics from DetailedCashFlowResult."""
        periods = result.periods
        total_units = metrics.total_units if hasattr(metrics, 'total_units') else st.session_state.get("target_units", 200)

        # Separate periods by phase (using header flags)
        predev_periods = [p for p in periods if p.header.is_predevelopment]
        construction_periods = [p for p in periods if p.header.is_construction]
        leaseup_periods = [p for p in periods if p.header.is_leaseup]
        ops_periods = [p for p in periods if p.header.is_operations and not p.header.is_reversion]
        reversion_period = [p for p in periods if p.header.is_reversion]

        # Total equity invested (sum of equity draws - negative values are outflows)
        total_equity_invested = abs(sum(p.equity.equity_drawn for p in periods if p.equity.equity_drawn < 0))

        # Total operating cash flows (levered)
        total_ops_levered_cf = sum(p.levered_cf for p in ops_periods + leaseup_periods)

        # Reversion proceeds
        reversion_proceeds = reversion_period[0].levered_cf if reversion_period else 0

        # Total returns to equity
        total_returns = total_ops_levered_cf + reversion_proceeds

        # Cash-on-cash by year (annualized operating cash flows)
        ops_cash_by_year = {}
        for p in ops_periods + leaseup_periods:
            year = (p.header.period - 1) // 12 + 1
            ops_cash_by_year[year] = ops_cash_by_year.get(year, 0) + p.levered_cf

        # Average annual cash-on-cash during operations
        if ops_cash_by_year and total_equity_invested > 0:
            avg_annual_coc = sum(ops_cash_by_year.values()) / len(ops_cash_by_year) / total_equity_invested
        else:
            avg_annual_coc = 0

        # DSCR calculations (during operations only)
        dscr_values = []
        for p in ops_periods:
            if p.permanent_debt.pmt_in_period > 0:
                dscr = p.operations.noi / p.permanent_debt.pmt_in_period
                dscr_values.append(dscr)

        avg_dscr = sum(dscr_values) / len(dscr_values) if dscr_values else 0
        min_dscr = min(dscr_values) if dscr_values else 0
        max_dscr = max(dscr_values) if dscr_values else 0

        # Debt metrics
        tdc = metrics.tdc
        debt = metrics.debt_amount
        equity = metrics.equity_required
        ltv = debt / (tdc * 1.0) if tdc > 0 else 0  # Debt to TDC as proxy
        debt_to_equity = debt / equity if equity > 0 else 0

        # Per unit metrics
        equity_per_unit = equity / total_units if total_units > 0 else 0
        debt_per_unit = debt / total_units if total_units > 0 else 0
        noi_per_unit = metrics.noi_annual / total_units if total_units > 0 else 0
        tdc_per_unit = tdc / total_units if total_units > 0 else 0

        # IRR Bifurcation - separate operations vs reversion contribution
        import numpy_financial as npf

        # Build full levered cash flow series
        full_levered_cfs = [p.levered_cf for p in periods]

        # Operations-only IRR: all cash flows except reversion
        # Replace reversion month with 0 to isolate operations contribution
        ops_only_cfs = full_levered_cfs.copy()
        if reversion_period:
            rev_month_idx = len(periods) - 1  # Reversion is last month
            ops_only_cfs[rev_month_idx] = 0

        try:
            ops_monthly_irr = npf.irr(ops_only_cfs)
            ops_only_irr = (1 + ops_monthly_irr) ** 12 - 1 if ops_monthly_irr and not np.isnan(ops_monthly_irr) else 0
        except:
            ops_only_irr = 0

        # Reversion-only IRR: equity contributions + reversion only (no operating cash flows)
        # Zero out all operating period positive cash flows
        rev_only_cfs = []
        for p in periods:
            if p.header.is_predevelopment or p.header.is_construction:
                # Keep equity contributions (negative cash flows during development)
                rev_only_cfs.append(p.levered_cf)
            elif p.header.is_reversion:
                # Keep reversion
                rev_only_cfs.append(p.levered_cf)
            else:
                # Zero out lease-up and operations
                rev_only_cfs.append(0)

        try:
            rev_monthly_irr = npf.irr(rev_only_cfs)
            rev_only_irr = (1 + rev_monthly_irr) ** 12 - 1 if rev_monthly_irr and not np.isnan(rev_monthly_irr) else 0
        except:
            rev_only_irr = 0

        # Total profit breakdown
        total_profit = total_returns - total_equity_invested
        ops_profit = total_ops_levered_cf
        rev_profit = reversion_proceeds - total_equity_invested  # Reversion minus equity return

        # Profit attribution percentages
        if total_profit > 0:
            ops_profit_pct = ops_profit / total_profit * 100
            rev_profit_pct = (total_profit - ops_profit) / total_profit * 100
        else:
            ops_profit_pct = 0
            rev_profit_pct = 0

        return {
            "label": label,
            "total_units": total_units,
            "total_equity_invested": total_equity_invested,
            "total_ops_levered_cf": total_ops_levered_cf,
            "reversion_proceeds": reversion_proceeds,
            "total_returns": total_returns,
            "avg_annual_coc": avg_annual_coc,
            "avg_dscr": avg_dscr,
            "min_dscr": min_dscr,
            "max_dscr": max_dscr,
            "ltv": ltv,
            "debt_to_equity": debt_to_equity,
            "equity_per_unit": equity_per_unit,
            "debt_per_unit": debt_per_unit,
            "noi_per_unit": noi_per_unit,
            "tdc_per_unit": tdc_per_unit,
            "tdc": tdc,
            "debt": debt,
            "equity": equity,
            "ops_cash_by_year": ops_cash_by_year,
            "ops_periods": ops_periods,
            "leaseup_periods": leaseup_periods,
            "reversion_period": reversion_period,
            "all_periods": periods,
            # IRR bifurcation
            "ops_only_irr": ops_only_irr,
            "rev_only_irr": rev_only_irr,
            "total_profit": total_profit,
            "ops_profit": ops_profit,
            "ops_profit_pct": ops_profit_pct,
            "rev_profit_pct": rev_profit_pct,
        }

    market_dev = calc_developer_metrics(market_result, market_metrics, "Market Rate")
    mixed_dev = calc_developer_metrics(mixed_result, mixed_metrics, "Mixed Income")

    # Create tabs for different metric views
    metrics_tab1, metrics_tab2, metrics_tab3, metrics_tab4 = st.tabs([
        "Returns to Equity", "Debt & Coverage", "Per Unit Analysis", "Cash Flow Charts"
    ])

    with metrics_tab1:
        st.markdown("#### Returns to Equity")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Market Rate**")
            st.metric("Total Equity Invested", f"${market_dev['total_equity_invested']:,.0f}")
            st.metric("Operating Period Returns", f"${market_dev['total_ops_levered_cf']:,.0f}")
            st.metric("Reversion Proceeds", f"${market_dev['reversion_proceeds']:,.0f}")
            st.metric("Total Returns to Equity", f"${market_dev['total_returns']:,.0f}")
            st.metric("Equity Multiple", f"{market_metrics.equity_multiple:.2f}x")
            st.metric("Avg Annual Cash-on-Cash", f"{market_dev['avg_annual_coc']:.1%}")

        with col2:
            st.markdown("**Mixed Income**")
            equity_diff = mixed_dev['total_equity_invested'] - market_dev['total_equity_invested']
            st.metric("Total Equity Invested", f"${mixed_dev['total_equity_invested']:,.0f}",
                     delta=f"${equity_diff:+,.0f}", delta_color="inverse")

            ops_diff = mixed_dev['total_ops_levered_cf'] - market_dev['total_ops_levered_cf']
            st.metric("Operating Period Returns", f"${mixed_dev['total_ops_levered_cf']:,.0f}",
                     delta=f"${ops_diff:+,.0f}")

            rev_diff = mixed_dev['reversion_proceeds'] - market_dev['reversion_proceeds']
            st.metric("Reversion Proceeds", f"${mixed_dev['reversion_proceeds']:,.0f}",
                     delta=f"${rev_diff:+,.0f}")

            total_diff = mixed_dev['total_returns'] - market_dev['total_returns']
            st.metric("Total Returns to Equity", f"${mixed_dev['total_returns']:,.0f}",
                     delta=f"${total_diff:+,.0f}")

            mult_diff = mixed_metrics.equity_multiple - market_metrics.equity_multiple
            st.metric("Equity Multiple", f"{mixed_metrics.equity_multiple:.2f}x",
                     delta=f"{mult_diff:+.2f}x")

            coc_diff = mixed_dev['avg_annual_coc'] - market_dev['avg_annual_coc']
            st.metric("Avg Annual Cash-on-Cash", f"{mixed_dev['avg_annual_coc']:.1%}",
                     delta=f"{coc_diff:+.1%}")

        # Profit distribution visualization
        st.markdown("#### Profit Distribution")

        fig_profit = go.Figure()

        categories = ['Operating Returns', 'Reversion Proceeds']
        market_values = [market_dev['total_ops_levered_cf'], market_dev['reversion_proceeds']]
        mixed_values = [mixed_dev['total_ops_levered_cf'], mixed_dev['reversion_proceeds']]

        fig_profit.add_trace(go.Bar(
            name='Market Rate',
            x=categories,
            y=market_values,
            marker_color='#1f77b4'
        ))
        fig_profit.add_trace(go.Bar(
            name='Mixed Income',
            x=categories,
            y=mixed_values,
            marker_color='#ff7f0e'
        ))

        fig_profit.update_layout(
            barmode='group',
            yaxis_tickformat="$,.0f",
            height=350,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        )

        st.plotly_chart(fig_profit, use_container_width=True)

        # IRR Bifurcation Section
        st.markdown("#### IRR Bifurcation: Operations vs Reversion")
        st.caption("Shows how much of the total IRR is driven by operating cash flows vs exit proceeds")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Market Rate**")
            st.metric("Total Levered IRR", f"{market_metrics.levered_irr:.2%}")
            st.metric("Operations-Only IRR", f"{market_dev['ops_only_irr']:.2%}",
                     help="IRR if the project had no exit/reversion")
            st.metric("Reversion-Only IRR", f"{market_dev['rev_only_irr']:.2%}",
                     help="IRR if there were no operating cash flows")
            st.divider()
            st.metric("Profit from Operations", f"{market_dev['ops_profit_pct']:.1f}%",
                     help="% of total profit attributable to operating cash flows")
            st.metric("Profit from Reversion", f"{market_dev['rev_profit_pct']:.1f}%",
                     help="% of total profit attributable to exit proceeds")

        with col2:
            st.markdown("**Mixed Income**")
            irr_diff = (mixed_metrics.levered_irr - market_metrics.levered_irr) * 10000
            st.metric("Total Levered IRR", f"{mixed_metrics.levered_irr:.2%}",
                     delta=f"{irr_diff:+.0f} bps")

            ops_irr_diff = (mixed_dev['ops_only_irr'] - market_dev['ops_only_irr']) * 10000
            st.metric("Operations-Only IRR", f"{mixed_dev['ops_only_irr']:.2%}",
                     delta=f"{ops_irr_diff:+.0f} bps",
                     help="IRR if the project had no exit/reversion")

            rev_irr_diff = (mixed_dev['rev_only_irr'] - market_dev['rev_only_irr']) * 10000
            st.metric("Reversion-Only IRR", f"{mixed_dev['rev_only_irr']:.2%}",
                     delta=f"{rev_irr_diff:+.0f} bps",
                     help="IRR if there were no operating cash flows")

            st.divider()

            ops_pct_diff = mixed_dev['ops_profit_pct'] - market_dev['ops_profit_pct']
            st.metric("Profit from Operations", f"{mixed_dev['ops_profit_pct']:.1f}%",
                     delta=f"{ops_pct_diff:+.1f}%", delta_color="off",
                     help="% of total profit attributable to operating cash flows")

            rev_pct_diff = mixed_dev['rev_profit_pct'] - market_dev['rev_profit_pct']
            st.metric("Profit from Reversion", f"{mixed_dev['rev_profit_pct']:.1f}%",
                     delta=f"{rev_pct_diff:+.1f}%", delta_color="off",
                     help="% of total profit attributable to exit proceeds")

        # IRR Bifurcation visualization
        st.markdown("#### IRR Breakdown Comparison")

        fig_irr = go.Figure()

        irr_categories = ['Operations-Only IRR', 'Reversion-Only IRR', 'Total IRR']
        market_irrs = [market_dev['ops_only_irr'] * 100, market_dev['rev_only_irr'] * 100,
                       market_metrics.levered_irr * 100]
        mixed_irrs = [mixed_dev['ops_only_irr'] * 100, mixed_dev['rev_only_irr'] * 100,
                      mixed_metrics.levered_irr * 100]

        fig_irr.add_trace(go.Bar(
            name='Market Rate',
            x=irr_categories,
            y=market_irrs,
            marker_color='#1f77b4',
            text=[f"{v:.1f}%" for v in market_irrs],
            textposition='outside'
        ))
        fig_irr.add_trace(go.Bar(
            name='Mixed Income',
            x=irr_categories,
            y=mixed_irrs,
            marker_color='#ff7f0e',
            text=[f"{v:.1f}%" for v in mixed_irrs],
            textposition='outside'
        ))

        fig_irr.update_layout(
            barmode='group',
            yaxis_title="IRR (%)",
            yaxis_ticksuffix="%",
            height=400,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        )

        st.plotly_chart(fig_irr, use_container_width=True)

        # Deal character assessment
        st.markdown("#### Deal Character Assessment")

        def assess_deal_character(ops_pct, rev_pct):
            """Determine if deal is yield-driven or appreciation-driven."""
            if ops_pct >= 60:
                return "Yield Play", "Returns primarily driven by operating cash flows"
            elif rev_pct >= 60:
                return "Appreciation Play", "Returns primarily driven by exit/sale proceeds"
            else:
                return "Balanced", "Returns split between operations and exit"

        market_char, market_desc = assess_deal_character(market_dev['ops_profit_pct'], market_dev['rev_profit_pct'])
        mixed_char, mixed_desc = assess_deal_character(mixed_dev['ops_profit_pct'], mixed_dev['rev_profit_pct'])

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Market Rate: {market_char}**\n\n{market_desc}")
        with col2:
            st.info(f"**Mixed Income: {mixed_char}**\n\n{mixed_desc}")

    with metrics_tab2:
        st.markdown("#### Debt Dependence & Coverage")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Market Rate**")
            st.metric("Debt Amount", f"${market_dev['debt']:,.0f}")
            st.metric("Debt-to-TDC", f"{(market_dev['debt']/market_dev['tdc']*100):.1f}%")
            st.metric("Debt-to-Equity", f"{market_dev['debt_to_equity']:.2f}x")
            st.divider()
            st.metric("Avg DSCR (Operations)", f"{market_dev['avg_dscr']:.2f}x")
            st.metric("Min DSCR", f"{market_dev['min_dscr']:.2f}x")
            st.metric("Max DSCR", f"{market_dev['max_dscr']:.2f}x")

        with col2:
            st.markdown("**Mixed Income**")
            debt_diff = mixed_dev['debt'] - market_dev['debt']
            st.metric("Debt Amount", f"${mixed_dev['debt']:,.0f}",
                     delta=f"${debt_diff:+,.0f}", delta_color="inverse")

            d2tdc_market = market_dev['debt']/market_dev['tdc']*100
            d2tdc_mixed = mixed_dev['debt']/mixed_dev['tdc']*100
            st.metric("Debt-to-TDC", f"{d2tdc_mixed:.1f}%",
                     delta=f"{d2tdc_mixed - d2tdc_market:+.1f}%", delta_color="inverse")

            dte_diff = mixed_dev['debt_to_equity'] - market_dev['debt_to_equity']
            st.metric("Debt-to-Equity", f"{mixed_dev['debt_to_equity']:.2f}x",
                     delta=f"{dte_diff:+.2f}x", delta_color="inverse")

            st.divider()

            dscr_diff = mixed_dev['avg_dscr'] - market_dev['avg_dscr']
            st.metric("Avg DSCR (Operations)", f"{mixed_dev['avg_dscr']:.2f}x",
                     delta=f"{dscr_diff:+.2f}x")

            min_dscr_diff = mixed_dev['min_dscr'] - market_dev['min_dscr']
            st.metric("Min DSCR", f"{mixed_dev['min_dscr']:.2f}x",
                     delta=f"{min_dscr_diff:+.2f}x")

            max_dscr_diff = mixed_dev['max_dscr'] - market_dev['max_dscr']
            st.metric("Max DSCR", f"{mixed_dev['max_dscr']:.2f}x",
                     delta=f"{max_dscr_diff:+.2f}x")

        # Capital stack comparison
        st.markdown("#### Capital Stack Comparison")

        fig_stack = make_subplots(rows=1, cols=2, specs=[[{'type':'pie'}, {'type':'pie'}]],
                                  subplot_titles=['Market Rate', 'Mixed Income'])

        # Market Rate capital stack
        market_tif = market_metrics.tif_value
        fig_stack.add_trace(go.Pie(
            labels=['Equity', 'Debt', 'TIF'],
            values=[market_dev['equity'], market_dev['debt'], market_tif],
            marker_colors=['#2ecc71', '#e74c3c', '#3498db'],
            textinfo='label+percent',
            textposition='inside',
            hole=0.3
        ), row=1, col=1)

        # Mixed Income capital stack
        mixed_tif = mixed_metrics.tif_value
        fig_stack.add_trace(go.Pie(
            labels=['Equity', 'Debt', 'TIF'],
            values=[mixed_dev['equity'], mixed_dev['debt'], mixed_tif],
            marker_colors=['#2ecc71', '#e74c3c', '#3498db'],
            textinfo='label+percent',
            textposition='inside',
            hole=0.3
        ), row=1, col=2)

        fig_stack.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_stack, use_container_width=True)

    with metrics_tab3:
        st.markdown("#### Per Unit Analysis")

        total_units = market_dev['total_units']

        per_unit_data = {
            "Metric": [
                "TDC per Unit",
                "Equity per Unit",
                "Debt per Unit",
                "Annual NOI per Unit",
                "Annual GPR per Unit",
            ],
            "Market Rate": [
                f"${market_dev['tdc_per_unit']:,.0f}",
                f"${market_dev['equity_per_unit']:,.0f}",
                f"${market_dev['debt_per_unit']:,.0f}",
                f"${market_dev['noi_per_unit']:,.0f}",
                f"${market_metrics.gpr_annual / total_units:,.0f}",
            ],
            "Mixed Income": [
                f"${mixed_dev['tdc_per_unit']:,.0f}",
                f"${mixed_dev['equity_per_unit']:,.0f}",
                f"${mixed_dev['debt_per_unit']:,.0f}",
                f"${mixed_dev['noi_per_unit']:,.0f}",
                f"${mixed_metrics.gpr_annual / total_units:,.0f}",
            ],
            "Difference": [
                f"${mixed_dev['tdc_per_unit'] - market_dev['tdc_per_unit']:+,.0f}",
                f"${mixed_dev['equity_per_unit'] - market_dev['equity_per_unit']:+,.0f}",
                f"${mixed_dev['debt_per_unit'] - market_dev['debt_per_unit']:+,.0f}",
                f"${mixed_dev['noi_per_unit'] - market_dev['noi_per_unit']:+,.0f}",
                f"${(mixed_metrics.gpr_annual - market_metrics.gpr_annual) / total_units:+,.0f}",
            ],
        }

        df_per_unit = pd.DataFrame(per_unit_data)
        st.dataframe(df_per_unit, use_container_width=True, hide_index=True)

        # Per unit bar chart
        st.markdown("#### Per Unit Comparison")

        metrics_names = ['TDC', 'Equity', 'Debt', 'Annual NOI']
        market_per_unit = [market_dev['tdc_per_unit'], market_dev['equity_per_unit'],
                          market_dev['debt_per_unit'], market_dev['noi_per_unit']]
        mixed_per_unit = [mixed_dev['tdc_per_unit'], mixed_dev['equity_per_unit'],
                         mixed_dev['debt_per_unit'], mixed_dev['noi_per_unit']]

        fig_per_unit = go.Figure()
        fig_per_unit.add_trace(go.Bar(
            name='Market Rate',
            x=metrics_names,
            y=market_per_unit,
            marker_color='#1f77b4'
        ))
        fig_per_unit.add_trace(go.Bar(
            name='Mixed Income',
            x=metrics_names,
            y=mixed_per_unit,
            marker_color='#ff7f0e'
        ))

        fig_per_unit.update_layout(
            barmode='group',
            yaxis_tickformat="$,.0f",
            yaxis_title="$ per Unit",
            height=350,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        )

        st.plotly_chart(fig_per_unit, use_container_width=True)

    with metrics_tab4:
        st.markdown("#### Operating Period Cash Flows")
        st.caption("Excludes reversion period to show operational detail")

        # Get operations-only cash flows (exclude reversion)
        market_ops_periods = market_dev['leaseup_periods'] + market_dev['ops_periods']
        mixed_ops_periods = mixed_dev['leaseup_periods'] + mixed_dev['ops_periods']

        market_ops_levered = [p.levered_cf for p in market_ops_periods]
        mixed_ops_levered = [p.levered_cf for p in mixed_ops_periods]
        ops_months = [p.header.period for p in market_ops_periods]

        fig_ops = go.Figure()
        fig_ops.add_trace(go.Scatter(
            x=ops_months, y=market_ops_levered,
            mode='lines', name='Market Rate',
            line=dict(color='#1f77b4', width=2)
        ))
        fig_ops.add_trace(go.Scatter(
            x=ops_months, y=mixed_ops_levered,
            mode='lines', name='Mixed Income',
            line=dict(color='#ff7f0e', width=2)
        ))

        fig_ops.update_layout(
            title="Levered Cash Flows - Lease-Up & Operations Only",
            xaxis_title="Month",
            yaxis_title="Cash Flow ($)",
            yaxis_tickformat="$,.0f",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            height=400,
        )

        st.plotly_chart(fig_ops, use_container_width=True)

        # Annual Cash-on-Cash comparison
        st.markdown("#### Annual Cash-on-Cash Returns")

        market_years = sorted(market_dev['ops_cash_by_year'].keys())
        mixed_years = sorted(mixed_dev['ops_cash_by_year'].keys())
        all_years = sorted(set(market_years) | set(mixed_years))

        if all_years and market_dev['total_equity_invested'] > 0 and mixed_dev['total_equity_invested'] > 0:
            market_coc = [market_dev['ops_cash_by_year'].get(y, 0) / market_dev['total_equity_invested'] * 100 for y in all_years]
            mixed_coc = [mixed_dev['ops_cash_by_year'].get(y, 0) / mixed_dev['total_equity_invested'] * 100 for y in all_years]

            fig_coc = go.Figure()
            fig_coc.add_trace(go.Bar(
                name='Market Rate',
                x=[f"Year {y}" for y in all_years],
                y=market_coc,
                marker_color='#1f77b4'
            ))
            fig_coc.add_trace(go.Bar(
                name='Mixed Income',
                x=[f"Year {y}" for y in all_years],
                y=mixed_coc,
                marker_color='#ff7f0e'
            ))

            fig_coc.update_layout(
                title="Annual Cash-on-Cash Return by Year",
                barmode='group',
                yaxis_title="Cash-on-Cash (%)",
                yaxis_ticksuffix="%",
                height=350,
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            )

            st.plotly_chart(fig_coc, use_container_width=True)

        # Reversion Comparison
        st.markdown("#### Exit / Reversion Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Market Rate**")
            if market_dev['reversion_period']:
                rev_p = market_dev['reversion_period'][0]
                sale_proceeds = rev_p.investment.reversion
                loan_payoff = rev_p.permanent_debt.payoff
                net_to_equity = rev_p.levered_cf
                st.metric("Sale Proceeds", f"${sale_proceeds:,.0f}")
                st.metric("Loan Payoff", f"${loan_payoff:,.0f}")
                st.metric("Net to Equity", f"${net_to_equity:,.0f}")

        with col2:
            st.markdown("**Mixed Income**")
            if mixed_dev['reversion_period']:
                rev_p = mixed_dev['reversion_period'][0]
                sale_proceeds = rev_p.investment.reversion
                loan_payoff = rev_p.permanent_debt.payoff
                net_to_equity = rev_p.levered_cf

                market_rev = market_dev['reversion_period'][0] if market_dev['reversion_period'] else None
                market_sale = market_rev.investment.reversion if market_rev else 0
                market_payoff = market_rev.permanent_debt.payoff if market_rev else 0
                market_net = market_rev.levered_cf if market_rev else 0

                sale_diff = sale_proceeds - market_sale
                st.metric("Sale Proceeds", f"${sale_proceeds:,.0f}",
                         delta=f"${sale_diff:+,.0f}")

                payoff_diff = loan_payoff - market_payoff
                st.metric("Loan Payoff", f"${loan_payoff:,.0f}",
                         delta=f"${payoff_diff:+,.0f}", delta_color="inverse")

                net_diff = net_to_equity - market_net
                st.metric("Net to Equity", f"${net_to_equity:,.0f}",
                         delta=f"${net_diff:+,.0f}")

        # Reversion vs TDC comparison
        st.markdown("#### Exit Value vs Development Cost")

        fig_exit = go.Figure()

        scenarios = ['Market Rate', 'Mixed Income']
        tdc_values = [market_dev['tdc'], mixed_dev['tdc']]

        market_sale = market_dev['reversion_period'][0].investment.reversion if market_dev['reversion_period'] else 0
        mixed_sale = mixed_dev['reversion_period'][0].investment.reversion if mixed_dev['reversion_period'] else 0
        sale_values = [market_sale, mixed_sale]

        fig_exit.add_trace(go.Bar(
            name='Total Development Cost',
            x=scenarios,
            y=tdc_values,
            marker_color='#e74c3c'
        ))
        fig_exit.add_trace(go.Bar(
            name='Exit Sale Value',
            x=scenarios,
            y=sale_values,
            marker_color='#2ecc71'
        ))

        fig_exit.update_layout(
            barmode='group',
            yaxis_tickformat="$,.0f",
            height=350,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        )

        st.plotly_chart(fig_exit, use_container_width=True)

    # ========== DIAGNOSTIC SECTION ==========
    with st.expander("🔧 IRR Diagnostic (Compare with Excel)", expanded=False):
        st.markdown("""
        **Use this section to compare key values with your Excel model to identify discrepancies.**
        """)

        # Get timeline info
        predev_months = st.session_state.get("predevelopment_months", 18)
        const_months = st.session_state.get("construction_months", 24)
        leaseup_months = st.session_state.get("leaseup_months", 12)
        ops_months = st.session_state.get("operations_months", 60)
        total_months = predev_months + const_months + leaseup_months + ops_months

        st.markdown("#### Timeline")
        st.markdown(f"- Predevelopment: Months 1-{predev_months}")
        st.markdown(f"- Construction: Months {predev_months+1}-{predev_months+const_months}")
        st.markdown(f"- Lease-up: Months {predev_months+const_months+1}-{predev_months+const_months+leaseup_months}")
        st.markdown(f"- Operations: Months {predev_months+const_months+leaseup_months+1}-{total_months}")
        st.markdown("- **Equity invested first** (land at Month 1, then soft costs through predevelopment, then construction until exhausted)")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Market Rate")
            st.markdown(f"**TDC:** ${market_metrics.tdc:,.0f}")
            st.markdown(f"**Construction Loan:** ${market_metrics.debt_amount:,.0f}")
            st.markdown(f"**Equity Required:** ${market_metrics.equity_required:,.0f}")
            st.markdown(f"**TIF Lump Sum:** ${market_metrics.tif_value:,.0f}")
            st.divider()
            st.markdown(f"**Annual GPR:** ${market_metrics.gpr_annual:,.0f}")
            st.markdown(f"**Annual NOI:** ${market_metrics.noi_annual:,.0f}")
            st.markdown(f"**Yield on Cost:** {market_metrics.yield_on_cost:.2%}")
            st.divider()
            if market_dev['reversion_period']:
                rev = market_dev['reversion_period'][0]
                st.markdown(f"**Sale Price:** ${rev.investment.reversion:,.0f}")
                st.markdown(f"**Loan Payoff:** ${rev.permanent_debt.payoff:,.0f}")
                st.markdown(f"**Net to Equity:** ${rev.levered_cf:,.0f}")
            st.divider()
            st.markdown(f"**Levered IRR:** {market_metrics.levered_irr:.2%}")
            st.markdown(f"**Unlevered IRR:** {market_metrics.unlevered_irr:.2%}")

        with col2:
            st.markdown("#### Mixed Income")
            st.markdown(f"**TDC:** ${mixed_metrics.tdc:,.0f}")
            st.markdown(f"**Construction Loan:** ${mixed_metrics.debt_amount:,.0f}")
            st.markdown(f"**Equity Required:** ${mixed_metrics.equity_required:,.0f}")
            st.markdown(f"**TIF Lump Sum:** ${mixed_metrics.tif_value:,.0f}")
            st.divider()
            st.markdown(f"**Annual GPR:** ${mixed_metrics.gpr_annual:,.0f}")
            st.markdown(f"**Annual NOI:** ${mixed_metrics.noi_annual:,.0f}")
            st.markdown(f"**Yield on Cost:** {mixed_metrics.yield_on_cost:.2%}")
            st.divider()
            if mixed_dev['reversion_period']:
                rev = mixed_dev['reversion_period'][0]
                st.markdown(f"**Sale Price:** ${rev.investment.reversion:,.0f}")
                st.markdown(f"**Loan Payoff:** ${rev.permanent_debt.payoff:,.0f}")
                st.markdown(f"**Net to Equity:** ${rev.levered_cf:,.0f}")
            st.divider()
            st.markdown(f"**Levered IRR:** {mixed_metrics.levered_irr:.2%}")
            st.markdown(f"**Unlevered IRR:** {mixed_metrics.unlevered_irr:.2%}")

        st.divider()
        st.markdown("#### Cash Flow Summary (First 5 Years of Operations)")

        # Show monthly CF summary
        ops_start = predev_months + const_months + leaseup_months + 1
        market_ops = [p for p in market_result.periods if p.header.is_operations and not p.header.is_reversion][:60]
        mixed_ops = [p for p in mixed_result.periods if p.header.is_operations and not p.header.is_reversion][:60]

        if market_ops:
            # Annual totals
            market_annual = {}
            mixed_annual = {}
            for i, (m_p, x_p) in enumerate(zip(market_ops, mixed_ops)):
                year = i // 12 + 1
                market_annual[year] = market_annual.get(year, 0) + m_p.levered_cf
                mixed_annual[year] = mixed_annual.get(year, 0) + x_p.levered_cf

            cf_data = {
                "Year": list(market_annual.keys()),
                "Market Levered CF": [f"${v:,.0f}" for v in market_annual.values()],
                "Mixed Levered CF": [f"${v:,.0f}" for v in mixed_annual.values()],
            }
            st.dataframe(pd.DataFrame(cf_data), use_container_width=True, hide_index=True)

        st.info("""
        **Equity Draw Waterfall:** Equity is drawn first starting at Month 1 (land acquisition),
        then soft costs through predevelopment, then construction costs until exhausted.
        Debt kicks in only after equity is fully deployed. This matches standard real estate
        development funding where sponsor equity is drawn down first to protect lender.
        """)


def render_matrix_tab(inputs: ProjectInputs):
    """Render the scenario matrix analysis tab.

    This tab is for testing multiple tier/incentive combinations to find
    optimal packages. Primary incentive configuration is on the Scenarios tab.
    """
    st.header("Scenario Matrix Analysis")
    st.markdown("""
    Test multiple incentive tier and toggle combinations to find packages that meet your target returns.
    Configure your primary scenario on the **Scenarios** tab.
    """)

    # Current configuration summary
    current_tier = st.session_state.get("selected_tier", 2)
    affordable_pct_raw = st.session_state.get("affordable_pct", 20.0)
    affordable_pct = affordable_pct_raw if affordable_pct_raw <= 1 else affordable_pct_raw / 100
    ami_level = st.session_state.get("ami_level", "50%")

    st.info(f"**Current Configuration:** Tier {current_tier} | {affordable_pct:.0%} Affordable | {ami_level} AMI")

    st.divider()

    # Matrix configuration
    st.subheader("Matrix Configuration")

    col1, col2 = st.columns(2)

    with col1:
        tier_options = st.multiselect(
            "Tiers to Test",
            options=[1, 2, 3],
            default=[1, 2, 3],
            format_func=lambda x: f"Tier {x}",
            help="Select which incentive tiers to include in the matrix"
        )

    with col2:
        st.caption("**Tier Descriptions:**")
        st.caption("- Tier 1: 5% @ 30% AMI (deep affordability)")
        st.caption("- Tier 2: 20% @ 50% AMI (moderate, more units)")
        st.caption("- Tier 3: 10% @ 50% AMI (moderate, balanced)")

    st.divider()

    # Run button
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

        st.subheader("Results")

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

        # Results table with explicit incentive columns
        table_data = []
        for i, r in enumerate(results, 1):
            combo = r.combination

            # Get tier value (IncentiveTier enum has .value)
            tier_num = combo.tier.value if hasattr(combo.tier, 'value') else combo.tier

            table_data.append({
                "Rank": i,
                "Tier": tier_num,
                "TIF Lump": "✓" if combo.tif_lump_sum else "",
                "TIF Stream": "✓" if combo.tif_stream else "",
                "SMART": "✓" if combo.smart_fee_waiver else "",
                "Abatement": "✓" if combo.tax_abatement else "",
                "Buydown": "✓" if combo.interest_buydown else "",
                "Mkt IRR": r.market_metrics.levered_irr * 100,  # Convert to percentage
                "Mix IRR": r.mixed_metrics.levered_irr * 100,
                "Δ bps": r.comparison.irr_difference_bps,
                "Target": "✓" if r.comparison.meets_target else "",
            })

        df = pd.DataFrame(table_data)

        # Style the dataframe with sortable columns
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn("#", width="small"),
                "Tier": st.column_config.NumberColumn("Tier", width="small"),
                "TIF Lump": st.column_config.TextColumn("TIF Lump", width="small"),
                "TIF Stream": st.column_config.TextColumn("TIF Stream", width="small"),
                "SMART": st.column_config.TextColumn("SMART", width="small"),
                "Abatement": st.column_config.TextColumn("Abate", width="small"),
                "Buydown": st.column_config.TextColumn("Buydown", width="small"),
                "Mkt IRR": st.column_config.NumberColumn("Mkt IRR", width="small", format="%.2f%%"),
                "Mix IRR": st.column_config.NumberColumn("Mix IRR", width="small", format="%.2f%%"),
                "Δ bps": st.column_config.NumberColumn("Δ bps", width="small", format="%+d"),
                "Target": st.column_config.TextColumn("✓?", width="small"),
            }
        )

        # Legend
        st.caption("✓ = Incentive active | Click column headers to sort | TIF and Abatement are mutually exclusive")

        # Best combination details
        if results and results[0].comparison.meets_target:
            st.success(f"""
            **Best Combination: {results[0].combination.name}**

            This combination achieves **{results[0].comparison.irr_difference_bps:+d} bps**
            improvement over market rate.
            """)

            # Option to apply best combination
            if st.button("Apply Best Combination to Scenarios Tab"):
                best = results[0]
                # Extract tier from combination name and update session state
                st.session_state["matrix_applied"] = True
                st.info("Navigate to the **Scenarios** tab to see the updated configuration.")


def render_detailed_cashflow_tab(inputs: ProjectInputs):
    """Render the detailed cash flow tab with separate Market Rate and Mixed Income views.

    Uses the cached results from run_analysis() - the SINGLE SOURCE OF TRUTH.
    No recalculation or overriding needed since calculate_deal() already computed
    all periods and derived metrics.
    """
    st.subheader("Detailed Cash Flow Analysis")

    # Get cached results from run_analysis (single source of truth)
    # These are DetailedCashFlowResult objects with full period data
    market_result = st.session_state.get("_cached_market_result")
    mixed_result = st.session_state.get("_cached_mixed_result")

    if not market_result or not mixed_result:
        st.warning("Calculating results... Please wait.")
        return

    # Get TIF lump sum for display
    tif_lump_sum = st.session_state.get("calculated_tif_lump_sum", 0)
    tif_enabled = st.session_state.get("tif_lump_sum", True)

    try:
        # =====================================================================
        # COMPARISON OPERATING STATEMENT
        # =====================================================================
        st.markdown("### Stabilized Operating Statement Comparison")

        _render_operating_statement_comparison(market_result, mixed_result, inputs.target_units)

        st.divider()

        # =====================================================================
        # SUB-TABS FOR DETAILED CASH FLOWS
        # =====================================================================
        cf_tab1, cf_tab2 = st.tabs(["Market Rate Cash Flows", "Mixed Income Cash Flows"])

        with cf_tab1:
            _render_scenario_cashflow(
                result=market_result,
                scenario_name="Market Rate",
                inputs=inputs,
                has_incentives=False,
                tif_lump_sum=0,
                key_prefix="detailed_market",
            )

        with cf_tab2:
            _render_scenario_cashflow(
                result=mixed_result,
                scenario_name="Mixed Income",
                inputs=inputs,
                has_incentives=True,
                tif_lump_sum=tif_lump_sum if tif_enabled else 0,
                key_prefix="detailed_mixed",
            )

    except Exception as e:
        st.error(f"Error rendering detailed cash flow: {e}")
        import traceback
        st.code(traceback.format_exc())


def _render_operating_statement_comparison(market_result, mixed_result, total_units: int):
    """Render side-by-side operating statement comparison."""
    # Get stabilized period data
    market_stab = market_result.periods[-1] if market_result.periods else None
    mixed_stab = mixed_result.periods[-1] if mixed_result.periods else None

    if not market_stab or not mixed_stab:
        st.warning("No stabilized period data available")
        return

    # Build comparison data - use operations fields which are floats
    def get_annual(val):
        return val * 12 if val else 0

    # Extract values from the nested dataclass structure
    market_ops = market_stab.operations
    mixed_ops = mixed_stab.operations

    # Get debt service from permanent debt row (pmt_in_period = total payment)
    market_debt_svc = market_stab.permanent_debt.pmt_in_period if market_stab.permanent_debt else 0
    mixed_debt_svc = mixed_stab.permanent_debt.pmt_in_period if mixed_stab.permanent_debt else 0

    rows = [
        ("Gross Potential Rent", get_annual(market_ops.gpr), get_annual(mixed_ops.gpr)),
        ("Less: Vacancy", get_annual(market_ops.less_vacancy), get_annual(mixed_ops.less_vacancy)),
        ("Effective Gross Income", get_annual(market_ops.egi), get_annual(mixed_ops.egi)),
        ("", None, None),  # Spacer
        ("Operating Expenses", get_annual(market_ops.less_opex_ex_taxes), get_annual(mixed_ops.less_opex_ex_taxes)),
        ("Property Taxes", get_annual(market_ops.less_property_taxes), get_annual(mixed_ops.less_property_taxes)),
        ("Net Operating Income", get_annual(market_ops.noi), get_annual(mixed_ops.noi)),
        ("", None, None),  # Spacer
        ("Debt Service", -get_annual(market_debt_svc), -get_annual(mixed_debt_svc)),
        ("Cash Flow Before Tax", get_annual(market_stab.levered_cf), get_annual(mixed_stab.levered_cf)),
    ]

    col1, col2, col3, col4 = st.columns([2, 1.2, 1.2, 1.2])

    with col1:
        st.markdown("**Line Item**")
    with col2:
        st.markdown("**Market Rate**")
    with col3:
        st.markdown("**Mixed Income**")
    with col4:
        st.markdown("**Difference**")

    for label, market_val, mixed_val in rows:
        col1, col2, col3, col4 = st.columns([2, 1.2, 1.2, 1.2])

        with col1:
            if label == "":
                st.markdown("---")
            elif label in ["Effective Gross Income", "Net Operating Income", "Cash Flow Before Tax"]:
                st.markdown(f"**{label}**")
            else:
                st.markdown(label)

        if market_val is not None:
            diff = mixed_val - market_val

            with col2:
                if label in ["Effective Gross Income", "Net Operating Income", "Cash Flow Before Tax"]:
                    st.markdown(f"**${market_val:,.0f}**")
                else:
                    st.markdown(f"${market_val:,.0f}")

            with col3:
                if label in ["Effective Gross Income", "Net Operating Income", "Cash Flow Before Tax"]:
                    st.markdown(f"**${mixed_val:,.0f}**")
                else:
                    st.markdown(f"${mixed_val:,.0f}")

            with col4:
                color = "green" if diff >= 0 else "red"
                if label in ["Effective Gross Income", "Net Operating Income", "Cash Flow Before Tax"]:
                    st.markdown(f"**:{color}[${diff:+,.0f}]**")
                else:
                    st.markdown(f":{color}[${diff:+,.0f}]")
        else:
            with col2:
                st.markdown("---")
            with col3:
                st.markdown("---")
            with col4:
                st.markdown("---")

    # Summary metrics
    st.divider()
    col1, col2, col3, col4 = st.columns(4)

    market_noi = get_annual(market_stab.operations.noi)
    mixed_noi = get_annual(mixed_stab.operations.noi)

    with col1:
        st.metric("Market NOI", f"${market_noi:,.0f}")
    with col2:
        st.metric("Mixed NOI", f"${mixed_noi:,.0f}")
    with col3:
        noi_diff = mixed_noi - market_noi
        st.metric("NOI Difference", f"${noi_diff:+,.0f}",
                 delta=f"{noi_diff/market_noi*100:+.1f}%" if market_noi else None,
                 delta_color="off")
    with col4:
        st.metric("NOI/Unit (Mixed)", f"${mixed_noi/total_units:,.0f}")


def _render_scenario_cashflow(result, scenario_name: str, inputs: ProjectInputs,
                              has_incentives: bool, tif_lump_sum: float,
                              key_prefix: str = "detailed"):
    """Render detailed cash flow for a single scenario."""
    # Calculate equity multiple
    total_equity_out = abs(sum(p.levered_cf for p in result.periods if p.levered_cf < 0))
    total_distributions = sum(p.levered_cf for p in result.periods if p.levered_cf > 0)
    equity_multiple = total_distributions / total_equity_out if total_equity_out > 0 else 0

    # Deal summary header
    st.markdown(f"#### {scenario_name} Summary")
    render_deal_summary_header(
        tdc=result.sources_uses.tdc,
        units=inputs.target_units,
        equity=result.sources_uses.equity,
        senior_debt=result.sources_uses.construction_loan,
        levered_irr=result.levered_irr,
        equity_multiple=equity_multiple,
    )

    if has_incentives and tif_lump_sum > 0:
        st.success(f"**TIF Lump Sum Included:** ${tif_lump_sum:,.0f}")

    st.divider()

    # Sources & Uses with Drawdown Schedule
    st.markdown("#### Sources & Uses")
    render_sources_uses(result.sources_uses, key_prefix=key_prefix)

    st.divider()

    # Sources Drawdown Schedule
    st.markdown("#### Capital Drawdown Schedule")
    _render_drawdown_schedule(result, tif_lump_sum if has_incentives else 0)

    st.divider()

    # IRR Summary
    st.markdown("#### Return Metrics")
    render_irr_summary(result)

    st.divider()

    # Performance Metrics Tracking
    st.markdown("#### Performance Metrics by Period")
    _render_performance_metrics(result)

    st.divider()

    # Period aggregation selector
    aggregation = st.radio(
        "View by",
        options=["monthly", "quarterly", "annual"],
        index=2,  # Default to annual
        key=f"cf_aggregation_{scenario_name.lower().replace(' ', '_')}",
        horizontal=True
    )

    # Render detailed cash flow table
    is_mixed = scenario_name.lower() == "mixed income"
    render_detailed_cashflow_table(
        result,
        aggregation=aggregation,
        is_mixed_income=is_mixed,
        tif_lump_sum=tif_lump_sum if has_incentives and is_mixed else 0.0
    )

    st.divider()

    # Exit waterfall
    render_exit_waterfall(result)

    # Export buttons
    st.markdown("#### Export Data")
    col1, col2 = st.columns(2)

    with col1:
        csv_data = export_cashflow_to_csv(result)
        st.download_button(
            label=f"📄 Export {scenario_name} CSV",
            data=csv_data,
            file_name=f"cashflow_{scenario_name.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            key=f"export_csv_{scenario_name.lower().replace(' ', '_')}",
        )

    with col2:
        xlsx_data = export_cashflow_to_excel(result, scenario_name)
        st.download_button(
            label=f"📊 Export {scenario_name} Excel",
            data=xlsx_data,
            file_name=f"cashflow_{scenario_name.lower().replace(' ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"export_xlsx_{scenario_name.lower().replace(' ', '_')}",
        )


def _render_drawdown_schedule(result, tif_lump_sum: float):
    """Render sources drawdown schedule showing period-by-period funding."""
    import pandas as pd

    periods = result.periods
    if not periods:
        st.warning("No period data available")
        return

    # Get total sources
    su = result.sources_uses
    total_equity = su.equity
    total_debt = su.construction_loan
    total_tif = tif_lump_sum

    # Build drawdown tracking
    # Assumes: Equity first, then TIF (if any), then Debt
    drawdown_data = []
    cumulative_equity = 0
    cumulative_tif = 0
    cumulative_debt = 0
    cumulative_uses = 0

    # Group by quarter for readability
    quarters = {}
    for p in periods:
        if p.header.period <= result.construction_end:
            q = (p.header.period - 1) // 3 + 1
            if q not in quarters:
                quarters[q] = {"uses": 0, "periods": []}
            quarters[q]["uses"] += abs(min(0, p.investment.unlevered_cf))
            quarters[q]["periods"].append(p)

    # Simplified drawdown logic
    remaining_equity = total_equity
    remaining_tif = total_tif
    remaining_debt = total_debt

    for q in sorted(quarters.keys())[:12]:  # Show first 12 quarters (3 years)
        uses_this_q = quarters[q]["uses"]

        # Draw from equity first
        equity_draw = min(remaining_equity, uses_this_q)
        remaining_equity -= equity_draw
        cumulative_equity += equity_draw
        uses_remaining = uses_this_q - equity_draw

        # Then TIF
        tif_draw = min(remaining_tif, uses_remaining)
        remaining_tif -= tif_draw
        cumulative_tif += tif_draw
        uses_remaining -= tif_draw

        # Then debt
        debt_draw = min(remaining_debt, uses_remaining)
        remaining_debt -= debt_draw
        cumulative_debt += debt_draw

        cumulative_uses += uses_this_q

        drawdown_data.append({
            "Quarter": f"Q{q}",
            "Uses": uses_this_q,
            "Equity Draw": equity_draw,
            "TIF Draw": tif_draw,
            "Debt Draw": debt_draw,
            "Cum. Equity": cumulative_equity,
            "Cum. TIF": cumulative_tif,
            "Cum. Debt": cumulative_debt,
            "Equity Avail.": remaining_equity,
            "TIF Avail.": remaining_tif,
            "Debt Avail.": remaining_debt,
        })

    df = pd.DataFrame(drawdown_data)

    # Format for display
    for col in df.columns:
        if col != "Quarter":
            df[col] = df[col].apply(lambda x: f"${x:,.0f}" if x > 0 else "-")

    st.dataframe(df, use_container_width=True, hide_index=True, height=300)

    # Summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Equity", f"${total_equity:,.0f}")
    with col2:
        st.metric("Total TIF", f"${total_tif:,.0f}" if total_tif > 0 else "$0")
    with col3:
        st.metric("Total Debt", f"${total_debt:,.0f}")
    with col4:
        st.metric("Total Sources", f"${total_equity + total_tif + total_debt:,.0f}")


def _render_performance_metrics(result):
    """Render DSCR, ROI, ROE, Cash-on-Cash tracking by period."""
    import pandas as pd

    periods = result.periods
    if not periods:
        st.warning("No period data available")
        return

    # Get equity for CoC calculation
    total_equity = result.sources_uses.equity

    # Build metrics by year
    metrics_data = []
    years_seen = set()

    for p in periods:
        year = p.header.period // 12 + 1
        if year in years_seen or year > 10:  # Show max 10 years
            continue

        # Find all periods in this year
        year_periods = [per for per in periods if per.header.period // 12 + 1 == year]
        if not year_periods:
            continue

        years_seen.add(year)

        # Annual metrics
        annual_noi = sum(per.operations.noi for per in year_periods)
        annual_ds = sum(per.permanent_debt.pmt_in_period for per in year_periods)
        annual_cf = sum(per.levered_cf for per in year_periods)

        # DSCR = NOI / Debt Service
        dscr = annual_noi / annual_ds if annual_ds > 0 else 0

        # Cash-on-Cash = Annual CF / Equity
        coc = annual_cf / total_equity if total_equity > 0 else 0

        # Cumulative metrics for ROI/ROE
        cumulative_cf = sum(per.levered_cf for per in periods if per.header.period <= year_periods[-1].header.period)

        metrics_data.append({
            "Year": year,
            "Annual NOI": annual_noi,
            "Debt Service": annual_ds,
            "DSCR": dscr,
            "Annual CF": annual_cf,
            "Cash-on-Cash": coc,
            "Cumulative CF": cumulative_cf,
        })

    if not metrics_data:
        return

    df = pd.DataFrame(metrics_data)

    # Format for display
    df_display = df.copy()
    df_display["Annual NOI"] = df_display["Annual NOI"].apply(lambda x: f"${x:,.0f}")
    df_display["Debt Service"] = df_display["Debt Service"].apply(lambda x: f"${x:,.0f}")
    df_display["DSCR"] = df_display["DSCR"].apply(lambda x: f"{x:.2f}x")
    df_display["Annual CF"] = df_display["Annual CF"].apply(lambda x: f"${x:,.0f}")
    df_display["Cash-on-Cash"] = df_display["Cash-on-Cash"].apply(lambda x: f"{x:.1%}")
    df_display["Cumulative CF"] = df_display["Cumulative CF"].apply(lambda x: f"${x:,.0f}")

    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # Chart: DSCR over time
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**DSCR Trend**")
        chart_df = df[["Year", "DSCR"]].set_index("Year")
        st.line_chart(chart_df, height=200)

    with col2:
        st.markdown("**Cash-on-Cash Return**")
        chart_df = df[["Year", "Cash-on-Cash"]].set_index("Year")
        chart_df["Cash-on-Cash"] = chart_df["Cash-on-Cash"] * 100  # Convert to %
        st.line_chart(chart_df, height=200)


def render_property_tax_tab(inputs: ProjectInputs):
    """Render the property tax engine tab with TIF analysis."""
    # Import TIF analysis components
    from ui.components.tif_analysis_view import (
        render_tax_rate_breakdown,
        render_smart_fee_waiver,
        render_city_tax_abatement,
    )
    import pandas as pd

    st.subheader("Property Tax & TIF Analysis")

    # Get Austin tax stack
    tax_stack = get_austin_tax_stack()

    # Calculate TDC (Total Development Cost)
    hard_costs = inputs.target_units * inputs.hard_cost_per_unit
    soft_costs = hard_costs * inputs.soft_cost_pct
    land_cost = inputs.land_cost  # Simplified: 1 acre

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

        # Show selected tier info
        selected_tier = st.session_state.get("selected_tier", 2)
        affordable_pct_display = st.session_state.get("affordable_pct", 20.0)
        if affordable_pct_display > 1:
            affordable_pct_display = affordable_pct_display  # Already percentage
        else:
            affordable_pct_display = affordable_pct_display * 100  # Convert to percentage
        ami_level = st.session_state.get("ami_level", "50%")

        st.info(f"**Using Tier {selected_tier}** from Scenarios tab: {affordable_pct_display:.0f}% affordable at {ami_level} AMI")

        # Inputs for TIF calculation
        col1, col2, col3 = st.columns(3)

        # Helper to parse currency input (handles commas)
        def parse_currency(val: str) -> float:
            if not val:
                return 0.0
            return float(val.replace(",", "").replace("$", "").strip() or 0)

        with col1:
            # Initialize with formatted value
            if "tif_base_value_str" not in st.session_state:
                st.session_state["tif_base_value_str"] = f"{inputs.existing_assessed_value:,.0f}"
            else:
                # Reformat to ensure commas on each render
                current = parse_currency(st.session_state["tif_base_value_str"])
                st.session_state["tif_base_value_str"] = f"{current:,.0f}"

            tif_base_str = st.text_input(
                "Base Assessed Value",
                key="tif_base_value_str",
                help="Assessed value before development",
            )
            tif_base_value = parse_currency(tif_base_str)

        with col2:
            # Initialize with formatted value
            if "tif_new_value_str" not in st.session_state:
                st.session_state["tif_new_value_str"] = f"{tdc:,.0f}"
            else:
                # Reformat to ensure commas on each render
                current = parse_currency(st.session_state["tif_new_value_str"])
                st.session_state["tif_new_value_str"] = f"{current:,.0f}"

            tif_new_str = st.text_input(
                "New Assessed Value (TDC)",
                key="tif_new_value_str",
                help="Typically equals TDC",
            )
            tif_new_value = parse_currency(tif_new_str)

        with col3:
            # Get affordable % from session state (set by Scenarios tab tier selection)
            affordable_pct_raw = st.session_state.get("affordable_pct", 20.0)
            affordable_pct = affordable_pct_raw / 100.0 if affordable_pct_raw > 1 else affordable_pct_raw
            calculated_affordable_units = max(1, int(inputs.target_units * affordable_pct))

            # Detect tier change and reset affordable units
            prev_tier = st.session_state.get("_tif_prev_tier", None)
            current_tier = st.session_state.get("selected_tier", 2)
            if prev_tier is not None and prev_tier != current_tier:
                # Tier changed - update affordable units to match new tier
                st.session_state["tif_affordable_units"] = calculated_affordable_units
            st.session_state["_tif_prev_tier"] = current_tier

            tif_affordable_units = st.number_input(
                "Affordable Units",
                min_value=1,
                value=calculated_affordable_units,
                step=1,
                key="tif_affordable_units",
                help=f"Based on {affordable_pct:.0%} affordable from Scenarios tab tier selection",
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            tif_term = st.number_input(
                "TIF Term (years)",
                min_value=5,
                max_value=40,
                value=20,
                step=1,
                key="prop_tif_term",
                help="Number of years for TIF increment stream",
            )

        with col2:
            tif_cap_rate = st.slider(
                "TIF Cap Rate",
                min_value=5.0,
                max_value=15.0,
                value=9.5,
                step=0.25,
                format="%.2f%%",
                key="prop_tif_cap_rate",
                help="Cap rate for capitalizing annual tax increment to lump sum",
            ) / 100

        with col3:
            tif_escalation_rate = st.slider(
                "Increment Escalation Rate",
                min_value=0.0,
                max_value=5.0,
                value=2.0,
                step=0.25,
                format="%.2f%%",
                key="prop_tif_escalation",
                help="Annual growth rate for tax increment",
            ) / 100

        st.divider()

        # Calculate TIF increment cashflows
        incremental_value = tif_new_value - tif_base_value
        city_rate = tax_stack.tif_participating_rate_decimal
        annual_city_increment = incremental_value * city_rate

        # Show calculation summary
        st.markdown("### TIF Calculation")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Incremental Value", f"${incremental_value:,.0f}",
                     help="New AV - Base AV")
        with col2:
            st.metric("City (TIF) Rate", f"{city_rate:.4%}",
                     help="Combined rate of TIF-participating authorities")
        with col3:
            st.metric("Year 1 City Increment", f"${annual_city_increment:,.0f}",
                     help="Annual tax increment captured by TIF")

        st.divider()

        # Build and display TIF cashflow schedule
        st.markdown("### TIF Increment Cashflow Schedule")
        st.caption("Shows the tax increment stream over the TIF term in nominal and present value terms.")

        tif_cashflows = []
        total_nominal = 0
        total_pv = 0

        for year in range(1, tif_term + 1):
            # Escalate the increment
            year_increment = annual_city_increment * ((1 + tif_escalation_rate) ** (year - 1))
            # PV using cap rate as discount
            pv_factor = 1 / ((1 + tif_cap_rate) ** year)
            year_pv = year_increment * pv_factor

            total_nominal += year_increment
            total_pv += year_pv

            tif_cashflows.append({
                "Year": year,
                "Increment (Nominal)": year_increment,
                "PV Factor": pv_factor,
                "Increment (PV)": year_pv,
                "Cumulative Nominal": total_nominal,
                "Cumulative PV": total_pv,
            })

        cf_df = pd.DataFrame(tif_cashflows)

        # Display chart and table
        col1, col2 = st.columns([1, 1])

        with col1:
            # Chart showing nominal vs PV
            chart_df = cf_df[["Year", "Cumulative Nominal", "Cumulative PV"]].copy()
            chart_df = chart_df.set_index("Year")
            chart_df.columns = ["Cumulative (Nominal)", "Cumulative (PV)"]
            st.line_chart(chart_df, height=300)

        with col2:
            # Formatted table
            cf_display = cf_df.copy()
            cf_display["Increment (Nominal)"] = cf_display["Increment (Nominal)"].apply(lambda x: f"${x:,.0f}")
            cf_display["PV Factor"] = cf_display["PV Factor"].apply(lambda x: f"{x:.4f}")
            cf_display["Increment (PV)"] = cf_display["Increment (PV)"].apply(lambda x: f"${x:,.0f}")
            cf_display["Cumulative Nominal"] = cf_display["Cumulative Nominal"].apply(lambda x: f"${x:,.0f}")
            cf_display["Cumulative PV"] = cf_display["Cumulative PV"].apply(lambda x: f"${x:,.0f}")

            st.dataframe(cf_display, use_container_width=True, hide_index=True, height=300)

        st.divider()

        # TIF Lump Sum Result
        st.markdown("### TIF Lump Sum Value")

        # Simple capitalization: Year 1 Increment / Cap Rate
        tif_lump_sum_simple = annual_city_increment / tif_cap_rate if tif_cap_rate > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("TIF Lump Sum", f"${tif_lump_sum_simple:,.0f}",
                     help="Year 1 Increment ÷ Cap Rate")
        with col2:
            per_unit = tif_lump_sum_simple / tif_affordable_units if tif_affordable_units > 0 else 0
            st.metric("Per Affordable Unit", f"${per_unit:,.0f}")
        with col3:
            pct_of_tdc = tif_lump_sum_simple / tif_new_value if tif_new_value > 0 else 0
            st.metric("% of TDC", f"{pct_of_tdc:.1%}")
        with col4:
            st.metric("Total PV of Stream", f"${total_pv:,.0f}",
                     help="Sum of discounted increments over TIF term")

        # Store in session state
        st.session_state["calculated_tif_lump_sum"] = tif_lump_sum_simple

        # Calculation explanation
        with st.expander("Calculation Method", expanded=False):
            st.markdown(f"""
**Simple Capitalization Method:**
- TIF Lump Sum = Year 1 City Increment ÷ Cap Rate
- TIF Lump Sum = ${annual_city_increment:,.0f} ÷ {tif_cap_rate:.2%} = **${tif_lump_sum_simple:,.0f}**

**Present Value Method (shown in table):**
- Sum of annual increments discounted at cap rate over {tif_term} years
- Total PV = **${total_pv:,.0f}**

The simple capitalization method is typically used for quick estimates.
The PV method shows the actual value of the increment stream.
            """)

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
    """Render the detailed Sources & Uses tab with scenario comparison."""
    st.subheader("Sources & Uses")

    # Scenario tabs - same pattern as Detailed Cash Flows
    su_tab1, su_tab2 = st.tabs(["Market Rate", "Mixed Income"])

    with su_tab1:
        _render_sources_uses_for_scenario(inputs, is_mixed_income=False)

    with su_tab2:
        _render_sources_uses_for_scenario(inputs, is_mixed_income=True)


def _render_sources_uses_for_scenario(inputs: ProjectInputs, is_mixed_income: bool):
    """Render Sources & Uses for a single scenario.

    IMPORTANT: Uses cached results from calculate_deal() for consistency.
    No duplicate calculations - reads from the SINGLE SOURCE OF TRUTH.
    """
    scenario_name = "Mixed Income" if is_mixed_income else "Market Rate"
    key_prefix = "mixed_" if is_mixed_income else "market_"

    # Get cached result from the main calculation (SINGLE SOURCE OF TRUTH)
    cache_key = "_cached_mixed_result" if is_mixed_income else "_cached_market_result"
    cached_result = st.session_state.get(cache_key)

    if cached_result is None:
        st.warning(f"No cached results for {scenario_name}. Please run the analysis first.")
        return

    # Extract SourcesUses from cached result
    su = cached_result.sources_uses

    # Show scenario header
    st.markdown(f"### {scenario_name}")

    # Show incentives for mixed income
    if is_mixed_income:
        fee_waivers = 0.0
        tif_lump_sum = 0.0
        if inputs.incentive_config is not None:
            if inputs.incentive_config.toggles.smart_fee_waiver:
                affordable_units = round(inputs.target_units * inputs.affordable_pct)
                fee_waivers = inputs.incentive_config.get_waiver_amount(affordable_units)
            if inputs.incentive_config.toggles.tif_lump_sum:
                tif_lump_sum = st.session_state.get("calculated_tif_lump_sum", 0)

        if fee_waivers > 0 or tif_lump_sum > 0:
            incentives_msg = []
            if fee_waivers > 0:
                incentives_msg.append(f"SMART Fee Waiver: ${fee_waivers:,.0f}")
            if tif_lump_sum > 0:
                incentives_msg.append(f"TIF Lump Sum: ${tif_lump_sum:,.0f}")
            st.success(f"**Incentives Applied:** {', '.join(incentives_msg)}")

    # Render the simple SourcesUses table using cached data
    # Uses render_sources_uses from detailed_cashflow_view which handles SourcesUses
    render_sources_uses(su, key_prefix=key_prefix)

    # Show per-unit metrics
    st.markdown("#### Per-Unit Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        tdc_per_unit = su.tdc / inputs.target_units if inputs.target_units > 0 else 0
        st.metric("TDC / Unit", f"${tdc_per_unit:,.0f}")
    with col2:
        equity_per_unit = su.equity / inputs.target_units if inputs.target_units > 0 else 0
        st.metric("Equity / Unit", f"${equity_per_unit:,.0f}")
    with col3:
        loan_per_unit = su.construction_loan / inputs.target_units if inputs.target_units > 0 else 0
        st.metric("Loan / Unit", f"${loan_per_unit:,.0f}")


def render_monte_carlo_tab(inputs: ProjectInputs):
    """Render Monte Carlo simulation tab."""
    st.header("🎲 Monte Carlo Simulation")
    st.markdown("""
    Run Monte Carlo simulations to understand how uncertainty in key inputs affects project returns.
    Select which inputs to vary, configure their distributions, and run the simulation.
    """)

    # Initialize session state for Monte Carlo
    if "mc_results" not in st.session_state:
        st.session_state.mc_results = None
    if "mc_running" not in st.session_state:
        st.session_state.mc_running = False

    # Configuration section
    st.subheader("Simulation Configuration")

    col1, col2 = st.columns(2)

    with col1:
        n_iterations = st.number_input(
            "Number of Iterations",
            min_value=100,
            max_value=10000,
            value=st.session_state.get("mc_iterations", 500),
            step=100,
            help="More iterations = more accurate but slower"
        )
        st.session_state.mc_iterations = n_iterations

        confidence_level = st.slider(
            "Confidence Level (%)",
            min_value=80,
            max_value=99,
            value=int(st.session_state.get("mc_confidence", 0.95) * 100),
            step=1,
            help="Confidence level for intervals"
        ) / 100  # Convert from % to decimal
        st.session_state.mc_confidence = confidence_level

    with col2:
        seed = st.number_input(
            "Random Seed (for reproducibility)",
            min_value=1,
            max_value=99999,
            value=st.session_state.get("mc_seed", 42),
            help="Same seed = same results"
        )
        st.session_state.mc_seed = seed

        target_irr = st.number_input(
            "Target IRR for Probability Calculation (%)",
            min_value=0.0,
            max_value=50.0,
            value=st.session_state.get("mc_target_irr", 15.0),
            step=1.0,
            format="%.0f",
            help="Calculate P(IRR > target)"
        ) / 100  # Convert from % to decimal
        st.session_state.mc_target_irr = target_irr

    # Input selection
    st.subheader("Select Inputs to Vary")
    st.caption("Check inputs to include in the simulation. Configure min/mode/max values for each.")

    # Define available inputs with their default ranges
    available_inputs = {
        "hard_costs": {
            "label": "Hard Costs",
            "category": "costs",
            "default_mean": inputs.hard_cost_per_unit * inputs.target_units,
            "default_cv": 0.10,
            "distribution": DistributionType.LOGNORMAL,
            "unit": "$",
        },
        "soft_cost_pct": {
            "label": "Soft Cost %",
            "category": "costs",
            "default_min": 0.15,
            "default_mode": inputs.soft_cost_pct,
            "default_max": 0.30,
            "distribution": DistributionType.TRIANGULAR,
            "unit": "%",
        },
        "exit_cap_rate": {
            "label": "Exit Cap Rate",
            "category": "market",
            "default_min": 0.045,
            "default_mode": inputs.exit_cap_rate,
            "default_max": 0.070,
            "distribution": DistributionType.TRIANGULAR,
            "unit": "%",
        },
        "market_rent_growth": {
            "label": "Rent Growth",
            "category": "market",
            "default_min": 0.00,
            "default_mode": inputs.market_rent_growth,
            "default_max": 0.05,
            "distribution": DistributionType.TRIANGULAR,
            "unit": "%",
        },
        "vacancy_rate": {
            "label": "Vacancy Rate",
            "category": "market",
            "default_min": 0.03,
            "default_mode": inputs.vacancy_rate,
            "default_max": 0.12,
            "distribution": DistributionType.TRIANGULAR,
            "unit": "%",
        },
        "perm_rate": {
            "label": "Perm Loan Rate",
            "category": "financing",
            "default_min": 0.045,
            "default_mode": inputs.perm_rate,
            "default_max": 0.085,
            "distribution": DistributionType.TRIANGULAR,
            "unit": "%",
        },
        "construction_rate": {
            "label": "Const. Loan Rate",
            "category": "financing",
            "default_min": 0.055,
            "default_mode": inputs.construction_rate,
            "default_max": 0.095,
            "distribution": DistributionType.TRIANGULAR,
            "unit": "%",
        },
        "construction_months": {
            "label": "Construction (mo)",
            "category": "timing",
            "default_min": inputs.construction_months - 6,
            "default_mode": inputs.construction_months,
            "default_max": inputs.construction_months + 12,
            "distribution": DistributionType.PERT,
            "unit": "months",
        },
        "leaseup_months": {
            "label": "Lease-Up (mo)",
            "category": "timing",
            "default_min": max(3, inputs.leaseup_months - 3),
            "default_mode": inputs.leaseup_months,
            "default_max": inputs.leaseup_months + 6,
            "distribution": DistributionType.PERT,
            "unit": "months",
        },
    }

    # Create input configuration UI - organized by category in columns
    selected_inputs = []
    distributions = []

    # Group inputs by category
    categories = {
        "costs": {"title": "Development Costs", "inputs": []},
        "market": {"title": "Market Assumptions", "inputs": []},
        "financing": {"title": "Financing", "inputs": []},
        "timing": {"title": "Timing", "inputs": []},
    }

    for key, config in available_inputs.items():
        categories[config["category"]]["inputs"].append((key, config))

    # Render in a 4-column layout
    cols = st.columns(4)

    for col_idx, (cat_key, cat_data) in enumerate(categories.items()):
        with cols[col_idx]:
            st.markdown(f"**{cat_data['title']}**")

            for key, config in cat_data["inputs"]:
                # Checkbox for inclusion
                include = st.checkbox(
                    config['label'],
                    value=st.session_state.get(f"mc_include_{key}", key in ["exit_cap_rate", "hard_costs", "vacancy_rate"]),
                    key=f"mc_include_{key}"
                )

                if include:
                    selected_inputs.append(key)

                    if config["distribution"] == DistributionType.LOGNORMAL:
                        # Lognormal uses mean and CV
                        mean_val = st.number_input(
                            "Mean",
                            value=float(config["default_mean"]),
                            key=f"mc_mean_{key}",
                            format="%.0f",
                            label_visibility="collapsed"
                        )
                        cv_val = st.slider(
                            "CV",
                            min_value=0.05,
                            max_value=0.30,
                            value=config["default_cv"],
                            key=f"mc_cv_{key}",
                            help="Coefficient of Variation"
                        )
                        distributions.append(InputDistribution(
                            parameter=key,
                            distribution=DistributionType.LOGNORMAL,
                            mean=mean_val,
                            std=mean_val * cv_val,
                            clip_min=mean_val * 0.5,
                        ))
                    else:
                        # Triangular/PERT uses min, mode, max
                        is_pct = config["unit"] == "%"
                        multiplier = 100 if is_pct else 1
                        format_str = "%.1f" if is_pct else "%.0f"

                        min_val = st.number_input(
                            "Min",
                            value=float(config["default_min"]) * multiplier,
                            key=f"mc_min_{key}",
                            format=format_str,
                            label_visibility="collapsed"
                        ) / multiplier

                        mode_val = st.number_input(
                            "Mode",
                            value=float(config["default_mode"]) * multiplier,
                            key=f"mc_mode_{key}",
                            format=format_str,
                            label_visibility="collapsed"
                        ) / multiplier

                        max_val = st.number_input(
                            "Max",
                            value=float(config["default_max"]) * multiplier,
                            key=f"mc_max_{key}",
                            format=format_str,
                            label_visibility="collapsed"
                        ) / multiplier

                        # Show range summary
                        if is_pct:
                            st.caption(f"{min_val*100:.1f}% - {mode_val*100:.1f}% - {max_val*100:.1f}%")
                        elif config["unit"] == "months":
                            st.caption(f"{min_val:.0f} - {mode_val:.0f} - {max_val:.0f} mo")

                        distributions.append(InputDistribution(
                            parameter=key,
                            distribution=config["distribution"],
                            min_value=min_val,
                            mode=mode_val,
                            max_value=max_val,
                        ))
                else:
                    # Show current value when not included
                    if config["distribution"] == DistributionType.LOGNORMAL:
                        st.caption(f"Fixed: ${config['default_mean']:,.0f}")
                    elif config["unit"] == "%":
                        st.caption(f"Fixed: {config['default_mode']*100:.1f}%")
                    else:
                        st.caption(f"Fixed: {config['default_mode']:.0f} {config['unit']}")

    # Run simulation button
    st.divider()

    if len(distributions) == 0:
        st.warning("Select at least one input to vary.")
        run_disabled = True
    else:
        run_disabled = False
        st.info(f"**{len(distributions)} inputs selected** for simulation with **{n_iterations:,} iterations**")

    if st.button("🚀 Run Monte Carlo Simulation", disabled=run_disabled, type="primary"):
        st.session_state.mc_running = True

        # Build base inputs from session state
        construction_type = st.session_state.get("construction_type", "podium_5over1")
        efficiency = get_efficiency(construction_type)
        unit_mix = get_unit_mix_from_session_state(efficiency)
        total_units = inputs.target_units

        # Calculate monthly GPR using allocate_units -> calculate_gpr
        allocations = allocate_units(
            total_units, unit_mix, 0.0,  # 0% affordable for base
            inputs.ami_level, inputs.market_rent_psf
        )
        gpr_result = calculate_gpr(allocations)

        base_inputs = BaseInputs(
            land_cost=inputs.land_cost * 2,  # Assuming 2 acres
            hard_costs=inputs.hard_cost_per_unit * total_units,
            soft_cost_pct=inputs.soft_cost_pct,
            construction_ltc=inputs.construction_ltc,
            construction_rate=inputs.construction_rate,
            start_date=inputs.predevelopment_start,
            predevelopment_months=inputs.predevelopment_months,
            construction_months=inputs.construction_months,
            leaseup_months=inputs.leaseup_months,
            operations_months=inputs.operations_months,
            monthly_gpr_at_stabilization=gpr_result.total_gpr_monthly,
            vacancy_rate=inputs.vacancy_rate,
            leaseup_pace=inputs.leaseup_pace,
            max_occupancy=inputs.max_occupancy,
            annual_opex_per_unit=4500,  # Default
            total_units=total_units,
            existing_assessed_value=inputs.existing_assessed_value,
            assessed_value_basis=AssessedValueBasis.TDC,
            market_rent_growth=inputs.market_rent_growth,
            opex_growth=inputs.opex_growth,
            prop_tax_growth=inputs.property_tax_growth,
            perm_rate=inputs.perm_rate,
            perm_amort_years=inputs.perm_amort_years,
            perm_ltv_max=inputs.perm_ltv_max,
            perm_dscr_min=inputs.perm_dscr_min,
            exit_cap_rate=inputs.exit_cap_rate,
            affordable_pct=inputs.affordable_pct,
            affordable_rent_discount=0.40 if inputs.affordable_pct > 0 else 0.0,
        )

        config = MonteCarloConfig(
            base_inputs=base_inputs,
            distributions=distributions,
            n_iterations=n_iterations,
            seed=seed,
            confidence_level=confidence_level,
            parallel=True,
            use_unified_engine=True,  # Use calculate_deal() for consistency with main app
        )

        # Run with progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(completed, total):
            progress_bar.progress(completed / total)
            status_text.text(f"Running iteration {completed:,} of {total:,}...")

        try:
            result = run_monte_carlo(config, target_irr=target_irr, progress_callback=update_progress)
            st.session_state.mc_results = result
            progress_bar.progress(1.0)
            status_text.text("✅ Simulation complete!")
        except Exception as e:
            st.error(f"Error running simulation: {e}")
            st.session_state.mc_results = None

        st.session_state.mc_running = False

    # Display results
    if st.session_state.mc_results is not None:
        result = st.session_state.mc_results

        st.divider()
        st.subheader("📈 Simulation Results")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Mean Levered IRR",
                f"{result.levered_irr_mean:.1%}",
                delta=f"±{result.levered_irr_std:.1%} std"
            )

        with col2:
            st.metric(
                "Median Levered IRR",
                f"{result.levered_irr_median:.1%}"
            )

        with col3:
            st.metric(
                f"{int(confidence_level*100)}% Confidence Interval",
                f"[{result.levered_irr_ci_lower:.1%}, {result.levered_irr_ci_upper:.1%}]"
            )

        with col4:
            st.metric(
                f"P(IRR > {target_irr:.0%})",
                f"{result.prob_levered_above_target:.1%}"
            )

        # Detailed results tabs
        res_tab1, res_tab2, res_tab3 = st.tabs(["📊 Distribution", "🎯 Sensitivity", "📋 Summary"])

        with res_tab1:
            st.subheader("IRR Distribution")

            # Create histogram data
            levered_irrs = result.get_levered_irr_distribution()

            # Use pandas for histogram
            hist_df = pd.DataFrame({"Levered IRR": levered_irrs * 100})  # Convert to %

            st.bar_chart(hist_df["Levered IRR"].value_counts(bins=30).sort_index())

            # Percentiles table
            st.subheader("Percentiles")
            percentile_data = {
                "Percentile": ["5th", "10th", "25th", "50th (Median)", "75th", "90th", "95th"],
                "Levered IRR": [
                    f"{result.levered_irr_percentiles[5]:.2%}",
                    f"{result.levered_irr_percentiles[10]:.2%}",
                    f"{result.levered_irr_percentiles[25]:.2%}",
                    f"{result.levered_irr_percentiles[50]:.2%}",
                    f"{result.levered_irr_percentiles[75]:.2%}",
                    f"{result.levered_irr_percentiles[90]:.2%}",
                    f"{result.levered_irr_percentiles[95]:.2%}",
                ],
                "Unlevered IRR": [
                    f"{result.unlevered_irr_percentiles[5]:.2%}",
                    f"{result.unlevered_irr_percentiles[10]:.2%}",
                    f"{result.unlevered_irr_percentiles[25]:.2%}",
                    f"{result.unlevered_irr_percentiles[50]:.2%}",
                    f"{result.unlevered_irr_percentiles[75]:.2%}",
                    f"{result.unlevered_irr_percentiles[90]:.2%}",
                    f"{result.unlevered_irr_percentiles[95]:.2%}",
                ],
            }
            st.dataframe(pd.DataFrame(percentile_data), hide_index=True, use_container_width=True)

        with res_tab2:
            st.subheader("Sensitivity Analysis")
            st.markdown("Which inputs have the most impact on IRR?")

            # Sort by absolute correlation
            sorted_sens = sorted(
                result.sensitivities,
                key=lambda s: abs(s.correlation_levered),
                reverse=True
            )

            sens_data = {
                "Input": [s.parameter.replace("_", " ").title() for s in sorted_sens],
                "Correlation (Levered)": [s.correlation_levered for s in sorted_sens],
                "Correlation (Unlevered)": [s.correlation_unlevered for s in sorted_sens],
                "Impact": ["🔴 High" if abs(s.correlation_levered) > 0.5
                          else "🟡 Medium" if abs(s.correlation_levered) > 0.2
                          else "🟢 Low" for s in sorted_sens],
            }

            st.dataframe(pd.DataFrame(sens_data), hide_index=True, use_container_width=True)

            st.markdown("""
            **Interpretation:**
            - **Positive correlation**: Higher input value → Higher IRR
            - **Negative correlation**: Higher input value → Lower IRR
            - **Magnitude**: Closer to ±1.0 means stronger relationship
            """)

        with res_tab3:
            st.subheader("Full Summary")
            st.code(result.summary())


def _precalculate_derived_values(inputs: ProjectInputs):
    """Pre-calculate derived values so they're available to all tabs.

    This runs on every rerun to keep values in sync with inputs.
    IMPORTANT: TIF lump sum calculation uses the same session state keys
    as the Property Tax tab to ensure consistency.
    """
    from src.calculations.property_tax import get_austin_tax_stack
    from src.models.lookups import CONSTRUCTION_TYPE_PARAMS

    # Get construction params
    ct_params = CONSTRUCTION_TYPE_PARAMS.get(inputs.construction_type)
    units_per_acre = ct_params.units_per_acre if ct_params else 66

    # Estimate TDC for TIF calculation (used as default for tif_new_value_str)
    hard_costs = inputs.hard_cost_per_unit * inputs.target_units
    hard_costs_with_contingency = hard_costs * (1 + inputs.hard_cost_contingency_pct)
    soft_costs = hard_costs_with_contingency * inputs.soft_cost_pct * (1 + inputs.soft_cost_contingency_pct)
    predevelopment = hard_costs * inputs.predevelopment_cost_pct
    land_cost = inputs.land_cost * (inputs.target_units / units_per_acre)
    estimated_tdc = land_cost + hard_costs_with_contingency + predevelopment + soft_costs
    # Add ~10% for financing costs
    estimated_tdc *= 1.10

    # Store estimated TDC for reference
    st.session_state["estimated_tdc"] = estimated_tdc

    # Helper to parse currency input (handles commas)
    def parse_currency(val: str) -> float:
        if not val:
            return 0.0
        return float(val.replace(",", "").replace("$", "").strip() or 0)

    # Use the SAME session state keys as Property Tax tab for TIF calculation
    # This ensures manual adjustments on the Property Tax tab flow through correctly

    # Base value: use Property Tax tab's tif_base_value_str if set, else existing_assessed_value
    if "tif_base_value_str" in st.session_state:
        base_value = parse_currency(st.session_state["tif_base_value_str"])
    else:
        base_value = st.session_state.get("existing_assessed_value", 5_000_000)

    # New value: use Property Tax tab's tif_new_value_str if set, else estimated TDC
    if "tif_new_value_str" in st.session_state:
        new_value = parse_currency(st.session_state["tif_new_value_str"])
    else:
        new_value = estimated_tdc

    # Cap rate: use Property Tax tab's prop_tif_cap_rate if set, else default
    if "prop_tif_cap_rate" in st.session_state:
        tif_cap_rate = st.session_state["prop_tif_cap_rate"] / 100  # Slider stores as percentage
    else:
        tif_cap_rate = 0.095  # Default

    # Calculate TIF lump sum using consistent inputs
    tax_stack = get_austin_tax_stack()
    increment = max(0, new_value - base_value)
    annual_city_increment = increment * tax_stack.tif_participating_rate_decimal

    if tif_cap_rate > 0:
        tif_lump_sum = annual_city_increment / tif_cap_rate
        st.session_state["calculated_tif_lump_sum"] = tif_lump_sum


def main():
    """Main application entry point."""
    st.title("🏠 Austin Affordable Housing Incentive Calculator")
    st.caption("Model multifamily developments and analyze incentive impacts on returns")

    # Main content tabs - render FIRST so inputs update session state
    tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "⚙️ Project Inputs",
        "🏢 Unit Mix",
        "🏛️ Property Tax",
        "💵 Sources & Uses",
        "🎁 Incentives",
        "💰 Detailed Cash Flows",
        "📊 Results",
        "🎲 Monte Carlo",
        "🔍 Spreadsheet Debug"
    ])

    # Get inputs from session state
    inputs = get_session_state_inputs()

    # Pre-calculate derived values (TIF, etc.) so they're available to all tabs
    _precalculate_derived_values(inputs)

    # Get scenario configuration
    model_config = get_model_config_from_session()

    # Use cached results from sidebar (single source of truth)
    # The sidebar calls run_analysis() and caches results in session state
    market_result = st.session_state.get("_cached_market_result")
    mixed_result = st.session_state.get("_cached_mixed_result")
    market_metrics = st.session_state.get("_cached_market_metrics")
    mixed_metrics = st.session_state.get("_cached_mixed_metrics")

    if market_result and mixed_result and market_metrics and mixed_metrics:
        # Build comparison from cached results
        from src.calculations.metrics import ScenarioComparison
        irr_diff_bps = int((mixed_metrics.levered_irr - market_metrics.levered_irr) * 10000)
        comparison = ScenarioComparison(
            market=market_metrics,
            mixed_income=mixed_metrics,
            irr_difference_bps=irr_diff_bps,
            noi_gap_annual=market_metrics.noi_annual - mixed_metrics.noi_annual,
            equity_difference=market_metrics.equity_required - mixed_metrics.equity_required,
            tdc_difference=market_metrics.tdc - mixed_metrics.tdc,
            total_incentive_value=st.session_state.get("calculated_tif_lump_sum", 0.0),
            incentive_irr_impact_bps=irr_diff_bps,
            meets_target=irr_diff_bps >= 150,
        )
        analysis_success = True
    else:
        # Fallback: run analysis if cache is not available
        try:
            if model_config.mode == ModelMode.COMPARISON:
                market_result, mixed_result, market_metrics, mixed_metrics, comparison = run_analysis(
                    inputs,
                    scenario_a=model_config.scenario_a,
                    scenario_b=model_config.scenario_b,
                )
            else:
                market_result, mixed_result, market_metrics, mixed_metrics, comparison = run_analysis(
                    inputs,
                    scenario_a=model_config.single_scenario if model_config.single_project_type == ProjectType.MARKET_RATE else None,
                    scenario_b=model_config.single_scenario if model_config.single_project_type == ProjectType.MIXED_INCOME else None,
                )
            # Cache results so tabs can use them
            st.session_state["_cached_market_result"] = market_result
            st.session_state["_cached_mixed_result"] = mixed_result
            st.session_state["_cached_market_metrics"] = market_metrics
            st.session_state["_cached_mixed_metrics"] = mixed_metrics
            analysis_success = True
        except Exception as e:
            st.error(f"Error running analysis: {e}")
            analysis_success = False

    with tab0:
        render_project_inputs_tab()

    with tab1:
        render_unit_mix_tab()

    with tab2:
        render_property_tax_tab(inputs)

    with tab3:
        render_sources_uses_tab(inputs)

    with tab4:
        render_matrix_tab(inputs)

    with tab5:
        render_detailed_cashflow_tab(inputs)

    with tab6:
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

    with tab7:
        render_monte_carlo_tab(inputs)

    with tab8:
        # Spreadsheet Debug View - only render when tab is selected
        # Uses cached results from session state
        if market_result and mixed_result:
            render_full_debug_page(market_result, mixed_result)
        else:
            st.warning("Run analysis first to see debug view. Results are calculated when inputs change.")
            st.info("Go to the Results tab or check the sidebar to trigger calculation.")

    # Render sidebar AFTER tabs so input widgets have updated session state
    render_sidebar()


if __name__ == "__main__":
    main()
