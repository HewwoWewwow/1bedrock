"""Unit mix configuration component for the Streamlit UI."""

import streamlit as st
from typing import Dict

from src.models.lookups import (
    ConstructionType,
    CONSTRUCTION_TYPE_PARAMS,
    AMI_RENT_LIMITS,
)


# Unit types in order from smallest to largest
UNIT_TYPES = ["studio", "1br", "2br", "3br", "4br"]
UNIT_LABELS = {"studio": "Studio", "1br": "1-Bedroom", "2br": "2-Bedroom", "3br": "3-Bedroom", "4br": "4-Bedroom"}

# Default NSF values per unit type (from spreadsheet v10c, assumes 0.85 efficiency)
DEFAULT_NSF = {"studio": 510, "1br": 638, "2br": 765, "3br": 978, "4br": 1233}

# Default allocations (as percentages, e.g., 12 = 12%)
DEFAULT_ALLOCATIONS = {"studio": 12, "1br": 25, "2br": 45, "3br": 15, "4br": 3}

# Default $/NSF values (derived from spreadsheet v10c: monthly_rent / nsf)
DEFAULT_RENT_PSF = {
    "studio": 4.1176,  # $2,100 / 510 NSF
    "1br": 4.1144,     # $2,625 / 638 NSF
    "2br": 4.0000,     # $3,060 / 765 NSF
    "3br": 3.8804,     # $3,795 / 978 NSF
    "4br": 3.7632,     # $4,640 / 1233 NSF
}

# Default monthly rents (from spreadsheet v10c, for quick estimate mode)
DEFAULT_RENT_MONTHLY = {
    "studio": 2100,
    "1br": 2625,
    "2br": 3060,
    "3br": 3795,
    "4br": 4640,
}

# Rent input modes
RENT_MODE_PSF = "$/NSF (recommended)"
RENT_MODE_MONTHLY = "$/month"

# AMI tiers
AMI_TIERS = ["30%", "50%", "60%", "80%", "100%"]


def get_efficiency(construction_type: str) -> float:
    """Get efficiency factor for construction type."""
    try:
        ct = ConstructionType(construction_type)
        return CONSTRUCTION_TYPE_PARAMS[ct].efficiency
    except (ValueError, KeyError):
        return 0.85


def _get_tier_default_ami() -> str:
    """Get the default AMI level based on the selected incentive tier."""
    from src.models.incentives import IncentiveTier, TIER_REQUIREMENTS

    selected_tier = st.session_state.get("selected_tier", 2)
    try:
        tier_enum = IncentiveTier(selected_tier)
        return str(TIER_REQUIREMENTS[tier_enum]["ami_level"])
    except (ValueError, KeyError):
        return "50%"


def _initialize_session_state_defaults():
    """Initialize unit mix session state with defaults if not set."""
    # Get the default AMI based on selected tier
    default_ami = _get_tier_default_ami()

    # Check if tier has changed and reset AMI distributions if so
    current_tier = st.session_state.get("selected_tier", 2)
    prev_tier = st.session_state.get("_unit_mix_prev_tier", None)

    tier_changed = prev_tier is not None and prev_tier != current_tier
    if tier_changed:
        # Reset AMI distributions to match new tier
        for unit_type in UNIT_TYPES:
            for tier in AMI_TIERS:
                key = f"{unit_type}_ami_{tier}_pct"
                st.session_state[key] = 100 if tier == default_ami else 0

    # Track current tier for next check
    st.session_state["_unit_mix_prev_tier"] = current_tier

    for unit_type in UNIT_TYPES:
        # Allocation
        if f"{unit_type}_alloc_pct" not in st.session_state:
            st.session_state[f"{unit_type}_alloc_pct"] = DEFAULT_ALLOCATIONS[unit_type]
        # NSF
        if f"{unit_type}_nsf" not in st.session_state:
            st.session_state[f"{unit_type}_nsf"] = DEFAULT_NSF[unit_type]
        # Rent PSF
        if f"{unit_type}_rent_psf" not in st.session_state:
            st.session_state[f"{unit_type}_rent_psf"] = DEFAULT_RENT_PSF[unit_type]
        # Rent monthly
        if f"{unit_type}_rent_monthly" not in st.session_state:
            st.session_state[f"{unit_type}_rent_monthly"] = DEFAULT_RENT_MONTHLY[unit_type]
        # AMI distributions (default based on selected tier's AMI level)
        for tier in AMI_TIERS:
            key = f"{unit_type}_ami_{tier}_pct"
            if key not in st.session_state:
                st.session_state[key] = 100 if tier == default_ami else 0

    # Rent input mode
    if "rent_input_mode" not in st.session_state:
        st.session_state["rent_input_mode"] = RENT_MODE_PSF


def render_unit_mix_tab():
    """Render the unit mix tab with sub-tabs for Market Rate and Mixed Income."""
    from src.models.scenario_config import ModelMode, ProjectType

    # Initialize defaults
    _initialize_session_state_defaults()

    # Get mode
    mode = st.session_state.get("model_mode", ModelMode.COMPARISON)
    is_comparison = mode == ModelMode.COMPARISON
    is_single_mixed = (mode == ModelMode.SINGLE_PROJECT and
                       st.session_state.get("single_project_type") == ProjectType.MIXED_INCOME)

    if is_comparison or is_single_mixed:
        # Show sub-tabs
        tab1, tab2 = st.tabs(["Market Rate Unit Mix", "Mixed Income Unit Mix"])

        with tab1:
            st.caption("Define unit allocation, sizes, and market rents. These flow to the Mixed Income tab.")
            render_market_rate_unit_mix()

        with tab2:
            st.caption("Assign affordable units to AMI tiers. Unit allocation and rents inherited from Market Rate.")
            render_mixed_income_unit_mix()
    else:
        # Single project market rate - just show market rate form
        st.caption("Define unit allocation, sizes, and market rents.")
        render_market_rate_unit_mix()


def render_market_rate_unit_mix():
    """Render the market rate unit mix configuration (no AMI columns)."""
    # Get key values
    construction_type = st.session_state.get("construction_type", "podium_midrise_5over1")
    efficiency = get_efficiency(construction_type)
    target_units = st.session_state.get("target_units", 200)

    st.markdown(f"**Construction:** {construction_type.replace('_', ' ').title()} | "
                f"**Efficiency:** {efficiency:.0%} | "
                f"**Total Units:** {target_units}")

    st.divider()

    # =========================================================================
    # SECTION 1: Unit Allocations (simplified - no AMI)
    # =========================================================================
    st.markdown("### Unit Allocations")

    # Header row
    cols = st.columns([1.5, 1.0, 1.0])
    with cols[0]:
        st.markdown("**Unit Type**")
    with cols[1]:
        st.markdown("**Allocation %**")
    with cols[2]:
        st.markdown("**Units**")

    # Track totals
    total_alloc = 0.0
    total_units_calc = 0

    for unit_type in UNIT_TYPES:
        cols = st.columns([1.5, 1.0, 1.0])

        with cols[0]:
            st.markdown(f"**{UNIT_LABELS[unit_type]}**")

        with cols[1]:
            alloc_pct = st.number_input(
                f"Alloc {unit_type}",
                min_value=0,
                max_value=100,
                value=st.session_state[f"{unit_type}_alloc_pct"],
                step=1,
                key=f"{unit_type}_alloc_pct",
                label_visibility="collapsed"
            )
            alloc = alloc_pct / 100.0

        units_of_type = round(target_units * alloc)
        total_alloc += alloc
        total_units_calc += units_of_type

        with cols[2]:
            st.markdown(f"{units_of_type}")

    # Totals row
    st.divider()
    cols = st.columns([1.5, 1.0, 1.0])
    with cols[0]:
        st.markdown("**TOTAL**")
    with cols[1]:
        if abs(total_alloc - 1.0) > 0.01:
            st.error(f"{total_alloc:.0%}")
        else:
            st.success(f"{total_alloc:.0%}")
    with cols[2]:
        st.markdown(f"**{total_units_calc}**")

    if abs(total_alloc - 1.0) > 0.01:
        st.warning(f"Allocations sum to {total_alloc:.0%}. Adjust to reach 100%.")

    st.divider()

    # =========================================================================
    # SECTION 2: NSF by Unit Type
    # =========================================================================
    st.markdown("### Net Square Footage (NSF)")
    st.caption(f"NSF = Net Square Feet | GSF = NSF / {efficiency:.0%} efficiency")

    cols = st.columns([1.5, 1.0, 1.0, 1.0])
    with cols[0]:
        st.markdown("**Unit Type**")
    with cols[1]:
        st.markdown("**NSF/Unit**")
    with cols[2]:
        st.markdown("**GSF/Unit**")
    with cols[3]:
        st.markdown("**Total NSF**")

    total_nsf = 0
    total_gsf = 0

    for unit_type in UNIT_TYPES:
        cols = st.columns([1.5, 1.0, 1.0, 1.0])

        with cols[0]:
            st.markdown(f"**{UNIT_LABELS[unit_type]}**")

        with cols[1]:
            nsf = st.number_input(
                f"NSF {unit_type}",
                min_value=300,
                max_value=2500,
                value=st.session_state[f"{unit_type}_nsf"],
                step=25,
                key=f"{unit_type}_nsf",
                label_visibility="collapsed"
            )

        gsf = int(nsf / efficiency) if efficiency > 0 else nsf
        st.session_state[f"{unit_type}_gsf"] = gsf

        alloc_pct = st.session_state.get(f"{unit_type}_alloc_pct", DEFAULT_ALLOCATIONS[unit_type])
        units_of_type = round(target_units * alloc_pct / 100.0)

        type_nsf = nsf * units_of_type
        type_gsf = gsf * units_of_type
        total_nsf += type_nsf
        total_gsf += type_gsf

        with cols[2]:
            st.markdown(f"{gsf:,}")

        with cols[3]:
            st.markdown(f"{type_nsf:,}")

    st.divider()
    cols = st.columns([1.5, 1.0, 1.0, 1.0])
    with cols[0]:
        st.markdown("**TOTAL**")
    with cols[1]:
        st.markdown("")
    with cols[2]:
        st.markdown(f"**{total_gsf:,}**")
    with cols[3]:
        st.markdown(f"**{total_nsf:,}**")

    st.divider()

    # =========================================================================
    # SECTION 3: Market Rents
    # =========================================================================
    st.markdown("### Market Rents")

    rent_mode_col, _ = st.columns([2, 3])
    with rent_mode_col:
        rent_mode = st.radio(
            "Rent input mode",
            options=[RENT_MODE_PSF, RENT_MODE_MONTHLY],
            horizontal=True,
            key="rent_input_mode",
            help="$/NSF: rent adjusts with unit size. $/month: enter explicit rent."
        )

    use_psf_mode = (rent_mode == RENT_MODE_PSF)

    if use_psf_mode:
        cols = st.columns([1.5, 1.0, 1.0, 1.2])
        with cols[0]:
            st.markdown("**Unit Type**")
        with cols[1]:
            st.markdown("**$/NSF**")
        with cols[2]:
            st.markdown("**$/Month**")
        with cols[3]:
            st.markdown("**Total Monthly**")
    else:
        cols = st.columns([1.5, 1.0, 1.0, 1.2])
        with cols[0]:
            st.markdown("**Unit Type**")
        with cols[1]:
            st.markdown("**$/Month**")
        with cols[2]:
            st.markdown("**$/NSF**")
        with cols[3]:
            st.markdown("**Total Monthly**")

    total_monthly_rent = 0

    for unit_type in UNIT_TYPES:
        nsf = st.session_state.get(f"{unit_type}_nsf", DEFAULT_NSF[unit_type])
        alloc_pct = st.session_state.get(f"{unit_type}_alloc_pct", DEFAULT_ALLOCATIONS[unit_type])
        units_of_type = round(target_units * alloc_pct / 100.0)

        if use_psf_mode:
            cols = st.columns([1.5, 1.0, 1.0, 1.2])

            with cols[0]:
                st.markdown(f"**{UNIT_LABELS[unit_type]}**")

            with cols[1]:
                rent_psf = st.number_input(
                    f"$/NSF {unit_type}",
                    min_value=1.00,
                    max_value=10.00,
                    value=st.session_state[f"{unit_type}_rent_psf"],
                    step=0.05,
                    format="%.2f",
                    key=f"{unit_type}_rent_psf",
                    label_visibility="collapsed"
                )

            market_rent = nsf * rent_psf

            with cols[2]:
                st.markdown(f"${market_rent:,.0f}")

        else:
            cols = st.columns([1.5, 1.0, 1.0, 1.2])

            with cols[0]:
                st.markdown(f"**{UNIT_LABELS[unit_type]}**")

            with cols[1]:
                market_rent = st.number_input(
                    f"$/mo {unit_type}",
                    min_value=500,
                    max_value=15000,
                    value=st.session_state[f"{unit_type}_rent_monthly"],
                    step=25,
                    key=f"{unit_type}_rent_monthly",
                    label_visibility="collapsed"
                )

            rent_psf = market_rent / nsf if nsf > 0 else 0

            with cols[2]:
                st.markdown(f"${rent_psf:.2f}")

        type_rent = market_rent * units_of_type
        total_monthly_rent += type_rent

        with cols[3]:
            st.markdown(f"${type_rent:,.0f}")

    st.divider()
    cols = st.columns([1.5, 1.0, 1.0, 1.2])
    with cols[0]:
        st.markdown("**TOTAL**")
    with cols[1]:
        st.markdown("")
    with cols[2]:
        st.markdown("")
    with cols[3]:
        st.markdown(f"**${total_monthly_rent:,.0f}**")

    # GPR/EGI Summary for Market Rate
    st.divider()
    st.markdown("### Market Rate Revenue Summary")

    vacancy_rate = st.session_state.get("vacancy_rate_pct", 6.0) / 100.0
    annual_gpr = total_monthly_rent * 12
    annual_egi = annual_gpr * (1 - vacancy_rate)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Monthly GPR", f"${total_monthly_rent:,.0f}")
    with col2:
        st.metric("Annual GPR", f"${annual_gpr:,.0f}")
    with col3:
        st.metric(f"Annual EGI ({1-vacancy_rate:.0%} occ)", f"${annual_egi:,.0f}")


def render_mixed_income_unit_mix():
    """Render the mixed income unit mix with AMI distribution."""
    from src.models.incentives import IncentiveTier, TIER_REQUIREMENTS

    # Get key values
    construction_type = st.session_state.get("construction_type", "podium_midrise_5over1")
    efficiency = get_efficiency(construction_type)
    target_units = st.session_state.get("target_units", 200)

    # Get affordable % from session state
    affordable_pct_raw = st.session_state.get("affordable_pct", 20.0)
    affordable_pct = affordable_pct_raw / 100.0 if affordable_pct_raw > 1 else affordable_pct_raw

    total_affordable = round(target_units * affordable_pct)
    total_market = target_units - total_affordable

    # Show selected tier info
    selected_tier = st.session_state.get("selected_tier", 2)
    tier_names = {1: "Tier 1 - Deep Affordability", 2: "Tier 2 - Moderate (More Units)", 3: "Tier 3 - Moderate (Balanced)"}
    try:
        tier_enum = IncentiveTier(selected_tier)
        tier_ami = TIER_REQUIREMENTS[tier_enum]["ami_level"]
    except (ValueError, KeyError):
        tier_ami = "50%"

    st.info(f"**Selected Tier:** {tier_names.get(selected_tier, f'Tier {selected_tier}')} | "
            f"**Default AMI:** {tier_ami} *(change tier on Scenarios tab)*")

    st.markdown(f"**Total Units:** {target_units} | "
                f"**Affordable:** {total_affordable} ({affordable_pct:.0%}) | "
                f"**Market:** {total_market}")

    st.divider()

    # =========================================================================
    # SECTION 1: Unit Summary (inherited from market rate, read-only)
    # =========================================================================
    st.markdown("### Unit Allocation (from Market Rate)")
    st.caption("Allocation inherited from Market Rate tab. Affordable units distributed proportionally.")

    # Header row
    cols = st.columns([1.5, 0.8, 0.8, 0.8, 0.8])
    with cols[0]:
        st.markdown("**Unit Type**")
    with cols[1]:
        st.markdown("**Alloc %**")
    with cols[2]:
        st.markdown("**Total**")
    with cols[3]:
        st.markdown("**Market**")
    with cols[4]:
        st.markdown("**Afford.**")

    unit_counts = {}
    for unit_type in UNIT_TYPES:
        cols = st.columns([1.5, 0.8, 0.8, 0.8, 0.8])

        alloc_pct = st.session_state.get(f"{unit_type}_alloc_pct", DEFAULT_ALLOCATIONS[unit_type])
        alloc = alloc_pct / 100.0
        units_of_type = round(target_units * alloc)
        affordable_of_type = round(total_affordable * alloc) if total_affordable > 0 else 0
        market_of_type = units_of_type - affordable_of_type

        unit_counts[unit_type] = {
            "total": units_of_type,
            "market": market_of_type,
            "affordable": affordable_of_type,
        }

        with cols[0]:
            st.markdown(f"**{UNIT_LABELS[unit_type]}**")
        with cols[1]:
            st.markdown(f"{alloc_pct}%")
        with cols[2]:
            st.markdown(f"{units_of_type}")
        with cols[3]:
            st.markdown(f"{market_of_type}")
        with cols[4]:
            st.markdown(f"{affordable_of_type}")

    st.divider()

    # =========================================================================
    # SECTION 2: AMI Distribution for Affordable Units
    # =========================================================================
    st.markdown("### AMI Distribution")
    st.caption("Assign affordable units to AMI tiers. Percentages should sum to 100% per row.")

    # Header row
    cols = st.columns([1.5, 0.8] + [0.8] * len(AMI_TIERS) + [0.8])
    with cols[0]:
        st.markdown("**Unit Type**")
    with cols[1]:
        st.markdown("**Afford.**")
    for i, tier in enumerate(AMI_TIERS):
        with cols[2 + i]:
            st.markdown(f"**{tier}**")
    with cols[-1]:
        st.markdown("**Check**")

    ami_unit_counts = {tier: 0 for tier in AMI_TIERS}

    for unit_type in UNIT_TYPES:
        affordable_of_type = unit_counts[unit_type]["affordable"]

        cols = st.columns([1.5, 0.8] + [0.8] * len(AMI_TIERS) + [0.8])

        with cols[0]:
            st.markdown(f"**{UNIT_LABELS[unit_type]}**")

        with cols[1]:
            st.markdown(f"{affordable_of_type}")

        if affordable_of_type > 0:
            row_total_pct = 0
            for i, tier in enumerate(AMI_TIERS):
                with cols[2 + i]:
                    ami_pct = st.number_input(
                        f"{tier} {unit_type}",
                        min_value=0,
                        max_value=100,
                        value=st.session_state[f"{unit_type}_ami_{tier}_pct"],
                        step=25,
                        key=f"{unit_type}_ami_{tier}_pct",
                        label_visibility="collapsed"
                    )
                    row_total_pct += ami_pct
                    ami_units = round(affordable_of_type * ami_pct / 100.0)
                    ami_unit_counts[tier] += ami_units

            with cols[-1]:
                if abs(row_total_pct - 100) <= 1:
                    st.success("100%")
                else:
                    st.error(f"{row_total_pct}%")
        else:
            for i in range(len(AMI_TIERS)):
                with cols[2 + i]:
                    st.markdown("-")
            with cols[-1]:
                st.markdown("-")

    # Totals row
    st.divider()
    cols = st.columns([1.5, 0.8] + [0.8] * len(AMI_TIERS) + [0.8])
    with cols[0]:
        st.markdown("**TOTAL**")
    with cols[1]:
        st.markdown(f"**{total_affordable}**")
    for i, tier in enumerate(AMI_TIERS):
        with cols[2 + i]:
            if ami_unit_counts[tier] > 0:
                st.markdown(f"**{ami_unit_counts[tier]}**")
            else:
                st.markdown("-")
    with cols[-1]:
        st.markdown("")

    st.divider()

    # =========================================================================
    # SECTION 3: Monthly Rents Summary
    # =========================================================================
    st.markdown("### Monthly Rents")
    st.caption("Market rents from Market Rate tab. Affordable rents are AMI limits.")

    # Get rent mode
    rent_mode = st.session_state.get("rent_input_mode", RENT_MODE_PSF)
    use_psf_mode = (rent_mode == RENT_MODE_PSF)

    # Header
    cols = st.columns([1.5, 0.9, 0.9] + [0.8] * len(AMI_TIERS))
    with cols[0]:
        st.markdown("**Unit Type**")
    with cols[1]:
        st.markdown("**Market $/mo**")
    with cols[2]:
        st.markdown("**Mkt Total**")
    for i, tier in enumerate(AMI_TIERS):
        with cols[3 + i]:
            st.markdown(f"**{tier}**")

    total_market_rent = 0
    total_affordable_rent = 0
    ami_rent_totals = {tier: 0 for tier in AMI_TIERS}

    for unit_type in UNIT_TYPES:
        nsf = st.session_state.get(f"{unit_type}_nsf", DEFAULT_NSF[unit_type])
        affordable_of_type = unit_counts[unit_type]["affordable"]
        market_of_type = unit_counts[unit_type]["market"]

        # Get market rent
        if use_psf_mode:
            rent_psf = st.session_state.get(f"{unit_type}_rent_psf", DEFAULT_RENT_PSF[unit_type])
            market_rent = nsf * rent_psf
        else:
            market_rent = st.session_state.get(f"{unit_type}_rent_monthly", DEFAULT_RENT_MONTHLY[unit_type])

        cols = st.columns([1.5, 0.9, 0.9] + [0.8] * len(AMI_TIERS))

        with cols[0]:
            st.markdown(f"**{UNIT_LABELS[unit_type]}**")

        with cols[1]:
            st.markdown(f"${market_rent:,.0f}")

        market_total = market_rent * market_of_type
        total_market_rent += market_total

        with cols[2]:
            st.markdown(f"${market_total:,.0f}")

        # AMI rents
        for i, tier in enumerate(AMI_TIERS):
            ami_rent = AMI_RENT_LIMITS.get(tier, {}).get(unit_type, 0)
            ami_pct = st.session_state.get(f"{unit_type}_ami_{tier}_pct", 0) / 100.0
            ami_units = round(affordable_of_type * ami_pct)
            tier_total = ami_rent * ami_units
            ami_rent_totals[tier] += tier_total
            total_affordable_rent += tier_total

            with cols[3 + i]:
                if ami_units > 0:
                    st.markdown(f"${tier_total:,.0f}")
                else:
                    st.markdown("-")

    # Totals
    st.divider()
    cols = st.columns([1.5, 0.9, 0.9] + [0.8] * len(AMI_TIERS))
    with cols[0]:
        st.markdown("**TOTAL**")
    with cols[1]:
        st.markdown("")
    with cols[2]:
        st.markdown(f"**${total_market_rent:,.0f}**")
    for i, tier in enumerate(AMI_TIERS):
        with cols[3 + i]:
            if ami_rent_totals[tier] > 0:
                st.markdown(f"**${ami_rent_totals[tier]:,.0f}**")
            else:
                st.markdown("-")

    # GPR/EGI Summary for Mixed Income
    st.divider()
    st.markdown("### Mixed Income Revenue Summary")

    vacancy_rate = st.session_state.get("mixed_vacancy_rate_pct",
                                        st.session_state.get("vacancy_rate_pct", 6.0)) / 100.0
    total_monthly_rent = total_market_rent + total_affordable_rent
    annual_gpr = total_monthly_rent * 12
    annual_egi = annual_gpr * (1 - vacancy_rate)

    # Compare to market rate
    market_rate_gpr = _calculate_market_rate_gpr()
    gpr_difference = annual_gpr - market_rate_gpr

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Monthly GPR", f"${total_monthly_rent:,.0f}")
    with col2:
        # Use delta_color="off" for neutral presentation - lower GPR is expected, not bad
        delta_str = f"{gpr_difference:,.0f} vs Market" if market_rate_gpr else None
        st.metric("Annual GPR", f"${annual_gpr:,.0f}",
                 delta=delta_str,
                 delta_color="off")
    with col3:
        st.metric(f"Annual EGI ({1-vacancy_rate:.0%} occ)", f"${annual_egi:,.0f}")

    # Breakdown
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Market Rent (monthly)", f"${total_market_rent:,.0f}")
    with col2:
        st.metric("Affordable Rent (monthly)", f"${total_affordable_rent:,.0f}")


def _calculate_market_rate_gpr() -> float:
    """Calculate annual GPR for market rate scenario."""
    target_units = st.session_state.get("target_units", 200)
    rent_mode = st.session_state.get("rent_input_mode", RENT_MODE_PSF)
    use_psf_mode = (rent_mode == RENT_MODE_PSF)

    total_monthly = 0
    for unit_type in UNIT_TYPES:
        nsf = st.session_state.get(f"{unit_type}_nsf", DEFAULT_NSF[unit_type])
        alloc_pct = st.session_state.get(f"{unit_type}_alloc_pct", DEFAULT_ALLOCATIONS[unit_type])
        units = round(target_units * alloc_pct / 100.0)

        if use_psf_mode:
            rent_psf = st.session_state.get(f"{unit_type}_rent_psf", DEFAULT_RENT_PSF[unit_type])
            rent = nsf * rent_psf
        else:
            rent = st.session_state.get(f"{unit_type}_rent_monthly", DEFAULT_RENT_MONTHLY[unit_type])

        total_monthly += rent * units

    return total_monthly * 12


def get_unit_mix_from_session_state(efficiency: float = 0.85) -> Dict:
    """Build unit mix dict from session state values.

    Handles both rent input modes:
    - $/NSF mode: uses rent_psf as the driver
    - $/month mode: uses market_rent_monthly as the driver
    """
    from src.models.project import UnitMixEntry

    # Initialize defaults if needed
    _initialize_session_state_defaults()

    # Check which rent mode is active
    rent_mode = st.session_state.get("rent_input_mode", RENT_MODE_PSF)
    use_psf_mode = (rent_mode == RENT_MODE_PSF)

    unit_mix = {}
    for unit_type in UNIT_TYPES:
        nsf = st.session_state.get(f"{unit_type}_nsf", DEFAULT_NSF[unit_type])
        gsf = int(nsf / efficiency) if efficiency > 0 else nsf
        alloc_pct = st.session_state.get(f"{unit_type}_alloc_pct", DEFAULT_ALLOCATIONS[unit_type])
        alloc = alloc_pct / 100.0  # Convert to decimal

        # Get rent based on mode
        if use_psf_mode:
            # $/NSF mode: rent_psf is the input, market_rent_monthly derived
            rent_psf = st.session_state.get(f"{unit_type}_rent_psf", DEFAULT_RENT_PSF[unit_type])
            market_rent_monthly = 0.0  # Will be calculated from rent_psf × nsf
        else:
            # $/month mode: market_rent_monthly is the input
            market_rent_monthly = float(st.session_state.get(f"{unit_type}_rent_monthly", DEFAULT_RENT_MONTHLY[unit_type]))
            rent_psf = 0.0  # Will be ignored since market_rent_monthly is set

        # Get the dominant AMI tier for this unit type
        ami_tier = "50%"  # Default
        max_pct = 0
        for tier in AMI_TIERS:
            pct = st.session_state.get(f"{unit_type}_ami_{tier}_pct", 0)
            if pct > max_pct:
                max_pct = pct
                ami_tier = tier

        unit_mix[unit_type] = UnitMixEntry(
            unit_type=unit_type,
            gsf=gsf,
            allocation=alloc,
            nsf=nsf,
            rent_psf=rent_psf,
            market_rent_monthly=market_rent_monthly,
            ami_tier=ami_tier,
        )

    return unit_mix
