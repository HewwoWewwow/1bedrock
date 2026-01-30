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
# Stored with precision to minimize rounding error when calculating monthly rent
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


def render_unit_mix_tab():
    """Render the redesigned unit mix configuration tab."""
    st.subheader("Unit Mix Configuration")

    # Get key values from session state
    construction_type = st.session_state.get("construction_type", "podium_midrise_5over1")
    efficiency = get_efficiency(construction_type)
    target_units = st.session_state.get("target_units", 200)
    affordable_pct = st.session_state.get("affordable_pct", 0.20)
    total_affordable = round(target_units * affordable_pct)
    total_market = target_units - total_affordable

    st.caption(f"**Construction:** {construction_type.replace('_', ' ').title()} | "
               f"**Efficiency:** {efficiency:.0%} | "
               f"**Total Units:** {target_units} | "
               f"**Affordable:** {total_affordable} ({affordable_pct:.0%})")

    st.divider()

    # =========================================================================
    # SECTION 1: Unit Allocations
    # =========================================================================
    st.markdown("### Unit Allocations")
    st.caption("Enter allocation % for each unit type. Affordable units can be distributed across AMI tiers.")

    # Header row
    cols = st.columns([1.5, 0.8, 0.8, 0.8, 0.8] + [0.7] * len(AMI_TIERS))
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
    for i, tier in enumerate(AMI_TIERS):
        with cols[5 + i]:
            st.markdown(f"**{tier}**")

    # Track totals
    total_alloc = 0.0
    total_units_calc = 0
    total_market_calc = 0
    total_affordable_calc = 0
    ami_totals = {tier: 0 for tier in AMI_TIERS}

    # Data rows for each unit type
    for unit_type in UNIT_TYPES:
        cols = st.columns([1.5, 0.8, 0.8, 0.8, 0.8] + [0.7] * len(AMI_TIERS))

        with cols[0]:
            st.markdown(f"**{UNIT_LABELS[unit_type]}**")

        with cols[1]:
            alloc_pct = st.number_input(
                f"Alloc {unit_type}",
                min_value=0,
                max_value=100,
                value=DEFAULT_ALLOCATIONS[unit_type],
                step=1,
                key=f"{unit_type}_alloc_pct",
                label_visibility="collapsed"
            )
            alloc = alloc_pct / 100.0  # Convert to decimal

        # Calculate units
        units_of_type = round(target_units * alloc)
        affordable_of_type = round(total_affordable * alloc) if total_affordable > 0 else 0
        market_of_type = units_of_type - affordable_of_type

        total_alloc += alloc
        total_units_calc += units_of_type
        total_market_calc += market_of_type
        total_affordable_calc += affordable_of_type

        with cols[2]:
            st.markdown(f"{units_of_type}")

        with cols[3]:
            st.markdown(f"{market_of_type}")

        with cols[4]:
            st.markdown(f"{affordable_of_type}")

        # AMI distribution inputs (as % of affordable units of this type)
        if affordable_of_type > 0:
            for i, tier in enumerate(AMI_TIERS):
                with cols[5 + i]:
                    ami_pct_input = st.number_input(
                        f"{tier} {unit_type}",
                        min_value=0,
                        max_value=100,
                        value=0 if tier != "50%" else 100,  # Default all to 50% AMI
                        step=25,
                        key=f"{unit_type}_ami_{tier}_pct",
                        label_visibility="collapsed"
                    )
                    ami_pct = ami_pct_input / 100.0
                    # Convert percentage to units
                    ami_units = round(affordable_of_type * ami_pct)
                    ami_totals[tier] += ami_units
        else:
            for i in range(len(AMI_TIERS)):
                with cols[5 + i]:
                    st.markdown("-")

    # Totals row
    st.divider()
    cols = st.columns([1.5, 0.8, 0.8, 0.8, 0.8] + [0.7] * len(AMI_TIERS))
    with cols[0]:
        st.markdown("**TOTAL**")
    with cols[1]:
        if abs(total_alloc - 1.0) > 0.01:
            st.error(f"{total_alloc:.0%}")
        else:
            st.success(f"{total_alloc:.0%}")
    with cols[2]:
        st.markdown(f"**{total_units_calc}**")
    with cols[3]:
        st.markdown(f"**{total_market_calc}**")
    with cols[4]:
        st.markdown(f"**{total_affordable_calc}**")
    for i, tier in enumerate(AMI_TIERS):
        with cols[5 + i]:
            if ami_totals[tier] > 0:
                st.markdown(f"**{ami_totals[tier]}**")
            else:
                st.markdown("-")

    # Validation
    if abs(total_alloc - 1.0) > 0.01:
        st.warning(f"Allocations sum to {total_alloc:.0%}. Adjust to reach 100%.")

    st.divider()

    # =========================================================================
    # SECTION 2: NSF by Unit Type
    # =========================================================================
    st.markdown("### Net Square Footage (NSF)")
    st.caption(f"NSF = Net Square Feet | GSF = NSF / {efficiency:.0%} efficiency")

    # Header row
    cols = st.columns([1.5, 0.8, 0.9, 0.9, 0.9] + [0.8] * len(AMI_TIERS))
    with cols[0]:
        st.markdown("**Unit Type**")
    with cols[1]:
        st.markdown("**NSF/Unit**")
    with cols[2]:
        st.markdown("**Total NSF**")
    with cols[3]:
        st.markdown("**Market**")
    with cols[4]:
        st.markdown("**Afford.**")
    for i, tier in enumerate(AMI_TIERS):
        with cols[5 + i]:
            st.markdown(f"**{tier}**")

    # Track NSF and GSF totals
    nsf_totals = {"total": 0, "market": 0, "affordable": 0}
    nsf_ami_totals = {tier: 0 for tier in AMI_TIERS}
    gsf_totals = {"total": 0, "market": 0, "affordable": 0}
    gsf_ami_totals = {tier: 0 for tier in AMI_TIERS}

    for unit_type in UNIT_TYPES:
        cols = st.columns([1.5, 0.8, 0.9, 0.9, 0.9] + [0.8] * len(AMI_TIERS))

        with cols[0]:
            st.markdown(f"**{UNIT_LABELS[unit_type]}**")

        with cols[1]:
            nsf = st.number_input(
                f"NSF {unit_type}",
                min_value=300,
                max_value=2500,
                value=DEFAULT_NSF[unit_type],
                step=25,
                key=f"{unit_type}_nsf",
                label_visibility="collapsed"
            )

        # Calculate GSF from NSF
        gsf = int(nsf / efficiency) if efficiency > 0 else nsf
        st.session_state[f"{unit_type}_gsf"] = gsf

        # Get unit counts
        alloc_pct = st.session_state.get(f"{unit_type}_alloc_pct", DEFAULT_ALLOCATIONS[unit_type])
        alloc = alloc_pct / 100.0
        units_of_type = round(target_units * alloc)
        affordable_of_type = round(total_affordable * alloc) if total_affordable > 0 else 0
        market_of_type = units_of_type - affordable_of_type

        # Calculate NSF totals
        total_nsf = nsf * units_of_type
        market_nsf = nsf * market_of_type
        affordable_nsf = nsf * affordable_of_type

        nsf_totals["total"] += total_nsf
        nsf_totals["market"] += market_nsf
        nsf_totals["affordable"] += affordable_nsf

        # Calculate GSF totals (for summary)
        gsf_totals["total"] += gsf * units_of_type
        gsf_totals["market"] += gsf * market_of_type
        gsf_totals["affordable"] += gsf * affordable_of_type

        with cols[2]:
            st.markdown(f"{total_nsf:,}")

        with cols[3]:
            st.markdown(f"{market_nsf:,}")

        with cols[4]:
            st.markdown(f"{affordable_nsf:,}")

        # AMI NSF breakdown
        if affordable_of_type > 0:
            for i, tier in enumerate(AMI_TIERS):
                ami_pct_val = st.session_state.get(f"{unit_type}_ami_{tier}_pct", 0)
                ami_pct = ami_pct_val / 100.0
                ami_units = round(affordable_of_type * ami_pct)
                ami_nsf = nsf * ami_units
                ami_gsf = gsf * ami_units
                nsf_ami_totals[tier] += ami_nsf
                gsf_ami_totals[tier] += ami_gsf
                with cols[5 + i]:
                    if ami_nsf > 0:
                        st.markdown(f"{ami_nsf:,}")
                    else:
                        st.markdown("-")
        else:
            for i in range(len(AMI_TIERS)):
                with cols[5 + i]:
                    st.markdown("-")

    # Totals rows
    st.divider()

    # NSF Totals
    cols = st.columns([1.5, 0.8, 0.9, 0.9, 0.9] + [0.8] * len(AMI_TIERS))
    with cols[0]:
        st.markdown("**TOTAL NSF**")
    with cols[1]:
        st.markdown("")
    with cols[2]:
        st.markdown(f"**{nsf_totals['total']:,}**")
    with cols[3]:
        st.markdown(f"**{nsf_totals['market']:,}**")
    with cols[4]:
        st.markdown(f"**{nsf_totals['affordable']:,}**")
    for i, tier in enumerate(AMI_TIERS):
        with cols[5 + i]:
            if nsf_ami_totals[tier] > 0:
                st.markdown(f"**{nsf_ami_totals[tier]:,}**")
            else:
                st.markdown("-")

    # GSF Totals
    cols = st.columns([1.5, 0.8, 0.9, 0.9, 0.9] + [0.8] * len(AMI_TIERS))
    with cols[0]:
        st.markdown("**TOTAL GSF**")
    with cols[1]:
        st.markdown("")
    with cols[2]:
        st.markdown(f"**{gsf_totals['total']:,}**")
    with cols[3]:
        st.markdown(f"**{gsf_totals['market']:,}**")
    with cols[4]:
        st.markdown(f"**{gsf_totals['affordable']:,}**")
    for i, tier in enumerate(AMI_TIERS):
        with cols[5 + i]:
            if gsf_ami_totals[tier] > 0:
                st.markdown(f"**{gsf_ami_totals[tier]:,}**")
            else:
                st.markdown("-")

    st.divider()

    # =========================================================================
    # SECTION 3: Rents by Unit Type and AMI
    # =========================================================================
    st.markdown("### Monthly Rents")

    # Rent input mode toggle
    rent_mode_col, _ = st.columns([2, 3])
    with rent_mode_col:
        rent_mode = st.radio(
            "Rent input mode",
            options=[RENT_MODE_PSF, RENT_MODE_MONTHLY],
            horizontal=True,
            key="rent_input_mode",
            help="$/NSF: rent adjusts with unit size. $/month: enter explicit rent for quick estimates."
        )

    use_psf_mode = (rent_mode == RENT_MODE_PSF)

    # --- Per-Unit Rent Table ---
    st.markdown("#### Rent per Unit")
    if use_psf_mode:
        st.caption("$/NSF is the driver - monthly rent adjusts with unit size. Affordable rents are AMI limits.")
    else:
        st.caption("Enter monthly rent directly. $/NSF shown for reference. Affordable rents are AMI limits.")

    # Header row - columns change based on mode
    if use_psf_mode:
        cols = st.columns([1.5, 0.8, 1.0] + [0.9] * len(AMI_TIERS))
        with cols[0]:
            st.markdown("**Unit Type**")
        with cols[1]:
            st.markdown("**$/NSF**")
        with cols[2]:
            st.markdown("**Market $/mo**")
    else:
        cols = st.columns([1.5, 1.0, 0.8] + [0.9] * len(AMI_TIERS))
        with cols[0]:
            st.markdown("**Unit Type**")
        with cols[1]:
            st.markdown("**Market $/mo**")
        with cols[2]:
            st.markdown("**$/NSF**")
    for i, tier in enumerate(AMI_TIERS):
        with cols[3 + i]:
            st.markdown(f"**{tier}**")

    # Store data for the totals table
    rent_data = {}

    for unit_type in UNIT_TYPES:
        nsf = st.session_state.get(f"{unit_type}_nsf", DEFAULT_NSF[unit_type])

        if use_psf_mode:
            # $/NSF mode: $/NSF is input, monthly rent is calculated
            cols = st.columns([1.5, 0.8, 1.0] + [0.9] * len(AMI_TIERS))

            with cols[0]:
                st.markdown(f"**{UNIT_LABELS[unit_type]}**")

            with cols[1]:
                rent_psf = st.number_input(
                    f"$/NSF {unit_type}",
                    min_value=1.00,
                    max_value=10.00,
                    value=DEFAULT_RENT_PSF[unit_type],
                    step=0.05,
                    format="%.2f",
                    key=f"{unit_type}_rent_psf",
                    label_visibility="collapsed"
                )

            # Calculate monthly rent from NSF × $/NSF
            market_rent = nsf * rent_psf

            with cols[2]:
                st.markdown(f"**${market_rent:,.0f}**")

        else:
            # $/month mode: monthly rent is input, $/NSF is calculated for reference
            cols = st.columns([1.5, 1.0, 0.8] + [0.9] * len(AMI_TIERS))

            with cols[0]:
                st.markdown(f"**{UNIT_LABELS[unit_type]}**")

            with cols[1]:
                market_rent = st.number_input(
                    f"$/mo {unit_type}",
                    min_value=500,
                    max_value=15000,
                    value=DEFAULT_RENT_MONTHLY[unit_type],
                    step=25,
                    key=f"{unit_type}_rent_monthly",
                    label_visibility="collapsed"
                )

            # Calculate $/NSF for reference
            rent_psf = market_rent / nsf if nsf > 0 else 0

            with cols[2]:
                st.markdown(f"${rent_psf:.2f}")

        # Get unit counts
        alloc_pct = st.session_state.get(f"{unit_type}_alloc_pct", DEFAULT_ALLOCATIONS[unit_type])
        alloc = alloc_pct / 100.0
        units_of_type = round(target_units * alloc)
        affordable_of_type = round(total_affordable * alloc) if total_affordable > 0 else 0
        market_of_type = units_of_type - affordable_of_type

        # Store for totals table
        rent_data[unit_type] = {
            "market_rent": market_rent,
            "market_units": market_of_type,
            "affordable_units": affordable_of_type,
            "ami_rents": {},
            "ami_units": {}
        }

        # Affordable rents by AMI tier
        for i, tier in enumerate(AMI_TIERS):
            ami_rent = AMI_RENT_LIMITS.get(tier, {}).get(unit_type, 0)
            ami_pct_val = st.session_state.get(f"{unit_type}_ami_{tier}_pct", 0)
            ami_pct = ami_pct_val / 100.0
            ami_units = round(affordable_of_type * ami_pct)

            rent_data[unit_type]["ami_rents"][tier] = ami_rent
            rent_data[unit_type]["ami_units"][tier] = ami_units

            with cols[3 + i]:
                if ami_rent > 0:
                    st.markdown(f"${ami_rent:,}")
                else:
                    st.markdown("-")

    st.divider()

    # --- Total Monthly Rent Table ---
    st.markdown("#### Total Monthly Rent (All Units)")
    st.caption("Rent per unit × number of units")

    # Header row
    cols = st.columns([1.5, 1.0, 1.0] + [0.9] * len(AMI_TIERS))
    with cols[0]:
        st.markdown("**Unit Type**")
    with cols[1]:
        st.markdown("**Market**")
    with cols[2]:
        st.markdown("**Affordable**")
    for i, tier in enumerate(AMI_TIERS):
        with cols[3 + i]:
            st.markdown(f"**{tier}**")

    # Track totals
    total_market_rent = 0
    total_affordable_rent = 0
    ami_rent_totals = {tier: 0 for tier in AMI_TIERS}

    for unit_type in UNIT_TYPES:
        data = rent_data[unit_type]
        cols = st.columns([1.5, 1.0, 1.0] + [0.9] * len(AMI_TIERS))

        with cols[0]:
            st.markdown(f"**{UNIT_LABELS[unit_type]}**")

        # Market total
        market_total = data["market_rent"] * data["market_units"]
        total_market_rent += market_total
        with cols[1]:
            if market_total > 0:
                st.markdown(f"${market_total:,.0f}")
            else:
                st.markdown("-")

        # Affordable total (sum across AMI tiers)
        aff_total = sum(data["ami_rents"][t] * data["ami_units"][t] for t in AMI_TIERS)
        total_affordable_rent += aff_total
        with cols[2]:
            if aff_total > 0:
                st.markdown(f"${aff_total:,.0f}")
            else:
                st.markdown("-")

        # By AMI tier
        for i, tier in enumerate(AMI_TIERS):
            tier_total = data["ami_rents"][tier] * data["ami_units"][tier]
            ami_rent_totals[tier] += tier_total
            with cols[3 + i]:
                if tier_total > 0:
                    st.markdown(f"${tier_total:,.0f}")
                else:
                    st.markdown("-")

    # Totals row
    st.divider()
    cols = st.columns([1.5, 1.0, 1.0] + [0.9] * len(AMI_TIERS))
    with cols[0]:
        st.markdown("**TOTAL**")
    with cols[1]:
        st.markdown(f"**${total_market_rent:,.0f}**")
    with cols[2]:
        st.markdown(f"**${total_affordable_rent:,.0f}**")
    for i, tier in enumerate(AMI_TIERS):
        with cols[3 + i]:
            if ami_rent_totals[tier] > 0:
                st.markdown(f"**${ami_rent_totals[tier]:,.0f}**")
            else:
                st.markdown("-")

    # GPR summary
    st.divider()
    total_monthly_rent = total_market_rent + total_affordable_rent
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Monthly GPR", f"${total_monthly_rent:,.0f}")
    with col2:
        st.metric("Market Rent", f"${total_market_rent:,.0f}")
    with col3:
        st.metric("Affordable Rent", f"${total_affordable_rent:,.0f}")


def get_unit_mix_from_session_state(efficiency: float = 0.85) -> Dict:
    """Build unit mix dict from session state values.

    Handles both rent input modes:
    - $/NSF mode: uses rent_psf as the driver
    - $/month mode: uses market_rent_monthly as the driver
    """
    from src.models.project import UnitMixEntry

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
