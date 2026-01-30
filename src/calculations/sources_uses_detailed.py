"""Detailed Sources and Uses with flexible input methods."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict


class InputMethod(Enum):
    """How a line item value is determined."""
    DIRECT = "direct"           # Direct dollar input
    PER_UNIT = "per_unit"       # $/unit × units
    PER_GSF = "per_gsf"         # $/GSF × GSF
    PERCENT = "percent"         # % of a base (hard costs, TDC, etc.)
    CALCULATED = "calculated"   # Derived from other values


class LandCostMethod(Enum):
    """How land cost is determined."""
    DIRECT = "direct"           # Direct total input
    PER_ACRE = "per_acre"       # $/acre × acres needed


@dataclass
class LineItem:
    """A single line item in Sources or Uses."""
    name: str
    amount: float = 0.0
    input_method: InputMethod = InputMethod.DIRECT

    # Input values (used based on input_method)
    direct_input: float = 0.0
    per_unit_input: float = 0.0
    per_gsf_input: float = 0.0
    percent_input: float = 0.0      # As decimal (0.05 = 5%)
    percent_base: str = ""          # What the % is based on ("hard_costs", "tdc", etc.)

    # Reference values for display
    units: int = 0
    gsf: int = 0

    @property
    def per_unit(self) -> float:
        """Amount per unit."""
        return self.amount / self.units if self.units > 0 else 0.0

    @property
    def per_gsf(self) -> float:
        """Amount per GSF."""
        return self.amount / self.gsf if self.gsf > 0 else 0.0


@dataclass
class UsesDetail:
    """Detailed breakdown of Uses (costs)."""

    # === LAND & ACQUISITION ===
    land_cost_method: LandCostMethod = LandCostMethod.DIRECT
    land_direct: float = 0.0            # Direct input
    land_per_acre: float = 0.0          # $/acre input
    land_acres: float = 0.0             # Acres (calculated or input)
    land_total: float = 0.0             # Final land cost

    closing_costs: LineItem = field(default_factory=lambda: LineItem("Closing Costs"))
    due_diligence: LineItem = field(default_factory=lambda: LineItem("Due Diligence"))

    # === HARD COSTS ===
    hard_cost_method: InputMethod = InputMethod.PER_UNIT

    site_work: LineItem = field(default_factory=lambda: LineItem("Site Work"))
    building_construction: LineItem = field(default_factory=lambda: LineItem("Building Construction"))
    parking: LineItem = field(default_factory=lambda: LineItem("Parking"))
    common_areas: LineItem = field(default_factory=lambda: LineItem("Common Areas"))
    hard_cost_contingency_pct: float = 0.05  # 5% default
    hard_cost_contingency: float = 0.0
    hard_costs_subtotal: float = 0.0    # Before contingency
    hard_costs_total: float = 0.0       # With contingency

    # === SOFT COSTS ===
    architecture_engineering: LineItem = field(default_factory=lambda: LineItem(
        "Architecture & Engineering", input_method=InputMethod.PERCENT, percent_base="hard_costs"
    ))
    legal_accounting: LineItem = field(default_factory=lambda: LineItem("Legal & Accounting"))
    permits_fees: LineItem = field(default_factory=lambda: LineItem("Permits & Fees"))
    marketing: LineItem = field(default_factory=lambda: LineItem("Marketing"))
    developer_fee: LineItem = field(default_factory=lambda: LineItem(
        "Developer Fee", input_method=InputMethod.PERCENT, percent_base="hard_costs"
    ))
    soft_cost_contingency_pct: float = 0.05  # 5% default
    soft_cost_contingency: float = 0.0
    soft_costs_subtotal: float = 0.0    # Before contingency
    soft_costs_total: float = 0.0       # With contingency

    # === FINANCING COSTS ===
    construction_loan_fee_pct: float = 0.01  # 1% default
    construction_loan_fee: float = 0.0
    idc: float = 0.0                    # Interest during construction (calculated)
    permanent_loan_fee_pct: float = 0.01
    permanent_loan_fee: float = 0.0
    financing_costs_total: float = 0.0

    # === RESERVES ===
    operating_reserve_months: int = 3   # Months of operating expenses
    operating_reserve: float = 0.0
    leaseup_reserve_months: int = 6     # Months of debt service during lease-up
    leaseup_reserve: float = 0.0
    reserves_total: float = 0.0

    # === TOTALS ===
    total_land_acquisition: float = 0.0
    total_development_cost: float = 0.0  # TDC

    # Reference values
    target_units: int = 0
    total_gsf: int = 0


@dataclass
class SourcesDetail:
    """Detailed breakdown of Sources (funding)."""

    # === SENIOR DEBT ===
    construction_loan_ltc_cap: float = 0.65  # Max LTC (user input)
    construction_loan: float = 0.0
    construction_loan_ltc_actual: float = 0.0

    # === MEZZANINE / PREFERRED ===
    mezzanine_debt: float = 0.0
    mezzanine_rate: float = 0.12        # 12% default
    preferred_equity: float = 0.0
    preferred_return: float = 0.10      # 10% default

    # === GAP FUNDING ===
    tif_lump_sum: float = 0.0
    grants: float = 0.0
    fee_waivers: float = 0.0            # SMART program, etc.
    other_incentives: float = 0.0
    gap_funding_total: float = 0.0

    # === DEFERRED ITEMS ===
    deferred_developer_fee: float = 0.0
    deferred_developer_fee_pct: float = 0.0  # % of developer fee deferred

    # === EQUITY (RESIDUAL) ===
    equity_required: float = 0.0        # Calculated as residual
    equity_pct_of_tdc: float = 0.0

    # === TOTALS ===
    total_sources: float = 0.0

    # Reference
    target_units: int = 0


@dataclass
class SourcesUsesDetailed:
    """Complete detailed Sources & Uses."""
    uses: UsesDetail
    sources: SourcesDetail

    # Validation
    is_balanced: bool = False
    balance_difference: float = 0.0

    def validate(self) -> bool:
        """Check that sources equal uses."""
        self.balance_difference = abs(self.sources.total_sources - self.uses.total_development_cost)
        self.is_balanced = self.balance_difference < 1.0  # $1 tolerance
        return self.is_balanced


def calculate_line_item(
    item: LineItem,
    units: int,
    gsf: int,
    base_amounts: Dict[str, float],
) -> float:
    """Calculate line item amount based on its input method."""
    item.units = units
    item.gsf = gsf

    if item.input_method == InputMethod.DIRECT:
        item.amount = item.direct_input
    elif item.input_method == InputMethod.PER_UNIT:
        item.amount = item.per_unit_input * units
    elif item.input_method == InputMethod.PER_GSF:
        item.amount = item.per_gsf_input * gsf
    elif item.input_method == InputMethod.PERCENT:
        base = base_amounts.get(item.percent_base, 0)
        item.amount = base * item.percent_input
    # CALCULATED items are set externally

    return item.amount


def calculate_sources_uses_detailed(
    # Project parameters
    target_units: int,
    total_gsf: int,
    units_per_acre: float,

    # Land inputs
    land_cost_method: LandCostMethod,
    land_direct: float = 0.0,
    land_per_acre: float = 0.0,

    # Hard cost inputs (simplified for now - $/unit)
    hard_cost_per_unit: float = 155_000,
    hard_cost_contingency_pct: float = 0.05,

    # Soft cost inputs
    soft_cost_pct_of_hard: float = 0.30,
    developer_fee_pct: float = 0.04,
    soft_cost_contingency_pct: float = 0.05,

    # Financing inputs
    construction_ltc_cap: float = 0.65,
    construction_rate: float = 0.075,
    construction_months: int = 24,
    construction_loan_fee_pct: float = 0.01,

    # Reserves
    monthly_opex: float = 0.0,
    monthly_debt_service: float = 0.0,
    operating_reserve_months: int = 3,
    leaseup_reserve_months: int = 6,

    # Gap funding / incentives
    tif_lump_sum: float = 0.0,
    grants: float = 0.0,
    fee_waivers: float = 0.0,

    # Mezzanine / preferred
    mezzanine_debt: float = 0.0,
    mezzanine_rate: float = 0.12,
    preferred_equity: float = 0.0,
    preferred_return: float = 0.10,

    # Deferred
    deferred_developer_fee_pct: float = 0.0,

) -> SourcesUsesDetailed:
    """Calculate detailed sources and uses with iterative IDC solve."""

    uses = UsesDetail(target_units=target_units, total_gsf=total_gsf)
    sources = SourcesDetail(target_units=target_units)

    # === LAND ===
    uses.land_cost_method = land_cost_method
    uses.land_direct = land_direct
    uses.land_per_acre = land_per_acre

    if land_cost_method == LandCostMethod.DIRECT:
        uses.land_total = land_direct
        uses.land_acres = land_direct / land_per_acre if land_per_acre > 0 else 0
    else:
        # Calculate acres needed
        uses.land_acres = target_units / units_per_acre if units_per_acre > 0 else 0
        uses.land_total = uses.land_acres * land_per_acre

    uses.total_land_acquisition = uses.land_total

    # === HARD COSTS (simplified - will expand later) ===
    uses.hard_costs_subtotal = hard_cost_per_unit * target_units
    uses.hard_cost_contingency_pct = hard_cost_contingency_pct
    uses.hard_cost_contingency = uses.hard_costs_subtotal * hard_cost_contingency_pct
    uses.hard_costs_total = uses.hard_costs_subtotal + uses.hard_cost_contingency

    # === SOFT COSTS ===
    base_soft = uses.hard_costs_total * soft_cost_pct_of_hard
    developer_fee_amount = uses.hard_costs_total * developer_fee_pct

    uses.soft_costs_subtotal = base_soft + developer_fee_amount - fee_waivers
    uses.soft_costs_subtotal = max(0, uses.soft_costs_subtotal)
    uses.soft_cost_contingency_pct = soft_cost_contingency_pct
    uses.soft_cost_contingency = uses.soft_costs_subtotal * soft_cost_contingency_pct
    uses.soft_costs_total = uses.soft_costs_subtotal + uses.soft_cost_contingency

    # Store developer fee for deferred calculation
    uses.developer_fee.amount = developer_fee_amount

    # === COSTS BEFORE FINANCING ===
    costs_before_financing = (
        uses.total_land_acquisition +
        uses.hard_costs_total +
        uses.soft_costs_total
    )

    # === ITERATIVE IDC SOLVE ===
    monthly_rate = construction_rate / 12
    idc = 0.0
    tolerance = 100.0
    max_iterations = 50

    for _ in range(max_iterations):
        # TDC estimate
        tdc_estimate = costs_before_financing + idc

        # Size construction loan
        construction_loan = tdc_estimate * construction_ltc_cap

        # Loan fee
        loan_fee = construction_loan * construction_loan_fee_pct

        # Calculate new IDC (using average balance method)
        average_balance = construction_loan * 0.65  # S-curve approximation
        new_idc = average_balance * monthly_rate * construction_months

        if abs(new_idc - idc) < tolerance:
            idc = new_idc
            break
        idc = new_idc

    # === FINANCING COSTS ===
    uses.idc = idc
    uses.construction_loan_fee_pct = construction_loan_fee_pct
    uses.construction_loan_fee = construction_loan * construction_loan_fee_pct
    uses.financing_costs_total = uses.idc + uses.construction_loan_fee

    # === RESERVES ===
    uses.operating_reserve_months = operating_reserve_months
    uses.operating_reserve = monthly_opex * operating_reserve_months
    uses.leaseup_reserve_months = leaseup_reserve_months
    uses.leaseup_reserve = monthly_debt_service * leaseup_reserve_months
    uses.reserves_total = uses.operating_reserve + uses.leaseup_reserve

    # === TOTAL DEVELOPMENT COST ===
    uses.total_development_cost = (
        uses.total_land_acquisition +
        uses.hard_costs_total +
        uses.soft_costs_total +
        uses.financing_costs_total +
        uses.reserves_total
    )

    # === SOURCES ===
    sources.construction_loan_ltc_cap = construction_ltc_cap
    sources.construction_loan = uses.total_development_cost * construction_ltc_cap
    sources.construction_loan_ltc_actual = construction_ltc_cap

    # Gap funding
    sources.tif_lump_sum = tif_lump_sum
    sources.grants = grants
    sources.fee_waivers = fee_waivers
    sources.gap_funding_total = tif_lump_sum + grants + fee_waivers

    # Mezzanine / preferred
    sources.mezzanine_debt = mezzanine_debt
    sources.mezzanine_rate = mezzanine_rate
    sources.preferred_equity = preferred_equity
    sources.preferred_return = preferred_return

    # Deferred developer fee
    sources.deferred_developer_fee_pct = deferred_developer_fee_pct
    sources.deferred_developer_fee = uses.developer_fee.amount * deferred_developer_fee_pct

    # Equity (residual)
    sources.equity_required = (
        uses.total_development_cost -
        sources.construction_loan -
        sources.mezzanine_debt -
        sources.preferred_equity -
        sources.gap_funding_total -
        sources.deferred_developer_fee
    )
    sources.equity_required = max(0, sources.equity_required)
    sources.equity_pct_of_tdc = (
        sources.equity_required / uses.total_development_cost
        if uses.total_development_cost > 0 else 0
    )

    # Total sources
    sources.total_sources = (
        sources.construction_loan +
        sources.mezzanine_debt +
        sources.preferred_equity +
        sources.gap_funding_total +
        sources.deferred_developer_fee +
        sources.equity_required
    )

    # Build result
    result = SourcesUsesDetailed(uses=uses, sources=sources)
    result.validate()

    return result
