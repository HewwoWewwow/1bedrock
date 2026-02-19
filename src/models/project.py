"""Project data model containing all inputs for a development scenario."""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Dict

from .lookups import ConstructionType, DEFAULT_TAX_RATES
from .incentives import IncentiveConfig, IncentiveToggles


class Scenario(Enum):
    """Scenario type for comparison."""

    MARKET = "market"
    MIXED_INCOME = "mixed_income"


class TIFStartTiming(Enum):
    """When TIF stream payments begin."""

    LEASEUP = "leaseup"
    OPERATIONS = "operations"


class ReversionNOIBasis(Enum):
    """Which NOI to use for reversion valuation."""

    FORWARD = "forward"  # Next 12 months projected
    TRAILING = "trailing"  # Prior 12 months actual


@dataclass
class UnitMixEntry:
    """Single entry in the unit mix."""

    unit_type: str  # "studio", "1br", "2br", "3br", "4br"
    gsf: int  # Gross square feet per unit (can be calculated from nsf/efficiency)
    allocation: float  # Percentage of total units (0-1)
    nsf: int = 0  # Net square feet per unit (input - if 0, use gsf as-is)
    rent_psf: float = 0.0  # Monthly rent per NSF for market units (0 = use project default)
    market_rent_monthly: float = 0.0  # Explicit monthly market rent (0 = derive from rent_psf × nsf)
    ami_tier: str = ""  # AMI tier for affordable units of this type (e.g., "50%")

    def get_unit_count(self, total_units: int) -> int:
        """Calculate number of units of this type.

        Args:
            total_units: Total number of units in the project.

        Returns:
            Number of units of this type (rounded).
        """
        return round(total_units * self.allocation)

    def get_nsf(self) -> int:
        """Get NSF (returns nsf if set, otherwise estimates from gsf at 0.82 efficiency)."""
        return self.nsf if self.nsf > 0 else int(self.gsf * 0.82)

    def get_gsf_from_efficiency(self, efficiency: float) -> int:
        """Calculate GSF from NSF and efficiency factor.

        Args:
            efficiency: Net-to-gross efficiency ratio (e.g., 0.82)

        Returns:
            Gross square feet.
        """
        if self.nsf > 0 and efficiency > 0:
            return int(self.nsf / efficiency)
        return self.gsf


@dataclass
class ProjectInputs:
    """Complete input parameters for a development project.

    This dataclass contains all inputs needed to run a DCF analysis
    for either a market-rate or mixed-income scenario.
    """

    # === Project Timing ===
    predevelopment_start: date = field(default_factory=lambda: date(2026, 1, 1))
    predevelopment_months: int = 18
    construction_months: int = 24
    leaseup_months: int = 12
    operations_months: int = 12

    # === Land & Construction ===
    land_cost: float = 3_000_000.0  # Total land cost (lump sum)
    target_units: int = 200
    hard_cost_per_unit: float = 175_000  # Excel default
    soft_cost_pct: float = 0.30  # As % of hard costs (design, legal, etc.)
    predevelopment_cost_pct: float = 0.0972  # As % of hard costs (separate from soft costs)
    hard_cost_contingency_pct: float = 0.05  # 5% contingency on hard costs
    soft_cost_contingency_pct: float = 0.05  # 5% contingency on soft costs
    developer_fee_pct: float = 0.04  # Developer fee as % of hard costs
    construction_type: ConstructionType = ConstructionType.PODIUM_5OVER1

    # === Unit Mix ===
    # Default values from Austin TIF Model spreadsheet v10c
    # NSF assumes 0.85 efficiency; rents calculated as rent_psf × nsf
    # Rent PSF pattern: $3.50 (studio/1br), $3.40 (2br), $3.30 (3br), $3.20 (4br)
    unit_mix: Dict[str, UnitMixEntry] = field(default_factory=lambda: {
        "studio": UnitMixEntry("studio", gsf=600, allocation=0.12, nsf=510, rent_psf=3.50),
        "1br": UnitMixEntry("1br", gsf=750, allocation=0.25, nsf=638, rent_psf=3.50),
        "2br": UnitMixEntry("2br", gsf=900, allocation=0.45, nsf=765, rent_psf=3.40),
        "3br": UnitMixEntry("3br", gsf=1150, allocation=0.15, nsf=978, rent_psf=3.30),
        "4br": UnitMixEntry("4br", gsf=1450, allocation=0.03, nsf=1233, rent_psf=3.20),
    })

    # === Rents ===
    market_rent_psf: float = 2.50  # Monthly rent per GSF for market units

    # === Operating Assumptions ===
    vacancy_rate: float = 0.06  # Stabilized vacancy
    leaseup_pace: float = 0.08  # Percentage occupied per month during lease-up
    max_occupancy: float = 0.94  # Stabilized occupancy ceiling

    # OpEx per unit per year
    opex_utilities: float = 1200.0
    opex_management_pct: float = 0.05  # As percentage of EGI
    opex_maintenance: float = 1500.0
    opex_misc: float = 650.0
    reserves_pct: float = 0.02  # As percentage of EGI

    # === Growth Rates (Annual) ===
    market_rent_growth: float = 0.02
    affordable_rent_growth: float = 0.01
    opex_growth: float = 0.03
    property_tax_growth: float = 0.01

    # === Financing ===
    # Construction Loan
    construction_rate: float = 0.075
    construction_ltc: float = 0.65  # Loan-to-cost
    idc_leaseup_months: int = 4  # Months of lease-up included in IDC (net of NOI offset)

    # === Reserves ===
    operating_reserve_months: int = 3  # Months of OpEx held in reserve
    leaseup_reserve_months: int = 6  # Months of debt service held during lease-up

    # Permanent Loan
    perm_rate: float = 0.06
    perm_amort_years: int = 20
    perm_ltv_max: float = 0.65  # Maximum loan-to-value
    perm_dscr_min: float = 1.25  # Minimum debt service coverage ratio

    # === Property Taxes ===
    existing_assessed_value: float = 5_000_000.0
    tax_rates: Dict[str, float] = field(default_factory=lambda: DEFAULT_TAX_RATES.copy())

    # === Exit Assumptions ===
    exit_cap_rate: float = 0.055
    reversion_noi_basis: ReversionNOIBasis = ReversionNOIBasis.FORWARD
    selling_costs_pct: float = 0.02  # Broker fees, transfer taxes, etc.

    # === Incentives (Mixed-Income Only) ===
    selected_tier: int = 2
    affordable_pct: float = 0.20  # Percentage of units that are affordable
    ami_level: str = "50%"
    incentive_config: IncentiveConfig | None = None
    tif_start_timing: TIFStartTiming = TIFStartTiming.OPERATIONS

    @property
    def total_months(self) -> int:
        """Total project duration including reversion month."""
        return (
            self.predevelopment_months
            + self.construction_months
            + self.leaseup_months
            + self.operations_months
            + 1  # Reversion month
        )

    @property
    def construction_start_month(self) -> int:
        """Month when construction starts (1-indexed)."""
        return self.predevelopment_months + 1

    @property
    def leaseup_start_month(self) -> int:
        """Month when lease-up starts (1-indexed)."""
        return self.predevelopment_months + self.construction_months + 1

    @property
    def operations_start_month(self) -> int:
        """Month when stabilized operations start (1-indexed)."""
        return self.leaseup_start_month + self.leaseup_months

    @property
    def reversion_month(self) -> int:
        """Month when reversion (sale) occurs (1-indexed)."""
        return self.operations_start_month + self.operations_months

    @property
    def total_tax_rate(self) -> float:
        """Total tax rate per $100 of assessed value."""
        return sum(self.tax_rates.values())

    @property
    def annual_opex_per_unit(self) -> float:
        """Total annual operating expenses per unit (excluding management %)."""
        return self.opex_utilities + self.opex_maintenance + self.opex_misc

    def get_weighted_avg_gsf(self) -> float:
        """Calculate weighted average GSF per unit.

        Returns:
            Weighted average gross square feet per unit.
        """
        total_weight = 0.0
        weighted_gsf = 0.0
        for entry in self.unit_mix.values():
            weighted_gsf += entry.gsf * entry.allocation
            total_weight += entry.allocation
        return weighted_gsf / total_weight if total_weight > 0 else 0.0

    def get_affordable_unit_count(self) -> int:
        """Calculate number of affordable units.

        Returns:
            Number of affordable units (rounded).
        """
        return round(self.target_units * self.affordable_pct)

    def get_market_unit_count(self) -> int:
        """Calculate number of market-rate units.

        Returns:
            Number of market-rate units.
        """
        return self.target_units - self.get_affordable_unit_count()

    def validate(self) -> list[str]:
        """Validate inputs and return list of errors.

        Returns:
            List of validation error messages. Empty if valid.
        """
        errors = []

        # Unit count validation
        if not 10 <= self.target_units <= 1000:
            errors.append(f"target_units must be 10-1000, got {self.target_units}")

        # Percentage validations
        if not 0 <= self.affordable_pct <= 1:
            errors.append(f"affordable_pct must be 0-1, got {self.affordable_pct}")
        if not 0 <= self.vacancy_rate <= 0.25:
            errors.append(f"vacancy_rate must be 0-0.25, got {self.vacancy_rate}")

        # Rate validations
        if not 0.03 <= self.construction_rate <= 0.15:
            errors.append(f"construction_rate must be 0.03-0.15, got {self.construction_rate}")
        if not 0.03 <= self.perm_rate <= 0.12:
            errors.append(f"perm_rate must be 0.03-0.12, got {self.perm_rate}")
        if not 0.04 <= self.exit_cap_rate <= 0.12:
            errors.append(f"exit_cap_rate must be 0.04-0.12, got {self.exit_cap_rate}")

        # Cost validations
        if not 50000 <= self.hard_cost_per_unit <= 500000:
            errors.append(
                f"hard_cost_per_unit must be $50k-$500k, got ${self.hard_cost_per_unit:,.0f}"
            )

        # Unit mix must sum to ~1.0
        allocation_sum = sum(entry.allocation for entry in self.unit_mix.values())
        if not 0.99 <= allocation_sum <= 1.01:
            errors.append(f"unit_mix allocations must sum to 1.0, got {allocation_sum:.2f}")

        # Affordable unit sanity check
        affordable_count = self.get_affordable_unit_count()
        market_count = self.get_market_unit_count()
        if affordable_count > self.target_units:
            errors.append(
                f"affordable units ({affordable_count}) exceed target_units ({self.target_units})"
            )
        if market_count < 0:
            errors.append(
                f"market units ({market_count}) is negative — affordable_pct too high"
            )

        # Weighted average GSF sanity check
        avg_gsf = self.get_weighted_avg_gsf()
        if avg_gsf < 200:
            errors.append(f"weighted avg GSF is implausibly low ({avg_gsf:.0f} SF)")
        elif avg_gsf > 3000:
            errors.append(f"weighted avg GSF is implausibly high ({avg_gsf:.0f} SF)")

        # Incentive validation
        if self.incentive_config is not None:
            toggles = self.incentive_config.toggles
            # TIF lump sum and stream are mutually exclusive
            if toggles.tif_lump_sum and toggles.tif_stream:
                errors.append("Cannot enable both TIF lump sum and TIF stream")
            # Tax abatement and TIF are mutually exclusive (warning, not error)
            # This is enforced in IncentiveToggles.__post_init__

        return errors

    def copy(self) -> "ProjectInputs":
        """Create a deep copy of the project inputs.

        Returns:
            New ProjectInputs instance with copied values.
        """
        return ProjectInputs(
            predevelopment_start=self.predevelopment_start,
            predevelopment_months=self.predevelopment_months,
            construction_months=self.construction_months,
            leaseup_months=self.leaseup_months,
            operations_months=self.operations_months,
            land_cost=self.land_cost,
            target_units=self.target_units,
            hard_cost_per_unit=self.hard_cost_per_unit,
            soft_cost_pct=self.soft_cost_pct,
            predevelopment_cost_pct=self.predevelopment_cost_pct,
            hard_cost_contingency_pct=self.hard_cost_contingency_pct,
            soft_cost_contingency_pct=self.soft_cost_contingency_pct,
            developer_fee_pct=self.developer_fee_pct,
            construction_type=self.construction_type,
            unit_mix={k: UnitMixEntry(
                          v.unit_type, v.gsf, v.allocation,
                          nsf=v.nsf, rent_psf=v.rent_psf,
                          market_rent_monthly=v.market_rent_monthly, ami_tier=v.ami_tier)
                      for k, v in self.unit_mix.items()},
            market_rent_psf=self.market_rent_psf,
            vacancy_rate=self.vacancy_rate,
            leaseup_pace=self.leaseup_pace,
            max_occupancy=self.max_occupancy,
            opex_utilities=self.opex_utilities,
            opex_management_pct=self.opex_management_pct,
            opex_maintenance=self.opex_maintenance,
            opex_misc=self.opex_misc,
            reserves_pct=self.reserves_pct,
            market_rent_growth=self.market_rent_growth,
            affordable_rent_growth=self.affordable_rent_growth,
            opex_growth=self.opex_growth,
            property_tax_growth=self.property_tax_growth,
            construction_rate=self.construction_rate,
            construction_ltc=self.construction_ltc,
            idc_leaseup_months=self.idc_leaseup_months,
            operating_reserve_months=self.operating_reserve_months,
            leaseup_reserve_months=self.leaseup_reserve_months,
            perm_rate=self.perm_rate,
            perm_amort_years=self.perm_amort_years,
            perm_ltv_max=self.perm_ltv_max,
            perm_dscr_min=self.perm_dscr_min,
            existing_assessed_value=self.existing_assessed_value,
            tax_rates=self.tax_rates.copy(),
            exit_cap_rate=self.exit_cap_rate,
            reversion_noi_basis=self.reversion_noi_basis,
            selling_costs_pct=self.selling_costs_pct,
            selected_tier=self.selected_tier,
            affordable_pct=self.affordable_pct,
            ami_level=self.ami_level,
            incentive_config=self.incentive_config,
            tif_start_timing=self.tif_start_timing,
        )
