"""Scenario configuration for single project or comparison mode."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any


class ModelMode(Enum):
    """Model operating mode."""
    SINGLE_PROJECT = "single"      # One scenario only
    COMPARISON = "comparison"       # Side-by-side comparison


class ProjectType(Enum):
    """Type of project for single project mode."""
    MARKET_RATE = "market_rate"
    MIXED_INCOME = "mixed_income"


class TIFTreatment(Enum):
    """How TIF benefits are applied to the project."""
    NONE = "none"                           # No TIF benefits
    LUMP_SUM_CAPITAL = "lump_sum"           # Upfront capital grant, then pay standard taxes
    TAX_ABATEMENT = "abatement"             # Reduced property taxes for a period
    TIF_STREAM = "stream"                   # Reimbursement of increment from participating entities


@dataclass
class TIFConfig:
    """TIF-specific configuration."""
    treatment: TIFTreatment = TIFTreatment.NONE

    # Lump sum settings
    lump_sum_amount: float = 0.0            # Dollar amount if lump sum
    lump_sum_pct_of_tdc: float = 0.0        # OR as % of TDC
    use_lump_sum_pct: bool = False          # Toggle between $ and %

    # Abatement settings
    abatement_pct: float = 0.0              # % of taxes abated (0-100%)
    abatement_years: int = 0                # Duration of abatement
    abatement_participating_only: bool = True  # Only participating entities

    # Stream settings (reimbursement of increment)
    stream_pct_of_increment: float = 0.0    # % of increment captured (typically 50-100%)
    stream_years: int = 0                   # Duration of stream
    stream_participating_entities: list = field(default_factory=list)  # Which entities participate

    # Common
    tif_start_delay_months: int = 0         # Delay from stabilization to TIF start


@dataclass
class ScenarioInputs:
    """Inputs specific to a single scenario (can differ between comparison scenarios).

    NOTE ON UNIT COUNTS:
    The total_units field should be determined from separate site/massing analysis.
    Unit count depends on many factors including:
    - Site area (less setbacks, easements)
    - Zoning (base FAR, height limits)
    - Any bonuses (FAR bonus, height bonus) and how that area is allocated
    - Building efficiency (gross to net)
    - Parking requirements and how they're met (surface vs structured)

    This model is for financial analysis, not site planning. Users should
    input unit counts that have been validated through proper test-fit studies.
    """
    name: str = "Scenario"

    # Unit configuration
    # NOTE: This should come from site/massing analysis, not calculated here
    total_units: int = 200

    # Affordable configuration
    affordable_pct: float = 0.0             # % of units that are affordable
    ami_level: str = "50%"                  # AMI level for affordable units

    # TIF configuration
    tif_config: TIFConfig = field(default_factory=TIFConfig)

    # Fee waivers / incentives (non-TIF)
    smart_fee_waiver: bool = False
    smart_fee_waiver_amount: float = 0.0    # Calculated based on units/affordable %

    # Other scenario-specific overrides (if needed)
    hard_cost_adjustment_pct: float = 0.0   # Adjust hard costs (e.g., for different construction)

    @property
    def affordable_units(self) -> int:
        """Number of affordable units."""
        return int(self.total_units * self.affordable_pct)

    @property
    def market_units(self) -> int:
        """Number of market rate units."""
        return self.total_units - self.affordable_units


@dataclass
class ModelConfig:
    """Complete model configuration."""
    mode: ModelMode = ModelMode.COMPARISON

    # For single project mode
    single_project_type: ProjectType = ProjectType.MIXED_INCOME
    single_scenario: ScenarioInputs = field(default_factory=ScenarioInputs)

    # For comparison mode
    scenario_a: ScenarioInputs = field(default_factory=lambda: ScenarioInputs(
        name="Market Rate",
        affordable_pct=0.0,
        tif_config=TIFConfig(treatment=TIFTreatment.NONE),
    ))
    scenario_b: ScenarioInputs = field(default_factory=lambda: ScenarioInputs(
        name="Mixed Income",
        affordable_pct=0.20,
        tif_config=TIFConfig(treatment=TIFTreatment.TIF_STREAM),
    ))

    def get_active_scenarios(self) -> list:
        """Get list of active scenarios based on mode."""
        if self.mode == ModelMode.SINGLE_PROJECT:
            return [self.single_scenario]
        else:
            return [self.scenario_a, self.scenario_b]


@dataclass
class SharedInputs:
    """Inputs shared across all scenarios."""
    # Timing
    predevelopment_months: int = 18
    construction_months: int = 24
    leaseup_months: int = 12
    operations_months: int = 60

    # Land
    land_cost_per_acre: float = 1_000_000
    units_per_acre: float = 66              # Density

    # Construction costs (per unit basis)
    hard_cost_per_unit: float = 155_000
    hard_cost_contingency_pct: float = 0.05
    soft_cost_pct: float = 0.30             # % of hard costs
    soft_cost_contingency_pct: float = 0.05
    developer_fee_pct: float = 0.04         # % of hard costs

    # Revenue
    market_rent_psf: float = 2.50
    vacancy_rate: float = 0.06
    leaseup_pace: float = 0.08              # Monthly absorption
    max_occupancy: float = 0.94

    # Operating expenses (per unit annual)
    opex_utilities: float = 1_200
    opex_maintenance: float = 1_500
    opex_management_pct: float = 0.05       # % of EGI
    opex_misc: float = 650

    # Escalations
    market_rent_growth: float = 0.02
    affordable_rent_growth: float = 0.01
    opex_growth: float = 0.03
    property_tax_growth: float = 0.01

    # Financing - Construction
    construction_rate: float = 0.075
    construction_ltc: float = 0.65
    construction_loan_fee_pct: float = 0.01

    # Financing - Permanent
    perm_rate: float = 0.06
    perm_amort_years: int = 25
    perm_ltv_max: float = 0.65
    perm_dscr_min: float = 1.25

    # Financing - Mezzanine (optional)
    mezzanine_enabled: bool = False
    mezzanine_amount: float = 0.0
    mezzanine_rate: float = 0.12

    # Financing - Preferred (optional)
    preferred_enabled: bool = False
    preferred_amount: float = 0.0
    preferred_return: float = 0.10

    # Exit
    exit_cap_rate: float = 0.055

    # Property tax
    existing_assessed_value: float = 5_000_000


def create_default_market_scenario(total_units: int = 200) -> ScenarioInputs:
    """Create default market rate scenario.

    Args:
        total_units: Number of units (should be from site/massing analysis)
    """
    return ScenarioInputs(
        name="Market Rate",
        total_units=total_units,
        affordable_pct=0.0,
        tif_config=TIFConfig(treatment=TIFTreatment.NONE),
        smart_fee_waiver=False,
    )


def create_default_mixed_income_scenario(
    total_units: int = 200,
    affordable_pct: float = 0.20,
    ami_level: str = "50%",
    tif_treatment: TIFTreatment = TIFTreatment.TIF_STREAM,
) -> ScenarioInputs:
    """Create default mixed income scenario.

    Args:
        total_units: Number of units (should be from site/massing analysis)
        affordable_pct: Percentage of units that are affordable
        ami_level: AMI level for affordable units
        tif_treatment: How TIF benefits are applied
    """
    return ScenarioInputs(
        name="Mixed Income",
        total_units=total_units,
        affordable_pct=affordable_pct,
        ami_level=ami_level,
        tif_config=TIFConfig(
            treatment=tif_treatment,
            stream_pct_of_increment=1.0,  # 100% of increment
            stream_years=20,
        ),
        smart_fee_waiver=True,
    )
