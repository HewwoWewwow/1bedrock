"""Incentive tier configurations and structures."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


class IncentiveTier(Enum):
    """Incentive tiers with increasing affordability requirements."""

    TIER_1 = 1
    TIER_2 = 2
    TIER_3 = 3


@dataclass(frozen=True)
class SmartWaiver:
    """SMART fee waiver configuration by tier."""

    waiver_pct: float  # Percentage of fees waived (0-1)
    amount_per_unit: float  # Dollar amount per affordable unit


@dataclass(frozen=True)
class AbatementTerms:
    """Tax abatement terms by tier."""

    years: int  # Duration of abatement
    pct: float  # Percentage of taxes abated (0-1)
    applicability: str  # "affordable_units" or "all_units"


@dataclass(frozen=True)
class TIFParams:
    """TIF (Tax Increment Financing) parameters by tier."""

    term_years: int  # Duration of TIF
    rate: float  # Discount/escalation rate
    cap_rate: float  # Cap rate for TIF valuation


@dataclass
class IncentiveToggles:
    """Boolean toggles for which incentives are enabled."""

    smart_fee_waiver: bool = False
    tax_abatement: bool = False
    tif_lump_sum: bool = False
    tif_stream: bool = False
    interest_buydown: bool = False
    height_bonus: bool = False

    def __post_init__(self) -> None:
        """Enforce mutual exclusivity rules.

        TIF stream and TIF lump sum are mutually exclusive.
        If both enabled, lump sum takes precedence.

        Tax abatement and TIF are mutually exclusive.
        If both enabled, TIF takes precedence.
        """
        # TIF lump sum and stream are mutually exclusive
        if self.tif_lump_sum and self.tif_stream:
            self.tif_stream = False

        # Tax abatement and TIF are mutually exclusive - TIF takes precedence
        if (self.tif_lump_sum or self.tif_stream) and self.tax_abatement:
            self.tax_abatement = False


@dataclass
class IncentiveConfig:
    """Complete incentive configuration for a scenario."""

    tier: IncentiveTier
    affordable_pct: float  # Required affordable percentage (0-1)
    ami_level: str  # AMI level for affordable units (e.g., "50%")
    affordability_term_years: int  # Required compliance period
    toggles: IncentiveToggles = field(default_factory=IncentiveToggles)

    # Tier-specific parameters (populated from TIER_CONFIGS)
    smart_waiver: SmartWaiver | None = None
    abatement_terms: AbatementTerms | None = None
    tif_params: TIFParams | None = None
    buydown_bps: int = 0  # Interest rate buydown in basis points

    def get_waiver_amount(self, affordable_units: int) -> float:
        """Calculate total fee waiver amount.

        Args:
            affordable_units: Number of affordable units.

        Returns:
            Total waiver amount in dollars.
        """
        if not self.toggles.smart_fee_waiver or self.smart_waiver is None:
            return 0.0
        return affordable_units * self.smart_waiver.amount_per_unit

    def get_effective_buydown_bps(self) -> int:
        """Get effective interest rate buydown.

        Returns:
            Buydown in basis points, or 0 if not enabled.
        """
        if not self.toggles.interest_buydown:
            return 0
        return self.buydown_bps


# SMART fee waivers by tier (from spec)
SMART_WAIVERS: Dict[IncentiveTier, SmartWaiver] = {
    IncentiveTier.TIER_1: SmartWaiver(waiver_pct=1.00, amount_per_unit=35000),
    IncentiveTier.TIER_2: SmartWaiver(waiver_pct=0.50, amount_per_unit=17500),
    IncentiveTier.TIER_3: SmartWaiver(waiver_pct=0.25, amount_per_unit=7500),
}

# Tax abatement terms by tier (from spec)
ABATEMENT_TERMS: Dict[IncentiveTier, AbatementTerms] = {
    IncentiveTier.TIER_1: AbatementTerms(years=12, pct=1.00, applicability="affordable_units"),
    IncentiveTier.TIER_2: AbatementTerms(years=15, pct=1.00, applicability="affordable_units"),
    IncentiveTier.TIER_3: AbatementTerms(years=20, pct=1.00, applicability="affordable_units"),
}

# TIF parameters by tier (from spec)
# Cap rate of 9.5% used for capitalizing TIF lump sum value
TIF_PARAMS: Dict[IncentiveTier, TIFParams] = {
    IncentiveTier.TIER_1: TIFParams(term_years=20, rate=0.00, cap_rate=0.095),
    IncentiveTier.TIER_2: TIFParams(term_years=15, rate=0.02, cap_rate=0.095),
    IncentiveTier.TIER_3: TIFParams(term_years=10, rate=0.04, cap_rate=0.095),
}

# Interest rate buydown by tier (from spec)
BUYDOWN_BPS: Dict[IncentiveTier, int] = {
    IncentiveTier.TIER_1: 300,
    IncentiveTier.TIER_2: 150,
    IncentiveTier.TIER_3: 75,
}

# Tier requirements (from spec)
TIER_REQUIREMENTS: Dict[IncentiveTier, Dict[str, float | str | int]] = {
    IncentiveTier.TIER_1: {
        "affordable_pct": 0.05,
        "ami_level": "30%",
        "term_years": 10,
    },
    IncentiveTier.TIER_2: {
        "affordable_pct": 0.20,
        "ami_level": "50%",
        "term_years": 20,
    },
    IncentiveTier.TIER_3: {
        "affordable_pct": 0.10,
        "ami_level": "50%",
        "term_years": 20,
    },
}


def get_tier_config(tier: IncentiveTier, toggles: IncentiveToggles | None = None) -> IncentiveConfig:
    """Build an IncentiveConfig from tier requirements and lookups.

    Args:
        tier: The incentive tier.
        toggles: Which incentives are enabled. Defaults to all disabled.

    Returns:
        Complete IncentiveConfig with tier-specific parameters.
    """
    reqs = TIER_REQUIREMENTS[tier]
    return IncentiveConfig(
        tier=tier,
        affordable_pct=float(reqs["affordable_pct"]),
        ami_level=str(reqs["ami_level"]),
        affordability_term_years=int(reqs["term_years"]),
        toggles=toggles or IncentiveToggles(),
        smart_waiver=SMART_WAIVERS[tier],
        abatement_terms=ABATEMENT_TERMS[tier],
        tif_params=TIF_PARAMS[tier],
        buydown_bps=BUYDOWN_BPS[tier],
    )


# Pre-built tier configs with all incentives disabled
TIER_CONFIGS: Dict[IncentiveTier, IncentiveConfig] = {
    tier: get_tier_config(tier) for tier in IncentiveTier
}
