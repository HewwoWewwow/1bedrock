"""Property tax calculations with abatement support."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class PropertyTaxResult:
    """Results of property tax calculation."""

    assessed_value: float
    existing_assessed_value: float
    increment_value: float  # New value above existing base

    # Taxes by jurisdiction
    taxes_by_jurisdiction: Dict[str, float]
    existing_taxes_by_jurisdiction: Dict[str, float]
    increment_by_jurisdiction: Dict[str, float]

    # Totals
    total_tax_annual: float
    total_existing_tax_annual: float
    total_increment_annual: float
    city_increment_annual: float  # For TIF calculations

    # Abatement (if applicable)
    abatement_amount_annual: float
    net_tax_annual: float


def calculate_property_taxes(
    assessed_value: float,
    existing_assessed_value: float,
    tax_rates: Dict[str, float],
    abatement_pct: float = 0.0,
    abatement_base: str = "increment",  # "increment" or "total"
) -> PropertyTaxResult:
    """Calculate property taxes by jurisdiction with optional abatement.

    Property taxes are calculated per $100 of assessed value for each
    taxing jurisdiction. The increment (new value above existing base)
    is important for TIF calculations.

    Args:
        assessed_value: New assessed value after development.
        existing_assessed_value: Pre-development assessed value.
        tax_rates: Tax rates by jurisdiction (per $100 of assessed value).
        abatement_pct: Percentage of taxes abated (0-1).
        abatement_base: Whether abatement applies to "increment" or "total".

    Returns:
        PropertyTaxResult with taxes by jurisdiction and totals.

    Example:
        >>> result = calculate_property_taxes(
        ...     assessed_value=80_000_000,
        ...     existing_assessed_value=5_000_000,
        ...     tax_rates={"city": 0.574, "county": 0.3758, ...},
        ...     abatement_pct=0.20,
        ... )
        >>> result.total_tax_annual
        1656480.0  # Total before abatement
    """
    increment_value = assessed_value - existing_assessed_value

    # Calculate taxes by jurisdiction
    taxes_by_jurisdiction: Dict[str, float] = {}
    existing_taxes_by_jurisdiction: Dict[str, float] = {}
    increment_by_jurisdiction: Dict[str, float] = {}

    for jurisdiction, rate in tax_rates.items():
        # Rate is per $100 of assessed value
        taxes_by_jurisdiction[jurisdiction] = (assessed_value / 100) * rate
        existing_taxes_by_jurisdiction[jurisdiction] = (existing_assessed_value / 100) * rate
        increment_by_jurisdiction[jurisdiction] = (increment_value / 100) * rate

    # Calculate totals
    total_tax = sum(taxes_by_jurisdiction.values())
    total_existing = sum(existing_taxes_by_jurisdiction.values())
    total_increment = sum(increment_by_jurisdiction.values())

    # City increment for TIF
    city_increment = increment_by_jurisdiction.get("city", 0.0)

    # Calculate abatement
    if abatement_pct > 0:
        if abatement_base == "increment":
            abatement_amount = total_increment * abatement_pct
        else:  # "total"
            abatement_amount = total_tax * abatement_pct
    else:
        abatement_amount = 0.0

    net_tax = total_tax - abatement_amount

    return PropertyTaxResult(
        assessed_value=assessed_value,
        existing_assessed_value=existing_assessed_value,
        increment_value=increment_value,
        taxes_by_jurisdiction=taxes_by_jurisdiction,
        existing_taxes_by_jurisdiction=existing_taxes_by_jurisdiction,
        increment_by_jurisdiction=increment_by_jurisdiction,
        total_tax_annual=total_tax,
        total_existing_tax_annual=total_existing,
        total_increment_annual=total_increment,
        city_increment_annual=city_increment,
        abatement_amount_annual=abatement_amount,
        net_tax_annual=net_tax,
    )


def calculate_assessed_value(
    noi_annual: float,
    cap_rate: float,
) -> float:
    """Calculate assessed value from NOI using income approach.

    Assessed Value = NOI / Cap Rate

    Args:
        noi_annual: Annual Net Operating Income.
        cap_rate: Capitalization rate.

    Returns:
        Assessed value.
    """
    if cap_rate <= 0:
        raise ValueError("Cap rate must be positive")
    return noi_annual / cap_rate


def escalate_taxes(
    base_tax: float,
    years_elapsed: int,
    growth_rate: float,
) -> float:
    """Apply annual property tax escalation.

    Args:
        base_tax: Starting tax amount.
        years_elapsed: Number of years since base period.
        growth_rate: Annual growth rate (e.g., 0.02 for 2%).

    Returns:
        Escalated tax amount.
    """
    return base_tax * (1 + growth_rate) ** years_elapsed
