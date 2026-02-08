"""TIF Lump Sum Calculator.

Calculates the TIF lump sum based on:
- City's tax increment (participating entities only)
- Increment over base assessed value (not total value)
- Capitalized at the tier's cap rate
"""

from dataclasses import dataclass
from typing import Optional

from src.calculations.property_tax import TaxingAuthorityStack, get_austin_tax_stack
from src.models.incentives import IncentiveTier, TIF_PARAMS, TIFParams


@dataclass
class TIFCalculationResult:
    """Result of TIF lump sum calculation."""

    # Inputs
    new_assessed_value: float
    base_assessed_value: float
    incremental_value: float
    tier: IncentiveTier
    tif_params: TIFParams

    # Tax rates
    city_rate_decimal: float
    total_rate_decimal: float

    # Annual values
    annual_city_increment: float  # City's portion of increment taxes
    annual_total_increment: float  # All entities' portion

    # Capitalized values
    tif_lump_sum: float  # Capitalized city increment

    # Per-unit metrics
    affordable_units: int
    per_affordable_unit: float

    # Buildup schedule (year -> cumulative value)
    buildup_schedule_nominal: dict[int, float]
    buildup_schedule_real: dict[int, float]

    def summary(self) -> str:
        """Return a text summary of the calculation."""
        lines = [
            "TIF LUMP SUM CALCULATION",
            "=" * 50,
            f"Tier: {self.tier.name}",
            f"Cap Rate: {self.tif_params.cap_rate:.2%}",
            f"Term: {self.tif_params.term_years} years",
            f"Escalation Rate: {self.tif_params.rate:.2%}",
            "",
            "ASSESSED VALUES",
            f"  New Value (TDC):    ${self.new_assessed_value:>15,.0f}",
            f"  Base Value:         ${self.base_assessed_value:>15,.0f}",
            f"  Increment:          ${self.incremental_value:>15,.0f}",
            "",
            "TAX RATES",
            f"  City (TIF) Rate:    {self.city_rate_decimal:>15.4%}",
            f"  Total Rate:         {self.total_rate_decimal:>15.4%}",
            "",
            "ANNUAL INCREMENT",
            f"  City Increment:     ${self.annual_city_increment:>15,.0f}",
            f"  Total Increment:    ${self.annual_total_increment:>15,.0f}",
            "",
            "TIF LUMP SUM",
            f"  Capitalized Value:  ${self.tif_lump_sum:>15,.0f}",
            f"  Per Affordable Unit:${self.per_affordable_unit:>15,.0f}",
        ]
        return "\n".join(lines)


def calculate_tif_lump_sum_from_tier(
    new_assessed_value: float,
    base_assessed_value: float,
    tier: IncentiveTier,
    affordable_units: int,
    tax_stack: Optional[TaxingAuthorityStack] = None,
    override_cap_rate: Optional[float] = None,
    override_term_years: Optional[int] = None,
    override_escalation_rate: Optional[float] = None,
) -> TIFCalculationResult:
    """Calculate TIF lump sum using tier parameters.

    The TIF lump sum is calculated as:
        TIF Lump Sum = Annual City Increment / Cap Rate

    Where:
        - Annual City Increment = Incremental Value * City Tax Rate
        - Incremental Value = New Assessed Value - Base Assessed Value
        - Cap Rate is from the tier's TIF parameters

    Args:
        new_assessed_value: New assessed value (typically TDC)
        base_assessed_value: Existing assessed value before development
        tier: Incentive tier for TIF parameters
        affordable_units: Number of affordable units (for per-unit calc)
        tax_stack: Tax stack to use (defaults to Austin)
        override_cap_rate: Override the tier's cap rate
        override_term_years: Override the tier's term
        override_escalation_rate: Override the tier's escalation rate

    Returns:
        TIFCalculationResult with all calculation details
    """
    if tax_stack is None:
        tax_stack = get_austin_tax_stack()

    # Get tier parameters
    tif_params = TIF_PARAMS[tier]

    # Apply overrides
    cap_rate = override_cap_rate if override_cap_rate is not None else tif_params.cap_rate
    term_years = override_term_years if override_term_years is not None else tif_params.term_years
    escalation_rate = override_escalation_rate if override_escalation_rate is not None else tif_params.rate

    # Create effective params
    effective_params = TIFParams(
        term_years=term_years,
        rate=escalation_rate,
        cap_rate=cap_rate,
    )

    # Calculate increment
    incremental_value = max(0, new_assessed_value - base_assessed_value)

    # Get tax rates
    city_rate = tax_stack.tif_participating_rate_decimal
    total_rate = tax_stack.total_rate_decimal

    # Calculate annual increments
    annual_city_increment = incremental_value * city_rate
    annual_total_increment = incremental_value * total_rate

    # Calculate TIF lump sum (capitalized city increment)
    tif_lump_sum = annual_city_increment / cap_rate if cap_rate > 0 else 0

    # Per affordable unit
    per_affordable_unit = tif_lump_sum / affordable_units if affordable_units > 0 else 0

    # Build up schedule - showing cumulative value over time
    buildup_nominal: dict[int, float] = {}
    buildup_real: dict[int, float] = {}

    cumulative_nominal = 0.0
    cumulative_real = 0.0

    for year in range(1, term_years + 1):
        # Nominal: increment grows at escalation rate
        year_increment = annual_city_increment * ((1 + escalation_rate) ** (year - 1))
        cumulative_nominal += year_increment
        buildup_nominal[year] = cumulative_nominal

        # Real: discount back to present value
        discounted_increment = year_increment / ((1 + escalation_rate) ** (year - 1))
        cumulative_real += discounted_increment
        buildup_real[year] = cumulative_real

    return TIFCalculationResult(
        new_assessed_value=new_assessed_value,
        base_assessed_value=base_assessed_value,
        incremental_value=incremental_value,
        tier=tier,
        tif_params=effective_params,
        city_rate_decimal=city_rate,
        total_rate_decimal=total_rate,
        annual_city_increment=annual_city_increment,
        annual_total_increment=annual_total_increment,
        tif_lump_sum=tif_lump_sum,
        affordable_units=affordable_units,
        per_affordable_unit=per_affordable_unit,
        buildup_schedule_nominal=buildup_nominal,
        buildup_schedule_real=buildup_real,
    )


def get_tier_tif_defaults(tier: IncentiveTier) -> dict:
    """Get default TIF parameters for a tier.

    Returns:
        Dict with cap_rate, term_years, escalation_rate
    """
    params = TIF_PARAMS[tier]
    return {
        "cap_rate": params.cap_rate,
        "term_years": params.term_years,
        "escalation_rate": params.rate,
    }
