"""Land calculations for determining acres needed and land cost."""

from dataclasses import dataclass

from ..models.lookups import ConstructionType, CONSTRUCTION_TYPE_PARAMS


@dataclass
class LandResult:
    """Results of land calculations."""

    acres_needed: float
    land_cost: float
    units_per_acre: int


def calculate_land(
    target_units: int,
    construction_type: ConstructionType,
    land_cost_per_acre: float,
) -> LandResult:
    """Calculate land requirements and cost.

    The acres needed is determined by the construction type's density
    (units per acre). Land cost is simply acres * cost per acre.

    Args:
        target_units: Number of units to build.
        construction_type: Type of construction (determines density).
        land_cost_per_acre: Cost per acre in dollars.

    Returns:
        LandResult with acres needed, land cost, and units per acre.

    Example:
        >>> result = calculate_land(200, ConstructionType.PODIUM_5OVER1, 1_000_000)
        >>> result.acres_needed
        3.08  # 200 units / 65 units per acre
        >>> result.land_cost
        3076923.08
    """
    params = CONSTRUCTION_TYPE_PARAMS[construction_type]
    units_per_acre = params.units_per_acre

    acres_needed = target_units / units_per_acre
    land_cost = acres_needed * land_cost_per_acre

    return LandResult(
        acres_needed=acres_needed,
        land_cost=land_cost,
        units_per_acre=units_per_acre,
    )
