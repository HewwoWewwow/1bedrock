"""Unit allocation calculations for market and affordable units."""

from dataclasses import dataclass
from typing import Dict

from ..models.project import UnitMixEntry
from ..models.lookups import AMI_RENT_LIMITS


@dataclass
class UnitAllocation:
    """Allocation of units by type with market/affordable split."""

    unit_type: str
    gsf: int  # Gross square feet per unit
    nsf: int  # Net square feet per unit
    total_units: int
    market_units: int
    affordable_units: int
    market_rent_monthly: float  # Per unit
    affordable_rent_monthly: float  # Per unit (0 if no affordable)
    ami_tier: str = ""  # AMI tier for affordable units of this type


def allocate_units(
    target_units: int,
    unit_mix: Dict[str, UnitMixEntry],
    affordable_pct: float,
    ami_level: str,
    market_rent_psf: float,
    efficiency: float = 0.0,
) -> list[UnitAllocation]:
    """Allocate units by bedroom type for market and affordable pools.

    Affordable units are distributed proportionally across bedroom types
    based on the unit mix allocation percentages.

    Args:
        target_units: Total number of units in the project.
        unit_mix: Unit mix configuration with GSF and allocation %.
        affordable_pct: Percentage of units that are affordable (0-1).
        ami_level: AMI level string (e.g., "50%", "60%") - fallback if unit has no ami_tier.
        market_rent_psf: Monthly rent per NSF for market units - fallback if unit has no rent_psf.
        efficiency: Net-to-gross efficiency ratio (e.g., 0.85). When > 0, uses
            UnitMixEntry.get_gsf_from_efficiency() to derive GSF from NSF.

    Returns:
        List of UnitAllocation for each unit type.

    Example:
        >>> allocations = allocate_units(
        ...     target_units=200,
        ...     unit_mix={"studio": UnitMixEntry("studio", 600, 0.12), ...},
        ...     affordable_pct=0.20,
        ...     ami_level="50%",
        ...     market_rent_psf=2.50
        ... )
    """
    total_affordable = round(target_units * affordable_pct)
    allocations = []

    for unit_type, entry in unit_mix.items():
        # Calculate total units of this type
        total_of_type = round(target_units * entry.allocation)

        # Distribute affordable units proportionally
        affordable_of_type = round(total_affordable * entry.allocation)

        # Remaining are market units
        market_of_type = total_of_type - affordable_of_type

        # Get GSF (optionally derived from NSF and efficiency)
        gsf = entry.get_gsf_from_efficiency(efficiency) if efficiency > 0 else entry.gsf

        # Get NSF via the UnitMixEntry method (falls back to gsf * 0.82 if nsf not set)
        nsf = entry.get_nsf()

        # Calculate market rent:
        # 1. Use explicit market_rent_monthly if set
        # 2. Otherwise derive from rent_psf × nsf
        # 3. Fall back to project-level market_rent_psf × nsf
        if entry.market_rent_monthly > 0:
            market_rent = entry.market_rent_monthly
        else:
            rent_psf = entry.rent_psf if entry.rent_psf > 0 else market_rent_psf
            market_rent = nsf * rent_psf

        # Get AMI tier - use per-unit ami_tier if set, otherwise fallback to project-level
        unit_ami_tier = entry.ami_tier if entry.ami_tier else ami_level

        # Affordable rent from AMI table (or 0 if no affordable units)
        if affordable_of_type > 0 and unit_ami_tier in AMI_RENT_LIMITS:
            affordable_rent = float(AMI_RENT_LIMITS[unit_ami_tier].get(unit_type, 0))
        else:
            affordable_rent = 0.0

        allocations.append(
            UnitAllocation(
                unit_type=unit_type,
                gsf=gsf,
                nsf=nsf,
                total_units=total_of_type,
                market_units=market_of_type,
                affordable_units=affordable_of_type,
                market_rent_monthly=market_rent,
                affordable_rent_monthly=affordable_rent,
                ami_tier=unit_ami_tier,
            )
        )

    return allocations


def get_total_units(allocations: list[UnitAllocation]) -> int:
    """Get total unit count from allocations.

    Args:
        allocations: List of unit allocations.

    Returns:
        Total number of units.
    """
    return sum(a.total_units for a in allocations)


def get_total_affordable_units(allocations: list[UnitAllocation]) -> int:
    """Get total affordable unit count from allocations.

    Args:
        allocations: List of unit allocations.

    Returns:
        Total number of affordable units.
    """
    return sum(a.affordable_units for a in allocations)


def get_total_market_units(allocations: list[UnitAllocation]) -> int:
    """Get total market unit count from allocations.

    Args:
        allocations: List of unit allocations.

    Returns:
        Total number of market-rate units.
    """
    return sum(a.market_units for a in allocations)


def get_total_gsf(allocations: list[UnitAllocation]) -> int:
    """Get total gross square feet from allocations.

    Args:
        allocations: List of unit allocations.

    Returns:
        Total gross square feet.
    """
    return sum(a.gsf * a.total_units for a in allocations)
