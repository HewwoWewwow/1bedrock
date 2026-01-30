"""Lookup tables for construction types, AMI rents, and tax rates."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class ConstructionType(Enum):
    """Construction type determines density, efficiency, and parking requirements."""

    GARDEN = "garden"  # 2-3 stories
    SMALL_COURTYARD = "small_courtyard"  # 2-4 stories
    CORRIDOR = "corridor"  # Double-loaded corridor bar, 4-7 stories
    PODIUM_5OVER1 = "podium_midrise_5over1"  # 5-over-1 / 5-over-2
    WRAP = "wrap"  # Wrap around garage, 4-6 stories
    HIGHRISE = "highrise"  # 15+ stories


class ParkingType(Enum):
    """Parking configuration for construction types."""

    SURFACE_STRUCTURED = "surface_structured"
    PODIUM_BASED = "podium_based"
    GROUND_INTEGRATED = "ground_integrated"
    STRUCTURED = "structured"


@dataclass(frozen=True)
class ConstructionTypeParams:
    """Parameters for each construction type from spreadsheet lookup table."""

    floors: int  # Typical number of floors
    typical_units: int  # Typical total units for this type
    floorplate_gsf: int  # Gross SF per floor
    efficiency: float  # Net rentable / gross building SF
    units_per_acre: int  # Density (du/acre)
    units_per_floor: int  # Average units per floor
    parking_type: ParkingType  # Parking configuration
    parking_ratio: float = 1.0  # Spaces per unit (for cost calculations)


# Construction type lookup table from spreadsheet v10c Sheet1
CONSTRUCTION_TYPE_PARAMS: Dict[ConstructionType, ConstructionTypeParams] = {
    ConstructionType.GARDEN: ConstructionTypeParams(
        floors=2,
        typical_units=90,
        floorplate_gsf=11500,
        efficiency=0.805,
        units_per_acre=59,
        units_per_floor=14,
        parking_type=ParkingType.SURFACE_STRUCTURED,
        parking_ratio=1.5,
    ),
    ConstructionType.SMALL_COURTYARD: ConstructionTypeParams(
        floors=3,
        typical_units=90,
        floorplate_gsf=7250,
        efficiency=0.93,
        units_per_acre=56,
        units_per_floor=8,
        parking_type=ParkingType.SURFACE_STRUCTURED,
        parking_ratio=1.25,
    ),
    ConstructionType.CORRIDOR: ConstructionTypeParams(
        floors=5,
        typical_units=100,
        floorplate_gsf=16000,
        efficiency=0.86,
        units_per_acre=66,
        units_per_floor=21,
        parking_type=ParkingType.SURFACE_STRUCTURED,
        parking_ratio=1.0,
    ),
    ConstructionType.PODIUM_5OVER1: ConstructionTypeParams(
        floors=5,
        typical_units=200,
        floorplate_gsf=24000,
        efficiency=0.85,
        units_per_acre=66,
        units_per_floor=31,
        parking_type=ParkingType.PODIUM_BASED,
        parking_ratio=1.0,
    ),
    ConstructionType.WRAP: ConstructionTypeParams(
        floors=5,
        typical_units=240,
        floorplate_gsf=60000,
        efficiency=0.76,
        units_per_acre=90,
        units_per_floor=75,
        parking_type=ParkingType.GROUND_INTEGRATED,
        parking_ratio=1.0,
    ),
    ConstructionType.HIGHRISE: ConstructionTypeParams(
        floors=15,
        typical_units=600,
        floorplate_gsf=12500,
        efficiency=0.73,
        units_per_acre=275,
        units_per_floor=12,
        parking_type=ParkingType.STRUCTURED,
        parking_ratio=0.75,
    ),
}


@dataclass(frozen=True)
class UnitTypeParams:
    """Default parameters for each unit type."""

    gsf: int  # Gross square feet
    bedrooms: int


# Default unit type parameters from spec
UNIT_TYPE_DEFAULTS: Dict[str, UnitTypeParams] = {
    "studio": UnitTypeParams(gsf=600, bedrooms=0),
    "1br": UnitTypeParams(gsf=750, bedrooms=1),
    "2br": UnitTypeParams(gsf=900, bedrooms=2),
    "3br": UnitTypeParams(gsf=1150, bedrooms=3),
    "4br": UnitTypeParams(gsf=1450, bedrooms=4),
}


# AMI rent limits by AMI level and unit type (monthly)
# Source: City of Austin "Other" program rents (from spreadsheet v10c)
# TODO: Replace with API calls to HUD and City of Austin data portals
AMI_RENT_LIMITS: Dict[str, Dict[str, int]] = {
    "30%": {"studio": 703, "1br": 703, "2br": 803, "3br": 903, "4br": 1003},
    "50%": {"studio": 1171, "1br": 1171, "2br": 1338, "3br": 1506, "4br": 1672},
    "60%": {"studio": 1405, "1br": 1405, "2br": 1606, "3br": 1807, "4br": 2007},
    "80%": {"studio": 1498, "1br": 1498, "2br": 1712, "3br": 1925, "4br": 2139},
    "100%": {"studio": 1823, "1br": 1823, "2br": 2085, "3br": 2345, "4br": 2605},
}


# Default tax rates (per $100 of assessed value) from spec
DEFAULT_TAX_RATES: Dict[str, float] = {
    "city": 0.574,
    "county": 0.3758,
    "isd": 0.9252,
    "hospital": 0.098,
    "acc": 0.1034,
}


def get_total_tax_rate(tax_rates: Dict[str, float] | None = None) -> float:
    """Calculate total tax rate from all jurisdictions.

    Args:
        tax_rates: Tax rates by jurisdiction. Uses defaults if None.

    Returns:
        Total tax rate per $100 of assessed value.
    """
    rates = tax_rates or DEFAULT_TAX_RATES
    return sum(rates.values())


def get_ami_rent(ami_level: str, unit_type: str) -> int:
    """Get AMI rent limit for a given AMI level and unit type.

    Args:
        ami_level: AMI level string (e.g., "50%", "60%")
        unit_type: Unit type (e.g., "studio", "1br", "2br")

    Returns:
        Monthly rent limit in dollars.

    Raises:
        KeyError: If AMI level or unit type not found.
    """
    return AMI_RENT_LIMITS[ami_level][unit_type]
