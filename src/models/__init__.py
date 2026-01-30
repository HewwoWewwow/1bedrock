"""Data models for the Austin TIF calculation engine."""

from .lookups import (
    ConstructionType,
    ConstructionTypeParams,
    ParkingType,
    CONSTRUCTION_TYPE_PARAMS,
    UNIT_TYPE_DEFAULTS,
    AMI_RENT_LIMITS,
    DEFAULT_TAX_RATES,
)
from .incentives import (
    IncentiveTier,
    SmartWaiver,
    AbatementTerms,
    TIFParams,
    IncentiveConfig,
    TIER_CONFIGS,
)
from .project import (
    UnitMixEntry,
    ProjectInputs,
    Scenario,
)
from .scenario_config import (
    ModelMode,
    ProjectType,
    TIFTreatment,
    TIFConfig,
    ScenarioInputs,
    ModelConfig,
    SharedInputs,
)

__all__ = [
    "ConstructionType",
    "ConstructionTypeParams",
    "ParkingType",
    "CONSTRUCTION_TYPE_PARAMS",
    "UNIT_TYPE_DEFAULTS",
    "AMI_RENT_LIMITS",
    "DEFAULT_TAX_RATES",
    "IncentiveTier",
    "SmartWaiver",
    "AbatementTerms",
    "TIFParams",
    "IncentiveConfig",
    "TIER_CONFIGS",
    "UnitMixEntry",
    "ProjectInputs",
    "Scenario",
    "ModelMode",
    "ProjectType",
    "TIFTreatment",
    "TIFConfig",
    "ScenarioInputs",
    "ModelConfig",
    "SharedInputs",
]
