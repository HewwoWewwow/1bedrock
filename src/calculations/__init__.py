"""Calculation modules for the Austin TIF model."""

from .land import calculate_land, LandResult
from .units import allocate_units, UnitAllocation
from .revenue import calculate_gpr, GPRResult
from .costs import calculate_tdc, TDCResult, calculate_idc
from .taxes import calculate_property_taxes, PropertyTaxResult
from .debt import size_construction_loan, size_permanent_loan, ConstructionLoan, PermanentLoan
from .tif import calculate_tif_value, TIFResult
from .dcf import run_dcf, DCFResult, MonthlyCashFlow
from .metrics import calculate_metrics, ScenarioMetrics, compare_scenarios, ScenarioComparison

# Property tax module with detailed taxing authority breakdown
from .property_tax import (
    TaxingAuthority,
    TaxingAuthorityStack,
    get_austin_tax_stack,
    PropertyTaxPeriod,
    PropertyTaxSchedule,
    generate_property_tax_schedule,
    analyze_tif,
    TIFAnalysisResult,
    # TIF Lump Sum
    TIFLumpSumResult,
    calculate_tif_lump_sum,
    # TIF as Loan
    TIFLoanPayment,
    TIFLoanSchedule,
    calculate_tif_loan_schedule,
    solve_tif_term,
    # SMART Fee Waivers
    SMARTFeeWaiver,
    SMART_TIER_CONFIG,
    calculate_smart_fee_waiver,
    # City Tax Abatement
    CityTaxAbatement,
    calculate_city_tax_abatement,
)

__all__ = [
    "calculate_land",
    "LandResult",
    "allocate_units",
    "UnitAllocation",
    "calculate_gpr",
    "GPRResult",
    "calculate_tdc",
    "TDCResult",
    "calculate_idc",
    "calculate_property_taxes",
    "PropertyTaxResult",
    "size_construction_loan",
    "size_permanent_loan",
    "ConstructionLoan",
    "PermanentLoan",
    "calculate_tif_value",
    "TIFResult",
    "run_dcf",
    "DCFResult",
    "MonthlyCashFlow",
    "calculate_metrics",
    "ScenarioMetrics",
    "compare_scenarios",
    "ScenarioComparison",
    # Property tax detailed
    "TaxingAuthority",
    "TaxingAuthorityStack",
    "get_austin_tax_stack",
    "PropertyTaxPeriod",
    "PropertyTaxSchedule",
    "generate_property_tax_schedule",
    "analyze_tif",
    "TIFAnalysisResult",
    "TIFLumpSumResult",
    "calculate_tif_lump_sum",
    "TIFLoanPayment",
    "TIFLoanSchedule",
    "calculate_tif_loan_schedule",
    "solve_tif_term",
    "SMARTFeeWaiver",
    "SMART_TIER_CONFIG",
    "calculate_smart_fee_waiver",
    "CityTaxAbatement",
    "calculate_city_tax_abatement",
]
