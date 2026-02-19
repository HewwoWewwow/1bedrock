"""Calculation modules for the Austin TIF model."""

from .land import calculate_land, LandResult
from .units import allocate_units, UnitAllocation
from .revenue import calculate_gpr, GPRResult
from .costs import calculate_tdc, TDCResult, calculate_idc
from .taxes import calculate_property_taxes, PropertyTaxResult
from .debt import size_construction_loan, size_permanent_loan, ConstructionLoan, PermanentLoan
from .tif import calculate_tif_value, TIFResult
from .dcf import run_dcf, DCFResult, MonthlyCashFlow
from .metrics import (
    calculate_metrics,
    calculate_metrics_from_detailed,  # For unified engine
    ScenarioMetrics,
    compare_scenarios,
    ScenarioComparison,
)

# Detailed cashflow module (unified calculation engine)
from .detailed_cashflow import (
    AssessedValueBasis,
    generate_detailed_cash_flow,
    calculate_deal,  # Unified entry point - SINGLE SOURCE OF TRUTH
    DetailedCashFlowResult,
    DetailedPeriodCashFlow,
)

# Monte Carlo simulation module
from .monte_carlo import (
    DistributionType,
    InputDistribution,
    BaseInputs,
    MonteCarloConfig,
    IterationResult,
    SensitivityResult,
    MonteCarloResult,
    run_monte_carlo,
    create_cost_uncertainty_distributions,
    create_market_uncertainty_distributions,
    create_financing_uncertainty_distributions,
    create_timing_uncertainty_distributions,
    create_full_uncertainty_suite,
    # TIF Grid Search
    TIFGridPoint,
    TIFGridSearchResult,
    run_tif_grid_search,
    create_tif_distributions,
)

# TIF Calculator
from .tif_calculator import (
    TIFCalculationResult,
    calculate_tif_lump_sum_from_tier,
    get_tier_tif_defaults,
)

# Property tax module with detailed taxing authority breakdown
from .property_tax import (
    TaxingAuthority,
    TaxingAuthorityStack,
    get_austin_tax_stack,
    build_tax_stack_from_rates,
    PropertyTaxPeriod,
    PropertyTaxSchedule,
    generate_property_tax_schedule,
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
    "build_tax_stack_from_rates",
    "PropertyTaxPeriod",
    "PropertyTaxSchedule",
    "generate_property_tax_schedule",
    "TIFLoanPayment",
    "TIFLoanSchedule",
    "calculate_tif_loan_schedule",
    "solve_tif_term",
    "SMARTFeeWaiver",
    "SMART_TIER_CONFIG",
    "calculate_smart_fee_waiver",
    "CityTaxAbatement",
    "calculate_city_tax_abatement",
    # Detailed cashflow
    "AssessedValueBasis",
    "generate_detailed_cash_flow",
    "DetailedCashFlowResult",
    # Monte Carlo simulation
    "DistributionType",
    "InputDistribution",
    "BaseInputs",
    "MonteCarloConfig",
    "IterationResult",
    "SensitivityResult",
    "MonteCarloResult",
    "run_monte_carlo",
    "create_cost_uncertainty_distributions",
    "create_market_uncertainty_distributions",
    "create_financing_uncertainty_distributions",
    "create_timing_uncertainty_distributions",
    "create_full_uncertainty_suite",
    # TIF Grid Search
    "TIFGridPoint",
    "TIFGridSearchResult",
    "run_tif_grid_search",
    "create_tif_distributions",
    # TIF Calculator
    "TIFCalculationResult",
    "calculate_tif_lump_sum_from_tier",
    "get_tier_tif_defaults",
]
