"""Monthly DCF (Discounted Cash Flow) engine."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List

import numpy_financial as npf

from ..models.project import ProjectInputs, Scenario, ReversionNOIBasis, TIFStartTiming
from ..models.incentives import IncentiveConfig, IncentiveToggles, get_tier_config, IncentiveTier
from .land import calculate_land
from .units import allocate_units, get_total_units
from .revenue import calculate_gpr
from .costs import calculate_tdc, calculate_opex_monthly
from .taxes import calculate_property_taxes, calculate_assessed_value
from .debt import size_construction_loan, size_permanent_loan, calculate_loan_balance
from .tif import calculate_tif_value, get_tif_payment_for_month


class Phase(Enum):
    """Development phases."""

    PREDEVELOPMENT = "predevelopment"
    CONSTRUCTION = "construction"
    LEASEUP = "leaseup"
    OPERATIONS = "operations"
    REVERSION = "reversion"


@dataclass
class MonthlyCashFlow:
    """Cash flow for a single month."""

    month: int
    phase: Phase

    # Revenue
    gross_potential_rent: float = 0.0
    vacancy_loss: float = 0.0
    effective_gross_income: float = 0.0

    # Operating expenses
    operating_expenses: float = 0.0
    property_taxes: float = 0.0
    reserves: float = 0.0

    # NOI
    noi: float = 0.0

    # Incentives
    tif_payment: float = 0.0
    abatement_credit: float = 0.0

    # Financing
    construction_draw: float = 0.0
    construction_interest: float = 0.0
    perm_debt_service: float = 0.0

    # Equity flows
    equity_contribution: float = 0.0

    # Cash flows
    unlevered_cf: float = 0.0  # NOI - CapEx
    levered_cf: float = 0.0  # After debt service

    # Sale (reversion only)
    sale_proceeds: float = 0.0
    loan_payoff: float = 0.0
    net_sale_proceeds: float = 0.0

    # Occupancy
    occupancy_rate: float = 0.0


@dataclass
class DCFResult:
    """Complete DCF analysis result."""

    scenario: Scenario
    monthly_cash_flows: List[MonthlyCashFlow] = field(default_factory=list)

    # Key metrics
    tdc: float = 0.0
    equity_required: float = 0.0
    perm_loan_amount: float = 0.0
    stabilized_noi_annual: float = 0.0
    stabilized_value: float = 0.0

    # IRR (annualized)
    unlevered_irr: float = 0.0
    levered_irr: float = 0.0

    # Other metrics
    equity_multiple: float = 0.0
    yield_on_cost: float = 0.0

    # TIF value (if applicable)
    tif_capitalized_value: float = 0.0


def get_phase(month: int, inputs: ProjectInputs) -> Phase:
    """Determine the phase for a given month.

    Args:
        month: Month number (1-indexed).
        inputs: Project inputs with timing.

    Returns:
        Phase enum value.
    """
    if month <= inputs.predevelopment_months:
        return Phase.PREDEVELOPMENT
    elif month <= inputs.predevelopment_months + inputs.construction_months:
        return Phase.CONSTRUCTION
    elif month <= inputs.predevelopment_months + inputs.construction_months + inputs.leaseup_months:
        return Phase.LEASEUP
    elif month < inputs.reversion_month:
        return Phase.OPERATIONS
    else:
        return Phase.REVERSION


def run_dcf(
    inputs: ProjectInputs,
    scenario: Scenario = Scenario.MARKET,
) -> DCFResult:
    """Run complete monthly DCF analysis.

    This is the main entry point for the DCF engine. It:
    1. Calculates all upstream values (land, units, costs, etc.)
    2. Builds monthly cash flows for all 5 phases
    3. Calculates IRR and other return metrics

    Args:
        inputs: Complete project inputs.
        scenario: MARKET or MIXED_INCOME.

    Returns:
        DCFResult with cash flows and metrics.
    """
    result = DCFResult(scenario=scenario)

    # === Setup: Calculate upstream values ===

    # Land
    land_result = calculate_land(
        target_units=inputs.target_units,
        construction_type=inputs.construction_type,
        land_cost_per_acre=inputs.land_cost_per_acre,
    )

    # Unit allocation
    affordable_pct = inputs.affordable_pct if scenario == Scenario.MIXED_INCOME else 0.0
    ami_level = inputs.ami_level if scenario == Scenario.MIXED_INCOME else "50%"

    allocations = allocate_units(
        target_units=inputs.target_units,
        unit_mix=inputs.unit_mix,
        affordable_pct=affordable_pct,
        ami_level=ami_level,
        market_rent_psf=inputs.market_rent_psf,
    )
    total_units = get_total_units(allocations)
    affordable_units = sum(a.affordable_units for a in allocations)

    # GPR
    gpr_result = calculate_gpr(allocations)

    # Fee waiver (mixed-income only)
    fee_waiver = 0.0
    incentive_config: IncentiveConfig | None = None

    if scenario == Scenario.MIXED_INCOME and inputs.incentive_config is not None:
        incentive_config = inputs.incentive_config
        if incentive_config.toggles.smart_fee_waiver:
            fee_waiver = incentive_config.get_waiver_amount(affordable_units)

    # TDC
    tdc_result = calculate_tdc(
        land_cost=land_result.land_cost,
        target_units=inputs.target_units,
        hard_cost_per_unit=inputs.hard_cost_per_unit,
        soft_cost_pct=inputs.soft_cost_pct,
        construction_ltc=inputs.construction_ltc,
        construction_rate=inputs.construction_rate,
        construction_months=inputs.construction_months,
        fee_waiver_amount=fee_waiver,
    )
    result.tdc = tdc_result.tdc

    # Construction loan
    construction_loan = size_construction_loan(
        tdc=tdc_result.tdc,
        ltc_ratio=inputs.construction_ltc,
        interest_rate=inputs.construction_rate,
        construction_months=inputs.construction_months,
    )

    # Stabilized values (for permanent loan sizing)
    # Use GPR * (1 - vacancy) - OpEx - Taxes
    stabilized_egi_annual = gpr_result.total_gpr_annual * (1 - inputs.vacancy_rate)

    # OpEx at stabilization
    opex_fixed_annual = total_units * inputs.annual_opex_per_unit
    mgmt_annual = stabilized_egi_annual * inputs.opex_management_pct
    reserves_annual = stabilized_egi_annual * inputs.reserves_pct
    stabilized_opex_annual = opex_fixed_annual + mgmt_annual + reserves_annual

    # Property taxes (need assessed value from stabilized value)
    # Circular: need NOI for value, need value for taxes
    # Use iterative approach or estimate
    estimated_noi = stabilized_egi_annual - stabilized_opex_annual
    estimated_value = estimated_noi / inputs.exit_cap_rate

    tax_result = calculate_property_taxes(
        assessed_value=estimated_value,
        existing_assessed_value=inputs.existing_assessed_value,
        tax_rates=inputs.tax_rates,
    )

    # Stabilized NOI
    stabilized_noi_annual = stabilized_egi_annual - stabilized_opex_annual - tax_result.net_tax_annual
    result.stabilized_noi_annual = stabilized_noi_annual

    # Stabilized value
    stabilized_value = stabilized_noi_annual / inputs.exit_cap_rate
    result.stabilized_value = stabilized_value

    # TIF calculation (mixed-income with TIF enabled)
    tif_result = None
    if (
        scenario == Scenario.MIXED_INCOME
        and incentive_config is not None
        and (incentive_config.toggles.tif_stream or incentive_config.toggles.tif_lump_sum)
    ):
        tif_start_month = (
            inputs.operations_start_month
            if inputs.tif_start_timing == TIFStartTiming.OPERATIONS
            else inputs.leaseup_start_month
        )
        tif_type = "lump_sum" if incentive_config.toggles.tif_lump_sum else "stream"
        tif_params = incentive_config.tif_params

        if tif_params is not None:
            tif_result = calculate_tif_value(
                city_increment_annual=tax_result.city_increment_annual,
                tif_term_years=tif_params.term_years,
                tif_rate=tif_params.rate,
                tif_cap_rate=tif_params.cap_rate,
                tif_start_month=tif_start_month,
                tif_type=tif_type,
            )
            result.tif_capitalized_value = tif_result.capitalized_value

    # Permanent loan sizing
    rate_buydown_bps = 0
    if scenario == Scenario.MIXED_INCOME and incentive_config is not None:
        rate_buydown_bps = incentive_config.get_effective_buydown_bps()

    perm_loan = size_permanent_loan(
        stabilized_value=stabilized_value,
        stabilized_noi=stabilized_noi_annual,
        ltv_max=inputs.perm_ltv_max,
        dscr_min=inputs.perm_dscr_min,
        perm_rate=inputs.perm_rate,
        amort_years=inputs.perm_amort_years,
        rate_buydown_bps=rate_buydown_bps,
    )
    result.perm_loan_amount = perm_loan.loan_amount

    # Equity required
    equity = tdc_result.tdc - construction_loan.loan_amount
    if tif_result is not None and incentive_config is not None:
        if incentive_config.toggles.tif_lump_sum:
            equity -= tif_result.capitalized_value
    result.equity_required = equity

    # Yield on cost
    result.yield_on_cost = stabilized_noi_annual / tdc_result.tdc if tdc_result.tdc > 0 else 0

    # === Build Monthly Cash Flows ===

    cash_flows: List[MonthlyCashFlow] = []
    perm_loan_balance = perm_loan.loan_amount

    for month in range(1, inputs.total_months + 1):
        phase = get_phase(month, inputs)
        cf = MonthlyCashFlow(month=month, phase=phase)

        # Years since lease-up start (for escalation)
        if month >= inputs.leaseup_start_month:
            years_operating = (month - inputs.leaseup_start_month) // 12
        else:
            years_operating = 0

        if phase == Phase.PREDEVELOPMENT:
            # No cash flows during predevelopment in levered analysis
            # All costs are funded at construction start
            cf.levered_cf = 0.0
            cf.unlevered_cf = 0.0

        elif phase == Phase.CONSTRUCTION:
            # No operating cash flows during construction
            # Equity is invested at month 1 of construction (handled in IRR calc)
            cf.levered_cf = 0.0
            cf.unlevered_cf = 0.0

        elif phase == Phase.LEASEUP:
            # Ramping occupancy
            month_in_leaseup = month - inputs.leaseup_start_month + 1
            target_occupancy = min(month_in_leaseup * inputs.leaseup_pace, inputs.max_occupancy)
            cf.occupancy_rate = target_occupancy

            # Revenue with escalation
            market_growth = (1 + inputs.market_rent_growth) ** years_operating
            affordable_growth = (1 + inputs.affordable_rent_growth) ** years_operating

            cf.gross_potential_rent = (
                gpr_result.market_gpr_monthly * market_growth
                + gpr_result.affordable_gpr_monthly * affordable_growth
            )
            cf.vacancy_loss = cf.gross_potential_rent * (1 - cf.occupancy_rate)
            cf.effective_gross_income = cf.gross_potential_rent - cf.vacancy_loss

            # OpEx with escalation
            opex_growth = (1 + inputs.opex_growth) ** years_operating
            fixed_opex_monthly = (total_units * inputs.annual_opex_per_unit / 12) * opex_growth
            mgmt_monthly = cf.effective_gross_income * inputs.opex_management_pct
            reserves_monthly = cf.effective_gross_income * inputs.reserves_pct

            cf.operating_expenses = fixed_opex_monthly + mgmt_monthly
            cf.reserves = reserves_monthly

            # Property taxes with escalation
            tax_growth = (1 + inputs.property_tax_growth) ** years_operating
            cf.property_taxes = (tax_result.net_tax_annual / 12) * tax_growth

            # NOI
            cf.noi = cf.effective_gross_income - cf.operating_expenses - cf.reserves - cf.property_taxes

            # Debt service
            cf.perm_debt_service = perm_loan.monthly_payment

            # TIF stream payment
            if tif_result is not None and incentive_config is not None:
                if incentive_config.toggles.tif_stream:
                    cf.tif_payment = get_tif_payment_for_month(
                        tif_result, month, years_operating
                    )

            cf.unlevered_cf = cf.noi
            cf.levered_cf = cf.noi + cf.tif_payment - cf.perm_debt_service

        elif phase == Phase.OPERATIONS:
            # Stabilized operations
            cf.occupancy_rate = inputs.max_occupancy

            # Revenue with escalation
            market_growth = (1 + inputs.market_rent_growth) ** years_operating
            affordable_growth = (1 + inputs.affordable_rent_growth) ** years_operating

            cf.gross_potential_rent = (
                gpr_result.market_gpr_monthly * market_growth
                + gpr_result.affordable_gpr_monthly * affordable_growth
            )
            cf.vacancy_loss = cf.gross_potential_rent * inputs.vacancy_rate
            cf.effective_gross_income = cf.gross_potential_rent - cf.vacancy_loss

            # OpEx with escalation
            opex_growth = (1 + inputs.opex_growth) ** years_operating
            fixed_opex_monthly = (total_units * inputs.annual_opex_per_unit / 12) * opex_growth
            mgmt_monthly = cf.effective_gross_income * inputs.opex_management_pct
            reserves_monthly = cf.effective_gross_income * inputs.reserves_pct

            cf.operating_expenses = fixed_opex_monthly + mgmt_monthly
            cf.reserves = reserves_monthly

            # Property taxes with escalation
            tax_growth = (1 + inputs.property_tax_growth) ** years_operating
            cf.property_taxes = (tax_result.net_tax_annual / 12) * tax_growth

            # NOI
            cf.noi = cf.effective_gross_income - cf.operating_expenses - cf.reserves - cf.property_taxes

            # Debt service
            cf.perm_debt_service = perm_loan.monthly_payment

            # TIF stream payment
            if tif_result is not None and incentive_config is not None:
                if incentive_config.toggles.tif_stream:
                    cf.tif_payment = get_tif_payment_for_month(
                        tif_result, month, years_operating
                    )

            cf.unlevered_cf = cf.noi
            cf.levered_cf = cf.noi + cf.tif_payment - cf.perm_debt_service

        elif phase == Phase.REVERSION:
            # Sale event
            # Calculate reversion NOI - use trailing 12 months of actual NOI
            ops_cfs = [c for c in cash_flows if c.phase in [Phase.OPERATIONS, Phase.LEASEUP]]
            if ops_cfs:
                trailing_12_noi = sum(c.noi for c in ops_cfs[-12:])
                # Annualize if less than 12 months
                if len(ops_cfs[-12:]) < 12:
                    trailing_12_noi = trailing_12_noi * 12 / len(ops_cfs[-12:])
            else:
                trailing_12_noi = stabilized_noi_annual

            if inputs.reversion_noi_basis == ReversionNOIBasis.FORWARD:
                # Use forward projection with growth
                reversion_noi = trailing_12_noi * (1 + inputs.market_rent_growth)
            else:
                reversion_noi = trailing_12_noi

            # Sale price
            cf.sale_proceeds = reversion_noi / inputs.exit_cap_rate

            # Loan payoff (remaining balance)
            months_with_perm = inputs.leaseup_months + inputs.operations_months
            cf.loan_payoff = calculate_loan_balance(
                original_principal=perm_loan.loan_amount,
                monthly_rate=perm_loan.interest_rate / 12,
                monthly_payment=perm_loan.monthly_payment,
                months_elapsed=months_with_perm,
            )

            # Net proceeds
            selling_costs = cf.sale_proceeds * inputs.selling_costs_pct
            cf.net_sale_proceeds = cf.sale_proceeds - cf.loan_payoff - selling_costs

            cf.unlevered_cf = cf.sale_proceeds - selling_costs
            cf.levered_cf = cf.net_sale_proceeds

        cash_flows.append(cf)

    result.monthly_cash_flows = cash_flows

    # === Calculate IRR ===
    # Levered IRR: Equity invested at construction start, cash flows during ops, sale proceeds at end

    # Build levered cash flow array
    # Month 0: No cash flow (we'll prepend equity)
    # Month 1 (construction start): Equity investment
    # Operations/Lease-up: Operating cash flows
    # Reversion: Net sale proceeds

    levered_cfs: List[float] = []

    for cf in cash_flows:
        if cf.month == inputs.construction_start_month:
            # Equity investment at construction start
            levered_cfs.append(-result.equity_required)
        elif cf.phase in [Phase.PREDEVELOPMENT]:
            # No cash flows during predevelopment
            levered_cfs.append(0.0)
        elif cf.phase == Phase.CONSTRUCTION and cf.month != inputs.construction_start_month:
            # No cash flows during construction (except equity at start)
            levered_cfs.append(0.0)
        else:
            levered_cfs.append(cf.levered_cf)

    # Calculate IRR
    try:
        monthly_irr = npf.irr(levered_cfs)
        if monthly_irr is not None and not (monthly_irr != monthly_irr):  # Check for NaN
            result.levered_irr = (1 + monthly_irr) ** 12 - 1
        else:
            result.levered_irr = 0.0
    except Exception:
        result.levered_irr = 0.0

    # Unlevered IRR: TDC invested at construction start
    unlevered_cfs: List[float] = []

    for cf in cash_flows:
        if cf.month == inputs.construction_start_month:
            # TDC investment at construction start
            unlevered_cfs.append(-result.tdc)
        elif cf.phase in [Phase.PREDEVELOPMENT]:
            unlevered_cfs.append(0.0)
        elif cf.phase == Phase.CONSTRUCTION and cf.month != inputs.construction_start_month:
            unlevered_cfs.append(0.0)
        elif cf.phase == Phase.REVERSION:
            # Sale proceeds (gross, before debt)
            selling_costs = cf.sale_proceeds * inputs.selling_costs_pct
            unlevered_cfs.append(cf.sale_proceeds - selling_costs)
        else:
            unlevered_cfs.append(cf.noi)

    try:
        monthly_irr = npf.irr(unlevered_cfs)
        if monthly_irr is not None and not (monthly_irr != monthly_irr):
            result.unlevered_irr = (1 + monthly_irr) ** 12 - 1
        else:
            result.unlevered_irr = 0.0
    except Exception:
        result.unlevered_irr = 0.0

    # Equity multiple
    total_distributions = sum(cf.levered_cf for cf in cash_flows if cf.levered_cf > 0)
    if result.equity_required > 0:
        result.equity_multiple = total_distributions / result.equity_required
    else:
        result.equity_multiple = 0.0

    return result
