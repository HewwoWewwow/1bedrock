"""Monthly DCF (Discounted Cash Flow) engine.

DEPRECATED: This module contains the legacy DCF engine.
Use `calculate_deal()` from `detailed_cashflow.py` instead.
"""

import warnings
from dataclasses import dataclass, field
from typing import List

import numpy_financial as npf

from ..models.project import ProjectInputs, Scenario, ReversionNOIBasis, TIFStartTiming
from ..models.incentives import IncentiveConfig, IncentiveToggles, get_tier_config, IncentiveTier
from .land import calculate_land
from .units import allocate_units, get_total_units, get_total_affordable_units
from .revenue import calculate_gpr, calculate_egi, calculate_egi_with_leaseup, escalate_rent
from .costs import calculate_tdc, calculate_opex_monthly, escalate_opex
from .taxes import calculate_property_taxes, calculate_assessed_value, escalate_taxes
from .debt import size_construction_loan, size_permanent_loan, calculate_loan_balance
from .tif import calculate_tif_value, get_tif_payment_for_month
from .draw_schedule import generate_draw_schedule, DrawSchedule, Phase


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

    # Draw schedule (for equity timing)
    draw_schedule: DrawSchedule | None = None


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
        return Phase.LEASE_UP
    elif month < inputs.reversion_month:
        return Phase.OPERATIONS
    else:
        return Phase.REVERSION


def _calculate_operating_period(
    cf: MonthlyCashFlow,
    month: int,
    gpr_result,
    inputs: ProjectInputs,
    total_units: int,
    tax_result,
    perm_loan,
    years_operating: int,
    tif_result,
    incentive_config,
    is_leaseup: bool,
) -> None:
    """Populate revenue, opex, and NOI fields for a lease-up or operations month.

    Uses the dedicated escalation, EGI, and opex functions rather than
    inline calculations, ensuring a single source of truth for each formula.

    Args:
        cf: MonthlyCashFlow to populate (mutated in place).
        month: Current month number (1-indexed).
        gpr_result: GPR calculation result.
        inputs: Project inputs for rates and assumptions.
        total_units: Total unit count.
        tax_result: Property tax result for base taxes.
        perm_loan: Permanent loan sizing result.
        years_operating: Years since lease-up start (for escalation).
        tif_result: TIF result (or None).
        incentive_config: Incentive config (or None).
        is_leaseup: True for lease-up phase, False for stabilized operations.
    """
    # Revenue with escalation
    cf.gross_potential_rent = (
        escalate_rent(gpr_result.market_gpr_monthly, years_operating, inputs.market_rent_growth)
        + escalate_rent(gpr_result.affordable_gpr_monthly, years_operating, inputs.affordable_rent_growth)
    )

    # Occupancy and EGI
    if is_leaseup:
        month_in_leaseup = month - inputs.leaseup_start_month + 1
        egi, occupancy = calculate_egi_with_leaseup(
            cf.gross_potential_rent, month_in_leaseup,
            inputs.leaseup_months, inputs.leaseup_pace, inputs.max_occupancy,
        )
        cf.occupancy_rate = occupancy
        cf.effective_gross_income = egi
    else:
        cf.occupancy_rate = inputs.max_occupancy
        cf.effective_gross_income = calculate_egi(cf.gross_potential_rent, inputs.vacancy_rate)

    cf.vacancy_loss = cf.gross_potential_rent - cf.effective_gross_income

    # OpEx with escalation
    escalated_opex_per_unit = escalate_opex(
        inputs.annual_opex_per_unit, years_operating, inputs.opex_growth,
    )
    total_opex, _mgmt_fee, reserves = calculate_opex_monthly(
        total_units, escalated_opex_per_unit, cf.effective_gross_income,
        inputs.opex_management_pct, inputs.reserves_pct,
    )
    cf.operating_expenses = total_opex - reserves  # Fixed + management
    cf.reserves = reserves

    # Property taxes with escalation
    cf.property_taxes = escalate_taxes(
        tax_result.net_tax_annual / 12, years_operating, inputs.property_tax_growth,
    )

    # NOI
    cf.noi = cf.effective_gross_income - cf.operating_expenses - cf.reserves - cf.property_taxes

    # Debt service
    cf.perm_debt_service = perm_loan.monthly_payment

    # TIF stream payment — escalate from TIF start, not lease-up start
    if tif_result is not None and incentive_config is not None:
        if incentive_config.toggles.tif_stream:
            years_since_tif_start = max(0, (month - tif_result.start_month) // 12)
            cf.tif_payment = get_tif_payment_for_month(
                tif_result, month, years_since_tif_start,
            )

    cf.unlevered_cf = cf.noi
    cf.levered_cf = cf.noi + cf.tif_payment - cf.perm_debt_service


def _calculate_leaseup_period(
    cf: MonthlyCashFlow,
    month: int,
    gpr_result,
    inputs: ProjectInputs,
    total_units: int,
    tax_result,
    years_operating: int,
    tif_result,
    incentive_config,
) -> None:
    """Populate revenue, opex, and NOI fields for a lease-up month.

    During lease-up, the construction loan is still in place (perm hasn't closed).
    Construction loan interest continues to capitalize (not paid).
    NOI generated during lease-up offsets funding needs.
    From developer equity perspective, levered CF = 0 during lease-up.

    Args:
        cf: MonthlyCashFlow to populate (mutated in place).
        month: Current month number (1-indexed).
        gpr_result: GPR calculation result.
        inputs: Project inputs for rates and assumptions.
        total_units: Total unit count.
        tax_result: Property tax result for base taxes.
        years_operating: Years since lease-up start (for escalation).
        tif_result: TIF result (or None).
        incentive_config: Incentive config (or None).
    """
    # Revenue with escalation
    cf.gross_potential_rent = (
        escalate_rent(gpr_result.market_gpr_monthly, years_operating, inputs.market_rent_growth)
        + escalate_rent(gpr_result.affordable_gpr_monthly, years_operating, inputs.affordable_rent_growth)
    )

    # Occupancy ramp during lease-up
    month_in_leaseup = month - inputs.leaseup_start_month + 1
    egi, occupancy = calculate_egi_with_leaseup(
        cf.gross_potential_rent, month_in_leaseup,
        inputs.leaseup_months, inputs.leaseup_pace, inputs.max_occupancy,
    )
    cf.occupancy_rate = occupancy
    cf.effective_gross_income = egi
    cf.vacancy_loss = cf.gross_potential_rent - cf.effective_gross_income

    # OpEx with escalation
    escalated_opex_per_unit = escalate_opex(
        inputs.annual_opex_per_unit, years_operating, inputs.opex_growth,
    )
    total_opex, _mgmt_fee, reserves = calculate_opex_monthly(
        total_units, escalated_opex_per_unit, cf.effective_gross_income,
        inputs.opex_management_pct, inputs.reserves_pct,
    )
    cf.operating_expenses = total_opex - reserves
    cf.reserves = reserves

    # Property taxes with escalation
    cf.property_taxes = escalate_taxes(
        tax_result.net_tax_annual / 12, years_operating, inputs.property_tax_growth,
    )

    # NOI (may be negative during early lease-up)
    cf.noi = cf.effective_gross_income - cf.operating_expenses - cf.reserves - cf.property_taxes

    # During lease-up, construction loan is still in place
    # Interest accrues and capitalizes (no cash payment required)
    # NOI generated offsets construction loan draws (already in IDC calc)
    # NOI deficits add to construction loan draws (already in IDC calc)
    # From equity IRR perspective, there's no cash flow during lease-up
    cf.perm_debt_service = 0.0

    cf.unlevered_cf = cf.noi
    cf.levered_cf = 0.0  # No equity cash flow during lease-up (construction loan covers)


def run_dcf(
    inputs: ProjectInputs,
    scenario: Scenario = Scenario.MARKET,
) -> DCFResult:
    """Run complete monthly DCF analysis.

    .. deprecated::
        This function is DEPRECATED. Use `calculate_deal()` from
        `detailed_cashflow.py` instead. `calculate_deal()` is the
        single source of truth for all cash flow calculations.

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
    warnings.warn(
        "run_dcf() is deprecated. Use calculate_deal() from detailed_cashflow.py instead. "
        "calculate_deal() is the single source of truth for all calculations.",
        DeprecationWarning,
        stacklevel=2,
    )
    result = DCFResult(scenario=scenario)

    # === Setup: Calculate upstream values ===

    # Land cost is now a direct input (lump sum)
    land_cost = inputs.land_cost

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
    affordable_units = get_total_affordable_units(allocations)

    # GPR
    gpr_result = calculate_gpr(allocations)

    # Fee waiver (mixed-income only)
    fee_waiver = 0.0
    incentive_config: IncentiveConfig | None = None

    if scenario == Scenario.MIXED_INCOME and inputs.incentive_config is not None:
        incentive_config = inputs.incentive_config
        if incentive_config.toggles.smart_fee_waiver:
            fee_waiver = incentive_config.get_waiver_amount(affordable_units)

    # Estimate monthly OpEx for reserves
    monthly_opex = (
        inputs.opex_utilities +
        inputs.opex_maintenance +
        inputs.opex_misc
    ) / 12 * total_units

    # Estimate monthly debt service for reserves (using actual amortization)
    hard_costs_est = inputs.target_units * inputs.hard_cost_per_unit
    soft_costs_est = hard_costs_est * inputs.soft_cost_pct
    estimated_tdc = land_cost + hard_costs_est + soft_costs_est
    construction_loan_est = estimated_tdc * inputs.construction_ltc
    # Calculate actual P&I using amortization formula instead of 1.5x approximation
    monthly_rate = inputs.perm_rate / 12
    n_payments = inputs.perm_amort_years * 12
    if monthly_rate > 0 and n_payments > 0:
        # PMT formula: P * r * (1+r)^n / ((1+r)^n - 1)
        monthly_debt_service = construction_loan_est * monthly_rate * (1 + monthly_rate) ** n_payments / ((1 + monthly_rate) ** n_payments - 1)
    else:
        monthly_debt_service = 0.0

    # TDC (IDC includes construction + partial lease-up, net of NOI offset)
    tdc_result = calculate_tdc(
        land_cost=land_cost,
        target_units=inputs.target_units,
        hard_cost_per_unit=inputs.hard_cost_per_unit,
        soft_cost_pct=inputs.soft_cost_pct,
        construction_ltc=inputs.construction_ltc,
        construction_rate=inputs.construction_rate,
        construction_months=inputs.construction_months,
        leaseup_months=inputs.idc_leaseup_months,  # Net months for IDC (after NOI offset)
        predevelopment_cost_pct=inputs.predevelopment_cost_pct,
        fee_waiver_amount=fee_waiver,
        hard_cost_contingency_pct=inputs.hard_cost_contingency_pct,
        soft_cost_contingency_pct=inputs.soft_cost_contingency_pct,
        developer_fee_pct=inputs.developer_fee_pct,
        construction_loan_fee_pct=0.01,
        monthly_opex=monthly_opex,
        monthly_debt_service=monthly_debt_service,
        operating_reserve_months=inputs.operating_reserve_months,
        leaseup_reserve_months=inputs.leaseup_reserve_months,
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

    # Property taxes - use TDC as the assessed value basis
    # This matches typical tax assessment methodology for new developments
    # The TDC represents the cost basis which is how properties are initially assessed
    tax_result = calculate_property_taxes(
        assessed_value=tdc_result.tdc,
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
    # Perm loan takes out construction loan - they should be equal
    result.perm_loan_amount = construction_loan.loan_amount

    # Equity required
    equity = tdc_result.tdc - construction_loan.loan_amount
    tif_lump_sum_value = 0.0
    if tif_result is not None and incentive_config is not None:
        if incentive_config.toggles.tif_lump_sum:
            tif_lump_sum_value = tif_result.capitalized_value
            equity -= tif_lump_sum_value
    result.equity_required = equity

    # Generate draw schedule for proper equity timing
    # Draw waterfall: Equity FIRST → TIF SECOND → Debt THIRD
    draw_sched = generate_draw_schedule(
        tdc=tdc_result.tdc,
        equity=equity,
        construction_loan=construction_loan.loan_amount,
        predevelopment_months=inputs.predevelopment_months,
        construction_months=inputs.construction_months,
        construction_rate=inputs.construction_rate,
        leaseup_months=inputs.leaseup_months,
        tif_lump_sum=tif_lump_sum_value,
    )
    result.draw_schedule = draw_sched

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
            # Equity is drawn as costs are incurred during predevelopment
            # Use draw schedule to get the equity draw for this month
            period_draw = draw_sched.get_period(month)
            # equity_draw is negative (outflow), store as positive contribution
            cf.equity_contribution = -period_draw.equity_draw if period_draw.equity_draw < 0 else 0.0
            cf.levered_cf = 0.0
            cf.unlevered_cf = 0.0

        elif phase == Phase.CONSTRUCTION:
            # Equity continues to be drawn as costs are incurred during construction
            # Once equity is exhausted, TIF and then debt fund remaining costs
            period_draw = draw_sched.get_period(month)
            # equity_draw is negative (outflow), store as positive contribution
            cf.equity_contribution = -period_draw.equity_draw if period_draw.equity_draw < 0 else 0.0
            cf.levered_cf = 0.0
            cf.unlevered_cf = 0.0

        elif phase == Phase.LEASE_UP:
            # During lease-up, construction loan is still in place
            # Any NOI shortfall is funded by the construction loan
            # From developer equity perspective, levered CF = 0
            # Calculate NOI for informational purposes but don't use perm debt service
            _calculate_leaseup_period(
                cf, month, gpr_result, inputs, total_units,
                tax_result, years_operating,
                tif_result, incentive_config,
            )

        elif phase == Phase.OPERATIONS:
            _calculate_operating_period(
                cf, month, gpr_result, inputs, total_units,
                tax_result, perm_loan, years_operating,
                tif_result, incentive_config, is_leaseup=False,
            )

        elif phase == Phase.REVERSION:
            # Sale event
            # Calculate reversion NOI based on methodology
            if inputs.reversion_noi_basis == ReversionNOIBasis.FORWARD:
                # FORWARD: Use stabilized NOI projected 12 months forward from sale
                # This matches Excel methodology of using month 72 NOI for month 60 sale
                # Growth is applied from stabilization to 12 months past sale
                months_since_stabilization = inputs.operations_months + 12  # ops + 12 forward
                years_of_growth = months_since_stabilization / 12
                reversion_noi = stabilized_noi_annual * (1 + inputs.market_rent_growth) ** years_of_growth
            else:
                # TRAILING: Use actual trailing 12 months of NOI
                ops_cfs = [c for c in cash_flows if c.phase in [Phase.OPERATIONS, Phase.LEASE_UP]]
                if ops_cfs:
                    trailing_12_noi = sum(c.noi for c in ops_cfs[-12:])
                    # Annualize if less than 12 months
                    if len(ops_cfs[-12:]) < 12:
                        trailing_12_noi = trailing_12_noi * 12 / len(ops_cfs[-12:])
                else:
                    trailing_12_noi = stabilized_noi_annual
                reversion_noi = trailing_12_noi

            # Sale price
            cf.sale_proceeds = reversion_noi / inputs.exit_cap_rate

            # Loan payoff (remaining balance)
            # Perm loan starts at operations (after lease-up), not at lease-up start
            months_with_perm = inputs.operations_months
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
    # Levered IRR: Equity invested as costs are incurred (per draw schedule),
    # cash flows during ops, sale proceeds at end

    # Build levered cash flow array
    # Predevelopment/Construction: Equity contributions (negative) as costs are incurred
    # Operations/Lease-up: Operating cash flows
    # Reversion: Net sale proceeds

    levered_cfs: List[float] = []

    for cf in cash_flows:
        if cf.phase in [Phase.PREDEVELOPMENT, Phase.CONSTRUCTION]:
            # Equity contributions during development (negative = outflow)
            if cf.equity_contribution > 0:
                levered_cfs.append(-cf.equity_contribution)
            else:
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

    # Unlevered IRR: TDC invested as costs are incurred (per draw schedule)
    unlevered_cfs: List[float] = []

    for cf in cash_flows:
        if cf.phase in [Phase.PREDEVELOPMENT, Phase.CONSTRUCTION]:
            # TDC deployed as costs are incurred
            # Get the TDC draw for this period from draw schedule
            period_draw = draw_sched.get_period(cf.month)
            if period_draw.tdc_draw_total > 0:
                unlevered_cfs.append(-period_draw.tdc_draw_total)
            else:
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
