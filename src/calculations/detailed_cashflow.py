"""Detailed cash flow engine with full pro forma rows.

This module produces the detailed monthly cash flows matching the
Excel model structure, including:
- Development draws (TDC, equity, debt)
- Escalations
- GPR by scenario
- Operating expenses and property taxes
- NOI buildup
- Investment (unlevered) cash flows
- Debt service (construction and permanent)
- Levered cash flows and IRR
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import date
from dateutil.relativedelta import relativedelta
from enum import Enum
import numpy_financial as npf

from src.calculations.sources_uses import SourcesUses, calculate_sources_uses
from src.calculations.draw_schedule import (
    DrawSchedule, PeriodDraw, Phase,
    generate_draw_schedule_from_sources_uses
)
from src.calculations.property_tax import (
    TaxingAuthorityStack, PropertyTaxSchedule,
    calculate_assessed_value_schedule, generate_property_tax_schedule,
    get_austin_tax_stack, AssessedValueTiming
)


class LoanType(str, Enum):
    """Type of loan active in a period."""
    NONE = "none"
    CONSTRUCTION = "construction"
    PERMANENT = "permanent"


@dataclass
class PeriodHeader:
    """Header information for a period."""
    period: int  # 1-indexed
    date_from: date
    date_to: date

    # Phase flags
    is_predevelopment: bool
    is_construction: bool
    is_leaseup: bool
    is_operations: bool
    is_operations_ftm: bool  # First twelve months of operations
    is_reversion: bool
    is_leaseup_to_ftm_noi: bool  # Lease-up period with less than FTM NOI

    # Loan flags
    loan_type: LoanType


@dataclass
class DevelopmentRow:
    """Development section of cash flow."""
    # TDC tracking
    tdc_total: float
    tdc_bop: float
    draw_pct_total: float
    draw_pct_predev: float
    draw_pct_construction: float
    draw_dollars_total: float
    draw_dollars_predev: float
    draw_dollars_construction: float
    tdc_eop: float
    tdc_to_be_funded: float  # Amount to be funded this period


@dataclass
class EquityRow:
    """Equity section of cash flow."""
    equity_bop: float
    equity_drawn: float  # Negative = outflow
    equity_eop: float


@dataclass
class DebtSourceRow:
    """Debt sources section (construction loan availability)."""
    debt_bop: float
    to_be_financed: float
    debt_eop: float


@dataclass
class EscalationRow:
    """Escalation tracking."""
    active_count: int  # Years since base for escalation
    market_revenue_bump: float  # Cumulative %
    affordable_revenue_bump: float  # Cumulative %
    opex_bump: float  # Cumulative %
    prop_tax_bump: float  # Cumulative %


@dataclass
class GPRRow:
    """Gross Potential Rent section."""
    gpr_all_market: float  # If 100% market rate
    gpr_mixed_market: float  # Market portion of mixed scenario
    gpr_mixed_affordable: float  # Affordable portion of mixed scenario
    gpr_total: float  # Total GPR for the active scenario


@dataclass
class OpexRow:
    """Operating expenses section."""
    opex_ex_prop_taxes: float
    property_taxes: float
    total_opex: float


@dataclass
class TIFRow:
    """TIF incentive section."""
    # Tax amounts before incentive
    gross_property_taxes: float  # Full taxes without any incentive

    # Abatement (reduces taxes paid)
    abatement_amount: float  # Tax reduction from abatement

    # Net taxes actually paid
    net_property_taxes: float  # Gross - abatement

    # TIF stream reimbursement (received after paying taxes)
    tif_reimbursement: float  # City increment reimbursed to developer

    # Net TIF benefit for this period
    net_tif_benefit: float  # Reimbursement or abatement value


@dataclass
class OperationsRow:
    """Operations section (NOI buildup)."""
    leaseup_pct: float  # 0% to max_occupancy
    gpr: float
    vacancy_rate: float
    less_vacancy: float
    egi: float  # Effective Gross Income
    less_opex_ex_taxes: float
    less_property_taxes: float
    noi: float
    noi_reserve_in_tdc: float  # Operating deficit reserve

    # TIF-related (optional, may be None)
    tif: Optional['TIFRow'] = None
    tif_reimbursement: float = 0.0  # Included in NOI if TIF stream


@dataclass
class InvestmentRow:
    """Investment section (unlevered cash flow)."""
    dev_cost: float  # Development cost outflow (negative)
    reserves: float  # Replacement reserves (negative)
    reversion: float  # Sale proceeds (positive, reversion period only)
    unlevered_cf: float


@dataclass
class ConstructionDebtRow:
    """Construction loan section."""
    principal_bop: float
    debt_added: float
    interest_in_period: float
    repaid: float  # Payoff when converting to perm
    principal_eop: float
    net_cf: float  # Net cash flow from construction debt


@dataclass
class PermanentDebtRow:
    """Permanent loan section."""
    principal_bop: float
    pmt_in_period: float  # Total payment
    interest_pmt: float  # Interest portion
    principal_pmt: float  # Principal portion
    payoff: float  # Balance payoff at reversion
    principal_eop: float
    net_cf: float  # Net cash flow from perm debt


@dataclass
class MezzanineDebtRow:
    """Mezzanine debt section (junior debt, higher rate)."""
    principal_bop: float
    interest_rate: float  # Annual rate for reference
    pmt_in_period: float  # Total payment (usually I/O)
    interest_pmt: float  # Interest portion
    principal_pmt: float  # Principal portion (usually 0 for I/O)
    payoff: float  # Balance payoff at reversion
    principal_eop: float
    net_cf: float  # Net cash flow from mezz debt
    is_active: bool = False  # Whether mezz is part of capital stack


@dataclass
class PreferredEquityRow:
    """Preferred equity section (equity-like but with priority return)."""
    balance_bop: float
    preferred_return_rate: float  # Annual rate
    accrued_return: float  # Accrued preferred return
    paid_return: float  # Return paid this period
    payoff: float  # Balance + accrued at reversion
    balance_eop: float
    net_cf: float  # Net cash flow from preferred
    is_active: bool = False


@dataclass
class DetailedPeriodCashFlow:
    """Complete detailed cash flow for one period.

    Contains all rows from the Excel model structure.
    """
    # Header
    header: PeriodHeader

    # Development
    development: DevelopmentRow

    # Equity
    equity: EquityRow

    # Debt sources
    debt_source: DebtSourceRow

    # Escalations
    escalation: EscalationRow

    # GPR
    gpr: GPRRow

    # Operating expenses
    opex: OpexRow

    # Operations
    operations: OperationsRow

    # Investment (unlevered)
    investment: InvestmentRow

    # Construction debt
    construction_debt: ConstructionDebtRow

    # Permanent debt
    permanent_debt: PermanentDebtRow

    # Mezzanine debt (optional)
    mezzanine_debt: MezzanineDebtRow = None

    # Preferred equity (optional)
    preferred_equity: PreferredEquityRow = None

    # Final
    net_senior_debt_cf: float = 0.0  # Construction + perm
    net_mezz_debt_cf: float = 0.0    # Mezzanine
    net_preferred_cf: float = 0.0     # Preferred equity
    net_debt_cf: float = 0.0          # Combined all debt layers
    levered_cf: float = 0.0


@dataclass
class DetailedCashFlowResult:
    """Complete detailed cash flow result.

    Contains all periods plus summary metrics.
    """
    periods: List[DetailedPeriodCashFlow]

    # Inputs
    sources_uses: SourcesUses
    draw_schedule: DrawSchedule

    # Summary metrics
    total_periods: int
    unlevered_irr: float
    levered_irr: float

    # Phase boundaries
    predevelopment_end: int
    construction_end: int
    leaseup_end: int
    operations_end: int

    # Key totals
    total_equity_invested: float
    total_noi: float
    total_debt_service: float
    reversion_value: float

    def get_period(self, period: int) -> DetailedPeriodCashFlow:
        """Get cash flow for a specific period (1-indexed)."""
        if period < 1 or period > len(self.periods):
            raise IndexError(f"Period {period} out of range")
        return self.periods[period - 1]


def _calculate_monthly_pmt(
    principal: float,
    annual_rate: float,
    amort_years: int,
) -> Tuple[float, float, float]:
    """Calculate monthly payment split into interest and principal.

    Args:
        principal: Loan principal
        annual_rate: Annual interest rate
        amort_years: Amortization period in years

    Returns:
        Tuple of (total_payment, interest_portion, principal_portion)
    """
    if principal <= 0 or annual_rate <= 0 or amort_years <= 0:
        return (0.0, 0.0, 0.0)

    monthly_rate = annual_rate / 12
    n_payments = amort_years * 12

    # Calculate monthly payment
    pmt = principal * (monthly_rate * (1 + monthly_rate) ** n_payments) / \
          ((1 + monthly_rate) ** n_payments - 1)

    # First month's interest
    interest = principal * monthly_rate
    principal_pmt = pmt - interest

    return (pmt, interest, principal_pmt)


def generate_detailed_cash_flow(
    # Sources & Uses inputs
    land_cost: float,
    hard_costs: float,
    soft_cost_pct: float,
    construction_ltc: float,
    construction_rate: float,

    # Timing
    start_date: date,
    predevelopment_months: int,
    construction_months: int,
    leaseup_months: int,
    operations_months: int,

    # Revenue inputs
    monthly_gpr_at_stabilization: float,
    vacancy_rate: float,
    leaseup_pace: float,  # Monthly absorption rate
    max_occupancy: float,

    # Operating expense inputs
    annual_opex_per_unit: float,  # Annual operating expenses per unit (ex property taxes)
    total_units: int,

    # Property tax inputs
    existing_assessed_value: float,
    tax_stack: Optional[TaxingAuthorityStack] = None,
    assessment_growth_rate: float = 0.02,

    # Escalation inputs
    market_rent_growth: float = 0.02,
    opex_growth: float = 0.03,
    prop_tax_growth: float = 0.02,

    # Permanent loan inputs
    perm_rate: float = 0.06,
    perm_amort_years: int = 20,
    perm_ltv_max: float = 0.65,
    perm_dscr_min: float = 1.25,

    # Exit inputs
    exit_cap_rate: float = 0.055,
    reserves_pct: float = 0.005,  # % of EGI

    # Mixed income inputs (optional)
    affordable_pct: float = 0.0,
    affordable_rent_discount: float = 0.0,  # As decimal (e.g., 0.40 = 40% discount)
    affordable_rent_growth: float = 0.01,

    # Mezzanine debt inputs (optional)
    mezzanine_amount: float = 0.0,
    mezzanine_rate: float = 0.12,  # 12% default
    mezzanine_io: bool = True,  # Interest-only

    # Preferred equity inputs (optional)
    preferred_amount: float = 0.0,
    preferred_return: float = 0.10,  # 10% default

    # TIF treatment inputs (optional)
    tif_treatment: str = "none",  # "none", "lump_sum", "abatement", "stream"
    tif_lump_sum: float = 0.0,  # For lump_sum treatment (handled in S&U)
    tif_abatement_pct: float = 0.0,  # % of city taxes abated (0-1.0)
    tif_abatement_years: int = 0,  # Years of abatement
    tif_stream_pct: float = 0.0,  # % of city increment reimbursed (0-1.0)
    tif_stream_years: int = 0,  # Years of stream
    tif_start_delay_months: int = 0,  # Delay from stabilization to TIF start
) -> DetailedCashFlowResult:
    """Generate detailed cash flow for a development.

    This is the main entry point for the detailed cash flow engine.
    """
    # Default to Austin tax stack
    if tax_stack is None:
        tax_stack = get_austin_tax_stack()

    # Calculate sources & uses
    sources_uses = calculate_sources_uses(
        land_cost=land_cost,
        hard_costs=hard_costs,
        soft_cost_pct=soft_cost_pct,
        construction_ltc=construction_ltc,
        construction_rate=construction_rate,
        construction_months=construction_months,
        predevelopment_months=predevelopment_months,
    )

    # Generate draw schedule
    draw_schedule = generate_draw_schedule_from_sources_uses(
        sources_uses=sources_uses,
        predevelopment_months=predevelopment_months,
        construction_months=construction_months,
        construction_rate=construction_rate,
    )

    # Calculate phase boundaries
    predev_end = predevelopment_months
    construction_end = predev_end + construction_months
    leaseup_end = construction_end + leaseup_months
    operations_end = leaseup_end + operations_months
    total_periods = operations_end + 1  # +1 for reversion period

    # Calculate stabilized value for property tax assessment
    annual_noi_stabilized = monthly_gpr_at_stabilization * 12 * (1 - vacancy_rate) * 0.60  # Rough NOI margin
    stabilized_value = annual_noi_stabilized / exit_cap_rate

    # Calculate assessed values
    assessed_values = calculate_assessed_value_schedule(
        existing_value=existing_assessed_value,
        stabilized_value=stabilized_value,
        predevelopment_months=predevelopment_months,
        construction_months=construction_months,
        leaseup_months=leaseup_months,
        operations_months=operations_months,
        timing=AssessedValueTiming.AT_STABILIZATION,
        assessment_growth_rate=assessment_growth_rate,
    )

    # Size permanent loan at stabilization
    # Stabilized NOI
    stabilized_gpr = monthly_gpr_at_stabilization * (1 - vacancy_rate)
    stabilized_egi = stabilized_gpr
    stabilized_monthly_opex = (annual_opex_per_unit * total_units) / 12
    stabilized_monthly_taxes = stabilized_value * tax_stack.total_rate_decimal / 12
    stabilized_monthly_noi = stabilized_egi - stabilized_monthly_opex - stabilized_monthly_taxes
    stabilized_annual_noi = stabilized_monthly_noi * 12

    # LTV constraint
    ltv_loan = stabilized_value * perm_ltv_max

    # DSCR constraint
    annual_debt_service_max = stabilized_annual_noi / perm_dscr_min
    monthly_rate = perm_rate / 12
    n_payments = perm_amort_years * 12
    if monthly_rate > 0 and n_payments > 0:
        dscr_loan = annual_debt_service_max / 12 * \
                    ((1 + monthly_rate) ** n_payments - 1) / \
                    (monthly_rate * (1 + monthly_rate) ** n_payments)
    else:
        dscr_loan = 0

    # Take lesser of LTV and DSCR
    perm_loan_amount = min(ltv_loan, dscr_loan)

    # Generate period-by-period cash flows
    periods: List[DetailedPeriodCashFlow] = []

    # Track running balances
    const_loan_balance = 0.0
    perm_loan_balance = 0.0
    mezz_loan_balance = mezzanine_amount  # Funded at start of perm period
    preferred_balance = preferred_amount   # Funded at start of perm period
    preferred_accrued = 0.0               # Accrued but unpaid preferred return
    cumulative_occupancy = 0.0
    escalation_year = 0

    # Mezzanine and preferred active flags
    has_mezzanine = mezzanine_amount > 0
    has_preferred = preferred_amount > 0

    # Monthly rates for mezz/preferred
    mezz_monthly_rate = mezzanine_rate / 12
    pref_monthly_rate = preferred_return / 12

    # For IRR calculation
    unlevered_cfs: List[float] = []
    levered_cfs: List[float] = []

    # Track totals
    total_equity_invested = 0.0
    total_noi = 0.0
    total_debt_service = 0.0
    reversion_value = 0.0

    for period in range(1, total_periods + 1):
        # Determine dates
        period_start = start_date + relativedelta(months=period - 1)
        period_end = start_date + relativedelta(months=period) - relativedelta(days=1)

        # Determine phase
        is_predev = period <= predev_end
        is_construction = predev_end < period <= construction_end
        is_leaseup = construction_end < period <= leaseup_end
        is_operations = leaseup_end < period <= operations_end
        is_reversion = period == total_periods
        is_ftm = is_operations and (period - leaseup_end) <= 12

        # Determine loan type
        if period <= construction_end:
            loan_type = LoanType.CONSTRUCTION if period > predev_end else LoanType.NONE
        elif period == construction_end + 1:
            # Conversion period - both loans active briefly
            loan_type = LoanType.PERMANENT
        else:
            loan_type = LoanType.PERMANENT

        # Header
        header = PeriodHeader(
            period=period,
            date_from=period_start,
            date_to=period_end,
            is_predevelopment=is_predev,
            is_construction=is_construction,
            is_leaseup=is_leaseup,
            is_operations=is_operations,
            is_operations_ftm=is_ftm,
            is_reversion=is_reversion,
            is_leaseup_to_ftm_noi=is_leaseup,
            loan_type=loan_type,
        )

        # Development section (from draw schedule during predev/construction)
        if period <= construction_end:
            draw = draw_schedule.get_period(period)
            development = DevelopmentRow(
                tdc_total=sources_uses.tdc,
                tdc_bop=draw.tdc_bop,
                draw_pct_total=draw.tdc_draw_pct,
                draw_pct_predev=draw.tdc_draw_predev / sources_uses.tdc if sources_uses.tdc > 0 else 0,
                draw_pct_construction=draw.tdc_draw_construction / sources_uses.tdc if sources_uses.tdc > 0 else 0,
                draw_dollars_total=draw.tdc_draw_total,
                draw_dollars_predev=draw.tdc_draw_predev,
                draw_dollars_construction=draw.tdc_draw_construction,
                tdc_eop=draw.tdc_eop,
                tdc_to_be_funded=draw.tdc_draw_total,
            )
            equity = EquityRow(
                equity_bop=draw.equity_bop,
                equity_drawn=draw.equity_draw,
                equity_eop=draw.equity_eop,
            )
            debt_source = DebtSourceRow(
                debt_bop=draw.const_debt_bop,
                to_be_financed=draw.const_debt_draw,
                debt_eop=draw.const_debt_eop,
            )
            const_loan_balance = draw.const_debt_eop
            total_equity_invested += abs(draw.equity_draw)
        else:
            # No development draws after construction
            development = DevelopmentRow(
                tdc_total=sources_uses.tdc,
                tdc_bop=0,
                draw_pct_total=0,
                draw_pct_predev=0,
                draw_pct_construction=0,
                draw_dollars_total=0,
                draw_dollars_predev=0,
                draw_dollars_construction=0,
                tdc_eop=0,
                tdc_to_be_funded=0,
            )
            equity = EquityRow(
                equity_bop=0,
                equity_drawn=0,
                equity_eop=0,
            )
            debt_source = DebtSourceRow(
                debt_bop=0,
                to_be_financed=0,
                debt_eop=0,
            )

        # Escalations
        # Bump annually starting from lease-up
        if period == construction_end + 1:
            escalation_year = 0  # Reset at start of lease-up
        elif is_leaseup or is_operations or is_reversion:
            if (period - construction_end - 1) % 12 == 0 and period > construction_end + 1:
                escalation_year += 1

        market_bump = (1 + market_rent_growth) ** escalation_year - 1
        affordable_bump = (1 + affordable_rent_growth) ** escalation_year - 1
        opex_bump = (1 + opex_growth) ** escalation_year - 1
        tax_bump = (1 + prop_tax_growth) ** escalation_year - 1

        escalation = EscalationRow(
            active_count=escalation_year,
            market_revenue_bump=market_bump,
            affordable_revenue_bump=affordable_bump,
            opex_bump=opex_bump,
            prop_tax_bump=tax_bump,
        )

        # GPR
        # Apply escalation
        base_gpr_market = monthly_gpr_at_stabilization
        escalated_gpr_market = base_gpr_market * (1 + market_bump)

        # Mixed income GPR
        market_portion = 1 - affordable_pct
        affordable_portion = affordable_pct
        affordable_rent = base_gpr_market * (1 - affordable_rent_discount)
        escalated_affordable = affordable_rent * (1 + affordable_bump)

        gpr = GPRRow(
            gpr_all_market=escalated_gpr_market,
            gpr_mixed_market=escalated_gpr_market * market_portion,
            gpr_mixed_affordable=escalated_affordable * affordable_portion,
            gpr_total=escalated_gpr_market,  # Use market rate for now
        )

        # Operating expenses
        if period <= construction_end:
            # No opex during development
            opex = OpexRow(
                opex_ex_prop_taxes=0,
                property_taxes=0,
                total_opex=0,
            )
            tif_row = None
        else:
            # Apply escalation
            base_opex = (annual_opex_per_unit * total_units) / 12
            escalated_opex = base_opex * (1 + opex_bump)

            # Property taxes from assessed value
            if period <= len(assessed_values):
                assessed = assessed_values[period - 1]
            else:
                assessed = assessed_values[-1]

            # Calculate base property taxes (full amount)
            monthly_prop_tax_gross = assessed * tax_stack.total_rate_decimal / 12

            # Calculate TIF benefits based on treatment
            abatement_amount = 0.0
            tif_reimbursement = 0.0

            # Determine if TIF is active this period
            stabilization_period = construction_end + leaseup_months
            months_since_stabilization = max(0, period - stabilization_period)
            tif_start_period = stabilization_period + tif_start_delay_months

            if tif_treatment == "abatement" and period >= tif_start_period:
                # Check if within abatement term
                months_in_tif = period - tif_start_period
                if months_in_tif < tif_abatement_years * 12:
                    # Calculate city portion only (abatement applies to city tax only)
                    city_tax_portion = assessed * tax_stack.tif_participating_rate_decimal / 12
                    # Apply abatement to city portion
                    # For affordable units only (proportional)
                    affordable_share = affordable_pct
                    abatement_amount = city_tax_portion * tif_abatement_pct * affordable_share

            elif tif_treatment == "stream" and period >= tif_start_period:
                # Check if within stream term
                months_in_tif = period - tif_start_period
                if months_in_tif < tif_stream_years * 12:
                    # Calculate city's increment portion
                    increment_value = max(0, assessed - existing_assessed_value)
                    city_increment = increment_value * tax_stack.tif_participating_rate_decimal / 12
                    # Reimbursement is % of city's increment
                    tif_reimbursement = city_increment * tif_stream_pct

            # Net property taxes paid (after abatement)
            monthly_prop_tax_net = monthly_prop_tax_gross - abatement_amount

            # Create TIF row
            tif_row = TIFRow(
                gross_property_taxes=monthly_prop_tax_gross,
                abatement_amount=abatement_amount,
                net_property_taxes=monthly_prop_tax_net,
                tif_reimbursement=tif_reimbursement,
                net_tif_benefit=abatement_amount + tif_reimbursement,
            )

            opex = OpexRow(
                opex_ex_prop_taxes=escalated_opex,
                property_taxes=monthly_prop_tax_net,  # Net taxes after abatement
                total_opex=escalated_opex + monthly_prop_tax_net,
            )

        # Operations (NOI buildup)
        if period <= construction_end:
            # No operations during development
            operations = OperationsRow(
                leaseup_pct=0,
                gpr=0,
                vacancy_rate=0,
                less_vacancy=0,
                egi=0,
                less_opex_ex_taxes=0,
                less_property_taxes=0,
                noi=0,
                noi_reserve_in_tdc=0,
                tif=None,
                tif_reimbursement=0,
            )
            noi = 0
        else:
            # Calculate occupancy
            if is_leaseup:
                # Ramp up occupancy
                months_in_leaseup = period - construction_end
                cumulative_occupancy = min(max_occupancy, leaseup_pace * months_in_leaseup)
            else:
                cumulative_occupancy = max_occupancy

            # Calculate NOI
            period_gpr = gpr.gpr_total * cumulative_occupancy
            period_vacancy = period_gpr * vacancy_rate if cumulative_occupancy >= max_occupancy else 0
            period_egi = period_gpr - period_vacancy

            # TIF reimbursement adds to NOI
            tif_reimb = tif_row.tif_reimbursement if tif_row else 0.0

            # NOI = EGI - OpEx + TIF Reimbursement
            base_noi = period_egi - opex.total_opex
            adjusted_noi = base_noi + tif_reimb

            operations = OperationsRow(
                leaseup_pct=cumulative_occupancy,
                gpr=period_gpr,
                vacancy_rate=vacancy_rate if cumulative_occupancy >= max_occupancy else 0,
                less_vacancy=period_vacancy,
                egi=period_egi,
                less_opex_ex_taxes=opex.opex_ex_prop_taxes,
                less_property_taxes=opex.property_taxes,
                noi=adjusted_noi,
                noi_reserve_in_tdc=0,  # Tracked but not active
                tif=tif_row,
                tif_reimbursement=tif_reimb,
            )
            noi = operations.noi
            total_noi += noi

        # Investment (unlevered cash flow)
        dev_cost = -development.tdc_to_be_funded if period <= construction_end else 0
        reserves_amount = -operations.egi * reserves_pct if operations.egi > 0 else 0

        # Reversion
        if is_reversion:
            # Calculate sale price
            terminal_noi = noi * 12  # Annualize
            sale_price = terminal_noi / exit_cap_rate
            reversion_amount = sale_price
            reversion_value = sale_price
        else:
            reversion_amount = 0

        investment = InvestmentRow(
            dev_cost=dev_cost,
            reserves=reserves_amount,
            reversion=reversion_amount,
            unlevered_cf=dev_cost + noi + reserves_amount + reversion_amount,
        )

        # Construction debt
        if period <= construction_end:
            draw = draw_schedule.get_period(period)
            const_debt = ConstructionDebtRow(
                principal_bop=draw.const_debt_bop,
                debt_added=draw.const_debt_draw,
                interest_in_period=draw.const_debt_interest,
                repaid=0,
                principal_eop=draw.const_debt_eop,
                net_cf=draw.const_debt_draw,  # Inflow from draws
            )
        elif period == construction_end + 1:
            # Payoff construction loan
            const_debt = ConstructionDebtRow(
                principal_bop=const_loan_balance,
                debt_added=0,
                interest_in_period=0,
                repaid=-const_loan_balance,  # Payoff
                principal_eop=0,
                net_cf=-const_loan_balance,  # Outflow for payoff
            )
            const_loan_balance = 0
        else:
            const_debt = ConstructionDebtRow(
                principal_bop=0,
                debt_added=0,
                interest_in_period=0,
                repaid=0,
                principal_eop=0,
                net_cf=0,
            )

        # Permanent debt
        if period == construction_end + 1:
            # Perm loan funds
            perm_loan_balance = perm_loan_amount
            pmt, i_pmt, p_pmt = _calculate_monthly_pmt(perm_loan_amount, perm_rate, perm_amort_years)
            perm_debt = PermanentDebtRow(
                principal_bop=0,
                pmt_in_period=pmt,
                interest_pmt=i_pmt,
                principal_pmt=p_pmt,
                payoff=0,
                principal_eop=perm_loan_amount - p_pmt,
                net_cf=perm_loan_amount - pmt,  # Inflow from funding - first payment
            )
            perm_loan_balance = perm_loan_amount - p_pmt
            total_debt_service += pmt
        elif period > construction_end + 1 and not is_reversion:
            # Regular perm loan payments
            pmt, i_pmt, p_pmt = _calculate_monthly_pmt(perm_loan_balance, perm_rate, perm_amort_years)
            perm_debt = PermanentDebtRow(
                principal_bop=perm_loan_balance,
                pmt_in_period=pmt,
                interest_pmt=i_pmt,
                principal_pmt=p_pmt,
                payoff=0,
                principal_eop=perm_loan_balance - p_pmt,
                net_cf=-pmt,  # Outflow for payment
            )
            perm_loan_balance -= p_pmt
            total_debt_service += pmt
        elif is_reversion:
            # Payoff perm loan
            pmt, i_pmt, p_pmt = _calculate_monthly_pmt(perm_loan_balance, perm_rate, perm_amort_years)
            perm_debt = PermanentDebtRow(
                principal_bop=perm_loan_balance,
                pmt_in_period=pmt,
                interest_pmt=i_pmt,
                principal_pmt=p_pmt,
                payoff=-perm_loan_balance,
                principal_eop=0,
                net_cf=-pmt - perm_loan_balance,  # Payment + payoff
            )
            total_debt_service += pmt
            perm_loan_balance = 0
        else:
            perm_debt = PermanentDebtRow(
                principal_bop=0,
                pmt_in_period=0,
                interest_pmt=0,
                principal_pmt=0,
                payoff=0,
                principal_eop=0,
                net_cf=0,
            )

        # === MEZZANINE DEBT ===
        if has_mezzanine and is_operations:
            mezz_interest = mezz_loan_balance * mezz_monthly_rate
            if mezzanine_io:
                mezz_principal_pmt = 0.0
                mezz_pmt = mezz_interest
            else:
                # Amortizing mezz (rare but possible)
                mezz_pmt = mezz_interest * 1.2  # Rough approximation
                mezz_principal_pmt = mezz_pmt - mezz_interest

            if is_reversion:
                mezz_payoff = mezz_loan_balance
                mezz_loan_balance = 0
            else:
                mezz_payoff = 0
                mezz_loan_balance -= mezz_principal_pmt

            mezz_debt = MezzanineDebtRow(
                principal_bop=mezz_loan_balance + mezz_principal_pmt,
                interest_rate=mezzanine_rate,
                pmt_in_period=mezz_pmt,
                interest_pmt=mezz_interest,
                principal_pmt=mezz_principal_pmt,
                payoff=mezz_payoff,
                principal_eop=mezz_loan_balance,
                net_cf=-mezz_pmt - mezz_payoff,
                is_active=True,
            )
            total_debt_service += mezz_pmt
        else:
            mezz_debt = MezzanineDebtRow(
                principal_bop=mezz_loan_balance if has_mezzanine else 0,
                interest_rate=mezzanine_rate,
                pmt_in_period=0,
                interest_pmt=0,
                principal_pmt=0,
                payoff=0,
                principal_eop=mezz_loan_balance if has_mezzanine else 0,
                net_cf=0,
                is_active=has_mezzanine,
            )

        # === PREFERRED EQUITY ===
        if has_preferred and is_operations:
            pref_return_due = preferred_balance * pref_monthly_rate
            # Preferred return typically accrues and is paid from cash flow
            # For now, treat as paid if cash flow available
            pref_paid = pref_return_due  # Simplified: assume paid

            if is_reversion:
                pref_payoff = preferred_balance + preferred_accrued
                preferred_balance = 0
                preferred_accrued = 0
            else:
                pref_payoff = 0

            pref_equity = PreferredEquityRow(
                balance_bop=preferred_balance + (pref_payoff if is_reversion else 0),
                preferred_return_rate=preferred_return,
                accrued_return=preferred_accrued,
                paid_return=pref_paid,
                payoff=pref_payoff,
                balance_eop=preferred_balance,
                net_cf=-pref_paid - pref_payoff,
                is_active=True,
            )
        else:
            pref_equity = PreferredEquityRow(
                balance_bop=preferred_balance if has_preferred else 0,
                preferred_return_rate=preferred_return,
                accrued_return=0,
                paid_return=0,
                payoff=0,
                balance_eop=preferred_balance if has_preferred else 0,
                net_cf=0,
                is_active=has_preferred,
            )

        # Net debt and levered CF
        net_senior_debt_cf = const_debt.net_cf + perm_debt.net_cf
        net_mezz_debt_cf = mezz_debt.net_cf if has_mezzanine else 0
        net_preferred_cf = pref_equity.net_cf if has_preferred else 0
        net_debt_cf = net_senior_debt_cf + net_mezz_debt_cf + net_preferred_cf
        levered_cf = investment.unlevered_cf + net_debt_cf

        # Track for IRR
        unlevered_cfs.append(investment.unlevered_cf)
        levered_cfs.append(levered_cf)

        periods.append(DetailedPeriodCashFlow(
            header=header,
            development=development,
            equity=equity,
            debt_source=debt_source,
            escalation=escalation,
            gpr=gpr,
            opex=opex,
            operations=operations,
            investment=investment,
            construction_debt=const_debt,
            permanent_debt=perm_debt,
            mezzanine_debt=mezz_debt,
            preferred_equity=pref_equity,
            net_senior_debt_cf=net_senior_debt_cf,
            net_mezz_debt_cf=net_mezz_debt_cf,
            net_preferred_cf=net_preferred_cf,
            net_debt_cf=net_debt_cf,
            levered_cf=levered_cf,
        ))

    # Calculate IRRs
    try:
        monthly_unlevered_irr = npf.irr(unlevered_cfs)
        unlevered_irr = (1 + monthly_unlevered_irr) ** 12 - 1
    except:
        unlevered_irr = 0.0

    try:
        monthly_levered_irr = npf.irr(levered_cfs)
        levered_irr = (1 + monthly_levered_irr) ** 12 - 1
    except:
        levered_irr = 0.0

    return DetailedCashFlowResult(
        periods=periods,
        sources_uses=sources_uses,
        draw_schedule=draw_schedule,
        total_periods=total_periods,
        unlevered_irr=unlevered_irr,
        levered_irr=levered_irr,
        predevelopment_end=predev_end,
        construction_end=construction_end,
        leaseup_end=leaseup_end,
        operations_end=operations_end,
        total_equity_invested=total_equity_invested,
        total_noi=total_noi,
        total_debt_service=total_debt_service,
        reversion_value=reversion_value,
    )
