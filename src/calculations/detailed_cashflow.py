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


class AssessedValueBasis(str, Enum):
    """Basis for property tax assessed value calculation."""
    TDC = "tdc"  # Total Development Cost with escalations
    NOI = "noi"  # Capitalized NOI (income approach)

from src.calculations.sources_uses import SourcesUses, calculate_sources_uses
from src.calculations.debt import (
    calculate_interest_portion,
    calculate_principal_portion,
    size_permanent_loan,
)
from src.calculations.units import allocate_units, get_total_units, get_total_affordable_units
from src.calculations.revenue import calculate_gpr
from src.models.project import ProjectInputs, Scenario, TIFStartTiming
from src.calculations.draw_schedule import (
    DrawSchedule, PeriodDraw, Phase,
    generate_draw_schedule_from_sources_uses
)
from src.calculations.property_tax import (
    TaxingAuthorityStack, PropertyTaxSchedule,
    calculate_assessed_value_schedule, generate_property_tax_schedule,
    get_austin_tax_stack, build_tax_stack_from_rates, AssessedValueTiming
)
from src.calculations.trace import TraceContext, trace


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
    less_management_fee: float  # Management fee as % of EGI
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

    # Calculation trace (for transparency/auditing)
    trace_context: Optional[TraceContext] = None

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
    idc_leaseup_months: int,  # Additional IDC months during lease-up
    leaseup_months: int,
    operations_months: int,

    # Revenue inputs - GPR from proper unit allocation calculations
    # These values come from allocate_units() + calculate_gpr() in the caller
    market_gpr_monthly: float,           # Total monthly GPR from market-rate units
    affordable_gpr_monthly: float,       # Total monthly GPR from affordable units (at AMI rents)
    all_market_gpr_monthly: float,       # GPR if ALL units were at market rents (for comparison)
    vacancy_rate: float,
    leaseup_pace: float,  # Monthly absorption rate
    max_occupancy: float,

    # Operating expense inputs
    annual_opex_per_unit: float,  # Annual operating expenses per unit (ex property taxes)
    total_units: int,

    # Property tax inputs
    existing_assessed_value: float,

    # === Parameters with defaults below ===
    # Management fee
    management_fee_pct: float = 0.05,  # Management fee as % of EGI

    # Reserve inputs (added to TDC)
    operating_reserve_months: int = 3,  # Months of OpEx in reserve
    leaseup_reserve_months: int = 6,  # Months of leaseup deficit in reserve

    # Property tax options
    tax_stack: Optional[TaxingAuthorityStack] = None,
    assessment_growth_rate: float = 0.02,
    assessed_value_basis: AssessedValueBasis = AssessedValueBasis.TDC,  # TDC or NOI-based
    assessed_value_override: Optional[float] = None,  # Direct override for assessed value

    # Escalation inputs
    market_rent_growth: float = 0.02,
    opex_growth: float = 0.03,
    prop_tax_growth: float = 0.02,

    # Permanent loan inputs
    perm_rate: float = 0.06,
    perm_amort_years: int = 20,
    perm_ltv_max: float = 0.65,
    perm_dscr_min: float = 1.25,
    perm_ltc_max: Optional[float] = None,  # If set, cap perm at this % of TDC (loan-to-cost)

    # Exit inputs
    exit_cap_rate: float = 0.055,
    reserves_pct: float = 0.005,  # % of EGI
    selling_costs_pct: float = 0.02,  # Selling costs as % of sale price
    reversion_noi_basis: str = "forward",  # "forward" or "trailing" - which NOI to use for exit cap

    # IRR calculation options
    exclude_land_from_irr: bool = True,  # Land is recovered at sale, not a "cost"
    exclude_idc_from_irr: bool = True,   # IDC is financing cost, not development cost

    # Perm loan sizing options
    cap_perm_at_construction: bool = True,  # If True, perm loan ≤ construction loan (no cash-out)

    # Mixed income inputs (optional)
    affordable_pct: float = 0.0,
    affordable_rent_growth: float = 0.01,

    # Mezzanine debt inputs (optional)
    mezzanine_amount: float = 0.0,
    mezzanine_rate: float = 0.12,  # 12% default
    mezzanine_io: bool = True,  # Interest-only
    mezzanine_amort_years: int = 10,  # Amortization term if not IO

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
    # Detailed breakdown for display (optional)
    hard_cost_contingency: float = 0.0,
    soft_cost_contingency: float = 0.0,
    predevelopment_costs: float = 0.0,
    developer_fee: float = 0.0,
    loan_fee: float = 0.0,
) -> DetailedCashFlowResult:
    """Generate detailed cash flow for a development.

    This is the main entry point for the detailed cash flow engine.
    """
    # Default to Austin tax stack
    if tax_stack is None:
        tax_stack = get_austin_tax_stack()

    # Calculate base sources & uses (before reserves)
    sources_uses_base = calculate_sources_uses(
        land_cost=land_cost,
        hard_costs=hard_costs,
        soft_cost_pct=soft_cost_pct,
        construction_ltc=construction_ltc,
        construction_rate=construction_rate,
        construction_months=construction_months,
        predevelopment_months=predevelopment_months,
        leaseup_months=leaseup_months,
    )

    # Calculate reserves to add to TDC
    # Operating reserve: X months of stabilized operating expenses
    monthly_opex_estimate = (annual_opex_per_unit * total_units) / 12
    operating_reserve = monthly_opex_estimate * operating_reserve_months

    # Leaseup reserve: X months of projected deficit during leaseup
    # Calculate average occupancy from actual leaseup pace instead of hardcoded 0.5
    total_gpr_monthly_estimate = market_gpr_monthly + affordable_gpr_monthly
    # Linear lease-up: starts at 0, ends at min(leaseup_pace * leaseup_months, max_occupancy)
    final_leaseup_occupancy = min(leaseup_pace * leaseup_months, max_occupancy)
    avg_leaseup_occupancy = final_leaseup_occupancy / 2  # Average of linear ramp from 0 to final
    avg_leaseup_egi = total_gpr_monthly_estimate * avg_leaseup_occupancy
    avg_leaseup_deficit = max(0, monthly_opex_estimate - avg_leaseup_egi)
    leaseup_reserve = avg_leaseup_deficit * leaseup_reserve_months

    total_reserves = operating_reserve + leaseup_reserve

    # Recalculate sources & uses with reserves added to soft costs
    # Reserves are typically funded through equity, so add to soft cost equivalent
    reserve_as_soft_pct = total_reserves / hard_costs if hard_costs > 0 else 0
    effective_soft_cost_pct = soft_cost_pct + reserve_as_soft_pct

    sources_uses = calculate_sources_uses(
        land_cost=land_cost,
        hard_costs=hard_costs,
        soft_cost_pct=effective_soft_cost_pct,
        construction_ltc=construction_ltc,
        construction_rate=construction_rate,
        construction_months=construction_months,
        predevelopment_months=predevelopment_months,
        leaseup_months=leaseup_months,
    )

    # Store detailed breakdown values (for display) and reserves
    sources_uses.hard_cost_contingency = hard_cost_contingency
    sources_uses.soft_cost_contingency = soft_cost_contingency
    sources_uses.predevelopment_costs = predevelopment_costs
    sources_uses.developer_fee = developer_fee
    sources_uses.loan_fee = loan_fee
    sources_uses.reserves = total_reserves

    # TIF lump sum reduces equity requirement (front-loaded incentive)
    # The TIF is treated as an additional source of funds that offsets equity
    tif_lump_sum_applied = 0.0
    if tif_treatment == "lump_sum" and tif_lump_sum > 0:
        tif_lump_sum_applied = min(tif_lump_sum, sources_uses.equity)  # Can't exceed equity

    # Generate draw schedule (includes lease-up for construction loan interest)
    draw_schedule = generate_draw_schedule_from_sources_uses(
        sources_uses=sources_uses,
        predevelopment_months=predevelopment_months,
        construction_months=construction_months,
        construction_rate=construction_rate,
        leaseup_months=leaseup_months,
    )

    # Calculate phase boundaries
    predev_end = predevelopment_months
    construction_end = predev_end + construction_months
    leaseup_end = construction_end + leaseup_months
    operations_end = leaseup_end + operations_months
    total_periods = operations_end + 1  # +1 for reversion period

    # Calculate assessed value basis for property tax
    if assessed_value_override is not None:
        # Use directly specified assessed value
        assessment_base = assessed_value_override
    elif assessed_value_basis == AssessedValueBasis.TDC:
        # Use Total Development Cost as assessment basis (cost approach)
        # This is more typical of how assessors value new construction
        assessment_base = sources_uses.tdc
    else:
        # Use capitalized NOI (income approach)
        # Calculate NOI using actual inputs instead of rough 60% margin
        total_gpr_for_noi = market_gpr_monthly + affordable_gpr_monthly
        stabilized_egi_monthly = total_gpr_for_noi * (1 - vacancy_rate)
        monthly_opex = (annual_opex_per_unit * total_units) / 12
        monthly_mgmt = stabilized_egi_monthly * management_fee_pct
        # For initial assessment, exclude property taxes (they're circular with assessed value)
        annual_noi = (stabilized_egi_monthly - monthly_opex - monthly_mgmt) * 12
        assessment_base = annual_noi / exit_cap_rate

    # Calculate assessed values
    assessed_values = calculate_assessed_value_schedule(
        existing_value=existing_assessed_value,
        stabilized_value=assessment_base,
        predevelopment_months=predevelopment_months,
        construction_months=construction_months,
        leaseup_months=leaseup_months,
        operations_months=operations_months,
        timing=AssessedValueTiming.AT_STABILIZATION,
        assessment_growth_rate=assessment_growth_rate,
    )

    # Size permanent loan at stabilization using the canonical debt.py function
    # For loan sizing, always use income approach (what lenders use)
    # Stabilized NOI calculation using proper GPR values from unit allocations
    total_gpr_monthly = market_gpr_monthly + affordable_gpr_monthly

    stabilized_gpr = total_gpr_monthly * (1 - vacancy_rate)
    stabilized_egi = stabilized_gpr
    stabilized_monthly_opex = (annual_opex_per_unit * total_units) / 12
    stabilized_monthly_mgmt_fee = stabilized_egi * management_fee_pct  # Management fee as % of EGI
    stabilized_monthly_taxes = assessment_base * tax_stack.total_rate_decimal / 12
    stabilized_monthly_noi = stabilized_egi - stabilized_monthly_opex - stabilized_monthly_mgmt_fee - stabilized_monthly_taxes
    stabilized_annual_noi = stabilized_monthly_noi * 12

    # Trace GPR and NOI calculations
    trace("gpr.gpr_total", total_gpr_monthly, {
        "gpr.market_gpr_monthly": market_gpr_monthly,
        "gpr.affordable_gpr_monthly": affordable_gpr_monthly,
    })
    trace("operations.egi", stabilized_egi, {
        "operations.gpr": total_gpr_monthly,
        "inputs.vacancy_rate": vacancy_rate,
    })
    trace("operations.stabilized_noi", stabilized_annual_noi, {
        "operations.egi": stabilized_egi * 12,
        "opex.opex_ex_prop_taxes": stabilized_monthly_opex * 12,
        "operations.management_fee": stabilized_monthly_mgmt_fee * 12,
        "opex.property_taxes": stabilized_monthly_taxes * 12,
    })

    # Property value for LTV (always income approach for lenders)
    property_value_for_ltv = stabilized_annual_noi / exit_cap_rate

    # Trace property value
    trace("investment.stabilized_value", property_value_for_ltv, {
        "operations.stabilized_noi": stabilized_annual_noi,
        "inputs.exit_cap_rate": exit_cap_rate,
    })

    # Use canonical size_permanent_loan from debt.py (SINGLE SOURCE OF TRUTH)
    perm_loan_result = size_permanent_loan(
        stabilized_value=property_value_for_ltv,
        stabilized_noi=stabilized_annual_noi,
        ltv_max=perm_ltv_max,
        dscr_min=perm_dscr_min,
        perm_rate=perm_rate,
        amort_years=perm_amort_years,
        rate_buydown_bps=0,  # Buydown handled separately if needed
        tdc=sources_uses.tdc,
        ltc_max=perm_ltc_max,
        max_loan_cap=sources_uses.construction_loan if cap_perm_at_construction else None,
    )
    perm_loan_max = perm_loan_result.loan_amount

    # Trace perm loan
    trace("perm_loan.amount", perm_loan_max, {
        "investment.stabilized_value": property_value_for_ltv,
        "inputs.perm_ltv_max": perm_ltv_max,
        "operations.stabilized_noi": stabilized_annual_noi,
        "inputs.perm_dscr_min": perm_dscr_min,
    }, notes=f"Binding constraint: {perm_loan_result.binding_constraint}")

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
    fixed_perm_pmt = 0.0  # Set at perm loan origination, reused for all periods

    # Track NOI history for trailing NOI calculation at reversion
    noi_history: List[float] = []

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
        # Construction loan remains active through lease-up
        # Perm loan takes out construction at stabilization (leaseup_end + 1)
        if period <= leaseup_end:
            loan_type = LoanType.CONSTRUCTION if period > predev_end else LoanType.NONE
        elif period == leaseup_end + 1:
            # Conversion period - perm takes out construction
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

        # Development section (from draw schedule during predev/construction/leaseup)
        # Draw schedule now extends through lease-up because construction loan
        # remains outstanding with interest accruing until perm takeout
        if period <= leaseup_end:
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
            # No development draws after lease-up (perm loan has taken out construction)
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

        # GPR - using actual values from unit allocations, with escalation
        # all_market_gpr_monthly = what GPR would be if all units were market rate
        # market_gpr_monthly = actual GPR from market-rate units
        # affordable_gpr_monthly = actual GPR from affordable units (at AMI rents)

        # Use all_market for the comparison row (if not provided, derive from total)
        base_all_market = all_market_gpr_monthly if all_market_gpr_monthly > 0 else (market_gpr_monthly + affordable_gpr_monthly)
        escalated_all_market = base_all_market * (1 + market_bump)

        # Actual GPR from each unit type
        escalated_market = market_gpr_monthly * (1 + market_bump)
        escalated_affordable = affordable_gpr_monthly * (1 + affordable_bump)

        # Total blended GPR for this scenario
        gpr_mixed_total = escalated_market + escalated_affordable

        gpr = GPRRow(
            gpr_all_market=escalated_all_market,  # Hypothetical: all units at market rent
            gpr_mixed_market=escalated_market,     # Actual: market units only
            gpr_mixed_affordable=escalated_affordable,  # Actual: affordable units only
            gpr_total=gpr_mixed_total,  # Total blended GPR
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
                less_management_fee=0,
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
            # GPR is always max potential rent (100% occupied)
            period_gpr = gpr.gpr_total

            # Vacancy represents total economic loss from GPR
            # - During leaseup: vacancy = 1 - cumulative_occupancy (unleased units)
            # - At stabilization: vacancy = 1 - max_occupancy (stabilized vacancy rate)
            # The vacancy_rate input represents the stabilized economic vacancy (e.g., 6%)
            # max_occupancy = 1 - vacancy_rate at stabilization
            total_vacancy_pct = 1.0 - cumulative_occupancy
            period_vacancy = period_gpr * total_vacancy_pct
            period_egi = period_gpr - period_vacancy

            # TIF reimbursement adds to NOI
            tif_reimb = tif_row.tif_reimbursement if tif_row else 0.0

            # Management fee as % of EGI
            period_mgmt_fee = period_egi * management_fee_pct

            # NOI = EGI - Management Fee - OpEx + TIF Reimbursement
            base_noi = period_egi - period_mgmt_fee - opex.total_opex
            adjusted_noi = base_noi + tif_reimb

            operations = OperationsRow(
                leaseup_pct=cumulative_occupancy,
                gpr=period_gpr,
                vacancy_rate=total_vacancy_pct,  # Includes unleased units + economic vacancy
                less_vacancy=period_vacancy,
                egi=period_egi,
                less_management_fee=period_mgmt_fee,
                less_opex_ex_taxes=opex.opex_ex_prop_taxes,
                less_property_taxes=opex.property_taxes,
                noi=adjusted_noi,
                noi_reserve_in_tdc=0,  # Tracked but not active
                tif=tif_row,
                tif_reimbursement=tif_reimb,
            )
            noi = operations.noi
            total_noi += noi
            noi_history.append(noi)

        # Investment (unlevered cash flow)
        dev_cost = -development.tdc_to_be_funded if period <= construction_end else 0
        reserves_amount = -operations.egi * reserves_pct if operations.egi > 0 else 0

        # Reversion
        if is_reversion:
            # Calculate terminal NOI based on reversion_noi_basis
            if reversion_noi_basis == "trailing" and len(noi_history) >= 12:
                # Use last 12 months of actual NOI (trailing twelve months)
                trailing_12_months = noi_history[-12:]
                terminal_noi = sum(trailing_12_months)
            else:
                # Forward NOI: use current period's NOI annualized (default)
                terminal_noi = noi * 12

            # Calculate sale price using exit cap rate
            sale_price_gross = terminal_noi / exit_cap_rate

            # Deduct selling costs (broker fees, legal, closing costs, etc.)
            selling_costs = sale_price_gross * selling_costs_pct
            sale_price_net = sale_price_gross - selling_costs

            reversion_amount = sale_price_net
            reversion_value = sale_price_net

            # Trace reversion calculation
            trace("investment.reversion", reversion_amount, {
                "operations.terminal_noi": terminal_noi,
                "inputs.exit_cap_rate": exit_cap_rate,
                "inputs.selling_costs_pct": selling_costs_pct,
            }, period=period, notes=f"basis={reversion_noi_basis}")
        else:
            reversion_amount = 0

        investment = InvestmentRow(
            dev_cost=dev_cost,
            reserves=reserves_amount,
            reversion=reversion_amount,
            unlevered_cf=dev_cost + noi + reserves_amount + reversion_amount,
        )

        # Construction debt
        # Construction loan remains outstanding through lease-up
        # Perm loan takes it out at stabilization (first month of operations)
        if period <= leaseup_end:
            draw = draw_schedule.get_period(period)
            const_debt = ConstructionDebtRow(
                principal_bop=draw.const_debt_bop,
                debt_added=draw.const_debt_draw,
                interest_in_period=draw.const_debt_interest,
                repaid=0,
                principal_eop=draw.const_debt_eop,
                net_cf=draw.const_debt_draw,  # Inflow from draws (0 during lease-up)
            )
        elif period == leaseup_end + 1:
            # Payoff construction loan - perm loan takes out at stabilization
            # Save balance before zeroing - this caps the perm loan (no cash-out refi)
            const_loan_payoff_amount = const_loan_balance
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
        # Perm loan originates at stabilization (after lease-up), takes out construction loan
        perm_monthly_rate = perm_rate / 12 if perm_rate > 0 else 0.0
        if period == leaseup_end + 1:
            # Perm loan funds — calculate fixed payment at origination
            if cap_perm_at_construction:
                # Cap at construction loan payoff (no cash-out refi)
                perm_loan_amount = min(perm_loan_max, const_loan_payoff_amount)
            else:
                # Allow perm to be sized by LTV/DSCR only (may exceed construction loan)
                perm_loan_amount = perm_loan_max
            perm_loan_balance = perm_loan_amount
            fixed_perm_pmt, _, _ = _calculate_monthly_pmt(perm_loan_amount, perm_rate, perm_amort_years)
            i_pmt = calculate_interest_portion(perm_loan_balance, perm_monthly_rate)
            p_pmt = calculate_principal_portion(fixed_perm_pmt, i_pmt)
            perm_debt = PermanentDebtRow(
                principal_bop=0,
                pmt_in_period=fixed_perm_pmt,
                interest_pmt=i_pmt,
                principal_pmt=p_pmt,
                payoff=0,
                principal_eop=perm_loan_amount - p_pmt,
                net_cf=perm_loan_amount - fixed_perm_pmt,  # Inflow from funding - first payment
            )
            perm_loan_balance = perm_loan_amount - p_pmt
            total_debt_service += fixed_perm_pmt
        elif period > leaseup_end + 1 and not is_reversion:
            # Regular perm loan payments — use fixed payment from origination
            i_pmt = calculate_interest_portion(perm_loan_balance, perm_monthly_rate)
            p_pmt = calculate_principal_portion(fixed_perm_pmt, i_pmt)
            perm_debt = PermanentDebtRow(
                principal_bop=perm_loan_balance,
                pmt_in_period=fixed_perm_pmt,
                interest_pmt=i_pmt,
                principal_pmt=p_pmt,
                payoff=0,
                principal_eop=perm_loan_balance - p_pmt,
                net_cf=-fixed_perm_pmt,  # Outflow for payment
            )
            perm_loan_balance -= p_pmt
            total_debt_service += fixed_perm_pmt
        elif is_reversion:
            # Payoff perm loan — use fixed payment for final period split
            i_pmt = calculate_interest_portion(perm_loan_balance, perm_monthly_rate)
            p_pmt = calculate_principal_portion(fixed_perm_pmt, i_pmt)
            perm_debt = PermanentDebtRow(
                principal_bop=perm_loan_balance,
                pmt_in_period=fixed_perm_pmt,
                interest_pmt=i_pmt,
                principal_pmt=p_pmt,
                payoff=-perm_loan_balance,
                principal_eop=0,
                net_cf=-fixed_perm_pmt - perm_loan_balance,  # Payment + payoff
            )
            total_debt_service += fixed_perm_pmt
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
                # Amortizing mezz - use proper amortization calculation
                n_months = mezzanine_amort_years * 12
                if mezz_monthly_rate > 0 and n_months > 0:
                    mezz_pmt = -npf.pmt(mezz_monthly_rate, n_months, mezzanine_amount)
                else:
                    mezz_pmt = mezz_interest
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
    # Adjust unlevered cash flows based on exclusion settings
    adjusted_unlevered_cfs = unlevered_cfs.copy()

    if exclude_land_from_irr:
        # Add back land cost from first predevelopment period
        # Land is recovered at sale, so it's not a true "cost" in IRR terms
        adjusted_unlevered_cfs[0] += land_cost

    if exclude_idc_from_irr:
        # Add back IDC spread over construction periods
        # IDC is a financing cost, not a development cost
        idc_per_period = sources_uses.idc / construction_months
        for i in range(predev_end, construction_end):
            adjusted_unlevered_cfs[i] += idc_per_period

    # Adjust levered cash flows for TIF lump sum
    # TIF lump sum is a front-loaded incentive that reduces equity requirement
    # Add as positive inflow in period 1 to offset equity outflow
    adjusted_levered_cfs = levered_cfs.copy()
    if tif_lump_sum_applied > 0:
        adjusted_levered_cfs[0] += tif_lump_sum_applied
        total_equity_invested -= tif_lump_sum_applied  # Reduce net equity

    try:
        monthly_unlevered_irr = npf.irr(adjusted_unlevered_cfs)
        unlevered_irr = (1 + monthly_unlevered_irr) ** 12 - 1
    except:
        unlevered_irr = 0.0

    try:
        monthly_levered_irr = npf.irr(adjusted_levered_cfs)
        levered_irr = (1 + monthly_levered_irr) ** 12 - 1
    except:
        levered_irr = 0.0

    # Trace key return metrics
    trace("returns.unlevered_irr", unlevered_irr, {
        "investment.unlevered_cf": sum(adjusted_unlevered_cfs),
    })
    trace("returns.levered_irr", levered_irr, {
        "investment.levered_cf": sum(adjusted_levered_cfs),
        "sources_uses.equity": total_equity_invested,
    })
    trace("returns.yield_on_cost", total_noi / sources_uses.tdc if sources_uses.tdc > 0 else 0, {
        "operations.total_noi": total_noi,
        "sources_uses.tdc": sources_uses.tdc,
    })

    # Get the current trace context to attach to result
    current_trace = TraceContext.current()

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
        trace_context=current_trace,
    )


def calculate_deal(
    inputs: ProjectInputs,
    scenario: Scenario,
    tif_lump_sum: float = 0.0,
    tif_treatment: str = "none",
    tif_abatement_pct: float = 0.0,
    tif_abatement_years: int = 0,
    tif_stream_pct: float = 0.0,
    tif_stream_years: int = 0,
    tif_start_delay_months: Optional[int] = None,  # None = derive from inputs.tif_start_timing
    mezzanine_amount: float = 0.0,
    mezzanine_rate: float = 0.12,
    preferred_amount: float = 0.0,
    preferred_return: float = 0.10,
) -> DetailedCashFlowResult:
    """Calculate a complete deal with proper GPR from unit allocations.

    This is the SINGLE SOURCE OF TRUTH for all financial calculations.
    All summary metrics are derived from the period-by-period calculations.

    Args:
        inputs: Project inputs including unit mix, rents, costs, timing
        scenario: MARKET or MIXED_INCOME
        tif_lump_sum: TIF lump sum amount (if applicable)
        tif_treatment: "none", "lump_sum", "abatement", or "stream"
        tif_abatement_pct: Percentage of city taxes abated (0-1)
        tif_abatement_years: Years of abatement
        tif_stream_pct: Percentage of city increment reimbursed (0-1)
        tif_stream_years: Years of TIF stream
        tif_start_delay_months: Delay from stabilization to TIF start
        mezzanine_amount: Mezzanine debt amount
        mezzanine_rate: Mezzanine interest rate
        preferred_amount: Preferred equity amount
        preferred_return: Preferred return rate

    Returns:
        DetailedCashFlowResult with all periods and derived summary metrics.
        The IRR, TDC, equity, and all other metrics are calculated FROM
        the period-by-period cash flows.
    """
    # Determine affordable percentage based on scenario
    if scenario == Scenario.MARKET:
        affordable_pct = 0.0
    else:
        affordable_pct = inputs.affordable_pct

    # Determine TIF start delay from inputs.tif_start_timing if not explicitly set
    if tif_start_delay_months is None:
        if inputs.tif_start_timing == TIFStartTiming.LEASEUP:
            # TIF starts at construction end (beginning of leaseup)
            # Relative to stabilization, this is -leaseup_months
            tif_start_delay_months = -inputs.leaseup_months
        else:
            # OPERATIONS (default): TIF starts at stabilization
            tif_start_delay_months = 0

    # Calculate unit allocations with proper rent calculations
    allocations = allocate_units(
        target_units=inputs.target_units,
        unit_mix=inputs.unit_mix,
        affordable_pct=affordable_pct,
        ami_level=inputs.ami_level,
        market_rent_psf=inputs.market_rent_psf,
    )

    # Calculate GPR from allocations (proper AMI rent lookups)
    gpr_result = calculate_gpr(allocations)

    # Also calculate all-market GPR (for comparison display)
    # This is what GPR would be if all units were at market rents
    all_market_allocations = allocate_units(
        target_units=inputs.target_units,
        unit_mix=inputs.unit_mix,
        affordable_pct=0.0,  # All market
        ami_level=inputs.ami_level,
        market_rent_psf=inputs.market_rent_psf,
    )
    all_market_gpr = calculate_gpr(all_market_allocations)

    # Calculate hard costs WITH contingency
    hard_costs_base = inputs.target_units * inputs.hard_cost_per_unit
    hard_cost_contingency = hard_costs_base * inputs.hard_cost_contingency_pct
    hard_costs = hard_costs_base + hard_cost_contingency

    # Calculate effective soft cost percentage including:
    # - Base soft cost % (with contingency)
    # - Predevelopment costs (as % of hard costs)
    # - Developer fee (as % of hard costs)
    soft_cost_with_contingency = inputs.soft_cost_pct * (1 + inputs.soft_cost_contingency_pct)
    effective_soft_cost_pct = (
        soft_cost_with_contingency +
        inputs.predevelopment_cost_pct +
        inputs.developer_fee_pct
    )

    # Calculate breakdown dollar amounts for display
    soft_cost_base = hard_costs * inputs.soft_cost_pct
    soft_cost_contingency = soft_cost_base * inputs.soft_cost_contingency_pct
    predevelopment_costs = hard_costs * inputs.predevelopment_cost_pct
    developer_fee = hard_costs * inputs.developer_fee_pct

    # Calculate annual opex
    annual_opex_per_unit = (
        inputs.opex_utilities +
        inputs.opex_maintenance +
        inputs.opex_misc
    )

    # Get tax stack from inputs (uses input tax rates or defaults to Austin stack)
    if inputs.tax_rates:
        tax_stack = build_tax_stack_from_rates(inputs.tax_rates)
    else:
        tax_stack = get_austin_tax_stack()

    # Call the detailed cash flow generator with proper GPR values
    # Wrap in TraceContext to capture all calculation traces
    with TraceContext() as ctx:
        result = generate_detailed_cash_flow(
            # Sources & Uses inputs
            land_cost=inputs.land_cost,
            hard_costs=hard_costs,
            soft_cost_pct=effective_soft_cost_pct,
            construction_ltc=inputs.construction_ltc,
            construction_rate=inputs.construction_rate,

            # Timing
            start_date=inputs.predevelopment_start,
            predevelopment_months=inputs.predevelopment_months,
            construction_months=inputs.construction_months,
            idc_leaseup_months=inputs.idc_leaseup_months,
            leaseup_months=inputs.leaseup_months,
            operations_months=inputs.operations_months,

            # Revenue - PROPER GPR from unit allocations
            market_gpr_monthly=gpr_result.market_gpr_monthly,
            affordable_gpr_monthly=gpr_result.affordable_gpr_monthly,
            all_market_gpr_monthly=all_market_gpr.total_gpr_monthly,
            vacancy_rate=inputs.vacancy_rate,
            leaseup_pace=inputs.leaseup_pace,
            max_occupancy=inputs.max_occupancy,

            # Operating expenses
            annual_opex_per_unit=annual_opex_per_unit,
            total_units=inputs.target_units,

            # Property tax assessed value (required)
            existing_assessed_value=inputs.existing_assessed_value,

            # === Optional parameters with defaults ===
            # Management fee
            management_fee_pct=inputs.opex_management_pct,

            # Reserves
            operating_reserve_months=inputs.operating_reserve_months,
            leaseup_reserve_months=inputs.leaseup_reserve_months,

            # Property tax options
            tax_stack=tax_stack,
            assessment_growth_rate=inputs.property_tax_growth,

            # Escalation
            market_rent_growth=inputs.market_rent_growth,
            opex_growth=inputs.opex_growth,
            prop_tax_growth=inputs.property_tax_growth,

            # Permanent loan
            perm_rate=inputs.perm_rate,
            perm_amort_years=inputs.perm_amort_years,
            perm_ltv_max=inputs.perm_ltv_max,
            perm_dscr_min=inputs.perm_dscr_min,

            # Exit
            exit_cap_rate=inputs.exit_cap_rate,
            reserves_pct=inputs.reserves_pct,
            selling_costs_pct=inputs.selling_costs_pct,
            reversion_noi_basis=inputs.reversion_noi_basis,

            # Mixed income
            affordable_pct=affordable_pct,
            affordable_rent_growth=inputs.affordable_rent_growth,

            # TIF treatment
            tif_treatment=tif_treatment,
            tif_lump_sum=tif_lump_sum,
            tif_abatement_pct=tif_abatement_pct,
            tif_abatement_years=tif_abatement_years,
            tif_stream_pct=tif_stream_pct,
            tif_stream_years=tif_stream_years,
            tif_start_delay_months=tif_start_delay_months,

            # Additional capital
            mezzanine_amount=mezzanine_amount,
            mezzanine_rate=mezzanine_rate,
            preferred_amount=preferred_amount,
            preferred_return=preferred_return,

            # Detailed breakdown for display
            hard_cost_contingency=hard_cost_contingency,
            soft_cost_contingency=soft_cost_contingency,
            predevelopment_costs=predevelopment_costs,
            developer_fee=developer_fee,
        )
    return result
