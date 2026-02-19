"""Property tax engine with taxing authority stack and TIF increment calculation."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


@dataclass
class TaxingAuthority:
    """A single taxing jurisdiction.

    Attributes:
        name: Name of the taxing authority
        code: Short code for the authority
        rate_per_100: Tax rate per $100 of assessed value
        participates_in_tif: Whether this authority participates in TIF
    """
    name: str
    code: str
    rate_per_100: float
    participates_in_tif: bool = False

    @property
    def rate_decimal(self) -> float:
        """Tax rate as decimal (e.g., 0.02 for 2%)."""
        return self.rate_per_100 / 100


@dataclass
class TaxingAuthorityStack:
    """Complete stack of taxing authorities for a property.

    This represents all the jurisdictions that levy property taxes
    on a given property.
    """
    authorities: List[TaxingAuthority]

    @property
    def total_rate_per_100(self) -> float:
        """Total tax rate per $100 across all authorities."""
        return sum(a.rate_per_100 for a in self.authorities)

    @property
    def total_rate_decimal(self) -> float:
        """Total tax rate as decimal."""
        return self.total_rate_per_100 / 100

    @property
    def tif_participating_rate_per_100(self) -> float:
        """Tax rate per $100 for TIF-participating authorities only."""
        return sum(a.rate_per_100 for a in self.authorities if a.participates_in_tif)

    @property
    def tif_participating_rate_decimal(self) -> float:
        """TIF-participating rate as decimal."""
        return self.tif_participating_rate_per_100 / 100

    @property
    def non_tif_rate_per_100(self) -> float:
        """Tax rate per $100 for non-TIF authorities."""
        return sum(a.rate_per_100 for a in self.authorities if not a.participates_in_tif)

    def calculate_tax(self, assessed_value: float) -> float:
        """Calculate total property tax for an assessed value."""
        return assessed_value * self.total_rate_decimal

    def calculate_tax_by_authority(self, assessed_value: float) -> Dict[str, float]:
        """Calculate property tax breakdown by authority."""
        return {
            a.code: assessed_value * a.rate_decimal
            for a in self.authorities
        }

    def set_tif_participation(self, authority_codes: List[str]) -> None:
        """Set which authorities participate in TIF.

        Args:
            authority_codes: List of authority codes that participate
        """
        for a in self.authorities:
            a.participates_in_tif = a.code in authority_codes


# Default Austin taxing authority stack
def get_austin_tax_stack() -> TaxingAuthorityStack:
    """Get the default Austin taxing authority stack.

    Rates are based on 2024/2025 Austin rates.
    Rates are per $100 of assessed value.
    Total rate: approximately 2.0764%
    """
    return TaxingAuthorityStack(
        authorities=[
            TaxingAuthority("City of Austin", "COA", 0.5740, participates_in_tif=True),
            TaxingAuthority("Travis County", "TC", 0.3758, participates_in_tif=False),
            TaxingAuthority("Austin ISD", "AISD", 0.9252, participates_in_tif=False),
            TaxingAuthority("Austin Community College", "ACC", 0.1034, participates_in_tif=False),
            TaxingAuthority("Central Health", "CH", 0.0980, participates_in_tif=False),
        ]
    )


def build_tax_stack_from_rates(tax_rates: Dict[str, float]) -> TaxingAuthorityStack:
    """Build a TaxingAuthorityStack from a tax_rates dictionary.

    The tax_rates dictionary uses simple keys ("city", "county", "isd", "hospital", "acc")
    which map to the Austin taxing authorities. Only "city" participates in TIF.

    Args:
        tax_rates: Dictionary with keys "city", "county", "isd", "hospital", "acc"
                   and values as rate per $100 of assessed value.

    Returns:
        TaxingAuthorityStack with authorities matching the provided rates.
    """
    # Map simple keys to authority names and codes
    authority_mapping = {
        "city": ("City of Austin", "COA", True),  # Only city participates in TIF
        "county": ("Travis County", "TC", False),
        "isd": ("Austin ISD", "AISD", False),
        "hospital": ("Central Health", "CH", False),
        "acc": ("Austin Community College", "ACC", False),
    }

    authorities = []
    for key, rate in tax_rates.items():
        if key in authority_mapping:
            name, code, participates_in_tif = authority_mapping[key]
            authorities.append(
                TaxingAuthority(name, code, rate, participates_in_tif=participates_in_tif)
            )

    # If no authorities found, return default Austin stack
    if not authorities:
        return get_austin_tax_stack()

    return TaxingAuthorityStack(authorities=authorities)


@dataclass
class PropertyTaxPeriod:
    """Property tax calculation for a single period.

    Tracks assessed value, taxes by authority, and TIF increment.
    """
    period: int
    year: int  # Calendar year

    # Assessed values
    baseline_assessed_value: float  # Frozen baseline for TIF
    current_assessed_value: float  # Current period assessed value

    # Tax calculations
    total_tax: float  # Total property tax due
    tax_by_authority: Dict[str, float]  # Breakdown by authority

    # TIF increment (if applicable)
    increment_assessed_value: float  # Current - Baseline
    increment_tax_nominal: float  # Tax on increment (participating authorities only)

    # Inflation adjustment
    inflation_factor: float  # Cumulative inflation since baseline
    increment_tax_real: float  # Increment tax in real (baseline) dollars


@dataclass
class PropertyTaxSchedule:
    """Complete property tax schedule over the analysis period.

    Includes period-by-period tax calculations and TIF increment analysis.
    """
    periods: List[PropertyTaxPeriod]
    tax_stack: TaxingAuthorityStack

    # Baseline info
    baseline_period: int  # Period when baseline was frozen
    baseline_assessed_value: float

    # Summary metrics
    total_taxes_paid: float
    total_increment_nominal: float
    total_increment_real: float

    # Present value calculations
    npv_increment_nominal: float
    npv_increment_real: float
    discount_rate: float

    def get_period(self, period: int) -> PropertyTaxPeriod:
        """Get tax info for a specific period (1-indexed)."""
        for p in self.periods:
            if p.period == period:
                return p
        raise IndexError(f"Period {period} not found")


class AssessedValueTiming(str, Enum):
    """When the assessed value steps up."""
    AT_CO = "at_co"  # Step up at certificate of occupancy
    AT_STABILIZATION = "at_stabilization"  # Step up at stabilization
    GRADUAL = "gradual"  # Gradual step-up during lease-up


def calculate_assessed_value_schedule(
    existing_value: float,
    stabilized_value: float,
    predevelopment_months: int,
    construction_months: int,
    leaseup_months: int,
    operations_months: int,
    timing: AssessedValueTiming = AssessedValueTiming.AT_CO,
    assessment_growth_rate: float = 0.02,
) -> List[float]:
    """Calculate assessed value for each period.

    Args:
        existing_value: Current assessed value of the land
        stabilized_value: Assessed value at stabilization (based on income approach)
        predevelopment_months: Months of predevelopment
        construction_months: Months of construction
        leaseup_months: Months of lease-up
        operations_months: Months of stabilized operations
        timing: When the assessed value steps up
        assessment_growth_rate: Annual growth rate for assessed value

    Returns:
        List of assessed values for each period
    """
    total_periods = (
        predevelopment_months + construction_months +
        leaseup_months + operations_months + 1  # +1 for reversion
    )

    values = []
    monthly_growth = (1 + assessment_growth_rate) ** (1/12) - 1

    co_period = predevelopment_months + construction_months + 1
    stabilization_period = co_period + leaseup_months

    for period in range(1, total_periods + 1):
        if period <= predevelopment_months + construction_months:
            # During predevelopment/construction: existing value with growth
            months_elapsed = period - 1
            value = existing_value * ((1 + monthly_growth) ** months_elapsed)

        elif timing == AssessedValueTiming.AT_CO:
            # Step up at CO
            if period >= co_period:
                months_since_co = period - co_period
                value = stabilized_value * ((1 + monthly_growth) ** months_since_co)
            else:
                months_elapsed = period - 1
                value = existing_value * ((1 + monthly_growth) ** months_elapsed)

        elif timing == AssessedValueTiming.AT_STABILIZATION:
            # Step up at stabilization
            if period >= stabilization_period:
                months_since_stab = period - stabilization_period
                value = stabilized_value * ((1 + monthly_growth) ** months_since_stab)
            else:
                months_elapsed = period - 1
                value = existing_value * ((1 + monthly_growth) ** months_elapsed)

        elif timing == AssessedValueTiming.GRADUAL:
            # Gradual step-up during lease-up
            if period < co_period:
                months_elapsed = period - 1
                value = existing_value * ((1 + monthly_growth) ** months_elapsed)
            elif period < stabilization_period:
                # Linear interpolation during lease-up
                progress = (period - co_period + 1) / leaseup_months
                base_existing = existing_value * ((1 + monthly_growth) ** (period - 1))
                base_stabilized = stabilized_value
                value = base_existing + progress * (base_stabilized - base_existing)
            else:
                months_since_stab = period - stabilization_period
                value = stabilized_value * ((1 + monthly_growth) ** months_since_stab)

        values.append(value)

    return values


def generate_property_tax_schedule(
    tax_stack: TaxingAuthorityStack,
    assessed_values: List[float],
    baseline_period: int,
    inflation_rate: float = 0.02,
    discount_rate: float = 0.06,
    start_year: int = 2026,
) -> PropertyTaxSchedule:
    """Generate complete property tax schedule with TIF analysis.

    Args:
        tax_stack: Taxing authority stack
        assessed_values: List of assessed values for each period
        baseline_period: Period when TIF baseline is frozen (1-indexed)
        inflation_rate: Annual inflation rate for real dollar conversion
        discount_rate: Annual discount rate for NPV calculations
        start_year: Calendar year of period 1

    Returns:
        PropertyTaxSchedule with period-by-period analysis
    """
    if baseline_period < 1 or baseline_period > len(assessed_values):
        raise ValueError(f"Baseline period {baseline_period} out of range")

    baseline_value = assessed_values[baseline_period - 1]
    monthly_inflation = (1 + inflation_rate) ** (1/12) - 1
    monthly_discount = (1 + discount_rate) ** (1/12) - 1

    periods: List[PropertyTaxPeriod] = []
    total_taxes = 0.0
    total_increment_nominal = 0.0
    total_increment_real = 0.0
    npv_increment_nominal = 0.0
    npv_increment_real = 0.0

    for i, assessed_value in enumerate(assessed_values):
        period = i + 1
        year = start_year + (period - 1) // 12

        # Calculate inflation factor since baseline
        months_since_baseline = max(0, period - baseline_period)
        inflation_factor = (1 + monthly_inflation) ** months_since_baseline

        # Calculate total tax
        total_tax = tax_stack.calculate_tax(assessed_value)
        tax_by_authority = tax_stack.calculate_tax_by_authority(assessed_value)

        # Calculate TIF increment
        increment_value = max(0, assessed_value - baseline_value)
        increment_tax_nominal = increment_value * tax_stack.tif_participating_rate_decimal

        # Convert to real dollars
        if inflation_factor > 0:
            increment_tax_real = increment_tax_nominal / inflation_factor
        else:
            increment_tax_real = increment_tax_nominal

        # Discount for NPV (discount from period to period 1)
        discount_factor = 1 / ((1 + monthly_discount) ** (period - 1))

        periods.append(PropertyTaxPeriod(
            period=period,
            year=year,
            baseline_assessed_value=baseline_value,
            current_assessed_value=assessed_value,
            total_tax=total_tax,
            tax_by_authority=tax_by_authority,
            increment_assessed_value=increment_value,
            increment_tax_nominal=increment_tax_nominal,
            inflation_factor=inflation_factor,
            increment_tax_real=increment_tax_real,
        ))

        # Accumulate totals
        total_taxes += total_tax
        total_increment_nominal += increment_tax_nominal
        total_increment_real += increment_tax_real
        npv_increment_nominal += increment_tax_nominal * discount_factor
        npv_increment_real += increment_tax_real * discount_factor

    return PropertyTaxSchedule(
        periods=periods,
        tax_stack=tax_stack,
        baseline_period=baseline_period,
        baseline_assessed_value=baseline_value,
        total_taxes_paid=total_taxes,
        total_increment_nominal=total_increment_nominal,
        total_increment_real=total_increment_real,
        npv_increment_nominal=npv_increment_nominal,
        npv_increment_real=npv_increment_real,
        discount_rate=discount_rate,
    )


# ============================================================================
# TIF as Amortizing Loan
# ============================================================================

@dataclass
class TIFLoanPayment:
    """Single payment in the TIF-as-loan schedule."""
    period: int
    date_offset_months: int

    principal_bop: float  # Beginning of period balance
    payment: float  # Total payment (from increment)
    interest: float  # Interest portion
    principal_paid: float  # Principal portion
    principal_eop: float  # End of period balance

    # PV tracking (discounted)
    pv_bop: float
    pv_payment: float
    pv_eop: float


@dataclass
class TIFLoanSchedule:
    """Complete TIF-as-amortizing-loan schedule.

    This treats the TIF lump sum as a loan from the city to the developer,
    where the developer's regular property tax payments serve as loan payments.
    """
    principal: float
    interest_rate: float  # Monthly rate
    annual_rate: float  # Annual rate (for display)

    payments: List[TIFLoanPayment]

    # Summary
    term_months: int  # Months to full repayment
    term_years: float  # Years to full repayment
    total_payments: float  # Sum of all payments (nominal)
    total_interest: float  # Sum of interest
    total_principal: float  # Sum of principal (should = original principal)

    total_pv: float  # Sum of PV of payments
    balloon_at_term: float  # Remaining balance if term limited


def calculate_tif_loan_schedule(
    principal: float,
    annual_rate: float,
    monthly_payment: float,
    max_term_months: int = 360,  # 30 years max
    escalation_rate: float = 0.015,
    escalation_frequency: int = 12,  # Escalate every 12 months
) -> TIFLoanSchedule:
    """Calculate TIF repayment as an amortizing loan schedule.

    The developer pays property taxes. The city's portion of the increment
    is treated as "loan payments" that repay the TIF lump sum over time.

    Args:
        principal: TIF lump sum amount (initial "loan" balance)
        annual_rate: Interest/discount rate (e.g., 0.03 for 3%)
        monthly_payment: Initial monthly increment payment
        max_term_months: Maximum term (stops schedule even if not fully paid)
        escalation_rate: Annual escalation of payment
        escalation_frequency: How often payments escalate (months)

    Returns:
        TIFLoanSchedule with complete payment history
    """
    monthly_rate = annual_rate / 12

    payments = []
    balance = principal
    pv_balance = principal

    current_payment = monthly_payment
    total_payments = 0.0
    total_interest = 0.0
    total_principal = 0.0
    total_pv = 0.0

    for period in range(1, max_term_months + 1):
        if balance <= 0.01:  # Effectively paid off
            break

        # Escalate payment annually
        if period > 1 and (period - 1) % escalation_frequency == 0:
            current_payment *= (1 + escalation_rate)

        # Interest for this period
        interest = balance * monthly_rate

        # Principal portion
        principal_payment = min(current_payment - interest, balance)
        if principal_payment < 0:
            # Payment doesn't cover interest - negative amortization
            principal_payment = 0

        actual_payment = interest + principal_payment

        # New balance
        new_balance = balance - principal_payment

        # PV calculations
        pv_factor = 1 / ((1 + monthly_rate) ** period)
        pv_payment = actual_payment * pv_factor
        new_pv_balance = pv_balance - pv_payment

        payment = TIFLoanPayment(
            period=period,
            date_offset_months=period,
            principal_bop=balance,
            payment=actual_payment,
            interest=interest,
            principal_paid=principal_payment,
            principal_eop=new_balance,
            pv_bop=pv_balance,
            pv_payment=pv_payment,
            pv_eop=max(0, new_pv_balance),
        )
        payments.append(payment)

        # Update tracking
        total_payments += actual_payment
        total_interest += interest
        total_principal += principal_payment
        total_pv += pv_payment

        balance = new_balance
        pv_balance = max(0, new_pv_balance)

    # Balloon at end of term
    balloon = balance if len(payments) == max_term_months else 0.0

    return TIFLoanSchedule(
        principal=principal,
        interest_rate=monthly_rate,
        annual_rate=annual_rate,
        payments=payments,
        term_months=len(payments),
        term_years=len(payments) / 12,
        total_payments=total_payments,
        total_interest=total_interest,
        total_principal=total_principal,
        total_pv=total_pv,
        balloon_at_term=balloon,
    )


def solve_tif_term(
    principal: float,
    annual_rate: float,
    monthly_payment: float,
    escalation_rate: float = 0.015,
) -> float:
    """Solve for the term (in months) to fully repay TIF.

    This is an iterative calculation.

    Returns:
        Number of months to full repayment, or infinity if payment < interest
    """
    monthly_rate = annual_rate / 12

    # Check if payment covers first month's interest
    first_interest = principal * monthly_rate
    if monthly_payment <= first_interest:
        return float('inf')

    # Iterative solve
    balance = principal
    current_payment = monthly_payment

    for month in range(1, 600):  # Max 50 years
        if balance <= 0.01:
            return month

        # Escalate annually
        if month > 1 and (month - 1) % 12 == 0:
            current_payment *= (1 + escalation_rate)

        interest = balance * monthly_rate
        principal_paid = current_payment - interest
        balance -= principal_paid

    return float('inf')


# ============================================================================
# SMART Fee Waivers
# ============================================================================

@dataclass
class SMARTFeeWaiver:
    """SMART Housing fee waiver calculation."""
    tier: int
    affordable_pct: float
    ami_level: float

    waiver_pct: float  # 1.0 = 100%, 0.5 = 50%, etc.
    per_unit_amount: float
    total_waiver: float
    affordable_units: int

    deferment_to_co: bool = True  # Deferred to Certificate of Occupancy


# SMART fee waiver configuration by tier
SMART_TIER_CONFIG = {
    1: {"affordable_pct": 0.05, "ami": 0.30, "waiver_pct": 1.0, "per_unit": 35000, "term": 10},
    2: {"affordable_pct": 0.20, "ami": 0.50, "waiver_pct": 0.5, "per_unit": 17500, "term": 20},
    3: {"affordable_pct": 0.10, "ami": 0.50, "waiver_pct": 0.25, "per_unit": 7500, "term": 20},
}


def calculate_smart_fee_waiver(
    tier: int,
    total_units: int,
    custom_affordable_pct: Optional[float] = None,
    custom_per_unit: Optional[float] = None,
) -> SMARTFeeWaiver:
    """Calculate SMART Housing fee waiver.

    Args:
        tier: SMART tier (1, 2, or 3)
        total_units: Total project units
        custom_affordable_pct: Override affordable percentage if provided
        custom_per_unit: Override per-unit amount if provided

    Returns:
        SMARTFeeWaiver with calculated amounts
    """
    if tier not in SMART_TIER_CONFIG:
        tier = 2  # Default to tier 2

    config = SMART_TIER_CONFIG[tier]
    affordable_pct = custom_affordable_pct if custom_affordable_pct else config["affordable_pct"]
    per_unit = custom_per_unit if custom_per_unit else config["per_unit"]

    affordable_units = int(total_units * affordable_pct)
    total_waiver = affordable_units * per_unit

    return SMARTFeeWaiver(
        tier=tier,
        affordable_pct=affordable_pct,
        ami_level=config["ami"],
        waiver_pct=config["waiver_pct"],
        per_unit_amount=per_unit,
        total_waiver=total_waiver,
        affordable_units=affordable_units,
    )


# ============================================================================
# City Tax Abatement
# ============================================================================

@dataclass
class CityTaxAbatement:
    """City property tax abatement for affordable units."""
    city_rate: float
    affordable_share: float
    affordable_assessed_value: float
    annual_city_tax_on_affordable: float
    abatement_pct: float
    annual_abatement: float
    term_years: int
    total_abatement: float
    monthly_abatement: float


def calculate_city_tax_abatement(
    assessed_value: float,
    base_assessed_value: float,
    abatement_pct: float,
    affordable_units: int,
    total_units: int,
    term_years: int,
    tax_stack: Optional[TaxingAuthorityStack] = None,
) -> CityTaxAbatement:
    """Calculate city property tax abatement for affordable units.

    Only the City of Austin portion is abated, and only for affordable units.

    Args:
        assessed_value: Total assessed value
        base_assessed_value: Base (pre-development) assessed value
        abatement_pct: Percentage of city tax abated (e.g., 1.0 = 100%)
        affordable_units: Number of affordable units
        total_units: Total project units
        term_years: Abatement term
        tax_stack: Tax rate configuration (uses Austin defaults if not provided)

    Returns:
        CityTaxAbatement with calculated amounts
    """
    if tax_stack is None:
        tax_stack = get_austin_tax_stack()

    # Get city rate
    city_rate = 0.0
    for auth in tax_stack.authorities:
        if auth.code == "COA":
            city_rate = auth.rate_decimal
            break

    # Calculate proportional value for affordable units
    affordable_share = affordable_units / total_units if total_units > 0 else 0
    affordable_assessed = assessed_value * affordable_share

    # Annual city tax on affordable portion
    annual_city_tax = affordable_assessed * city_rate

    # Annual abatement
    annual_abatement = annual_city_tax * abatement_pct

    # Total over term
    total_abatement = annual_abatement * term_years

    return CityTaxAbatement(
        city_rate=city_rate,
        affordable_share=affordable_share,
        affordable_assessed_value=affordable_assessed,
        annual_city_tax_on_affordable=annual_city_tax,
        abatement_pct=abatement_pct,
        annual_abatement=annual_abatement,
        term_years=term_years,
        total_abatement=total_abatement,
        monthly_abatement=annual_abatement / 12,
    )
