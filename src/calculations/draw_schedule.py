"""Draw schedule engine with equity-first logic and S-curves."""

from dataclasses import dataclass, field
from typing import List, Literal
from enum import Enum
import math


class Phase(str, Enum):
    """Development phase."""
    PREDEVELOPMENT = "predevelopment"
    CONSTRUCTION = "construction"
    LEASE_UP = "lease_up"
    OPERATIONS = "operations"
    REVERSION = "reversion"


@dataclass
class PeriodDraw:
    """Single period in the draw schedule.

    Tracks TDC draws, equity, and debt for one month.
    """
    # Period info
    period: int  # 1-indexed period number
    phase: Phase

    # TDC tracking
    tdc_total: float  # Total TDC (static)
    tdc_bop: float  # TDC remaining at beginning of period
    tdc_draw_pct: float  # % of TDC drawn this period
    tdc_draw_predev: float  # $ drawn for predevelopment
    tdc_draw_construction: float  # $ drawn for construction
    tdc_draw_total: float  # Total $ drawn this period
    tdc_eop: float  # TDC remaining at end of period

    # Equity tracking
    equity_total: float  # Total equity commitment (static)
    equity_bop: float  # Equity remaining at beginning of period
    equity_draw: float  # Equity drawn this period (negative = outflow)
    equity_eop: float  # Equity remaining at end of period

    # Construction debt tracking
    const_debt_total: float  # Total construction loan commitment
    const_debt_bop: float  # Loan balance at beginning of period
    const_debt_draw: float  # Loan drawn this period
    const_debt_interest: float  # Interest accrued this period
    const_debt_eop: float  # Loan balance at end of period (incl. capitalized interest)

    # Flags
    is_equity_phase: bool  # True if equity is still being drawn
    is_debt_phase: bool  # True if debt is being drawn


@dataclass
class DrawSchedule:
    """Complete draw schedule for development.

    Contains period-by-period draws for the entire development timeline.
    """
    periods: List[PeriodDraw]
    total_periods: int

    # Summary totals
    total_tdc: float
    total_equity: float
    total_construction_debt: float
    total_idc_actual: float  # Actual IDC from period-by-period calc

    # Phase boundaries
    predevelopment_end: int  # Last period of predevelopment
    construction_end: int  # Last period of construction

    def get_period(self, period: int) -> PeriodDraw:
        """Get draw info for a specific period (1-indexed)."""
        if period < 1 or period > len(self.periods):
            raise IndexError(f"Period {period} out of range [1, {len(self.periods)}]")
        return self.periods[period - 1]

    def get_phase_periods(self, phase: Phase) -> List[PeriodDraw]:
        """Get all periods for a given phase."""
        return [p for p in self.periods if p.phase == phase]


def _s_curve_weights(n_periods: int, steepness: float = 4.0) -> List[float]:
    """Generate S-curve weights for draw schedule.

    Uses a logistic function to create S-curve distribution.
    Weights sum to 1.0.

    Args:
        n_periods: Number of periods
        steepness: Controls how steep the S-curve is (higher = steeper)

    Returns:
        List of weights that sum to 1.0
    """
    if n_periods <= 0:
        return []
    if n_periods == 1:
        return [1.0]

    # Generate logistic S-curve cumulative values
    cumulative = []
    for i in range(n_periods + 1):
        # Map i to range [-steepness, +steepness]
        x = steepness * (2 * i / n_periods - 1)
        # Logistic function
        cumulative.append(1 / (1 + math.exp(-x)))

    # Convert cumulative to period weights
    weights = []
    for i in range(n_periods):
        weight = cumulative[i + 1] - cumulative[i]
        weights.append(weight)

    # Normalize to sum to 1.0
    total = sum(weights)
    if total > 0:
        weights = [w / total for w in weights]

    return weights


def _flat_weights(n_periods: int) -> List[float]:
    """Generate flat (linear) weights.

    Each period gets equal weight.

    Args:
        n_periods: Number of periods

    Returns:
        List of equal weights that sum to 1.0
    """
    if n_periods <= 0:
        return []
    return [1.0 / n_periods] * n_periods


def generate_draw_schedule(
    tdc: float,
    equity: float,
    construction_loan: float,
    predevelopment_months: int,
    construction_months: int,
    construction_rate: float,
    predev_curve: Literal["flat", "s_curve"] = "s_curve",
    construction_curve: Literal["flat", "s_curve"] = "s_curve",
) -> DrawSchedule:
    """Generate complete draw schedule for development.

    Key logic:
    1. Equity draws FIRST until exhausted
    2. Then construction loan draws begin
    3. IDC capitalizes each period based on loan balance
    4. S-curves applied to predevelopment and construction draws

    Args:
        tdc: Total development cost
        equity: Total equity commitment
        construction_loan: Total construction loan commitment
        predevelopment_months: Number of months of predevelopment
        construction_months: Number of months of construction
        construction_rate: Annual interest rate on construction loan
        predev_curve: Draw curve type for predevelopment
        construction_curve: Draw curve type for construction

    Returns:
        DrawSchedule with period-by-period draws
    """
    monthly_rate = construction_rate / 12
    total_periods = predevelopment_months + construction_months

    # Determine how much TDC is drawn in each phase
    # Predevelopment typically covers land, soft costs, some hard costs
    # For simplicity, assume predevelopment = (land + soft costs) / TDC
    # But we don't have that breakdown here, so use a heuristic:
    # Predevelopment = ~15% of TDC (typical for soft costs + land deposits)
    predev_pct = 0.15
    construction_pct = 0.85

    predev_tdc = tdc * predev_pct
    construction_tdc = tdc * construction_pct

    # Generate draw weights
    if predev_curve == "s_curve":
        predev_weights = _s_curve_weights(predevelopment_months)
    else:
        predev_weights = _flat_weights(predevelopment_months)

    if construction_curve == "s_curve":
        construction_weights = _s_curve_weights(construction_months)
    else:
        construction_weights = _flat_weights(construction_months)

    periods: List[PeriodDraw] = []

    # Track running balances
    tdc_remaining = tdc
    equity_remaining = equity
    const_debt_balance = 0.0
    total_idc = 0.0

    period_num = 0

    # Predevelopment periods
    for i, weight in enumerate(predev_weights):
        period_num += 1
        draw_amount = predev_tdc * weight

        # TDC tracking
        tdc_bop = tdc_remaining
        tdc_draw = draw_amount
        tdc_remaining -= draw_amount
        tdc_eop = tdc_remaining

        # Equity draws first (all predevelopment is typically equity)
        equity_bop = equity_remaining
        equity_draw = -min(draw_amount, equity_remaining)  # Negative = outflow
        equity_remaining = max(0, equity_remaining - draw_amount)
        equity_eop = equity_remaining

        # No construction debt during predevelopment
        const_debt_bop = const_debt_balance
        const_debt_draw = 0.0
        const_debt_interest = 0.0
        const_debt_eop = const_debt_balance

        periods.append(PeriodDraw(
            period=period_num,
            phase=Phase.PREDEVELOPMENT,
            tdc_total=tdc,
            tdc_bop=tdc_bop,
            tdc_draw_pct=weight * predev_pct,
            tdc_draw_predev=draw_amount,
            tdc_draw_construction=0.0,
            tdc_draw_total=draw_amount,
            tdc_eop=tdc_eop,
            equity_total=equity,
            equity_bop=equity_bop,
            equity_draw=equity_draw,
            equity_eop=equity_eop,
            const_debt_total=construction_loan,
            const_debt_bop=const_debt_bop,
            const_debt_draw=const_debt_draw,
            const_debt_interest=const_debt_interest,
            const_debt_eop=const_debt_eop,
            is_equity_phase=equity_remaining > 0,
            is_debt_phase=False,
        ))

    # Construction periods
    for i, weight in enumerate(construction_weights):
        period_num += 1
        draw_amount = construction_tdc * weight

        # TDC tracking
        tdc_bop = tdc_remaining
        tdc_draw = draw_amount
        tdc_remaining -= draw_amount
        tdc_eop = tdc_remaining

        # Determine equity vs debt funding
        equity_bop = equity_remaining
        const_debt_bop = const_debt_balance

        if equity_remaining > 0:
            # Still have equity to draw
            equity_portion = min(draw_amount, equity_remaining)
            debt_portion = draw_amount - equity_portion
        else:
            # Equity exhausted, all debt
            equity_portion = 0.0
            debt_portion = draw_amount

        equity_draw = -equity_portion  # Negative = outflow
        equity_remaining = max(0, equity_remaining - equity_portion)
        equity_eop = equity_remaining

        # Construction debt draw
        const_debt_draw = debt_portion

        # Interest accrues on beginning balance + half of new draw (mid-period convention)
        interest_base = const_debt_bop + (const_debt_draw / 2)
        const_debt_interest = interest_base * monthly_rate
        total_idc += const_debt_interest

        # End balance includes draw + capitalized interest
        const_debt_balance = const_debt_bop + const_debt_draw + const_debt_interest
        const_debt_eop = const_debt_balance

        periods.append(PeriodDraw(
            period=period_num,
            phase=Phase.CONSTRUCTION,
            tdc_total=tdc,
            tdc_bop=tdc_bop,
            tdc_draw_pct=weight * construction_pct,
            tdc_draw_predev=0.0,
            tdc_draw_construction=draw_amount,
            tdc_draw_total=draw_amount,
            tdc_eop=tdc_eop,
            equity_total=equity,
            equity_bop=equity_bop,
            equity_draw=equity_draw,
            equity_eop=equity_eop,
            const_debt_total=construction_loan,
            const_debt_bop=const_debt_bop,
            const_debt_draw=const_debt_draw,
            const_debt_interest=const_debt_interest,
            const_debt_eop=const_debt_eop,
            is_equity_phase=equity_remaining > 0,
            is_debt_phase=const_debt_draw > 0,
        ))

    return DrawSchedule(
        periods=periods,
        total_periods=total_periods,
        total_tdc=tdc,
        total_equity=equity,
        total_construction_debt=construction_loan,
        total_idc_actual=total_idc,
        predevelopment_end=predevelopment_months,
        construction_end=predevelopment_months + construction_months,
    )


def generate_draw_schedule_from_sources_uses(
    sources_uses: "SourcesUses",
    predevelopment_months: int,
    construction_months: int,
    construction_rate: float,
    land_draw_period: int = 1,
    predev_curve: Literal["flat", "s_curve"] = "s_curve",
    construction_curve: Literal["flat", "s_curve"] = "s_curve",
) -> DrawSchedule:
    """Generate draw schedule from a SourcesUses object.

    This version uses the actual cost breakdown from SourcesUses
    to determine predevelopment vs construction draws.

    Args:
        sources_uses: SourcesUses object with capital stack
        predevelopment_months: Number of months of predevelopment
        construction_months: Number of months of construction
        construction_rate: Annual interest rate on construction loan
        land_draw_period: Period in which land is acquired (1 = first period)
        predev_curve: Draw curve type for predevelopment (excluding land)
        construction_curve: Draw curve type for construction

    Returns:
        DrawSchedule with period-by-period draws
    """
    # Import here to avoid circular dependency
    from src.calculations.sources_uses import SourcesUses

    monthly_rate = construction_rate / 12
    total_periods = predevelopment_months + construction_months

    # Predevelopment costs: land + soft costs
    # Land is typically drawn in period 1 (closing)
    # Soft costs are spread across predevelopment
    predev_land = sources_uses.land
    predev_soft = sources_uses.soft_costs
    predev_total = predev_land + predev_soft

    # Construction costs: hard costs + IDC
    construction_total = sources_uses.hard_costs + sources_uses.idc

    # Generate draw weights for soft costs during predevelopment
    if predev_curve == "s_curve":
        soft_cost_weights = _s_curve_weights(predevelopment_months)
    else:
        soft_cost_weights = _flat_weights(predevelopment_months)

    # Generate draw weights for construction
    if construction_curve == "s_curve":
        construction_weights = _s_curve_weights(construction_months)
    else:
        construction_weights = _flat_weights(construction_months)

    periods: List[PeriodDraw] = []

    # Track running balances
    tdc = sources_uses.tdc
    tdc_remaining = tdc
    equity = sources_uses.equity
    equity_remaining = equity
    construction_loan = sources_uses.construction_loan
    const_debt_balance = 0.0
    total_idc_actual = 0.0

    period_num = 0

    # Predevelopment periods
    for i in range(predevelopment_months):
        period_num += 1

        # Calculate draw amount
        soft_draw = predev_soft * soft_cost_weights[i]

        # Add land in the specified period
        if period_num == land_draw_period:
            draw_amount = predev_land + soft_draw
        else:
            draw_amount = soft_draw

        # TDC tracking
        tdc_bop = tdc_remaining
        tdc_remaining -= draw_amount
        tdc_eop = tdc_remaining

        # All predevelopment is equity
        equity_bop = equity_remaining
        equity_draw = -min(draw_amount, equity_remaining)
        equity_remaining = max(0, equity_remaining - draw_amount)
        equity_eop = equity_remaining

        # No construction debt during predevelopment
        const_debt_bop = const_debt_balance
        const_debt_draw = 0.0
        const_debt_interest = 0.0
        const_debt_eop = const_debt_balance

        periods.append(PeriodDraw(
            period=period_num,
            phase=Phase.PREDEVELOPMENT,
            tdc_total=tdc,
            tdc_bop=tdc_bop,
            tdc_draw_pct=draw_amount / tdc if tdc > 0 else 0,
            tdc_draw_predev=draw_amount,
            tdc_draw_construction=0.0,
            tdc_draw_total=draw_amount,
            tdc_eop=tdc_eop,
            equity_total=equity,
            equity_bop=equity_bop,
            equity_draw=equity_draw,
            equity_eop=equity_eop,
            const_debt_total=construction_loan,
            const_debt_bop=const_debt_bop,
            const_debt_draw=const_debt_draw,
            const_debt_interest=const_debt_interest,
            const_debt_eop=const_debt_eop,
            is_equity_phase=equity_remaining > 0,
            is_debt_phase=False,
        ))

    # Construction periods
    # Note: We draw hard_costs over construction, but IDC is calculated dynamically
    for i in range(construction_months):
        period_num += 1

        # Hard cost draw this period
        hard_cost_draw = sources_uses.hard_costs * construction_weights[i]

        # TDC tracking (IDC will be added as it accrues)
        tdc_bop = tdc_remaining

        # Determine equity vs debt funding for hard costs
        equity_bop = equity_remaining
        const_debt_bop = const_debt_balance

        if equity_remaining > 0:
            equity_portion = min(hard_cost_draw, equity_remaining)
            debt_portion = hard_cost_draw - equity_portion
        else:
            equity_portion = 0.0
            debt_portion = hard_cost_draw

        equity_draw = -equity_portion
        equity_remaining = max(0, equity_remaining - equity_portion)
        equity_eop = equity_remaining

        # Construction debt draw
        const_debt_draw = debt_portion

        # Interest accrues on beginning balance + half of new draw
        interest_base = const_debt_bop + (const_debt_draw / 2)
        const_debt_interest = interest_base * monthly_rate
        total_idc_actual += const_debt_interest

        # IDC is also funded - it's part of TDC
        # IDC is funded by debt (it's essentially the loan funding its own interest)
        # This increases the loan balance
        const_debt_balance = const_debt_bop + const_debt_draw + const_debt_interest
        const_debt_eop = const_debt_balance

        # Total draw for TDC tracking includes both hard costs and IDC
        total_draw = hard_cost_draw + const_debt_interest
        tdc_remaining -= total_draw
        tdc_eop = tdc_remaining

        periods.append(PeriodDraw(
            period=period_num,
            phase=Phase.CONSTRUCTION,
            tdc_total=tdc,
            tdc_bop=tdc_bop,
            tdc_draw_pct=total_draw / tdc if tdc > 0 else 0,
            tdc_draw_predev=0.0,
            tdc_draw_construction=total_draw,
            tdc_draw_total=total_draw,
            tdc_eop=tdc_eop,
            equity_total=equity,
            equity_bop=equity_bop,
            equity_draw=equity_draw,
            equity_eop=equity_eop,
            const_debt_total=construction_loan,
            const_debt_bop=const_debt_bop,
            const_debt_draw=const_debt_draw,
            const_debt_interest=const_debt_interest,
            const_debt_eop=const_debt_eop,
            is_equity_phase=equity_remaining > 0,
            is_debt_phase=const_debt_draw > 0 or const_debt_interest > 0,
        ))

    return DrawSchedule(
        periods=periods,
        total_periods=total_periods,
        total_tdc=tdc,
        total_equity=equity,
        total_construction_debt=construction_loan,
        total_idc_actual=total_idc_actual,
        predevelopment_end=predevelopment_months,
        construction_end=predevelopment_months + construction_months,
    )
