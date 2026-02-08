"""Monte Carlo simulation engine for DCF sensitivity analysis.

This module runs Monte Carlo simulations over the detailed cash flow model
to produce IRR distributions with confidence intervals. It enables:
- Probabilistic analysis of project returns
- Sensitivity identification across input parameters
- Risk assessment through confidence intervals

Typical usage:
    from src.calculations.monte_carlo import (
        MonteCarloConfig,
        InputDistribution,
        run_monte_carlo,
    )

    config = MonteCarloConfig(
        base_inputs=my_inputs,
        distributions=[
            InputDistribution("hard_costs", "normal", mean=25_000_000, std=2_500_000),
            InputDistribution("exit_cap_rate", "triangular", min=0.05, mode=0.055, max=0.065),
        ],
        n_iterations=1000,
    )

    result = run_monte_carlo(config)
    print(f"Mean Levered IRR: {result.levered_irr_mean:.2%}")
    print(f"95% CI: [{result.levered_irr_ci_lower:.2%}, {result.levered_irr_ci_upper:.2%}]")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal, Tuple, Any, Callable
from enum import Enum
import numpy as np
from datetime import date
import concurrent.futures
import multiprocessing

from src.calculations.detailed_cashflow import (
    generate_detailed_cash_flow,
    DetailedCashFlowResult,
    AssessedValueBasis,
)
from src.calculations.property_tax import get_austin_tax_stack, TaxingAuthorityStack


class DistributionType(str, Enum):
    """Supported probability distributions for inputs."""
    UNIFORM = "uniform"       # Equal probability between min and max
    NORMAL = "normal"         # Gaussian distribution
    TRIANGULAR = "triangular" # Triangular with mode (most likely value)
    LOGNORMAL = "lognormal"   # Lognormal (always positive, right-skewed)
    PERT = "pert"             # PERT/Beta distribution (weighted mode)


@dataclass
class InputDistribution:
    """Definition of a probability distribution for an input parameter.

    Attributes:
        parameter: Name of the DCF input parameter to vary
        distribution: Type of probability distribution
        min_value: Minimum value (for uniform, triangular, PERT)
        max_value: Maximum value (for uniform, triangular, PERT)
        mean: Mean value (for normal, lognormal)
        std: Standard deviation (for normal, lognormal)
        mode: Most likely value (for triangular, PERT)
        clip_min: Optional hard floor on sampled values
        clip_max: Optional hard ceiling on sampled values
    """
    parameter: str
    distribution: DistributionType

    # Distribution parameters
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    mode: Optional[float] = None

    # Clipping bounds
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None

    def sample(self, rng: np.random.Generator) -> float:
        """Draw a random sample from this distribution.

        Args:
            rng: NumPy random number generator

        Returns:
            Sampled value
        """
        if self.distribution == DistributionType.UNIFORM:
            value = rng.uniform(self.min_value, self.max_value)

        elif self.distribution == DistributionType.NORMAL:
            value = rng.normal(self.mean, self.std)

        elif self.distribution == DistributionType.TRIANGULAR:
            value = rng.triangular(self.min_value, self.mode, self.max_value)

        elif self.distribution == DistributionType.LOGNORMAL:
            # Convert mean/std to lognormal parameters
            # For lognormal, we need mu and sigma of the underlying normal
            m = self.mean
            s = self.std
            mu = np.log(m**2 / np.sqrt(s**2 + m**2))
            sigma = np.sqrt(np.log(1 + (s**2 / m**2)))
            value = rng.lognormal(mu, sigma)

        elif self.distribution == DistributionType.PERT:
            # PERT uses Beta distribution with weighted mode
            # Shape parameters from min, mode, max
            a = self.min_value
            b = self.mode
            c = self.max_value

            # Calculate mean (PERT weights mode 4x)
            pert_mean = (a + 4*b + c) / 6

            # Calculate alpha and beta for Beta distribution
            # Using standard PERT formula
            if c == a:
                value = b  # Degenerate case
            else:
                alpha = 1 + 4 * (b - a) / (c - a)
                beta = 1 + 4 * (c - b) / (c - a)
                # Sample from Beta and scale to [a, c]
                beta_sample = rng.beta(alpha, beta)
                value = a + beta_sample * (c - a)
        else:
            raise ValueError(f"Unknown distribution type: {self.distribution}")

        # Apply clipping if specified
        if self.clip_min is not None:
            value = max(value, self.clip_min)
        if self.clip_max is not None:
            value = min(value, self.clip_max)

        return value


@dataclass
class BaseInputs:
    """Base/default inputs for the DCF model.

    These values are used unless overridden by a distribution sample.
    """
    # Land & Construction
    land_cost: float = 2_000_000
    hard_costs: float = 25_000_000
    soft_cost_pct: float = 0.20
    construction_ltc: float = 0.65
    construction_rate: float = 0.07

    # Timing
    start_date: date = field(default_factory=lambda: date(2025, 1, 1))
    predevelopment_months: int = 6
    construction_months: int = 18
    leaseup_months: int = 6
    operations_months: int = 36

    # Revenue
    monthly_gpr_at_stabilization: float = 600_000
    vacancy_rate: float = 0.06
    leaseup_pace: float = 0.15
    max_occupancy: float = 0.94

    # Operating
    annual_opex_per_unit: float = 4_500
    total_units: int = 250

    # Property Tax
    existing_assessed_value: float = 1_000_000
    assessment_growth_rate: float = 0.02
    assessed_value_basis: AssessedValueBasis = AssessedValueBasis.TDC
    assessed_value_override: Optional[float] = None

    # Escalation
    market_rent_growth: float = 0.02
    opex_growth: float = 0.03
    prop_tax_growth: float = 0.02

    # Permanent Loan
    perm_rate: float = 0.06
    perm_amort_years: int = 20
    perm_ltv_max: float = 0.65
    perm_dscr_min: float = 1.25
    perm_ltc_max: Optional[float] = None

    # Exit
    exit_cap_rate: float = 0.055
    reserves_pct: float = 0.005

    # IRR Calculation
    exclude_land_from_irr: bool = True
    exclude_idc_from_irr: bool = True
    cap_perm_at_construction: bool = True

    # Mixed Income (optional)
    affordable_pct: float = 0.0
    affordable_rent_discount: float = 0.0
    affordable_rent_growth: float = 0.01

    # Mezzanine (optional)
    mezzanine_amount: float = 0.0
    mezzanine_rate: float = 0.12
    mezzanine_io: bool = True

    # Preferred (optional)
    preferred_amount: float = 0.0
    preferred_return: float = 0.10

    # TIF (optional)
    tif_treatment: str = "none"
    tif_lump_sum: float = 0.0
    tif_abatement_pct: float = 0.0
    tif_abatement_years: int = 0
    tif_stream_pct: float = 0.0
    tif_stream_years: int = 0
    tif_start_delay_months: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DCF function call."""
        return {
            "land_cost": self.land_cost,
            "hard_costs": self.hard_costs,
            "soft_cost_pct": self.soft_cost_pct,
            "construction_ltc": self.construction_ltc,
            "construction_rate": self.construction_rate,
            "start_date": self.start_date,
            "predevelopment_months": self.predevelopment_months,
            "construction_months": self.construction_months,
            "leaseup_months": self.leaseup_months,
            "operations_months": self.operations_months,
            "monthly_gpr_at_stabilization": self.monthly_gpr_at_stabilization,
            "vacancy_rate": self.vacancy_rate,
            "leaseup_pace": self.leaseup_pace,
            "max_occupancy": self.max_occupancy,
            "annual_opex_per_unit": self.annual_opex_per_unit,
            "total_units": self.total_units,
            "existing_assessed_value": self.existing_assessed_value,
            "assessment_growth_rate": self.assessment_growth_rate,
            "assessed_value_basis": self.assessed_value_basis,
            "assessed_value_override": self.assessed_value_override,
            "market_rent_growth": self.market_rent_growth,
            "opex_growth": self.opex_growth,
            "prop_tax_growth": self.prop_tax_growth,
            "perm_rate": self.perm_rate,
            "perm_amort_years": self.perm_amort_years,
            "perm_ltv_max": self.perm_ltv_max,
            "perm_dscr_min": self.perm_dscr_min,
            "perm_ltc_max": self.perm_ltc_max,
            "exit_cap_rate": self.exit_cap_rate,
            "reserves_pct": self.reserves_pct,
            "exclude_land_from_irr": self.exclude_land_from_irr,
            "exclude_idc_from_irr": self.exclude_idc_from_irr,
            "cap_perm_at_construction": self.cap_perm_at_construction,
            "affordable_pct": self.affordable_pct,
            "affordable_rent_discount": self.affordable_rent_discount,
            "affordable_rent_growth": self.affordable_rent_growth,
            "mezzanine_amount": self.mezzanine_amount,
            "mezzanine_rate": self.mezzanine_rate,
            "mezzanine_io": self.mezzanine_io,
            "preferred_amount": self.preferred_amount,
            "preferred_return": self.preferred_return,
            "tif_treatment": self.tif_treatment,
            "tif_lump_sum": self.tif_lump_sum,
            "tif_abatement_pct": self.tif_abatement_pct,
            "tif_abatement_years": self.tif_abatement_years,
            "tif_stream_pct": self.tif_stream_pct,
            "tif_stream_years": self.tif_stream_years,
            "tif_start_delay_months": self.tif_start_delay_months,
        }


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation.

    Attributes:
        base_inputs: Default input values for parameters not being varied
        distributions: List of distributions for parameters to vary
        n_iterations: Number of simulation iterations
        seed: Random seed for reproducibility
        confidence_level: Confidence level for intervals (default 0.95 = 95%)
        parallel: Whether to run iterations in parallel
        max_workers: Max parallel workers (None = CPU count)
    """
    base_inputs: BaseInputs
    distributions: List[InputDistribution]
    n_iterations: int = 1000
    seed: Optional[int] = None
    confidence_level: float = 0.95
    parallel: bool = True
    max_workers: Optional[int] = None


@dataclass
class IterationResult:
    """Result from a single Monte Carlo iteration."""
    iteration: int
    inputs: Dict[str, float]  # Sampled input values
    unlevered_irr: float
    levered_irr: float
    tdc: float
    equity: float
    noi: float
    reversion: float


@dataclass
class SensitivityResult:
    """Sensitivity analysis for a single input parameter."""
    parameter: str
    correlation_unlevered: float  # Correlation with unlevered IRR
    correlation_levered: float    # Correlation with levered IRR
    elasticity_unlevered: float   # % change in IRR per 1% change in input
    elasticity_levered: float


@dataclass
class MonteCarloResult:
    """Complete Monte Carlo simulation results.

    Contains all iteration results plus summary statistics.
    """
    # Configuration
    n_iterations: int
    confidence_level: float

    # All iteration results
    iterations: List[IterationResult]

    # Unlevered IRR statistics
    unlevered_irr_mean: float
    unlevered_irr_std: float
    unlevered_irr_median: float
    unlevered_irr_min: float
    unlevered_irr_max: float
    unlevered_irr_ci_lower: float  # Lower confidence bound
    unlevered_irr_ci_upper: float  # Upper confidence bound
    unlevered_irr_percentiles: Dict[int, float]  # 5th, 10th, 25th, 50th, 75th, 90th, 95th

    # Levered IRR statistics
    levered_irr_mean: float
    levered_irr_std: float
    levered_irr_median: float
    levered_irr_min: float
    levered_irr_max: float
    levered_irr_ci_lower: float
    levered_irr_ci_upper: float
    levered_irr_percentiles: Dict[int, float]

    # Sensitivity analysis
    sensitivities: List[SensitivityResult]

    # Probability metrics
    prob_unlevered_positive: float  # P(unlevered IRR > 0)
    prob_levered_positive: float    # P(levered IRR > 0)
    prob_unlevered_above_target: float  # P(unlevered IRR > target)
    prob_levered_above_target: float    # P(levered IRR > target)
    target_irr: float  # Target IRR for probability calculation

    def get_unlevered_irr_distribution(self) -> np.ndarray:
        """Get array of all unlevered IRR values."""
        return np.array([r.unlevered_irr for r in self.iterations])

    def get_levered_irr_distribution(self) -> np.ndarray:
        """Get array of all levered IRR values."""
        return np.array([r.levered_irr for r in self.iterations])

    def summary(self) -> str:
        """Return a formatted summary of results."""
        lines = [
            "=" * 60,
            "MONTE CARLO SIMULATION RESULTS",
            "=" * 60,
            f"Iterations: {self.n_iterations:,}",
            f"Confidence Level: {self.confidence_level:.0%}",
            "",
            "UNLEVERED IRR",
            "-" * 40,
            f"  Mean:   {self.unlevered_irr_mean:>8.2%}",
            f"  Median: {self.unlevered_irr_median:>8.2%}",
            f"  Std:    {self.unlevered_irr_std:>8.2%}",
            f"  Min:    {self.unlevered_irr_min:>8.2%}",
            f"  Max:    {self.unlevered_irr_max:>8.2%}",
            f"  {self.confidence_level:.0%} CI: [{self.unlevered_irr_ci_lower:>7.2%}, {self.unlevered_irr_ci_upper:>7.2%}]",
            "",
            "LEVERED IRR",
            "-" * 40,
            f"  Mean:   {self.levered_irr_mean:>8.2%}",
            f"  Median: {self.levered_irr_median:>8.2%}",
            f"  Std:    {self.levered_irr_std:>8.2%}",
            f"  Min:    {self.levered_irr_min:>8.2%}",
            f"  Max:    {self.levered_irr_max:>8.2%}",
            f"  {self.confidence_level:.0%} CI: [{self.levered_irr_ci_lower:>7.2%}, {self.levered_irr_ci_upper:>7.2%}]",
            "",
            "PROBABILITIES",
            "-" * 40,
            f"  P(Unlevered IRR > 0%):      {self.prob_unlevered_positive:>6.1%}",
            f"  P(Levered IRR > 0%):        {self.prob_levered_positive:>6.1%}",
            f"  P(Unlevered IRR > {self.target_irr:.0%}):    {self.prob_unlevered_above_target:>6.1%}",
            f"  P(Levered IRR > {self.target_irr:.0%}):      {self.prob_levered_above_target:>6.1%}",
            "",
            "SENSITIVITY (Correlation with Levered IRR)",
            "-" * 40,
        ]

        # Sort sensitivities by absolute correlation
        sorted_sens = sorted(
            self.sensitivities,
            key=lambda s: abs(s.correlation_levered),
            reverse=True
        )
        for s in sorted_sens[:10]:  # Top 10
            lines.append(f"  {s.parameter:<30} {s.correlation_levered:>+6.3f}")

        lines.append("=" * 60)
        return "\n".join(lines)


def _run_single_iteration(
    iteration: int,
    base_inputs: Dict[str, Any],
    distributions: List[InputDistribution],
    seed: int,
    tax_stack: TaxingAuthorityStack,
) -> IterationResult:
    """Run a single Monte Carlo iteration.

    Args:
        iteration: Iteration number
        base_inputs: Base input dictionary
        distributions: List of input distributions
        seed: Random seed for this iteration
        tax_stack: Tax stack to use

    Returns:
        IterationResult with sampled inputs and IRRs
    """
    # Create RNG for this iteration
    rng = np.random.default_rng(seed)

    # Start with base inputs
    inputs = base_inputs.copy()

    # Sample from distributions and override base inputs
    sampled = {}
    for dist in distributions:
        value = dist.sample(rng)
        sampled[dist.parameter] = value
        inputs[dist.parameter] = value

    # Run DCF
    try:
        result = generate_detailed_cash_flow(
            tax_stack=tax_stack,
            **inputs
        )

        return IterationResult(
            iteration=iteration,
            inputs=sampled,
            unlevered_irr=result.unlevered_irr,
            levered_irr=result.levered_irr,
            tdc=result.sources_uses.tdc,
            equity=result.total_equity_invested,
            noi=result.total_noi,
            reversion=result.reversion_value,
        )
    except Exception as e:
        # Return NaN for failed iterations
        return IterationResult(
            iteration=iteration,
            inputs=sampled,
            unlevered_irr=float('nan'),
            levered_irr=float('nan'),
            tdc=0,
            equity=0,
            noi=0,
            reversion=0,
        )


def run_monte_carlo(
    config: MonteCarloConfig,
    target_irr: float = 0.10,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> MonteCarloResult:
    """Run Monte Carlo simulation.

    Args:
        config: Simulation configuration
        target_irr: Target IRR for probability calculations
        progress_callback: Optional callback(completed, total) for progress updates

    Returns:
        MonteCarloResult with all statistics and iteration data
    """
    # Initialize
    base_inputs = config.base_inputs.to_dict()
    tax_stack = get_austin_tax_stack()

    # Create master RNG for reproducibility
    master_rng = np.random.default_rng(config.seed)

    # Generate seeds for each iteration
    iteration_seeds = master_rng.integers(0, 2**31, size=config.n_iterations)

    # Run iterations
    results: List[IterationResult] = []

    if config.parallel and config.n_iterations > 10:
        # Parallel execution
        max_workers = config.max_workers or min(multiprocessing.cpu_count(), 8)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _run_single_iteration,
                    i,
                    base_inputs,
                    config.distributions,
                    int(iteration_seeds[i]),
                    tax_stack,
                )
                for i in range(config.n_iterations)
            ]

            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                results.append(future.result())
                if progress_callback:
                    progress_callback(i + 1, config.n_iterations)
    else:
        # Sequential execution
        for i in range(config.n_iterations):
            result = _run_single_iteration(
                i,
                base_inputs,
                config.distributions,
                int(iteration_seeds[i]),
                tax_stack,
            )
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, config.n_iterations)

    # Sort by iteration number
    results.sort(key=lambda r: r.iteration)

    # Filter out failed iterations (NaN IRRs)
    valid_results = [r for r in results if not np.isnan(r.levered_irr)]

    if len(valid_results) == 0:
        raise ValueError("All iterations failed - check input distributions")

    # Calculate statistics
    unlevered_irrs = np.array([r.unlevered_irr for r in valid_results])
    levered_irrs = np.array([r.levered_irr for r in valid_results])

    # Percentiles
    percentile_levels = [5, 10, 25, 50, 75, 90, 95]

    unlevered_percentiles = {
        p: float(np.percentile(unlevered_irrs, p))
        for p in percentile_levels
    }
    levered_percentiles = {
        p: float(np.percentile(levered_irrs, p))
        for p in percentile_levels
    }

    # Confidence intervals
    alpha = 1 - config.confidence_level
    ci_lower = alpha / 2 * 100
    ci_upper = (1 - alpha / 2) * 100

    # Sensitivity analysis
    sensitivities = []
    for dist in config.distributions:
        param_values = np.array([r.inputs[dist.parameter] for r in valid_results])

        # Correlation
        if np.std(param_values) > 0:
            corr_unlevered = float(np.corrcoef(param_values, unlevered_irrs)[0, 1])
            corr_levered = float(np.corrcoef(param_values, levered_irrs)[0, 1])
        else:
            corr_unlevered = 0.0
            corr_levered = 0.0

        # Elasticity (approximate via regression)
        if np.std(param_values) > 0 and np.mean(param_values) != 0:
            # Normalized sensitivity
            param_mean = np.mean(param_values)
            irr_mean_unlev = np.mean(unlevered_irrs)
            irr_mean_lev = np.mean(levered_irrs)

            # Regression slope
            slope_unlev = np.cov(param_values, unlevered_irrs)[0, 1] / np.var(param_values)
            slope_lev = np.cov(param_values, levered_irrs)[0, 1] / np.var(param_values)

            # Elasticity = (dIRR/dParam) * (Param/IRR)
            elasticity_unlev = slope_unlev * param_mean / irr_mean_unlev if irr_mean_unlev != 0 else 0
            elasticity_lev = slope_lev * param_mean / irr_mean_lev if irr_mean_lev != 0 else 0
        else:
            elasticity_unlev = 0.0
            elasticity_lev = 0.0

        sensitivities.append(SensitivityResult(
            parameter=dist.parameter,
            correlation_unlevered=corr_unlevered,
            correlation_levered=corr_levered,
            elasticity_unlevered=elasticity_unlev,
            elasticity_levered=elasticity_lev,
        ))

    return MonteCarloResult(
        n_iterations=len(valid_results),
        confidence_level=config.confidence_level,
        iterations=results,

        # Unlevered stats
        unlevered_irr_mean=float(np.mean(unlevered_irrs)),
        unlevered_irr_std=float(np.std(unlevered_irrs)),
        unlevered_irr_median=float(np.median(unlevered_irrs)),
        unlevered_irr_min=float(np.min(unlevered_irrs)),
        unlevered_irr_max=float(np.max(unlevered_irrs)),
        unlevered_irr_ci_lower=float(np.percentile(unlevered_irrs, ci_lower)),
        unlevered_irr_ci_upper=float(np.percentile(unlevered_irrs, ci_upper)),
        unlevered_irr_percentiles=unlevered_percentiles,

        # Levered stats
        levered_irr_mean=float(np.mean(levered_irrs)),
        levered_irr_std=float(np.std(levered_irrs)),
        levered_irr_median=float(np.median(levered_irrs)),
        levered_irr_min=float(np.min(levered_irrs)),
        levered_irr_max=float(np.max(levered_irrs)),
        levered_irr_ci_lower=float(np.percentile(levered_irrs, ci_lower)),
        levered_irr_ci_upper=float(np.percentile(levered_irrs, ci_upper)),
        levered_irr_percentiles=levered_percentiles,

        # Sensitivities
        sensitivities=sensitivities,

        # Probabilities
        prob_unlevered_positive=float(np.mean(unlevered_irrs > 0)),
        prob_levered_positive=float(np.mean(levered_irrs > 0)),
        prob_unlevered_above_target=float(np.mean(unlevered_irrs > target_irr)),
        prob_levered_above_target=float(np.mean(levered_irrs > target_irr)),
        target_irr=target_irr,
    )


# Convenience functions for common distribution setups

def create_cost_uncertainty_distributions(
    base_hard_costs: float,
    hard_cost_cv: float = 0.10,  # Coefficient of variation (std/mean)
    soft_cost_range: Tuple[float, float] = (0.18, 0.25),
) -> List[InputDistribution]:
    """Create distributions for construction cost uncertainty.

    Args:
        base_hard_costs: Base hard cost estimate
        hard_cost_cv: Coefficient of variation for hard costs
        soft_cost_range: Min/max for soft cost percentage

    Returns:
        List of InputDistribution for cost parameters
    """
    return [
        InputDistribution(
            parameter="hard_costs",
            distribution=DistributionType.LOGNORMAL,
            mean=base_hard_costs,
            std=base_hard_costs * hard_cost_cv,
            clip_min=base_hard_costs * 0.8,
        ),
        InputDistribution(
            parameter="soft_cost_pct",
            distribution=DistributionType.TRIANGULAR,
            min_value=soft_cost_range[0],
            mode=(soft_cost_range[0] + soft_cost_range[1]) / 2,
            max_value=soft_cost_range[1],
        ),
    ]


def create_market_uncertainty_distributions(
    base_cap_rate: float = 0.055,
    cap_rate_range: Tuple[float, float] = (0.050, 0.065),
    rent_growth_range: Tuple[float, float] = (0.01, 0.04),
    vacancy_range: Tuple[float, float] = (0.04, 0.10),
) -> List[InputDistribution]:
    """Create distributions for market uncertainty.

    Args:
        base_cap_rate: Base exit cap rate
        cap_rate_range: Min/max for exit cap rate
        rent_growth_range: Min/max for rent growth
        vacancy_range: Min/max for vacancy rate

    Returns:
        List of InputDistribution for market parameters
    """
    return [
        InputDistribution(
            parameter="exit_cap_rate",
            distribution=DistributionType.TRIANGULAR,
            min_value=cap_rate_range[0],
            mode=base_cap_rate,
            max_value=cap_rate_range[1],
        ),
        InputDistribution(
            parameter="market_rent_growth",
            distribution=DistributionType.TRIANGULAR,
            min_value=rent_growth_range[0],
            mode=0.025,  # 2.5% most likely
            max_value=rent_growth_range[1],
        ),
        InputDistribution(
            parameter="vacancy_rate",
            distribution=DistributionType.TRIANGULAR,
            min_value=vacancy_range[0],
            mode=0.06,
            max_value=vacancy_range[1],
        ),
    ]


def create_financing_uncertainty_distributions(
    base_perm_rate: float = 0.06,
    perm_rate_range: Tuple[float, float] = (0.055, 0.075),
    base_construction_rate: float = 0.07,
    construction_rate_range: Tuple[float, float] = (0.065, 0.085),
) -> List[InputDistribution]:
    """Create distributions for financing uncertainty.

    Args:
        base_perm_rate: Base permanent loan rate
        perm_rate_range: Min/max for perm rate
        base_construction_rate: Base construction rate
        construction_rate_range: Min/max for construction rate

    Returns:
        List of InputDistribution for financing parameters
    """
    return [
        InputDistribution(
            parameter="perm_rate",
            distribution=DistributionType.TRIANGULAR,
            min_value=perm_rate_range[0],
            mode=base_perm_rate,
            max_value=perm_rate_range[1],
        ),
        InputDistribution(
            parameter="construction_rate",
            distribution=DistributionType.TRIANGULAR,
            min_value=construction_rate_range[0],
            mode=base_construction_rate,
            max_value=construction_rate_range[1],
        ),
    ]


def create_timing_uncertainty_distributions(
    base_construction_months: int = 18,
    construction_range: Tuple[int, int] = (15, 24),
    base_leaseup_months: int = 6,
    leaseup_range: Tuple[int, int] = (4, 12),
) -> List[InputDistribution]:
    """Create distributions for timing uncertainty.

    Args:
        base_construction_months: Base construction period
        construction_range: Min/max construction months
        base_leaseup_months: Base lease-up period
        leaseup_range: Min/max lease-up months

    Returns:
        List of InputDistribution for timing parameters
    """
    return [
        InputDistribution(
            parameter="construction_months",
            distribution=DistributionType.PERT,
            min_value=float(construction_range[0]),
            mode=float(base_construction_months),
            max_value=float(construction_range[1]),
            clip_min=float(construction_range[0]),
            clip_max=float(construction_range[1]),
        ),
        InputDistribution(
            parameter="leaseup_months",
            distribution=DistributionType.PERT,
            min_value=float(leaseup_range[0]),
            mode=float(base_leaseup_months),
            max_value=float(leaseup_range[1]),
            clip_min=float(leaseup_range[0]),
            clip_max=float(leaseup_range[1]),
        ),
    ]


def create_full_uncertainty_suite(
    base_hard_costs: float,
) -> List[InputDistribution]:
    """Create a comprehensive suite of distributions for full uncertainty analysis.

    This combines cost, market, financing, and timing uncertainty.

    Args:
        base_hard_costs: Base hard cost estimate

    Returns:
        List of all InputDistributions
    """
    distributions = []
    distributions.extend(create_cost_uncertainty_distributions(base_hard_costs))
    distributions.extend(create_market_uncertainty_distributions())
    distributions.extend(create_financing_uncertainty_distributions())
    distributions.extend(create_timing_uncertainty_distributions())
    return distributions


# ============================================================================
# TIF Grid Search for Optimal Configurations
# ============================================================================

@dataclass
class TIFGridPoint:
    """Single point in TIF grid search."""
    cap_rate: float
    term_years: int
    tif_lump_sum: float

    # Monte Carlo results at this point
    n_iterations: int
    market_irr_mean: float
    mixed_irr_mean: float
    irr_gap_mean_bps: float  # Mixed - Market in basis points
    irr_gap_std_bps: float

    # Probability metrics
    prob_gap_above_target: float  # P(gap >= target_bps)
    prob_gap_within_tolerance: float  # P(gap >= -tolerance_bps)

    # Confidence interval for gap
    gap_ci_lower_bps: float
    gap_ci_upper_bps: float
    gap_5th_percentile_bps: float
    gap_95th_percentile_bps: float

    def meets_target(self, target_bps: float = -250) -> bool:
        """Check if this configuration meets the target.

        Args:
            target_bps: Minimum acceptable gap (negative = mixed underperforms)

        Returns:
            True if 95th percentile of gap is above target
        """
        return self.gap_5th_percentile_bps >= target_bps


@dataclass
class TIFGridSearchResult:
    """Results from TIF grid search."""
    grid_points: List[TIFGridPoint]

    # Search parameters
    cap_rates_tested: List[float]
    terms_tested: List[int]
    target_gap_bps: float
    n_iterations_per_point: int

    # Best configurations
    best_by_probability: Optional[TIFGridPoint]  # Highest prob of meeting target
    best_by_gap: Optional[TIFGridPoint]  # Best mean gap
    minimum_viable: List[TIFGridPoint]  # All that meet target at 95% confidence

    def summary(self) -> str:
        """Return formatted summary of results."""
        lines = [
            "=" * 70,
            "TIF GRID SEARCH RESULTS",
            "=" * 70,
            f"Cap Rates Tested: {[f'{r:.1%}' for r in self.cap_rates_tested]}",
            f"Terms Tested: {self.terms_tested} years",
            f"Target Gap: {self.target_gap_bps:+.0f} bps (mixed - market)",
            f"Iterations per Point: {self.n_iterations_per_point}",
            "",
        ]

        if self.best_by_probability:
            bp = self.best_by_probability
            lines.extend([
                "BEST BY PROBABILITY OF MEETING TARGET",
                "-" * 50,
                f"  Cap Rate: {bp.cap_rate:.2%}",
                f"  Term: {bp.term_years} years",
                f"  TIF Lump Sum: ${bp.tif_lump_sum:,.0f}",
                f"  Mean IRR Gap: {bp.irr_gap_mean_bps:+.0f} bps",
                f"  P(Gap >= {self.target_gap_bps:+.0f} bps): {bp.prob_gap_within_tolerance:.1%}",
                f"  5th Percentile Gap: {bp.gap_5th_percentile_bps:+.0f} bps",
                "",
            ])

        if self.minimum_viable:
            lines.extend([
                f"CONFIGURATIONS MEETING TARGET ({len(self.minimum_viable)} found)",
                "-" * 50,
                f"{'Cap Rate':<12} {'Term':<8} {'Lump Sum':<15} {'Mean Gap':<12} {'P(Meet)':<10}",
                "-" * 50,
            ])
            for p in sorted(self.minimum_viable, key=lambda x: x.tif_lump_sum):
                lines.append(
                    f"{p.cap_rate:>10.2%}  {p.term_years:>5}yr  "
                    f"${p.tif_lump_sum:>12,.0f}  {p.irr_gap_mean_bps:>+8.0f} bps  "
                    f"{p.prob_gap_within_tolerance:>8.1%}"
                )
        else:
            lines.append("NO CONFIGURATIONS MEET TARGET - Consider higher cap rates or longer terms")

        lines.append("=" * 70)
        return "\n".join(lines)

    def get_heatmap_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get data formatted for heatmap visualization.

        Returns:
            Tuple of (cap_rates, terms, gap_matrix) for plotting
        """
        cap_rates = np.array(self.cap_rates_tested)
        terms = np.array(self.terms_tested)

        gap_matrix = np.zeros((len(terms), len(cap_rates)))

        for point in self.grid_points:
            i = self.terms_tested.index(point.term_years)
            j = self.cap_rates_tested.index(point.cap_rate)
            gap_matrix[i, j] = point.irr_gap_mean_bps

        return cap_rates, terms, gap_matrix


def _calculate_tif_lump_sum_for_grid(
    noi_delta_annual: float,
    cap_rate: float,
    term_years: int,
    discount_rate: float = 0.06,
    escalation_rate: float = 0.015,
) -> float:
    """Calculate TIF lump sum using NOI capitalization method.

    For grid search, we use the NOI method: lump_sum = NOI_delta / cap_rate
    This represents the present value of the income gap being filled.

    Args:
        noi_delta_annual: Annual NOI difference (market - mixed)
        cap_rate: Capitalization rate for lump sum calculation
        term_years: TIF term (affects escalated PV, not primary calc)
        discount_rate: Discount rate for PV method comparison
        escalation_rate: Annual escalation rate

    Returns:
        TIF lump sum amount
    """
    if cap_rate <= 0:
        return 0.0

    # Primary method: NOI capitalization
    lump_sum = noi_delta_annual / cap_rate

    return lump_sum


def run_tif_grid_search(
    base_inputs: BaseInputs,
    distributions: List[InputDistribution],
    cap_rates: List[float],
    terms: List[int],
    noi_delta_annual: float,
    target_gap_bps: float = -250,
    n_iterations: int = 100,
    seed: Optional[int] = None,
    confidence_level: float = 0.95,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> TIFGridSearchResult:
    """Run grid search over TIF cap rates and terms.

    For each (cap_rate, term) combination:
    1. Calculate TIF lump sum using that cap rate
    2. Run Monte Carlo with that TIF configuration
    3. Compare mixed-income IRR to market IRR
    4. Record probability of meeting target gap

    Args:
        base_inputs: Base inputs for DCF (should have affordable_pct > 0)
        distributions: Input distributions for Monte Carlo
        cap_rates: List of cap rates to test (e.g., [0.05, 0.06, 0.07])
        terms: List of TIF terms in years to test (e.g., [10, 15, 20])
        noi_delta_annual: Annual NOI gap between market and mixed scenarios
        target_gap_bps: Target minimum gap in bps (e.g., -250 means mixed can
                        underperform by at most 250 bps)
        n_iterations: Monte Carlo iterations per grid point
        seed: Random seed for reproducibility
        confidence_level: Confidence level for intervals
        progress_callback: Optional callback(completed, total)

    Returns:
        TIFGridSearchResult with all tested configurations
    """
    total_points = len(cap_rates) * len(terms)
    grid_points: List[TIFGridPoint] = []

    # Get tax stack for TIF calculations
    tax_stack = get_austin_tax_stack()

    # Master RNG
    master_rng = np.random.default_rng(seed)

    completed = 0

    for cap_rate in cap_rates:
        for term_years in terms:
            # Calculate TIF lump sum for this cap rate
            tif_lump_sum = _calculate_tif_lump_sum_for_grid(
                noi_delta_annual=noi_delta_annual,
                cap_rate=cap_rate,
                term_years=term_years,
            )

            # Create market scenario inputs (no TIF, no affordable)
            market_inputs_dict = base_inputs.to_dict()
            market_inputs_dict["affordable_pct"] = 0.0
            market_inputs_dict["tif_treatment"] = "none"
            market_inputs_dict["tif_lump_sum"] = 0.0

            # Create mixed-income scenario inputs (with TIF lump sum)
            mixed_inputs_dict = base_inputs.to_dict()
            mixed_inputs_dict["tif_treatment"] = "lump_sum"
            mixed_inputs_dict["tif_lump_sum"] = tif_lump_sum

            # Generate seeds for this grid point
            point_seed = int(master_rng.integers(0, 2**31))
            point_rng = np.random.default_rng(point_seed)
            iteration_seeds = point_rng.integers(0, 2**31, size=n_iterations)

            # Run iterations
            market_irrs = []
            mixed_irrs = []

            for i in range(n_iterations):
                iter_seed = int(iteration_seeds[i])
                iter_rng = np.random.default_rng(iter_seed)

                # Sample from distributions
                sampled = {}
                for dist in distributions:
                    sampled[dist.parameter] = dist.sample(iter_rng)

                # Apply samples to both scenarios
                market_inputs = market_inputs_dict.copy()
                mixed_inputs = mixed_inputs_dict.copy()
                for param, value in sampled.items():
                    market_inputs[param] = value
                    mixed_inputs[param] = value

                # Run DCFs
                try:
                    market_result = generate_detailed_cash_flow(
                        tax_stack=tax_stack,
                        **market_inputs
                    )
                    mixed_result = generate_detailed_cash_flow(
                        tax_stack=tax_stack,
                        **mixed_inputs
                    )

                    market_irrs.append(market_result.levered_irr)
                    mixed_irrs.append(mixed_result.levered_irr)
                except Exception:
                    # Skip failed iterations
                    pass

            if len(market_irrs) < 10:
                # Not enough valid iterations
                continue

            # Calculate statistics
            market_arr = np.array(market_irrs)
            mixed_arr = np.array(mixed_irrs)
            gap_arr = (mixed_arr - market_arr) * 10000  # Convert to bps

            alpha = 1 - confidence_level
            ci_lower = alpha / 2 * 100
            ci_upper = (1 - alpha / 2) * 100

            grid_point = TIFGridPoint(
                cap_rate=cap_rate,
                term_years=term_years,
                tif_lump_sum=tif_lump_sum,
                n_iterations=len(market_irrs),
                market_irr_mean=float(np.mean(market_arr)),
                mixed_irr_mean=float(np.mean(mixed_arr)),
                irr_gap_mean_bps=float(np.mean(gap_arr)),
                irr_gap_std_bps=float(np.std(gap_arr)),
                prob_gap_above_target=float(np.mean(gap_arr >= 0)),
                prob_gap_within_tolerance=float(np.mean(gap_arr >= target_gap_bps)),
                gap_ci_lower_bps=float(np.percentile(gap_arr, ci_lower)),
                gap_ci_upper_bps=float(np.percentile(gap_arr, ci_upper)),
                gap_5th_percentile_bps=float(np.percentile(gap_arr, 5)),
                gap_95th_percentile_bps=float(np.percentile(gap_arr, 95)),
            )
            grid_points.append(grid_point)

            completed += 1
            if progress_callback:
                progress_callback(completed, total_points)

    # Find best configurations
    if grid_points:
        # Best by probability of meeting target
        best_by_prob = max(grid_points, key=lambda p: p.prob_gap_within_tolerance)

        # Best by mean gap
        best_by_gap = max(grid_points, key=lambda p: p.irr_gap_mean_bps)

        # All that meet target at 95% confidence (5th percentile >= target)
        minimum_viable = [p for p in grid_points if p.gap_5th_percentile_bps >= target_gap_bps]
    else:
        best_by_prob = None
        best_by_gap = None
        minimum_viable = []

    return TIFGridSearchResult(
        grid_points=grid_points,
        cap_rates_tested=cap_rates,
        terms_tested=terms,
        target_gap_bps=target_gap_bps,
        n_iterations_per_point=n_iterations,
        best_by_probability=best_by_prob,
        best_by_gap=best_by_gap,
        minimum_viable=minimum_viable,
    )


def create_tif_distributions(
    base_cap_rate: float = 0.06,
    cap_rate_range: Tuple[float, float] = (0.05, 0.08),
    base_term: int = 15,
    term_range: Tuple[int, int] = (10, 25),
) -> List[InputDistribution]:
    """Create distributions for TIF uncertainty analysis.

    Note: These are for Monte Carlo over TIF parameters, not grid search.
    Grid search tests discrete combinations; these distributions are for
    sensitivity analysis within a single configuration.

    Args:
        base_cap_rate: Base TIF cap rate
        cap_rate_range: Min/max for cap rate
        base_term: Base TIF term in years
        term_range: Min/max term in years

    Returns:
        List of InputDistributions for TIF parameters
    """
    return [
        InputDistribution(
            parameter="tif_cap_rate",
            distribution=DistributionType.TRIANGULAR,
            min_value=cap_rate_range[0],
            mode=base_cap_rate,
            max_value=cap_rate_range[1],
        ),
        InputDistribution(
            parameter="tif_term_years",
            distribution=DistributionType.UNIFORM,
            min_value=float(term_range[0]),
            max_value=float(term_range[1]),
            clip_min=float(term_range[0]),
            clip_max=float(term_range[1]),
        ),
    ]
