"""Tests for Monte Carlo simulation engine."""

import pytest
import numpy as np
from datetime import date

from src.calculations.monte_carlo import (
    DistributionType,
    InputDistribution,
    BaseInputs,
    MonteCarloConfig,
    run_monte_carlo,
    create_cost_uncertainty_distributions,
    create_market_uncertainty_distributions,
    create_financing_uncertainty_distributions,
    create_timing_uncertainty_distributions,
    create_full_uncertainty_suite,
)
from src.calculations.detailed_cashflow import AssessedValueBasis


class TestInputDistribution:
    """Test InputDistribution sampling."""

    def test_uniform_distribution(self):
        """Uniform distribution samples within bounds."""
        dist = InputDistribution(
            parameter="test",
            distribution=DistributionType.UNIFORM,
            min_value=10.0,
            max_value=20.0,
        )
        rng = np.random.default_rng(42)

        samples = [dist.sample(rng) for _ in range(1000)]

        assert min(samples) >= 10.0
        assert max(samples) <= 20.0
        # Mean should be ~15 for uniform(10, 20)
        assert 14.0 < np.mean(samples) < 16.0

    def test_normal_distribution(self):
        """Normal distribution has correct mean and std."""
        dist = InputDistribution(
            parameter="test",
            distribution=DistributionType.NORMAL,
            mean=100.0,
            std=10.0,
        )
        rng = np.random.default_rng(42)

        samples = [dist.sample(rng) for _ in range(10000)]

        assert 98.0 < np.mean(samples) < 102.0
        assert 9.0 < np.std(samples) < 11.0

    def test_triangular_distribution(self):
        """Triangular distribution samples correctly."""
        dist = InputDistribution(
            parameter="test",
            distribution=DistributionType.TRIANGULAR,
            min_value=0.05,
            mode=0.06,
            max_value=0.08,
        )
        rng = np.random.default_rng(42)

        samples = [dist.sample(rng) for _ in range(1000)]

        assert min(samples) >= 0.05
        assert max(samples) <= 0.08
        # Mode should be near peak of density
        hist, edges = np.histogram(samples, bins=30)
        peak_idx = np.argmax(hist)
        peak_center = (edges[peak_idx] + edges[peak_idx + 1]) / 2
        assert 0.055 < peak_center < 0.065

    def test_lognormal_distribution(self):
        """Lognormal distribution is always positive."""
        dist = InputDistribution(
            parameter="test",
            distribution=DistributionType.LOGNORMAL,
            mean=25_000_000,
            std=2_500_000,
        )
        rng = np.random.default_rng(42)

        samples = [dist.sample(rng) for _ in range(1000)]

        assert min(samples) > 0  # Always positive
        # Mean should be approximately correct
        assert 20_000_000 < np.mean(samples) < 30_000_000

    def test_pert_distribution(self):
        """PERT distribution weights mode."""
        dist = InputDistribution(
            parameter="test",
            distribution=DistributionType.PERT,
            min_value=12,
            mode=18,
            max_value=24,
        )
        rng = np.random.default_rng(42)

        samples = [dist.sample(rng) for _ in range(1000)]

        assert min(samples) >= 12
        assert max(samples) <= 24
        # PERT mean = (min + 4*mode + max) / 6 = (12 + 72 + 24) / 6 = 18
        assert 16.5 < np.mean(samples) < 19.5

    def test_clipping(self):
        """Clipping bounds are respected."""
        dist = InputDistribution(
            parameter="test",
            distribution=DistributionType.NORMAL,
            mean=0.06,
            std=0.02,
            clip_min=0.04,
            clip_max=0.08,
        )
        rng = np.random.default_rng(42)

        samples = [dist.sample(rng) for _ in range(1000)]

        assert min(samples) >= 0.04
        assert max(samples) <= 0.08


class TestBaseInputs:
    """Test BaseInputs configuration."""

    def test_default_values(self):
        """BaseInputs has reasonable defaults."""
        inputs = BaseInputs()

        assert inputs.land_cost == 2_000_000
        assert inputs.hard_costs == 25_000_000
        assert inputs.construction_months == 18
        assert inputs.vacancy_rate == 0.06

    def test_to_dict(self):
        """to_dict returns all parameters."""
        inputs = BaseInputs(land_cost=3_000_000)
        d = inputs.to_dict()

        assert d["land_cost"] == 3_000_000
        assert "hard_costs" in d
        assert "perm_rate" in d
        assert "tif_treatment" in d


class TestConvenienceFunctions:
    """Test distribution creation convenience functions."""

    def test_cost_uncertainty_distributions(self):
        """Cost uncertainty creates hard_costs and soft_cost_pct."""
        dists = create_cost_uncertainty_distributions(25_000_000)

        params = {d.parameter for d in dists}
        assert "hard_costs" in params
        assert "soft_cost_pct" in params

    def test_market_uncertainty_distributions(self):
        """Market uncertainty creates exit_cap_rate, rent_growth, vacancy."""
        dists = create_market_uncertainty_distributions()

        params = {d.parameter for d in dists}
        assert "exit_cap_rate" in params
        assert "market_rent_growth" in params
        assert "vacancy_rate" in params

    def test_financing_uncertainty_distributions(self):
        """Financing uncertainty creates rate distributions."""
        dists = create_financing_uncertainty_distributions()

        params = {d.parameter for d in dists}
        assert "perm_rate" in params
        assert "construction_rate" in params

    def test_timing_uncertainty_distributions(self):
        """Timing uncertainty creates construction and leaseup months."""
        dists = create_timing_uncertainty_distributions()

        params = {d.parameter for d in dists}
        assert "construction_months" in params
        assert "leaseup_months" in params

    def test_full_uncertainty_suite(self):
        """Full suite combines all categories."""
        dists = create_full_uncertainty_suite(25_000_000)

        params = {d.parameter for d in dists}
        # Should have all major parameters
        assert len(dists) >= 8
        assert "hard_costs" in params
        assert "exit_cap_rate" in params
        assert "perm_rate" in params
        assert "construction_months" in params


class TestMonteCarloSimulation:
    """Test full Monte Carlo simulation."""

    def test_small_simulation(self):
        """Small simulation runs successfully."""
        base_inputs = BaseInputs(
            land_cost=2_000_000,
            hard_costs=25_000_000,
            monthly_gpr_at_stabilization=600_000,
            total_units=250,
            operations_months=12,  # Shorter for speed
        )

        distributions = [
            InputDistribution(
                parameter="exit_cap_rate",
                distribution=DistributionType.UNIFORM,
                min_value=0.05,
                max_value=0.06,
            ),
        ]

        config = MonteCarloConfig(
            base_inputs=base_inputs,
            distributions=distributions,
            n_iterations=10,
            seed=42,
            parallel=False,  # Sequential for testing
        )

        result = run_monte_carlo(config)

        # Basic checks
        assert result.n_iterations == 10
        assert len(result.iterations) == 10
        assert result.levered_irr_mean > 0
        assert result.unlevered_irr_mean > 0

    def test_reproducibility_with_seed(self):
        """Same seed produces same results."""
        base_inputs = BaseInputs(operations_months=12)
        distributions = [
            InputDistribution(
                parameter="exit_cap_rate",
                distribution=DistributionType.UNIFORM,
                min_value=0.05,
                max_value=0.06,
            ),
        ]

        config = MonteCarloConfig(
            base_inputs=base_inputs,
            distributions=distributions,
            n_iterations=5,
            seed=12345,
            parallel=False,
        )

        result1 = run_monte_carlo(config)
        result2 = run_monte_carlo(config)

        assert result1.levered_irr_mean == result2.levered_irr_mean
        assert result1.unlevered_irr_mean == result2.unlevered_irr_mean

    def test_confidence_intervals(self):
        """Confidence intervals are calculated correctly."""
        base_inputs = BaseInputs(operations_months=12)
        distributions = [
            InputDistribution(
                parameter="exit_cap_rate",
                distribution=DistributionType.UNIFORM,
                min_value=0.04,
                max_value=0.07,
            ),
        ]

        config = MonteCarloConfig(
            base_inputs=base_inputs,
            distributions=distributions,
            n_iterations=20,
            seed=42,
            confidence_level=0.90,
            parallel=False,
        )

        result = run_monte_carlo(config)

        # CI should bracket most values
        assert result.levered_irr_ci_lower < result.levered_irr_mean
        assert result.levered_irr_ci_upper > result.levered_irr_mean
        assert result.levered_irr_ci_lower >= result.levered_irr_min
        assert result.levered_irr_ci_upper <= result.levered_irr_max

    def test_sensitivity_analysis(self):
        """Sensitivity correlations are calculated."""
        base_inputs = BaseInputs(operations_months=12)
        distributions = [
            InputDistribution(
                parameter="exit_cap_rate",
                distribution=DistributionType.UNIFORM,
                min_value=0.04,
                max_value=0.07,
            ),
            InputDistribution(
                parameter="hard_costs",
                distribution=DistributionType.UNIFORM,
                min_value=22_000_000,
                max_value=28_000_000,
            ),
        ]

        config = MonteCarloConfig(
            base_inputs=base_inputs,
            distributions=distributions,
            n_iterations=30,
            seed=42,
            parallel=False,
        )

        result = run_monte_carlo(config)

        # Should have sensitivities for both parameters
        assert len(result.sensitivities) == 2

        params = {s.parameter for s in result.sensitivities}
        assert "exit_cap_rate" in params
        assert "hard_costs" in params

        # Exit cap should negatively correlate with IRR
        cap_sens = next(s for s in result.sensitivities if s.parameter == "exit_cap_rate")
        assert cap_sens.correlation_levered < 0  # Higher cap = lower IRR

    def test_probability_metrics(self):
        """Probability metrics are calculated."""
        base_inputs = BaseInputs(operations_months=12)
        distributions = [
            InputDistribution(
                parameter="exit_cap_rate",
                distribution=DistributionType.UNIFORM,
                min_value=0.05,
                max_value=0.06,
            ),
        ]

        config = MonteCarloConfig(
            base_inputs=base_inputs,
            distributions=distributions,
            n_iterations=10,
            seed=42,
            parallel=False,
        )

        result = run_monte_carlo(config, target_irr=0.15)

        # Probabilities should be between 0 and 1
        assert 0 <= result.prob_levered_positive <= 1
        assert 0 <= result.prob_unlevered_positive <= 1
        assert 0 <= result.prob_levered_above_target <= 1
        assert result.target_irr == 0.15

    def test_summary_output(self):
        """Summary output is formatted correctly."""
        base_inputs = BaseInputs(operations_months=12)
        distributions = [
            InputDistribution(
                parameter="exit_cap_rate",
                distribution=DistributionType.UNIFORM,
                min_value=0.05,
                max_value=0.06,
            ),
        ]

        config = MonteCarloConfig(
            base_inputs=base_inputs,
            distributions=distributions,
            n_iterations=10,
            seed=42,
            parallel=False,
        )

        result = run_monte_carlo(config)
        summary = result.summary()

        assert "MONTE CARLO SIMULATION RESULTS" in summary
        assert "UNLEVERED IRR" in summary
        assert "LEVERED IRR" in summary
        assert "PROBABILITIES" in summary
        assert "SENSITIVITY" in summary


class TestTIFGridSearch:
    """Test TIF grid search functionality."""

    def test_grid_search_runs(self):
        """Grid search completes without error."""
        from src.calculations.monte_carlo import run_tif_grid_search, TIFGridSearchResult

        base_inputs = BaseInputs(
            operations_months=12,
            affordable_pct=0.15,
            affordable_rent_discount=0.40,
        )

        distributions = [
            InputDistribution(
                parameter="exit_cap_rate",
                distribution=DistributionType.UNIFORM,
                min_value=0.05,
                max_value=0.06,
            ),
        ]

        result = run_tif_grid_search(
            base_inputs=base_inputs,
            distributions=distributions,
            cap_rates=[0.06, 0.07],
            terms=[10, 15],
            noi_delta_annual=100_000,
            target_gap_bps=-250,
            n_iterations=10,
            seed=42,
        )

        assert isinstance(result, TIFGridSearchResult)
        assert len(result.grid_points) == 4  # 2 cap rates x 2 terms
        assert result.target_gap_bps == -250

    def test_grid_point_meets_target(self):
        """Grid point correctly identifies target meeting."""
        from src.calculations.monte_carlo import TIFGridPoint

        # Point that meets target (5th percentile above -250)
        point_meets = TIFGridPoint(
            cap_rate=0.06,
            term_years=15,
            tif_lump_sum=1_000_000,
            n_iterations=100,
            market_irr_mean=0.35,
            mixed_irr_mean=0.34,
            irr_gap_mean_bps=-100,
            irr_gap_std_bps=50,
            prob_gap_above_target=0.8,
            prob_gap_within_tolerance=0.95,
            gap_ci_lower_bps=-200,
            gap_ci_upper_bps=0,
            gap_5th_percentile_bps=-200,  # Above -250
            gap_95th_percentile_bps=0,
        )
        assert point_meets.meets_target(-250)

        # Point that doesn't meet target
        point_fails = TIFGridPoint(
            cap_rate=0.06,
            term_years=15,
            tif_lump_sum=500_000,
            n_iterations=100,
            market_irr_mean=0.35,
            mixed_irr_mean=0.32,
            irr_gap_mean_bps=-300,
            irr_gap_std_bps=50,
            prob_gap_above_target=0.2,
            prob_gap_within_tolerance=0.40,
            gap_ci_lower_bps=-400,
            gap_ci_upper_bps=-200,
            gap_5th_percentile_bps=-400,  # Below -250
            gap_95th_percentile_bps=-200,
        )
        assert not point_fails.meets_target(-250)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
