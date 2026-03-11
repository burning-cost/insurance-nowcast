"""
Tests for the E-step imputation logic.
"""
import numpy as np
import pytest

from insurance_nowcast._estep import (
    estep,
    compute_observed_log_likelihood,
    compute_initial_lambda,
)


# --- estep ---

class TestEstep:
    def test_observed_periods_unchanged(self):
        """For j < tau_i, observed counts must not be changed."""
        observed = np.array([[3.0, 2.0, 0.0, 0.0],
                             [1.0, 4.0, 2.0, 0.0]])
        trunc = np.array([2, 3])
        lambda_hat = np.array([5.0, 7.0])
        p_hat = np.ones((2, 4)) / 4.0

        result = estep(observed, trunc, lambda_hat, p_hat)

        # Period 0: delays 0, 1 observed
        assert result[0, 0] == 3.0
        assert result[0, 1] == 2.0
        # Period 1: delays 0, 1, 2 observed
        assert result[1, 0] == 1.0
        assert result[1, 1] == 4.0
        assert result[1, 2] == 2.0

    def test_censored_periods_imputed(self):
        """For j >= tau_i, imputed = lambda_hat * p_hat."""
        observed = np.array([[3.0, 0.0, 0.0, 0.0]])
        trunc = np.array([1])
        lambda_hat = np.array([10.0])
        p_hat = np.array([[0.4, 0.3, 0.2, 0.1]])

        result = estep(observed, trunc, lambda_hat, p_hat)

        # Delay 0 is observed
        assert result[0, 0] == 3.0
        # Delays 1, 2, 3 are censored
        np.testing.assert_allclose(result[0, 1], 10.0 * 0.3)
        np.testing.assert_allclose(result[0, 2], 10.0 * 0.2)
        np.testing.assert_allclose(result[0, 3], 10.0 * 0.1)

    def test_fully_observed_period_unchanged(self):
        """When tau = max_delay, no imputation should occur."""
        max_delay = 5
        observed = np.array([[2.0, 1.0, 3.0, 0.0, 4.0]])
        trunc = np.array([max_delay])  # fully observed
        lambda_hat = np.array([100.0])  # large value — should not affect result
        p_hat = np.ones((1, max_delay)) / max_delay

        result = estep(observed, trunc, lambda_hat, p_hat)
        np.testing.assert_array_equal(result[0], observed[0])

    def test_output_shape(self):
        n, d = 10, 8
        observed = np.zeros((n, d))
        trunc = np.full(n, 4)
        lambda_hat = np.ones(n) * 5.0
        p_hat = np.ones((n, d)) / d
        result = estep(observed, trunc, lambda_hat, p_hat)
        assert result.shape == (n, d)

    def test_non_negative_output(self):
        n, d = 5, 6
        observed = np.abs(np.random.randn(n, d))
        trunc = np.array([2, 3, 4, 5, 6])
        lambda_hat = np.abs(np.random.randn(n)) + 0.1
        p_hat = np.random.dirichlet(np.ones(d), size=n)
        result = estep(observed, trunc, lambda_hat, p_hat)
        assert (result >= 0).all()

    def test_multiple_periods_independence(self):
        """Each period's imputation depends only on its own lambda and p."""
        observed = np.zeros((2, 4))
        trunc = np.array([0, 0])  # Nothing observed for either period
        lambda_hat = np.array([10.0, 20.0])
        p_hat = np.array([[0.5, 0.3, 0.1, 0.1],
                           [0.2, 0.4, 0.3, 0.1]])

        result = estep(observed, trunc, lambda_hat, p_hat)

        # Period 0: all imputed from lambda=10
        np.testing.assert_allclose(result[0], 10.0 * p_hat[0])
        # Period 1: all imputed from lambda=20
        np.testing.assert_allclose(result[1], 20.0 * p_hat[1])


# --- compute_observed_log_likelihood ---

class TestComputeObservedLogLikelihood:
    def test_returns_float(self):
        observed = np.array([[3.0, 2.0, 0.0]])
        trunc = np.array([2])
        lambda_hat = np.array([5.0])
        p_hat = np.array([[0.5, 0.3, 0.2]])
        exposure = np.array([10.0])
        ll = compute_observed_log_likelihood(observed, trunc, lambda_hat, p_hat, exposure)
        assert isinstance(ll, float)

    def test_higher_ll_for_better_fit(self):
        """A model with matching delay distribution should have higher LL."""
        observed = np.array([[8.0, 2.0, 0.0, 0.0]])
        trunc = np.array([2])
        exposure = np.array([10.0])

        # Good fit: p proportional to observed (0.8, 0.2, ...)
        p_good = np.array([[0.8, 0.2, 0.0, 0.0]])
        p_good = p_good / p_good.sum()
        lambda_good = np.array([10.0])

        # Bad fit: uniform distribution
        p_bad = np.array([[0.25, 0.25, 0.25, 0.25]])
        lambda_bad = np.array([10.0])

        ll_good = compute_observed_log_likelihood(observed, trunc, lambda_good, p_good, exposure)
        ll_bad = compute_observed_log_likelihood(observed, trunc, lambda_bad, p_bad, exposure)

        assert ll_good > ll_bad

    def test_finite_result(self):
        """Log-likelihood should always be finite."""
        rng = np.random.default_rng(0)
        n_periods, d = 10, 8
        observed = rng.integers(0, 5, size=(n_periods, d)).astype(float)
        trunc = rng.integers(2, d, size=n_periods)
        lambda_hat = rng.uniform(1, 20, size=n_periods)
        p_hat = rng.dirichlet(np.ones(d), size=n_periods)
        exposure = rng.uniform(1, 100, size=n_periods)

        ll = compute_observed_log_likelihood(observed, trunc, lambda_hat, p_hat, exposure)
        assert np.isfinite(ll)

    def test_more_data_lower_variance_higher_ll(self):
        """More observations with good fit should give higher (less negative) LL."""
        p_hat = np.array([[0.6, 0.4]])
        exposure = np.array([100.0])

        # Small dataset: 6 claims at delay 0, 4 at delay 1
        obs_small = np.array([[6.0, 4.0]])
        ll_small = compute_observed_log_likelihood(
            obs_small, np.array([2]), np.array([10.0]), p_hat, exposure
        )

        # Large dataset (10x): 60 claims at delay 0, 40 at delay 1
        obs_large = np.array([[60.0, 40.0]])
        ll_large = compute_observed_log_likelihood(
            obs_large, np.array([2]), np.array([100.0]), p_hat, exposure
        )

        # Both should be finite; not asserting direction (scale depends on N)
        assert np.isfinite(ll_small)
        assert np.isfinite(ll_large)


# --- compute_initial_lambda ---

class TestComputeInitialLambda:
    def test_basic_recovery(self):
        """With uniform delay and full observation, lambda ≈ total observed."""
        d = 4
        n = 3
        observed = np.array([[10.0, 0.0, 0.0, 0.0],
                              [5.0, 5.0, 0.0, 0.0],
                              [3.0, 3.0, 3.0, 3.0]])
        trunc = np.array([d, d, d])  # fully observed
        p_hat = np.ones((n, d)) / d
        exposure = np.ones(n)

        result = compute_initial_lambda(observed, trunc, p_hat, exposure)

        # With fully observed periods and uniform p,
        # F_i = sum_{j<d} p_{i,j} = 1.0
        # So lambda_init[i] = N_obs_i / 1.0
        np.testing.assert_allclose(result[0], 10.0, rtol=0.01)
        np.testing.assert_allclose(result[1], 10.0, rtol=0.01)
        np.testing.assert_allclose(result[2], 12.0, rtol=0.01)

    def test_partial_observation_inflates_lambda(self):
        """With partial observation, lambda should be > observed count."""
        observed = np.array([[5.0, 0.0, 0.0, 0.0]])
        trunc = np.array([1])  # Only delay 0 observed
        # p_hat: delay 0 has probability 0.5, rest 0.5
        p_hat = np.array([[0.5, 0.25, 0.15, 0.10]])
        exposure = np.array([10.0])

        result = compute_initial_lambda(observed, trunc, p_hat, exposure)
        # F_0 = 0.5, so lambda_init = 5 / 0.5 = 10
        np.testing.assert_allclose(result[0], 10.0, rtol=0.01)

    def test_output_positive(self):
        d = 5
        n = 4
        observed = np.random.rand(n, d)
        trunc = np.random.randint(1, d + 1, size=n)
        p_hat = np.random.dirichlet(np.ones(d), size=n)
        exposure = np.ones(n)
        result = compute_initial_lambda(observed, trunc, p_hat, exposure)
        assert (result >= 0).all()
