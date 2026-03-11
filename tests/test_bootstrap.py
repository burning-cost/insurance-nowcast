"""
Tests for bootstrap confidence interval computation.
"""
import numpy as np
import pandas as pd
import pytest

from insurance_nowcast._bootstrap import (
    analytic_completion_factor_variance,
)


class TestAnalyticCompletionFactorVariance:
    def test_output_shape(self):
        n, d = 5, 6
        p_hat = np.random.dirichlet(np.ones(d), size=n)
        trunc = np.array([3, 4, 5, 6, 6])
        lam = np.random.uniform(5, 15, n)
        n_obs = np.array([5, 8, 10, 12, 15]).astype(float)
        result = analytic_completion_factor_variance(p_hat, trunc, lam, n_obs)
        assert result.shape == (n,)

    def test_non_negative(self):
        n, d = 4, 6
        p_hat = np.random.dirichlet(np.ones(d), size=n)
        trunc = np.full(n, 3)
        lam = np.ones(n) * 10
        n_obs = np.ones(n) * 8
        result = analytic_completion_factor_variance(p_hat, trunc, lam, n_obs)
        assert (result >= 0).all()

    def test_more_obs_lower_se(self):
        """More observations should give lower standard error."""
        d = 6
        p_hat = np.array([[0.5, 0.3, 0.1, 0.05, 0.03, 0.02]])
        trunc = np.array([2])
        lam = np.array([10.0])

        se_few = analytic_completion_factor_variance(p_hat, trunc, lam, np.array([5.0]))
        se_many = analytic_completion_factor_variance(p_hat, trunc, lam, np.array([500.0]))

        assert se_few[0] > se_many[0]

    def test_finite_result(self):
        rng = np.random.default_rng(0)
        n, d = 8, 10
        p_hat = rng.dirichlet(np.ones(d), size=n)
        trunc = rng.integers(2, d + 1, size=n)
        lam = rng.uniform(5, 50, n)
        n_obs = rng.uniform(3, 30, n)
        result = analytic_completion_factor_variance(p_hat, trunc, lam, n_obs)
        assert np.isfinite(result).all()
