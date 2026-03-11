"""
Tests for completion factor and IBNR computation.
"""
import numpy as np
import pandas as pd
import pytest

from insurance_nowcast._completion import (
    predict_completion_factors,
    predict_ibnr_by_period,
    predict_development_pattern,
)
from insurance_nowcast._mstep import MultinomialGLMDelay, XGBoostDelay


def _make_test_data(n_periods=5, max_delay=6):
    rng = np.random.default_rng(42)
    occurrence_periods = np.arange(n_periods)
    # Truncation: older periods have more observed delays
    # Build dynamically to handle any n_periods
    trunc_values = []
    for i in range(n_periods):
        elapsed = n_periods - 1 - i  # older periods have more elapsed time
        trunc_values.append(min(max_delay, elapsed + 1))
    truncation_depth = np.array(trunc_values, dtype=int)
    # Ensure at least 1
    truncation_depth = np.maximum(truncation_depth, 1)
    # Delay distribution
    p_hat = rng.dirichlet(np.ones(max_delay), size=n_periods)
    observed = rng.integers(0, 10, size=(n_periods, max_delay)).astype(float)
    lambda_hat = rng.uniform(5, 20, size=n_periods)
    return occurrence_periods, truncation_depth, p_hat, observed, lambda_hat


class TestPredictCompletionFactors:
    def test_returns_dataframe(self):
        occ, trunc, p, obs, lam = _make_test_data()
        result = predict_completion_factors(occ, trunc, p, obs, lam, {})
        assert isinstance(result, pd.DataFrame)

    def test_n_rows_equals_n_periods(self):
        occ, trunc, p, obs, lam = _make_test_data(n_periods=7)
        result = predict_completion_factors(occ, trunc, p, obs, lam, {})
        assert len(result) == 7

    def test_completion_factor_ge_one(self):
        occ, trunc, p, obs, lam = _make_test_data()
        result = predict_completion_factors(occ, trunc, p, obs, lam, {})
        assert (result["completion_factor"] >= 1.0 - 1e-6).all()

    def test_fully_observed_cf_near_one(self):
        """Period with max truncation depth should have CF near 1."""
        n, d = 3, 6
        occ = np.array([0, 1, 2])
        trunc = np.array([d, d, d])
        # Concentrated at delay 0 => all probability within tau
        p = np.zeros((n, d))
        p[:, 0] = 1.0  # all claims at delay 0
        obs = np.zeros((n, d))
        obs[:, 0] = 5.0
        lam = np.array([5.0, 5.0, 5.0])
        result = predict_completion_factors(occ, trunc, p, obs, lam, {})
        np.testing.assert_allclose(result["completion_factor"].values, 1.0, atol=0.01)

    def test_tau_one_high_cf(self):
        """Period with tau=1 and mass in later delays should have high CF."""
        occ = np.array([0])
        trunc = np.array([1])  # Only delay 0 observed
        p = np.array([[0.1, 0.4, 0.3, 0.2]])  # Only 10% at delay 0
        obs = np.zeros((1, 4))
        lam = np.array([10.0])
        result = predict_completion_factors(occ, trunc, p, obs, lam, {})
        assert result["completion_factor"].values[0] > 5.0  # 1/0.1 = 10

    def test_required_columns(self):
        occ, trunc, p, obs, lam = _make_test_data()
        result = predict_completion_factors(occ, trunc, p, obs, lam, {})
        for col in ["occurrence_period", "completion_factor", "development_fraction",
                    "observed_claims", "estimated_ultimate_claims", "ibnr_count"]:
            assert col in result.columns

    def test_ibnr_equals_ultimate_minus_observed(self):
        occ, trunc, p, obs, lam = _make_test_data()
        result = predict_completion_factors(occ, trunc, p, obs, lam, {})
        diff = result["estimated_ultimate_claims"] - result["observed_claims"]
        expected_ibnr = diff.clip(lower=0)
        np.testing.assert_allclose(
            result["ibnr_count"].values,
            expected_ibnr.values,
            atol=0.01,
        )


class TestPredictIbnrByPeriod:
    def test_returns_dataframe(self):
        occ, trunc, p, obs, lam = _make_test_data()
        result = predict_ibnr_by_period(occ, trunc, p, lam, obs)
        assert isinstance(result, pd.DataFrame)

    def test_ibnr_non_negative(self):
        occ, trunc, p, obs, lam = _make_test_data()
        result = predict_ibnr_by_period(occ, trunc, p, lam, obs)
        assert (result["ibnr_count"] >= 0).all()

    def test_ibnr_fraction_in_range(self):
        occ, trunc, p, obs, lam = _make_test_data()
        result = predict_ibnr_by_period(occ, trunc, p, lam, obs)
        assert (result["ibnr_fraction"] >= 0).all()
        assert (result["ibnr_fraction"] <= 1.0).all()

    def test_full_observation_zero_ibnr(self):
        """Fully observed periods should have zero IBNR."""
        n, d = 3, 4
        occ = np.arange(n)
        trunc = np.full(n, d)  # All fully observed
        p = np.ones((n, d)) / d
        lam = np.array([10.0, 10.0, 10.0])
        obs = np.zeros((n, d))
        result = predict_ibnr_by_period(occ, trunc, p, lam, obs)
        np.testing.assert_allclose(result["ibnr_fraction"].values, 0.0, atol=1e-10)


class TestPredictDevelopmentPatternFn:
    def _fit_glm(self, n_classes=6):
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (60, 2))
        y = rng.integers(0, n_classes, 60)
        w = rng.uniform(1, 5, 60)
        model = MultinomialGLMDelay()
        model.fit(X, y, w, n_classes=n_classes)
        return model, n_classes

    def test_correct_shape(self):
        model, n_classes = self._fit_glm()
        X = np.random.rand(10, 2)
        result = predict_development_pattern(model, X, max_delay_periods=n_classes)
        assert result.shape == (10, n_classes)

    def test_sums_to_one(self):
        model, n_classes = self._fit_glm()
        X = np.random.rand(20, 2)
        result = predict_development_pattern(model, X, max_delay_periods=n_classes)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-5)

    def test_non_negative(self):
        model, n_classes = self._fit_glm()
        X = np.random.rand(20, 2)
        result = predict_development_pattern(model, X, max_delay_periods=n_classes)
        assert (result >= 0).all()
