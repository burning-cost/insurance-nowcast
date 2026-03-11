"""
Tests for truncation computation and data preparation.
"""
import numpy as np
import pandas as pd
import pytest

from insurance_nowcast._truncation import (
    compute_delay,
    compute_truncation_depth,
    prepare_observed_triangle,
    build_delay_training_data,
)


# --- compute_delay ---

class TestComputeDelay:
    def test_integer_periods(self):
        occ = pd.Series([0, 1, 2, 3])
        rep = pd.Series([2, 2, 3, 5])
        result = compute_delay(occ, rep)
        np.testing.assert_array_equal(result.values, [2, 1, 1, 2])

    def test_zero_delay(self):
        occ = pd.Series([5])
        rep = pd.Series([5])
        assert compute_delay(occ, rep).values[0] == 0

    def test_null_report(self):
        occ = pd.Series([0, 1])
        rep = pd.Series([2, np.nan])
        result = compute_delay(occ, rep)
        assert result.iloc[0] == 2
        assert pd.isna(result.iloc[1])

    def test_large_delay(self):
        occ = pd.Series([0])
        rep = pd.Series([24])
        assert compute_delay(occ, rep).values[0] == 24


# --- compute_truncation_depth ---

class TestComputeTruncationDepth:
    def test_basic(self):
        occ = pd.Series([0, 1, 2, 3])
        result = compute_truncation_depth(occ, eval_date=4, max_delay_periods=12)
        # elapsed = [4, 3, 2, 1], tau = elapsed + 1 = [5, 4, 3, 2], capped at 12
        np.testing.assert_array_equal(result.values, [5, 4, 3, 2])

    def test_cap_at_max_delay(self):
        occ = pd.Series([0])
        result = compute_truncation_depth(occ, eval_date=100, max_delay_periods=12)
        assert result.values[0] == 12

    def test_current_period_tau_is_one(self):
        occ = pd.Series([5])
        result = compute_truncation_depth(occ, eval_date=5, max_delay_periods=12)
        # elapsed = 0, tau = 1
        assert result.values[0] == 1

    def test_all_periods_fully_observed(self):
        occ = pd.Series([0, 1, 2])
        result = compute_truncation_depth(occ, eval_date=20, max_delay_periods=5)
        # All should be at max (5)
        np.testing.assert_array_equal(result.values, [5, 5, 5])

    def test_returns_integers(self):
        occ = pd.Series([0, 1])
        result = compute_truncation_depth(occ, eval_date=3, max_delay_periods=6)
        assert result.dtype in (np.int32, np.int64, int)


# --- prepare_observed_triangle ---

class TestPrepareObservedTriangle:
    def _make_df(self, n_periods=5, n_claims_per_period=10, max_delay=6):
        rng = np.random.default_rng(42)
        records = []
        for p in range(n_periods):
            for _ in range(n_claims_per_period):
                delay = rng.integers(0, max_delay)
                records.append({
                    "occurrence_period": p,
                    "report_period": float(p + delay),
                    "exposure": 1.0,
                    "feature_a": rng.uniform(0, 1),
                })
        return pd.DataFrame(records)

    def test_output_keys(self):
        df = self._make_df()
        result = prepare_observed_triangle(
            df, "occurrence_period", "report_period", "exposure",
            ["feature_a"], eval_date=6, max_delay_periods=6
        )
        for key in ["observed_counts", "occurrence_periods", "truncation_depth",
                    "exposure_by_period", "features_by_period", "period_to_idx",
                    "n_periods", "max_delay_periods"]:
            assert key in result, f"Missing key: {key}"

    def test_observed_counts_shape(self):
        df = self._make_df(n_periods=5, max_delay=6)
        result = prepare_observed_triangle(
            df, "occurrence_period", "report_period", "exposure",
            ["feature_a"], eval_date=6, max_delay_periods=6
        )
        assert result["observed_counts"].shape == (5, 6)

    def test_observed_counts_non_negative(self):
        df = self._make_df()
        result = prepare_observed_triangle(
            df, "occurrence_period", "report_period", "exposure",
            ["feature_a"], eval_date=6, max_delay_periods=6
        )
        assert (result["observed_counts"] >= 0).all()

    def test_delays_beyond_max_excluded(self):
        df = pd.DataFrame({
            "occurrence_period": [0, 0],
            "report_period": [10.0, 2.0],  # delay=10 beyond max_delay=6
            "exposure": [1.0, 1.0],
            "feature_a": [0.5, 0.5],
        })
        result = prepare_observed_triangle(
            df, "occurrence_period", "report_period", "exposure",
            ["feature_a"], eval_date=6, max_delay_periods=6
        )
        # Only delay=2 should be counted
        assert result["observed_counts"][0, 2] == 1
        assert result["observed_counts"].sum() == 1

    def test_null_report_periods_ignored(self):
        df = pd.DataFrame({
            "occurrence_period": [0, 0, 0],
            "report_period": [2.0, np.nan, 3.0],
            "exposure": [1.0, 1.0, 1.0],
            "feature_a": [0.5, 0.5, 0.5],
        })
        result = prepare_observed_triangle(
            df, "occurrence_period", "report_period", "exposure",
            ["feature_a"], eval_date=6, max_delay_periods=6
        )
        # 2 reported claims
        assert result["observed_counts"].sum() == 2

    def test_exposure_by_period_positive(self):
        df = self._make_df()
        result = prepare_observed_triangle(
            df, "occurrence_period", "report_period", "exposure",
            ["feature_a"], eval_date=6, max_delay_periods=6
        )
        assert (result["exposure_by_period"] > 0).all()

    def test_features_by_period_shape(self):
        df = self._make_df()
        result = prepare_observed_triangle(
            df, "occurrence_period", "report_period", "exposure",
            ["feature_a"], eval_date=6, max_delay_periods=6
        )
        assert result["features_by_period"].shape == (5, 1)

    def test_zero_exposure_rows_excluded(self):
        df = pd.DataFrame({
            "occurrence_period": [0, 0],
            "report_period": [1.0, 2.0],
            "exposure": [0.0, 1.0],  # first has zero exposure
            "feature_a": [0.5, 0.5],
        })
        result = prepare_observed_triangle(
            df, "occurrence_period", "report_period", "exposure",
            ["feature_a"], eval_date=6, max_delay_periods=6
        )
        # Only rows with positive exposure
        assert result["exposure_by_period"][0] == 1.0

    def test_period_to_idx_complete(self):
        df = self._make_df(n_periods=4)
        result = prepare_observed_triangle(
            df, "occurrence_period", "report_period", "exposure",
            ["feature_a"], eval_date=6, max_delay_periods=6
        )
        assert set(result["period_to_idx"].keys()) == {0, 1, 2, 3}


# --- build_delay_training_data ---

class TestBuildDelayTrainingData:
    def test_output_shapes(self):
        n_periods = 4
        max_delay = 5
        features = np.random.rand(n_periods, 2)
        imputed = np.abs(np.random.randn(n_periods, max_delay)) + 0.1
        trunc = np.array([3, 4, 5, 5])
        X, y, w = build_delay_training_data(imputed, features, trunc, max_delay)
        assert X.ndim == 2
        assert X.shape[1] == 2
        assert y.ndim == 1
        assert w.ndim == 1
        assert len(X) == len(y) == len(w)

    def test_labels_in_range(self):
        n_periods, max_delay = 3, 6
        features = np.random.rand(n_periods, 2)
        imputed = np.ones((n_periods, max_delay))
        trunc = np.array([3, 4, 6])
        X, y, w = build_delay_training_data(imputed, features, trunc, max_delay)
        assert y.min() >= 0
        assert y.max() < max_delay

    def test_weights_positive(self):
        n_periods, max_delay = 3, 4
        features = np.random.rand(n_periods, 2)
        imputed = np.abs(np.random.randn(n_periods, max_delay)) + 0.01
        trunc = np.array([2, 3, 4])
        X, y, w = build_delay_training_data(imputed, features, trunc, max_delay)
        assert (w > 0).all()

    def test_zero_imputed_row_excluded(self):
        n_periods, max_delay = 2, 4
        features = np.random.rand(n_periods, 2)
        imputed = np.zeros((n_periods, max_delay))
        imputed[0, 2] = 5.0  # Only one non-zero entry
        trunc = np.array([3, 4])
        X, y, w = build_delay_training_data(imputed, features, trunc, max_delay)
        assert len(X) == 1
        assert y[0] == 2
        assert w[0] == 5.0
