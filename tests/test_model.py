"""
Integration tests for ReportingDelayModel.

These are the highest-value tests — they exercise the full EM pipeline
and check that outputs are sensible (not necessarily statistically correct,
since that requires large datasets and many replications).
"""
import numpy as np
import pandas as pd
import pytest

from insurance_nowcast import ReportingDelayModel, NowcastSimulator


@pytest.fixture
def sim_data():
    sim = NowcastSimulator(
        n_occurrence_periods=15,
        max_delay_periods=8,
        base_frequency=0.08,
        delay_shape="geometric",
        random_state=42,
    )
    df = sim.generate(n_policies=300, eval_period=14)
    return df, sim


@pytest.fixture
def fitted_glm(sim_data):
    df, _ = sim_data
    model = ReportingDelayModel(
        occurrence_model="glm",
        delay_model="glm",
        max_delay_periods=8,
        max_em_iterations=5,
        n_bootstrap=0,
        random_state=42,
    )
    model.fit(
        df,
        occurrence_col="occurrence_period",
        report_col="report_period",
        exposure_col="exposure",
        feature_cols=["age_group", "risk_score", "channel"],
        eval_date=14,
    )
    return model


@pytest.fixture
def fitted_xgb(sim_data):
    df, _ = sim_data
    model = ReportingDelayModel(
        occurrence_model="xgboost",
        delay_model="xgboost",
        max_delay_periods=8,
        max_em_iterations=5,
        n_bootstrap=0,
        random_state=42,
    )
    model.fit(
        df,
        occurrence_col="occurrence_period",
        report_col="report_period",
        exposure_col="exposure",
        feature_cols=["age_group", "risk_score", "channel"],
        eval_date=14,
    )
    return model


# ---- Constructor tests ----

class TestReportingDelayModelConstructor:
    def test_default_construction(self):
        model = ReportingDelayModel()
        assert model.occurrence_model == "xgboost"
        assert model.delay_model == "xgboost"
        assert model.max_delay_periods == 24
        assert model.exposure_offset is True

    def test_custom_params(self):
        model = ReportingDelayModel(
            occurrence_model="glm",
            delay_model="glm",
            max_delay_periods=12,
            em_patience=5,
        )
        assert model.occurrence_model == "glm"
        assert model.max_delay_periods == 12

    def test_repr_unfitted(self):
        model = ReportingDelayModel()
        assert "unfitted" in repr(model)

    def test_repr_fitted(self, fitted_glm):
        assert "fitted" in repr(fitted_glm)


# ---- fit() tests ----

class TestReportingDelayModelFit:
    def test_fit_returns_self(self, sim_data):
        df, _ = sim_data
        model = ReportingDelayModel(max_em_iterations=3, n_bootstrap=0)
        result = model.fit(
            df, "occurrence_period", "report_period", "exposure",
            ["age_group", "risk_score", "channel"], eval_date=14
        )
        assert result is model

    def test_is_fitted_after_fit(self, fitted_glm):
        assert fitted_glm._is_fitted is True

    def test_convergence_history_populated(self, fitted_glm):
        assert len(fitted_glm.convergence_history_) > 0

    def test_convergence_history_finite(self, fitted_glm):
        for ll in fitted_glm.convergence_history_:
            assert np.isfinite(ll)

    def test_n_iter_positive(self, fitted_glm):
        assert fitted_glm.n_iter_ >= 1

    def test_p_hat_shape(self, fitted_glm):
        n_periods = len(fitted_glm.occurrence_periods_)
        assert fitted_glm.p_hat_.shape == (n_periods, 8)

    def test_p_hat_sums_to_one(self, fitted_glm):
        row_sums = fitted_glm.p_hat_.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_p_hat_non_negative(self, fitted_glm):
        assert (fitted_glm.p_hat_ >= 0).all()

    def test_lambda_hat_positive(self, fitted_glm):
        assert (fitted_glm.lambda_hat_ > 0).all()

    def test_occurrence_periods_sorted(self, fitted_glm):
        periods = fitted_glm.occurrence_periods_
        assert (periods[1:] >= periods[:-1]).all()

    def test_fit_without_features(self, sim_data):
        """Should work with empty feature_cols (intercept-only model)."""
        df, _ = sim_data
        model = ReportingDelayModel(max_em_iterations=3, n_bootstrap=0)
        model.fit(
            df, "occurrence_period", "report_period", "exposure",
            feature_cols=[], eval_date=14
        )
        assert model._is_fitted

    def test_xgb_fit(self, fitted_xgb):
        assert fitted_xgb._is_fitted
        assert len(fitted_xgb.convergence_history_) > 0

    def test_fit_raises_before_validation_error(self, sim_data):
        df, _ = sim_data
        model = ReportingDelayModel()
        with pytest.raises(ValueError):
            model.fit(
                df,
                occurrence_col="nonexistent_col",
                report_col="report_period",
                exposure_col="exposure",
                feature_cols=[],
                eval_date=14,
            )

    def test_fit_with_glm_occurrence_xgb_delay(self, sim_data):
        df, _ = sim_data
        model = ReportingDelayModel(
            occurrence_model="glm",
            delay_model="xgboost",
            max_em_iterations=3,
            n_bootstrap=0,
        )
        model.fit(
            df, "occurrence_period", "report_period", "exposure",
            ["age_group"], eval_date=14
        )
        assert model._is_fitted

    def test_fit_with_xgb_occurrence_glm_delay(self, sim_data):
        df, _ = sim_data
        model = ReportingDelayModel(
            occurrence_model="xgboost",
            delay_model="glm",
            max_em_iterations=3,
            n_bootstrap=0,
        )
        model.fit(
            df, "occurrence_period", "report_period", "exposure",
            ["age_group"], eval_date=14
        )
        assert model._is_fitted


# ---- predict_completion_factors() tests ----

class TestPredictCompletionFactors:
    def test_returns_dataframe(self, fitted_glm, sim_data):
        df, _ = sim_data
        result = fitted_glm.predict_completion_factors(include_ci=False)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self, fitted_glm):
        result = fitted_glm.predict_completion_factors(include_ci=False)
        for col in ["occurrence_period", "completion_factor", "observed_claims",
                    "estimated_ultimate_claims", "ibnr_count"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_completion_factors_ge_one(self, fitted_glm):
        result = fitted_glm.predict_completion_factors(include_ci=False)
        assert (result["completion_factor"] >= 1.0).all()

    def test_oldest_period_cf_near_one(self, fitted_glm):
        result = fitted_glm.predict_completion_factors(include_ci=False)
        oldest = result.sort_values("occurrence_period").iloc[0]
        assert oldest["completion_factor"] < 2.0, \
            f"Oldest period CF should be ~1, got {oldest['completion_factor']}"

    def test_latest_period_has_high_cf(self, fitted_glm):
        result = fitted_glm.predict_completion_factors(include_ci=False)
        latest = result.sort_values("occurrence_period").iloc[-1]
        assert latest["completion_factor"] > 1.0

    def test_completion_factor_finite(self, fitted_glm):
        result = fitted_glm.predict_completion_factors(include_ci=False)
        assert result["completion_factor"].isna().sum() == 0
        assert np.isfinite(result["completion_factor"].values).all()

    def test_ibnr_non_negative(self, fitted_glm):
        result = fitted_glm.predict_completion_factors(include_ci=False)
        assert (result["ibnr_count"] >= 0).all()

    def test_estimated_ultimate_ge_observed(self, fitted_glm):
        result = fitted_glm.predict_completion_factors(include_ci=False)
        assert (result["estimated_ultimate_claims"] >= result["observed_claims"] - 0.01).all()

    def test_one_row_per_period(self, fitted_glm):
        result = fitted_glm.predict_completion_factors(include_ci=False)
        n_periods = len(fitted_glm.occurrence_periods_)
        assert len(result) == n_periods

    def test_by_segment_returns_more_rows(self, fitted_glm, sim_data):
        df, _ = sim_data
        result_no_by = fitted_glm.predict_completion_factors(include_ci=False)
        result_with_by = fitted_glm.predict_completion_factors(
            df=df, by=["channel"], include_ci=False
        )
        assert len(result_with_by) >= len(result_no_by)

    def test_predict_without_fit_raises(self, sim_data):
        df, _ = sim_data
        model = ReportingDelayModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict_completion_factors()


# ---- predict_ibnr() tests ----

class TestPredictIbnr:
    def test_returns_dataframe(self, fitted_glm):
        result = fitted_glm.predict_ibnr()
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self, fitted_glm):
        result = fitted_glm.predict_ibnr()
        for col in ["occurrence_period", "ibnr_count", "ibnr_fraction",
                    "observed_claims", "estimated_ultimate_claims"]:
            assert col in result.columns

    def test_ibnr_count_non_negative(self, fitted_glm):
        result = fitted_glm.predict_ibnr()
        assert (result["ibnr_count"] >= 0).all()

    def test_ibnr_fraction_in_zero_one(self, fitted_glm):
        result = fitted_glm.predict_ibnr()
        assert (result["ibnr_fraction"] >= 0).all()
        assert (result["ibnr_fraction"] <= 1).all()

    def test_recent_periods_have_ibnr(self, fitted_glm):
        """Most recent periods should have non-trivial IBNR."""
        result = fitted_glm.predict_ibnr()
        recent = result.sort_values("occurrence_period").tail(3)
        assert recent["ibnr_fraction"].sum() > 0

    def test_predict_ibnr_without_fit_raises(self):
        model = ReportingDelayModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict_ibnr()


# ---- predict_development_pattern() tests ----

class TestPredictDevelopmentPattern:
    def test_returns_array(self, fitted_glm, sim_data):
        df, _ = sim_data
        X = df[["age_group", "risk_score", "channel"]].head(5)
        result = fitted_glm.predict_development_pattern(X)
        assert isinstance(result, np.ndarray)

    def test_shape(self, fitted_glm, sim_data):
        df, _ = sim_data
        n = 10
        X = df[["age_group", "risk_score", "channel"]].head(n)
        result = fitted_glm.predict_development_pattern(X)
        assert result.shape == (n, 8)

    def test_sums_to_one(self, fitted_glm, sim_data):
        df, _ = sim_data
        X = df[["age_group", "risk_score", "channel"]].head(20)
        result = fitted_glm.predict_development_pattern(X)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-5)

    def test_non_negative(self, fitted_glm, sim_data):
        df, _ = sim_data
        X = df[["age_group", "risk_score", "channel"]].head(20)
        result = fitted_glm.predict_development_pattern(X)
        assert (result >= 0).all()

    def test_missing_feature_raises(self, fitted_glm, sim_data):
        df, _ = sim_data
        X = df[["age_group"]].head(5)  # Missing risk_score, channel
        with pytest.raises(ValueError, match="missing"):
            fitted_glm.predict_development_pattern(X)

    def test_without_fit_raises(self, sim_data):
        df, _ = sim_data
        model = ReportingDelayModel()
        with pytest.raises(RuntimeError):
            model.predict_development_pattern(df.head(5))


# ---- Convergence behaviour ----

class TestConvergenceBehaviour:
    def test_convergence_history_length_le_max_iter(self, sim_data):
        df, _ = sim_data
        max_iter = 10
        model = ReportingDelayModel(
            occurrence_model="glm",
            delay_model="glm",
            max_em_iterations=max_iter,
            n_bootstrap=0,
        )
        model.fit(
            df, "occurrence_period", "report_period", "exposure",
            ["age_group"], eval_date=14
        )
        assert len(model.convergence_history_) <= max_iter

    def test_patience_stops_early(self, sim_data):
        """With patience=1, model should stop after 2 non-improving iterations."""
        df, _ = sim_data
        model = ReportingDelayModel(
            occurrence_model="glm",
            delay_model="glm",
            em_patience=2,
            convergence_tol=1e10,  # Very high tolerance = always "not improving"
            max_em_iterations=20,
            n_bootstrap=0,
        )
        model.fit(
            df, "occurrence_period", "report_period", "exposure",
            ["age_group"], eval_date=14
        )
        assert len(model.convergence_history_) <= 5  # Should stop early

    def test_xgb_convergence_history(self, fitted_xgb):
        for ll in fitted_xgb.convergence_history_:
            assert np.isfinite(ll)
