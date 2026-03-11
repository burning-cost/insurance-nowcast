"""
Tests for NowcastSimulator.
"""
import numpy as np
import pandas as pd
import pytest

from insurance_nowcast import NowcastSimulator


class TestNowcastSimulator:
    def test_basic_generation(self):
        sim = NowcastSimulator(n_occurrence_periods=10, max_delay_periods=5, random_state=0)
        df = sim.generate(n_policies=100, eval_period=9)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_required_columns(self):
        sim = NowcastSimulator(random_state=0)
        df = sim.generate(n_policies=100, eval_period=10)
        assert "occurrence_period" in df.columns
        assert "report_period" in df.columns
        assert "exposure" in df.columns

    def test_feature_columns_present(self):
        sim = NowcastSimulator(random_state=0)
        df = sim.generate(n_policies=100, eval_period=10)
        assert "age_group" in df.columns
        assert "risk_score" in df.columns
        assert "channel" in df.columns

    def test_occurrence_periods_in_range(self):
        n_periods = 12
        sim = NowcastSimulator(n_occurrence_periods=n_periods, random_state=0)
        df = sim.generate(n_policies=100, eval_period=n_periods - 1)
        assert df["occurrence_period"].min() >= 0
        assert df["occurrence_period"].max() < n_periods

    def test_report_period_null_for_ibnr(self):
        sim = NowcastSimulator(n_occurrence_periods=20, max_delay_periods=8, random_state=0)
        df = sim.generate(n_policies=200, eval_period=10)
        # Some recent-period claims should be IBNR (null report_period)
        has_ibnr = df["report_period"].isna().any()
        assert has_ibnr, "No IBNR claims generated — check eval_period"

    def test_report_period_not_before_occurrence(self):
        sim = NowcastSimulator(random_state=0)
        df = sim.generate(n_policies=200, eval_period=10)
        reported = df[df["report_period"].notna()]
        assert (reported["report_period"] >= reported["occurrence_period"]).all()

    def test_exposure_positive(self):
        sim = NowcastSimulator(random_state=0)
        df = sim.generate(n_policies=100, eval_period=10)
        assert (df["exposure"] > 0).all()

    def test_reproducible_with_seed(self):
        sim1 = NowcastSimulator(random_state=42)
        df1 = sim1.generate(n_policies=100, eval_period=10)
        sim2 = NowcastSimulator(random_state=42)
        df2 = sim2.generate(n_policies=100, eval_period=10)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        sim1 = NowcastSimulator(random_state=1)
        df1 = sim1.generate(n_policies=100, eval_period=10)
        sim2 = NowcastSimulator(random_state=999)
        df2 = sim2.generate(n_policies=100, eval_period=10)
        # Should not be identical
        assert not df1["occurrence_period"].equals(df2["occurrence_period"]) or \
               not df1["exposure"].equals(df2["exposure"])

    def test_geometric_delay_shape(self):
        sim = NowcastSimulator(delay_shape="geometric", random_state=0)
        df = sim.generate(n_policies=500, eval_period=20)
        assert len(df) > 0

    def test_logistic_delay_shape(self):
        sim = NowcastSimulator(delay_shape="logistic", random_state=0)
        df = sim.generate(n_policies=200, eval_period=10)
        assert len(df) > 0

    def test_mixed_delay_shape(self):
        sim = NowcastSimulator(delay_shape="mixed", random_state=0)
        df = sim.generate(n_policies=200, eval_period=10)
        assert len(df) > 0

    def test_invalid_delay_shape_raises(self):
        sim = NowcastSimulator(delay_shape="invalid_shape", random_state=0)
        with pytest.raises(ValueError, match="Unknown delay_shape"):
            sim.generate(n_policies=50, eval_period=5)

    def test_feature_effects_increase_frequency(self):
        """Higher feature effect should increase mean claim count."""
        sim_low = NowcastSimulator(
            feature_effects={"age_group": 0.0},
            base_frequency=0.1,
            random_state=42,
        )
        sim_high = NowcastSimulator(
            feature_effects={"age_group": 1.0},
            base_frequency=0.1,
            random_state=42,
        )
        df_low = sim_low.generate(n_policies=500, eval_period=20)
        df_high = sim_high.generate(n_policies=500, eval_period=20)
        assert len(df_high) > len(df_low)

    def test_get_true_completion_factors_shape(self):
        sim = NowcastSimulator(n_occurrence_periods=12, max_delay_periods=6, random_state=0)
        cf = sim.get_true_completion_factors(eval_period=11)
        assert isinstance(cf, pd.DataFrame)
        assert len(cf) == 12

    def test_true_completion_factors_columns(self):
        sim = NowcastSimulator(random_state=0)
        cf = sim.get_true_completion_factors(eval_period=10)
        assert "occurrence_period" in cf.columns
        assert "true_completion_factor" in cf.columns
        assert "true_development_fraction" in cf.columns

    def test_true_cf_first_period_near_one(self):
        """Fully observed period should have completion factor near 1."""
        sim = NowcastSimulator(
            n_occurrence_periods=12,
            max_delay_periods=6,
            random_state=0,
        )
        cf = sim.get_true_completion_factors(eval_period=11)
        # Period 0 has max observation (11 elapsed periods vs 6 max delay)
        cf_period0 = cf[cf["occurrence_period"] == 0]["true_completion_factor"].values[0]
        assert cf_period0 < 1.5, f"Completion factor for oldest period should be ~1, got {cf_period0}"

    def test_true_cf_latest_period_large(self):
        """Most recent period should have large completion factor."""
        sim = NowcastSimulator(n_occurrence_periods=12, max_delay_periods=6, random_state=0)
        cf = sim.get_true_completion_factors(eval_period=11)
        cf_latest = cf[cf["occurrence_period"] == 11]["true_completion_factor"].values[0]
        assert cf_latest >= 1.0

    def test_development_fractions_between_zero_and_one(self):
        sim = NowcastSimulator(random_state=0)
        cf = sim.get_true_completion_factors(eval_period=15)
        fracs = cf["true_development_fraction"]
        assert (fracs > 0).all()
        assert (fracs <= 1.0).all()

    def test_exposure_variability(self):
        """Variable exposure should produce non-uniform exposure distribution."""
        sim = NowcastSimulator(random_state=0)
        df = sim.generate(n_policies=200, eval_period=10, exposure_variability=0.5)
        exposure_std = df["exposure"].std()
        assert exposure_std > 0
