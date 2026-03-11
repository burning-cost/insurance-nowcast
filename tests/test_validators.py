"""
Tests for input validation.
"""
import numpy as np
import pandas as pd
import pytest

from insurance_nowcast._validators import validate_fit_inputs, validate_feature_cols


def _make_valid_df(n=50):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "occurrence_period": rng.integers(0, 5, n),
        "report_period": (rng.integers(0, 5, n) + rng.integers(0, 3, n)).astype(float),
        "exposure": rng.uniform(0.5, 2.0, n),
        "feature_a": rng.uniform(0, 1, n),
        "feature_b": rng.integers(0, 3, n).astype(float),
    })


class TestValidateFitInputs:
    def _call(self, df, **overrides):
        kwargs = dict(
            occurrence_col="occurrence_period",
            report_col="report_period",
            exposure_col="exposure",
            feature_cols=["feature_a", "feature_b"],
            eval_date=10,
            max_delay_periods=12,
            exposure_offset=True,
        )
        kwargs.update(overrides)
        validate_fit_inputs(df=df, **kwargs)

    def test_valid_input_passes(self):
        df = _make_valid_df()
        self._call(df)  # Should not raise

    def test_not_dataframe_raises(self):
        with pytest.raises(ValueError, match="DataFrame"):
            self._call({"a": [1, 2]})

    def test_empty_dataframe_raises(self):
        df = pd.DataFrame(columns=["occurrence_period", "report_period", "exposure", "feature_a", "feature_b"])
        with pytest.raises(ValueError, match="empty"):
            self._call(df)

    def test_missing_occurrence_col_raises(self):
        df = _make_valid_df().rename(columns={"occurrence_period": "occ"})
        with pytest.raises(ValueError, match="missing"):
            self._call(df)

    def test_missing_report_col_raises(self):
        df = _make_valid_df().rename(columns={"report_period": "rep"})
        with pytest.raises(ValueError, match="missing"):
            self._call(df)

    def test_null_occurrence_raises(self):
        df = _make_valid_df()
        df.loc[0, "occurrence_period"] = np.nan
        with pytest.raises(ValueError, match="null"):
            self._call(df)

    def test_all_null_report_raises(self):
        df = _make_valid_df()
        df["report_period"] = np.nan
        with pytest.raises(ValueError, match="null for all"):
            self._call(df)

    def test_report_before_occurrence_raises(self):
        df = _make_valid_df()
        # Set report_period to be less than occurrence_period
        df.loc[0, "report_period"] = -100.0
        df.loc[0, "occurrence_period"] = 5
        with pytest.raises(ValueError, match="report_period < occurrence_period"):
            self._call(df)

    def test_null_exposure_raises(self):
        df = _make_valid_df()
        df.loc[0, "exposure"] = np.nan
        with pytest.raises(ValueError, match="null"):
            self._call(df)

    def test_all_nonpositive_exposure_raises(self):
        df = _make_valid_df()
        df["exposure"] = -1.0
        with pytest.raises(ValueError, match="non-positive for all"):
            self._call(df)

    def test_some_nonpositive_exposure_warns(self):
        df = _make_valid_df()
        df.loc[0, "exposure"] = -1.0
        with pytest.warns(UserWarning, match="non-positive exposure"):
            self._call(df)

    def test_max_delay_zero_raises(self):
        df = _make_valid_df()
        with pytest.raises(ValueError, match="max_delay_periods must be >= 1"):
            self._call(df, max_delay_periods=0)

    def test_single_occurrence_period_raises(self):
        df = _make_valid_df()
        df["occurrence_period"] = 0
        with pytest.raises(ValueError, match="Only 1 distinct"):
            self._call(df)

    def test_null_feature_warns(self):
        df = _make_valid_df()
        df.loc[0, "feature_a"] = np.nan
        with pytest.warns(UserWarning, match="null values"):
            self._call(df)

    def test_delay_boundary_warns(self):
        """Should warn when >10% of observed delays are at boundary."""
        rng = np.random.default_rng(0)
        n = 100
        df = pd.DataFrame({
            "occurrence_period": [0] * n,
            "report_period": [12.0] * n,  # all at boundary = 100%
            "exposure": [1.0] * n,
            "feature_a": rng.uniform(0, 1, n),
            "feature_b": [0.0] * n,
        })
        with pytest.warns(UserWarning, match="max_delay_periods"):
            self._call(df, eval_date=20, max_delay_periods=12)


class TestValidateFeatureCols:
    def test_numeric_passes(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3, 4]})
        validate_feature_cols(df, ["a", "b"])  # Should not raise

    def test_string_col_raises(self):
        df = pd.DataFrame({"a": ["x", "y"]})
        with pytest.raises(ValueError, match="Non-numeric"):
            validate_feature_cols(df, ["a"])

    def test_object_col_raises(self):
        df = pd.DataFrame({"a": [object(), object()]})
        with pytest.raises(ValueError, match="Non-numeric"):
            validate_feature_cols(df, ["a"])

    def test_int_col_passes(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        validate_feature_cols(df, ["a"])  # Should not raise

    def test_bool_col_passes(self):
        """Boolean columns are numeric in pandas."""
        df = pd.DataFrame({"a": [True, False, True]})
        validate_feature_cols(df, ["a"])  # Should not raise
