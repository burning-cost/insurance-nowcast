"""
Shared fixtures for the insurance-nowcast test suite.

All fixtures use NowcastSimulator to generate synthetic data with known DGP.
This ensures tests are deterministic and can check parameter recovery.
"""
import numpy as np
import pandas as pd
import pytest

from insurance_nowcast import NowcastSimulator, ReportingDelayModel


# --- Small dataset fixtures (fast, used in unit tests) ---

@pytest.fixture
def small_sim():
    """Small simulator: 12 periods, 6 max delay, 200 policies."""
    return NowcastSimulator(
        n_occurrence_periods=12,
        max_delay_periods=6,
        base_frequency=0.10,
        delay_shape="geometric",
        random_state=42,
    )


@pytest.fixture
def small_df(small_sim):
    """Small synthetic claims dataset."""
    return small_sim.generate(n_policies=200, eval_period=11)


@pytest.fixture
def medium_sim():
    """Medium simulator: 24 periods, 12 max delay, 500 policies."""
    return NowcastSimulator(
        n_occurrence_periods=24,
        max_delay_periods=12,
        base_frequency=0.08,
        delay_shape="geometric",
        random_state=42,
    )


@pytest.fixture
def medium_df(medium_sim):
    """Medium synthetic claims dataset."""
    return medium_sim.generate(n_policies=500, eval_period=23)


@pytest.fixture
def glm_model_small(small_df):
    """Fitted GLM model on small dataset."""
    model = ReportingDelayModel(
        occurrence_model="glm",
        delay_model="glm",
        max_delay_periods=6,
        n_bootstrap=0,
        max_em_iterations=5,
        random_state=42,
    )
    model.fit(
        small_df,
        occurrence_col="occurrence_period",
        report_col="report_period",
        exposure_col="exposure",
        feature_cols=["age_group", "risk_score", "channel"],
        eval_date=11,
    )
    return model


@pytest.fixture
def xgb_model_small(small_df):
    """Fitted XGBoost model on small dataset."""
    model = ReportingDelayModel(
        occurrence_model="xgboost",
        delay_model="xgboost",
        max_delay_periods=6,
        n_bootstrap=0,
        max_em_iterations=5,
        random_state=42,
    )
    model.fit(
        small_df,
        occurrence_col="occurrence_period",
        report_col="report_period",
        exposure_col="exposure",
        feature_cols=["age_group", "risk_score", "channel"],
        eval_date=11,
    )
    return model


@pytest.fixture
def minimal_df():
    """Minimal valid DataFrame for testing edge cases."""
    rng = np.random.default_rng(0)
    n = 50
    return pd.DataFrame({
        "occurrence_period": rng.integers(0, 5, size=n),
        "report_period": rng.integers(0, 8, size=n).astype(float),
        "exposure": rng.uniform(0.5, 2.0, size=n),
        "feature_a": rng.uniform(0, 1, size=n),
        "feature_b": rng.integers(0, 3, size=n).astype(float),
    })
