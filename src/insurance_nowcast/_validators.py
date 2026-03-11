"""
Input validation for ReportingDelayModel.

Validates the input DataFrame and parameters before fitting. Provides clear,
actionable error messages — the kind a pricing actuary can act on without
needing to read the source code.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd


def validate_fit_inputs(
    df: pd.DataFrame,
    occurrence_col: str,
    report_col: str,
    exposure_col: str,
    feature_cols: list[str],
    eval_date: Any,
    max_delay_periods: int,
    exposure_offset: bool,
) -> None:
    """
    Validate all inputs to ReportingDelayModel.fit().

    Raises ValueError with a clear message if any check fails.
    Issues warnings for likely data problems that won't prevent fitting.

    Parameters
    ----------
    df : pd.DataFrame
        The input claims DataFrame.
    occurrence_col : str
        Column name for occurrence period.
    report_col : str
        Column name for report period (nullable for IBNR).
    exposure_col : str
        Column name for exposure (policy-years at risk).
    feature_cols : list[str]
        List of covariate column names.
    eval_date : int or comparable
        Evaluation date — the date at which the data is truncated.
    max_delay_periods : int
        Maximum observable reporting delay in periods.
    exposure_offset : bool
        Whether to use exposure as an offset in the occurrence model.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"df must be a pandas DataFrame, got {type(df).__name__}")

    if len(df) == 0:
        raise ValueError("df is empty — no data to fit on")

    # Check required columns exist
    required_cols = [occurrence_col, report_col, exposure_col] + list(feature_cols)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columns missing from df: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Occurrence period checks
    occ = df[occurrence_col]
    if occ.isna().any():
        raise ValueError(
            f"occurrence_col '{occurrence_col}' has {occ.isna().sum()} null values. "
            "Every row must have an occurrence period."
        )

    # Report period: null is allowed (= IBNR / not yet reported)
    rep = df[report_col]
    fully_reported_mask = rep.notna()
    if fully_reported_mask.sum() == 0:
        raise ValueError(
            f"report_col '{report_col}' is null for all rows. "
            "Need at least some observed claims to fit the delay distribution."
        )

    # Check report >= occurrence for observed claims
    obs = df[fully_reported_mask]
    if (obs[report_col] < obs[occurrence_col]).any():
        n_bad = (obs[report_col] < obs[occurrence_col]).sum()
        raise ValueError(
            f"{n_bad} rows have report_period < occurrence_period. "
            "A claim cannot be reported before it occurs."
        )

    # Exposure checks
    exp = df[exposure_col]
    if exp.isna().any():
        raise ValueError(
            f"exposure_col '{exposure_col}' has {exp.isna().sum()} null values. "
            "All rows need an exposure value."
        )
    if (exp <= 0).any():
        n_bad = (exp <= 0).sum()
        if n_bad == len(df):
            raise ValueError(
                f"exposure_col '{exposure_col}' is non-positive for all rows."
            )
        warnings.warn(
            f"{n_bad} rows have non-positive exposure. These rows will be excluded.",
            UserWarning,
            stacklevel=4,
        )

    # Feature column checks
    for col in feature_cols:
        if df[col].isna().any():
            n_null = df[col].isna().sum()
            warnings.warn(
                f"feature_col '{col}' has {n_null} null values. "
                "Consider imputing before fitting.",
                UserWarning,
                stacklevel=4,
            )

    # eval_date must be comparable to occurrence_col values
    try:
        _ = eval_date >= df[occurrence_col].max()
    except TypeError:
        raise ValueError(
            f"eval_date ({eval_date!r}) is not comparable to occurrence_col values "
            f"(dtype: {df[occurrence_col].dtype}). Use the same type for both."
        )

    # eval_date should be >= max occurrence period
    max_occ = df[occurrence_col].max()
    if eval_date < max_occ:
        warnings.warn(
            f"eval_date ({eval_date}) is before the latest occurrence period "
            f"({max_occ}). This means recent occurrence periods extend beyond "
            "the evaluation date — is this intentional?",
            UserWarning,
            stacklevel=4,
        )

    # max_delay_periods check
    if max_delay_periods < 1:
        raise ValueError(f"max_delay_periods must be >= 1, got {max_delay_periods}")
    if max_delay_periods > 120:
        warnings.warn(
            f"max_delay_periods={max_delay_periods} is very large (>10 years). "
            "Consider whether this is appropriate for your line of business.",
            UserWarning,
            stacklevel=4,
        )

    # Check whether delay boundary is being hit
    if fully_reported_mask.sum() > 0:
        delays = obs[report_col] - obs[occurrence_col]
        pct_at_boundary = (delays >= max_delay_periods).mean()
        if pct_at_boundary > 0.10:
            warnings.warn(
                f"{pct_at_boundary:.0%} of observed delays are at or beyond "
                f"max_delay_periods={max_delay_periods}. This suggests the "
                "maximum delay is too short — IBNR estimates will be understated. "
                "Consider increasing max_delay_periods.",
                UserWarning,
                stacklevel=4,
            )

    # Minimum data checks
    n_occurrence_periods = df[occurrence_col].nunique()
    if n_occurrence_periods < 2:
        raise ValueError(
            f"Only {n_occurrence_periods} distinct occurrence period(s) found. "
            "Need at least 2 to fit a development pattern."
        )

    fully_observed_periods = (
        df[fully_reported_mask][occurrence_col].nunique()
    )
    if fully_observed_periods == 0:
        raise ValueError(
            "No occurrence periods have any reported claims. "
            "Cannot initialise delay distribution."
        )


def validate_feature_cols(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Check that feature columns are numeric or can be encoded."""
    non_numeric = []
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric.append(col)
    if non_numeric:
        raise ValueError(
            f"Feature columns must be numeric. Non-numeric columns: {non_numeric}. "
            "Encode categorical variables before passing to fit() "
            "(e.g., pd.get_dummies() or sklearn LabelEncoder)."
        )
