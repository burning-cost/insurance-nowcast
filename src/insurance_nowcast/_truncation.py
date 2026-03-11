"""
Truncation computation and data preparation for the EM algorithm.

The key operation here is computing tau_i — the truncation depth for each
occurrence period. This tells us how many delay periods are observable for
each risk unit given the evaluation date. Everything beyond tau_i is censored
and must be imputed in the E-step.

The design principle: get this right once, test it thoroughly, and never
touch it again. Truncation bugs are the most common cause of incorrect IBNR
estimates in practice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any


def compute_delay(
    occurrence_period: pd.Series,
    report_period: pd.Series,
) -> pd.Series:
    """
    Compute the delay (in periods) between occurrence and report.

    Parameters
    ----------
    occurrence_period : pd.Series
        Occurrence period — integer or comparable type.
    report_period : pd.Series
        Report period — same type as occurrence_period. Null for IBNR.

    Returns
    -------
    pd.Series
        Delay in periods. Null where report_period is null.
    """
    delay = report_period - occurrence_period
    return delay


def compute_truncation_depth(
    occurrence_period: pd.Series,
    eval_date: Any,
    max_delay_periods: int,
) -> pd.Series:
    """
    Compute tau_i — the number of observable delay periods for each row.

    For occurrence period i evaluated at eval_date, the number of delay periods
    we can observe is min(max_delay_periods, eval_date - i). A value of d means
    delays 0, 1, ..., d-1 are potentially observable.

    Parameters
    ----------
    occurrence_period : pd.Series
        Occurrence period for each row.
    eval_date : int or comparable
        The evaluation date (must be same type as occurrence_period).
    max_delay_periods : int
        Maximum delay periods the model can represent.

    Returns
    -------
    pd.Series
        Truncation depth tau_i >= 1 for each row. Capped at max_delay_periods.
    """
    elapsed = eval_date - occurrence_period
    # elapsed = 0 means we're in the same period as eval_date — 1 delay period observable
    # elapsed = 1 means 2 delay periods observable (0 and 1), etc.
    tau = (elapsed + 1).clip(lower=1, upper=max_delay_periods)
    return tau.astype(int)


def prepare_observed_triangle(
    df: pd.DataFrame,
    occurrence_col: str,
    report_col: str,
    exposure_col: str,
    feature_cols: list[str],
    eval_date: Any,
    max_delay_periods: int,
) -> dict:
    """
    Convert raw claims DataFrame into the structured data the EM algorithm needs.

    This is the central data preparation step. The output is a dictionary
    containing:

    - ``observed_counts`` : shape (n_periods, max_delay_periods) — N_{i,j}, the
      raw observed claims count for occurrence period i and delay j.
    - ``occurrence_periods`` : sorted array of unique occurrence periods.
    - ``truncation_depth`` : shape (n_periods,) — tau_i for each occurrence period.
    - ``exposure_by_period`` : shape (n_periods,) — total exposure by occurrence period.
    - ``features_by_period`` : shape (n_periods, n_features) — mean features per period.
    - ``period_to_idx`` : dict mapping occurrence period value to row index.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame — one row per risk-occurrence-period observation.
    occurrence_col : str
        Column with occurrence period.
    report_col : str
        Column with report period (nullable).
    exposure_col : str
        Column with exposure.
    feature_cols : list[str]
        Covariate columns.
    eval_date : comparable
        Evaluation date.
    max_delay_periods : int
        Maximum delay periods.

    Returns
    -------
    dict
        Structured data for the EM algorithm.
    """
    # Work with positive-exposure rows only
    valid_mask = df[exposure_col] > 0
    df = df[valid_mask].copy()

    occurrence_periods = np.sort(df[occurrence_col].unique())
    n_periods = len(occurrence_periods)
    period_to_idx = {p: i for i, p in enumerate(occurrence_periods)}

    # Build observed triangle N_{i,j}
    observed_counts = np.zeros((n_periods, max_delay_periods), dtype=float)

    # Only use observed (reported) claims to fill the triangle
    reported_mask = df[report_col].notna()
    reported = df[reported_mask].copy()

    if len(reported) > 0:
        reported["_delay"] = reported[report_col] - reported[occurrence_col]
        reported["_period_idx"] = reported[occurrence_col].map(period_to_idx)

        # Only include delays within model window
        in_window = reported["_delay"] < max_delay_periods
        reported = reported[in_window]

        # For each row, add 1 claim to the triangle
        # (each row = one claim event in the input data)
        for _, row in reported.iterrows():
            i = int(row["_period_idx"])
            j = int(row["_delay"])
            observed_counts[i, j] += 1.0

    # Compute truncation depth per period
    occ_series = pd.Series(occurrence_periods)
    tau_series = compute_truncation_depth(occ_series, eval_date, max_delay_periods)
    truncation_depth = tau_series.values  # shape (n_periods,)

    # Exposure per occurrence period — sum across all risk units in that period
    exposure_by_period = (
        df.groupby(occurrence_col)[exposure_col].sum().loc[occurrence_periods].values
    )

    # Features per occurrence period — exposure-weighted mean
    feature_matrix = np.zeros((n_periods, len(feature_cols)), dtype=float)
    if feature_cols:
        for period, idx in period_to_idx.items():
            period_rows = df[df[occurrence_col] == period]
            if len(period_rows) == 0:
                continue
            exp = period_rows[exposure_col].values
            exp_sum = exp.sum()
            if exp_sum > 0:
                for k, fcol in enumerate(feature_cols):
                    feature_matrix[idx, k] = (period_rows[fcol].values * exp).sum() / exp_sum

    # Also store per-row features for prediction purposes
    # These are the raw features before period aggregation
    row_features = df[feature_cols].values if feature_cols else np.zeros((len(df), 0))
    row_occurrence_idx = df[occurrence_col].map(period_to_idx).values
    row_exposure = df[exposure_col].values

    return {
        "observed_counts": observed_counts,           # (n_periods, max_delay)
        "occurrence_periods": occurrence_periods,      # (n_periods,)
        "truncation_depth": truncation_depth,          # (n_periods,)
        "exposure_by_period": exposure_by_period,      # (n_periods,)
        "features_by_period": feature_matrix,          # (n_periods, n_features)
        "period_to_idx": period_to_idx,
        "n_periods": n_periods,
        "max_delay_periods": max_delay_periods,
        "feature_cols": feature_cols,
        # Per-row data for XGBoost fitting
        "row_features": row_features,                  # (n_rows, n_features)
        "row_occurrence_idx": row_occurrence_idx,      # (n_rows,) -> period index
        "row_exposure": row_exposure,                  # (n_rows,)
        "n_rows": len(df),
    }


def build_delay_training_data(
    imputed_counts: np.ndarray,
    features_by_period: np.ndarray,
    truncation_depth: np.ndarray,
    max_delay_periods: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the training data for the delay M-step.

    For multinomial regression with XGBoost's multi:softprob, we need:
    - One row per (occurrence_period, delay_period) pair that is observed or imputed
    - Features: the risk features for that occurrence period
    - Label: the delay period index (0 to max_delay_periods - 1)
    - Weight: the imputed count N_{i,j}^{(k)}

    This reshaping is the key to making XGBoost's per-observation weighting work
    with a multinomial output.

    Parameters
    ----------
    imputed_counts : np.ndarray
        Shape (n_periods, max_delay_periods). Full imputed count matrix.
    features_by_period : np.ndarray
        Shape (n_periods, n_features). Features per occurrence period.
    truncation_depth : np.ndarray
        Shape (n_periods,). tau_i for each period (not used for filtering here —
        all periods contribute, observed + imputed).
    max_delay_periods : int
        Number of delay periods.

    Returns
    -------
    X : np.ndarray, shape (n_rows, n_features)
    y : np.ndarray, shape (n_rows,) — delay period index (int)
    w : np.ndarray, shape (n_rows,) — sample weights (imputed counts)
    """
    n_periods = len(features_by_period)
    rows_X = []
    rows_y = []
    rows_w = []

    for i in range(n_periods):
        total_i = imputed_counts[i].sum()
        if total_i <= 0:
            continue
        for j in range(max_delay_periods):
            w_ij = imputed_counts[i, j]
            if w_ij <= 0:
                continue
            rows_X.append(features_by_period[i])
            rows_y.append(j)
            rows_w.append(w_ij)

    if not rows_X:
        # Fall back to uniform — shouldn't happen after validation
        X = features_by_period
        y = np.zeros(n_periods, dtype=int)
        w = np.ones(n_periods, dtype=float)
    else:
        X = np.array(rows_X, dtype=float)
        y = np.array(rows_y, dtype=int)
        w = np.array(rows_w, dtype=float)

    return X, y, w
