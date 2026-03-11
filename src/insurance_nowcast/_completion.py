"""
Completion factor and IBNR prediction.

Given a fitted ReportingDelayModel, these functions compute:
- Completion factors: the fraction of claims in each occurrence period that
  have been reported by the evaluation date, and its inverse (the multiplier
  to scale observed claims to estimated ultimate claims).
- IBNR counts: expected unreported claims per risk unit or per occurrence period.

For pricing actuaries: the completion factor tells you "multiply observed claims
by this to get ultimate claims". A completion factor of 1.25 means 25% of claims
in that period are still unreported at the evaluation date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any


def predict_completion_factors(
    occurrence_periods: np.ndarray,
    truncation_depth: np.ndarray,
    p_hat: np.ndarray,
    observed_counts: np.ndarray,
    lambda_hat: np.ndarray,
    feature_data: dict,
    by: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute completion factors by occurrence period.

    The completion factor for period i is:
        CF_i = 1 / F_i = 1 / sum_{j < tau_i} p_{i,j}

    where F_i is the fraction of claims expected to be reported by tau_i
    delay periods. CF_i >= 1 always (more claims ultimately than observed).

    For the most recent periods (small tau_i), CF_i is large. For periods
    near the start of the history (large tau_i), CF_i approaches 1.0.

    Parameters
    ----------
    occurrence_periods : np.ndarray
        Shape (n_periods,). The occurrence period values.
    truncation_depth : np.ndarray
        Shape (n_periods,). tau_i for each period.
    p_hat : np.ndarray
        Shape (n_periods, max_delay_periods). Delay probabilities.
    observed_counts : np.ndarray
        Shape (n_periods, max_delay_periods). Observed claims triangle.
    lambda_hat : np.ndarray
        Shape (n_periods,). Expected ultimate occurrence counts (exposure-adjusted).
    feature_data : dict
        The prepared data dict from _truncation.prepare_observed_triangle().
    by : list[str] or None
        Not used in this base function — segment-level CFs handled in
        ReportingDelayModel.predict_completion_factors().

    Returns
    -------
    pd.DataFrame
        Columns: occurrence_period, development_fraction, completion_factor,
        observed_claims, estimated_ultimate_claims.
    """
    n_periods = len(occurrence_periods)
    records = []

    for i in range(n_periods):
        tau = truncation_depth[i]
        # Development fraction F_i = P(reported by tau_i)
        f_i = p_hat[i, :tau].sum()
        f_i = np.clip(f_i, 1e-6, 1.0)

        completion_factor = 1.0 / f_i
        observed_i = observed_counts[i, :tau].sum()
        estimated_ultimate = lambda_hat[i]

        records.append({
            "occurrence_period": occurrence_periods[i],
            "development_fraction": float(f_i),
            "completion_factor": float(completion_factor),
            "observed_claims": float(observed_i),
            "estimated_ultimate_claims": float(estimated_ultimate),
            "ibnr_count": float(max(estimated_ultimate - observed_i, 0.0)),
        })

    return pd.DataFrame(records)


def predict_ibnr_by_period(
    occurrence_periods: np.ndarray,
    truncation_depth: np.ndarray,
    p_hat: np.ndarray,
    lambda_hat: np.ndarray,
    observed_counts: np.ndarray,
) -> pd.DataFrame:
    """
    Compute IBNR (Incurred But Not Reported) counts by occurrence period.

    IBNR_i = lambda_hat_i * (1 - F_i)
           = lambda_hat_i * sum_{j >= tau_i} p_{i,j}

    This is the expected number of claims that occurred in period i but
    have not yet been reported as of the evaluation date.

    Parameters
    ----------
    occurrence_periods : np.ndarray
    truncation_depth : np.ndarray
    p_hat : np.ndarray
    lambda_hat : np.ndarray
    observed_counts : np.ndarray

    Returns
    -------
    pd.DataFrame
        Columns: occurrence_period, ibnr_count, ibnr_fraction,
        observed_claims, estimated_ultimate_claims.
    """
    n_periods = len(occurrence_periods)
    records = []

    for i in range(n_periods):
        tau = truncation_depth[i]
        # Unreported fraction
        max_delay = p_hat.shape[1]
        unreported_fraction = p_hat[i, tau:].sum() if tau < max_delay else 0.0

        ibnr = lambda_hat[i] * unreported_fraction
        observed_i = observed_counts[i, :tau].sum()

        records.append({
            "occurrence_period": occurrence_periods[i],
            "ibnr_count": float(max(ibnr, 0.0)),
            "ibnr_fraction": float(np.clip(unreported_fraction, 0.0, 1.0)),
            "observed_claims": float(observed_i),
            "estimated_ultimate_claims": float(lambda_hat[i]),
        })

    return pd.DataFrame(records)


def predict_development_pattern(
    delay_model: Any,
    X: np.ndarray,
    max_delay_periods: int,
) -> np.ndarray:
    """
    Predict delay probability distribution for given risk features.

    Returns p_{i,j} — the probability that a claim with risk profile X[i]
    will be reported with a delay of j periods. Each row sums to 1.

    Parameters
    ----------
    delay_model : fitted delay model
        Must have a predict_proba(X) method.
    X : np.ndarray, shape (n, p)
        Risk features.
    max_delay_periods : int
        Number of delay periods.

    Returns
    -------
    np.ndarray, shape (n, max_delay_periods)
    """
    proba = delay_model.predict_proba(X)
    # Ensure correct shape and normalisation
    if proba.shape[1] != max_delay_periods:
        raise RuntimeError(
            f"Delay model returned {proba.shape[1]} classes, "
            f"expected {max_delay_periods}"
        )
    # Normalise rows to sum to 1 (numerical safety)
    row_sums = proba.sum(axis=1, keepdims=True)
    return proba / np.maximum(row_sums, 1e-15)


def compute_segment_completion_factors(
    df: pd.DataFrame,
    occurrence_col: str,
    by: list[str],
    occurrence_periods: np.ndarray,
    period_to_idx: dict,
    truncation_depth: np.ndarray,
    p_hat: np.ndarray,
    lambda_hat: np.ndarray,
    observed_counts: np.ndarray,
    exposure_col: str,
    feature_cols: list[str],
    delay_model: Any,
    occurrence_model: Any,
) -> pd.DataFrame:
    """
    Compute completion factors at segment level (by occurrence_period + groupby columns).

    This is the more complex output that pricing actuaries actually want:
    "what is the completion factor for motor BI in the north-west in Q2?"

    For each segment (combination of occurrence_period and by columns), we:
    1. Take the subset of data in that segment
    2. Predict the delay distribution for those risk characteristics
    3. Compute the segment-level development fraction and completion factor

    Parameters
    ----------
    df : pd.DataFrame
        The original input DataFrame.
    occurrence_col : str
    by : list[str]
        Additional grouping columns.
    occurrence_periods : np.ndarray
    period_to_idx : dict
    truncation_depth : np.ndarray
    p_hat : np.ndarray
    lambda_hat : np.ndarray
    observed_counts : np.ndarray
    exposure_col : str
    feature_cols : list[str]
    delay_model : fitted delay model
    occurrence_model : fitted occurrence model

    Returns
    -------
    pd.DataFrame
    """
    max_delay = p_hat.shape[1]
    group_cols = [occurrence_col] + list(by)
    records = []

    for group_keys, group_df in df.groupby(group_cols):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)

        occ_period = group_keys[0]
        if occ_period not in period_to_idx:
            continue
        period_idx = period_to_idx[occ_period]
        tau = truncation_depth[period_idx]

        # Exposure-weighted mean features for this segment
        exp = group_df[exposure_col].values
        if exp.sum() <= 0:
            continue
        if feature_cols:
            X_seg = (group_df[feature_cols].values * exp[:, None]).sum(axis=0) / exp.sum()
            X_seg = X_seg.reshape(1, -1)
        else:
            X_seg = np.zeros((1, 0))

        # Predict delay distribution for this segment's risk profile
        if feature_cols:
            p_seg = delay_model.predict_proba(X_seg)[0]
        else:
            # No features — use period-level delay distribution
            p_seg = p_hat[period_idx]

        # Normalise
        p_seg = p_seg / max(p_seg.sum(), 1e-15)

        # Development fraction
        f_seg = p_seg[:tau].sum()
        f_seg = np.clip(f_seg, 1e-6, 1.0)
        completion_factor = 1.0 / f_seg

        # Observed claims in this segment
        obs_mask = group_df.apply(
            lambda r: (
                r.get("_delay", -1) is not None
                and r.name is not None
            ),
            axis=1
        )
        # Count observed claims in this segment from original data
        # (the group_df from the original df, not the triangle)
        observed_i = len(group_df)  # each row = one claim or one risk unit?
        # Actually group_df contains risk units (rows from df), not just claims
        # The triangle has the actual claim counts — use period-level observed
        # and scale by segment exposure fraction
        total_exp_period = df[df[occurrence_col] == occ_period][exposure_col].sum()
        seg_exp_frac = exp.sum() / max(total_exp_period, 1e-10)
        observed_claims_seg = observed_counts[period_idx, :tau].sum() * seg_exp_frac
        estimated_ultimate_seg = lambda_hat[period_idx] * seg_exp_frac

        record = {}
        for i, key in enumerate(group_keys):
            record[group_cols[i]] = key
        record.update({
            "development_fraction": float(f_seg),
            "completion_factor": float(completion_factor),
            "observed_claims": float(observed_claims_seg),
            "estimated_ultimate_claims": float(estimated_ultimate_seg),
            "ibnr_count": float(max(estimated_ultimate_seg - observed_claims_seg, 0.0)),
            "segment_exposure": float(exp.sum()),
        })
        records.append(record)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)
