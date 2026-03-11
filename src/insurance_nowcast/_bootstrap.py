"""
Bootstrap confidence intervals for completion factors and IBNR counts.

Bootstrap CIs on completion factors are important for pricing actuaries.
A completion factor of 1.40 ± 0.30 should be treated very differently from
1.40 ± 0.02 — the former implies the IBNR estimate is highly uncertain and
deserves additional margin.

We use a non-parametric bootstrap: resample occurrence periods with replacement,
refit the model, and compute completion factors. This preserves the
period-level correlation structure rather than resampling individual claims.

Computational note: bootstrapping a full EM fit n_bootstrap times is expensive.
Default n_bootstrap=100 gives adequate CI estimates for practical purposes.
Consider n_bootstrap=50 for exploratory work.
"""
from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd


def bootstrap_completion_factors(
    fit_fn: Callable,
    predict_fn: Callable,
    df: pd.DataFrame,
    occurrence_col: str,
    n_bootstrap: int = 100,
    confidence_level: float = 0.90,
    random_state: int = 42,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Compute bootstrap confidence intervals for completion factors.

    Resamples occurrence periods (with replacement) and refits the model
    n_bootstrap times. The CI is the empirical percentile interval.

    Parameters
    ----------
    fit_fn : callable
        Function that takes a resampled DataFrame and returns a fitted model.
    predict_fn : callable
        Function that takes a fitted model and returns a completion factor
        DataFrame (must include 'occurrence_period' and 'completion_factor').
    df : pd.DataFrame
        Original training data.
    occurrence_col : str
        Occurrence period column name.
    n_bootstrap : int
        Number of bootstrap replications. Default 100.
    confidence_level : float
        Confidence level for the interval. Default 0.90 (90% CI).
    random_state : int
        Random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        Columns: occurrence_period, ci_lower, ci_upper, ci_mean.
        Join to the main completion factor table on occurrence_period.
    """
    rng = np.random.default_rng(random_state)
    occurrence_periods = df[occurrence_col].unique()
    n_periods = len(occurrence_periods)

    # Store bootstrap CF values per period
    cf_bootstrap: dict[Any, list[float]] = {p: [] for p in occurrence_periods}

    alpha = (1 - confidence_level) / 2
    n_failed = 0

    for b in range(n_bootstrap):
        if verbose and (b + 1) % 10 == 0:
            print(f"Bootstrap {b + 1}/{n_bootstrap}")

        # Resample occurrence periods with replacement
        sampled_periods = rng.choice(occurrence_periods, size=n_periods, replace=True)

        # Build resampled DataFrame
        dfs = []
        for period in sampled_periods:
            dfs.append(df[df[occurrence_col] == period])
        df_boot = pd.concat(dfs, ignore_index=True)

        # Fit model on resampled data
        try:
            model_boot = fit_fn(df_boot)
            cf_df = predict_fn(model_boot)
            for _, row in cf_df.iterrows():
                p = row["occurrence_period"]
                if p in cf_bootstrap:
                    cf_bootstrap[p].append(float(row["completion_factor"]))
        except Exception as e:
            n_failed += 1
            if verbose:
                warnings.warn(f"Bootstrap replicate {b + 1} failed: {e}", UserWarning)
            continue

    if n_failed > 0.2 * n_bootstrap:
        warnings.warn(
            f"{n_failed}/{n_bootstrap} bootstrap replicates failed. "
            "CI estimates may be unreliable. Check model convergence.",
            UserWarning,
            stacklevel=2,
        )

    records = []
    for period in occurrence_periods:
        vals = cf_bootstrap[period]
        if len(vals) < 5:
            records.append({
                "occurrence_period": period,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "ci_mean": np.nan,
                "n_bootstrap_ok": len(vals),
            })
        else:
            records.append({
                "occurrence_period": period,
                "ci_lower": float(np.percentile(vals, 100 * alpha)),
                "ci_upper": float(np.percentile(vals, 100 * (1 - alpha))),
                "ci_mean": float(np.mean(vals)),
                "n_bootstrap_ok": len(vals),
            })

    return pd.DataFrame(records)


def analytic_completion_factor_variance(
    p_hat: np.ndarray,
    truncation_depth: np.ndarray,
    lambda_hat: np.ndarray,
    n_obs_by_period: np.ndarray,
) -> np.ndarray:
    """
    Approximate analytic variance of completion factors using the delta method.

    For CF_i = 1/F_i where F_i = sum_{j<tau_i} p_{i,j}, the variance is
    approximately:
        Var(CF_i) ≈ (1/F_i^2)^2 * Var(F_i)
                  ≈ CF_i^4 * Var(F_i)

    where Var(F_i) is estimated from the Poisson sampling variance of the
    observed delay counts. This is a rough approximation — bootstrap CIs
    are more reliable for sparse data.

    Parameters
    ----------
    p_hat : np.ndarray, shape (n_periods, max_delay)
    truncation_depth : np.ndarray, shape (n_periods,)
    lambda_hat : np.ndarray, shape (n_periods,)
    n_obs_by_period : np.ndarray, shape (n_periods,) — observed claim counts

    Returns
    -------
    np.ndarray, shape (n_periods,) — approximate standard error of CF_i
    """
    n_periods = len(lambda_hat)
    se_cf = np.zeros(n_periods)

    for i in range(n_periods):
        tau = truncation_depth[i]
        f_i = p_hat[i, :tau].sum()
        f_i = max(f_i, 1e-6)
        cf_i = 1.0 / f_i
        n_obs = max(n_obs_by_period[i], 1.0)

        # Approximate variance of F_i from Poisson (counts/lambda)
        # Var(F_i) ≈ F_i * (1 - F_i) / n_obs
        var_f = f_i * (1 - f_i) / n_obs
        # Delta method: Var(CF_i) ≈ (d CF / d F)^2 * Var(F) = CF^4 * Var(F)
        var_cf = (cf_i ** 4) * var_f
        se_cf[i] = np.sqrt(max(var_cf, 0.0))

    return se_cf
