"""
E-step of the ML-EM nowcasting algorithm.

The E-step computes the expected complete data given the current model
parameters. For observed delays (j <= tau_i), we use the actual counts.
For censored delays (j > tau_i), we impute using the current occurrence
rate and delay probability estimates.

Mathematically, for censored (i, j):
    N_{i,j}^{(k)} = lambda^{(k-1)}(x_i) * p_j^{(k-1)}(x_i)

where lambda is the expected occurrence count (already incorporating exposure)
and p_j is the delay probability from the current iteration's delay model.
"""
from __future__ import annotations

import numpy as np


def estep(
    observed_counts: np.ndarray,
    truncation_depth: np.ndarray,
    lambda_hat: np.ndarray,
    p_hat: np.ndarray,
) -> np.ndarray:
    """
    Run the E-step: compute imputed complete-data counts N_{i,j}^{(k)}.

    For each occurrence period i:
    - If j < tau_i: use observed count N_{i,j} (no imputation needed)
    - If j >= tau_i: impute as lambda_hat[i] * p_hat[i, j]

    Note: we use >= tau_i rather than > tau_i. tau_i is the number of
    observable delay periods — delays 0, 1, ..., tau_i - 1 are observed.
    Delays tau_i, tau_i + 1, ..., d - 1 are censored.

    Parameters
    ----------
    observed_counts : np.ndarray
        Shape (n_periods, max_delay_periods). Observed N_{i,j} for j < tau_i,
        zeros elsewhere.
    truncation_depth : np.ndarray
        Shape (n_periods,). tau_i for each period. Integer valued.
    lambda_hat : np.ndarray
        Shape (n_periods,). Expected occurrence count for each period,
        lambda(x_i) * exposure_i (already exposure-adjusted).
    p_hat : np.ndarray
        Shape (n_periods, max_delay_periods). Delay probability distribution
        for each period. Each row sums to 1.

    Returns
    -------
    np.ndarray
        Shape (n_periods, max_delay_periods). Imputed complete data counts.
        For j < tau_i: equals observed_counts[i, j].
        For j >= tau_i: equals lambda_hat[i] * p_hat[i, j].
    """
    n_periods, max_delay = observed_counts.shape
    imputed = np.zeros_like(observed_counts, dtype=float)

    # Vectorised implementation using period-level mask
    for i in range(n_periods):
        tau = truncation_depth[i]
        # Observed delays: 0, 1, ..., tau - 1
        imputed[i, :tau] = observed_counts[i, :tau]
        # Censored delays: tau, tau + 1, ..., max_delay - 1
        if tau < max_delay:
            imputed[i, tau:] = lambda_hat[i] * p_hat[i, tau:]

    return imputed


def compute_observed_log_likelihood(
    observed_counts: np.ndarray,
    truncation_depth: np.ndarray,
    lambda_hat: np.ndarray,
    p_hat: np.ndarray,
    exposure: np.ndarray,
) -> float:
    """
    Compute the observed-data log-likelihood for convergence monitoring.

    The observed-data likelihood marginalises over the unobserved censored
    counts. For a Poisson-Multinomial model with truncation, the observed-data
    log-likelihood is:

        ell_obs = sum_i [ -lambda_i * F_i + N_i^obs * log(lambda_i * F_i) ]
                + sum_i sum_{j < tau_i} [ N_{i,j} * log(p_{i,j}) ]
                - sum_i log(N_i^obs!)  (constant in lambda, p)

    where F_i = sum_{j < tau_i} p_{i,j} is the probability of being observed
    by the truncation depth (the "development fraction" or 1/completion_factor).

    We track this quantity across EM iterations to monitor convergence.
    ML-EM has no guaranteed monotone increase, but oscillation is a warning sign.

    Parameters
    ----------
    observed_counts : np.ndarray
        Shape (n_periods, max_delay_periods). Raw observed counts.
    truncation_depth : np.ndarray
        Shape (n_periods,). tau_i for each period.
    lambda_hat : np.ndarray
        Shape (n_periods,). Expected occurrence counts (exposure-adjusted).
    p_hat : np.ndarray
        Shape (n_periods, max_delay_periods). Delay probabilities.
    exposure : np.ndarray
        Shape (n_periods,). Exposure per period (for reference; lambda_hat
        should already incorporate exposure).

    Returns
    -------
    float
        Observed-data log-likelihood.
    """
    n_periods, max_delay = observed_counts.shape
    ll = 0.0

    for i in range(n_periods):
        tau = truncation_depth[i]
        # Probability of being observed = sum of p_{i,j} for j < tau
        f_i = p_hat[i, :tau].sum()
        f_i = max(f_i, 1e-15)  # numerical safety

        # Expected observed count
        expected_obs = lambda_hat[i] * f_i
        expected_obs = max(expected_obs, 1e-15)

        # Actual observed count
        n_obs_i = observed_counts[i, :tau].sum()

        # Poisson contribution: -expected + n_obs * log(expected)
        ll += -expected_obs + n_obs_i * np.log(expected_obs)

        # Delay distribution contribution for observed delays
        for j in range(tau):
            n_ij = observed_counts[i, j]
            if n_ij > 0:
                p_ij = max(p_hat[i, j], 1e-15)
                # Conditional log-prob: log(p_{i,j} / F_i) for each observed claim
                ll += n_ij * (np.log(p_ij) - np.log(f_i))

    return ll


def compute_initial_lambda(
    observed_counts: np.ndarray,
    truncation_depth: np.ndarray,
    p_hat: np.ndarray,
    exposure: np.ndarray,
) -> np.ndarray:
    """
    Compute initial lambda estimates from observed data.

    Before the first M-step, we need starting values. The method-of-moments
    estimator divides observed claims by the fraction of delay periods that are
    observable.

    Parameters
    ----------
    observed_counts : np.ndarray
        Shape (n_periods, max_delay_periods).
    truncation_depth : np.ndarray
        Shape (n_periods,). tau_i.
    p_hat : np.ndarray
        Shape (n_periods, max_delay_periods). Initial delay probabilities.
    exposure : np.ndarray
        Shape (n_periods,). Exposure per period.

    Returns
    -------
    np.ndarray
        Shape (n_periods,). Initial lambda estimates.
    """
    n_periods = len(exposure)
    lambda_init = np.zeros(n_periods, dtype=float)

    for i in range(n_periods):
        tau = truncation_depth[i]
        n_obs_i = observed_counts[i, :tau].sum()
        f_i = p_hat[i, :tau].sum()
        f_i = max(f_i, 0.01)  # avoid division by near-zero
        lambda_init[i] = n_obs_i / f_i

    return lambda_init
