"""
ReportingDelayModel — ML-Enhanced EM Nowcasting for insurance claims.

Implements the Wilsens/Antonio/Claeskens (arXiv:2512.07335) algorithm adapted
for insurance pricing:

1. Joint Poisson-Multinomial model for occurrence and reporting delay
2. XGBoost (or GLM) in the M-step — additive construction across EM iterations
3. Exposure offset in the Poisson M-step — critical for pricing, absent from the paper
4. Segment-level completion factor output for pricing GLM adjustment
5. Bootstrap confidence intervals

The key difference from standard IBNR triangles: this model conditions on risk
covariates. A motor BI claim from a young driver has a different delay distribution
than a motor PD claim from an experienced fleet driver. Aggregate LDFs wash this
out; this model doesn't.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from ._validators import validate_fit_inputs, validate_feature_cols
from ._truncation import prepare_observed_triangle, build_delay_training_data
from ._estep import estep, compute_observed_log_likelihood, compute_initial_lambda
from ._mstep import build_occurrence_model, build_delay_model
from ._completion import (
    predict_completion_factors,
    predict_ibnr_by_period,
    predict_development_pattern,
    compute_segment_completion_factors,
)
from ._bootstrap import bootstrap_completion_factors, analytic_completion_factor_variance


class ReportingDelayModel:
    """
    ML-enhanced EM nowcasting for insurance claims reporting delays.

    Jointly estimates claim occurrence rates and reporting delay distributions
    using an Expectation-Maximisation algorithm. The M-step uses XGBoost
    (or Poisson/multinomial GLM) to model covariate effects on both occurrence
    frequency and delay speed.

    The primary output is segment-level completion factors — the multipliers
    needed to scale observed (immature) claim counts to estimated ultimate
    counts. These are direct inputs to a pricing frequency GLM.

    This is an insurance-specific adaptation of Wilsens, Antonio & Claeskens
    (arXiv:2512.07335), with the key addition of an exposure offset in the
    occurrence model — essential for pricing use but absent from the paper.

    Parameters
    ----------
    occurrence_model : str
        Model type for occurrence M-step. One of 'glm', 'xgboost'. Default 'xgboost'.
        XGBoost is the better performer per the paper's experiments; GLM is faster
        and more interpretable on small datasets.
    delay_model : str
        Model type for delay M-step. One of 'glm', 'xgboost'. Default 'xgboost'.
    max_delay_periods : int
        Maximum reporting delay in periods. Default 24. Set this to the 95th or
        99th percentile of observed delays for your line of business — too small
        understates IBNR. The model warns if >10% of observed delays hit this
        boundary.
    exposure_offset : bool
        If True, adds log(exposure) as offset in the Poisson occurrence model.
        Default True. Keep this True for pricing use — turning it off conflates
        frequency rate with volume.
    em_patience : int
        Stop if observed log-likelihood does not improve for this many iterations.
        Default 10.
    max_em_iterations : int
        Hard upper limit on EM iterations. Default 50.
    convergence_tol : float
        Minimum improvement in log-likelihood to reset patience counter.
        Default 1e-4.
    n_bootstrap : int
        Bootstrap replications for confidence intervals. Default 100. Set to 0
        to skip bootstrap (CI columns will be NaN).
    bootstrap_confidence : float
        Confidence level for bootstrap CIs. Default 0.90.
    random_state : int
        Random seed for reproducibility. Default 42.
    verbose : bool
        Print convergence information during fitting. Default False.
    occurrence_model_kwargs : dict or None
        Additional keyword arguments passed to the occurrence model constructor.
    delay_model_kwargs : dict or None
        Additional keyword arguments passed to the delay model constructor.

    Attributes
    ----------
    convergence_history_ : list[float]
        Observed log-likelihood at each EM iteration. Use to assess convergence.
    n_iter_ : int
        Number of EM iterations completed.
    is_converged_ : bool
        True if the algorithm met the convergence criterion.
    p_hat_ : np.ndarray
        Final delay probability matrix, shape (n_periods, max_delay_periods).
    lambda_hat_ : np.ndarray
        Final expected occurrence counts per period (exposure-adjusted).
    occurrence_periods_ : np.ndarray
        Sorted occurrence period values seen during fit.
    truncation_depth_ : np.ndarray
        Truncation depth tau_i for each occurrence period.

    Examples
    --------
    >>> from insurance_nowcast import ReportingDelayModel
    >>> model = ReportingDelayModel(max_delay_periods=12)
    >>> model.fit(
    ...     df,
    ...     occurrence_col='accident_month',
    ...     report_col='report_month',
    ...     exposure_col='exposure',
    ...     feature_cols=['vehicle_class', 'age_band', 'territory'],
    ...     eval_date=202312,
    ... )
    >>> cf = model.predict_completion_factors(df, by=['vehicle_class'])
    """

    def __init__(
        self,
        occurrence_model: str = "xgboost",
        delay_model: str = "xgboost",
        max_delay_periods: int = 24,
        exposure_offset: bool = True,
        em_patience: int = 10,
        max_em_iterations: int = 50,
        convergence_tol: float = 1e-4,
        n_bootstrap: int = 100,
        bootstrap_confidence: float = 0.90,
        random_state: int = 42,
        verbose: bool = False,
        occurrence_model_kwargs: dict | None = None,
        delay_model_kwargs: dict | None = None,
    ) -> None:
        self.occurrence_model = occurrence_model
        self.delay_model = delay_model
        self.max_delay_periods = max_delay_periods
        self.exposure_offset = exposure_offset
        self.em_patience = em_patience
        self.max_em_iterations = max_em_iterations
        self.convergence_tol = convergence_tol
        self.n_bootstrap = n_bootstrap
        self.bootstrap_confidence = bootstrap_confidence
        self.random_state = random_state
        self.verbose = verbose
        self.occurrence_model_kwargs = occurrence_model_kwargs or {}
        self.delay_model_kwargs = delay_model_kwargs or {}

        # Set after fit()
        self._is_fitted = False
        self.convergence_history_: list[float] = []
        self.n_iter_: int = 0
        self.is_converged_: bool = False

    def fit(
        self,
        df: pd.DataFrame,
        occurrence_col: str,
        report_col: str,
        exposure_col: str,
        feature_cols: list[str],
        eval_date: Any,
    ) -> "ReportingDelayModel":
        """
        Fit the ML-EM nowcasting model.

        Runs the EM algorithm until convergence (by patience criterion) or
        max_em_iterations is reached. Each EM iteration:
        1. E-step: impute unreported claim counts using current model predictions
        2. M-step (occurrence): refit occurrence model on imputed total counts
        3. M-step (delay): refit delay model on imputed (period, delay) counts

        Parameters
        ----------
        df : pd.DataFrame
            Individual claims data. One row per claim event, plus one row per
            risk unit (for exposure). See module docstring for column requirements.
        occurrence_col : str
            Column for occurrence period (e.g., accident month as integer YYYYMM).
        report_col : str
            Column for report period. Null/NaN for unreported (IBNR) claims.
        exposure_col : str
            Column for policy exposure (policy-years at risk).
        feature_cols : list[str]
            Risk covariate columns. Must be numeric.
        eval_date : int or comparable
            Evaluation date — claims reported after this date are treated as
            unreported. Must be the same type as occurrence_col.

        Returns
        -------
        self
        """
        # --- Validation ---
        validate_fit_inputs(
            df=df,
            occurrence_col=occurrence_col,
            report_col=report_col,
            exposure_col=exposure_col,
            feature_cols=feature_cols,
            eval_date=eval_date,
            max_delay_periods=self.max_delay_periods,
            exposure_offset=self.exposure_offset,
        )
        if feature_cols:
            validate_feature_cols(df, feature_cols)

        # Store for later use in predict methods
        self._occurrence_col = occurrence_col
        self._report_col = report_col
        self._exposure_col = exposure_col
        self._feature_cols = list(feature_cols)
        self._eval_date = eval_date
        self._df_fit = df.copy()

        # --- Data preparation ---
        data = prepare_observed_triangle(
            df=df,
            occurrence_col=occurrence_col,
            report_col=report_col,
            exposure_col=exposure_col,
            feature_cols=feature_cols,
            eval_date=eval_date,
            max_delay_periods=self.max_delay_periods,
        )

        observed_counts = data["observed_counts"]
        occurrence_periods = data["occurrence_periods"]
        truncation_depth = data["truncation_depth"]
        exposure_by_period = data["exposure_by_period"]
        features_by_period = data["features_by_period"]
        n_periods = data["n_periods"]

        # If no features, use intercept-only (column of ones)
        if features_by_period.shape[1] == 0:
            features_by_period = np.ones((n_periods, 1))

        # --- Initialise delay distribution ---
        # Start with uniform delay distribution
        p_hat = np.ones((n_periods, self.max_delay_periods), dtype=float)
        p_hat = p_hat / p_hat.sum(axis=1, keepdims=True)

        # Initialise lambda from observed counts + uniform delay
        lambda_hat = compute_initial_lambda(
            observed_counts, truncation_depth, p_hat, exposure_by_period
        )
        lambda_hat = np.maximum(lambda_hat, 1e-6)

        # --- Build model objects ---
        occ_kwargs = {
            "exposure_offset": self.exposure_offset,
            "random_state": self.random_state,
        }
        occ_kwargs.update(self.occurrence_model_kwargs)

        delay_kwargs = {
            "random_state": self.random_state,
        }
        delay_kwargs.update(self.delay_model_kwargs)

        occ_model = build_occurrence_model(self.occurrence_model, **occ_kwargs)
        del_model = build_delay_model(self.delay_model, **delay_kwargs)

        # --- EM loop ---
        self.convergence_history_ = []
        best_ll = -np.inf
        patience_counter = 0

        for iteration in range(self.max_em_iterations):
            # E-step: compute imputed complete data
            imputed_counts = estep(
                observed_counts=observed_counts,
                truncation_depth=truncation_depth,
                lambda_hat=lambda_hat,
                p_hat=p_hat,
            )

            # M-step (occurrence): fit Poisson model on imputed total counts
            imputed_totals = imputed_counts.sum(axis=1)  # N_i^{(k)}

            try:
                occ_model.fit(
                    X=features_by_period,
                    y_counts=imputed_totals,
                    exposure=exposure_by_period,
                    **_maybe_em_iter(self.occurrence_model, iteration),
                )
            except Exception as e:
                warnings.warn(
                    f"EM iteration {iteration + 1}: occurrence model fit failed: {e}. "
                    "Keeping previous parameters.",
                    UserWarning,
                    stacklevel=2,
                )

            # M-step (delay): fit multinomial model on imputed delay counts
            X_delay, y_delay, w_delay = build_delay_training_data(
                imputed_counts=imputed_counts,
                features_by_period=features_by_period,
                truncation_depth=truncation_depth,
                max_delay_periods=self.max_delay_periods,
            )

            try:
                del_model.fit(
                    X=X_delay,
                    y=y_delay,
                    sample_weight=w_delay,
                    n_classes=self.max_delay_periods,
                    **_maybe_em_iter(self.delay_model, iteration),
                )
            except Exception as e:
                warnings.warn(
                    f"EM iteration {iteration + 1}: delay model fit failed: {e}. "
                    "Keeping previous parameters.",
                    UserWarning,
                    stacklevel=2,
                )

            # Update predictions
            lambda_hat = occ_model.predict(
                X=features_by_period,
                exposure=exposure_by_period,
            )
            lambda_hat = np.maximum(lambda_hat, 1e-8)

            p_hat_raw = del_model.predict_proba(features_by_period)
            # Ensure correct shape
            if p_hat_raw.shape[1] != self.max_delay_periods:
                raise RuntimeError(
                    f"Delay model returned {p_hat_raw.shape[1]} classes, "
                    f"expected {self.max_delay_periods}"
                )
            # Normalise rows
            row_sums = p_hat_raw.sum(axis=1, keepdims=True)
            p_hat = p_hat_raw / np.maximum(row_sums, 1e-15)

            # Compute observed log-likelihood
            ll = compute_observed_log_likelihood(
                observed_counts=observed_counts,
                truncation_depth=truncation_depth,
                lambda_hat=lambda_hat,
                p_hat=p_hat,
                exposure=exposure_by_period,
            )
            self.convergence_history_.append(ll)

            if self.verbose:
                print(f"EM iter {iteration + 1:3d}: log-lik = {ll:.4f}")

            # Convergence check
            if ll > best_ll + self.convergence_tol:
                best_ll = ll
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.em_patience:
                if self.verbose:
                    print(f"Converged at iteration {iteration + 1}")
                self.is_converged_ = True
                break
        else:
            # Ran to max iterations
            if self.verbose:
                print(f"Reached max iterations ({self.max_em_iterations})")
            self.is_converged_ = False

        # Store fitted objects and results
        self._occ_model = occ_model
        self._del_model = del_model
        self.p_hat_ = p_hat
        self.lambda_hat_ = lambda_hat
        self.occurrence_periods_ = occurrence_periods
        self.truncation_depth_ = truncation_depth
        self._observed_counts = observed_counts
        self._exposure_by_period = exposure_by_period
        self._features_by_period = features_by_period
        self._period_to_idx = data["period_to_idx"]
        self.n_iter_ = iteration + 1
        self._is_fitted = True

        if not self.is_converged_ and self.verbose:
            warnings.warn(
                f"EM did not converge within {self.max_em_iterations} iterations. "
                "Consider increasing max_em_iterations or adjusting regularisation.",
                UserWarning,
                stacklevel=2,
            )

        return self

    def predict_completion_factors(
        self,
        df: pd.DataFrame | None = None,
        by: list[str] | None = None,
        include_ci: bool = True,
    ) -> pd.DataFrame:
        """
        Compute completion factors by occurrence period (and optionally by segment).

        The completion factor for period i is CF_i = 1 / F_i, where F_i is the
        fraction of claims expected to have been reported by the evaluation date.
        Multiply observed claims by CF_i to estimate ultimate claims.

        CF_i = 1.0 for periods where tau_i = max_delay_periods (fully developed).
        CF_i can be very large for very recent periods where few claims are reported.

        Parameters
        ----------
        df : pd.DataFrame or None
            If None, uses the training data. If provided, uses this for segment
            grouping when by is specified.
        by : list[str] or None
            Additional grouping columns. If provided, returns completion factors
            at the intersection of occurrence_period and these columns.
        include_ci : bool
            Whether to include bootstrap confidence intervals. Default True.
            Only relevant if n_bootstrap > 0 was set at model initialisation.

        Returns
        -------
        pd.DataFrame
            Columns: occurrence_period, [by cols], development_fraction,
            completion_factor, observed_claims, estimated_ultimate_claims,
            ibnr_count, [ci_lower, ci_upper if include_ci].
        """
        self._check_fitted()

        if df is None:
            df = self._df_fit

        if by:
            result = compute_segment_completion_factors(
                df=df,
                occurrence_col=self._occurrence_col,
                by=by,
                occurrence_periods=self.occurrence_periods_,
                period_to_idx=self._period_to_idx,
                truncation_depth=self.truncation_depth_,
                p_hat=self.p_hat_,
                lambda_hat=self.lambda_hat_,
                observed_counts=self._observed_counts,
                exposure_col=self._exposure_col,
                feature_cols=self._feature_cols,
                delay_model=self._del_model,
                occurrence_model=self._occ_model,
            )
        else:
            result = predict_completion_factors(
                occurrence_periods=self.occurrence_periods_,
                truncation_depth=self.truncation_depth_,
                p_hat=self.p_hat_,
                observed_counts=self._observed_counts,
                lambda_hat=self.lambda_hat_,
                feature_data={},
            )

        if include_ci and self.n_bootstrap > 0 and by is None:
            ci_df = self._compute_bootstrap_ci()
            result = result.merge(ci_df, on="occurrence_period", how="left")

        return result

    def predict_ibnr(
        self,
        df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Estimate IBNR (Incurred But Not Reported) counts by occurrence period.

        IBNR_i is the expected number of claims that occurred in period i but
        have not yet been reported as of the evaluation date. Sum across all
        periods to get total IBNR loading for the portfolio.

        Parameters
        ----------
        df : pd.DataFrame or None
            Not currently used — present for API consistency. Future versions
            will support out-of-sample IBNR prediction.

        Returns
        -------
        pd.DataFrame
            Columns: occurrence_period, ibnr_count, ibnr_fraction,
            observed_claims, estimated_ultimate_claims.
        """
        self._check_fitted()

        return predict_ibnr_by_period(
            occurrence_periods=self.occurrence_periods_,
            truncation_depth=self.truncation_depth_,
            p_hat=self.p_hat_,
            lambda_hat=self.lambda_hat_,
            observed_counts=self._observed_counts,
        )

    def predict_development_pattern(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Predict the delay probability distribution for given risk profiles.

        Returns p_{i,j} — the probability that a claim with features X[i] will
        be reported with a delay of j periods after occurrence. Each row sums to 1.

        Parameters
        ----------
        X : pd.DataFrame
            Risk features. Must contain the same feature columns used in fit().

        Returns
        -------
        np.ndarray, shape (n_risks, max_delay_periods)
            Delay probability distribution for each risk profile.
        """
        self._check_fitted()

        if self._feature_cols:
            missing = [c for c in self._feature_cols if c not in X.columns]
            if missing:
                raise ValueError(
                    f"Feature columns missing from X: {missing}"
                )
            X_arr = X[self._feature_cols].values.astype(float)
        else:
            X_arr = np.ones((len(X), 1))

        return predict_development_pattern(
            delay_model=self._del_model,
            X=X_arr,
            max_delay_periods=self.max_delay_periods,
        )

    def _compute_ibnr_df(self) -> pd.DataFrame:
        """Internal helper for diagnostics."""
        return self.predict_ibnr()

    def _compute_bootstrap_ci(self) -> pd.DataFrame:
        """Compute bootstrap CIs for period-level completion factors."""
        if hasattr(self, "_bootstrap_ci_cache_"):
            return self._bootstrap_ci_cache_

        occ_col = self._occurrence_col
        rep_col = self._report_col
        exp_col = self._exposure_col
        feat_cols = self._feature_cols
        eval_date = self._eval_date
        max_delay = self.max_delay_periods

        # Build fit and predict closures for bootstrap
        model_config = dict(
            occurrence_model=self.occurrence_model,
            delay_model=self.delay_model,
            max_delay_periods=max_delay,
            exposure_offset=self.exposure_offset,
            em_patience=self.em_patience,
            max_em_iterations=self.max_em_iterations,
            convergence_tol=self.convergence_tol,
            n_bootstrap=0,  # No nested bootstrap
            random_state=self.random_state,
            verbose=False,
        )

        def fit_fn(df_boot: pd.DataFrame) -> "ReportingDelayModel":
            m = ReportingDelayModel(**model_config)
            m.fit(df_boot, occ_col, rep_col, exp_col, feat_cols, eval_date)
            return m

        def predict_fn(m: "ReportingDelayModel") -> pd.DataFrame:
            return m.predict_completion_factors(include_ci=False)

        ci_df = bootstrap_completion_factors(
            fit_fn=fit_fn,
            predict_fn=predict_fn,
            df=self._df_fit,
            occurrence_col=occ_col,
            n_bootstrap=self.n_bootstrap,
            confidence_level=self.bootstrap_confidence,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        self._bootstrap_ci_cache_ = ci_df
        return ci_df

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "Model has not been fitted. Call fit() first."
            )

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"ReportingDelayModel("
            f"occurrence_model={self.occurrence_model!r}, "
            f"delay_model={self.delay_model!r}, "
            f"max_delay_periods={self.max_delay_periods}, "
            f"status={status})"
        )


def _maybe_em_iter(model_type: str, iteration: int) -> dict:
    """Return em_iteration kwarg only for XGBoost models (which support it)."""
    if model_type == "xgboost":
        return {"em_iteration": iteration}
    return {}
