"""
NowcastSimulator — generate realistic insurance claims data with known delay structure.

The simulator is the backbone of our test suite. By generating data from a
known data-generating process (DGP), we can verify that the EM algorithm
recovers the true parameters to within acceptable precision.

The DGP follows the Wilsens et al. framework:
- Occurrence: N_i ~ Poisson(lambda(x_i) * exposure_i)
- Delay: N_{i,j} | N_i ~ Multinomial(p_j(x_i))
- Truncation: claims with j >= tau_i are not yet reported at eval_date

True delay distributions are parameterised as either:
- Geometric: p_j ∝ (1-rho)^j * rho — exponential decay
- Logistic: delay probabilities derived from a logistic function of features

This mirrors real insurance data where high-risk claims (BI) have slower
reporting than low-risk claims (PD).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


class NowcastSimulator:
    """
    Generate synthetic insurance claims data with known reporting delay structure.

    Useful for:
    - Testing that ReportingDelayModel recovers known parameters
    - Benchmarking convergence speed
    - Generating examples for documentation and notebooks

    The DGP is:
        N_i ~ Poisson(base_frequency * exposure_i * exp(sum_k beta_k * x_{i,k}))
        N_{i,j} | N_i ~ Multinomial(p_j(x_i))
        p_j(x_i) = softmax of delay_logits_j(x_i)

    Parameters
    ----------
    n_occurrence_periods : int
        Number of occurrence periods (e.g., months). Default 24.
    max_delay_periods : int
        Maximum reporting delay in periods. Default 12.
    base_frequency : float
        Baseline claim frequency (claims per unit exposure). Default 0.05.
    delay_shape : str
        Shape of delay distribution. One of 'geometric', 'logistic', 'mixed'.
        Default 'geometric'.
    feature_effects : dict or None
        Maps feature name to its log-frequency effect. E.g. {'age_group': 0.3}
        means claims are e^0.3 ≈ 1.35x more frequent for age_group=1.
    delay_feature_effects : dict or None
        Maps feature name to its effect on delay speed. Positive = faster reporting.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_occurrence_periods: int = 24,
        max_delay_periods: int = 12,
        base_frequency: float = 0.05,
        delay_shape: str = "geometric",
        feature_effects: dict | None = None,
        delay_feature_effects: dict | None = None,
        random_state: int = 42,
    ) -> None:
        self.n_occurrence_periods = n_occurrence_periods
        self.max_delay_periods = max_delay_periods
        self.base_frequency = base_frequency
        self.delay_shape = delay_shape
        self.feature_effects = feature_effects or {}
        self.delay_feature_effects = delay_feature_effects or {}
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def generate(
        self,
        n_policies: int = 1000,
        eval_period: int | None = None,
        exposure_per_policy: float = 1.0,
        exposure_variability: float = 0.0,
    ) -> pd.DataFrame:
        """
        Generate a synthetic claims dataset.

        Returns a DataFrame where each row is a risk unit (policy) in an
        occurrence period. Claims are represented by the observed claim count,
        with individual claims expanded into separate rows for use with
        ReportingDelayModel.

        Parameters
        ----------
        n_policies : int
            Number of policies per occurrence period. Default 1000.
        eval_period : int or None
            The evaluation period (= last occurrence period by default).
            Claims reported with delay >= tau_i are set as IBNR (null report_period).
        exposure_per_policy : float
            Mean exposure per policy-period. Default 1.0 (annual).
        exposure_variability : float
            If > 0, exposure varies as Gamma(1/variability, variability). Default 0.

        Returns
        -------
        pd.DataFrame
            Columns: occurrence_period, report_period (nullable), exposure,
            plus any feature columns.
        """
        if eval_period is None:
            eval_period = self.n_occurrence_periods - 1

        records = []

        for period in range(self.n_occurrence_periods):
            # Generate policy-level features
            features = self._generate_features(n_policies)

            # Generate exposure
            if exposure_variability > 0:
                alpha = 1.0 / exposure_variability
                beta = exposure_per_policy * exposure_variability
                exposure = self._rng.gamma(alpha, beta, size=n_policies)
            else:
                exposure = np.full(n_policies, exposure_per_policy)

            # Compute claim frequency for each policy
            log_rate = np.log(self.base_frequency)
            for feat_name, effect in self.feature_effects.items():
                if feat_name in features:
                    log_rate = log_rate + effect * features[feat_name]

            lambda_i = np.exp(log_rate) * exposure

            # Generate claim counts (Poisson)
            claim_counts = self._rng.poisson(lambda_i)

            # For each policy with claims, generate delay periods
            tau_i = min(self.max_delay_periods, eval_period - period + 1)
            tau_i = max(tau_i, 0)

            for pol_idx in range(n_policies):
                n_claims = claim_counts[pol_idx]
                if n_claims == 0:
                    # Risk unit row with no claims
                    row = {
                        "occurrence_period": period,
                        "report_period": None,  # no claim = no report
                        "exposure": float(exposure[pol_idx]),
                        "has_claim": False,
                    }
                    for feat_name, feat_vals in features.items():
                        row[feat_name] = feat_vals[pol_idx]
                    records.append(row)
                    continue

                # Compute delay distribution for this policy
                p_delay = self._compute_delay_probs(features, pol_idx)

                for _ in range(n_claims):
                    delay = self._rng.choice(
                        self.max_delay_periods,
                        p=p_delay,
                    )
                    report_period = period + delay

                    # Is this claim reported by eval_period?
                    if delay < tau_i:
                        reported_period = report_period
                    else:
                        reported_period = None  # IBNR

                    row = {
                        "occurrence_period": period,
                        "report_period": reported_period,
                        "exposure": float(exposure[pol_idx]) / max(n_claims, 1),
                        "has_claim": True,
                    }
                    for feat_name, feat_vals in features.items():
                        row[feat_name] = feat_vals[pol_idx]
                    records.append(row)

        df = pd.DataFrame(records)
        # Keep only rows with claims (or a sample of no-claim rows for exposure)
        # Actually, we need to structure this properly:
        # Each row in the output = one claim or one risk unit
        # For the model to work, we need one row per risk unit per occurrence period
        # Let's restructure
        return self._restructure_as_risk_units(
            n_policies, eval_period, exposure_per_policy, exposure_variability
        )

    def _restructure_as_risk_units(
        self,
        n_policies: int,
        eval_period: int,
        exposure_per_policy: float,
        exposure_variability: float,
    ) -> pd.DataFrame:
        """
        Generate data in risk-unit format (one row per risk unit per period).

        This is the correct input format for ReportingDelayModel: each row
        represents a risk unit (policy) with its exposure and features, and
        claim events are separate rows.
        """
        claim_records = []
        exposure_records = []

        for period in range(self.n_occurrence_periods):
            features = self._generate_features(n_policies)

            if exposure_variability > 0:
                alpha = 1.0 / exposure_variability
                beta = exposure_per_policy * exposure_variability
                exposure = self._rng.gamma(alpha, beta, size=n_policies)
            else:
                exposure = np.full(n_policies, exposure_per_policy)

            log_rate = np.log(self.base_frequency)
            for feat_name, effect in self.feature_effects.items():
                if feat_name in features:
                    log_rate = log_rate + effect * features[feat_name]

            lambda_i = np.exp(log_rate) * exposure
            claim_counts = self._rng.poisson(lambda_i)

            tau_i = min(self.max_delay_periods, eval_period - period + 1)
            tau_i = max(tau_i, 0)

            for pol_idx in range(n_policies):
                n_claims = claim_counts[pol_idx]

                # Exposure row (always present, even with 0 claims)
                exp_row = {
                    "occurrence_period": period,
                    "report_period": np.nan,  # no claim
                    "exposure": float(exposure[pol_idx]),
                }
                for feat_name, feat_vals in features.items():
                    exp_row[feat_name] = feat_vals[pol_idx]
                exposure_records.append(exp_row)

                if n_claims == 0:
                    continue

                # Individual claim rows
                p_delay = self._compute_delay_probs(features, pol_idx)

                for _ in range(n_claims):
                    delay = self._rng.choice(self.max_delay_periods, p=p_delay)

                    if delay < tau_i:
                        reported = float(period + delay)
                    else:
                        reported = np.nan  # IBNR

                    claim_row = {
                        "occurrence_period": period,
                        "report_period": reported,
                        "exposure": float(exposure[pol_idx]),
                    }
                    for feat_name, feat_vals in features.items():
                        claim_row[feat_name] = feat_vals[pol_idx]
                    claim_records.append(claim_row)

        # Combine: use exposure records as the base (one row per policy-period)
        # In the input format expected by ReportingDelayModel, we use
        # aggregated exposure rows where report_period = NaN if no claim
        # Actually the model expects one row per claim occurrence
        # plus the exposure information needs to be on each row.
        # Let's return claim-level rows with exposure per claim:
        if not claim_records:
            # No claims at all
            df = pd.DataFrame(exposure_records)
            df["report_period"] = np.nan
            return df

        df_claims = pd.DataFrame(claim_records)
        return df_claims

    def _generate_features(self, n: int) -> dict[str, np.ndarray]:
        """Generate feature arrays for n policies."""
        features = {}
        # Default features
        features["age_group"] = self._rng.integers(0, 3, size=n).astype(float)
        features["risk_score"] = self._rng.uniform(0, 1, size=n)
        features["channel"] = self._rng.integers(0, 2, size=n).astype(float)
        return features

    def _compute_delay_probs(
        self,
        features: dict[str, np.ndarray],
        pol_idx: int,
    ) -> np.ndarray:
        """
        Compute delay probability vector for a single policy.

        Returns
        -------
        np.ndarray, shape (max_delay_periods,)
            Probability of being reported at each delay.
        """
        d = self.max_delay_periods

        if self.delay_shape == "geometric":
            # Default: geometric decay, rho ~ 0.4 => most claims in first few periods
            rho = 0.4
            for feat_name, effect in self.delay_feature_effects.items():
                if feat_name in features:
                    # Positive effect = faster reporting (higher rho)
                    rho += effect * features[feat_name][pol_idx]
            rho = np.clip(rho, 0.05, 0.95)
            raw = rho * (1 - rho) ** np.arange(d)

        elif self.delay_shape == "logistic":
            # Sigmoid-shaped: slow start, peak in middle, tail off
            center = d / 3
            for feat_name, effect in self.delay_feature_effects.items():
                if feat_name in features:
                    center += effect * features[feat_name][pol_idx]
            center = np.clip(center, 1, d - 1)
            x = np.arange(d)
            raw = np.exp(-(x - center) ** 2 / (2 * (d / 6) ** 2))

        elif self.delay_shape == "mixed":
            # Mixture: immediate (j=0) and delayed (j=3-6) peak
            raw = np.zeros(d)
            raw[0] = 0.3  # quick-reported claims
            peak2 = min(4, d - 1)
            raw[peak2] = 0.5
            for j in range(1, d):
                raw[j] += 0.2 * np.exp(-0.5 * j)
        else:
            raise ValueError(f"Unknown delay_shape: {self.delay_shape!r}")

        # Normalise to sum to 1
        raw = np.maximum(raw, 0)
        total = raw.sum()
        if total <= 0:
            raw = np.ones(d) / d
        else:
            raw = raw / total

        return raw

    def get_true_completion_factors(
        self,
        eval_period: int | None = None,
    ) -> pd.DataFrame:
        """
        Compute the true (DGP) completion factors for each occurrence period.

        Uses the average delay distribution across all policies.

        Parameters
        ----------
        eval_period : int or None
            Evaluation period. Defaults to n_occurrence_periods - 1.

        Returns
        -------
        pd.DataFrame
            Columns: occurrence_period, true_development_fraction,
            true_completion_factor.
        """
        if eval_period is None:
            eval_period = self.n_occurrence_periods - 1

        # Average delay distribution (no feature effects for simplicity)
        mock_features = self._generate_features(100)
        p_avg = np.mean(
            [self._compute_delay_probs(mock_features, i) for i in range(100)],
            axis=0,
        )

        records = []
        for period in range(self.n_occurrence_periods):
            tau = min(self.max_delay_periods, eval_period - period + 1)
            tau = max(tau, 0)
            f = p_avg[:tau].sum()
            f = np.clip(f, 1e-6, 1.0)
            records.append({
                "occurrence_period": period,
                "true_development_fraction": float(f),
                "true_completion_factor": float(1.0 / f),
            })

        return pd.DataFrame(records)
