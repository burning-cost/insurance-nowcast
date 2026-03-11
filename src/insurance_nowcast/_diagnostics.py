"""
ReportingDelayDiagnostic — visualisation tools for fitted models.

These plots are designed for an actuarial review workflow:
1. Convergence plot: did the EM algorithm converge? Any oscillation?
2. Development pattern: do the delay distributions look sensible?
3. IBNR by period: where is the unreported claims loading concentrated?
4. Delay distribution: compare delay patterns across risk segments.

All methods are static and take a fitted ReportingDelayModel as input.
matplotlib is optional — import will raise a clear error if not installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._model import ReportingDelayModel


class ReportingDelayDiagnostic:
    """
    Visualisation and diagnostic tools for a fitted ReportingDelayModel.

    All methods are static — instantiation is optional.

    Examples
    --------
    >>> from insurance_nowcast import ReportingDelayDiagnostic
    >>> diag = ReportingDelayDiagnostic()
    >>> diag.plot_convergence(model)
    >>> diag.plot_development_pattern(model)
    """

    @staticmethod
    def plot_convergence(
        model: "ReportingDelayModel",
        ax=None,
        figsize: tuple = (10, 4),
    ):
        """
        Plot the observed-data log-likelihood across EM iterations.

        A well-converged model shows a curve that flattens out. Oscillation
        (up-down pattern) is a warning sign — consider reducing learning rate
        or increasing regularisation.

        Parameters
        ----------
        model : ReportingDelayModel
            A fitted model.
        ax : matplotlib Axes or None
        figsize : tuple

        Returns
        -------
        matplotlib Figure
        """
        _check_matplotlib()
        import matplotlib.pyplot as plt

        if not hasattr(model, "convergence_history_") or not model.convergence_history_:
            raise RuntimeError(
                "Model has no convergence history. Has it been fitted? "
                "Check model.convergence_history_"
            )

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        history = model.convergence_history_
        ax.plot(range(1, len(history) + 1), history, "b-o", markersize=4)
        ax.set_xlabel("EM Iteration")
        ax.set_ylabel("Observed Log-Likelihood")
        ax.set_title("EM Algorithm Convergence")

        # Annotate final iteration
        ax.axhline(history[-1], color="gray", linestyle="--", alpha=0.5)
        ax.annotate(
            f"Final: {history[-1]:.2f}",
            xy=(len(history), history[-1]),
            xytext=(max(1, len(history) - 3), history[-1] + abs(history[-1]) * 0.01),
            fontsize=9,
        )

        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_development_pattern(
        model: "ReportingDelayModel",
        periods_to_show: list | None = None,
        ax=None,
        figsize: tuple = (12, 5),
    ):
        """
        Plot the cumulative delay distribution (development pattern) by occurrence period.

        Shows how reporting accumulates over time. Each line is one occurrence
        period. Lines that plateau early have fast reporting; lines with long
        flat tails have slow reporting.

        The horizontal dashed line at the latest truncation point shows where
        data becomes unavailable.

        Parameters
        ----------
        model : ReportingDelayModel
            A fitted model.
        periods_to_show : list or None
            Specific occurrence periods to plot. None = all.
        ax : matplotlib Axes or None
        figsize : tuple

        Returns
        -------
        matplotlib Figure
        """
        _check_matplotlib()
        import matplotlib.pyplot as plt

        _assert_fitted(model)

        p_hat = model.p_hat_  # (n_periods, max_delay)
        occ_periods = model.occurrence_periods_
        trunc = model.truncation_depth_

        if periods_to_show is not None:
            mask = np.isin(occ_periods, periods_to_show)
            p_hat = p_hat[mask]
            occ_periods = occ_periods[mask]
            trunc = trunc[mask]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        cmap = plt.get_cmap("viridis")
        n = len(occ_periods)

        for idx, (period, tau) in enumerate(zip(occ_periods, trunc)):
            colour = cmap(idx / max(n - 1, 1))
            cumulative = np.cumsum(p_hat[idx])
            ax.plot(
                range(len(cumulative)),
                cumulative,
                color=colour,
                alpha=0.7,
                linewidth=1.5,
                label=f"Period {period}" if n <= 10 else None,
            )
            # Mark truncation point
            if tau < model.max_delay_periods:
                ax.axvline(tau, color=colour, linestyle=":", alpha=0.3)

        ax.axhline(1.0, color="black", linestyle="--", alpha=0.3)
        ax.set_xlabel("Delay Periods")
        ax.set_ylabel("Cumulative Reporting Fraction")
        ax.set_title("Development Pattern by Occurrence Period")
        ax.set_ylim(0, 1.05)
        if n <= 10:
            ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_ibnr_by_period(
        model: "ReportingDelayModel",
        ax=None,
        figsize: tuple = (10, 5),
    ):
        """
        Bar chart of IBNR count and observed claims by occurrence period.

        Shows the stacked breakdown of:
        - Observed claims (reported by eval_date)
        - IBNR (estimated unreported claims)

        Use this to understand which periods drive the total IBNR loading.

        Parameters
        ----------
        model : ReportingDelayModel
        ax : matplotlib Axes or None
        figsize : tuple

        Returns
        -------
        matplotlib Figure
        """
        _check_matplotlib()
        import matplotlib.pyplot as plt

        _assert_fitted(model)

        if not hasattr(model, "_ibnr_df"):
            model._ibnr_df = model._compute_ibnr_df()

        df = model._ibnr_df

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        x = np.arange(len(df))
        ax.bar(
            x,
            df["observed_claims"],
            label="Observed",
            colour="steelblue",
            alpha=0.8,
        )
        ax.bar(
            x,
            df["ibnr_count"],
            bottom=df["observed_claims"],
            label="IBNR (estimated)",
            colour="coral",
            alpha=0.8,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(df["occurrence_period"], rotation=45, ha="right")
        ax.set_xlabel("Occurrence Period")
        ax.set_ylabel("Claim Count")
        ax.set_title("Observed Claims vs IBNR by Occurrence Period")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_delay_distribution(
        model: "ReportingDelayModel",
        X: pd.DataFrame,
        labels: list[str] | None = None,
        ax=None,
        figsize: tuple = (10, 5),
    ):
        """
        Plot delay probability mass functions for specific risk profiles.

        Useful for understanding how the delay distribution differs across
        risk segments. Pass one row per risk profile you want to compare.

        Parameters
        ----------
        model : ReportingDelayModel
            A fitted model.
        X : pd.DataFrame
            Risk features — one row per profile to compare.
        labels : list[str] or None
            Labels for each row of X.
        ax : matplotlib Axes or None
        figsize : tuple

        Returns
        -------
        matplotlib Figure
        """
        _check_matplotlib()
        import matplotlib.pyplot as plt

        _assert_fitted(model)

        p = model.predict_development_pattern(X)
        n_profiles = len(X)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        if labels is None:
            labels = [f"Profile {i + 1}" for i in range(n_profiles)]

        for i, label in enumerate(labels):
            ax.plot(
                range(model.max_delay_periods),
                p[i],
                "o-",
                label=label,
                markersize=4,
                alpha=0.8,
            )

        ax.set_xlabel("Delay Periods")
        ax.set_ylabel("Probability")
        ax.set_title("Delay Probability Distribution by Risk Profile")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig


# ---- Helpers ------------------------------------------------------------

def _check_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        raise ImportError(
            "matplotlib is required for diagnostic plots. "
            "Install with: pip install insurance-nowcast[plots]"
        )


def _assert_fitted(model: "ReportingDelayModel") -> None:
    if not getattr(model, "_is_fitted", False):
        raise RuntimeError(
            "Model has not been fitted yet. Call model.fit() first."
        )
