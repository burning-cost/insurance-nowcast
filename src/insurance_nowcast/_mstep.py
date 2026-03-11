"""
M-step model fitting for the ML-EM nowcasting algorithm.

The M-step updates the occurrence and delay models by maximising the
complete-data log-likelihood on the imputed dataset from the E-step.

Two backends are supported:
- 'glm': Poisson GLM for occurrence, multinomial logit for delay.
  Lower capacity, faster, works well on small datasets.
- 'xgboost': XGBoost with base_margin for exposure offset in occurrence,
  multi:softprob for delay. Better for non-linear effects and large datasets.

The XGBoost implementation follows Wilsens et al. (2024): trees are added
to the previous model rather than refitting from scratch. This is the
'additive boosting across EM iterations' approach that provides de facto
monotone likelihood improvement.
"""
from __future__ import annotations

from typing import Any

import numpy as np


# ---- GLM occurrence model wrapper ----------------------------------------

class PoissonGLMOccurrence:
    """
    Poisson GLM occurrence model using sklearn.

    Wraps sklearn's PoissonRegressor to provide a consistent interface
    for the M-step. The exposure offset enters as a fixed offset term
    in the log-rate: log(lambda_i) = f(x_i) + log(e_i).

    For sklearn, we implement the offset by including log(exposure) as a
    fixed feature column and setting its coefficient to 1.0 during training.
    In practice, we use the sample_weight mechanism to approximate the offset:
    the target is the rate (N/E) and we weight by E.
    """

    def __init__(
        self,
        exposure_offset: bool = True,
        max_iter: int = 200,
        alpha: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.exposure_offset = exposure_offset
        self.max_iter = max_iter
        self.alpha = alpha
        self.random_state = random_state
        self._model = None

    def fit(
        self,
        X: np.ndarray,
        y_counts: np.ndarray,
        exposure: np.ndarray,
    ) -> "PoissonGLMOccurrence":
        """
        Fit Poisson GLM on imputed occurrence counts.

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            Risk features.
        y_counts : np.ndarray, shape (n,)
            Imputed total occurrence counts per period (N_i^{(k)}).
        exposure : np.ndarray, shape (n,)
            Exposure per period.
        """
        from sklearn.linear_model import PoissonRegressor

        if self.exposure_offset:
            # Model: E[Y] = exp(f(X) + log(E)) = E * exp(f(X))
            # Equivalently, rate = Y/E follows exp(f(X))
            # Use rate as target with sample_weight = exposure
            y_rate = y_counts / np.maximum(exposure, 1e-10)
            sample_weight = exposure
        else:
            y_rate = y_counts
            sample_weight = None

        self._model = PoissonRegressor(
            alpha=self.alpha,
            max_iter=self.max_iter,
            fit_intercept=True,
        )
        # Ensure no negative counts
        y_rate = np.maximum(y_rate, 0.0)
        self._model.fit(X, y_rate, sample_weight=sample_weight)
        self._exposure_used = exposure.copy()
        return self

    def predict(
        self,
        X: np.ndarray,
        exposure: np.ndarray,
    ) -> np.ndarray:
        """
        Predict expected occurrence counts.

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
        exposure : np.ndarray, shape (n,)

        Returns
        -------
        np.ndarray, shape (n,)
            Expected occurrence counts (lambda * exposure).
        """
        if self._model is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        rate = self._model.predict(X)
        if self.exposure_offset:
            return rate * exposure
        return rate


# ---- XGBoost occurrence model wrapper ------------------------------------

class XGBoostOccurrence:
    """
    XGBoost Poisson occurrence model with exposure offset via base_margin.

    The exposure offset is implemented using XGBoost's base_margin parameter:
        F(x_i) = f_{trees}(x_i) + log(e_i)

    New trees are added to the previous model at each EM iteration (additive
    construction across iterations), which provides de facto monotone
    improvement — the key property from Wilsens et al.
    """

    def __init__(
        self,
        exposure_offset: bool = True,
        n_estimators_per_iter: int = 50,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        min_child_weight: float = 10.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 5.0,
        reg_alpha: float = 0.0,
        random_state: int = 42,
    ) -> None:
        self.exposure_offset = exposure_offset
        self.n_estimators_per_iter = n_estimators_per_iter
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.random_state = random_state
        self._booster = None
        self._n_features = None

    def fit(
        self,
        X: np.ndarray,
        y_counts: np.ndarray,
        exposure: np.ndarray,
        em_iteration: int = 0,
    ) -> "XGBoostOccurrence":
        """
        Fit or update the XGBoost occurrence model.

        On the first EM iteration (em_iteration == 0), creates a new booster.
        On subsequent iterations, adds new trees to the existing model.

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
        y_counts : np.ndarray, shape (n,)
            Imputed occurrence counts.
        exposure : np.ndarray, shape (n,)
        em_iteration : int
            Current EM iteration number (0-indexed).
        """
        import xgboost as xgb

        self._n_features = X.shape[1]
        exposure = np.maximum(exposure, 1e-10)

        if self.exposure_offset:
            base_margin = np.log(exposure)
        else:
            base_margin = np.zeros(len(X))

        dtrain = xgb.DMatrix(
            X,
            label=y_counts,
            base_margin=base_margin,
        )

        params = {
            "objective": "count:poisson",
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "reg_alpha": self.reg_alpha,
            "seed": self.random_state,
            "verbosity": 0,
            "poisson_max_delta_step": 0.7,
        }

        if em_iteration == 0 or self._booster is None:
            self._booster = xgb.train(
                params,
                dtrain,
                num_boost_round=self.n_estimators_per_iter,
                verbose_eval=False,
            )
        else:
            # Additive construction: add trees to existing model
            self._booster = xgb.train(
                params,
                dtrain,
                num_boost_round=self.n_estimators_per_iter,
                xgb_model=self._booster,
                verbose_eval=False,
            )

        return self

    def predict(
        self,
        X: np.ndarray,
        exposure: np.ndarray,
    ) -> np.ndarray:
        """
        Predict expected occurrence counts (lambda * exposure).
        """
        import xgboost as xgb

        if self._booster is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        exposure = np.maximum(exposure, 1e-10)
        if self.exposure_offset:
            base_margin = np.log(exposure)
        else:
            base_margin = np.zeros(len(X))

        dtest = xgb.DMatrix(X, base_margin=base_margin)
        return self._booster.predict(dtest)


# ---- GLM delay model wrapper --------------------------------------------

class MultinomialGLMDelay:
    """
    Multinomial delay model using sklearn LogisticRegression (softmax).

    Fits a multinomial logistic regression on the (occurrence_period,
    delay_period) training data, with imputed counts as sample weights.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self._model = None
        self._n_classes = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        n_classes: int,
    ) -> "MultinomialGLMDelay":
        """
        Fit multinomial delay model.

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            Features — one row per (occurrence_period, delay_period) pair.
        y : np.ndarray, shape (n,)
            Delay period index (0, 1, ..., n_classes - 1).
        sample_weight : np.ndarray, shape (n,)
            Imputed counts N_{i,j}^{(k)}.
        n_classes : int
            Number of delay classes (= max_delay_periods).
        """
        from sklearn.linear_model import LogisticRegression

        self._n_classes = n_classes
        self._model = LogisticRegression(
            solver="lbfgs",
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            fit_intercept=True,
        )
        # Ensure all classes are present — add pseudo-observations if needed
        X_train, y_train, w_train = _ensure_all_classes(X, y, sample_weight, n_classes)
        self._model.fit(X_train, y_train, sample_weight=w_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict delay probability distribution.

        Returns
        -------
        np.ndarray, shape (n, n_classes)
            Softmax probabilities summing to 1 across delay periods.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.predict_proba(X)


# ---- XGBoost delay model wrapper ----------------------------------------

class XGBoostDelay:
    """
    XGBoost multinomial delay model using multi:softprob objective.

    Each (occurrence_period, delay_period) pair is one training row,
    weighted by the imputed count N_{i,j}^{(k)}. The target is the
    delay period index. XGBoost's softprob output gives probabilities
    across all delay periods.

    Like the occurrence model, trees are added additively across EM
    iterations rather than refitting from scratch.
    """

    def __init__(
        self,
        n_estimators_per_iter: int = 50,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        min_child_weight: float = 5.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 5.0,
        reg_alpha: float = 0.0,
        random_state: int = 42,
    ) -> None:
        self.n_estimators_per_iter = n_estimators_per_iter
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.random_state = random_state
        self._booster = None
        self._n_classes = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        n_classes: int,
        em_iteration: int = 0,
    ) -> "XGBoostDelay":
        """
        Fit or update XGBoost delay model.

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
        y : np.ndarray, shape (n,) — delay period index
        sample_weight : np.ndarray, shape (n,)
        n_classes : int
        em_iteration : int
        """
        import xgboost as xgb

        self._n_classes = n_classes

        dtrain = xgb.DMatrix(
            X,
            label=y,
            weight=sample_weight,
        )

        params = {
            "objective": "multi:softprob",
            "num_class": n_classes,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "reg_alpha": self.reg_alpha,
            "seed": self.random_state,
            "verbosity": 0,
        }

        if em_iteration == 0 or self._booster is None:
            self._booster = xgb.train(
                params,
                dtrain,
                num_boost_round=self.n_estimators_per_iter,
                verbose_eval=False,
            )
        else:
            self._booster = xgb.train(
                params,
                dtrain,
                num_boost_round=self.n_estimators_per_iter,
                xgb_model=self._booster,
                verbose_eval=False,
            )

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict delay probability distribution.

        Returns
        -------
        np.ndarray, shape (n, n_classes)
        """
        import xgboost as xgb

        if self._booster is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        dtest = xgb.DMatrix(X)
        raw = self._booster.predict(dtest)
        # XGBoost returns flat array for multi:softprob — reshape
        if raw.ndim == 1:
            raw = raw.reshape(-1, self._n_classes)
        return raw


# ---- Helper functions ---------------------------------------------------

def _ensure_all_classes(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ensure all class labels are present in the training data.

    sklearn's multinomial LogisticRegression requires all classes to be seen
    during fit. For sparse delay data, some delay periods may have zero weight.
    We add tiny pseudo-observations for missing classes.
    """
    present = set(y.astype(int))
    missing = set(range(n_classes)) - present
    if not missing:
        return X, y, w

    x_pseudo = np.zeros((len(missing), X.shape[1] if X.ndim > 1 else 1), dtype=float)
    if X.ndim == 1:
        x_pseudo = x_pseudo.ravel()
    y_pseudo = np.array(sorted(missing), dtype=int)
    w_pseudo = np.full(len(missing), 1e-10)

    X_out = np.vstack([X, x_pseudo])
    y_out = np.concatenate([y, y_pseudo])
    w_out = np.concatenate([w, w_pseudo])
    return X_out, y_out, w_out


def build_occurrence_model(model_type: str, **kwargs: Any) -> Any:
    """
    Factory function for occurrence models.

    Parameters
    ----------
    model_type : str
        One of 'glm', 'xgboost'.
    **kwargs
        Passed to the model constructor.
    """
    if model_type == "glm":
        return PoissonGLMOccurrence(**kwargs)
    elif model_type == "xgboost":
        return XGBoostOccurrence(**kwargs)
    else:
        raise ValueError(
            f"Unknown occurrence_model '{model_type}'. "
            "Supported: 'glm', 'xgboost'"
        )


def build_delay_model(model_type: str, **kwargs: Any) -> Any:
    """
    Factory function for delay models.

    Parameters
    ----------
    model_type : str
        One of 'glm', 'xgboost'.
    **kwargs
        Passed to the model constructor.
    """
    if model_type == "glm":
        return MultinomialGLMDelay(**kwargs)
    elif model_type == "xgboost":
        return XGBoostDelay(**kwargs)
    else:
        raise ValueError(
            f"Unknown delay_model '{model_type}'. "
            "Supported: 'glm', 'xgboost'"
        )
