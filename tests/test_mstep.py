"""
Tests for M-step model fitting (GLM and XGBoost backends).
"""
import numpy as np
import pytest

from insurance_nowcast._mstep import (
    PoissonGLMOccurrence,
    XGBoostOccurrence,
    MultinomialGLMDelay,
    XGBoostDelay,
    build_occurrence_model,
    build_delay_model,
    _ensure_all_classes,
)


# ---- PoissonGLMOccurrence ----

class TestPoissonGLMOccurrence:
    def _fit_basic(self, exposure_offset=True):
        rng = np.random.default_rng(42)
        n = 50
        X = rng.uniform(0, 1, size=(n, 2))
        exposure = rng.uniform(0.5, 2.0, size=n)
        y = rng.poisson(0.05 * exposure)
        model = PoissonGLMOccurrence(exposure_offset=exposure_offset, max_iter=200)
        model.fit(X, y.astype(float), exposure)
        return model, X, exposure, y

    def test_fit_returns_self(self):
        model = PoissonGLMOccurrence()
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (20, 2))
        y = rng.poisson(2.0, size=20).astype(float)
        exp = np.ones(20)
        result = model.fit(X, y, exp)
        assert result is model

    def test_predict_shape(self):
        model, X, exposure, _ = self._fit_basic()
        preds = model.predict(X, exposure)
        assert preds.shape == (len(X),)

    def test_predict_positive(self):
        model, X, exposure, _ = self._fit_basic()
        preds = model.predict(X, exposure)
        assert (preds > 0).all()

    def test_predict_before_fit_raises(self):
        model = PoissonGLMOccurrence()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(np.ones((3, 2)), np.ones(3))

    def test_exposure_offset_scales_prediction(self):
        """With exposure_offset=True, predictions scale with exposure."""
        rng = np.random.default_rng(0)
        X = np.ones((10, 1)) * 0.5
        y = np.ones(10) * 5.0
        exp_low = np.ones(10) * 1.0
        exp_high = np.ones(10) * 10.0

        model = PoissonGLMOccurrence(exposure_offset=True)
        model.fit(X, y, exp_low)

        pred_low = model.predict(X, exp_low)
        pred_high = model.predict(X, exp_high)

        # Prediction with 10x exposure should be ~10x higher
        ratio = pred_high.mean() / pred_low.mean()
        assert 5 < ratio < 20  # Approximate

    def test_no_exposure_offset(self):
        model, X, exposure, _ = self._fit_basic(exposure_offset=False)
        preds = model.predict(X, exposure)
        assert preds.shape == (len(X),)
        assert (preds > 0).all()


# ---- XGBoostOccurrence ----

class TestXGBoostOccurrence:
    def _make_data(self, n=40):
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (n, 3))
        exposure = rng.uniform(0.5, 2.0, n)
        y = rng.poisson(0.08 * exposure)
        return X, exposure, y.astype(float)

    def test_fit_returns_self(self):
        X, exp, y = self._make_data()
        model = XGBoostOccurrence(n_estimators_per_iter=5)
        result = model.fit(X, y, exp, em_iteration=0)
        assert result is model

    def test_predict_shape(self):
        X, exp, y = self._make_data()
        model = XGBoostOccurrence(n_estimators_per_iter=5)
        model.fit(X, y, exp, em_iteration=0)
        preds = model.predict(X, exp)
        assert preds.shape == (len(X),)

    def test_predict_positive(self):
        X, exp, y = self._make_data()
        model = XGBoostOccurrence(n_estimators_per_iter=5)
        model.fit(X, y, exp, em_iteration=0)
        preds = model.predict(X, exp)
        assert (preds > 0).all()

    def test_additive_iteration_increases_trees(self):
        """Second EM iteration should add trees to the model."""
        X, exp, y = self._make_data()
        model = XGBoostOccurrence(n_estimators_per_iter=10)
        model.fit(X, y, exp, em_iteration=0)
        n_trees_first = model._booster.num_boosted_rounds()
        model.fit(X, y, exp, em_iteration=1)
        n_trees_second = model._booster.num_boosted_rounds()
        assert n_trees_second > n_trees_first

    def test_predict_before_fit_raises(self):
        model = XGBoostOccurrence()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(np.ones((3, 2)), np.ones(3))

    def test_exposure_offset_available(self):
        X, exp, y = self._make_data()
        model = XGBoostOccurrence(n_estimators_per_iter=5, exposure_offset=True)
        model.fit(X, y, exp, em_iteration=0)
        preds_1 = model.predict(X, np.ones(len(X)))
        preds_10 = model.predict(X, np.ones(len(X)) * 10)
        # 10x exposure should give ~10x prediction
        ratio = preds_10.mean() / preds_1.mean()
        assert 5 < ratio < 20


# ---- MultinomialGLMDelay ----

class TestMultinomialGLMDelay:
    def _make_delay_data(self, n=100, n_classes=6):
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (n, 2))
        y = rng.integers(0, n_classes, n)
        w = rng.uniform(0.5, 5.0, n)
        return X, y, w

    def test_fit_returns_self(self):
        X, y, w = self._make_delay_data()
        model = MultinomialGLMDelay()
        result = model.fit(X, y, w, n_classes=6)
        assert result is model

    def test_predict_proba_shape(self):
        X, y, w = self._make_delay_data(n=100, n_classes=6)
        model = MultinomialGLMDelay()
        model.fit(X, y, w, n_classes=6)
        proba = model.predict_proba(X[:5])
        assert proba.shape == (5, 6)

    def test_predict_proba_sums_to_one(self):
        X, y, w = self._make_delay_data(n=100, n_classes=6)
        model = MultinomialGLMDelay()
        model.fit(X, y, w, n_classes=6)
        proba = model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_non_negative(self):
        X, y, w = self._make_delay_data()
        model = MultinomialGLMDelay()
        model.fit(X, y, w, n_classes=6)
        proba = model.predict_proba(X)
        assert (proba >= 0).all()

    def test_predict_before_fit_raises(self):
        model = MultinomialGLMDelay()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(np.ones((3, 2)))

    def test_missing_classes_handled(self):
        """Should not raise even when some classes absent from training data."""
        # Only classes 0, 1, 2 present
        X = np.ones((30, 2))
        y = np.array([0] * 10 + [1] * 10 + [2] * 10)
        w = np.ones(30)
        model = MultinomialGLMDelay()
        model.fit(X, y, w, n_classes=6)  # n_classes=6 but only 3 present
        proba = model.predict_proba(np.ones((5, 2)))
        assert proba.shape == (5, 6)


# ---- XGBoostDelay ----

class TestXGBoostDelay:
    def _make_delay_data(self, n=100, n_classes=8):
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (n, 3))
        y = rng.integers(0, n_classes, n)
        w = rng.uniform(0.5, 5.0, n)
        return X, y, w

    def test_fit_returns_self(self):
        X, y, w = self._make_delay_data()
        model = XGBoostDelay(n_estimators_per_iter=10)
        result = model.fit(X, y, w, n_classes=8, em_iteration=0)
        assert result is model

    def test_predict_proba_shape(self):
        X, y, w = self._make_delay_data(n_classes=8)
        model = XGBoostDelay(n_estimators_per_iter=10)
        model.fit(X, y, w, n_classes=8, em_iteration=0)
        proba = model.predict_proba(X[:10])
        assert proba.shape == (10, 8)

    def test_predict_proba_sums_to_one(self):
        X, y, w = self._make_delay_data()
        model = XGBoostDelay(n_estimators_per_iter=10)
        model.fit(X, y, w, n_classes=8, em_iteration=0)
        proba = model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_proba_non_negative(self):
        X, y, w = self._make_delay_data()
        model = XGBoostDelay(n_estimators_per_iter=10)
        model.fit(X, y, w, n_classes=8, em_iteration=0)
        proba = model.predict_proba(X)
        assert (proba >= 0).all()

    def test_additive_construction(self):
        X, y, w = self._make_delay_data()
        model = XGBoostDelay(n_estimators_per_iter=10)
        model.fit(X, y, w, n_classes=8, em_iteration=0)
        n0 = model._booster.num_boosted_rounds()
        model.fit(X, y, w, n_classes=8, em_iteration=1)
        n1 = model._booster.num_boosted_rounds()
        assert n1 > n0

    def test_predict_before_fit_raises(self):
        model = XGBoostDelay()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(np.ones((3, 2)))


# ---- _ensure_all_classes ----

class TestEnsureAllClasses:
    def test_all_present_unchanged(self):
        X = np.ones((6, 2))
        y = np.array([0, 1, 2, 3, 4, 5])
        w = np.ones(6)
        X2, y2, w2 = _ensure_all_classes(X, y, w, n_classes=6)
        assert len(X2) == 6
        np.testing.assert_array_equal(y2, y)

    def test_missing_class_added(self):
        X = np.ones((3, 2))
        y = np.array([0, 1, 2])  # class 3 missing
        w = np.ones(3)
        X2, y2, w2 = _ensure_all_classes(X, y, w, n_classes=4)
        assert 3 in y2
        assert len(X2) == 4

    def test_tiny_weight_for_pseudo(self):
        X = np.ones((2, 2))
        y = np.array([0, 1])
        w = np.array([5.0, 3.0])
        X2, y2, w2 = _ensure_all_classes(X, y, w, n_classes=3)
        # pseudo-obs for class 2 should have tiny weight
        pseudo_mask = y2 == 2
        assert w2[pseudo_mask].max() < 1e-5


# ---- Factory functions ----

class TestFactoryFunctions:
    def test_build_occurrence_glm(self):
        model = build_occurrence_model("glm", exposure_offset=True)
        assert isinstance(model, PoissonGLMOccurrence)

    def test_build_occurrence_xgboost(self):
        model = build_occurrence_model("xgboost", exposure_offset=True)
        assert isinstance(model, XGBoostOccurrence)

    def test_build_occurrence_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown occurrence_model"):
            build_occurrence_model("neural_network")

    def test_build_delay_glm(self):
        model = build_delay_model("glm")
        assert isinstance(model, MultinomialGLMDelay)

    def test_build_delay_xgboost(self):
        model = build_delay_model("xgboost")
        assert isinstance(model, XGBoostDelay)

    def test_build_delay_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown delay_model"):
            build_delay_model("lightgbm")
