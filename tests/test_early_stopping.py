"""Early stopping is confined to the search; the round count is frozen after it."""

from __future__ import annotations

import numpy as np
import pytest
from catboost import CatBoostClassifier
from sklearn.datasets import make_classification
import xgboost as xgb

from src.models import FitLocalPosWeightClassifier, model_seed
from src.seeds import ES_MAX_ROUNDS, ES_PATIENCE


def _data(n=500, seed=0):
    X, y = make_classification(
        n_samples=n, n_features=8, n_informative=5, weights=[0.75, 0.25],
        random_state=seed,
    )
    fit_index = np.arange(0, int(n * 0.8))
    stop_index = np.arange(int(n * 0.8), n)
    return X[fit_index], y[fit_index], X[stop_index], y[stop_index]


def _xgb_wrapper(cap=300):
    return FitLocalPosWeightClassifier(
        estimator=xgb.XGBClassifier(
            n_estimators=cap,
            eval_metric="logloss",
            verbosity=0,
            random_state=model_seed("XGBoost"),
            tree_method="hist",
        )
    )


def _cat_wrapper(cap=300):
    return FitLocalPosWeightClassifier(
        estimator=CatBoostClassifier(
            iterations=cap,
            eval_metric="Logloss",
            random_seed=model_seed("CatBoost"),
            verbose=0,
            allow_writing_files=False,
        )
    )


def test_xgboost_stops_and_is_deterministic():
    X_fit, y_fit, X_stop, y_stop = _data()
    first = _xgb_wrapper().fit(X_fit, y_fit, eval_set=(X_stop, y_stop))
    second = _xgb_wrapper().fit(X_fit, y_fit, eval_set=(X_stop, y_stop))
    assert first.best_iteration_ is not None
    assert first.best_iteration_ == second.best_iteration_
    np.testing.assert_array_equal(
        first.predict_proba(X_stop), second.predict_proba(X_stop)
    )


def test_catboost_stops_and_is_deterministic():
    X_fit, y_fit, X_stop, y_stop = _data()
    first = _cat_wrapper().fit(X_fit, y_fit, eval_set=(X_stop, y_stop))
    second = _cat_wrapper().fit(X_fit, y_fit, eval_set=(X_stop, y_stop))
    assert first.best_iteration_ is not None
    assert first.best_iteration_ == second.best_iteration_
    np.testing.assert_allclose(
        first.predict_proba(X_stop), second.predict_proba(X_stop), rtol=1e-9
    )


def test_easily_separable_data_stops_before_the_cap():
    X, y = make_classification(
        n_samples=600, n_features=6, n_informative=6, n_redundant=0,
        class_sep=4.0, random_state=1,
    )
    fitted = _xgb_wrapper(cap=ES_MAX_ROUNDS).fit(
        X[:480], y[:480], eval_set=(X[480:], y[480:])
    )
    assert fitted.best_iteration_ is not None
    assert fitted.best_iteration_ + 1 < ES_MAX_ROUNDS


def test_plain_fit_has_no_evaluation_set_and_no_best_iteration():
    """Refit, calibration and threshold fits never see an evaluation set."""
    X_fit, y_fit, _, _ = _data()
    fitted = _xgb_wrapper(cap=25).fit(X_fit, y_fit)
    assert fitted.best_iteration_ is None
    assert fitted.estimator_.get_params()["n_estimators"] == 25


def test_patience_is_the_frozen_constant():
    X_fit, y_fit, X_stop, y_stop = _data()
    fitted = _xgb_wrapper().fit(X_fit, y_fit, eval_set=(X_stop, y_stop))
    assert fitted.estimator_.get_params()["early_stopping_rounds"] == ES_PATIENCE
