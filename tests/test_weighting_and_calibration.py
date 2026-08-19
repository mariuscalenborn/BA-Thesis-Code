"""Class weighting stays fit-local; the calibrator stays unweighted."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from src.models import (
    MODEL_NAMES,
    BalancedEBMClassifier,
    FitLocalPosWeightClassifier,
    build_estimator,
    build_pipeline,
    fit_calibrated,
    positive_class_column,
    predict_positive_proba,
)
from src.seeds import SEEDS


def _imbalanced(y, ratio):
    """Indices with roughly ``ratio`` negatives per positive."""
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    take = min(len(negative), int(ratio * len(positive)))
    return np.concatenate([positive, negative[:take]])


def test_pos_weight_is_recomputed_for_every_fit_subset(synthetic_xy):
    X, y = synthetic_xy
    estimator = build_estimator("XGBoost", use_gpu=False, ebm_bags=2)
    assert isinstance(estimator, FitLocalPosWeightClassifier)

    balanced = _imbalanced(y, ratio=1)
    skewed = _imbalanced(y, ratio=6)

    first = clone(estimator).fit(X.iloc[balanced].select_dtypes("number").fillna(0), y[balanced])
    second = clone(estimator).fit(X.iloc[skewed].select_dtypes("number").fillna(0), y[skewed])
    assert first.scale_pos_weight_ != second.scale_pos_weight_
    assert second.scale_pos_weight_ > first.scale_pos_weight_


def test_clone_resets_the_learned_weight(synthetic_xy):
    X, y = synthetic_xy
    numeric = X.select_dtypes("number").fillna(0)
    fitted = build_estimator("CatBoost", use_gpu=False, ebm_bags=2)
    fitted.set_params(estimator__iterations=10)
    fitted.fit(numeric, y)
    assert hasattr(fitted, "scale_pos_weight_")
    assert not hasattr(clone(fitted), "scale_pos_weight_")


def test_logistic_regression_uses_balanced_weights(synthetic_config):
    pipeline = build_pipeline("LR", synthetic_config, use_gpu=False, ebm_bags=2)
    assert pipeline.named_steps["clf"].named_steps["lr"].class_weight == "balanced"


def test_ebm_weights_are_computed_inside_fit(synthetic_config, synthetic_xy):
    X, y = synthetic_xy
    estimator = build_estimator("EBM", use_gpu=False, ebm_bags=2)
    assert isinstance(estimator, BalancedEBMClassifier)

    recorded = {}
    original = estimator.estimator.__class__.fit

    def spy(self, X_inner, y_inner, sample_weight=None, **kwargs):
        recorded["sample_weight"] = sample_weight
        return original(self, X_inner, y_inner, sample_weight=sample_weight, **kwargs)

    estimator.estimator.__class__.fit = spy
    try:
        estimator.fit(X.select_dtypes("number").fillna(0), y)
    finally:
        estimator.estimator.__class__.fit = original

    weights = recorded["sample_weight"]
    assert weights is not None
    assert weights[y == 1].mean() > weights[y == 0].mean()


def test_calibrator_receives_no_sample_weight(synthetic_config, synthetic_xy):
    X, y = synthetic_xy
    from sklearn.calibration import CalibratedClassifierCV

    recorded = {}
    original = CalibratedClassifierCV.fit

    def spy(self, X_inner, y_inner, sample_weight=None, **kwargs):
        recorded["sample_weight"] = sample_weight
        recorded["ensemble"] = self.ensemble
        return original(self, X_inner, y_inner, sample_weight=sample_weight, **kwargs)

    CalibratedClassifierCV.fit = spy
    try:
        pipeline = build_pipeline("LR", synthetic_config, use_gpu=False, ebm_bags=2)
        fit_calibrated(pipeline, X, y, cv=_calibration_folds(y))
    finally:
        CalibratedClassifierCV.fit = original

    assert recorded["sample_weight"] is None
    assert recorded["ensemble"] is False


def _calibration_folds(y):
    return list(
        StratifiedKFold(n_splits=3, shuffle=True, random_state=SEEDS["calibration_cv"])
        .split(np.zeros(len(y), dtype=np.uint8), y)
    )


def test_calibration_uses_three_folds_and_one_final_base_estimator(
    synthetic_config, synthetic_xy
):
    """``ensemble=False`` means one base model that saw all training data."""
    X, y = synthetic_xy
    pipeline = build_pipeline("LR", synthetic_config, use_gpu=False, ebm_bags=2)
    calibrated = fit_calibrated(pipeline, X, y, cv=_calibration_folds(y))

    assert calibrated.ensemble is False
    assert len(calibrated.calibrated_classifiers_) == 1
    base = calibrated.calibrated_classifiers_[0].estimator
    assert base.named_steps["clf"].named_steps["lr"].n_features_in_ > 0


def test_raw_and_calibrated_predictions_are_different_objects(
    synthetic_config, synthetic_xy
):
    X, y = synthetic_xy
    pipeline = build_pipeline("LR", synthetic_config, use_gpu=False, ebm_bags=2)
    raw = clone(pipeline).fit(X, y)
    calibrated = fit_calibrated(clone(pipeline), X, y, cv=_calibration_folds(y))

    p_raw = predict_positive_proba(raw, X)
    p_calibrated = predict_positive_proba(calibrated, X)
    assert not np.allclose(p_raw, p_calibrated)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_positive_class_column_is_verified_not_assumed(
    name, synthetic_config, synthetic_xy
):
    X, y = synthetic_xy
    pipeline = build_pipeline(name, synthetic_config, use_gpu=False, ebm_bags=2)
    if name == "CatBoost":
        pipeline.set_params(clf__estimator__iterations=10)
    if name == "XGBoost":
        pipeline.set_params(clf__estimator__n_estimators=10)
    pipeline.fit(X, y)
    assert positive_class_column(pipeline) == 1


def test_positive_class_column_raises_on_a_missing_label():
    class Fake:
        classes_ = np.asarray([0, 2])

    with pytest.raises(ValueError, match="not uniquely present"):
        positive_class_column(Fake())
