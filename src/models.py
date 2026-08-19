"""Model construction and probability calibration for the four model families."""

from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

from src.datasets import DatasetConfig
from src.preprocessing import build_preprocessor
from src.seeds import ES_PATIENCE, ES_MAX_ROUNDS, model_seed


MODEL_NAMES = ("LR", "EBM", "XGBoost", "CatBoost")
POSITIVE_LABEL = 1


class BalancedEBMClassifier(ClassifierMixin, BaseEstimator):
    """Clone-compatible EBM wrapper that weights only the base estimator fit."""

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        base = self.estimator
        if base is None:
            base = ExplainableBoostingClassifier(
                random_state=model_seed("EBM"), n_jobs=-1
            )
        self.estimator_ = clone(base)
        weights = compute_sample_weight(class_weight="balanced", y=y)
        self.estimator_.fit(X, y, sample_weight=weights)
        self.classes_ = np.asarray(self.estimator_.classes_)
        if hasattr(self.estimator_, "n_features_in_"):
            self.n_features_in_ = self.estimator_.n_features_in_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.estimator_.predict(X)


class FitLocalPosWeightClassifier(ClassifierMixin, BaseEstimator):
    """Fit-local class weighting and early stopping for the boosted ensembles."""

    def __init__(self, estimator=None, weight_param="scale_pos_weight"):
        self.estimator = estimator
        self.weight_param = weight_param

    def fit(self, X, y, eval_set=None):
        if self.estimator is None:
            raise ValueError("FitLocalPosWeightClassifier requires an inner estimator")
        y = np.asarray(y)
        n_negative = int((y == 0).sum())
        n_positive = int((y == POSITIVE_LABEL).sum())
        self.scale_pos_weight_ = float(n_negative / max(n_positive, 1))
        self.estimator_ = clone(self.estimator)
        self.estimator_.set_params(**{self.weight_param: self.scale_pos_weight_})

        self.best_iteration_ = None
        if eval_set is None:
            if isinstance(self.estimator_, CatBoostClassifier):
                self.estimator_.fit(X, y, verbose=False)
            else:
                self.estimator_.fit(X, y)
        elif isinstance(self.estimator_, CatBoostClassifier):
            self.estimator_.set_params(eval_metric="AUC")
            self.estimator_.fit(
                X, y,
                eval_set=eval_set,
                early_stopping_rounds=ES_PATIENCE,
                use_best_model=True,
                verbose=False,
            )
            best = self.estimator_.get_best_iteration()
            self.best_iteration_ = int(best) if best is not None else None
        else:
            self.estimator_.set_params(
                early_stopping_rounds=ES_PATIENCE, eval_metric="auc"
            )
            self.estimator_.fit(X, y, eval_set=[eval_set], verbose=False)
            best = getattr(self.estimator_, "best_iteration", None)
            self.best_iteration_ = int(best) if best is not None else None

        self.classes_ = np.asarray(self.estimator_.classes_)
        if hasattr(self.estimator_, "n_features_in_"):
            self.n_features_in_ = self.estimator_.n_features_in_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.estimator_.predict(X)


def build_estimator(name: str, *, use_gpu: bool, ebm_bags: int):
    """Construct one of the four frozen model families with fit-local weighting."""
    if name == "LR":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=model_seed("LR"),
            )),
        ])
    if name == "EBM":
        return BalancedEBMClassifier(
            estimator=ExplainableBoostingClassifier(
                outer_bags=ebm_bags,
                interactions=10,
                random_state=model_seed("EBM"),
                n_jobs=-1,
            )
        )
    if name == "XGBoost":
        return FitLocalPosWeightClassifier(
            estimator=xgb.XGBClassifier(
                n_estimators=ES_MAX_ROUNDS,
                eval_metric="logloss",
                verbosity=0,
                random_state=model_seed("XGBoost"),
                tree_method="hist",
                device="cuda" if use_gpu else "cpu",
            )
        )
    if name == "CatBoost":
        return FitLocalPosWeightClassifier(
            estimator=CatBoostClassifier(
                iterations=ES_MAX_ROUNDS,
                eval_metric="Logloss",
                random_seed=model_seed("CatBoost"),
                verbose=0,
                allow_writing_files=False,
                thread_count=-1,
                task_type="GPU" if use_gpu else "CPU",
            )
        )
    raise ValueError(f"Unknown model: {name}")


def build_pipeline(name: str, config: DatasetConfig, *, use_gpu: bool, ebm_bags: int) -> Pipeline:
    """Preprocessing plus estimator as one cloneable, refittable object."""
    return Pipeline([
        ("prep", build_preprocessor(config)),
        ("clf", build_estimator(name, use_gpu=use_gpu, ebm_bags=ebm_bags)),
    ])


def inner_estimator(pipeline: Pipeline):
    """Return the object that actually carries the learned decision function."""
    estimator = pipeline.named_steps["clf"]
    if isinstance(estimator, (BalancedEBMClassifier, FitLocalPosWeightClassifier)):
        return estimator.estimator_
    return estimator


def positive_class_column(estimator) -> int:
    """Index of the positive class in ``predict_proba`` output."""
    classes = np.asarray(getattr(estimator, "classes_"))
    matches = np.flatnonzero(classes == POSITIVE_LABEL)
    if len(matches) != 1:
        raise ValueError(
            f"Positive label {POSITIVE_LABEL} not uniquely present in classes_={classes!r}"
        )
    return int(matches[0])


def predict_positive_proba(estimator, X) -> np.ndarray:
    """Probability of default, with the class index verified rather than assumed."""
    probabilities = estimator.predict_proba(X)
    return np.asarray(probabilities)[:, positive_class_column(estimator)]


def fit_calibrated(pipeline, X, y, *, cv, method: str = "isotonic"):
    """Fit training-only probability calibration around a selected pipeline."""
    calibrated = CalibratedClassifierCV(
        estimator=clone(pipeline),
        method=method,
        cv=cv,
        ensemble=False,
        n_jobs=1,
    )
    return calibrated.fit(X, y)
