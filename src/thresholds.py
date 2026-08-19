"""Training-only operating thresholds and the frozen economic configuration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.base import clone

from src.evaluation import (
    EMP_IMPLEMENTATION_VERSION,
    EMP_P0,
    EMP_P1,
    emp_credit_scoring,
)
from src.models import predict_positive_proba
from src.run_utils import write_json_atomic


def base_rate_threshold(y_train) -> float:
    """Training default rate; the descriptive operating point."""
    return float(np.asarray(y_train).mean())


def oof_raw_scores(pipeline, X_train, y_train, folds) -> np.ndarray:
    """Cross-fitted raw scores for every training observation, exactly once."""
    y_train = np.asarray(y_train)
    predictions = np.full(len(y_train), np.nan, dtype=float)
    covered = np.zeros(len(y_train), dtype=int)
    for fit_index, held_out_index in folds:
        model = clone(pipeline)
        model.fit(_take(X_train, fit_index), y_train[fit_index])
        predictions[held_out_index] = predict_positive_proba(
            model, _take(X_train, held_out_index)
        )
        covered[held_out_index] += 1
    if not np.all(covered == 1):
        raise ValueError(
            "Out-of-fold coverage is not exactly one prediction per training row: "
            f"{np.bincount(covered).tolist()}"
        )
    if np.isnan(predictions).any():
        raise ValueError("Out-of-fold predictions contain missing values")
    return predictions


def emp_cutoff_from_oof(y_train, p_oof, roi, *, prior_default) -> tuple[float, float]:
    """Rejected fraction and score cut-off from training out-of-fold scores."""
    _, eta = emp_credit_scoring(y_train, p_oof, roi, prior_default=prior_default)
    cutoff = float(np.quantile(np.asarray(p_oof, dtype=float), 1.0 - eta))
    return cutoff, float(eta)


def build_economic_config(
    *,
    dataset,
    roi,
    prior_default,
    thresholds_by_model,
    roi_provenance,
) -> dict:
    """The immutable cost model, frozen before the test partition is opened."""
    return {
        "dataset": dataset,
        "emp_implementation_version": EMP_IMPLEMENTATION_VERSION,
        "prior_default_train": float(prior_default),
        "prior_non_default_train": float(1.0 - prior_default),
        "prior_source": "training partition, frozen before test access",
        "roi": float(roi),
        "roi_provenance": roi_provenance,
        "loss_given_default_mixture": {
            "distribution": "Beta mixture per Verbraken et al. (2014)",
            "p0": EMP_P0,
            "p1": EMP_P1,
            "density": 1.0 - EMP_P0 - EMP_P1,
        },
        "thresholds": thresholds_by_model,
        "note": (
            "The observed test default rate may be reported descriptively but must "
            "not replace prior_default_train inside the EMP calculation."
        ),
    }


def write_economic_config(path, config: dict) -> None:
    write_json_atomic(path, config)


def load_economic_config(path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. The economic configuration must be frozen before "
            "any test-partition analysis runs."
        )
    return json.loads(path.read_text())


def _take(X, index):
    """Row selection that works for DataFrames and arrays alike."""
    if hasattr(X, "iloc"):
        return X.iloc[index]
    return np.asarray(X)[index]
