"""Paired stratified bootstrap on the fixed test partition."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

from src.evaluation import emp_credit_scoring
from src.seeds import BOOTSTRAP_CI_LEVEL, BOOTSTRAP_CI_METHOD


BOOTSTRAP_METRICS = ("auc", "pr_auc", "brier", "emp", "f1_fix", "balacc_fix")


def paired_stratified_indices(y_test, *, n_bootstrap, random_state) -> np.ndarray:
    """One ``(n_bootstrap, n_test)`` index matrix, shared by all models."""
    y_test = np.asarray(y_test)
    positive = np.flatnonzero(y_test == 1)
    negative = np.flatnonzero(y_test == 0)
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("Stratified bootstrap needs both classes in the test partition")
    rng = np.random.default_rng(random_state)
    drawn_positive = rng.integers(0, len(positive), size=(n_bootstrap, len(positive)))
    drawn_negative = rng.integers(0, len(negative), size=(n_bootstrap, len(negative)))
    return np.concatenate(
        [positive[drawn_positive], negative[drawn_negative]], axis=1
    ).astype(np.int64)


def _replicate_metrics(y, p_raw, p_calibrated, *, emp_cutoff, base_rate_threshold, roi, prior_default):
    """The six bootstrap metrics on one resampled test set."""
    emp, _ = emp_credit_scoring(y, p_raw, roi, prior_default=prior_default)
    yhat_fixed = (p_calibrated > base_rate_threshold).astype(int)
    return {
        "auc": roc_auc_score(y, p_raw),
        "pr_auc": average_precision_score(y, p_raw),
        "brier": brier_score_loss(y, p_calibrated),
        "emp": emp,
        "f1_fix": f1_score(y, yhat_fixed, zero_division=0),
        "balacc_fix": balanced_accuracy_score(y, yhat_fixed),
    }


def bootstrap_model_metrics(
    y_test,
    scores_by_model,
    thresholds_by_model,
    *,
    roi,
    prior_default,
    indices,
):
    """Per-model point estimates, bootstrap distributions and percentile intervals."""
    y_test = np.asarray(y_test)
    draws = {}
    summary = []
    for name, (p_raw, p_calibrated) in scores_by_model.items():
        thresholds = thresholds_by_model[name]
        p_raw = np.asarray(p_raw, dtype=float)
        p_calibrated = np.asarray(p_calibrated, dtype=float)
        point = _replicate_metrics(
            y_test, p_raw, p_calibrated,
            emp_cutoff=thresholds["emp_cutoff"],
            base_rate_threshold=thresholds["base_rate_threshold"],
            roi=roi,
            prior_default=prior_default,
        )
        replicates = {metric: np.empty(len(indices)) for metric in BOOTSTRAP_METRICS}
        for replicate, index in enumerate(indices):
            values = _replicate_metrics(
                y_test[index], p_raw[index], p_calibrated[index],
                emp_cutoff=thresholds["emp_cutoff"],
                base_rate_threshold=thresholds["base_rate_threshold"],
                roi=roi,
                prior_default=prior_default,
            )
            for metric in BOOTSTRAP_METRICS:
                replicates[metric][replicate] = values[metric]
        draws[name] = replicates
        for metric in BOOTSTRAP_METRICS:
            summary.append({
                "model": name,
                "metric": metric,
                "point": float(point[metric]),
                "bootstrap_mean": float(np.mean(replicates[metric])),
                "ci_low": float(np.quantile(replicates[metric], 0.025)),
                "ci_high": float(np.quantile(replicates[metric], 0.975)),
                "ci_method": BOOTSTRAP_CI_METHOD,
                "ci_level": BOOTSTRAP_CI_LEVEL,
                "n_bootstrap": int(len(indices)),
            })
    return summary, draws


def bootstrap_pairwise_deltas(draws, summary):
    """Paired between-model differences with percentile intervals."""
    point_lookup = {
        (row["model"], row["metric"]): row["point"] for row in summary
    }
    records = []
    for left, right in combinations(draws, 2):
        for metric in BOOTSTRAP_METRICS:
            difference = draws[left][metric] - draws[right][metric]
            records.append({
                "model_a": left,
                "model_b": right,
                "metric": metric,
                "delta_point": float(
                    point_lookup[(left, metric)] - point_lookup[(right, metric)]
                ),
                "delta_bootstrap_mean": float(np.mean(difference)),
                "delta_ci_low": float(np.quantile(difference, 0.025)),
                "delta_ci_high": float(np.quantile(difference, 0.975)),
                "share_delta_positive": float(np.mean(difference > 0)),
                "ci_method": BOOTSTRAP_CI_METHOD,
                "ci_level": BOOTSTRAP_CI_LEVEL,
                "n_bootstrap": int(len(difference)),
            })
    return records
