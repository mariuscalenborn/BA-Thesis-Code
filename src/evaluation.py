"""Metrics, economic evaluation and explanation analysis."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import shap
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.models import BalancedEBMClassifier, FitLocalPosWeightClassifier


EMP_P0 = 0.55
EMP_P1 = 0.10
EMP_IMPLEMENTATION_VERSION = "verbraken2014-hull-v4.1"


def _roc_hull(y, scores):
    fpr, tpr, _ = roc_curve(y, scores, pos_label=1)
    hull = []
    for x, y_value in zip(fpr.tolist(), tpr.tolist()):
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            cross = (x2 - x1) * (y_value - y2) - (y2 - y1) * (x - x2)
            if cross >= 0:
                hull.pop()
            else:
                break
        hull.append((x, y_value))
    return np.asarray([p[1] for p in hull]), np.asarray([p[0] for p in hull])


def emp_credit_scoring(y, scores, roi, *, prior_default):
    """Expected Maximum Profit and the optimal rejected fraction."""
    if prior_default is None:
        raise ValueError(
            "emp_credit_scoring requires an explicit frozen prior; see economic_config.json"
        )
    prior_default = float(prior_default)
    if not 0.0 < prior_default < 1.0:
        raise ValueError(f"prior_default must lie in (0, 1), got {prior_default}")

    f0, f1 = _roc_hull(np.asarray(y), np.asarray(scores, dtype=float))
    pi0 = prior_default
    pi1 = 1.0 - pi0
    density = 1.0 - EMP_P0 - EMP_P1
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (pi1 * roi / pi0) * np.diff(f1) / np.diff(f0)
    lam = np.concatenate(([0.0], slope))
    lam = np.concatenate((lam[lam < 1.0], [1.0]))
    lower, upper = lam[:-1], lam[1:]
    n_segments = len(lower)
    f0, f1 = f0[:n_segments], f1[:n_segments]
    emp = (
        np.sum(density * (upper - lower) * (
            pi0 * f0 * (upper + lower) / 2.0 - roi * f1 * pi1
        ))
        + (pi0 * f0[-1] - roi * pi1 * f1[-1]) * EMP_P1
    )
    eta = (
        np.sum(density * (upper - lower) * (pi0 * f0 + pi1 * f1))
        + EMP_P1 * (pi0 * f0[-1] + pi1 * f1[-1])
    )
    return float(emp), float(eta)


def metrics_suite(
    y_test,
    p_raw,
    p_calibrated,
    *,
    emp_cutoff,
    base_rate_threshold,
    roi,
    prior_default,
):
    """The full metric row for one model on the test partition."""
    emp, eta = emp_credit_scoring(y_test, p_raw, roi, prior_default=prior_default)
    yhat_emp = (p_raw > emp_cutoff).astype(int)
    yhat_fixed = (p_calibrated > base_rate_threshold).astype(int)
    return {
        "auc": roc_auc_score(y_test, p_raw),
        "pr_auc": average_precision_score(y_test, p_raw),
        "brier": brier_score_loss(y_test, p_calibrated),
        "emp": emp,
        "eta": eta,
        "precision": precision_score(y_test, yhat_emp, zero_division=0),
        "recall": recall_score(y_test, yhat_emp, zero_division=0),
        "f1": f1_score(y_test, yhat_emp, zero_division=0),
        "precision_fix": precision_score(y_test, yhat_fixed, zero_division=0),
        "recall_fix": recall_score(y_test, yhat_fixed, zero_division=0),
        "f1_fix": f1_score(y_test, yhat_fixed, zero_division=0),
        "balacc_fix": balanced_accuracy_score(y_test, yhat_fixed),
        "test_default_rate": float(np.mean(np.asarray(y_test) == 1)),
        "prior_default_frozen": float(prior_default),
    }


METRIC_KEYS = (
    "auc", "pr_auc", "brier", "emp", "eta",
    "precision", "recall", "f1",
    "precision_fix", "recall_fix", "f1_fix", "balacc_fix",
)


def compute_attributions(
    name,
    estimator,
    X_background,
    X_explain,
    feature_names,
    *,
    background_size=500,
    random_state,
):
    """Per-case attributions from the uncalibrated final model."""
    X_background = np.asarray(X_background, dtype=float)
    X_explain = np.asarray(X_explain, dtype=float)
    if isinstance(estimator, (BalancedEBMClassifier, FitLocalPosWeightClassifier)):
        estimator = estimator.estimator_

    if name == "LR":
        scaler = estimator.named_steps["scaler"]
        classifier = estimator.named_steps["lr"]
        rng = np.random.default_rng(random_state)
        background_idx = rng.choice(
            len(X_background), min(background_size, len(X_background)), replace=False
        )
        background = scaler.transform(X_background[background_idx])
        explainer = shap.LinearExplainer(
            classifier, background, feature_perturbation="interventional"
        )
        values = explainer.shap_values(scaler.transform(X_explain))
    elif name == "EBM":
        local = estimator.explain_local(X_explain)
        scores = np.asarray([local.data(i)["scores"] for i in range(len(X_explain))])
        values = np.zeros((len(X_explain), len(feature_names)), dtype=float)
        for term_index, features in enumerate(estimator.term_features_):
            for feature_index in features:
                values[:, feature_index] += scores[:, term_index] / len(features)
    else:
        explainer = shap.TreeExplainer(
            estimator, feature_perturbation="tree_path_dependent"
        )
        values = explainer.shap_values(X_explain)
        values = values[1] if isinstance(values, list) else values
        values = values.values if hasattr(values, "values") else values
        values = np.asarray(values)
        if values.ndim == 3 and values.shape[-1] == 2:
            values = values[:, :, 1]
        if values.shape[1] == len(feature_names) + 1:
            values = values[:, :-1]
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError(f"{name}: unexpected attribution shape {values.shape}")
    return values


def aggregate_to_source_features(values, feature_names, source_map):
    """Sum encoded-column attributions back onto their source feature."""
    values = np.asarray(values, dtype=float)
    sources = [source_map.get(str(name), str(name)) for name in feature_names]
    ordered = sorted(set(sources))
    position = {source: index for index, source in enumerate(ordered)}
    aggregated = np.zeros((values.shape[0], len(ordered)), dtype=float)
    for column, source in enumerate(sources):
        aggregated[:, position[source]] += values[:, column]
    return aggregated, ordered


def mean_abs_importance(abs_attributions):
    return np.asarray(abs_attributions).mean(axis=0)


def topk_jaccard(importances, k=10):
    """Mean pairwise Jaccard overlap of the top-k sets of several rankings."""
    sets = [set(np.argsort(-np.asarray(importance))[:k]) for importance in importances]
    values = [len(a & b) / len(a | b) for a, b in combinations(sets, 2)]
    return float(np.mean(values)) if values else np.nan


def kendall_w(importances, top=20):
    """Concordance of several rankings over the union of their top-``top`` items."""
    ranks = np.asarray([rankdata(-np.asarray(importance)) for importance in importances])
    keep = sorted(set().union(*[
        set(np.argsort(rank_vector)[:top]) for rank_vector in ranks
    ]))
    restricted = np.apply_along_axis(rankdata, 1, ranks[:, keep])
    n_raters, n_items = restricted.shape
    rank_sums = restricted.sum(axis=0)
    numerator = ((rank_sums - rank_sums.mean()) ** 2).sum()
    denominator = n_raters ** 2 * (n_items ** 3 - n_items)
    return float(12.0 * numerator / denominator) if denominator > 0 else np.nan


def case_bootstrap_stability(
    attributions_by_model,
    source_features,
    *,
    n_bootstrap=1000,
    random_state,
    top_k=10,
    top_n=20,
):
    """Explanation stability under case resampling, paired across models."""
    names = list(attributions_by_model)
    matrices = [np.asarray(attributions_by_model[name], dtype=np.float32) for name in names]
    n_cases = matrices[0].shape[0]
    if any(matrix.shape[0] != n_cases for matrix in matrices):
        raise ValueError("All models must explain the same cases for a paired bootstrap")

    point = {name: mean_abs_importance(matrix) for name, matrix in zip(names, matrices)}
    point_jaccard = topk_jaccard(list(point.values()), k=top_k)
    point_kendall = kendall_w(list(point.values()), top=top_n)

    rng = np.random.default_rng(random_state)
    n_features = len(source_features)
    membership = {name: np.zeros(n_features, dtype=np.int64) for name in names}
    rank_draws = {name: np.empty((n_bootstrap, n_features), dtype=np.float32) for name in names}
    jaccard_draws = np.empty(n_bootstrap, dtype=float)
    kendall_draws = np.empty(n_bootstrap, dtype=float)

    for replicate in range(n_bootstrap):
        case_index = rng.integers(0, n_cases, size=n_cases)
        importances = []
        for name, matrix in zip(names, matrices):
            importance = matrix[case_index].mean(axis=0)
            importances.append(importance)
            membership[name][np.argsort(-importance)[:top_k]] += 1
            rank_draws[name][replicate] = rankdata(-importance)
        jaccard_draws[replicate] = topk_jaccard(importances, k=top_k)
        kendall_draws[replicate] = kendall_w(importances, top=top_n)

    per_model = []
    for name in names:
        importance = point[name]
        order = np.argsort(-importance)
        for rank_position, feature_index in enumerate(order, start=1):
            ranks = rank_draws[name][:, feature_index]
            per_model.append({
                "model": name,
                "feature": source_features[feature_index],
                "mean_abs_attribution": float(importance[feature_index]),
                "point_rank": int(rank_position),
                "in_top10_frequency": float(membership[name][feature_index] / n_bootstrap),
                "rank_ci_low": float(np.quantile(ranks, 0.025)),
                "rank_median": float(np.median(ranks)),
                "rank_ci_high": float(np.quantile(ranks, 0.975)),
            })

    cross_model = []
    for left, right in combinations(names, 2):
        correlation = spearmanr(point[left], point[right]).statistic
        cross_model.append({
            "model_a": left,
            "model_b": right,
            "spearman": float(correlation),
            "jaccard_top10": topk_jaccard([point[left], point[right]], k=top_k),
        })

    summary = {
        "n_bootstrap": int(n_bootstrap),
        "n_cases": int(n_cases),
        "ci_level": 0.95,
        "uncertainty_scope": "case_selection_only",
        "scope_note": (
            "Paired resampling of the explained applicants with all fitted models "
            "held fixed. Retraining variability is not covered."
        ),
        "cross_model_jaccard_top10": point_jaccard,
        "cross_model_jaccard_top10_ci_low": float(np.quantile(jaccard_draws, 0.025)),
        "cross_model_jaccard_top10_ci_high": float(np.quantile(jaccard_draws, 0.975)),
        "cross_model_kendall_w_top20": point_kendall,
        "cross_model_kendall_w_top20_ci_low": float(np.quantile(kendall_draws, 0.025)),
        "cross_model_kendall_w_top20_ci_high": float(np.quantile(kendall_draws, 0.975)),
    }
    return summary, per_model, cross_model
