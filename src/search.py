"""Hyperparameter search on the training partition only."""

from __future__ import annotations

import json
from time import perf_counter

import numpy as np
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, ParameterSampler, RandomizedSearchCV

from src.seeds import ES_MAX_ROUNDS, ES_PATIENCE, SEEDS, SEARCH_TIE_EPSILON


PARAM_GRIDS = {
    "LR": {
        "clf__lr__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "EBM": {
        "clf__estimator__interactions": [10, 20],
        "clf__estimator__max_leaves": [2, 3, 5],
        "clf__estimator__max_bins": [256, 512],
        "clf__estimator__learning_rate": [0.005, 0.01, 0.02, 0.05],
        "clf__estimator__min_samples_leaf": [2, 5],
    },
    "XGBoost": {
        "clf__estimator__max_depth": [4, 6, 8],
        "clf__estimator__learning_rate": [0.03, 0.05, 0.1],
        "clf__estimator__subsample": [0.8, 1.0],
        "clf__estimator__colsample_bytree": [0.8, 1.0],
        "clf__estimator__min_child_weight": [1, 5],
        "clf__estimator__gamma": [0, 0.1],
    },
    "CatBoost": {
        "clf__estimator__depth": [4, 6, 8],
        "clf__estimator__learning_rate": [0.03, 0.05, 0.1],
        "clf__estimator__l2_leaf_reg": [1, 3, 5],
    },
}

DEFAULT_BUDGETS = {"LR": 6, "EBM": 20, "XGBoost": 30, "CatBoost": 20}

COMPLEXITY_KEYS = {
    "LR": ("clf__lr__C",),
    "EBM": (
        "clf__estimator__interactions",
        "clf__estimator__max_leaves",
        "clf__estimator__max_bins",
        "clf__estimator__learning_rate",
    ),
    "XGBoost": (
        "clf__estimator__max_depth",
        "clf__estimator__learning_rate",
    ),
    "CatBoost": (
        "clf__estimator__depth",
        "clf__estimator__learning_rate",
    ),
}


def _json_ready(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _canonical(params):
    return {key: _json_ready(value) for key, value in sorted(params.items())}


def select_best(name: str, cv_results: dict) -> dict:
    """Deterministic candidate selection with a recorded tie-break trace."""
    means = np.asarray(cv_results["mean_test_score"], dtype=float)
    stds = np.asarray(cv_results["std_test_score"], dtype=float)
    params = [_canonical(candidate) for candidate in cv_results["params"]]

    best_mean = float(np.nanmax(means))
    contenders = [
        index
        for index in range(len(means))
        if np.isfinite(means[index]) and best_mean - means[index] <= SEARCH_TIE_EPSILON
    ]
    trace = {
        "rule": "max mean roc_auc -> min std -> simpler configuration -> serialised order",
        "tie_epsilon": SEARCH_TIE_EPSILON,
        "n_candidates": int(len(means)),
        "n_tied_on_mean": int(len(contenders)),
        "resolved_by": "mean",
    }

    if len(contenders) > 1:
        best_std = float(np.nanmin(stds[contenders]))
        narrowed = [
            index
            for index in contenders
            if np.isfinite(stds[index]) and stds[index] - best_std <= SEARCH_TIE_EPSILON
        ]
        if narrowed and len(narrowed) < len(contenders):
            trace["resolved_by"] = "std"
        contenders = narrowed or contenders
        trace["n_tied_on_std"] = int(len(contenders))

    if len(contenders) > 1:
        keys = COMPLEXITY_KEYS.get(name, ())

        def complexity(index):
            return tuple(
                float(params[index][key])
                for key in keys
                if key in params[index] and params[index][key] is not None
            )

        simplest = min(complexity(index) for index in contenders)
        narrowed = [index for index in contenders if complexity(index) == simplest]
        if narrowed and len(narrowed) < len(contenders):
            trace["resolved_by"] = "complexity"
        contenders = narrowed or contenders

    if len(contenders) > 1:
        trace["resolved_by"] = "serialised_order"
        contenders = [
            min(contenders, key=lambda index: json.dumps(params[index], sort_keys=True))
        ]

    best_index = contenders[0]
    return {
        "model": name,
        "best_index": int(best_index),
        "best_params": params[best_index],
        "mean_validation_auc": float(means[best_index]),
        "std_validation_auc": float(stds[best_index]),
        "tie_break": trace,
    }


ES_MODELS = ("XGBoost", "CatBoost")
ES_ROUNDS_PARAM = {
    "XGBoost": "clf__estimator__n_estimators",
    "CatBoost": "clf__estimator__iterations",
}


class _ESSearchResult:
    """Minimal stand-in for a fitted sklearn searcher: carries ``cv_results_``."""

    def __init__(self, cv_results):
        self.cv_results_ = cv_results


def _take(X, index):
    return X.iloc[index] if hasattr(X, "iloc") else X[index]


def _run_es_search(name, pipeline, X, y, folds, budget):
    """Candidate loop for the boosted ensembles with fold-wise early stopping."""
    from src.models import predict_positive_proba

    y = np.asarray(y)
    sampled = list(ParameterSampler(
        PARAM_GRIDS[name], n_iter=budget, random_state=SEEDS["randomized_search"]
    ))
    n_folds = len(folds)
    cv_results = {
        "params": [dict(candidate) for candidate in sampled],
        "mean_fit_time": np.zeros(len(sampled)),
        "mean_score_time": np.zeros(len(sampled)),
    }
    for fold in range(n_folds):
        cv_results[f"split{fold}_test_score"] = np.zeros(len(sampled))
        cv_results[f"split{fold}_best_iteration"] = np.zeros(len(sampled), dtype=int)

    for index, params in enumerate(sampled):
        fit_times, score_times, scores = [], [], []
        for fold, (fit_index, validation_index) in enumerate(folds):
            candidate = clone(pipeline).set_params(**params)
            preprocessor = candidate.named_steps["prep"]
            classifier = candidate.named_steps["clf"]

            started = perf_counter()
            X_fit = preprocessor.fit_transform(_take(X, fit_index), y[fit_index])
            X_validation = preprocessor.transform(_take(X, validation_index))
            classifier.fit(
                X_fit, y[fit_index],
                eval_set=(X_validation, y[validation_index]),
            )
            fit_times.append(perf_counter() - started)

            started = perf_counter()
            scores.append(roc_auc_score(
                y[validation_index],
                predict_positive_proba(classifier, X_validation),
            ))
            score_times.append(perf_counter() - started)
            cv_results[f"split{fold}_test_score"][index] = scores[-1]
            cv_results[f"split{fold}_best_iteration"][index] = int(
                classifier.best_iteration_ if classifier.best_iteration_ is not None
                else ES_MAX_ROUNDS - 1
            )
        cv_results["mean_fit_time"][index] = float(np.mean(fit_times))
        cv_results["mean_score_time"][index] = float(np.mean(score_times))

    score_matrix = np.vstack([
        cv_results[f"split{fold}_test_score"] for fold in range(n_folds)
    ])
    cv_results["mean_test_score"] = score_matrix.mean(axis=0)
    cv_results["std_test_score"] = score_matrix.std(axis=0)
    cv_results["rank_test_score"] = rankdata(
        -cv_results["mean_test_score"], method="min"
    ).astype(int)
    return _ESSearchResult(cv_results)


def run_search(
    name: str,
    pipeline,
    X,
    y,
    cv_folds,
    *,
    budgets=None,
    n_jobs: int = 1,
):
    """Run the training-only search for one model family."""
    budgets = DEFAULT_BUDGETS if budgets is None else budgets
    grid = PARAM_GRIDS[name]
    folds = [
        (np.asarray(train_idx, dtype=int), np.asarray(validation_idx, dtype=int))
        for train_idx, validation_idx in cv_folds
    ]

    if name in ES_MODELS:
        search = _run_es_search(name, pipeline, X, y, folds, int(budgets[name]))
        selection = select_best(name, search.cv_results_)
        best_iterations = [
            int(search.cv_results_[f"split{fold}_best_iteration"][selection["best_index"]])
            for fold in range(len(folds))
        ]
        rounds = int(np.median(best_iterations)) + 1
        selection["best_params"] = {
            **selection["best_params"], ES_ROUNDS_PARAM[name]: rounds,
        }
        selection["early_stopping"] = {
            "patience": ES_PATIENCE,
            "round_cap": ES_MAX_ROUNDS,
            "per_fold_best_iteration": best_iterations,
            "refit_rounds": rounds,
            "rule": "median best iteration over the folds + 1",
        }
        selection["search"] = {
            "class": "ESRandomizedSearch",
            "n_candidates": int(budgets[name]),
            "search_seed": SEEDS["randomized_search"],
        }
        selection["n_folds"] = len(folds)
        return search, selection

    common = {
        "scoring": "roc_auc",
        "cv": folds,
        "refit": False,
        "error_score": "raise",
        "n_jobs": n_jobs,
        "return_train_score": False,
    }
    if name == "LR":
        search = GridSearchCV(pipeline, grid, **common)
        strategy = {"class": "GridSearchCV", "n_candidates": 6, "search_seed": None}
    else:
        search = RandomizedSearchCV(
            pipeline,
            grid,
            n_iter=budgets[name],
            random_state=SEEDS["randomized_search"],
            **common,
        )
        strategy = {
            "class": "RandomizedSearchCV",
            "n_candidates": int(budgets[name]),
            "search_seed": SEEDS["randomized_search"],
        }

    search.fit(X, y)
    selection = select_best(name, search.cv_results_)
    selection["search"] = strategy
    selection["n_folds"] = len(folds)
    return search, selection


def cv_results_records(cv_results: dict, n_folds: int) -> list[dict]:
    """Flatten ``cv_results_`` into one persistable record per candidate."""
    records = []
    for index, params in enumerate(cv_results["params"]):
        record = {
            "candidate_index": index,
            "params": _canonical(params),
            "mean_validation_auc": float(cv_results["mean_test_score"][index]),
            "std_validation_auc": float(cv_results["std_test_score"][index]),
            "sklearn_rank": int(cv_results["rank_test_score"][index]),
            "mean_fit_time_seconds": float(cv_results["mean_fit_time"][index]),
            "mean_score_time_seconds": float(cv_results["mean_score_time"][index]),
        }
        for fold in range(n_folds):
            key = f"split{fold}_test_score"
            if key in cv_results:
                record[f"fold{fold + 1}_validation_auc"] = float(cv_results[key][index])
        records.append(record)
    return records
