"""Hyperparameter search: training-only, shared folds, deterministic selection."""

from __future__ import annotations

import numpy as np
import pytest

from src.models import MODEL_NAMES, build_pipeline
from src.search import (
    DEFAULT_BUDGETS,
    PARAM_GRIDS,
    cv_results_records,
    run_search,
    select_best,
)
from src.seeds import SEARCH_TIE_EPSILON, SEEDS
from src.splits import make_fixed_split, make_training_folds


@pytest.fixture
def training_setup(synthetic_config, synthetic_frame):
    features = synthetic_config.feature_columns(synthetic_frame.columns)
    y = synthetic_frame["target"].to_numpy(dtype=int)
    train_index, test_index = make_fixed_split(y)
    folds = make_training_folds(
        y[train_index], n_splits=5, random_state=SEEDS["hyperparameter_cv"]
    )
    return {
        "config": synthetic_config,
        "X_train": synthetic_frame[features].iloc[train_index].reset_index(drop=True),
        "y_train": y[train_index],
        "train_index": train_index,
        "test_index": test_index,
        "folds": folds,
    }


def test_exactly_five_training_folds_are_used(training_setup):
    folds = training_setup["folds"]
    assert len(folds) == 5
    n_train = len(training_setup["y_train"])
    covered = np.concatenate([validation for _, validation in folds])
    assert np.array_equal(np.sort(covered), np.arange(n_train))


def test_the_search_never_sees_a_test_row(training_setup):
    """Fold indices are local to the training partition, so they cannot address it."""
    n_train = len(training_setup["y_train"])
    for fit_index, validation_index in training_setup["folds"]:
        assert fit_index.max() < n_train
        assert validation_index.max() < n_train


def test_all_model_families_reuse_the_identical_folds(training_setup):
    seen = []
    for name in MODEL_NAMES:
        folds = make_training_folds(
            training_setup["y_train"], n_splits=5, random_state=SEEDS["hyperparameter_cv"]
        )
        seen.append([(fit.tolist(), val.tolist()) for fit, val in folds])
    assert all(entry == seen[0] for entry in seen)


def test_grid_search_enumerates_the_logistic_regression_space(training_setup):
    pipeline = build_pipeline(
        "LR", training_setup["config"], use_gpu=False, ebm_bags=2
    )
    search, selection = run_search(
        "LR", pipeline, training_setup["X_train"], training_setup["y_train"],
        training_setup["folds"],
    )
    assert selection["search"]["class"] == "GridSearchCV"
    assert len(search.cv_results_["params"]) == len(PARAM_GRIDS["LR"]["clf__lr__C"])
    assert selection["n_folds"] == 5


def test_randomized_search_honours_the_budget(training_setup):
    pipeline = build_pipeline(
        "EBM", training_setup["config"], use_gpu=False, ebm_bags=2
    )
    budgets = {**DEFAULT_BUDGETS, "EBM": 3}
    search, selection = run_search(
        "EBM", pipeline, training_setup["X_train"], training_setup["y_train"],
        training_setup["folds"], budgets=budgets,
    )
    assert selection["search"]["class"] == "RandomizedSearchCV"
    assert selection["search"]["search_seed"] == SEEDS["randomized_search"]
    assert len(search.cv_results_["params"]) == 3


def test_es_search_honours_the_budget_and_returns_rounds(training_setup):
    pipeline = build_pipeline(
        "XGBoost", training_setup["config"], use_gpu=False, ebm_bags=2
    )
    budgets = {**DEFAULT_BUDGETS, "XGBoost": 3}
    search, selection = run_search(
        "XGBoost", pipeline, training_setup["X_train"], training_setup["y_train"],
        training_setup["folds"], budgets=budgets,
    )
    assert selection["search"]["class"] == "ESRandomizedSearch"
    assert selection["search"]["search_seed"] == SEEDS["randomized_search"]
    assert len(search.cv_results_["params"]) == 3
    rounds = selection["best_params"]["clf__estimator__n_estimators"]
    stopping = selection["early_stopping"]
    assert rounds == stopping["refit_rounds"] >= 1
    assert len(stopping["per_fold_best_iteration"]) == 5
    assert rounds == int(np.median(stopping["per_fold_best_iteration"])) + 1
    records = cv_results_records(search.cv_results_, 5)
    assert len(records) == 3
    for record in records:
        assert {f"fold{index}_validation_auc" for index in range(1, 6)} <= set(record)


def test_cv_results_records_carry_per_fold_scores(training_setup):
    pipeline = build_pipeline(
        "LR", training_setup["config"], use_gpu=False, ebm_bags=2
    )
    search, _ = run_search(
        "LR", pipeline, training_setup["X_train"], training_setup["y_train"],
        training_setup["folds"],
    )
    records = cv_results_records(search.cv_results_, 5)
    assert len(records) == 6
    for record in records:
        assert {f"fold{index}_validation_auc" for index in range(1, 6)} <= set(record)
        assert "mean_validation_auc" in record and "std_validation_auc" in record


def test_selection_prefers_the_highest_mean_validation_auc():
    cv_results = {
        "mean_test_score": np.asarray([0.70, 0.80, 0.75]),
        "std_test_score": np.asarray([0.01, 0.02, 0.01]),
        "params": [
            {"clf__lr__C": 0.1}, {"clf__lr__C": 1.0}, {"clf__lr__C": 10.0},
        ],
    }
    selection = select_best("LR", cv_results)
    assert selection["best_params"] == {"clf__lr__C": 1.0}
    assert selection["tie_break"]["resolved_by"] == "mean"


def test_selection_breaks_a_mean_tie_on_the_standard_deviation():
    cv_results = {
        "mean_test_score": np.asarray([0.80, 0.80]),
        "std_test_score": np.asarray([0.05, 0.01]),
        "params": [{"clf__lr__C": 0.1}, {"clf__lr__C": 10.0}],
    }
    selection = select_best("LR", cv_results)
    assert selection["best_params"] == {"clf__lr__C": 10.0}
    assert selection["tie_break"]["resolved_by"] == "std"


def test_selection_falls_back_to_the_simpler_configuration():
    cv_results = {
        "mean_test_score": np.asarray([0.80, 0.80]),
        "std_test_score": np.asarray([0.01, 0.01]),
        "params": [{"clf__lr__C": 100.0}, {"clf__lr__C": 0.01}],
    }
    selection = select_best("LR", cv_results)
    assert selection["best_params"] == {"clf__lr__C": 0.01}
    assert selection["tie_break"]["resolved_by"] == "complexity"


def test_selection_is_deterministic_under_a_full_tie():
    cv_results = {
        "mean_test_score": np.asarray([0.80, 0.80]),
        "std_test_score": np.asarray([0.01, 0.01]),
        "params": [{"unknown_key": "b"}, {"unknown_key": "a"}],
    }
    first = select_best("Unregistered", cv_results)
    second = select_best("Unregistered", dict(cv_results))
    assert first["best_params"] == second["best_params"] == {"unknown_key": "a"}
    assert first["tie_break"]["resolved_by"] == "serialised_order"


def test_differences_below_the_epsilon_count_as_a_tie():
    cv_results = {
        "mean_test_score": np.asarray([0.80, 0.80 - SEARCH_TIE_EPSILON / 2]),
        "std_test_score": np.asarray([0.05, 0.01]),
        "params": [{"clf__lr__C": 100.0}, {"clf__lr__C": 0.01}],
    }
    selection = select_best("LR", cv_results)
    assert selection["tie_break"]["n_tied_on_mean"] == 2
    assert selection["best_params"] == {"clf__lr__C": 0.01}


def test_boosting_rounds_are_not_searched():
    """Rounds come from early stopping on the validation fold, not the grid."""
    assert "clf__estimator__n_estimators" not in PARAM_GRIDS["XGBoost"]
    assert "clf__estimator__iterations" not in PARAM_GRIDS["CatBoost"]
