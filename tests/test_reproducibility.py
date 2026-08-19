"""Two runs with identical inputs must produce identical scientific artefacts."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone

from src.models import build_pipeline, fit_calibrated, predict_positive_proba
from src.preprocessing import preprocessing_report
from src.search import cv_results_records, run_search
from src.seeds import (
    REPRODUCIBILITY_ATOL,
    REPRODUCIBILITY_RTOL,
    SEEDS,
    seed_everything,
)
from src.splits import make_fixed_split, make_training_folds
from src.thresholds import emp_cutoff_from_oof, oof_raw_scores


VOLATILE_FIELDS = {
    "mean_fit_time_seconds", "mean_score_time_seconds", "elapsed_seconds",
    "search_seconds", "refit_seconds", "total_seconds", "written_utc",
    "created_utc", "timestamp_utc",
}


@pytest.fixture
def setup(synthetic_config, synthetic_frame):
    features = synthetic_config.feature_columns(synthetic_frame.columns)
    y = synthetic_frame["target"].to_numpy(dtype=int)
    train_index, test_index = make_fixed_split(y)
    return {
        "config": synthetic_config,
        "X_train": synthetic_frame[features].iloc[train_index].reset_index(drop=True),
        "y_train": y[train_index],
        "X_test": synthetic_frame[features].iloc[test_index].reset_index(drop=True),
        "y": y,
    }


def _strip_volatile(records):
    return [
        {key: value for key, value in record.items() if key not in VOLATILE_FIELDS}
        for record in records
    ]


def test_the_split_is_bitwise_reproducible(setup):
    first = make_fixed_split(setup["y"])
    second = make_fixed_split(setup["y"])
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_every_fold_family_is_bitwise_reproducible(setup):
    for key in ("hyperparameter_cv", "calibration_cv", "threshold_oof_cv"):
        first = make_training_folds(setup["y_train"], n_splits=3, random_state=SEEDS[key])
        second = make_training_folds(setup["y_train"], n_splits=3, random_state=SEEDS[key])
        for (fit_a, val_a), (fit_b, val_b) in zip(first, second):
            assert np.array_equal(fit_a, fit_b)
            assert np.array_equal(val_a, val_b)


def test_sampled_candidates_and_selection_are_reproducible(setup):
    folds = make_training_folds(
        setup["y_train"], n_splits=3, random_state=SEEDS["hyperparameter_cv"]
    )
    outcomes = []
    for _ in range(2):
        seed_everything()
        pipeline = build_pipeline(
            "XGBoost", setup["config"], use_gpu=False, ebm_bags=2
        )
        search, selection = run_search(
            "XGBoost", pipeline, setup["X_train"], setup["y_train"], folds,
            budgets={"LR": 6, "EBM": 2, "XGBoost": 3, "CatBoost": 2},
        )
        outcomes.append((
            _strip_volatile(cv_results_records(search.cv_results_, len(folds))),
            selection["best_params"],
        ))

    assert outcomes[0][0] == outcomes[1][0]
    assert outcomes[0][1] == outcomes[1][1]


def test_fitted_preprocessing_parameters_are_reproducible(setup):
    reports = []
    for _ in range(2):
        seed_everything()
        pipeline = build_pipeline("LR", setup["config"], use_gpu=False, ebm_bags=2)
        pipeline.fit(setup["X_train"], setup["y_train"])
        reports.append(preprocessing_report(pipeline.named_steps["prep"]))

    for key in ("missingness_filter", "winsorization", "imputation_median",
                "imputation_most_frequent", "category_schema", "n_encoded_features"):
        assert reports[0][key] == reports[1][key], key


def test_predictions_and_thresholds_are_reproducible(setup):
    folds = make_training_folds(
        setup["y_train"], n_splits=3, random_state=SEEDS["threshold_oof_cv"]
    )
    prior = float(setup["y_train"].mean())
    results = []
    for _ in range(2):
        seed_everything()
        template = build_pipeline(
            "LR", setup["config"], use_gpu=False, ebm_bags=2
        ).set_params(clf__lr__C=1.0)
        fitted = clone(template).fit(setup["X_train"], setup["y_train"])
        calibrated = fit_calibrated(
            clone(template), setup["X_train"], setup["y_train"], cv=folds
        )
        oof = oof_raw_scores(
            clone(template), setup["X_train"], setup["y_train"], folds
        )
        cutoff, eta = emp_cutoff_from_oof(
            setup["y_train"], oof, 0.2644, prior_default=prior
        )
        results.append({
            "p_raw": predict_positive_proba(fitted, setup["X_test"]),
            "p_calibrated": predict_positive_proba(calibrated, setup["X_test"]),
            "cutoff": cutoff,
            "eta": eta,
        })

    for key in ("p_raw", "p_calibrated"):
        assert np.allclose(
            results[0][key], results[1][key],
            rtol=REPRODUCIBILITY_RTOL, atol=REPRODUCIBILITY_ATOL,
        ), key
    assert results[0]["cutoff"] == pytest.approx(
        results[1]["cutoff"], rel=REPRODUCIBILITY_RTOL, abs=REPRODUCIBILITY_ATOL
    )
    assert results[0]["eta"] == pytest.approx(
        results[1]["eta"], rel=REPRODUCIBILITY_RTOL, abs=REPRODUCIBILITY_ATOL
    )


def test_bootstrap_indices_are_reproducible(setup):
    from src.bootstrap import paired_stratified_indices

    y_test = setup["y"][make_fixed_split(setup["y"])[1]]
    first = paired_stratified_indices(
        y_test, n_bootstrap=25, random_state=SEEDS["bootstrap"]
    )
    second = paired_stratified_indices(
        y_test, n_bootstrap=25, random_state=SEEDS["bootstrap"]
    )
    assert np.array_equal(first, second)


def test_the_tolerances_are_pinned_rather_than_ad_hoc():
    assert REPRODUCIBILITY_RTOL > 0
    assert REPRODUCIBILITY_ATOL > 0
