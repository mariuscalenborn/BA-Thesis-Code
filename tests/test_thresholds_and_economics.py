"""Thresholds and the cost model come from training data and are frozen."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from src.evaluation import EMP_P0, EMP_P1, emp_credit_scoring, metrics_suite
from src.models import build_pipeline
from src.seeds import SEEDS
from src.splits import make_fixed_split, make_training_folds
from src.thresholds import (
    base_rate_threshold,
    build_economic_config,
    emp_cutoff_from_oof,
    load_economic_config,
    oof_raw_scores,
    write_economic_config,
)


@pytest.fixture
def trained(synthetic_config, synthetic_frame):
    features = synthetic_config.feature_columns(synthetic_frame.columns)
    y = synthetic_frame["target"].to_numpy(dtype=int)
    train_index, test_index = make_fixed_split(y)
    X_train = synthetic_frame[features].iloc[train_index].reset_index(drop=True)
    y_train = y[train_index]
    folds = make_training_folds(
        y_train, n_splits=5, random_state=SEEDS["threshold_oof_cv"]
    )
    pipeline = build_pipeline(
        "LR", synthetic_config, use_gpu=False, ebm_bags=2
    ).set_params(clf__lr__C=1.0)
    return {
        "X_train": X_train, "y_train": y_train, "folds": folds,
        "pipeline": pipeline,
        "X_test": synthetic_frame[features].iloc[test_index].reset_index(drop=True),
        "y_test": y[test_index],
    }


def test_every_training_row_gets_exactly_one_oof_prediction(trained):
    oof = oof_raw_scores(
        clone(trained["pipeline"]), trained["X_train"], trained["y_train"],
        trained["folds"],
    )
    assert len(oof) == len(trained["y_train"])
    assert not np.isnan(oof).any()


def test_incomplete_fold_coverage_is_refused(trained):
    truncated = [
        (fit_index, validation_index[:-1])
        for fit_index, validation_index in trained["folds"]
    ]
    with pytest.raises(ValueError, match="not exactly one prediction"):
        oof_raw_scores(
            clone(trained["pipeline"]), trained["X_train"], trained["y_train"], truncated
        )


def test_base_rate_threshold_comes_from_the_training_partition(trained):
    assert base_rate_threshold(trained["y_train"]) == pytest.approx(
        float(trained["y_train"].mean())
    )


def test_test_labels_cannot_move_either_threshold(trained):
    oof = oof_raw_scores(
        clone(trained["pipeline"]), trained["X_train"], trained["y_train"],
        trained["folds"],
    )
    prior = float(trained["y_train"].mean())
    cutoff, eta = emp_cutoff_from_oof(
        trained["y_train"], oof, 0.2644, prior_default=prior
    )
    fixed = base_rate_threshold(trained["y_train"])

    flipped = 1 - trained["y_test"]
    assert flipped.sum() != trained["y_test"].sum()
    cutoff_again, eta_again = emp_cutoff_from_oof(
        trained["y_train"], oof, 0.2644, prior_default=prior
    )
    assert cutoff_again == cutoff and eta_again == eta
    assert base_rate_threshold(trained["y_train"]) == fixed


def test_the_cutoff_is_cross_fitted_not_in_sample(trained):
    """Cross-fitting changes the cut-off; it is not a refactor."""
    pipeline = clone(trained["pipeline"]).fit(trained["X_train"], trained["y_train"])
    from src.models import predict_positive_proba

    in_sample = predict_positive_proba(pipeline, trained["X_train"])
    oof = oof_raw_scores(
        clone(trained["pipeline"]), trained["X_train"], trained["y_train"],
        trained["folds"],
    )
    prior = float(trained["y_train"].mean())
    _, eta_in_sample = emp_credit_scoring(
        trained["y_train"], in_sample, 0.2644, prior_default=prior
    )
    cutoff_in_sample = float(np.quantile(in_sample, 1.0 - eta_in_sample))
    cutoff_oof, _ = emp_cutoff_from_oof(
        trained["y_train"], oof, 0.2644, prior_default=prior
    )
    assert cutoff_in_sample != cutoff_oof


def test_emp_requires_an_explicit_prior(trained):
    with pytest.raises(ValueError, match="explicit frozen prior"):
        emp_credit_scoring(trained["y_test"], np.linspace(0, 1, len(trained["y_test"])),
                           0.2644, prior_default=None)


def test_emp_uses_the_frozen_prior_not_the_evaluated_sample():
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=400) < 0.3).astype(int)
    scores = rng.uniform(size=400)
    train_prior = 0.30
    with_train_prior = emp_credit_scoring(y, scores, 0.2644, prior_default=train_prior)
    with_other_prior = emp_credit_scoring(y, scores, 0.2644, prior_default=0.10)
    assert with_train_prior != with_other_prior


def test_metrics_suite_reports_the_test_default_rate_only_descriptively(trained):
    rng = np.random.default_rng(1)
    p_raw = rng.uniform(size=len(trained["y_test"]))
    p_calibrated = rng.uniform(size=len(trained["y_test"]))
    metrics = metrics_suite(
        trained["y_test"], p_raw, p_calibrated,
        emp_cutoff=0.5, base_rate_threshold=0.3, roi=0.2644, prior_default=0.30,
    )
    assert metrics["prior_default_frozen"] == 0.30
    assert metrics["test_default_rate"] == pytest.approx(trained["y_test"].mean())


def test_economic_config_records_everything_the_metric_consumes(tmp_path):
    config = build_economic_config(
        dataset="synthetic",
        roi=0.2644,
        prior_default=0.30,
        thresholds_by_model={"LR": {"emp_cutoff": 0.6, "eta_oof": 0.2,
                                    "base_rate_threshold": 0.3}},
        roi_provenance="Verbraken baseline",
    )
    path = tmp_path / "economic_config.json"
    write_economic_config(path, config)
    loaded = load_economic_config(path)

    assert loaded["prior_default_train"] == 0.30
    assert loaded["loss_given_default_mixture"]["p0"] == EMP_P0
    assert loaded["loss_given_default_mixture"]["p1"] == EMP_P1
    assert loaded["emp_implementation_version"]
    assert "must not replace prior_default_train" in loaded["note"]


def test_a_missing_economic_config_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="frozen before"):
        load_economic_config(tmp_path / "absent.json")


def test_calibration_folds_are_three_and_stratified(trained):
    folds = list(
        StratifiedKFold(n_splits=3, shuffle=True, random_state=SEEDS["calibration_cv"])
        .split(np.zeros(len(trained["y_train"]), dtype=np.uint8), trained["y_train"])
    )
    assert len(folds) == 3
    for _, validation_index in folds:
        share = trained["y_train"][validation_index].mean()
        assert abs(share - trained["y_train"].mean()) < 0.05
