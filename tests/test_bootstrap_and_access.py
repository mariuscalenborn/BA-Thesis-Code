"""Paired bootstrap and test-partition governance."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.bootstrap import (
    BOOTSTRAP_METRICS,
    bootstrap_model_metrics,
    bootstrap_pairwise_deltas,
    paired_stratified_indices,
)
from src.seeds import SEEDS
from src.test_access import (
    ANALYSIS_PLAN_NAME,
    PERMITTED_ANALYSES,
    GuardedTestPartition,
    ProtocolAccessError,
    assert_no_fit,
    write_analysis_plan,
)


@pytest.fixture
def test_scores():
    rng = np.random.default_rng(7)
    n = 300
    y = (rng.uniform(size=n) < 0.25).astype(int)
    signal = y * 0.6 + rng.uniform(size=n) * 0.6
    return {
        "y": y,
        "scores": {
            "LR": (np.clip(signal, 0, 1), np.clip(signal * 0.9, 0, 1)),
            "XGBoost": (np.clip(signal + 0.05 * rng.normal(size=n), 0, 1),
                        np.clip(signal * 0.92, 0, 1)),
        },
        "thresholds": {
            "LR": {"emp_cutoff": 0.6, "base_rate_threshold": 0.25},
            "XGBoost": {"emp_cutoff": 0.62, "base_rate_threshold": 0.25},
        },
    }


def test_bootstrap_preserves_the_class_counts(test_scores):
    y = test_scores["y"]
    indices = paired_stratified_indices(y, n_bootstrap=50, random_state=SEEDS["bootstrap"])
    assert indices.shape == (50, len(y))
    for row in indices:
        assert (y[row] == 1).sum() == (y == 1).sum()
        assert (y[row] == 0).sum() == (y == 0).sum()


def test_bootstrap_is_seeded(test_scores):
    first = paired_stratified_indices(test_scores["y"], n_bootstrap=20, random_state=4220)
    second = paired_stratified_indices(test_scores["y"], n_bootstrap=20, random_state=4220)
    assert np.array_equal(first, second)


def test_the_same_indices_serve_every_model(test_scores):
    """Pairing is what makes the difference interval a statement about the difference."""
    indices = paired_stratified_indices(
        test_scores["y"], n_bootstrap=30, random_state=SEEDS["bootstrap"]
    )
    summary, draws = bootstrap_model_metrics(
        test_scores["y"], test_scores["scores"], test_scores["thresholds"],
        roi=0.2644, prior_default=0.25, indices=indices,
    )
    assert set(draws) == set(test_scores["scores"])
    for name in draws:
        for metric in BOOTSTRAP_METRICS:
            assert len(draws[name][metric]) == 30

    deltas = bootstrap_pairwise_deltas(draws, summary)
    auc_delta = next(row for row in deltas if row["metric"] == "auc")
    assert auc_delta["delta_ci_low"] <= auc_delta["delta_point"] <= auc_delta["delta_ci_high"]


def test_thresholds_stay_fixed_across_replicates(test_scores):
    """A re-optimised threshold inside a replicate would inflate every interval."""
    indices = paired_stratified_indices(
        test_scores["y"], n_bootstrap=10, random_state=SEEDS["bootstrap"]
    )
    thresholds = {
        name: dict(values) for name, values in test_scores["thresholds"].items()
    }
    bootstrap_model_metrics(
        test_scores["y"], test_scores["scores"], thresholds,
        roi=0.2644, prior_default=0.25, indices=indices,
    )
    assert thresholds == test_scores["thresholds"]


def test_confidence_intervals_are_ordered(test_scores):
    indices = paired_stratified_indices(
        test_scores["y"], n_bootstrap=40, random_state=SEEDS["bootstrap"]
    )
    summary, _ = bootstrap_model_metrics(
        test_scores["y"], test_scores["scores"], test_scores["thresholds"],
        roi=0.2644, prior_default=0.25, indices=indices,
    )
    for row in summary:
        assert row["ci_low"] <= row["ci_high"]
        assert row["ci_method"] == "percentile"
        assert row["n_bootstrap"] == 40


def test_bootstrap_needs_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        paired_stratified_indices(np.zeros(10, dtype=int), n_bootstrap=5, random_state=1)


def test_no_delong_module_exists():
    """Uncertainty is quantified through one procedure, not two."""
    with pytest.raises(ModuleNotFoundError):
        __import__("src.delong")


def _access(tmp_path):
    write_analysis_plan(
        tmp_path / ANALYSIS_PLAN_NAME,
        datasets=["synthetic"],
        models=["LR"],
    )
    return GuardedTestPartition(tmp_path, repo_root=tmp_path, script="test.py")


def test_analysis_plan_lists_every_permitted_analysis(tmp_path):
    plan = write_analysis_plan(
        tmp_path / ANALYSIS_PLAN_NAME, datasets=["synthetic"], models=["LR"]
    )
    assert set(plan["permitted_analyses"]) == set(PERMITTED_ANALYSES)
    assert plan["primary_analysis"] == "primary_model_comparison"


def test_an_unplanned_analysis_is_refused(tmp_path, synthetic_frame):
    access = _access(tmp_path)
    with pytest.raises(ProtocolAccessError, match="not in the frozen analysis plan"):
        access.open(
            synthetic_frame, synthetic_frame["target"].to_numpy(), np.arange(5),
            analysis="peeking", dataset="synthetic",
        )


def test_an_unplanned_dataset_is_refused(tmp_path, synthetic_frame):
    access = _access(tmp_path)
    with pytest.raises(ProtocolAccessError, match="not covered"):
        access.open(
            synthetic_frame, synthetic_frame["target"].to_numpy(), np.arange(5),
            analysis="primary_model_comparison", dataset="other",
        )


def test_access_without_a_frozen_cost_model_is_refused(tmp_path, synthetic_frame):
    access = _access(tmp_path)
    with pytest.raises(ProtocolAccessError, match="economic configuration must"):
        access.open(
            synthetic_frame, synthetic_frame["target"].to_numpy(), np.arange(5),
            analysis="primary_model_comparison", dataset="synthetic",
            economic_config_path=tmp_path / "absent.json",
        )


def test_every_access_is_logged(tmp_path, synthetic_frame):
    access = _access(tmp_path)
    y = synthetic_frame["target"].to_numpy()
    access.open(synthetic_frame, y, np.arange(5),
                analysis="primary_model_comparison", dataset="synthetic", model="LR")
    access.open(synthetic_frame, y, np.arange(5),
                analysis="paired_bootstrap", dataset="synthetic", model="LR")

    lines = (tmp_path / "test_access_log.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert [record["analysis"] for record in records] == [
        "primary_model_comparison", "paired_bootstrap"
    ]
    assert all(record["fit_occurred"] is False for record in records)
    assert all(record["n_test_rows"] == 5 for record in records)


def test_a_missing_analysis_plan_is_an_error(tmp_path):
    with pytest.raises(ProtocolAccessError, match="frozen before"):
        GuardedTestPartition(tmp_path, repo_root=tmp_path, script="test.py")


def test_final_evaluation_refuses_an_unfitted_estimator(synthetic_config):
    from src.models import build_pipeline

    with pytest.raises(ProtocolAccessError, match="must never fit"):
        assert_no_fit(build_pipeline("LR", synthetic_config, use_gpu=False, ebm_bags=2))
