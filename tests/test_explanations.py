"""Explanations describe the uncalibrated model and compare source features."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation import (
    aggregate_to_source_features,
    case_bootstrap_stability,
    compute_attributions,
    kendall_w,
    mean_abs_importance,
    topk_jaccard,
)
from src.models import MODEL_NAMES, build_pipeline, fit_calibrated, inner_estimator
from src.preprocessing import encoded_feature_names, source_feature_map
from src.seeds import SEEDS
from src.splits import make_training_folds


@pytest.fixture
def fitted_pipelines(synthetic_config, synthetic_xy):
    X, y = synthetic_xy
    pipelines = {}
    for name in MODEL_NAMES:
        pipeline = build_pipeline(name, synthetic_config, use_gpu=False, ebm_bags=2)
        if name == "XGBoost":
            pipeline.set_params(clf__estimator__n_estimators=20)
        if name == "CatBoost":
            pipeline.set_params(clf__estimator__iterations=20)
        pipelines[name] = pipeline.fit(X, y)
    return pipelines, X, y


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_attribution_width_matches_the_encoded_feature_names(
    name, synthetic_config, fitted_pipelines
):
    pipelines, X, _ = fitted_pipelines
    pipeline = pipelines[name]
    preprocessor = pipeline.named_steps["prep"]
    names = encoded_feature_names(preprocessor)
    values = compute_attributions(
        name, inner_estimator(pipeline),
        preprocessor.transform(X), preprocessor.transform(X.iloc[:40]),
        names, background_size=50, random_state=SEEDS["explanation_sampling"],
    )
    assert values.shape == (40, len(names))


def test_the_explanation_object_is_never_the_calibration_wrapper(
    synthetic_config, fitted_pipelines
):
    """The calibrator wraps the model; explaining it would describe the wrapper."""
    pipelines, X, y = fitted_pipelines
    calibrated = fit_calibrated(
        build_pipeline("LR", synthetic_config, use_gpu=False, ebm_bags=2),
        X, y,
        cv=make_training_folds(y, n_splits=3, random_state=SEEDS["calibration_cv"]),
    )
    assert not hasattr(calibrated, "named_steps")
    assert hasattr(inner_estimator(pipelines["LR"]), "named_steps")


def test_one_hot_levels_are_aggregated_onto_their_source_feature(
    synthetic_config, fitted_pipelines
):
    pipelines, X, _ = fitted_pipelines
    preprocessor = pipelines["LR"].named_steps["prep"]
    names = encoded_feature_names(preprocessor)
    mapping = source_feature_map(preprocessor, synthetic_config)

    values = np.ones((5, len(names)))
    aggregated, sources = aggregate_to_source_features(values, names, mapping)

    assert "colour" in sources
    assert not any(source.startswith("colour_") for source in sources)
    colour_levels = sum(1 for name in names if mapping[name] == "colour")
    assert aggregated[0, sources.index("colour")] == pytest.approx(colour_levels)


def test_every_model_explains_the_identical_cases(synthetic_config, fitted_pipelines):
    pipelines, X, _ = fitted_pipelines
    rng = np.random.default_rng(SEEDS["explanation_sampling"])
    case_index = np.sort(rng.choice(len(X), 40, replace=False))

    widths = set()
    for name, pipeline in pipelines.items():
        preprocessor = pipeline.named_steps["prep"]
        names = encoded_feature_names(preprocessor)
        values = compute_attributions(
            name, inner_estimator(pipeline),
            preprocessor.transform(X), preprocessor.transform(X.iloc[case_index]),
            names, background_size=50, random_state=SEEDS["explanation_sampling"],
        )
        aggregated, sources = aggregate_to_source_features(
            np.abs(values), names, source_feature_map(preprocessor, synthetic_config)
        )
        widths.add(tuple(sources))
        assert aggregated.shape[0] == 40
    assert len(widths) == 1


def test_case_bootstrap_is_paired_and_labelled_as_case_selection_only():
    rng = np.random.default_rng(3)
    features = [f"f{index}" for index in range(12)]
    attributions = {
        "LR": rng.uniform(size=(80, 12)),
        "XGBoost": rng.uniform(size=(80, 12)),
    }
    summary, per_model, cross_model = case_bootstrap_stability(
        attributions, features, n_bootstrap=25,
        random_state=SEEDS["explanation_sampling"],
    )

    assert summary["uncertainty_scope"] == "case_selection_only"
    assert "Retraining variability is not covered" in summary["scope_note"]
    assert summary["n_bootstrap"] == 25 and summary["n_cases"] == 80
    assert len(per_model) == 2 * len(features)
    assert len(cross_model) == 1

    for row in per_model:
        assert 0.0 <= row["in_top10_frequency"] <= 1.0
        assert row["rank_ci_low"] <= row["rank_median"] <= row["rank_ci_high"]
    assert (
        summary["cross_model_jaccard_top10_ci_low"]
        <= summary["cross_model_jaccard_top10_ci_high"]
    )


def test_case_bootstrap_refuses_unpaired_inputs():
    with pytest.raises(ValueError, match="same cases"):
        case_bootstrap_stability(
            {"LR": np.zeros((10, 4)), "XGBoost": np.zeros((9, 4))},
            ["a", "b", "c", "d"], n_bootstrap=5, random_state=1,
        )


def test_jaccard_and_kendall_behave_on_known_inputs():
    identical = [np.arange(20, 0, -1), np.arange(20, 0, -1)]
    assert topk_jaccard(identical) == pytest.approx(1.0)
    assert kendall_w(identical) == pytest.approx(1.0)

    reversed_pair = [np.arange(20, 0, -1), np.arange(1, 21)]
    assert topk_jaccard(reversed_pair) < 1.0


def test_mean_abs_importance_averages_over_cases():
    values = np.asarray([[1.0, 3.0], [3.0, 1.0]])
    assert np.allclose(mean_abs_importance(values), [2.0, 2.0])
