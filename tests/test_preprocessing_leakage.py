"""The preprocessing pipeline must learn nothing from data it was not fitted on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    UNKNOWN_LEVEL,
    MissingnessFilter,
    Winsorizer,
    build_preprocessor,
    encoded_feature_names,
    preprocessing_report,
    source_feature_map,
)


def _fit(config, frame, rows=slice(0, 300)):
    features = config.feature_columns(frame.columns)
    preprocessor = build_preprocessor(config)
    preprocessor.fit(frame[features].iloc[rows])
    return preprocessor, features


def test_medians_ignore_held_out_rows(synthetic_config, synthetic_frame):
    preprocessor, features = _fit(synthetic_config, synthetic_frame)
    before = preprocessing_report(preprocessor)["imputation_median"]

    mutated = synthetic_frame.copy()
    mutated.loc[300:, "age"] = 1e9
    again, _ = _fit(synthetic_config, mutated)
    assert preprocessing_report(again)["imputation_median"] == before


def test_winsorization_thresholds_ignore_held_out_rows(synthetic_config, synthetic_frame):
    preprocessor, _ = _fit(synthetic_config, synthetic_frame)
    before = preprocessing_report(preprocessor)["winsorization"]

    mutated = synthetic_frame.copy()
    mutated.loc[300:, "income"] = 1e12
    again, _ = _fit(synthetic_config, mutated)
    assert preprocessing_report(again)["winsorization"] == before


def test_retained_columns_ignore_held_out_rows(synthetic_config, synthetic_frame):
    preprocessor, _ = _fit(synthetic_config, synthetic_frame)
    before = preprocessor.named_steps["missing"].columns_to_keep_

    mutated = synthetic_frame.copy()
    mutated.loc[300:, "mostly_missing"] = np.nan
    mutated.loc[300:, "income"] = np.nan
    again, _ = _fit(synthetic_config, mutated)
    assert again.named_steps["missing"].columns_to_keep_ == before


def test_category_schema_ignores_held_out_levels(synthetic_config, synthetic_frame):
    """The tail-only level 'chartreuse' must not enter a schema fitted without it."""
    preprocessor, features = _fit(synthetic_config, synthetic_frame)
    schema = preprocessing_report(preprocessor)["category_schema"]
    assert "chartreuse" not in schema["colour"]
    assert UNKNOWN_LEVEL in schema["colour"]


def test_unseen_category_encodes_as_all_zero_and_is_counted(
    synthetic_config, synthetic_frame
):
    preprocessor, features = _fit(synthetic_config, synthetic_frame)
    tail = synthetic_frame[features].iloc[-3:]
    transformed = preprocessor.transform(tail)

    names = encoded_feature_names(preprocessor)
    colour_columns = [
        index for index, name in enumerate(names) if name.startswith("colour_")
    ]
    assert np.allclose(transformed[:, colour_columns], 0.0)

    diagnostics = preprocessing_report(preprocessor)["unseen_categories"][-1]
    assert diagnostics["n_rows_with_unseen_level"] == 3
    assert diagnostics["by_feature"]["colour"]["unseen_levels"] == ["chartreuse"]


def test_unknown_is_a_first_class_level_not_an_all_zero_row(
    synthetic_config, synthetic_frame
):
    """Missing must stay distinguishable from an unseen category."""
    preprocessor, features = _fit(synthetic_config, synthetic_frame)
    names = encoded_feature_names(preprocessor)
    unknown_column = list(names).index(f"colour_{UNKNOWN_LEVEL}")

    missing_row = synthetic_frame[features].iloc[[0]].copy()
    missing_row["colour"] = None
    encoded = preprocessor.transform(missing_row)
    assert encoded[0, unknown_column] == 1.0


def test_unknown_level_exists_even_when_the_fit_subset_has_no_missing(
    synthetic_config, synthetic_frame
):
    complete = synthetic_frame[synthetic_frame["colour"].notna()].iloc[:200]
    features = synthetic_config.feature_columns(synthetic_frame.columns)
    preprocessor = build_preprocessor(synthetic_config)
    preprocessor.fit(complete[features])

    schema = preprocessing_report(preprocessor)["category_schema"]
    assert UNKNOWN_LEVEL in schema["colour"]


def test_scaler_statistics_ignore_held_out_rows(synthetic_config, synthetic_frame):
    from sklearn.base import clone

    from src.models import build_pipeline

    features = synthetic_config.feature_columns(synthetic_frame.columns)
    y = synthetic_frame["target"].to_numpy(dtype=int)
    pipeline = build_pipeline("LR", synthetic_config, use_gpu=False, ebm_bags=2)
    pipeline.fit(synthetic_frame[features].iloc[:300], y[:300])
    before = pipeline.named_steps["clf"].named_steps["scaler"].mean_.copy()

    mutated = synthetic_frame.copy()
    mutated.loc[300:, "income"] = 1e12
    again = clone(pipeline)
    again.fit(mutated[features].iloc[:300], y[:300])
    assert np.allclose(again.named_steps["clf"].named_steps["scaler"].mean_, before)


def test_missingness_filter_is_fit_subset_local(synthetic_config, synthetic_frame):
    """A column may survive in one fold and be dropped in another."""
    features = synthetic_config.feature_columns(synthetic_frame.columns)
    dense = build_preprocessor(synthetic_config)
    dense.fit(synthetic_frame[features].iloc[:120])
    sparse = build_preprocessor(synthetic_config)
    sparse.fit(synthetic_frame[features].iloc[200:])

    assert "mostly_missing" in dense.named_steps["missing"].columns_to_keep_
    assert "mostly_missing" not in sparse.named_steps["missing"].columns_to_keep_


def test_transform_never_refits(synthetic_config, synthetic_frame):
    preprocessor, features = _fit(synthetic_config, synthetic_frame)
    report_before = preprocessing_report(preprocessor)
    preprocessor.transform(synthetic_frame[features].iloc[300:])
    report_after = preprocessing_report(preprocessor)

    for key in ("imputation_median", "winsorization", "category_schema",
                "missingness_filter"):
        assert report_after[key] == report_before[key]


def test_auxiliary_and_target_columns_never_reach_the_matrix(
    synthetic_config, synthetic_frame
):
    preprocessor, features = _fit(synthetic_config, synthetic_frame)
    assert "aux_value" not in features
    assert "target" not in features
    assert "source_row_id" not in features
    assert not any(
        name.startswith(("aux_value", "target", "source_row_id"))
        for name in encoded_feature_names(preprocessor)
    )


def test_encoded_columns_map_back_to_source_features(synthetic_config, synthetic_frame):
    preprocessor, _ = _fit(synthetic_config, synthetic_frame)
    mapping = source_feature_map(preprocessor, synthetic_config)
    for name in encoded_feature_names(preprocessor):
        if name.startswith("colour_"):
            assert mapping[name] == "colour"
        elif name == "income":
            assert mapping[name] == "income"


def test_winsorizer_clips_only_the_upper_tail():
    frame = pd.DataFrame({"value": [-100.0, 1.0, 2.0, 3.0, 1000.0]})
    winsorizer = Winsorizer(columns=("value",), quantile=0.99).fit(frame)
    transformed = winsorizer.transform(frame)
    assert transformed["value"].min() == -100.0
    assert transformed["value"].max() < 1000.0


def test_missingness_filter_rejects_missing_columns_at_transform():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, 2.0]})
    fitted = MissingnessFilter(threshold=0.4).fit(frame)
    with pytest.raises(ValueError, match="lacks fitted columns"):
        fitted.transform(frame[["a"]])
